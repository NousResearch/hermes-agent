"""A dead LOCAL proxy must not freeze all LLM traffic.

When a loopback proxy (Clash / mihomo / V2Ray on ``127.0.0.1``) exits, the stale
``HTTP(S)_PROXY`` captured at launch otherwise routes every LLM call to a refused
port (``WinError 10061`` / ``ECONNREFUSED``) until the whole app restarts. The
fix probes loopback proxies with a short-lived cache and returns ``None`` (direct)
when the port refuses connections; remote proxies are never probed. Gated by the
``agent.proxy_fallback_direct`` config key (default True, fail-open).

Tests assert behaviour contracts (live vs dead loopback port, remote passthrough,
the config gate) and exercise real loopback sockets — no source-text snapshots.
"""
import socket

import pytest

from agent.process_bootstrap import _get_proxy_for_base_url, _proxy_fallback_direct_enabled
from agent.proxy_bypass import (
    proxy_endpoint_alive,
    proxy_is_loopback,
    reset_proxy_liveness_cache,
)

_PROXY_KEYS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
               "NO_PROXY", "no_proxy")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Reset the liveness cache between tests so a probe in one test can't pin the next."""
    reset_proxy_liveness_cache()
    yield
    reset_proxy_liveness_cache()


@pytest.fixture
def proxy_env(monkeypatch):
    for key in _PROXY_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def live_loopback_port():
    """A port on 127.0.0.1 that is actually listening (bind port 0 -> kernel-assigned free port)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(16)
    yield server
    server.close()


@pytest.fixture
def dead_loopback_port():
    """A port on 127.0.0.1 that is very likely not listening (grab a free port, then don't use it)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens here now
    return port


# ── proxy_is_loopback / proxy_endpoint_alive (real sockets) ──────────────────

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:7897", "https://localhost:8443", "socks5://[::1]:1080",
])
def test_loopback_detection(url):
    assert proxy_is_loopback(url)


@pytest.mark.parametrize("url", [
    "http://proxy.corp:3128", "http://10.0.0.1:8080", "http://8.8.8.8:3128",
])
def test_remote_proxy_is_not_loopback(url):
    assert not proxy_is_loopback(url)


def test_live_loopback_proxy_reports_alive(live_loopback_port):
    assert proxy_endpoint_alive(f"http://127.0.0.1:{live_loopback_port.getsockname()[1]}") is True


def test_dead_loopback_proxy_reports_dead(dead_loopback_port):
    assert proxy_endpoint_alive(f"http://127.0.0.1:{dead_loopback_port}") is False


def test_remote_proxy_is_never_probed(dead_loopback_port):
    """A remote proxy is reported alive even when its port is unreachable — we don't dial it."""
    # 8.8.8.8:1 is not a port we can reach; the guard must return True without touching the socket.
    assert proxy_endpoint_alive(f"http://10.9.8.7:{dead_loopback_port}") is True


def test_disabled_probe_is_a_noop(dead_loopback_port, proxy_env):
    """``enabled=False`` skips probing entirely — the legacy behaviour, no socket dial."""
    assert proxy_endpoint_alive(f"http://127.0.0.1:{dead_loopback_port}", enabled=False) is True


def test_cache_bounds_probe_count(monkeypatch, live_loopback_port):
    """Within the TTL window the port is dialled once; after a cache reset it dials again."""
    url = f"http://127.0.0.1:{live_loopback_port.getsockname()[1]}"
    calls = {"n": 0}
    real_connect = socket.create_connection

    def counting_connect(*args, **kwargs):
        calls["n"] += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", counting_connect)
    assert proxy_endpoint_alive(url) is True
    assert proxy_endpoint_alive(url) is True  # cached, no new dial
    assert calls["n"] == 1
    reset_proxy_liveness_cache()
    assert proxy_endpoint_alive(url) is True
    assert calls["n"] == 2


# ── _get_proxy_for_base_url integration (the transport decision) ─────────────

def test_live_loopback_proxy_still_routes_via_proxy(proxy_env, live_loopback_port):
    port = live_loopback_port.getsockname()[1]
    proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == f"http://127.0.0.1:{port}"


def test_dead_loopback_proxy_falls_back_to_direct(proxy_env, dead_loopback_port):
    proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{dead_loopback_port}")
    # Default gate is on -> the dead loopback endpoint is bypassed, so dial direct.
    assert _get_proxy_for_base_url("https://api.openai.com/v1") is None


def test_dead_loopback_proxy_with_no_base_url(proxy_env, dead_loopback_port):
    """Even a request with no base_url must not be sent to a dead loopback proxy."""
    proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{dead_loopback_port}")
    assert _get_proxy_for_base_url(None) is None


def test_remote_dead_proxy_keeps_the_proxy(proxy_env, dead_loopback_port):
    """A remote proxy that is down is NOT our business — keep routing through it (no probe)."""
    proxy_env.setenv("HTTPS_PROXY", f"http://10.9.8.7:{dead_loopback_port}")
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == f"http://10.9.8.7:{dead_loopback_port}"


def test_config_gate_off_restores_legacy(proxy_env, dead_loopback_port, monkeypatch):
    """``agent.proxy_fallback_direct: false`` (or the gate read False) -> dead loopback still routes
    through the proxy, exactly the pre-fix symptom."""
    proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{dead_loopback_port}")
    monkeypatch.setattr("agent.process_bootstrap._proxy_fallback_direct_enabled", lambda: False)
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == f"http://127.0.0.1:{dead_loopback_port}"


def test_no_proxy_bypass_precedes_liveness(proxy_env, live_loopback_port):
    """A host in NO_PROXY bypasses the proxy even when the loopback proxy is alive (unchanged path)."""
    proxy_env.setenv("HTTPS_PROXY", f"http://127.0.0.1:{live_loopback_port.getsockname()[1]}")
    proxy_env.setenv("NO_PROXY", "internal.example.com")
    assert _get_proxy_for_base_url("https://internal.example.com/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == f"http://127.0.0.1:{live_loopback_port.getsockname()[1]}"


def test_no_proxy_env_at_all(proxy_env):
    """No proxy configured -> no probe, direct connection (the fast path)."""
    assert _get_proxy_for_base_url("https://api.openai.com/v1") is None


# ── the config gate itself ───────────────────────────────────────────────────

def test_gate_defaults_to_true_when_unset(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    assert _proxy_fallback_direct_enabled() is True


def test_gate_honours_config_false(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"proxy_fallback_direct": False}},
    )
    assert _proxy_fallback_direct_enabled() is False


def test_gate_fails_open_on_config_error(monkeypatch):
    def _boom():
        raise OSError("config unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert _proxy_fallback_direct_enabled() is True
