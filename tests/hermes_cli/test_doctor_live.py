"""Tests for ``hermes doctor --live`` — opt-in bounded real-call tool-backend probes.

All probes are mocked at the HTTP/client layer; no real network calls are made.
"""

from __future__ import annotations

import argparse
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from hermes_cli import doctor_live
from hermes_cli.doctor_live import (
    maybe_run_live_checks,
    run_live_checks,
)
from tools import browser_tool_install as bt_install

# Captured before the autouse fixture below stubs doctor_live._browser_available
# to a constant, so TestBrowserAvailableNpxRung can exercise the real function.
_real_browser_available = doctor_live._browser_available

# The exact environment Clash Verge / mihomo exports (issue #118159): a bracketed-IPv6
# NO_PROXY entry that httpx 0.28.1 cannot parse into a proxy mount.
_CLASH_NO_PROXY = "127.0.0.1,localhost,::1,[::1]"


@pytest.fixture
def clash_env(monkeypatch):
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
    monkeypatch.setenv("no_proxy", _CLASH_NO_PROXY)


class _Probes(BaseHTTPRequestHandler):
    """Minimal 200-origin for live-path tests; every request is a bare metadata GET."""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass  # keep test output clean


def _args(live: bool = True) -> argparse.Namespace:
    return argparse.Namespace(live=live)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip backend credentials so each test opts in explicitly."""
    for var in ("FIRECRAWL_API_KEY", "FAL_KEY", "OPENAI_API_KEY",
                "ELEVENLABS_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # Default: empty config, no MCP servers, local tts/stt.
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    # Default: browser not installed.
    monkeypatch.setattr(doctor_live, "_browser_available", lambda: False)


class TestLiveFlagGating:

    def test_no_live_flag_means_zero_probes(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            doctor_live, "run_live_checks",
            lambda *a, **k: called.append(True))
        result = maybe_run_live_checks(_args(live=False), [])
        assert result is None
        assert called == []



    def test_live_check_crash_never_propagates(self, monkeypatch, capsys):
        def _boom(*a, **k):
            raise RuntimeError("probe subsystem exploded")

        monkeypatch.setattr(doctor_live, "run_live_checks", _boom)
        # Must not raise.
        maybe_run_live_checks(_args(live=True), [])


class TestConfiguredOnlySelection:
        # No issues appended for skips.

    def test_unconfigured_backends_do_not_touch_network(self, monkeypatch):
        def _no_net(*a, **k):
            raise AssertionError("HTTP call made for unconfigured backend")

        monkeypatch.setattr(doctor_live, "_http_get", _no_net)
        results = run_live_checks([])
        assert all(r.status == "skip" for r in results)

    def test_firecrawl_probed_when_key_present(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        calls = []

        def _fake_get(url, headers=None, timeout=None):
            calls.append(url)
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _fake_get)
        results = {r.name: r for r in run_live_checks([])}
        assert results["Firecrawl"].status == "pass"
        assert any("firecrawl" in u for u in calls)

    def test_firecrawl_invalid_key_fails_and_appends_issue(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-bad")
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=401))
        issues: list[str] = []
        results = {r.name: r for r in run_live_checks(issues)}
        assert results["Firecrawl"].status == "fail"
        assert any("FIRECRAWL" in i or "Firecrawl" in i for i in issues)


    def test_mcp_servers_probed_per_configured_server(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"mcp_servers": {"alpha": {"url": "https://x"},
                                     "beta": {"command": "foo"}}})
        probed = []
        monkeypatch.setattr(
            doctor_live, "_probe_mcp_server",
            lambda name, cfg, timeout: probed.append(name) or [("t", "d")])
        results = [r for r in run_live_checks([]) if r.name.startswith("MCP")]
        assert sorted(probed) == ["alpha", "beta"]
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)

    def test_tts_local_provider_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"tts": {"provider": "edge"}})
        results = {r.name: r for r in run_live_checks([])}
        assert results["TTS"].status == "skip"

    def test_tts_openai_probed_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"tts": {"provider": "openai"}})
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=200))
        results = {r.name: r for r in run_live_checks([])}
        assert results["TTS"].status == "pass"

    def test_stt_groq_probed_with_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"stt": {"provider": "groq"}})
        monkeypatch.setattr(
            doctor_live, "_http_get",
            lambda *a, **k: SimpleNamespace(status_code=200))
        results = {r.name: r for r in run_live_checks([])}
        assert results["STT"].status == "pass"

    def test_stt_provider_configured_but_key_missing_warns(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"stt": {"provider": "groq"}})
        results = {r.name: r for r in run_live_checks([])}
        assert results["STT"].status == "warn"

    def test_browser_probed_when_available(self, monkeypatch):
        monkeypatch.setattr(doctor_live, "_browser_available", lambda: True)
        monkeypatch.setattr(
            doctor_live, "_launch_browser_probe",
            lambda timeout: (True, "about:blank ok"))
        results = {r.name: r for r in run_live_checks([])}
        assert results["Browser"].status == "pass"


@pytest.mark.parametrize("kind", ["NO_PROXY", "no_proxy"])
@pytest.mark.parametrize("entry", ["[::1]", "[::1]:8080", "::1/128"])
def test_http_get_sanitizes_bracketed_ipv6_no_proxy(monkeypatch, kind, entry):
    # The Clash Verge / mihomo export from #118159: httpx 0.28.1 turns a bracketed or
    # CIDR IPv6 NO_PROXY entry into an unparseable mount pattern and raises InvalidURL
    # at client construction, before any request. doctor --live's _http_get must not.
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(kind, f"127.0.0.1,localhost,::1,{entry}")
    # A bare local-socket GET proves the client was constructible; 127.0.0.1's own
    # NO_PROXY membership keeps it out of any proxy's reach.
    with HTTPServer(("127.0.0.1", 0), _Probes) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            resp = doctor_live._http_get(
                f"http://127.0.0.1:{httpd.server_port}/", timeout=5.0)
            assert resp.status_code == 200
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def test_run_live_checks_sanitizes_clash_env_and_probes_pass(clash_env, monkeypatch):
    # Red-on-base end-to-end: _keyed_probe -> _http_get -> bare httpx.get raised
    # InvalidURL under this env; _run_one's catch-all dressed it up as
    # "(Invalid port: ':1]')" — a backend-failure verdict for a healthy service.
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

    # Real HTTP origin: a 200 from an actual socket proves both client construction
    # and the request reaching a reachable origin. The remote-credential probes would
    # send real auth material or 401 without proving either.
    with HTTPServer(("127.0.0.1", 0), _Probes) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            monkeypatch.setitem(
                doctor_live._KEYED_PROBES, "Firecrawl",
                (f"http://127.0.0.1:{httpd.server_port}/v2/team/credit-usage",
                 "FIRECRAWL_API_KEY", "Bearer"))
            issues: list[str] = []
            results = {r.name: r for r in run_live_checks(issues)}
            assert results["Firecrawl"].status == "pass"
            assert issues == []
            assert "Invalid port" not in (results["Firecrawl"].detail or "")
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
    # The seam rewrote the entry in place, the established pattern of this seam
    # (chat transports and run_probes already do exactly this).
    assert "[::1]" not in os.environ["NO_PROXY"]
    assert os.environ["no_proxy"] == "127.0.0.1,localhost,::1"


def test_run_live_checks_leaves_a_clean_environment_untouched(monkeypatch):
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "example.com,10.0.0.0/8")
    # _http_get is untouched by the sanitize-once design; a socket-level 200 proves
    # the live path still works end to end when there is nothing to sanitize.
    with HTTPServer(("127.0.0.1", 0), _Probes) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            assert doctor_live._http_get(
                f"http://127.0.0.1:{httpd.server_port}/", timeout=5.0).status_code == 200
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
    assert os.environ["NO_PROXY"] == "example.com,10.0.0.0/8"


class TestBrowserAvailable:
    """Live probes use the same passive selection as browser execution."""

    def _block_path_and_node_modules_checks(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda *a, **k: None)
        monkeypatch.setattr("hermes_cli.doctor.HERMES_HOME", tmp_path / "home")
        monkeypatch.setattr("hermes_cli.doctor.PROJECT_ROOT", tmp_path / "root")

    def test_true_when_installed_browser_resolves(self, monkeypatch, tmp_path):
        self._block_path_and_node_modules_checks(monkeypatch, tmp_path)

        monkeypatch.setattr(bt_install, "_find_agent_browser", lambda **_kw: "/pm/agent-browser")

        assert _real_browser_available() is True

    def test_false_when_nothing_resolves(self, monkeypatch, tmp_path):
        self._block_path_and_node_modules_checks(monkeypatch, tmp_path)

        def _raise(**_kw):
            raise FileNotFoundError("agent-browser CLI not found")

        monkeypatch.setattr(bt_install, "_find_agent_browser", _raise)

        assert _real_browser_available() is False


class TestFailureIsolation:
    def test_one_probe_raising_does_not_stop_others(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setenv("FAL_KEY", "fal-test")

        def _get(url, headers=None, timeout=None):
            if "firecrawl" in url:
                raise RuntimeError("connection reset")
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _get)
        issues: list[str] = []
        results = {r.name: r for r in run_live_checks(issues)}
        assert results["Firecrawl"].status == "fail"
        assert results["FAL"].status == "pass"

    def test_mcp_probe_failure_isolated_per_server(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"mcp_servers": {"bad": {"url": "https://x"},
                                     "good": {"url": "https://y"}}})

        def _probe(name, cfg, timeout):
            if name == "bad":
                raise ConnectionError("refused")
            return [("tool", "desc")]

        monkeypatch.setattr(doctor_live, "_probe_mcp_server", _probe)
        results = {r.name: r for r in run_live_checks([])}
        assert results["MCP: bad"].status == "fail"
        assert results["MCP: good"].status == "pass"


class TestTimeoutHandling:
    def test_timeout_reported_as_fail(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

        def _slow(*a, **k):
            raise TimeoutError("timed out")

        monkeypatch.setattr(doctor_live, "_http_get", _slow)
        results = {r.name: r for r in run_live_checks([])}
        assert results["Firecrawl"].status == "fail"
        assert "time" in (results["Firecrawl"].detail or "").lower()

    def test_probe_timeout_bounded_and_configurable(self, monkeypatch):
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"doctor": {"live_probe_timeout": 3}})
        seen = {}

        def _get(url, headers=None, timeout=None):
            seen["timeout"] = timeout
            return SimpleNamespace(status_code=200)

        monkeypatch.setattr(doctor_live, "_http_get", _get)
        run_live_checks([])
        assert seen["timeout"] == 3



class TestReadOnly:

    def test_skips_never_append_issues(self, capsys):
        issues: list[str] = []
        run_live_checks(issues)
        assert issues == []
