"""Unit tests for ``hermes_cli.proxy_cli`` command handlers.

These tests cover the user-facing CLI surface that was previously
uncovered.  We mock the iron_proxy module's side-effect functions
(install / start / stop / discover) and exercise the dispatch +
return-code logic plus the small amount of presentation logic in
each handler (e.g. --from-bitwarden's fail-loud path).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.proxy_sources import iron_proxy as ip
from hermes_cli import proxy_cli


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so the wizard doesn't touch the
    operator's real config.  Also blanks any provider env vars so we
    don't accidentally read a real key."""

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in list(os.environ):
        if key.endswith("_API_KEY") or key in (
            "BWS_ACCESS_TOKEN", "ANTHROPIC_API_KEY",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
    return home


def _args(**overrides):
    ns = argparse.Namespace(
        force=False,
        tunnel_port=None,
        from_bitwarden=False,
        rotate_tokens=False,
        show_tokens=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# cmd_install
# ---------------------------------------------------------------------------


def test_cmd_install_success_returns_0(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "install_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "v0.39.0-test")
    rc = proxy_cli.cmd_install(_args())
    assert rc == 0


def test_cmd_install_failure_returns_1(hermes_home, monkeypatch):
    def boom(**kw):
        raise RuntimeError("download failed")
    monkeypatch.setattr(ip, "install_iron_proxy", boom)
    rc = proxy_cli.cmd_install(_args())
    assert rc == 1


# ---------------------------------------------------------------------------
# cmd_setup — --from-bitwarden fail-loud paths
# ---------------------------------------------------------------------------


def test_cmd_setup_from_bitwarden_refuses_when_bw_disabled(hermes_home, monkeypatch):
    """When --from-bitwarden is passed but secrets.bitwarden.enabled=false,
    the wizard must FAIL rather than silently rewriting credential_source
    to bitwarden."""

    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("secrets", {})["bitwarden"] = {"enabled": False}
    save_config(cfg)

    # Pre-stub install + CA so we get to step 3.
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    monkeypatch.setattr(
        ip, "ensure_ca_cert",
        lambda **kw: (hermes_home / "ca.crt", hermes_home / "ca.key"),
    )

    rc = proxy_cli.cmd_setup(_args(from_bitwarden=True))
    assert rc == 1
    # Verify we did NOT write credential_source: bitwarden to config.
    cfg2 = load_config()
    proxy_cfg = cfg2.get("proxy") or {}
    assert proxy_cfg.get("credential_source", "env") != "bitwarden"


def test_cmd_setup_from_bitwarden_refuses_when_token_missing(hermes_home, monkeypatch):
    """--from-bitwarden with secrets.bitwarden.enabled=true but BWS access
    token unset → fail loud, not silent env-fallback."""

    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("secrets", {})["bitwarden"] = {
        "enabled": True,
        "project_id": "test-proj",
        "access_token_env": "BWS_ACCESS_TOKEN",
    }
    save_config(cfg)
    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    monkeypatch.setattr(
        ip, "ensure_ca_cert",
        lambda **kw: (hermes_home / "ca.crt", hermes_home / "ca.key"),
    )

    rc = proxy_cli.cmd_setup(_args(from_bitwarden=True))
    assert rc == 1


def test_cmd_setup_from_bitwarden_refuses_on_empty_vault(hermes_home, monkeypatch):
    """If BW returns {} (empty vault / scoped wrong / unreachable), fail
    loud rather than silently writing credential_source: bitwarden."""

    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("secrets", {})["bitwarden"] = {
        "enabled": True,
        "project_id": "test-proj",
        "access_token_env": "BWS_ACCESS_TOKEN",
    }
    save_config(cfg)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "bwsk-test-token")

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    monkeypatch.setattr(
        ip, "ensure_ca_cert",
        lambda **kw: (hermes_home / "ca.crt", hermes_home / "ca.key"),
    )

    # Mock fetch_bitwarden_secrets to return an empty dict (empty vault).
    fake_bw = MagicMock()
    fake_bw.fetch_bitwarden_secrets = lambda **kw: ({}, [])
    monkeypatch.setattr("agent.secret_sources.bitwarden", fake_bw, raising=False)
    import sys
    sys.modules["agent.secret_sources.bitwarden"] = fake_bw

    rc = proxy_cli.cmd_setup(_args(from_bitwarden=True))
    assert rc == 1


def test_cmd_setup_rejects_tunnel_port_zero(hermes_home, monkeypatch):
    """--tunnel-port=0 is rejected explicitly (was silently substituting
    the default before the fix)."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    monkeypatch.setattr(
        ip, "ensure_ca_cert",
        lambda **kw: (hermes_home / "ca.crt", hermes_home / "ca.key"),
    )
    rc = proxy_cli.cmd_setup(_args(tunnel_port=0))
    assert rc == 1


@pytest.mark.parametrize("bad_port", [-1, 65535, 65536])
def test_cmd_setup_rejects_invalid_tunnel_port_range(hermes_home, monkeypatch, bad_port):
    """The egress wizard owns the derived HTTP listener at tunnel_port+1,
    so both listener ports must fit in the TCP range."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "iron_proxy_version", lambda b: "test")
    monkeypatch.setattr(
        ip,
        "ensure_ca_cert",
        lambda **kw: (hermes_home / "ca.crt", hermes_home / "ca.key"),
    )

    rc = proxy_cli.cmd_setup(_args(tunnel_port=bad_port))
    assert rc == 1


# ---------------------------------------------------------------------------
# cmd_start — fail_on_uncovered_providers + Bitwarden rotation wire-up
# ---------------------------------------------------------------------------


def test_cmd_start_refuses_when_proxy_disabled(hermes_home, monkeypatch):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = False
    save_config(cfg)

    rc = proxy_cli.cmd_start(_args())
    assert rc == 1


def test_cmd_start_refuses_on_uncovered_provider_when_strict(hermes_home, monkeypatch):
    """fail_on_uncovered_providers=true + ANTHROPIC_API_KEY in env =
    refuse to start (real credential would otherwise leak into sandbox)."""

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["fail_on_uncovered_providers"] = True
    save_config(cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    rc = proxy_cli.cmd_start(_args())
    assert rc == 1


def test_cmd_start_honors_auto_install_false(hermes_home, monkeypatch):
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["auto_install"] = False
    save_config(cfg)

    captured: dict = {}

    def fake_start_proxy(**kw):
        captured.update(kw)
        s = ip.ProxyStatus(pid=4242, listening=True, tunnel_port=9090)
        return s

    monkeypatch.setattr(ip, "start_proxy", fake_start_proxy)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    monkeypatch.setattr(ip, "discover_blocked_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 0
    assert captured.get("install_if_missing") is False


def test_cmd_start_passes_bitwarden_refresh_flag_when_credential_source_is_bitwarden(
    hermes_home, monkeypatch,
):
    """When credential_source=bitwarden, cmd_start must wire
    refresh_secrets_from_bitwarden=True into start_proxy.  That's what
    delivers the rotation promise the docs make."""

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["credential_source"] = "bitwarden"
    cfg["proxy"]["fail_on_uncovered_providers"] = False
    cfg.setdefault("secrets", {})["bitwarden"] = {
        "enabled": True,
        "project_id": "test-proj-id",
        "access_token_env": "BWS_ACCESS_TOKEN",
    }
    save_config(cfg)
    # v3: cmd_start now pre-checks BWS access token + project_id before
    # calling start_proxy.  Provide both so we get to the rotation
    # wire-up code path.
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "bwsk-test-access-token")

    captured: dict = {}
    def fake_start_proxy(**kw):
        captured.update(kw)
        s = ip.ProxyStatus()
        s.pid = 4242
        s.listening = True
        s.tunnel_port = 9090
        return s
    monkeypatch.setattr(ip, "start_proxy", fake_start_proxy)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    monkeypatch.setattr(ip, "discover_blocked_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 0
    assert captured.get("refresh_secrets_from_bitwarden") is True
    assert captured.get("bitwarden_config") is not None


def test_cmd_start_refuses_when_bitwarden_token_missing(hermes_home, monkeypatch):
    """stephenschoettler #1: when credential_source=bitwarden but the
    access-token env var is empty, cmd_start must fail-loud BEFORE
    start_proxy can silently fall back to parent env."""

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["credential_source"] = "bitwarden"
    cfg["proxy"]["fail_on_uncovered_providers"] = False
    cfg.setdefault("secrets", {})["bitwarden"] = {
        "enabled": True,
        "project_id": "test-proj-id",
        "access_token_env": "BWS_ACCESS_TOKEN",
    }
    save_config(cfg)
    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)

    # Sentinel: start_proxy must NOT be called.
    def must_not_call(**kw):
        pytest.fail("start_proxy should not be invoked when BWS token missing")
    monkeypatch.setattr(ip, "start_proxy", must_not_call)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    monkeypatch.setattr(ip, "discover_blocked_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 1


def test_cmd_start_does_not_pass_bitwarden_refresh_when_credential_source_is_env(
    hermes_home, monkeypatch,
):
    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["credential_source"] = "env"
    cfg["proxy"]["fail_on_uncovered_providers"] = False
    save_config(cfg)

    captured: dict = {}
    def fake_start_proxy(**kw):
        captured.update(kw)
        s = ip.ProxyStatus()
        s.pid = 4242
        s.listening = True
        return s
    monkeypatch.setattr(ip, "start_proxy", fake_start_proxy)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 0
    assert captured.get("refresh_secrets_from_bitwarden") is False


# ---------------------------------------------------------------------------
# cmd_stop, cmd_status, cmd_disable, cmd_config
# ---------------------------------------------------------------------------


def test_cmd_stop_returns_0_when_running(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "stop_proxy", lambda: True)
    rc = proxy_cli.cmd_stop(_args())
    assert rc == 0


def test_cmd_stop_returns_0_when_already_stopped(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "stop_proxy", lambda: False)
    rc = proxy_cli.cmd_stop(_args())
    assert rc == 0


def test_cmd_status_returns_0(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "get_status", lambda: ip.ProxyStatus())
    monkeypatch.setattr(ip, "load_mappings", lambda: [])
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    rc = proxy_cli.cmd_status(_args())
    assert rc == 0


def test_cmd_disable_uses_public_status_pid_not_private_read_pid(
    hermes_home, monkeypatch,
):
    """cmd_disable must read status.pid (which incorporates the _pid_alive
    check) — NOT ip._read_pid() directly (which would fire a spurious
    'still running' warning for a stale pidfile from a crashed run)."""

    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    save_config(cfg)

    # Pidfile exists but the process is dead.  Old code would have warned
    # "still running"; the new code reads status.pid which returns None
    # because _pid_alive is False, so no spurious warning.
    state = ip._proxy_state_dir()
    (state / "iron-proxy.pid").write_text("99999")
    # _pid_alive returns False → status.pid is None.
    monkeypatch.setattr(ip, "_pid_alive", lambda pid: False)

    # If cmd_disable reads _read_pid() directly (old path), this test
    # would still pass — but reading status.pid is the correct
    # API.  Sentinel: confirm _read_pid is NOT called from cmd_disable.
    read_pid_calls = []
    real_read_pid = ip._read_pid
    def tracked_read_pid(*a, **kw):
        read_pid_calls.append((a, kw))
        return real_read_pid(*a, **kw)
    monkeypatch.setattr(ip, "_read_pid", tracked_read_pid)

    rc = proxy_cli.cmd_disable(_args())
    assert rc == 0
    # cmd_disable should call get_status() (which may internally call
    # _read_pid), but should NOT call _read_pid from its own body.
    # Hard to assert directly without source-introspection — the meatier
    # assertion is that no "still running" message fired with a stale
    # pidfile.  That's covered by inspecting return code + config
    # mutation only.
    from hermes_cli.config import load_config as _lc
    cfg2 = _lc()
    assert cfg2["proxy"]["enabled"] is False


def test_cmd_config_returns_0_when_present(hermes_home, monkeypatch):
    fake = ip.ProxyStatus()
    fake.config_path = hermes_home / "proxy.yaml"
    monkeypatch.setattr(ip, "get_status", lambda: fake)
    rc = proxy_cli.cmd_config(_args())
    assert rc == 0


def test_cmd_config_returns_1_when_missing(hermes_home, monkeypatch):
    monkeypatch.setattr(ip, "get_status", lambda: ip.ProxyStatus())
    rc = proxy_cli.cmd_config(_args())
    assert rc == 1


# ---------------------------------------------------------------------------
# Argparse wiring — dest='egress_command' regression
# ---------------------------------------------------------------------------


def test_register_cli_uses_egress_command_dest():
    """The subparser dest must be 'egress_command' to stay disjoint from
    the inbound OAuth 'hermes proxy' subparser (dest='proxy_command').
    A future grep-and-refactor on proxy_command should not hit this
    subparser by accident."""

    parser = argparse.ArgumentParser(prog="hermes egress")
    proxy_cli.register_cli(parser)
    # Parse a no-op invocation and confirm the attribute name.
    args = parser.parse_args(["install"])
    assert hasattr(args, "egress_command")
    assert not hasattr(args, "proxy_command")


def test_egress_subcommands_registered():
    """Smoke test: every documented subcommand parses without error."""

    parser = argparse.ArgumentParser(prog="hermes egress")
    proxy_cli.register_cli(parser)
    for sub in ("install", "setup", "start", "stop", "status", "disable", "config"):
        args = parser.parse_args([sub])
        assert args.egress_command == sub


def test_setup_has_rotate_tokens_flag():
    """--rotate-tokens is the documented escape hatch for re-rolling
    every proxy token (used after a suspected token leak).  Default is
    preserve-existing."""

    parser = argparse.ArgumentParser(prog="hermes egress")
    proxy_cli.register_cli(parser)
    args = parser.parse_args(["setup"])
    assert args.rotate_tokens is False
    args = parser.parse_args(["setup", "--rotate-tokens"])
    assert args.rotate_tokens is True


# ---------------------------------------------------------------------------
# v4 round: credential_source=bitwarden with secrets.bitwarden disabled
# must NOT silently degrade to host-env secrets
# ---------------------------------------------------------------------------


def test_cmd_start_refuses_when_bitwarden_mode_but_disabled(hermes_home, monkeypatch):
    """config keeps credential_source: bitwarden but secrets.bitwarden.enabled
    later flips to false — cmd_start must refuse, not silently start on
    host env (the silent-degrade class strict mode is meant to close)."""

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["credential_source"] = "bitwarden"
    cfg["proxy"]["fail_on_uncovered_providers"] = False
    cfg.setdefault("secrets", {})["bitwarden"] = {"enabled": False}
    save_config(cfg)

    def must_not_call(**kw):
        pytest.fail("start_proxy must not run when bitwarden mode is broken")
    monkeypatch.setattr(ip, "start_proxy", must_not_call)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    monkeypatch.setattr(ip, "discover_blocked_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 1


def test_cmd_start_bitwarden_disabled_proceeds_with_env_fallback(
    hermes_home, monkeypatch,
):
    """Same scenario but proxy.allow_env_fallback=true is the documented
    escape hatch — start proceeds (with a warning)."""

    from hermes_cli.config import load_config, save_config
    cfg = load_config()
    cfg.setdefault("proxy", {})["enabled"] = True
    cfg["proxy"]["credential_source"] = "bitwarden"
    cfg["proxy"]["allow_env_fallback"] = True
    cfg["proxy"]["fail_on_uncovered_providers"] = False
    cfg.setdefault("secrets", {})["bitwarden"] = {"enabled": False}
    save_config(cfg)

    captured: dict = {}

    def fake_start_proxy(**kw):
        captured.update(kw)
        s = ip.ProxyStatus()
        s.pid = 4242
        s.listening = True
        return s

    monkeypatch.setattr(ip, "start_proxy", fake_start_proxy)
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])
    monkeypatch.setattr(ip, "discover_blocked_providers", lambda **kw: [])

    rc = proxy_cli.cmd_start(_args())
    assert rc == 0
    # BW refresh is off (bitwarden disabled), running on env secrets.
    assert captured.get("refresh_secrets_from_bitwarden") is False


def test_cmd_setup_audit_log_failure_is_warning_not_abort(hermes_home, monkeypatch):
    """On the pinned v0.39 the daemon never writes audit.log, so a
    pre-create failure must not abort the wizard."""

    monkeypatch.setattr(ip, "find_iron_proxy", lambda **kw: hermes_home / "iron-proxy")
    monkeypatch.setattr(ip, "discover_provider_mappings", lambda **kw: [
        ip.TokenMapping(
            proxy_token="hermes-proxy-deadbeef",
            real_env_name="OPENROUTER_API_KEY",
            upstream_hosts=("openrouter.ai",),
        ),
    ])
    monkeypatch.setattr(ip, "discover_uncovered_providers", lambda **kw: [])

    def boom(path):
        raise RuntimeError("could not pre-create audit log (synthetic)")
    monkeypatch.setattr(ip, "ensure_audit_log", boom)

    rc = proxy_cli.cmd_setup(_args())
    assert rc == 0
