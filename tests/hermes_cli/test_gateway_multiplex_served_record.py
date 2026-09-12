"""The live multiplexer's recorded ``served_profiles`` decides "served"; every start verb honours it.

- Probe: ``named_profile_served_by_running_multiplexer`` reads the pid-verified default
  ``gateway_state.json`` first (env-only opt-in on the default profile is invisible to ``hermes -p X``);
  config/env derivation is only the fallback when the key is absent.
- ``gateway start`` / ``install`` / ``restart`` refuse (exit 78) like ``run`` does, so a served profile
  never gets a permanently failing systemd unit or a launchd respawn loop; ``--force`` overrides.
- ``hermes status`` / ``cron status`` on a satellite say running-via-multiplexer; the default lists served.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os

import pytest


@pytest.fixture
def served_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    (root / "profiles" / "coder").mkdir(parents=True)
    (root / "profiles" / "other").mkdir(parents=True)
    (root / "config.yaml").write_text("model: {default: x}\n")  # NO multiplex flag: env-only opt-in
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "hermes_home": str(root), "served_profiles": ["default", "coder"]}))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    return root


def test_probe_trusts_live_record_over_cli_side_config(served_root):
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    assert named_profile_served_by_running_multiplexer("coder") is True
    assert named_profile_served_by_running_multiplexer("other") is False
    # Config says multiplex on, but the running gateway did not pick up 'other': the record wins.
    (served_root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    assert named_profile_served_by_running_multiplexer("other") is False


def test_probe_falls_back_to_config_only_without_recorded_key(served_root):
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    (served_root / "gateway_state.json").write_text(json.dumps({"pid": os.getpid()}))
    assert named_profile_served_by_running_multiplexer("coder") is False
    (served_root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    assert named_profile_served_by_running_multiplexer("coder") is True


@pytest.mark.parametrize("verb", ["start", "install", "restart"])
def test_service_verbs_refuse_served_profile_with_exit_78(served_root, monkeypatch, verb):
    import hermes_cli.gateway as gw
    calls: list = []
    monkeypatch.setattr(gw, "_service_backend", lambda: "systemd")
    monkeypatch.setattr(gw, "_service_call", lambda backend, v, system: calls.append(v))
    monkeypatch.setattr(gw, "_install_systemd_from_cli", lambda *a, **k: calls.append("install"))
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda v: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda *a, **k: "systemd")
    monkeypatch.setattr(gw, "is_managed", lambda: False)
    monkeypatch.setattr(gw, "is_termux", lambda: False)
    fn = getattr(gw, f"_cmd_{verb}")
    ns = argparse.Namespace(system=False, all=False, force=False, run_as_user=None)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        fn(ns)
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE and calls == []

    ns.force = True
    with contextlib.redirect_stdout(io.StringIO()):
        fn(ns)
    assert calls, f"--force must let `gateway {verb}` reach the service manager"


def test_status_surfaces_agree_for_a_satellite_profile(served_root, monkeypatch):
    import hermes_cli.gateway as gw
    import hermes_cli.status as st
    import hermes_cli.cron as cr
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(gw, "get_gateway_runtime_snapshot",
                        lambda system=False: gw.GatewayRuntimeSnapshot(manager="systemd (user)"))
    monkeypatch.setattr(cr, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr(cr, "_print_active_jobs_summary", lambda jobs: None)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st._render_gateway(None)
    assert "running" in buf.getvalue() and "multiplexer" in buf.getvalue()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cr.cron_status()
    assert "NOT fire" not in buf.getvalue() and "multiplexer" in buf.getvalue()


def test_dashboard_liveness_ladder_reports_served_profile_running(served_root):
    """`/api/status?profile=X` and `/api/messaging/platforms?profile=X` share this ladder: a served
    profile has no gateway.pid/gateway_state.json, so without the multiplexer rung the dashboard said
    "stopped" while `hermes -p X status` said running. Alpha's `<X>:<platform>` entries project as its own."""
    from gateway.status import profile_platforms_from_multiplexer, resolve_gateway_liveness
    (served_root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(served_root), "gateway_state": "running",
        "served_profiles": ["default", "coder"],
        "platforms": {"api_server": {"state": "connected"}, "coder:telegram": {"state": "connected"}}}))
    coder = served_root / "profiles" / "coder"
    live = resolve_gateway_liveness(profile_dir=coder, health_probe=None, use_cache=False)
    assert live.running is True and live.pid == os.getpid() and live.source == "multiplexer"
    assert profile_platforms_from_multiplexer(live.runtime, "coder") == {"telegram": {"state": "connected"}}
    # An unserved profile keeps the historical "stopped" answer.
    other = resolve_gateway_liveness(profile_dir=served_root / "profiles" / "other", health_probe=None, use_cache=False)
    assert other.running is False


def test_dashboard_lifecycle_verbs_target_the_multiplexer(served_root, monkeypatch):
    """`gateway restart` for a served profile restarts the multiplexer (a `-p X` child only exits 78 into
    the action log); `start`/`stop` refuse; a profile with its own gateway is managed normally."""
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server_gateway import _gateway_subcommand, multiplexed_profile_refusal
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: False)
    assert _gateway_subcommand("coder", "restart") == ["gateway", "restart"]
    assert multiplexed_profile_refusal("coder", "stop") and multiplexed_profile_refusal("coder", "start")
    assert _gateway_subcommand("other", "restart") == ["-p", "other", "gateway", "restart"]
    assert multiplexed_profile_refusal("other", "stop") is None
    # coder started its own gateway with --force: it is that gateway the verbs address.
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: True)
    assert _gateway_subcommand("coder", "restart") == ["-p", "coder", "gateway", "restart"]
    assert multiplexed_profile_refusal("coder", "stop") is None


def test_cli_stop_refuses_for_a_served_profile_without_its_own_gateway(served_root, monkeypatch):
    import hermes_cli.gateway as gw
    monkeypatch.setattr(gw, "find_gateway_pids", lambda *a, **k: [])
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit) as exc:
        gw._cmd_stop(argparse.Namespace(system=False, all=False))
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE
