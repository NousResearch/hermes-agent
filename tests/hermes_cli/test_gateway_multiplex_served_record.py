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
def served_root(tmp_path, monkeypatch, real_live_default_gateway_pid):
    root = tmp_path / "hermes"
    for name in ("coder", "other"):
        (root / "profiles" / name).mkdir(parents=True)
        (root / "profiles" / name / "config.yaml").write_text("{}\n")  # identity marker
    (root / "config.yaml").write_text("model: {default: x}\n")  # NO multiplex flag: env-only opt-in
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}))
    (root / "gateway_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
         "served_profiles": ["default", "coder"]}))
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    import hermes_constants
    import gateway.status as status
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    from hermes_cli import gateway_multiplex_served
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", lambda: os.getpid())
    return root


@pytest.fixture
def real_live_default_gateway_pid():
    from hermes_cli import gateway_multiplex_served
    return gateway_multiplex_served.live_default_gateway_pid


def test_probe_trusts_live_record_over_cli_side_config(served_root):
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    assert named_profile_served_by_running_multiplexer("coder") is True
    assert named_profile_served_by_running_multiplexer("other") is False
    # Config says multiplex on, but the running gateway did not pick up 'other': the record wins.
    (served_root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    assert named_profile_served_by_running_multiplexer("other") is False


def test_probe_falls_back_to_config_only_without_recorded_key(served_root):
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    (served_root / "gateway_state.json").write_text(json.dumps(
        {"pid": os.getpid(), "hermes_home": str(served_root), "gateway_state": "running"}))
    assert named_profile_served_by_running_multiplexer("coder") is False
    (served_root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n")
    assert named_profile_served_by_running_multiplexer("coder") is True


def test_probe_survives_a_missing_default_pid_file(served_root):
    """A launch-service-managed multiplexer can be live with no ``gateway.pid``: a replace/cleanup path
    unlinks it while the process keeps serving. Keying liveness off that file alone made every surface
    (``hermes -p X status``, ``cron list``, the dashboard ladder) say "not running" about the gateway
    that was in fact serving the profile."""
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    from hermes_cli.gateway_multiplex_served import live_default_gateway_pid
    (served_root / "gateway.pid").unlink()
    assert live_default_gateway_pid() == os.getpid()
    assert named_profile_served_by_running_multiplexer("coder") is True


def test_setup_wizard_skips_service_install_for_profile_served_by_multiplexer(
    served_root, monkeypatch, capsys,
):
    """Gateway setup must not create a standalone service for an already served profile."""
    import hermes_cli.gateway as gw

    calls: list[str] = []
    monkeypatch.setattr(gw, "_is_service_installed", lambda: False)
    monkeypatch.setattr(gw, "_is_service_running", lambda: False)
    monkeypatch.setattr(gw, "_service_backend", lambda: "systemd")
    monkeypatch.setattr(gw, "prompt_yes_no", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        gw, "install_linux_gateway_from_setup", lambda **kwargs: calls.append("install"),
    )
    monkeypatch.setattr(
        gw, "_setup_service_action", lambda *args, **kwargs: calls.append("start"),
    )

    gw._wizard_post_setup()

    assert calls == []
    assert "already served by the default multiplexer" in capsys.readouterr().out


def test_setup_gateway_service_step_skips_install_for_served_profile(served_root, monkeypatch, capsys):
    """``hermes -p <profile> setup gateway`` (and ``hermes setup`` / ``hermes import``) reach the service
    step through ``ensure_gateway_service``: a served profile gets the multiplexer note and no unit/plist,
    while a profile the live record does not list is still installed (#111958)."""
    import hermes_cli.gateway as gw

    calls: list[str] = []
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "_is_service_running", lambda: False)
    monkeypatch.setattr(gw, "_is_service_installed", lambda: False)
    monkeypatch.setattr(gw, "has_conflicting_systemd_units", lambda: False)
    monkeypatch.setattr(gw, "systemd_install", lambda **kwargs: calls.append("install"))
    monkeypatch.setattr(gw, "systemd_start", lambda *args, **kwargs: calls.append("start"))

    assert gw.ensure_gateway_service(context="setup") is True
    assert calls == []
    assert "already served by the default multiplexer" in capsys.readouterr().out

    monkeypatch.setenv("HERMES_HOME", str(served_root / "profiles" / "other"))  # not in the live record
    assert gw.ensure_gateway_service(context="setup") is True
    assert calls == ["install", "start"]


def test_recycled_pid_does_not_lend_a_stale_record_its_served_profiles(
    served_root, monkeypatch, real_live_default_gateway_pid,
):
    """A stale default record whose PID now belongs to an unrelated process (start time differs, command
    line is not a gateway's) must not make its ``served_profiles`` authoritative: bare PID existence
    once did, so `hermes -p coder gateway start` exited 78 for a multiplexer that was long gone."""
    import subprocess
    import gateway.status as status
    from hermes_cli import gateway_multiplex_served
    from hermes_cli.gateway import named_profile_served_by_running_multiplexer
    from hermes_cli.gateway_multiplex_served import recorded_served_profiles
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", real_live_default_gateway_pid)
    child = subprocess.Popen(["sleep", "60"])
    try:
        stale_start = (status._get_process_start_time(child.pid) or 10**9) - 4242
        for name in ("gateway.pid", "gateway_state.json"):
            (served_root / name).write_text(json.dumps({
                "pid": child.pid, "hermes_home": str(served_root), "gateway_state": "running",
                "start_time": stale_start, "served_profiles": ["default", "coder"]}))
        assert gateway_multiplex_served.live_default_gateway_pid() is None
        assert recorded_served_profiles(served_root) is None
        assert named_profile_served_by_running_multiplexer("coder") is False
    finally:
        child.kill()
        child.wait()


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


def test_satellite_gateway_identity_does_not_imply_cron_health(served_root, monkeypatch):
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
    assert "Scheduler host: default-profile multiplexer" in buf.getvalue()
    # A live scheduler host alone does not prove this satellite's ticker is healthy.
    assert "has not reported a heartbeat" in buf.getvalue()
    assert "will fire automatically" not in buf.getvalue()


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
    plats = profile_platforms_from_multiplexer(live.runtime, "coder")
    assert plats["telegram"] == {"state": "connected"}
    # The default's api_server is coder's too (served at /p/coder/), not a missing adapter.
    assert plats["api_server"]["state"] == "connected"
    # An unserved profile keeps the historical "stopped" answer.
    other = resolve_gateway_liveness(profile_dir=served_root / "profiles" / "other", health_probe=None, use_cache=False)
    assert other.running is False


def test_dashboard_lifecycle_verbs_target_the_multiplexer(served_root, monkeypatch):
    """`gateway restart` for a served profile restarts the multiplexer (a `-p X` child only exits 78 into
    the action log); `start`/`stop` refuse; a profile with its own gateway is managed normally."""
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_server_gateway import _gateway_subcommand, _profile_action_environment, multiplexed_profile_refusal
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda home: False)
    # This process's own HERMES_HOME is coder's; the restart child must still run under the DEFAULT
    # home (the multiplexer's) — a bare `gateway restart` here would inherit coder's home and exit 78.
    restart = _gateway_subcommand("coder", "restart")
    assert restart[-2:] == ["gateway", "restart"] and "coder" not in restart
    assert _profile_action_environment(restart)["HERMES_HOME"] == str(served_root)
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
