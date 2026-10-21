"""``hermes gateway restart`` without an installed service unit must actually restart.

Linger is a systemd-unit concern. With no unit installed the detached ``run_gateway`` IS the
restart, yet the linger check ran unconditionally and bailed with exit 0 on any Linux login
session whose linger is off — the Desktop/dashboard "Restart gateway" action then reported
success while nothing restarted and the saved credentials never loaded.
"""
from types import SimpleNamespace

import pytest


@pytest.mark.platforms("linux")
def test_restart_without_service_unit_runs_gateway_even_without_linger(monkeypatch):
    from hermes_cli import gateway as gw

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_linger_status", lambda *a, **k: (False, "Linger=no"))
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    started = []
    monkeypatch.setattr(gw, "run_gateway", lambda **k: started.append(k))

    gw._cmd_restart(SimpleNamespace(system=False, all=False))

    assert started, "no unit installed: restart must fall through to run_gateway, linger or not"


@pytest.mark.platforms("linux")
def test_detached_restart_without_service_never_runs_the_gateway_in_the_caller(monkeypatch):
    """Desktop runs `gateway restart --detach` as a bounded, piped client. In the foreground the
    client became the gateway: a Desktop child its timeout killed, holding app quit open."""
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_runtime_start as start

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    foreground, detached = [], []
    monkeypatch.setattr(gw, "run_gateway", lambda **k: foreground.append(k))
    monkeypatch.setattr(start, "spawn_unmanaged_gateway", lambda home, **k: detached.append(home))

    with pytest.raises(SystemExit) as exited:
        gw._cmd_restart(SimpleNamespace(system=False, all=False, detach=True))

    assert exited.value.code == 0
    assert not foreground, "--detach must not turn the restarting client into the gateway"
    assert detached == [gw.get_hermes_home()]


@pytest.mark.platforms("linux")
def test_restart_with_systemd_unit_still_reports_missing_linger(monkeypatch):
    from hermes_cli import gateway as gw

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: "systemd")

    def _failing_service_call(*a, **k):
        import subprocess
        raise subprocess.CalledProcessError(1, ["systemctl"])

    monkeypatch.setattr(gw, "_service_call", _failing_service_call)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: True)
    monkeypatch.setattr(gw, "get_systemd_linger_status", lambda *a, **k: (False, "Linger=no"))
    started = []
    monkeypatch.setattr(gw, "run_gateway", lambda **k: started.append(k))

    gw._cmd_restart(SimpleNamespace(system=False, all=False))

    assert not started, "a broken unit must surface the linger hint, not double-start a detached gateway"


@pytest.mark.platforms("linux")
def test_detached_restart_keeps_force_for_the_replacement(monkeypatch):
    """`restart --force --detach`: the old owner needed `--force` (a named profile under a
    multiplexer); a replacement launched without it exits on the run-side guard."""
    from hermes_cli import gateway as gw
    from hermes_cli import gateway_runtime_start as start

    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_via_service_manager_if_s6", lambda *a, **k: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gw, "stop_profile_gateway", lambda: True)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "run_gateway", lambda **k: pytest.fail("the caller became the gateway"))
    launched, real_popen = [], start.subprocess.Popen

    def _popen(cmd, **kwargs):  # only the gateway launch is faked; process scans stay real
        if "hermes_cli.main" not in cmd:
            return real_popen(cmd, **kwargs)
        launched.append(cmd)
        return SimpleNamespace()

    monkeypatch.setattr(start.subprocess, "Popen", _popen)

    with pytest.raises(SystemExit) as exited:
        gw._cmd_restart(SimpleNamespace(system=False, all=False, force=True, detach=True))

    assert exited.value.code == 0
    assert len(launched) == 1 and launched[0][-4:] == ["gateway", "run", "--quiet", "--force"]


@pytest.mark.platforms("linux")
def test_detached_restart_all_never_runs_the_host_gateway_in_the_caller(monkeypatch, tmp_path):
    """`restart --all --detach` from the root profile with no service took the in-process path.
    The detached replacement is pinned to the root: a bare child follows the sticky active_profile."""
    import sys

    import hermes_constants
    from hermes_cli import gateway as gw

    (tmp_path / "profiles" / "work").mkdir(parents=True)
    (tmp_path / "active_profile").write_text("work\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(gw, "_refuse_from_inside_gateway", lambda *a, **k: None)
    monkeypatch.setattr(gw, "_dispatch_all_via_service_manager_if_s6", lambda verb: False)
    monkeypatch.setattr(gw, "_host_multiplexer_for_all_verb", lambda: None)
    monkeypatch.setattr(gw, "_stop_installed_service", lambda system: False)
    monkeypatch.setattr(gw, "kill_gateway_processes", lambda **k: 1)
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **k: None)
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda **k: None)
    monkeypatch.setattr(gw, "_discard_dead_host_record", lambda: False)
    monkeypatch.setattr(gw, "_installed_service_kind_for", lambda windows: None)
    monkeypatch.setattr(gw, "get_python_path", lambda: sys.executable)
    monkeypatch.setattr(gw, "run_gateway", lambda **k: pytest.fail("the caller became the host gateway"))
    launched = []
    monkeypatch.setattr(gw.subprocess, "Popen", lambda cmd, **k: launched.append(cmd) or SimpleNamespace())

    gw._cmd_restart(SimpleNamespace(system=False, all=True, force=False, detach=True))

    assert len(launched) == 1
    assert launched[0][-5:] == ["--profile", "default", "gateway", "run", "--replace"]
