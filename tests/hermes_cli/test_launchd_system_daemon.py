"""System launchd jobs must not be treated as manual or replaced by user agents."""
from pathlib import Path
import plistlib
import subprocess

import pytest

from hermes_cli import gateway as gw, gateway_launchd as ld
from hermes_cli import update_cmd_fleet as fleet
from hermes_cli.update_inventory import _detect_supervisor_for_pid

pytestmark = pytest.mark.platforms("macos")


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(gw, "PROJECT_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(gw, "get_python_path", lambda: "/fixture/python")
    monkeypatch.setattr(gw, "_prepare_service_launcher", lambda **kw: None)
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    system = tmp_path / "LaunchDaemons"
    system.mkdir()
    user = tmp_path / "LaunchAgents" / "ai.hermes.gateway.plist"
    monkeypatch.setattr(ld, "SYSTEM_LAUNCHDAEMONS_DIR", system, raising=False)
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: user)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "_resolved_launchd_domain", None)
    monkeypatch.setattr(gw, "find_gateway_pids", lambda **kw: [4201])
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **kw: 4201)
    monkeypatch.setattr(gw, "_request_gateway_self_restart", lambda pid: False)
    monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", lambda *a, **kw: False)
    monkeypatch.setattr(gw, "probe_gateway_loop_liveness", lambda pid: "responsive")
    monkeypatch.setattr(gw, "_wait_for_api_server_port_free", lambda: None)
    monkeypatch.setattr(gw, "_get_restart_exit_wait_budget", lambda: 0)
    monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda pid: False)
    monkeypatch.setattr(gw, "_launchd_unsupported_marker_exists", lambda: False)
    monkeypatch.setattr(gw, "launchd_gateway_labels_for_install", lambda: ["ai.hermes.gateway"])
    plist = system / user.name
    data = {"Label": "ai.hermes.gateway", "UserName": "operator",
            "EnvironmentVariables": {"HERMES_HOME": str(home)},
            "ProgramArguments": ["hermes", "gateway", "run", "--external-supervisor"]}
    plist.write_bytes(plistlib.dumps(data))
    original = plist.read_bytes()
    calls = []
    state = {"pid": 4200, "deny": False, "replace": True, "list_denied": False}

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["launchctl", "print"]:
            if cmd[2].startswith("system/ai.hermes.gateway"):
                return subprocess.CompletedProcess(cmd, 0, f"state = running\npid = {state['pid']}\n", "")
            return subprocess.CompletedProcess(cmd, 113, "", "not found")
        if cmd[:2] == ["launchctl", "list"]:
            return subprocess.CompletedProcess(cmd, 113 if state["list_denied"] else 0, "PID Status Label\n", "")
        if cmd[:2] == ["launchctl", "kickstart"]:
            if state["deny"]:
                raise subprocess.CalledProcessError(1, cmd, stderr="Operation not permitted")
            if state["replace"]:
                state["pid"] += 100
            return subprocess.CompletedProcess(cmd, 0, "", "")
        pytest.fail(f"unexpected external command: {cmd}")

    monkeypatch.setattr(gw.subprocess, "run", run)
    # The osascript launchd leader is not the Python gateway process.
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def cmdline(self):
            return ["hermes", "gateway", "run", "--external-supervisor"] if self.pid == 4201 else ["osascript"]
        def children(self, recursive=False):
            return [Process(4201)] if self.pid == 4200 else []
    monkeypatch.setattr("psutil.Process", Process)
    return system, user, plist, original, calls, state, home


@pytest.mark.parametrize("behavior", ["snapshot", "classification", "legacy-fleet", "foreign-home", "profile-switch", "status"])
def test_system_daemon_detection(daemon, monkeypatch, capsys, behavior):
    system, user, plist, original, calls, state, home = daemon
    if behavior == "snapshot":
        snapshot = gw.get_gateway_runtime_snapshot()
        assert snapshot.service_installed and snapshot.service_running
        assert not snapshot.has_process_service_mismatch
        assert gw._is_service_installed()
    elif behavior == "status":
        gw.launchd_status()
        output = capsys.readouterr().out
        assert "system/ai.hermes.gateway" in output
        assert "supervised by launchd" in output and "login" in output
        assert "stale" not in output and "manually" not in output
    elif behavior == "profile-switch":
        assert gw.get_gateway_runtime_snapshot().service_installed
        monkeypatch.setenv("HERMES_HOME", str(home.parent / "other-install"))
        assert not gw.get_gateway_runtime_snapshot().service_installed
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert gw.get_gateway_runtime_snapshot().service_installed
    elif behavior == "classification":
        service_pids = gw._get_service_pids(all_profiles=True)
        assert {4200, 4201} <= service_pids
        assert _detect_supervisor_for_pid(4201, service_pids) == "launchd"
    elif behavior == "legacy-fleet":
        data = plistlib.loads(plist.read_bytes())
        data["Label"] = "ai.hermes.gateway-01234567"
        legacy = system / (data["Label"] + ".plist")
        legacy.write_bytes(plistlib.dumps(data))
        assert data["Label"] in gw.legacy_launchd_labels_for_install()
    else:
        from hermes_cli.update_fleet_scope import launchd_label_foreign_home
        foreign = home.parent / "other-install"
        data = plistlib.loads(plist.read_bytes())
        data["EnvironmentVariables"]["HERMES_HOME"] = str(foreign)
        plist.write_bytes(plistlib.dumps(data))
        assert launchd_label_foreign_home("ai.hermes.gateway", {home}) == str(foreign)
    assert not user.exists()


@pytest.mark.parametrize("action", ["install", "refresh", "start", "restart", "foreign-restart", "foreign-stop", "update", "ancestor-update", "delayed-self-restart", "denied", "unchanged-pid", "fleet-list-denied", "fleet-sibling", "fleet-wrapper-drain", "graceful-restart", "denied-eio", "uninstall", "denied-stop"])
def test_system_daemon_lifecycle_never_creates_user_agent(daemon, monkeypatch, action):
    system, user, plist, original, calls, state, home = daemon
    if action == "install":
        gw.launchd_install(force=True)
    elif action == "refresh":
        assert gw.refresh_launchd_plist_if_needed() is False
    elif action == "start":
        gw.launchd_start()
    elif action == "restart":
        gw.launchd_restart()
    elif action in {"foreign-restart", "foreign-stop"}:
        data = plistlib.loads(original)
        data["EnvironmentVariables"]["HERMES_HOME"] = str(home.parent / "foreign")
        plist.write_bytes(plistlib.dumps(data))
        original = plist.read_bytes()
        operation = gw.launchd_restart if action == "foreign-restart" else gw.launchd_stop
        with pytest.raises(RuntimeError, match="foreign or unreadable"):
            operation()
        assert calls == []
    elif action == "ancestor-update":
        monkeypatch.setattr(gw, "_is_pid_ancestor_of_current_process", lambda pid: True)
        monkeypatch.setattr(gw, "_request_gateway_self_restart", lambda pid: True)
        pending = set()
        assert fleet._restart_launchd_gateway_after_update(self_restart_pending=pending) == (["ai.hermes.gateway"], [])
        assert 4201 in pending, "the fleet verifier identifies the gateway, not its wrapper"
    elif action == "delayed-self-restart":
        monkeypatch.setattr(gw, "_request_gateway_self_restart", lambda pid: True)
        polls = []
        def respawn(interval):
            polls.append(state["pid"])
            state["pid"] = 4300
        monkeypatch.setattr(ld.time, "sleep", respawn)
        assert fleet._restart_launchd_gateway_after_update() == (["ai.hermes.gateway"], [])
        assert polls == [4200], "restart must settle before the verifier can run"
        assert not any(c[1] == "kickstart" for c in calls)
    elif action == "graceful-restart":
        monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", lambda *a, **kw: True)
        observed = []
        def wait(label, old_pid, timeout, *, domain):
            observed.append((old_pid, domain))
            return True
        monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", wait)
        gw.launchd_restart()
        assert observed == [(4200, "system")]
        assert not any(c[1] == "kickstart" for c in calls)
    elif action == "uninstall":
        with pytest.raises(RuntimeError, match="operator-managed"):
            gw.launchd_uninstall()
    elif action == "denied-stop":
        monkeypatch.setattr(gw, "_mark_planned_stop", lambda: None)
        monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **kw: pytest.fail("must not kill a daemon as a manual fallback"))
        original_run = gw.subprocess.run
        def run(cmd, **kw):
            if cmd[1] == "bootout":
                raise subprocess.CalledProcessError(5, cmd, stderr="Operation not permitted")
            return original_run(cmd, **kw)
        monkeypatch.setattr(gw.subprocess, "run", run)
        with pytest.raises(subprocess.CalledProcessError):
            gw.launchd_stop()
    elif action == "denied-eio":
        def deny(label, domain):
            raise subprocess.CalledProcessError(5, ["launchctl", "kickstart", "-k", f"{domain}/{label}"], stderr="Input/output error")
        # Actual subprocess error policy, including the no-detached-fallback guarantee.
        original_run = gw.subprocess.run
        def run(cmd, **kw):
            if cmd[1] == "kickstart":
                calls.append(cmd)
                deny(cmd[-1].split("/")[-1], "system")
            return original_run(cmd, **kw)
        monkeypatch.setattr(gw.subprocess, "run", run)
        with pytest.raises(subprocess.CalledProcessError):
            gw.launchd_restart()
    elif action.startswith("fleet-"):
        state["list_denied"] = action == "fleet-list-denied"
        drained = []
        if action == "fleet-wrapper-drain":
            def graceful(pid, **kw):
                drained.append(pid)
                state["pid"] += 100
                return True
            monkeypatch.setattr(gw, "_graceful_restart_via_sigusr1", graceful)
        if action == "fleet-sibling":
            data = plistlib.loads(original)
            data["Label"] = "ai.hermes.gateway-01234567"
            (system / (data["Label"] + ".plist")).write_bytes(plistlib.dumps(data))
        monkeypatch.setattr(fleet, "_gateway_home_for_pid", lambda pid: home)
        restarted, failed = [], []
        if action == "fleet-wrapper-drain":
            monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway-other")
            monkeypatch.setattr(fleet, "_restart_launchd_gateway_after_update", lambda **kw: ([], []))
        fleet._restart_macos_launchd_gateways(restarted, failed, 0, require_supervision=True)
        assert "ai.hermes.gateway" in restarted and failed == []
        if action == "fleet-wrapper-drain":
            assert drained == [4201], "SIGUSR1 must reach Python, never its osascript leader"
        if action == "fleet-sibling":
            assert "ai.hermes.gateway-01234567" in restarted
            assert any(c[-1] == "system/ai.hermes.gateway-01234567" and c[1] == "kickstart" for c in calls)
    else:
        state["deny"] = action == "denied"
        state["replace"] = action != "unchanged-pid"
        if action == "unchanged-pid":
            # Exercise the real poll once, without a twenty-second test delay.
            real_wait = gw.wait_for_launchd_gateway_supervision
            monkeypatch.setattr(gw, "wait_for_launchd_gateway_supervision",
                                lambda **kw: real_wait(timeout=0, **kw))
        restarted, failed = fleet._restart_launchd_gateway_after_update()
        if action == "update":
            assert restarted == ["ai.hermes.gateway"] and failed == []
        else:
            assert restarted == [] and failed == ["ai.hermes.gateway"]
    assert plist.read_bytes() == original
    assert not user.exists()
    assert not any(c[1] in {"bootstrap", "bootout", "submit"} for c in calls)
    if action in {"start", "restart", "update", "denied", "unchanged-pid"}:
        assert any(c[:2] == ["launchctl", "kickstart"] and c[-1] == "system/ai.hermes.gateway" for c in calls)
