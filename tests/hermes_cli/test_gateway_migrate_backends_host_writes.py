"""The host operations the backend migration uses to change the machine (``SystemHost``).

A hand-started old backend is stopped by its recorded pid only while that pid is still the same process
(a real child process here; a pid that now names another process, or one with no recorded start time, is
never signalled). A launchd job is retired with ``launchctl disable`` plus a ``.disabled`` rename of its real
plist file, and re-enabled exactly (rename back, then ``launchctl enable``) on rollback and restore. Where a
service manager answers (``launchctl``, ``systemctl``, ``systemd-run``) it is the one stand-in; files, modes
and processes are real.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import gateway_migrate_backends_host as host_module
from hermes_cli.gateway_migrate_backends_host import SystemHost
from tests.waiting import wait_until


@pytest.fixture
def sleeper():
    """A hand-started backend stand-in. A thread reaps it the moment it exits (as the shell that started
    a real one does), so an exited process never lingers as this test's zombie."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    reaper = threading.Thread(target=proc.wait, daemon=True)
    reaper.start()
    yield proc
    if proc.poll() is None:
        proc.kill()
    reaper.join(timeout=10)


@pytest.mark.platforms("linux")
def test_a_hand_started_backend_is_stopped_only_while_its_pid_is_still_the_same_process(tmp_path, sleeper):
    host = SystemHost(tmp_path)
    started = host.start_time(sleeper.pid)
    assert started is not None

    host._stop_manual({"supervisor": "manual", "pid": sleeper.pid, "start_time": started + 1})  # pid reused
    assert sleeper.poll() is None  # another process now: never signalled

    host._stop_manual({"supervisor": "manual", "pid": sleeper.pid, "start_time": started})
    wait_until(lambda: sleeper.poll() is not None, timeout=30, what="the old backend to exit")
    assert not host.same_process(sleeper.pid, started)


def test_a_launchd_job_is_disabled_and_set_aside_then_enabled_exactly_again(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def launchctl(args):
        calls.append(list(args))
        return subprocess.CompletedProcess(["launchctl", *args], 0, "", "")

    monkeypatch.setattr(host_module, "_launchctl", launchctl)
    plist = tmp_path / "local.hermes-dashboard.plist"
    plist.write_bytes(b"<plist/>")
    rec = {"supervisor": "launchd", "scope": "gui/501", "unit": "local.hermes-dashboard", "unit_path": str(plist)}
    host = SystemHost(tmp_path)

    host.old_disable(rec)
    assert calls == [["disable", "gui/501/local.hermes-dashboard"]]
    assert not plist.exists() and plist.with_name(plist.name + ".disabled").read_bytes() == b"<plist/>"

    host.old_enable(rec)
    assert calls[-1] == ["enable", "gui/501/local.hermes-dashboard"]
    assert plist.read_bytes() == b"<plist/>" and not plist.with_name(plist.name + ".disabled").exists()


def test_without_a_recorded_start_time_a_live_pid_is_never_the_same_process(tmp_path, sleeper):
    host = SystemHost(tmp_path)

    assert host.same_process(sleeper.pid, None) is False
    with pytest.raises(RuntimeError, match="no recorded start time"):
        host._stop_manual({"supervisor": "manual", "pid": sleeper.pid, "start_time": None})
    assert sleeper.poll() is None


class _LaunchdGatewayHost(SystemHost):
    def gateway_kind(self):
        return "launchd"


def test_on_macos_the_gateway_agent_is_written_but_not_loaded_before_the_old_job_is_out(tmp_path, monkeypatch):
    from hermes_cli import gateway as gw
    installs: list[dict] = []
    monkeypatch.setattr(gw, "launchd_install", lambda **kw: installs.append(kw))

    _LaunchdGatewayHost(tmp_path).install_gateway()

    assert installs == [{"start_now": False}]  # RunAtLoad: loading would start it beside the old job


def _systemctl_stub(states):
    calls: list[list[str]] = []

    def systemctl(scope, args, timeout=30):
        calls.append(list(args))
        out = states.pop(0) + "\n" if args[0] == "is-enabled" else ""
        return subprocess.CompletedProcess(["systemctl", *args], 0, out, "")
    return systemctl, calls


@pytest.mark.parametrize(("state", "disable", "enable"), [
    ("enabled", ["disable", "u.service"], ["enable", "u.service"]),
    ("enabled-runtime", ["disable", "--runtime", "u.service"], ["enable", "--runtime", "u.service"]),
    ("static", None, None), ("disabled", None, None)])
def test_a_systemd_unit_gets_back_exactly_the_enablement_it_had(tmp_path, monkeypatch, state, disable, enable):
    systemctl, calls = _systemctl_stub([state])
    monkeypatch.setattr(host_module, "_systemctl", systemctl)
    rec = {"supervisor": "systemd", "scope": "user", "unit": "u.service", "unit_file_state": state}
    host = SystemHost(tmp_path)

    host.old_disable(rec)
    host.old_enable(rec)

    assert calls == [c for c in (disable, enable) if c] + [["is-enabled", "u.service"]]


def test_a_restore_that_does_not_reach_the_recorded_enablement_fails(tmp_path, monkeypatch):
    systemctl, _calls = _systemctl_stub(["enabled"])  # it was enabled-runtime: now persistent
    monkeypatch.setattr(host_module, "_systemctl", systemctl)
    rec = {"supervisor": "systemd", "scope": "user", "unit": "u.service", "unit_file_state": "enabled-runtime"}

    with pytest.raises(RuntimeError, match="not enabled-runtime as before"):
        SystemHost(tmp_path).old_enable(rec)


class _InsideUnitHost(SystemHost):
    def gateway_kind(self):
        return "systemd"

    def own_unit(self):
        return "hermes-dashboard.service"


def test_a_detached_run_is_never_started_inside_the_callers_unit(tmp_path, monkeypatch):
    def failing_systemd_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, "", "Failed to start transient service unit")

    def popen(*a, **kw):
        raise AssertionError("fell back to a new session inside the caller's unit")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(host_module.subprocess, "run", failing_systemd_run)
    monkeypatch.setattr(host_module.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="outside hermes-dashboard.service"):
        _InsideUnitHost(tmp_path).spawn_detached(["gateway", "migrate", "--backends", "--apply"], tmp_path / "log")


def test_existing_env_is_private_before_the_first_secret_is_written(tmp_path, monkeypatch):
    from hermes_cli import config
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=x\n", encoding="utf-8")
    os.chmod(env, 0o644)
    modes: list[int] = []
    real = config.save_env_value

    def spy(key, value):
        modes.append(stat.S_IMODE(Path(env).stat().st_mode))
        real(key, value)

    monkeypatch.setattr(config, "save_env_value", spy)

    SystemHost(tmp_path).set_env({"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "u",
                                  "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": "t2-throwaway-pw"})

    assert modes == [0o600, 0o600] and stat.S_IMODE(env.stat().st_mode) == 0o600


def test_a_live_hand_started_backend_is_found_only_by_a_backend_command_line_on_the_port(tmp_path, monkeypatch):
    from hermes_cli.gateway_migrate_backends import BackendItem

    def item(pid, argv):
        return BackendItem(supervisor="manual", running=True, pid=pid, argv=argv, this_home=True)

    host = SystemHost(tmp_path)
    monkeypatch.setattr(host, "backends", lambda: [
        item(11, ["/venv/bin/hermes", "chat"]),  # not a backend: no address at all
        item(12, ["/venv/bin/hermes", "dashboard", "--port", "9200"]),
        item(13, ["/venv/bin/hermes", "dashboard", "--no-open", "--port", "9119"])])
    monkeypatch.setattr(host, "pid_listens_on", lambda pid, port: True)
    monkeypatch.setattr(host, "start_time", lambda pid: 100 + pid)

    assert host.live_manual_backend(9119) == {"pid": 13, "start_time": 113}
    assert host.live_manual_backend(9300) is None


def test_a_hand_started_backend_is_started_again_from_its_raw_command_line(tmp_path, monkeypatch):
    started: list[list[str]] = []

    class Child:
        def __init__(self, argv, **kw):
            started.append(list(argv))
            self.pid = 4321

    monkeypatch.setattr(host_module.subprocess, "Popen", Child)
    monkeypatch.setattr(host_module, "profile_args", lambda home: [])
    raw = ["/venv/bin/hermes", "dashboard", "--no-open", "--port", "9119", "--auth-token", "tok-raw"]
    rec = {"supervisor": "manual", "argv": [*raw[:-1], "<redacted>"]}  # what the report and manifest show

    assert SystemHost(tmp_path).old_start(rec, {"PATH": "/usr/bin"}, raw) == 4321
    assert started[0][-2:] == ["--auth-token", "tok-raw"]
    with pytest.raises(RuntimeError, match="start it again by hand"):
        SystemHost(tmp_path).old_start(rec, {"PATH": "/usr/bin"}, None)


def test_a_hand_started_backend_is_never_started_with_the_callers_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(host_module.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
    rec = {"supervisor": "manual", "argv": ["/venv/bin/hermes", "dashboard"]}

    with pytest.raises(RuntimeError, match="environment was not recorded"):
        SystemHost(tmp_path).old_start(rec, None, ["/venv/bin/hermes", "dashboard"])


def test_no_gateway_identity_is_unknown_not_absent(tmp_path, monkeypatch):
    from gateway import status
    monkeypatch.setattr(status, "live_gateway_pid_for_home", lambda home: None)  # also an unreadable record

    assert SystemHost(tmp_path).gateway_listens(9119) is None




def test_the_post_move_check_reads_the_gateways_status_without_a_proxy(tmp_path, monkeypatch):
    """The plan's probe of an old backend never calls /api/status (it reads /api/health); the verification
    of the migrated gateway does, for its gateway_state, and like the probe never through a proxy."""
    import io
    import json
    import urllib.request

    seen, handlers = [], []

    class Answer(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class Opener:
        def open(self, request, timeout=None):
            seen.append(request.full_url)
            return Answer(json.dumps({"gateway_state": "running", "auth_required": True}).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *h: (handlers.extend(h), Opener())[1])

    got = SystemHost(tmp_path).gateway_status("127.0.0.1", 9119)

    assert got["gateway_state"] == "running" and seen == ["http://127.0.0.1:9119/api/status"]
    (handler,) = handlers
    assert isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
