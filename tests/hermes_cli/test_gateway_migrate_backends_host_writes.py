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
from types import SimpleNamespace

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

    _located_systemd_run(monkeypatch)
    monkeypatch.setattr(host_module.subprocess, "run", failing_systemd_run)
    monkeypatch.setattr(host_module.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="outside hermes-dashboard.service"):
        _InsideUnitHost(tmp_path).spawn_detached(["gateway", "migrate", "--backends", "--apply"], tmp_path / "log")


def _located_systemd_run(monkeypatch, path="/opt/managed/bin/systemd-run"):
    from hermes_platform import resolver
    monkeypatch.setattr(resolver, "locate_command",
                        lambda name, *a, **kw: SimpleNamespace(command=path if name == "systemd-run" else None))


def test_a_detached_run_uses_the_managed_systemd_run_and_the_shared_child_env(tmp_path, monkeypatch):
    """The transient unit is started with the resolver's systemd-run (not a bare PATH lookup), and its env
    comes from build_subprocess_env, pinned to this home."""
    from tools.environments import local
    _located_systemd_run(monkeypatch)
    built: list[dict] = []
    real_build = local.build_subprocess_env
    monkeypatch.setattr(local, "build_subprocess_env", lambda *a, **kw: (built.append(kw), real_build(*a, **kw))[1])
    ran: list[list[str]] = []
    monkeypatch.setattr(host_module.subprocess, "run",
                        lambda cmd, **kw: (ran.append(cmd), subprocess.CompletedProcess(cmd, 0, "", ""))[1])

    out = _InsideUnitHost(tmp_path).spawn_detached(["gateway", "migrate", "--backends", "--apply"], tmp_path / "log")

    assert out["unit"].startswith("hermes-backend-migration-")
    assert ran[0][0] == "/opt/managed/bin/systemd-run"
    assert f"--setenv=HERMES_HOME={tmp_path}" in ran[0]
    assert built == [{"scrub_secrets": False, "inherit_profile_home": False, "extra": {"HERMES_HOME": str(tmp_path)}}]


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


def test_the_working_directory_is_read_without_proc(tmp_path, monkeypatch):
    """macOS and Windows have no /proc: psutil reads it there."""
    import psutil

    def no_proc(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(host_module.os, "readlink", no_proc)
    monkeypatch.setattr(psutil.Process, "cwd", lambda self: "/Users/u/.hermes/hermes-agent")

    assert SystemHost(tmp_path).process_cwd(os.getpid()) == "/Users/u/.hermes/hermes-agent"



def test_the_backend_a_rollback_started_again_is_hand_started_for_this_home(tmp_path, monkeypatch):
    """macOS: the respawned backend's parent is launchd (ppid 1), which the scan reports as "unknown"."""
    import json

    from hermes_cli import gateway_migrate_backends as scan
    from hermes_cli.gateway_migrate_backends import BackendItem

    items = [BackendItem(supervisor="unknown", running=True, pid=5001, argv=["hermes", "dashboard"]),
             BackendItem(supervisor="unknown", running=True, pid=5002, argv=["hermes", "dashboard"])]
    monkeypatch.setattr(scan, "collect_backends", lambda: items)
    (tmp_path / "backend_migration.json").write_text(json.dumps({
        "version": 1, "state": "rolled_back", "steps": [], "old_backend": {}, "gateway": {}, "backups": {},
        "dial": {}, "home": str(tmp_path), "old_backend_respawned": {"pid": 5001, "start_time": 6001}}),
        encoding="utf-8")
    host = SystemHost(tmp_path)
    monkeypatch.setattr(host, "start_time", lambda pid: 1000 + pid)

    found = host.backends()

    assert [(i.pid, i.supervisor, i.this_home) for i in found] == [(5001, "manual", True), (5002, "unknown", None)]


@pytest.mark.parametrize("case", ["cloned-manifest", "scan-says-another-home"])
def test_a_respawned_backend_is_never_claimed_for_another_home(tmp_path, monkeypatch, case):
    """`hermes profile create --clone-all` copies backend_migration.json: the clone must not claim (and so
    later stop) the source profile's backend; and a backend the scan placed in another home stays there."""
    import json

    from hermes_cli import gateway_migrate_backends as scan
    from hermes_cli.gateway_migrate_backends import BackendItem

    source, clone = tmp_path / "source", tmp_path / "profiles" / "clone"
    source.mkdir()
    clone.mkdir(parents=True)
    home = clone if case == "cloned-manifest" else source
    item = BackendItem(supervisor="unknown", running=True, pid=5001, argv=["hermes", "dashboard"],
                       this_home=None if case == "cloned-manifest" else False)
    monkeypatch.setattr(scan, "collect_backends", lambda: [item])
    (home / "backend_migration.json").write_text(json.dumps({
        "version": 1, "state": "restored", "steps": [], "old_backend": {}, "gateway": {}, "backups": {},
        "dial": {}, "home": str(source), "old_backend_respawned": {"pid": 5001, "start_time": 6001}}),
        encoding="utf-8")
    host = SystemHost(home)
    monkeypatch.setattr(host, "start_time", lambda pid: 1000 + pid)

    (found,) = host.backends()

    assert found.supervisor == "unknown" and found.this_home is not True


def _foreign_same_named_unit(tmp_path, monkeypatch):
    """A custom root's ``profiles/work`` beside a user unit ``hermes-gateway-work`` that pins the account's
    own ``~/.hermes/profiles/work``: same unit name, another home."""
    from hermes_cli import gateway as gw
    account = tmp_path / "account"
    mine = tmp_path / "srv" / "team" / "profiles" / "work"
    other = account / ".hermes" / "profiles" / "work"
    for d in (mine, other):
        d.mkdir(parents=True)
    unit = tmp_path / "units" / "hermes-gateway-work.service"
    unit.parent.mkdir()
    unit.write_text(f"[Service]\nEnvironment=HERMES_HOME={other}\nExecStart=/usr/bin/hermes gateway run\n",
                    encoding="utf-8")
    monkeypatch.setenv("HOME", str(account))
    monkeypatch.setenv("HERMES_HOME", str(mine))
    monkeypatch.setattr(gw, "get_systemd_unit_path", lambda system=False: unit)
    monkeypatch.setattr(gw, "get_service_name", lambda: "hermes-gateway-work")
    calls: list = []
    monkeypatch.setattr(host_module, "_systemctl", lambda *a, **kw: calls.append(a) or subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(gw, "systemd_uninstall", lambda **kw: calls.append(("uninstall", kw)))

    class _Host(SystemHost):
        def gateway_kind(self):
            return "systemd"

    return _Host(mine), str(other), calls


def test_a_same_named_unit_of_another_home_is_found_and_never_driven(tmp_path, monkeypatch):
    host, other, calls = _foreign_same_named_unit(tmp_path, monkeypatch)

    assert host.foreign_gateway_home() == other
    for verb in ("restart", "stop", "start"):
        with pytest.raises(RuntimeError, match="belongs to another Hermes home"):
            host.gateway_verb(verb)
    with pytest.raises(RuntimeError, match="belongs to another Hermes home"):
        host.uninstall_gateway()
    assert calls == []


def test_the_unit_of_this_home_is_driven(tmp_path, monkeypatch):
    host, _other, calls = _foreign_same_named_unit(tmp_path, monkeypatch)
    from hermes_cli import gateway as gw
    gw.get_systemd_unit_path().write_text(f"[Service]\nEnvironment=HERMES_HOME={host.home}\n", encoding="utf-8")

    assert host.foreign_gateway_home() is None
    host.gateway_verb("restart")
    assert calls and calls[0][1] == ["restart", "hermes-gateway-work"]


def test_a_launchagent_of_another_home_is_found_and_never_driven(tmp_path, monkeypatch):
    from hermes_cli import gateway as gw
    mine, other = tmp_path / "srv" / "team" / "profiles" / "work", tmp_path / "account" / ".hermes" / "profiles" / "work"
    for d in (mine, other):
        d.mkdir(parents=True)
    plist = tmp_path / "ai.hermes.gateway-work.plist"
    plist.write_text(f"<plist><dict><key>EnvironmentVariables</key><dict><key>HERMES_HOME</key>"
                     f"<string>{other}</string></dict></dict></plist>", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(mine))
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist)
    called: list = []
    monkeypatch.setattr(gw, "_service_call", lambda *a: called.append(a))
    monkeypatch.setattr(gw, "launchd_uninstall", lambda: called.append("uninstall"))

    class _Host(SystemHost):
        def gateway_kind(self):
            return "launchd"

    host = _Host(mine)
    assert host.foreign_gateway_home() == str(other)
    with pytest.raises(RuntimeError, match="belongs to another Hermes home"):
        host.gateway_verb("restart")
    with pytest.raises(RuntimeError, match="belongs to another Hermes home"):
        host.uninstall_gateway()
    assert called == []
