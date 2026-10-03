"""Linux acceptance: real updater scope/lock/completion versus disposable services.

No source update or production service is executed. Only the completion payload is
an inert fixture; isolation, lock, parent transport and receipt APIs are real.
Set HERMES_E2E_ARTIFACTS to retain commands, observations and inherited stdio.
These acceptance tests exercise an already implemented fix, not its original RED.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

import pytest


REPO = Path(__file__).resolve().parents[2]

# The launcher remains in the service; its update subprocess is the thing under test.
LAUNCHER = r'''
import json, os, signal, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
def atomic(name, data):
    p = root / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(data))
    t.replace(p)
atomic('launcher.json', {'pid': os.getpid(), 'cgroup': Path('/proc/self/cgroup').read_text(),
                          'invocation': os.environ['INVOCATION_ID']})
try:
    fd = os.open(root / 'launched.once', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
except FileExistsError:
    pass
else:
    os.close(fd)
    command = json.loads((root / 'command.json').read_text())
    with (root / 'stdout.log').open('wb') as out, (root / 'stderr.log').open('wb') as err:
        subprocess.Popen(command, stdout=out, stderr=err)
signal.pause()
'''

WORKER = r'''
import json, os, sys
from pathlib import Path
repo, directory, mode = sys.argv[1:4]
root = Path(directory)
sys.path.insert(0, repo)
def atomic(name, data):
    p = root / name
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(data))
    t.replace(p)
def identity():
    return {'pid': os.getpid(), 'cgroup': Path('/proc/self/cgroup').read_text(),
            'argv': sys.argv, 'orig_argv': sys.orig_argv, 'cwd': os.getcwd(),
            'home': os.environ['HERMES_HOME'], 'action': os.environ['HERMES_ACTION_ID'],
            'invocation': os.environ.get('INVOCATION_ID'),
            'stdout': os.readlink('/proc/self/fd/1'), 'stderr': os.readlink('/proc/self/fd/2')}
# Preserve the original service identity across the re-exec, atomically.
if not (root / 'before.json').exists():
    atomic('before.json', identity())
if mode != 'unsafe':
    from hermes_cli.update_process import isolate_update_process
    isolate_update_process()
atomic('isolated.json', identity())
from hermes_cli.update_lock import UpdateLock
from hermes_cli import update_receipt
from hermes_cli.update_completion import run_completion
marker = Path(os.environ['HERMES_HOME']) / '.hermes-update-in-progress'
assert not marker.exists(), 'isolation must precede the mutating handler/lock'
with UpdateLock() as lock:
    assert lock.acquired
    update_receipt.begin_update_receipt(correlation_id=os.environ['HERMES_ACTION_ID'])
    receipt = update_receipt._current.get().data
    atomic('parent.json', {**identity(), 'receipt': receipt, 'lock': str(marker)})
    print('PARENT_BEFORE_COMPLETION', flush=True)
    print('PARENT_STDERR_BEFORE_COMPLETION', file=sys.stderr, flush=True)
    result = run_completion({'schema': 1, 'source': str(root / 'fixture'),
                             'home': os.environ['HERMES_HOME'], 'receipt': receipt,
                             'repository': repo, 'directory': directory,
                             'socket': json.loads((root / 'socket.json').read_text())})
    atomic('result.json', result)
    assert result['exit_code'] == 0 and result['receipt']['outcome'] == 'success'
    # The real completion published the terminal receipt; do not finalize twice.
    print('PARENT_AFTER_COMPLETION', flush=True)
    print('PARENT_STDERR_AFTER_COMPLETION', file=sys.stderr, flush=True)
atomic('done.json', {'exit_code': 0, 'lock_exists': marker.exists(), **identity()})
'''

COMPLETION = r'''
import json, os, socket, sys
from pathlib import Path
request_path, result_path = map(Path, sys.argv[1:3])
request = json.loads(request_path.read_text())
root = Path(request['directory'])
sys.path.insert(0, request['repository'])
def atomic(path, data):
    t = path.with_suffix('.tmp')
    t.write_text(json.dumps(data))
    t.replace(path)
sock = socket.socket(socket.AF_UNIX)
sock.bind('\0' + request['socket'])
sock.listen(1)
sock.settimeout(20)
atomic(root / 'child.json', {'pid': os.getpid(), 'cgroup': Path('/proc/self/cgroup').read_text(),
                            'home': os.environ['HERMES_HOME'], 'action': os.environ['HERMES_ACTION_ID'],
                            'sid': os.getsid(0), 'request': request})
print('CHILD_READY_STDOUT', flush=True)
print('CHILD_READY_STDERR', file=sys.stderr, flush=True)
try:
    connection, _ = sock.accept()
    with connection:
        connection.settimeout(3)
        assert connection.recv(64) == b'release'
        connection.sendall(b'accepted')
finally:
    sock.close()
from hermes_cli.update_completion import _resume_receipt
from hermes_cli.update_receipt import finalize_pending_update_receipt
_resume_receipt(request['receipt'])
receipt_path = finalize_pending_update_receipt(0, 'harmless live topology acceptance')
assert receipt_path is not None
receipt = json.loads(receipt_path.read_text())
atomic(result_path, {'schema': 1, 'update_id': request['receipt']['update_id'],
                    'exit_code': 0, 'receipt': receipt, 'windows_resume': None})
print('CHILD_AFTER_RELEASE_STDOUT', flush=True)
print('CHILD_AFTER_RELEASE_STDERR', file=sys.stderr, flush=True)
'''


def _wait(observe, *, timeout=20):
    """Poll an actual predicate, not a blind startup sleep."""
    deadline = time.monotonic() + timeout
    while True:
        value = observe()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(f"readiness deadline ({timeout}s): {observe}")
        time.sleep(0.025)


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _process(pid):
    """A surviving zombie is not a running updater; access errors are errors."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2:].split()[0]
    except FileNotFoundError:
        return None


def _live(pid):
    return _process(pid) not in (None, "Z", "X")


def _cgroup(pid):
    return Path(f"/proc/{pid}/cgroup").read_text()


class OwnedRun:
    def __init__(self, root, home, bus_env):
        self.root, self.home, self.bus_env = root, home, bus_env
        self.token = uuid.uuid4().hex
        self.unit = f"hermes-scope-live-test-{self.token}.service"
        self.scope = None
        self.records = []
        self.proc = None
        root.mkdir()
        (root / "fixture/hermes_cli").mkdir(parents=True)
        (root / "os-home").mkdir()
        home.mkdir(parents=True, exist_ok=True)
        for name, script in (("launcher.py", LAUNCHER), ("worker.py", WORKER),
                             ("fixture/hermes_cli/update_completion.py", COMPLETION)):
            (root / name).write_text(script)
        (root / "socket.json").write_text(json.dumps(f"hermes-scope-live-{self.token}"))

    def note(self, assertion, **facts):
        self.records.append({"assertion": assertion, **facts})
        # Persist immediately, even when a later assertion fails.
        (self.root / "evidence.json").write_text(json.dumps(self.records, indent=2))

    def command(self, command, *, check=True):
        result = subprocess.run(command, env=self.bus_env, capture_output=True,
                                text=True, timeout=20)
        self.note("command execution", command=command, exit_code=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
        if check:
            assert result.returncode == 0, result.stderr
        return result

    def state(self, unit):
        output = self.command(["systemctl", "--user", "show", unit,
                               "-p", "LoadState", "-p", "ActiveState", "-p", "MainPID",
                               "-p", "ControlGroup", "-p", "KillMode", "-p", "TimeoutStopUSec"],
                              check=False).stdout
        return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)

    def start(self, mode="safe", *, service=True):
        env = {"HOME": str(self.root / "os-home"), "HERMES_HOME": str(self.home),
               "HERMES_ACTION_ID": self.token, "PYTHONUNBUFFERED": "1",
               "TMPDIR": str(self.root), "PATH": os.environ["PATH"]}
        argv = [sys.executable, "-u", str(self.root / "worker.py"), str(REPO),
                str(self.root), mode, "update", "--profile", "argument with spaces", "λ",
                "${HOME}", "$HOME", "${HERMES_ACTION_ID}", "two ${HOME} spaces",
                "$$", "${HERMES_SCOPE_UNSET_LITERAL}", ""]
        (self.root / "command.json").write_text(json.dumps(argv))
        if service:
            command = ["systemd-run", "--user", "--unit", self.unit, "--collect",
                       "--property", "KillMode=mixed", "--property", "TimeoutStopSec=500ms",
                       "--property", f"WorkingDirectory={self.root}",
                       "--property", f"StandardOutput=append:{self.root / 'launcher.log'}",
                       "--property", f"StandardError=append:{self.root / 'launcher.log'}"]
            command += [f"--setenv={key}={value}" for key, value in env.items()]
            command += [sys.executable, "-u", str(self.root / "launcher.py"), str(self.root)]
            self.command(command)
            launcher = _wait(lambda: _json(self.root / "launcher.json"))
            assert self.unit in launcher["cgroup"]
            state = self.state(self.unit)
            assert state["KillMode"] == "mixed" and state["TimeoutStopUSec"] == "500ms"
            self.note("launching service owns launcher", launcher=launcher, state=state)
        else:
            with (self.root / "stdout.log").open("wb") as out, (self.root / "stderr.log").open("wb") as err:
                self.proc = subprocess.Popen(argv, cwd=self.root, env=env, stdout=out, stderr=err)
            self.note("external ordinary-shell subprocess", command=argv, invocation_absent=True)
        isolated = _wait(lambda: _json(self.root / "isolated.json"))
        if service and mode == "safe":
            match = re.search(r"/(hermes-worker-update-[0-9a-f]{32}\.scope)(?:/|\n)", isolated["cgroup"])
            assert match, isolated
            self.scope = match[1]
            assert self.unit not in isolated["cgroup"]
            self.note("helper moved updater into its own real scope", scope=self.scope, isolated=isolated)
        parent = _wait(lambda: _json(self.root / "parent.json"))
        child = _wait(lambda: _json(self.root / "child.json"))
        before = _json(self.root / "before.json")
        for key in ("orig_argv", "argv", "cwd", "home", "action", "stdout", "stderr"):
            assert parent[key] == before[key], key
        assert parent["orig_argv"] == argv
        assert parent["home"] == child["home"] == str(self.home)
        assert parent["action"] == child["action"] == self.token
        assert parent["cgroup"] == child["cgroup"]
        assert child["sid"] == child["pid"]  # real run_completion starts a new session
        assert self.live_lock(parent)
        self.note("argv/cwd/stdio/home/action preserved; parent lock and completion ready",
                  before=before, parent=parent, child=child)
        return parent, child

    def live_lock(self, parent):
        from hermes_cli.update_lock import read_live_update
        holder = read_live_update(path=Path(parent["lock"]))
        return holder is not None and holder.pid == parent["pid"]

    def release(self):
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(3)
            connection.connect("\0" + _json(self.root / "socket.json"))
            connection.sendall(b"release")
            assert connection.recv(64) == b"accepted"
        done = _wait(lambda: _json(self.root / "done.json"))
        result = _json(self.root / "result.json")
        latest = _json(self.home / "logs/update_receipts/latest.json")
        assert done["exit_code"] == result["exit_code"] == latest["exit_code"] == 0
        assert not done["lock_exists"]
        assert result["schema"] == 1
        assert result["update_id"] == result["receipt"]["update_id"] == latest["update_id"] == self.token
        assert latest == result["receipt"]
        assert latest["outcome"] == "success" and latest["finished_at"]
        archive = list((self.home / "logs/update_receipts").glob(f"update_*_{self.token}.json"))
        assert len(archive) == 1 and _json(archive[0]) == latest
        _wait(lambda: not _live(done["pid"]))
        child = _json(self.root / "child.json")
        _wait(lambda: not _live(child["pid"]))
        stdout, stderr = (self.root / "stdout.log").read_text(), (self.root / "stderr.log").read_text()
        for marker in ("PARENT_BEFORE_COMPLETION", "PARENT_AFTER_COMPLETION", "CHILD_READY_STDOUT",
                       "CHILD_READY_STDERR", "CHILD_AFTER_RELEASE_STDOUT", "CHILD_AFTER_RELEASE_STDERR"):
            assert marker in stdout, marker
        for marker in ("PARENT_STDERR_BEFORE_COMPLETION", "PARENT_STDERR_AFTER_COMPLETION"):
            assert marker in stderr, marker
        if self.proc is not None:
            assert self.proc.wait(timeout=3) == 0
        self.note("completion released; correlated terminal success; lock removed; inherited stdio persisted",
                  done=done, result=result, archive=str(archive[0]), stdout=stdout, stderr=stderr)

    def cleanup(self):
        # Never glob-stop units: names must have been created/observed by this run.
        isolated = _json(self.root / "isolated.json")
        if self.scope is None and isolated:
            match = re.search(r"/(hermes-worker-update-[0-9a-f]{32}\.scope)(?:/|\n)", isolated["cgroup"])
            if match:
                self.scope = match[1]
        assert self.unit == f"hermes-scope-live-test-{self.token}.service"
        units = [self.unit] + ([self.scope] if self.scope else [])
        for unit in units:
            self.command(["systemctl", "--user", "stop", unit], check=False)
            self.command(["systemctl", "--user", "reset-failed", unit], check=False)
            state = _wait(lambda: self._collected(unit))
            assert state["LoadState"] == "not-found" and state["ActiveState"] == "inactive"
            self.note("owned unit collected", unit=unit, state=state)
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=3)
        for name in ("launcher.json", "before.json", "isolated.json", "parent.json", "child.json"):
            record = _json(self.root / name)
            if record:
                _wait(lambda: not _live(record["pid"]))
                self.note("owned process not running", record=name, pid=record["pid"], state=_process(record["pid"]))
        # Abstract Unix sockets disappear on process exit; confirm it cannot be reached.
        with socket.socket(socket.AF_UNIX) as connection:
            connection.settimeout(1)
            with pytest.raises(ConnectionRefusedError):
                connection.connect("\0" + _json(self.root / "socket.json"))
        self.note("test-owned abstract socket gone", socket=_json(self.root / "socket.json"))
        artifacts = os.environ.get("HERMES_E2E_ARTIFACTS")
        if artifacts:
            target = Path(artifacts) / f"scope-live-{self.root.name}-{self.token}"
            shutil.copytree(self.root, target)
            shutil.copytree(self.home, target / "receipt-home")
            print(f"LIVE_EVIDENCE={target}")

    def _collected(self, unit):
        state = self.state(unit)
        return state if state.get("LoadState") == "not-found" and state.get("ActiveState") == "inactive" else None


@pytest.fixture
def bus_env():
    from tools.process_registry import systemd_user_bus_env
    env = systemd_user_bus_env()
    if shutil.which("systemd-run") is None:
        pytest.skip("requires a reachable local systemd user manager")
    result = subprocess.run(["systemctl", "--user", "show-environment"], env=env,
                            capture_output=True, text=True, timeout=5)
    if result.returncode:
        pytest.skip(f"requires a reachable local systemd user manager: {result.stderr}")
    return env


@pytest.mark.platforms("linux")
def test_updater_and_completion_survive_owned_service_stop_restart_profiles_a_b_a(tmp_path, bus_env):
    # Two actual homes, selected A -> B -> A across distinct source-updater processes.
    for index, (operation, profile) in enumerate((("stop", "A"), ("restart", "B"), ("stop", "A"))):
        run = OwnedRun(tmp_path / f"safe-{index}-{operation}-{profile}", tmp_path / f"profile-{profile}", bus_env)
        try:
            parent, child = run.start()
            original_launcher = _json(run.root / "launcher.json")
            observer = _json(run.root / "before.json")
            assert observer["pid"] != parent["pid"]
            assert run.unit in observer["cgroup"] and run.unit not in parent["cgroup"]
            assert run.live_lock(parent)  # the old service's observer is NOT the lock holder
            run.note("old service owns only a lock-free launcher observer", observer=observer,
                     updater_pid=parent["pid"], lock_holder_pid=parent["pid"])
            run.command(["systemctl", "--user", operation, run.unit])
            _wait(lambda: not _live(original_launcher["pid"]) and not _live(observer["pid"]))
            if operation == "restart":
                replacement = _wait(lambda: (record if (record := _json(run.root / "launcher.json"))["pid"]
                                              != original_launcher["pid"] else None))
                assert replacement["invocation"] != original_launcher["invocation"]
                assert run.state(run.unit)["ActiveState"] == "active"
                run.note("service restarted with a distinct live invocation", replacement=replacement)
            else:
                assert run.state(run.unit)["ActiveState"] == "inactive"
            assert _live(parent["pid"]) and _live(child["pid"])
            assert _cgroup(parent["pid"]) == _cgroup(child["pid"]) == parent["cgroup"]
            assert run.live_lock(parent)
            assert parent["receipt"]["outcome"] == "running" and parent["receipt"]["finished_at"] is None
            assert not (run.root / "done.json").exists()
            assert run.state(run.scope)["ActiveState"] == "active"
            run.note("after service disruption updater and completion live in same independent scope; lock held",
                     operation=operation, parent_pid=parent["pid"], child_pid=child["pid"],
                     cgroup=_cgroup(parent["pid"]), lock=Path(parent["lock"]).read_text())
            run.release()
        finally:
            run.cleanup()


@pytest.mark.platforms("linux")
# The guard mistakes an inert harness's literal argv "update" for the real CLI.
# This fixture never invokes main/cmd_update/PM; all service targets are owned.
@pytest.mark.live_system_guard_bypass
def test_external_shell_safe_path_and_unsafe_service_control_discriminate_topology(tmp_path, bus_env):
    ordinary = OwnedRun(tmp_path / "ordinary-shell", tmp_path / "ordinary-home", bus_env)
    try:
        parent, child = ordinary.start(service=False)
        assert parent["invocation"] is None
        assert parent["pid"] == _json(ordinary.root / "before.json")["pid"]
        assert parent["cgroup"] == _json(ordinary.root / "before.json")["cgroup"]
        ordinary.note("ordinary shell helper is a safe no-op; parent PID/cgroup unchanged")
        ordinary.release()
    finally:
        ordinary.cleanup()
    unsafe = OwnedRun(tmp_path / "unsafe-control", tmp_path / "unsafe-home", bus_env)
    try:
        parent, child = unsafe.start("unsafe")
        assert unsafe.unit in parent["cgroup"] == child["cgroup"]
        unsafe.command(["systemctl", "--user", "stop", unsafe.unit])
        _wait(lambda: not _live(parent["pid"]) and not _live(child["pid"]))
        assert not (unsafe.root / "done.json").exists()
        assert not (unsafe.root / "result.json").exists()
        assert not (unsafe.home / "logs/update_receipts/latest.json").exists()
        from hermes_cli.update_lock import read_live_update
        assert read_live_update(path=Path(parent["lock"])) is None
        assert not Path(parent["lock"]).exists()
        unsafe.note("unsafe control killed both updater and completion despite setsid; no terminal success",
                    parent_pid=parent["pid"], child_pid=child["pid"],
                    parent_state=_process(parent["pid"]), child_state=_process(child["pid"]))
    finally:
        unsafe.cleanup()
