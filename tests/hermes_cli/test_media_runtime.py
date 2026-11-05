"""Lifecycle tests for ``MediaRuntimeController`` (media sidecar supervisor).

Every behavior the controller promises gets a real test, and the heavy ones run
against REAL processes: the fake backend below honors the proven Studio launch
contract (``--api-only --host 127.0.0.1 --port <port> --no-cloudflare --password -``, password on stdin
only, ``/api/health`` 200, ``Unsloth Studio running on http://127.0.0.1:<port>``
on stdout) and even reports its own argv/environ over ``/_debug/argv_env`` so
leak tests check what the CHILD actually received. Nothing here touches the
network beyond 127.0.0.1 ephemeral ports; the admission store is rooted in
``tmp_path``.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

import psutil
import pytest

from hermes_cli.local_runtime import media_runtime as mr
from hermes_cli.local_runtime.admission import AdmissionStore, Grant
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")

_BOOT = "test-boot"
_RESOURCE = "media:studio"

# ---------------------------------------------------------------------------
# fake backend (written to tmp_path per test; mode/receipt baked in so the
# child needs NO inherited env beyond the controller's strict disposable one)
# ---------------------------------------------------------------------------

_BACKEND = '''
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODE = "__MODE__"
RECEIPT = "__RECEIPT__"

args = sys.argv[1:]


def _flag(name):
    return args[args.index(name) + 1]


# Mirror the REAL studio/backend/run.py argparse contract: it accepts ONLY these flags (the short
# -H/-p belong to the `unsloth studio` typer CLI, not to run.py) and exits 2 on anything else.
_ACCEPTED = {"--api-only": 0, "--host": 1, "--port": 1, "--password": 1, "--no-cloudflare": 0}
_i = 0
while _i < len(args):
    if args[_i] not in _ACCEPTED:
        sys.stderr.write("run.py: error: unrecognized arguments: " + " ".join(args[_i:]) + "\\n")
        sys.exit(2)
    _i += 1 + _ACCEPTED[args[_i]]

assert "--api-only" in args, "api-only flag required"
assert "--no-cloudflare" in args, "a public tunnel must be explicitly off"
HOST = _flag("--host")
PORT = int(_flag("--port"))
assert _flag("--password") == "-", "password must be '-' (stdin)"
PASSWORD = sys.stdin.readline().strip()
del PASSWORD  # received over stdin; never echoed anywhere

if MODE == "exit":
    sys.exit(3)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            healthy = MODE != "unhealthy"
            self._reply(200 if healthy else 503,
                        {"status": "ok" if healthy else "loading"})
        elif self.path == "/_debug/argv_env":
            self._reply(200, {"argv": list(sys.argv), "env": dict(os.environ)})
        else:
            self._reply(404, {"error": "not found"})


server = ThreadingHTTPServer((HOST, PORT), Handler)
REPORTED = "0.0.0.0" if MODE == "badhost" else HOST
print(f"Unsloth Studio running on http://{REPORTED}:{PORT}", flush=True)
print(f"TAURI_PORT={PORT}", flush=True)
if RECEIPT != "None":
    payload = {"pid": os.getpid()}
    if MODE == "grandchild":
        payload["grandchild"] = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"]).pid
    Path(RECEIPT).write_text(json.dumps(payload), encoding="utf-8")
server.serve_forever()
'''


def _write_backend(tmp_path: Path, mode: str) -> tuple[Path, Path]:
    script = tmp_path / f"backend_{mode}.py"
    receipt = script.with_suffix(".receipt.json")
    script.write_text(
        _BACKEND.replace("__MODE__", mode).replace(
            "__RECEIPT__", str(receipt) if mode == "grandchild" else "None"),
        encoding="utf-8")
    return script, receipt


def _controller(tmp_path: Path, store: AdmissionStore, **over):
    kwargs = dict(owner_scope="scope-a", data_root=tmp_path / "runtime",
                  admission=store, boot_id_fn=lambda: _BOOT,
                  health_timeout=5.0, lease_seconds=30.0,
                  # these tests exercise process lifecycle only; the fake backend
                  # and decoys have no auth API (the channel has its own suite)
                  require_channel_auth=False)
    kwargs.update(over)
    return mr.MediaRuntimeController(**kwargs)


def _wait(predicate, *, timeout=15.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


def _gone(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


def _http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


class _FakeProc:
    """A live decoy pid behind a proc stub that ignores every stop signal."""

    def __init__(self, inner: subprocess.Popen):
        self.pid = inner.pid
        import io
        self.stdin = io.StringIO()
        self._inner = inner

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd="stub", timeout=timeout or 0)


@pytest.fixture(autouse=True)
def _isolate_spawn_ledger(tmp_path, monkeypatch):
    """Real ``spawn_owned`` registers in the machine-root ledger; tests keep
    that bookkeeping in tmp_path so nothing touches the real Hermes root."""
    from hermes_cli import process_identity as pi
    monkeypatch.setattr(pi, "_ledger_path", lambda: tmp_path / "spawn-ledger.jsonl")


@pytest.fixture
def live():
    """Tracks controllers/stubs so no test leaks a process or a grant."""
    controllers: list = []
    decoys: list = []

    def track(obj):
        (controllers if isinstance(obj, mr.MediaRuntimeController) else decoys).append(obj)
        return obj

    yield track
    for ctl in controllers:
        with contextlib.suppress(Exception):
            ctl.stop(timeout=10)
    for stub in decoys:
        with contextlib.suppress(Exception):
            stub.kill()
            stub.wait(timeout=5)


def _decoy_proc(live):
    inner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    return live(_FakeProc(inner)), inner


# ---------------------------------------------------------------------------
# (a) admission refusal: state unchanged, no process, reason surfaces
# ---------------------------------------------------------------------------

def test_start_refused_by_admission_leaves_state_and_no_process(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    holder = store.acquire(_RESOURCE, owner_scope="other", controller_id="other-1",
                           pid=424242, process_start_identity="holder-identity",
                           engine="unsloth-studio", model="", boot_id=_BOOT,
                           lease_seconds=3600.0)
    assert isinstance(holder, Grant)
    spawn_calls: list = []

    def no_spawn(cmd, **kwargs):
        spawn_calls.append(cmd)
        raise AssertionError("must not spawn when refused")

    ctl = live(_controller(tmp_path, store, spawn=no_spawn))
    status = ctl.start(python=sys.executable, run_py=_write_backend(tmp_path, "ok")[0])
    assert status.state == "stopped"          # unchanged, NOT failed
    assert status.instance_id is not None     # identity minted, then denied
    assert "busy" in status.detail
    assert status.pid is None
    assert spawn_calls == []
    assert store.status(_RESOURCE).controller_id == "other-1"  # incumbent untouched


# ---------------------------------------------------------------------------
# (b) fresh instance_id + fresh password every start; password never leaks
# ---------------------------------------------------------------------------

def test_fresh_identity_and_password_never_leak(tmp_path, monkeypatch, live):
    generated: list = []
    real_token = mr.secrets.token_urlsafe

    def spy(n=32):
        value = real_token(n)
        generated.append(value)
        return value

    monkeypatch.setattr(mr.secrets, "token_urlsafe", spy)
    script = _write_backend(tmp_path, "ok")[0]
    ctl = live(_controller(tmp_path, AdmissionStore(tmp_path / "adm")))
    observed: list = []  # per run: (instance_id, argv_env dump, log tail, reprs)
    try:
        for _ in range(2):
            status = ctl.start(python=sys.executable, run_py=str(script))
            assert status.state == "running", status.detail
            observed.append((status.instance_id,
                             json.dumps(_http_json(status.base_url + "/_debug/argv_env")),
                             ctl.log_tail(), repr(status), repr(ctl.__dict__)))
            assert ctl.stop().state == "stopped"
    finally:
        run_passwords = list(generated)  # snapshot: the failing start below mints one more
        exceptions: list = []
        try:
            # one failing start so exceptions are inspected too
            bad = _write_backend(tmp_path, "unhealthy")[0]
            strict = _controller(tmp_path, AdmissionStore(tmp_path / "adm2"), health_timeout=0.5)
            live(strict)
            strict.start(python=sys.executable, run_py=str(bad))
        except Exception as exc:
            exceptions.append(exc)

    assert len(run_passwords) == 2
    assert len(set(run_passwords)) == 2, "password reused across starts"
    assert observed[0][0] != observed[1][0], "instance_id reused across starts"
    haystacks = [surface for run in observed for surface in run[1:]]
    haystacks.extend(repr(exc) + str(exc) for exc in exceptions)
    for password in generated:
        assert password, "empty password"
        for haystack in haystacks:
            assert password not in haystack, "password leaked into a status/log/repr/exception surface"
        argv_env = json.loads(observed[0][1])
        child_argv = argv_env["argv"]
        assert "--password" in child_argv
        assert child_argv[child_argv.index("--password") + 1] == "-"
        assert not any(password in str(arg) for arg in child_argv), "password in child argv"
        assert not any(password in str(value) for value in argv_env["env"].values()), "password in child env"

    # the child env is the strict disposable one, not the inherited shell.
    # macOS libSystem injects __CF_USER_TEXT_ENCODING into every exec'd child;
    # that is OS plumbing, not inheritance, so it is excused explicitly.
    child_env = json.loads(observed[0][1])["env"]
    assert "PYTHONPATH" not in child_env
    assert child_env["HOME"] == str(tmp_path / "runtime" / "home")
    assert child_env["HF_HUB_OFFLINE"] == "1"
    inherited = {name for name in child_env
                 if name not in mr.disposable_environment(tmp_path / "env-probe")
                 and not name.startswith("__CF_")}
    assert all(mr._STRICT_KEEP.match(name) for name in inherited), \
        f"child inherited off-allowlist env: {sorted(inherited)}"


# ---------------------------------------------------------------------------
# (c) the spawn seam always goes through owned=True, and the argv contract holds
# ---------------------------------------------------------------------------

def test_spawn_goes_through_owned_true(tmp_path, monkeypatch, live):
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: True)
    seen: dict = {}

    def spy_spawn(cmd, *, owned, **kwargs):
        seen["owned"] = owned
        seen["cmd"] = cmd
        port = cmd[cmd.index("--port") + 1]
        kwargs["stdout"].write(f"Unsloth Studio running on http://127.0.0.1:{port}\n")
        kwargs["stdout"].flush()
        stub, _inner = _decoy_proc(live)
        return stub, None

    ctl = live(_controller(tmp_path, AdmissionStore(tmp_path / "adm"), spawn=spy_spawn))
    status = ctl.start(python=sys.executable, run_py=_write_backend(tmp_path, "ok")[0])
    assert status.state == "running", status.detail
    assert seen["owned"] is True, "sidecar must spawn owned=True"
    cmd = seen["cmd"]
    assert cmd[2:] == ["--api-only", "--host", "127.0.0.1", "--port", cmd[cmd.index("--port") + 1],
                       "--no-cloudflare", "--password", "-"]
    assert int(cmd[cmd.index("--port") + 1]) != 8888


# ---------------------------------------------------------------------------
# (d) health timeout / early exit: tree gone, exit verified, grant released
# ---------------------------------------------------------------------------

def test_health_timeout_tears_down_and_releases(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script = _write_backend(tmp_path, "unhealthy")[0]
    ctl = live(_controller(tmp_path, store, health_timeout=1.5))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed"
    assert "health" in status.detail.lower()
    assert status.exit_code is not None, "child exit code must be reported"
    assert status.pid is not None
    _wait(lambda: _gone(status.pid), what="unhealthy sidecar to die")
    _wait(lambda: store.status(_RESOURCE).residency == "released",
          what="grant released after verified exit")


def test_early_child_exit_reports_code_and_releases(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script = _write_backend(tmp_path, "exit")[0]
    ctl = live(_controller(tmp_path, store, health_timeout=5.0))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed"
    assert status.exit_code == 3
    assert "rc=3" in status.detail
    assert _gone(status.pid)
    assert store.status(_RESOURCE).residency == "released"


# ---------------------------------------------------------------------------
# (e) + (i) tick(): renewal keeps the epoch; a SIGKILLed child -> failed +
# released grant, with the REAL process verified gone
# ---------------------------------------------------------------------------

def test_tick_renews_and_detects_sigkill_crash(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script = _write_backend(tmp_path, "ok")[0]
    ctl = live(_controller(tmp_path, store))
    running = ctl.start(python=sys.executable, run_py=str(script))
    assert running.state == "running", running.detail
    epoch_before = running.epoch

    ticked = ctl.tick()
    assert ticked.state == "running"
    assert ticked.epoch == epoch_before, "renew must not change the epoch"
    assert store.status(_RESOURCE).residency == "resident"
    deadline_before = store.status(_RESOURCE).deadline

    def renewed():
        ctl.tick()
        return store.status(_RESOURCE).deadline > deadline_before

    wait_until(renewed, what="a tick to renew the lease")

    os.kill(running.pid, signal.SIGKILL)
    status = status_after_ticks = None
    for _ in range(100):
        status = ctl.tick()
        if status.state == "failed":
            break
        time.sleep(0.05)
    assert status is not None and status.state == "failed", status_after_ticks
    assert status.exit_code == -9
    _wait(lambda: _gone(running.pid), what="SIGKILLed pid to be reaped")
    _wait(lambda: store.status(_RESOURCE).residency == "released",
          what="crash-recovery release")
    again = ctl.tick()
    assert again.state == "failed", "tick must never auto-restart"


# ---------------------------------------------------------------------------
# (f) + (i) stop(): children first, verified exit, THEN release; the whole
# real tree (child + grandchild) must be gone
# ---------------------------------------------------------------------------

def test_stop_kills_whole_tree_then_releases(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script, receipt = _write_backend(tmp_path, "grandchild")
    ctl = live(_controller(tmp_path, store))
    running = ctl.start(python=sys.executable, run_py=str(script))
    assert running.state == "running", running.detail
    _wait(lambda: receipt.exists(), what="backend receipt")
    ids = json.loads(receipt.read_text(encoding="utf-8"))
    grandchild = ids["grandchild"]
    assert ids["pid"] == running.pid
    assert not _gone(grandchild)

    status = ctl.stop(timeout=10)
    assert status.state == "stopped", status.detail
    assert _gone(running.pid), "sidecar process survived stop"
    assert _gone(grandchild), "grandchild survived stop (tree not killed)"
    assert store.status(_RESOURCE).residency == "released"
    assert ctl.status().state == "stopped"


class _StubbornSpawn:
    """Spawns a real live decoy but never lets it die (exit unverifiable)."""

    def __init__(self, live):
        self.live = live

    def __call__(self, cmd, *, owned, **kwargs):
        port = cmd[cmd.index("--port") + 1]
        kwargs["stdout"].write(f"TAURI_PORT={port}\n")
        kwargs["stdout"].flush()
        stub, _inner = _decoy_proc(self.live)
        return stub, None


def test_stop_without_verified_exit_keeps_grant(tmp_path, monkeypatch, live):
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: True)
    store = AdmissionStore(tmp_path / "adm")
    ctl = live(_controller(tmp_path, store, spawn=_StubbornSpawn(live)))
    running = ctl.start(python=sys.executable, run_py=_write_backend(tmp_path, "ok")[0])
    assert running.state == "running", running.detail

    status = ctl.stop(timeout=0.5)
    assert status.state in ("stopping", "failed")
    assert "verif" in status.detail.lower()
    record = store.status(_RESOURCE)
    assert record.residency != "released", "grant must stay held while exit is unverified"
    assert record.residency in ("loading", "resident", "draining")


class _CountingStubbornSpawn(_StubbornSpawn):
    def __init__(self, live):
        super().__init__(live)
        self.spawns = 0

    def __call__(self, cmd, *, owned, **kwargs):
        self.spawns += 1
        return super().__call__(cmd, owned=owned, **kwargs)


def test_a_second_start_never_drops_a_sidecar_that_a_failed_start_could_not_stop(tmp_path, monkeypatch, live):
    """A start that fails while its child cannot be proven gone lands in 'failed' WITH the live process handle. A second start() used to
    blank that handle: the sidecar kept running (GPU and RAM held, grant still live), nothing could ever stop it, and stop() then
    reported 'stopped (no live sidecar)'."""
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: False)         # the start fails; the stubborn child cannot be stopped
    store = AdmissionStore(tmp_path / "adm")
    spawn = _CountingStubbornSpawn(live)
    ctl = live(_controller(tmp_path, store, spawn=spawn, health_timeout=0.6))
    script = _write_backend(tmp_path, "ok")[0]

    first = ctl.start(python=sys.executable, run_py=script)
    assert first.state == "failed"
    held = ctl._proc
    assert held is not None and held.poll() is None, "precondition: the failed start left a live child behind"
    record_before = store.status(_RESOURCE)

    second = ctl.start(python=sys.executable, run_py=script)
    assert second.state == "failed"
    assert "stop" in second.detail.lower() and "running" in second.detail.lower(), second.detail
    assert ctl._proc is held, "the only handle to the live sidecar was dropped"
    assert spawn.spawns == 1, "a second sidecar must not be launched next to the first"
    assert store.status(_RESOURCE).epoch == record_before.epoch, "the live grant must not be touched"

    third = ctl.start(python=sys.executable, run_py=script)
    assert third.detail == second.detail, "repeated attempts must not pile text onto the detail"

    stopped = ctl.stop(timeout=0.5)
    assert "no live sidecar" not in stopped.detail, "stop() must not claim there is nothing to stop while the child still runs"
    assert ctl._proc is held


def test_a_failed_start_whose_child_is_gone_can_simply_be_started_again(tmp_path, monkeypatch, live):
    """Control: the guard is only for a LIVE child. A failed start that was torn down and verified must not wedge the controller."""
    store = AdmissionStore(tmp_path / "adm")
    ctl = live(_controller(tmp_path, store, health_timeout=1.0))
    script = _write_backend(tmp_path, "ok")[0]
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: False)
    first = ctl.start(python=sys.executable, run_py=script)
    assert first.state == "failed"
    assert ctl._proc is None or ctl._proc.poll() is not None, "precondition: that child really is gone"
    monkeypatch.undo()
    second = ctl.start(python=sys.executable, run_py=script)
    assert second.state == "running", second.detail


def test_unreadable_identity_after_spawn_fails_closed(tmp_path, monkeypatch, live):
    """A LIVE process whose identity can no longer be read (AccessDenied-style)
    is ambiguous, not gone: the grant must stay held."""
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: True)
    seen: set[int] = set()

    def identity(pid: int) -> str:
        if pid in seen:                 # first read works (spawn), later reads fail
            raise PermissionError("identity unreadable")
        seen.add(pid)
        return f"ident-{pid}"

    store = AdmissionStore(tmp_path / "adm")
    ctl = live(_controller(tmp_path, store, spawn=_StubbornSpawn(live), identity_fn=identity))
    assert ctl.start(python=sys.executable,
                     run_py=_write_backend(tmp_path, "ok")[0]).state == "running"

    status = ctl.stop(timeout=0.5)
    assert status.state in ("stopping", "failed")
    assert "verif" in status.detail.lower()
    assert store.status(_RESOURCE).residency != "released", (
        "an ambiguous (live, unreadable-identity) process must never release the grant")


def test_identity_unreadable_from_the_start_fails_closed(tmp_path, monkeypatch, live):
    """No recorded identity to compare against and the pid is still live: we
    cannot tell our child from a recycled pid, so exit is NOT verified."""
    monkeypatch.setattr(mr, "_probe_health", lambda url, timeout=1.0: True)

    def identity(pid: int) -> str:
        raise PermissionError("identity unreadable")

    store = AdmissionStore(tmp_path / "adm")
    ctl = live(_controller(tmp_path, store, spawn=_StubbornSpawn(live), identity_fn=identity))
    assert ctl.start(python=sys.executable,
                     run_py=_write_backend(tmp_path, "ok")[0]).state == "running"

    status = ctl.stop(timeout=0.5)
    assert status.state in ("stopping", "failed")
    assert store.status(_RESOURCE).residency != "released"


def test_a_reaped_child_is_verified_gone_even_if_identity_is_unreadable(tmp_path, live):
    """Positive control for the fail-closed rule: once WE have reaped the Popen
    handle the kernel has released that pid, so exit IS provable without any
    identity comparison (otherwise unreadable identity would wedge every stop)."""
    def identity(pid: int) -> str:
        raise PermissionError("identity unreadable")

    store = AdmissionStore(tmp_path / "adm")
    ctl = live(_controller(tmp_path, store, identity_fn=identity))
    running = ctl.start(python=sys.executable, run_py=_write_backend(tmp_path, "ok")[0])
    assert running.state == "running", running.detail

    status = ctl.stop(timeout=10)
    assert status.state == "stopped", status.detail
    assert store.status(_RESOURCE).residency == "released"


# ---------------------------------------------------------------------------
# (g) fenced renew: stop OUR sidecar, go failed, never touch the new holder
# ---------------------------------------------------------------------------

def test_fenced_renew_stops_own_sidecar_only(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script = _write_backend(tmp_path, "ok")[0]
    ctl = live(_controller(tmp_path, store))
    running = ctl.start(python=sys.executable, run_py=str(script))
    assert running.state == "running", running.detail
    old_epoch = running.epoch

    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE admission_resources SET epoch = epoch + 1 WHERE resource = ?",
                 (_RESOURCE,))
    conn.commit()
    conn.close()

    status = ctl.tick()
    assert status.state == "failed"
    assert "fenc" in status.detail.lower()
    _wait(lambda: _gone(running.pid), what="own sidecar stopped after fencing")
    record = store.status(_RESOURCE)
    assert record.epoch == old_epoch + 1, "controller must not rewrite the new holder's epoch"
    assert record.residency in ("loading", "resident", "draining"), \
        "controller must not release a grant it no longer holds"


# ---------------------------------------------------------------------------
# (h) loopback only: a non-127.0.0.1 reported bind address is a failure
# ---------------------------------------------------------------------------

def test_non_loopback_reported_host_fails(tmp_path, live):
    store = AdmissionStore(tmp_path / "adm")
    script = _write_backend(tmp_path, "badhost")[0]
    ctl = live(_controller(tmp_path, store, health_timeout=3.0))
    status = ctl.start(python=sys.executable, run_py=str(script))
    assert status.state == "failed"
    assert "loopback" in status.detail.lower()
    assert "0.0.0.0" in status.detail
    assert _gone(status.pid)
    _wait(lambda: store.status(_RESOURCE).residency == "released", what="grant released")
