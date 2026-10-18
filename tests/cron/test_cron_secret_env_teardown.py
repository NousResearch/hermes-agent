"""Job-scoped ``secret_env`` (CORP-764 formal review): the secret values and the script lock are
let go only once every process of the run is confirmed gone.

When that cannot be confirmed (a member still exiting at the bound, a failed wait, a teardown
that does not kill), the run is reported failed and the resolver context stays entered: the
script stays locked and the values held until a retry confirms the tree gone. A start that fails
after the script was resumed ends its tree the same way.

Real ``_run_job_script`` with the real resolver context and real processes against a temp
``HERMES_HOME``; the injected faults sit at the job's WinAPI waits or its ``terminate``. Windows
only: secret_env fails closed elsewhere.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import threading
import time
import types

import psutil
import pytest

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
NAME = "NOTICE_REPORT_SLACK_TOKEN"

_IDENTITY = """\
import json, pathlib, psutil
def record(path):
    me = psutil.Process()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": me.pid, "created": me.create_time()}))
    tmp.replace(path)
"""

_CHILD = _IDENTITY + """\
import sys, time
record(pathlib.Path(sys.argv[1]))
time.sleep(60)
"""

_PARENT = _IDENTITY + f"""\
import subprocess, sys, time
here = pathlib.Path(sys.argv[0]).parent
subprocess.Popen(
    [sys.executable, "-c", {_CHILD!r}, str(here / "child.ident")], stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
deadline = time.monotonic() + 30
while not (here / "child.ident").exists() and time.monotonic() < deadline:
    time.sleep(0.02)
record(here / "root.ident")
"""


class FakeBackend:
    def resolve(self, target):
        return FAKE_VALUE


def _alive(identity) -> bool:
    try:
        proc = psutil.Process(identity["pid"])
        if proc.create_time() != identity["created"]:
            return False
        proc.wait(timeout=0)
        return False
    except psutil.TimeoutExpired:
        return True
    except psutil.NoSuchProcess:
        return False


def _identities(scripts) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(scripts.glob("*.ident"))]


def _locked(script) -> bool:
    try:
        with open(script, "r+b"):  # windows-footgun: ok (binary)
            return False
    except PermissionError:
        return True


class _InjectedBaseException(BaseException):
    """Not an Exception: what an `except Exception` lets through (SystemExit, KeyboardInterrupt)."""


@pytest.fixture
def uncaught_thread_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    return errors


def _wait(predicate, timeout=20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


@pytest.fixture
def secret_run(tmp_path, monkeypatch):
    """A pinned spawner script and a resolver that records, when its context is left, whether
    each process of the run was still alive (and whether the script was still locked)."""
    from cron import scheduler_script
    from cron.secret_env import resolve_for_script

    home = tmp_path / ".hermes"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(scheduler_script, "_SECRET_TREE_EXIT_TIMEOUT", 5.0)
    monkeypatch.setattr(scheduler_script, "_SECRET_TREE_RETRY_INTERVAL", 0.05)
    script = scripts / "spawner.py"
    script.write_text(_PARENT, encoding="utf-8")
    job = {"id": "teardown", "secret_env": {NAME: "wincred://hermes/teardown"},
           "secret_env_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    exits: list[dict] = []

    @contextlib.contextmanager
    def resolver(path):
        with resolve_for_script(job, path, backends={"wincred": FakeBackend()}) as values:
            try:
                yield values
            finally:
                exits.append({"alive": [_alive(i) for i in _identities(scripts)],
                              "processes": len(_identities(scripts)), "locked": _locked(script)})

    try:
        yield script, resolver, exits
    finally:
        for identity in _identities(scripts):
            if _alive(identity):
                psutil.Process(identity["pid"]).kill()
        assert _wait(lambda: not scheduler_script._held_secret_runs), "a held run was never released"


class _FaultyApi:
    """The job's kernel32 with some calls answering `faults[name]` until `allow` is set; a
    callable fault is called with the real function and the arguments."""

    def __init__(self, api, faults, allow):
        self._api, self._faults, self._allow = api, faults, allow

    def __getattr__(self, name):
        real = getattr(self._api, name)
        if name not in self._faults:
            return real
        fault = self._faults[name]

        def call(*args):
            if self._allow.is_set():
                return real(*args)
            return fault(real, *args) if callable(fault) else fault
        return call


def _denied(real, *args):
    ctypes.set_last_error(5)  # ERROR_ACCESS_DENIED
    return 0


_API_FAULTS = {
    "still_exiting": {"WaitForSingleObject": 0x102},  # WAIT_TIMEOUT
    "wait_failed": {"WaitForSingleObject": 0xFFFFFFFF},  # WAIT_FAILED
    # A listed member the owner may not open, which the kill did not reach either.
    "unopenable": {"OpenProcess": _denied, "TerminateJobObject": 1},
}


def _faulty_jobs(monkeypatch, faults, allow):
    from hermes_cli.local_runtime import processes

    real_new_job = processes.new_job

    def new_job():
        job = real_new_job()
        job._api = _FaultyApi(job._api, faults, allow)
        return job

    monkeypatch.setattr(processes, "new_job", new_job)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("fault", [*_API_FAULTS, "not_killed"])
def test_unconfirmed_tree_keeps_values_and_lock_until_confirmed(secret_run, monkeypatch, fault):
    from cron import scheduler_script
    from hermes_cli.local_runtime import processes

    script, resolver, exits = secret_run
    allow = threading.Event()
    if fault == "not_killed":
        real_terminate = processes._WindowsJob.terminate

        def terminate(job, timeout):
            if not allow.is_set():
                raise processes.JobShutdownError("injected: teardown did not run")
            real_terminate(job, timeout)

        monkeypatch.setattr(processes._WindowsJob, "terminate", terminate)
    else:
        _faulty_jobs(monkeypatch, _API_FAULTS[fault], allow)

    success, output = scheduler_script._run_job_script(
        str(script), secret_env_resolver=resolver)

    assert (success, output) == (False, scheduler_script._SECRET_TEARDOWN_FAILED)
    assert FAKE_VALUE not in output
    # Not confirmed gone: the resolver context was never left and the script is still locked.
    assert exits == []
    assert _locked(script)
    assert len(scheduler_script._held_secret_runs) == 1
    if fault in ("not_killed", "unopenable"):
        assert any(_alive(i) for i in _identities(script.parent))

    allow.set()
    assert _wait(lambda: exits), "the held run was never released"
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]
    assert _wait(lambda: not _locked(script))


@pytest.mark.platforms("windows")
def test_start_failure_after_resume_ends_the_tree_before_release(secret_run, monkeypatch):
    from cron import scheduler_script

    script, resolver, exits = secret_run
    real_resume = psutil.Process.resume

    def resume_then_fail(proc):
        real_resume(proc)
        # The script ran and its detached child holds the values before the start fails.
        assert _wait(lambda: len(_identities(script.parent)) == 2)
        raise psutil.AccessDenied(proc.pid)

    monkeypatch.setattr(psutil.Process, "resume", resume_then_fail)

    success, output = scheduler_script._run_job_script(
        str(script), secret_env_resolver=resolver)

    assert success is False and output.startswith("Script execution failed")
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]
    assert not _locked(script)
    assert not scheduler_script._held_secret_runs


@pytest.mark.platforms("windows")
def test_ordinary_run_confirms_the_tree_gone_before_release(secret_run):
    from cron import scheduler_script

    script, resolver, exits = secret_run

    success, _output = scheduler_script._run_job_script(str(script), secret_env_resolver=resolver)

    assert success is True
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]
    assert not _locked(script)


def _listed_then_dropped(state):
    """A BasicProcessIdList query that, once it has listed a member, lists nobody."""
    def query(real, handle, info_class, info, size, returned):
        ok = real(handle, info_class, info, size, returned)
        listing = info._obj
        if ok and state["seen"]:
            listing.NumberOfAssignedProcesses = listing.NumberOfProcessIdsInList = 0
            state["dropped"] += 1
        elif ok and listing.NumberOfProcessIdsInList:
            state["seen"] = True
        return ok
    return query


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("unheld", ["OpenProcess", "IsProcessInJob"])
def test_member_never_held_is_not_taken_for_gone_when_the_listing_drops_it(
        secret_run, monkeypatch, unheld):
    from cron import scheduler_script

    script, resolver, exits = secret_run
    allow = threading.Event()
    state = {"seen": False, "dropped": 0}
    _faulty_jobs(monkeypatch, {
        unheld: _denied,
        # The kill reports success, yet the member lives on: its process never signals.
        "TerminateJobObject": 1,
        "QueryInformationJobObject": _listed_then_dropped(state),
    }, allow)

    success, output = scheduler_script._run_job_script(
        str(script), secret_env_resolver=resolver)

    assert (success, output) == (False, scheduler_script._SECRET_TEARDOWN_FAILED)
    # The member left the listing while it was still running, and the run stays held.
    assert state["dropped"] > 0
    assert any(_alive(i) for i in _identities(script.parent))
    assert exits == []
    assert _locked(script)
    assert len(scheduler_script._held_secret_runs) == 1

    allow.set()
    assert _wait(lambda: exits), "the held run was never released"
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("fault", ["unexpected_error", "base_exception", "retry_not_started"])
def test_held_run_keeps_its_retry_through_faults_and_is_released_once(
        secret_run, monkeypatch, uncaught_thread_errors, fault):
    from cron import scheduler, scheduler_script
    from hermes_cli.local_runtime import processes

    script, resolver, exits = secret_run
    allow, allow_start = threading.Event(), threading.Event()
    attempts = []
    real_terminate = processes._WindowsJob.terminate

    def terminate(job, timeout):
        if allow.is_set():
            return real_terminate(job, timeout)
        attempts.append(fault)
        if fault == "unexpected_error" and len(attempts) > 1:
            raise RuntimeError("injected: unexpected teardown error")
        if fault == "base_exception" and len(attempts) > 1:
            raise _InjectedBaseException("injected: teardown raised a BaseException")
        raise processes.JobShutdownError("injected: teardown did not run")

    monkeypatch.setattr(processes._WindowsJob, "terminate", terminate)
    if fault == "retry_not_started":
        class Thread(threading.Thread):
            def start(self):
                if not allow_start.is_set():
                    raise RuntimeError("can't start new thread")
                super().start()

        monkeypatch.setattr(scheduler_script, "threading",
                            types.SimpleNamespace(**{**vars(threading), "Thread": Thread}))

    success, output = scheduler_script._run_job_script(
        str(script), secret_env_resolver=resolver)

    assert (success, output) == (False, scheduler_script._SECRET_TEARDOWN_FAILED)
    if fault == "retry_not_started":
        time.sleep(0.5)
        scheduler.tick(verbose=False)  # the retry still cannot start: nothing is released
        assert attempts == [fault] and scheduler_script._held_secret_worker is None
        allow_start.set()
        scheduler.tick(verbose=False)
    assert _wait(lambda: len(attempts) >= 4), "the held run is no longer retried"
    assert scheduler_script._held_secret_worker.is_alive()
    worker = scheduler_script._held_secret_worker
    assert exits == []
    assert _locked(script)
    assert len(scheduler_script._held_secret_runs) == 1

    allow.set()
    assert _wait(lambda: not scheduler_script._held_secret_runs)
    assert _wait(lambda: scheduler_script._held_secret_worker is None)
    scheduler.tick(verbose=False)
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]
    assert not _locked(script)
    # The worker that met the faults lived through them and did the release itself.
    assert _wait(lambda: not worker.is_alive())
    assert uncaught_thread_errors == []


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("error", [RuntimeError, _InjectedBaseException])
def test_resolver_exit_failure_after_confirmation_drops_the_run_once(
        secret_run, monkeypatch, caplog, uncaught_thread_errors, error):
    from cron import scheduler_script
    from hermes_cli.local_runtime import processes

    script, resolver, exits = secret_run
    allow = threading.Event()
    real_terminate = processes._WindowsJob.terminate

    def terminate(job, timeout):
        if not allow.is_set():
            raise processes.JobShutdownError("injected: teardown did not run")
        real_terminate(job, timeout)

    monkeypatch.setattr(processes._WindowsJob, "terminate", terminate)

    closes = []

    @contextlib.contextmanager
    def failing_exit(path):
        with resolver(path) as values:
            yield values
        closes.append(path)
        raise error("injected: resolver exit failed")

    success, _output = scheduler_script._run_job_script(
        str(script), secret_env_resolver=failing_exit)
    assert success is False and exits == []

    allow.set()
    assert _wait(lambda: not scheduler_script._held_secret_runs)
    assert _wait(lambda: scheduler_script._held_secret_worker is None)
    assert exits == [{"alive": [False, False], "processes": 2, "locked": True}]
    assert len(closes) == 1
    assert not _locked(script)
    assert "its cleanup ran, but leaving its resolver context failed" in caplog.text
    assert uncaught_thread_errors == []
