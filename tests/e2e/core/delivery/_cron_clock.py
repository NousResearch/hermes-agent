"""Lane-private harness for the C13 virtual-clock cron soak (test_cron_virtual_clock_soak.py).

What is virtual and what stays real:

* ``VirtualClock`` is ONE epoch value in a file shared by every process of a scenario. It is
  installed at the single wall-clock seam — ``hermes_time.datetime`` (every ``_hermes_now``
  binding in cron/jobs.py, scheduler.py, executions.py, occurrences via jobs, ... resolves
  ``hermes_time.now()`` -> ``datetime.now(tz)`` at call time) — plus the ``time.time()`` reads of
  cron.jobs / cron.scheduler / cron.executions (in-flight ages, ledger handoff grace, ticker
  heartbeat marker). ``time.monotonic`` stays real on purpose: it only paces REAL waits (lock
  deadlines, heartbeat thread cadence, reap/GC throttles).
* The fire-claim heartbeat cadence is time-scaled (60 s -> ``HEARTBEAT_REAL_SECONDS``) so a
  run that spans many virtual minutes still refreshes its lease the way a real one would.
* Fakes: the agent (``cron.scheduler.run_job``, the one function that would build an AIAgent)
  and the platform wire (``tools.send_message_tool._send_to_platform``). Both append JSONL
  records. Everything between — the real ``InProcessCronScheduler.start`` ticker loop,
  ``tick()``, due scan, pending slots, fire claims, heartbeat, executions ledger, delivery
  routing, ``mark_job_run`` — is production code.

``SchedulerHost`` runs the real ticker loop in a thread with a stepping ``stop_event``; running
this file as a script runs the same loop in a separate OS process (``ChildHost``), stepped via
files, so two processes can share one ``HERMES_HOME`` and one can be SIGKILLed mid-run. Two
tickers never contend for a FIRE: the tick lock admits one per instant and the winner advances
``next_run_at`` before it releases. Fire-claim contention is the external-provider path
(``CronScheduler.fire_due``, "exactly one of N replicas runs a job"): ``ReplicaHost`` runs a
replica in a second process, and ``install_claim_barrier`` parks each replica's
``claim_job_for_fire`` until every replica of the round is inside it, so the store CAS, not
timing, decides every fire.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time as _real_time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

REAL_DATETIME = _dt.datetime
HEARTBEAT_REAL_SECONDS = 0.05
POLL_SECONDS = 0.005
DEADLINE_SECONDS = 120.0
HOLD_DEADLINE_SECONDS = 300.0

_CURRENT_EXECUTION: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "c13_current_execution", default=None)


def wait_until(predicate: Callable[[], Any], what: str, timeout: float = DEADLINE_SECONDS):
    """Poll ``predicate`` until truthy (returned) or raise after a generous real deadline."""
    deadline = _real_time.monotonic() + timeout
    delay = POLL_SECONDS / 8
    while True:
        value = predicate()
        if value:
            return value
        if _real_time.monotonic() >= deadline:
            raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")
        _real_time.sleep(delay)
        delay = min(delay * 2, POLL_SECONDS)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict) -> None:
    line = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


class Control:
    """Filesystem layout of one scenario (shared by the parent and child hosts)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.clock_file = self.root / "clock"
        self.runs = self.root / "runs.jsonl"
        self.sink = self.root / "sink.jsonl"
        self.behaviors = self.root / "behaviors.json"
        self.gates = self.root / "gates"
        self.child = self.root / "child"
        self.replica = self.root / "replica"
        self.claims = self.root / "claims.jsonl"

    def ensure(self) -> "Control":
        for d in (self.root, self.gates, self.child, self.replica):
            d.mkdir(parents=True, exist_ok=True)
        if not self.behaviors.exists():
            self.set_behaviors({})
        return self

    def set_behaviors(self, behaviors: Dict[str, dict]) -> None:
        _atomic_write(self.behaviors, json.dumps(behaviors))

    def behavior(self, name: str) -> dict:
        try:
            return json.loads(self.behaviors.read_text(encoding="utf-8")).get(name) or {}
        except (OSError, ValueError):
            return {}

    def hold_file(self, name: str) -> Path:
        return self.gates / f"{name}.hold"


class VirtualClock:
    """File-backed epoch seconds; every process of a scenario reads the same instant."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def now_ts(self) -> float:
        return float(self.path.read_text(encoding="utf-8"))

    def set(self, ts: float) -> None:
        _atomic_write(self.path, repr(float(ts)))

    def advance(self, seconds: float) -> float:
        ts = self.now_ts() + seconds
        self.set(ts)
        return ts


def _virtual_datetime_class(clock: VirtualClock):
    class VirtualDatetime(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):  # noqa: D401 - datetime API
            return REAL_DATETIME.fromtimestamp(clock.now_ts(), tz)

    return VirtualDatetime


class _TimeProxy:
    """Stand-in for the ``time`` module inside cron modules: virtual ``time()``, real rest."""

    def __init__(self, clock: VirtualClock):
        self._clock = clock

    def time(self) -> float:
        return self._clock.now_ts()

    def __getattr__(self, name):
        return getattr(_real_time, name)


class FakeAgent:
    """Replaces ``cron.scheduler.run_job``; returns the real 4-tuple contract."""

    def __init__(self, clock: VirtualClock, control: Control):
        self.clock = clock
        self.control = control

    def __call__(self, job, *, execution_id=None, **_kwargs):
        name = job.get("name") or job["id"]
        _CURRENT_EXECUTION.set(execution_id)
        record = {
            "event": "start", "exec": execution_id, "job_id": job["id"], "name": name,
            "instant": job.get("_scheduled_instant"), "fired_at": self.clock.now_ts(),
            "pid": os.getpid(),
        }
        _append_jsonl(self.control.runs, record)
        behavior = self.control.behavior(name)
        hold = self.control.hold_file(name)
        if hold.exists():
            _atomic_write(self.control.gates / f"{name}.entered", str(execution_id))
            deadline = _real_time.monotonic() + HOLD_DEADLINE_SECONDS
            while hold.exists() and _real_time.monotonic() < deadline:
                _real_time.sleep(POLL_SECONDS)
        _append_jsonl(self.control.runs, dict(record, event="end", ended_at=self.clock.now_ts()))
        output = f"# {name}\n\nexec={execution_id}\n"
        if behavior.get("agent") == "fail":
            return False, output, "", f"simulated provider outage exec={execution_id}", None
        return True, output, f"report for {name} exec={execution_id}", None, None


class FakeSink:
    """Replaces ``tools.send_message_tool._send_to_platform`` (the standalone wire)."""

    def __init__(self, clock: VirtualClock, control: Control):
        self.clock = clock
        self.control = control

    def _behavior_for(self, message: str) -> dict:
        try:
            behaviors = json.loads(self.control.behaviors.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        for name, behavior in behaviors.items():
            if name in message:
                return behavior or {}
        return {}

    async def __call__(self, platform, pconfig, chat_id, message, thread_id=None,
                       media_files=None, **_kwargs):
        import asyncio

        execution_id = _CURRENT_EXECUTION.get()
        behavior = self._behavior_for(message)
        if behavior.get("delivery") == "slow":
            await asyncio.sleep(0.25)
        ok = behavior.get("delivery") != "fail"
        _append_jsonl(self.control.sink, {
            "exec": execution_id, "ok": ok, "chat_id": str(chat_id),
            "platform": getattr(platform, "value", str(platform)),
            "exec_in_text": bool(execution_id) and f"exec={execution_id}" in message,
            "at": self.clock.now_ts(), "pid": os.getpid(),
        })
        if not ok:
            return {"error": "simulated 502 from platform"}
        return {"success": True, "message_id": f"m-{execution_id}"}


def install(clock: VirtualClock, control: Control, setattr_fn=setattr) -> None:
    """Install the virtual clock + fakes. ``setattr_fn`` is ``monkeypatch.setattr`` in pytest."""
    import cron.executions as executions
    import cron.jobs as jobs
    import cron.scheduler as scheduler
    import hermes_time
    import tools.send_message_tool as send_message_tool

    setattr_fn(hermes_time, "datetime", _virtual_datetime_class(clock))
    proxy = _TimeProxy(clock)
    for module in (jobs, scheduler, executions):
        setattr_fn(module, "time", proxy)
    setattr_fn(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", HEARTBEAT_REAL_SECONDS)
    setattr_fn(scheduler, "run_job", FakeAgent(clock, control))
    setattr_fn(send_message_tool, "_send_to_platform", FakeSink(clock, control))
    # Durability is not under test (SIGKILL keeps the page cache); per-write fsync of
    # jobs.json/markers dominates wall time at virtual cadence. Atomic renames stay real.
    setattr_fn(os, "fsync", lambda _fd: None)
    import hermes_cli.sqlite_util as sqlite_util

    real_open_db = sqlite_util.open_db

    def open_db_no_full_sync(*args, **kwargs):  # same connection, WAL, schema; no per-commit fsync
        kwargs["synchronous_full"] = False
        conn = real_open_db(*args, **kwargs)
        conn.execute("PRAGMA synchronous=OFF")  # also skips the checkpoint-on-close fsync
        return conn

    setattr_fn(sqlite_util, "open_db", open_db_no_full_sync)
    # Worktree GC prunes the REAL git checkout the test runs from; hygiene, not scheduling.
    setattr_fn(scheduler, "_maybe_run_worktree_maintenance", lambda: None)
    hermes_time.reset_cache()


# --- stepping ticker hosts -------------------------------------------------------------------

class _StepGate:
    """``stop_event`` for ``InProcessCronScheduler.start``: each ``wait()`` parks the real ticker
    loop until the driver releases the next iteration."""

    def __init__(self):
        self.cv = threading.Condition()
        self.idle = 0
        self.go = 0
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def wait(self, timeout=None) -> bool:
        with self.cv:
            self.idle += 1
            self.cv.notify_all()
            deadline = _real_time.monotonic() + HOLD_DEADLINE_SECONDS
            while not self.stopped and self.go < self.idle:
                remaining = deadline - _real_time.monotonic()
                if remaining <= 0:
                    self.stopped = True
                    break
                self.cv.wait(min(remaining, 1.0))
            return self.stopped


class SchedulerHost:
    """The real in-process ticker loop, one iteration per ``release()``."""

    label = "inproc"

    def __init__(self):
        self.gate: Optional[_StepGate] = None
        self.thread: Optional[threading.Thread] = None
        self.errors: List[BaseException] = []

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        """Boot the loop (startup recovery + the boot tick run immediately)."""
        from cron.scheduler_provider import InProcessCronScheduler

        gate = self.gate = _StepGate()

        def _run():
            try:
                InProcessCronScheduler().start(gate, interval=60)
            except BaseException as exc:  # pragma: no cover - surfaced by wait_idle
                self.errors.append(exc)

        self.thread = threading.Thread(target=_run, name="c13-ticker", daemon=True)
        self.thread.start()
        self.wait_idle()

    def release(self) -> None:
        with self.gate.cv:
            self.gate.go += 1
            self.gate.cv.notify_all()

    def wait_idle(self) -> None:
        gate = self.gate

        def _idle():
            if self.errors:
                raise AssertionError(f"ticker thread died: {self.errors[0]!r}")
            with gate.cv:
                return gate.idle >= gate.go + 1

        wait_until(_idle, "in-process ticker iteration")

    def tick(self) -> None:
        self.release()
        self.wait_idle()

    def stop(self) -> None:
        if self.gate is not None:
            self.gate.set()
        if self.thread is not None:
            self.thread.join(timeout=DEADLINE_SECONDS)
            assert not self.thread.is_alive(), "ticker thread did not stop"
        self.thread = None


class _FileGate:
    """Child-process ``stop_event`` stepped through ``<control>/child/{go,idle,stop}``."""

    def __init__(self, control: Control):
        self.dir = control.child
        self.idle = 0

    def is_set(self) -> bool:
        return (self.dir / "stop").exists()

    def set(self) -> None:
        (self.dir / "stop").write_text("1", encoding="utf-8")

    def _go(self) -> int:
        try:
            return int((self.dir / "go").read_text(encoding="utf-8") or 0)
        except (OSError, ValueError):
            return 0

    def wait(self, timeout=None) -> bool:
        self.idle += 1
        _atomic_write(self.dir / "idle", str(self.idle))
        deadline = _real_time.monotonic() + HOLD_DEADLINE_SECONDS
        while not self.is_set() and self._go() < self.idle:
            if _real_time.monotonic() >= deadline:
                return True
            _real_time.sleep(POLL_SECONDS)
        return self.is_set()


class ChildHost:
    """Same real ticker loop in a separate OS process sharing the scenario's HERMES_HOME."""

    label = "child"

    def __init__(self, control: Control, env: Dict[str, str], repo_root: Path):
        self.control = control
        self.env = env
        self.repo_root = repo_root
        self.proc: Optional[subprocess.Popen] = None
        self.go = 0

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _idle(self) -> int:
        try:
            return int((self.control.child / "idle").read_text(encoding="utf-8") or 0)
        except (OSError, ValueError):
            return 0

    def start(self) -> None:
        for name in ("go", "idle", "stop"):
            (self.control.child / name).unlink(missing_ok=True)
        self.go = 0
        log = open(self.control.child / "stderr.log", "a", encoding="utf-8")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(self.control.root)],
            cwd=str(self.repo_root), env=self.env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log)
        log.close()
        self.wait_idle(timeout=180.0)

    def release(self) -> None:
        self.go += 1
        _atomic_write(self.control.child / "go", str(self.go))

    def wait_idle(self, timeout: float = DEADLINE_SECONDS) -> None:
        def _idle():
            if self.proc.poll() is not None:
                tail = (self.control.child / "stderr.log").read_text(encoding="utf-8")[-4000:]
                raise AssertionError(f"child ticker exited rc={self.proc.returncode}:\n{tail}")
            return self._idle() >= self.go + 1

        wait_until(_idle, "child ticker iteration", timeout=timeout)

    def tick(self) -> None:
        self.release()
        self.wait_idle()

    def kill(self) -> None:
        """SIGKILL: the process dies with whatever it had in flight (a crash, not a drain)."""
        if self.proc is not None and self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)  # windows-footgun: ok (POSIX-only harness)
            self.proc.wait(timeout=DEADLINE_SECONDS)

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            (self.control.child / "stop").write_text("1", encoding="utf-8")
            try:
                self.proc.wait(timeout=DEADLINE_SECONDS)
            except subprocess.TimeoutExpired:
                self.kill()


def tick_together(hosts: Iterable[Any]) -> None:
    """Release every live host at the same virtual instant, then wait for all of them."""
    live = [h for h in hosts if h is not None and h.alive]
    for host in live:
        host.release()
    for host in live:
        host.wait_idle()


# --- external-provider replicas contending for one fire ----------------------------------------

_CLAIM_ROUND: List[Optional[int]] = [None]  # one round in flight per process at a time


def replica_scheduler():
    """A minimal external ``CronScheduler``: the base class's real ``fire_due`` (store CAS claim,
    audit attempt, shared ``run_one_job``) is the whole firing path; nothing ticks."""
    from cron.scheduler_provider import CronScheduler

    class ReplicaScheduler(CronScheduler):
        @property
        def name(self) -> str:
            return "replica"

        def start(self, stop_event, **_kwargs) -> None:
            stop_event.wait()

    return ReplicaScheduler()


def install_claim_barrier(control: Control, setattr_fn=setattr, parties: int = 2) -> None:
    """Wrap ``cron.jobs.claim_job_for_fire`` (``claim_fire`` imports it at call time): each call
    waits until ``parties`` replicas have entered the claim for the current round, then runs the
    REAL claim and logs who won. It only delays, never changes a decision."""
    import cron.jobs as jobs

    real = jobs.claim_job_for_fire

    def claim(job_id, **kwargs):
        rnd = _CLAIM_ROUND[0]
        (control.replica / f"arrived-{rnd}-{os.getpid()}").touch()
        wait_until(lambda: len(list(control.replica.glob(f"arrived-{rnd}-*"))) >= parties,
                   f"{parties} replicas inside the fire claim of round {rnd}")
        result = real(job_id, **kwargs)
        _append_jsonl(control.claims, {"round": rnd, "pid": os.getpid(), "job_id": job_id,
                                       "won": bool(result)})
        return result

    setattr_fn(jobs, "claim_job_for_fire", claim)


def fire_as_replica(provider, job_id: str, rnd: int) -> bool:
    _CLAIM_ROUND[0] = rnd
    return provider.fire_due(job_id)


class ReplicaHost:
    """A second replica in its own OS process: fires ``req-<n>.json`` requests in order and
    answers ``res-<n>.json`` with whether its ``fire_due`` claimed the fire."""

    def __init__(self, control: Control, env: Dict[str, str], repo_root: Path):
        self.control, self.env, self.repo_root = control, env, repo_root
        self.proc: Optional[subprocess.Popen] = None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else None

    def _check(self) -> None:
        if self.proc.poll() is not None:
            tail = (self.control.replica / "stderr.log").read_text(encoding="utf-8")[-4000:]
            raise AssertionError(f"replica exited rc={self.proc.returncode}:\n{tail}")

    def start(self) -> None:
        log = open(self.control.replica / "stderr.log", "a", encoding="utf-8")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(self.control.root), "replica"],
            cwd=str(self.repo_root), env=self.env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        log.close()

        def ready():
            self._check()
            return (self.control.replica / "ready").exists()

        wait_until(ready, "replica process ready", timeout=180.0)

    def request(self, rnd: int, job_id: str) -> None:
        _atomic_write(self.control.replica / f"req-{rnd}.json", json.dumps({"job_id": job_id}))

    def result(self, rnd: int) -> bool:
        path = self.control.replica / f"res-{rnd}.json"

        def done():
            self._check()
            return path.exists()

        wait_until(done, f"replica result for round {rnd}")
        return json.loads(path.read_text(encoding="utf-8"))["won"]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            (self.control.replica / "stop").write_text("1", encoding="utf-8")
            try:
                self.proc.wait(timeout=DEADLINE_SECONDS)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=DEADLINE_SECONDS)


def _replica_main(control: Control) -> None:
    install(VirtualClock(control.clock_file), control)
    install_claim_barrier(control)
    provider = replica_scheduler()
    (control.replica / "ready").touch()
    rnd = 0
    while True:
        rnd += 1
        req = control.replica / f"req-{rnd}.json"
        deadline = _real_time.monotonic() + HOLD_DEADLINE_SECONDS
        while not req.exists():
            if (control.replica / "stop").exists() or _real_time.monotonic() >= deadline:
                return
            _real_time.sleep(POLL_SECONDS)
        won = fire_as_replica(provider, json.loads(req.read_text(encoding="utf-8"))["job_id"], rnd)
        _atomic_write(control.replica / f"res-{rnd}.json", json.dumps({"won": won}))


# --- ledger / store readers ------------------------------------------------------------------

def ledger_rows(home: Path, where: str = "") -> List[dict]:
    path = Path(home) / "cron" / "executions.db"
    if not path.exists():
        return []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        sql = "SELECT * FROM executions" + (f" WHERE {where}" if where else "")
        return [dict(r) for r in conn.execute(sql)]
    finally:
        conn.close()


def non_terminal(home: Path, ignore: Iterable[str] = ()) -> List[dict]:
    ignore = set(ignore)
    return [r for r in ledger_rows(home, "status IN ('claimed', 'running')")
            if r["id"] not in ignore]


def store_jobs(home: Path) -> List[dict]:
    path = Path(home) / "cron" / "jobs.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("jobs", [])


def _child_main(control_root: str) -> None:
    control = Control(Path(control_root))
    clock = VirtualClock(control.clock_file)
    install(clock, control)
    from cron.scheduler_provider import InProcessCronScheduler

    InProcessCronScheduler().start(_FileGate(control), interval=60)


if __name__ == "__main__":  # child ticker (or replica) process
    sys.path.insert(0, os.getcwd())
    if sys.argv[2:] == ["replica"]:
        _replica_main(Control(Path(sys.argv[1])))
    else:
        _child_main(sys.argv[1])
