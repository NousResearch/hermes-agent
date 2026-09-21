"""Quiet ``hermes chat -Q`` helpers: bind this session's key and resume nested notifies.

Bot Mode delivers a local DM as ``hermes -p <bot> chat -Q --query-file``. Interactive
chat binds ``set_current_session_key(self.session_id)`` around the turn; the quiet
path did not, so a nested ``message_agent`` notify inherited the dispatcher's
``HERMES_SESSION_KEY`` and never woke the recipient. Quiet also printed and exited
after one turn, so a nested teammate reply that finished during the one-shot linger
was never injected as a follow-up.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, MutableMapping

# Nested A→B→C is one extra turn; this caps a runaway message_agent chain.
_MAX_QUIET_NOTIFY_ROUNDS = 8

# Last line a Kanban worker leaves in its own log: ``[kanban-worker-exit] rc=<code>``. A per-tick
# ``hermes kanban dispatch`` process never reaped the worker, so ``os.waitpid`` cannot tell it how
# the worker exited; the trailer is the process-independent witness the dead-worker sweep reads
# instead, so a clean exit without a terminal board call is booked as the same protocol violation
# (and a 75 as the same rate-limit requeue) whichever process notices the death.
KANBAN_WORKER_EXIT_TRAILER = "[kanban-worker-exit] rc="


def exit_single_query(code: int) -> None:
    """``sys.exit(code)`` for a one-shot turn; a Kanban worker first writes the exit trailer to its log."""
    if os.environ.get("HERMES_KANBAN_TASK"):
        with contextlib.suppress(Exception):
            # stderr: stdout may be the ``--stream-json`` record stream, and the worker log
            # captures both streams.
            print(f"\n{KANBAN_WORKER_EXIT_TRAILER}{int(code)}", file=sys.stderr, flush=True)
    sys.exit(code)


# A spawner that bounds only the TURN (the cron Bot Chat lane) hands the quiet child a report
# path here. The child records the turn's outcome there the moment the turn ends, BEFORE the
# one-shot exit linger, so the spawner can book the delivery and stop waiting while the linger
# keeps protecting nested ``notify_on_complete`` replies. Popped before the turn runs (same
# contract as HERMES_TURN_AUTHOR): nothing the turn spawns inherits it, and a nested one-shot
# never writes over its host's report — the record also carries the writer's pid.
TURN_REPORT_FILE_ENV = "HERMES_QUIET_TURN_REPORT_FILE"


def take_turn_report_path(environ: MutableMapping[str, str] = os.environ) -> str | None:
    """Read and remove the spawner's turn-report path so subprocesses started during the turn do not inherit it."""
    return environ.pop(TURN_REPORT_FILE_ENV, None) or None


def write_turn_report(path: str | None, *, exit_code: int, error: str = "", reply: str = "") -> None:
    """Atomically record ``{pid, exit_code, error, reply}`` at *path*; a no-op without a path. Never
    raises: the report is the spawner's convenience, the turn itself is already persisted. ``reply``
    is what the run will print — a spawner booking a lingering child from its report relays it."""
    if not path:
        return
    from utils import atomic_json_write

    record = {"pid": os.getpid(), "exit_code": int(exit_code), "error": str(error or ""), "reply": str(reply or "")}
    # 0600 from creation: the record now carries the turn's answer, like the 0600 query file beside it.
    with contextlib.suppress(Exception):
        atomic_json_write(path, record, indent=None, mode=0o600)


def read_turn_report(path: str, pid: int) -> dict | None:
    """The child's turn report, or None while absent, unreadable, or written by another process."""
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("pid") != pid:
        return None
    return record


# After the child reports its turn, a child with nothing to linger for exits at once; a spawner
# that needs only the outcome gives it that long so its real exit code and stream tails are
# booked instead of the report's summary.
REPORTED_TURN_EXIT_GRACE_SECONDS = 2.0


def activity_heartbeat_path(report_path: str) -> str:
    """Path of the activity heartbeat a quiet child refreshes BESIDE its turn report.

    A separate file, never a ``running`` field inside the report: the report is the child's
    TERMINAL record, and a pre-heartbeat spawner must never mistake a live turn's marker for
    an outcome — a field it does not know would surface as ``report.get(...)`` misses, a file
    it does not know is inert. Stamped with the writer's pid like the report, so a stale
    heartbeat from an earlier child at the same tmp path is never evidence of life.
    """
    return f"{report_path}.activity"


def start_turn_activity_heartbeat(agent, report_path: str | None, *, poll_seconds: float = 2.0):
    """Refresh the turn's activity heartbeat while *agent* keeps making progress.

    The spawner (``run_reported_turn``) bounds a delivery turn by IDLE time, not wall-clock —
    the rule the cron inactivity watchdog already applies to its own agent sessions — and this
    is the liveness signal it reads. The timestamp is the agent's OWN activity clock
    (``get_activity_summary()["last_activity_ts"]``, the same source the watchdog samples
    in-process), so "active" means the same thing on both sides of the subprocess boundary.
    Returns the stop event, or None when there is nothing to report into (no report path —
    unbounded spawner — or an agent without an activity clock: no heartbeat means the spawner
    keeps its wall-clock bound). The first write lands only once the agent has recorded
    activity: a child wedged before its turn's first event stays under the wall cap.
    """
    if not report_path:
        return None
    if not callable(getattr(agent, "get_activity_summary", None)):
        return None  # no activity clock to publish: the spawner keeps its wall-clock bound
    path = activity_heartbeat_path(report_path)
    stop = threading.Event()

    def _refresh() -> None:
        from utils import atomic_json_write

        while not stop.wait(poll_seconds):
            try:
                snapshot = agent.get_activity_summary() or {}
                ts = snapshot.get("last_activity_ts")
            except Exception:
                continue  # no evidence yet: stay under the spawner's wall-clock bound
            if ts is None:
                continue
            try:
                atomic_json_write(path, {"pid": os.getpid(), "ts": float(ts)},
                                  indent=None, mode=0o600)
            except Exception:
                return  # the spawner's tmp lane vanished mid-turn; it falls back to the wall cap

    threading.Thread(target=_refresh, daemon=True, name="quiet-turn-activity").start()
    return stop


def read_activity_heartbeat(report_path: str, pid: int) -> float | None:
    """The child's last activity timestamp (wall clock), or None while absent, unreadable, or
    written by another process — silence is never evidence of life."""
    try:
        with open(activity_heartbeat_path(report_path), encoding="utf-8") as fh:
            beat = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(beat, dict) or beat.get("pid") != pid:
        return None
    try:
        return float(beat["ts"])
    except (KeyError, TypeError, ValueError):
        return None


def _kill_at_cap(proc, drain, argv: list, timeout: float) -> None:
    """Hard-interrupt the child at a cap, keeping the kill-window report recovery: a killed
    child cannot run further, but the turn may have ENDED (and delivered) in the window
    between the last report check and the kill landing — the caller re-reads the report once
    and books that instead of misreporting a delivered turn as a timeout."""
    proc.kill()
    drain.join(timeout=5.0)


def run_reported_turn(argv: list, *, env: MutableMapping[str, str], report_path: str, timeout: float,
                      exit_grace: float | None = REPORTED_TURN_EXIT_GRACE_SECONDS, cwd: str | None = None,
                      encoding: str | None = None) -> subprocess.CompletedProcess:
    """Run one ``hermes chat -Q`` delivery child; *timeout* bounds the TURN, not the process.

    The child records its turn at *report_path* (``write_turn_report``) the moment the turn ends,
    then runs the one-shot exit linger for nested ``notify_on_complete`` replies — bounded by
    ``terminal.oneshot_completion_wait_seconds``, whose default equals the delivery caps, so
    waiting for process exit booked every delivered turn that left a reply pending as a timeout
    and killed the linger (#113608, #114980). A child that exits is booked from its real exit
    code and streams. A child still lingering once its report exists is booked from the report
    and left running (a daemon thread drains and reaps it): after *exit_grace* seconds for a
    spawner that needs only the outcome, or at the cap when *exit_grace* is None, for a spawner
    that relays the printed answer — a teammate's reply during the linger may still become it.
    Only a turn that never ends is killed, as ``subprocess.TimeoutExpired``.

    *cwd* pins the child's directory (a spawner sitting in a reaped scratch workspace must not
    hand its dead cwd on — the child dies at CLI startup, #102941). The pipes decode lossily
    everywhere: a stray non-UTF-8 byte (a grandchild sharing the pipe interleaving a partial
    multi-byte write) must not raise in the drain thread and take the reply and the failure tail
    with it (#105582). Without an explicit *encoding* they decode as UTF-8 only on win32, where
    the child is guaranteed UTF-8 (hermes_bootstrap reconfigures its streams even under
    PYTHONIOENCODING=cp1252) while the gateway parent is not started in UTF-8 mode, so the
    locale default mangled or lost accented replies (#115894); on POSIX the child keeps the
    locale codec, so the locale default stays correct there (#66566).
    """
    from hermes_cli._subprocess_compat import windows_hide_flags

    if encoding is None and sys.platform == "win32":
        encoding = "utf-8"
    proc = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding=encoding, errors="replace", env={**env, TURN_REPORT_FILE_ENV: report_path},
        cwd=cwd, creationflags=windows_hide_flags())
    streams: dict = {}

    def _drain() -> None:
        streams["out"], streams["err"] = proc.communicate()

    drain = threading.Thread(target=_drain, name=f"quiet-turn-drain-{proc.pid}", daemon=True)
    drain.start()
    # The cap bounds IDLE time, not wall-clock — the rule the cron inactivity watchdog already
    # applies to its own agent sessions ("a stalled session is hard-interrupted ... while a
    # long-but-active job is never cut off"). A child that refreshes the activity heartbeat
    # beside its turn report keeps the lane alive for as long as it keeps making progress; only
    # a turn silent for the full cap is killed. A child that never heartbeats (an older build,
    # or one wedged before its first activity) stays under the original wall-clock bound.
    idle_deadline = time.monotonic() + timeout
    wall_deadline = idle_deadline  # until a first heartbeat is seen, the two are the same clock
    last_beat = read_activity_heartbeat(report_path, proc.pid)
    heartbeat_seen = last_beat is not None
    report = None
    while True:
        drain.join(timeout=exit_grace if report is not None and exit_grace is not None else 0.25)
        if not drain.is_alive():
            return subprocess.CompletedProcess(argv, proc.returncode, streams.get("out", ""), streams.get("err", ""))
        if report is not None and exit_grace is not None:
            break
        # Re-read while waiting for the cap: a follow-up turn rewrites the report with its answer.
        report = read_turn_report(report_path, proc.pid) or report
        beat = read_activity_heartbeat(report_path, proc.pid)
        if beat is not None:
            heartbeat_seen = True
            # Wall clock at the heartbeat write, converted to this loop's monotonic clock by
            # anchoring the first sample; every later beat moves the idle deadline, never back.
            if last_beat is None:
                idle_deadline = time.monotonic() + timeout
            else:
                idle_deadline += max(beat - last_beat, 0.0)
            last_beat = beat
        now = time.monotonic()
        if now >= wall_deadline and not heartbeat_seen:
            # No heartbeat ever: the original wall-clock cap.
            report = read_turn_report(report_path, proc.pid)
            if report is not None:
                break
            _kill_at_cap(proc, drain, argv, timeout)
            report = read_turn_report(report_path, proc.pid)
            if report is not None:
                break
            raise subprocess.TimeoutExpired(argv, timeout)
        if now >= idle_deadline:
            # A heartbeating turn went silent for the full cap.
            report = read_turn_report(report_path, proc.pid)
            if report is not None:
                break
            _kill_at_cap(proc, drain, argv, timeout)
            report = read_turn_report(report_path, proc.pid)
            if report is not None:
                break
            raise subprocess.TimeoutExpired(argv, timeout)
    # Turn over, child still lingering for a nested reply: not this spawner's wait.
    return subprocess.CompletedProcess(
        argv, int(report["exit_code"]), report.get("reply") or "", report.get("error") or "")


@contextlib.contextmanager
def bind_quiet_session_key(session_id: str):
    """Bind the approval/session key to *this* quiet session for the enclosing ``with`` block."""
    from tools.approval_context import reset_current_session_key, set_current_session_key

    token = set_current_session_key(session_id or "default")
    try:
        yield
    finally:
        reset_current_session_key(token)


def _diagnostic_only_wake_muted(events) -> bool:
    """True when every drained event is an automatic diagnostic AND the CLI policy suppresses them."""
    from agent.notification_presentation import diagnostic_process_event
    from gateway.warning_notifications import warning_notifications_enabled

    if not events or not all(diagnostic_process_event(e) for e in events if isinstance(e, dict)):
        return False
    return not warning_notifications_enabled("cli")


def quiet_notify_linger_seconds() -> float:
    """Total linger budget for one quiet run: the shared ``terminal.oneshot_completion_wait_seconds``.

    One budget covers the drain loop here AND the later ``_wait_for_oneshot_background_completions``
    pass, so a stuck ``notify_on_complete`` child cannot stack round-after-round waits on top of the
    finalize re-wait (pre-fix worst case: 8 rounds x 600s + 600s).
    """
    from tools.process_registry import ProcessRegistry

    return ProcessRegistry._oneshot_completion_wait_seconds()


def continue_quiet_notify_completions(
    session_id: str,
    run_turn: Callable[[str], Any],
    *,
    owns_event=None,
    max_rounds: int = _MAX_QUIET_NOTIFY_ROUNDS,
    linger_budget: float | None = None,
) -> Any:
    """Linger for ``notify_on_complete`` work, then run owned completion texts as follow-up turns.

    Returns the last ``run_turn`` result, or ``None`` when nothing owned completed. The whole
    loop shares ONE linger budget (default: ``terminal.oneshot_completion_wait_seconds``) — a
    process that times out is waited on no further this run: after the current round's drained
    texts run, the loop stops (the finalize linger still covers it once, bounded, via the
    budget handshake below).
    """
    from tools.process_registry import process_registry
    from tools.async_delegation import claim_event_delivery, complete_event_delivery

    last: Any = None
    key = session_id or ""
    if linger_budget is None:
        linger_budget = quiet_notify_linger_seconds()
    deadline = time.monotonic() + max(float(linger_budget), 0.0)
    for _ in range(max_rounds):
        wait = process_registry.wait_for_pending_completions(None, timeout=max(deadline - time.monotonic(), 0.0))
        drained = []
        for event, text in process_registry.drain_notifications(session_key=key, owns_event=owns_event):
            # Durable async_delegation events carry a delivery ledger: without the
            # claim/complete handshake the row stays delivery_state='pending' and
            # restore_undelivered_completions re-queues it on the next process start,
            # injecting the same result twice. Same contract as every other drain consumer.
            claim = claim_event_delivery(event, "cli-quiet")
            if claim is None:
                continue
            complete_event_delivery(event, claim)
            drained.append((event, text))
        # Every drained event type carries formatted text (completions, watch matches,
        # async_delegation results): drain_notifications POPS owned events off the queue,
        # so filtering by type here would consume-and-silently-drop owned
        # async_delegation results. Keep everything that rendered.
        texts = [text for _event, text in drained if text]
        if texts:
            follow = run_turn("\n\n".join(texts))
            # Same admission rule as the interactive CLI turn: a wake made ONLY of automatic
            # diagnostics (early failure / watch notices) still runs, but under suppression its
            # reply never displaces the requested one-shot answer on stdout.
            if not _diagnostic_only_wake_muted([event for event, text in drained if text]):
                last = follow
        if wait.get("timed_out"):
            break
        if not texts:
            return last
    return last


def adopt_unanswered_turn(cli: Any, query: Any, environ: MutableMapping[str, str] = os.environ) -> bool:
    """A dispatcher's re-run of a failed delivery turn resumes the DM its first attempt already
    persisted instead of appending it again. Returns True when the tail row was adopted.

    The failed attempt's turn-start persist left the DM as the transcript's unanswered tail row. A
    fresh process cannot know that by itself (``_DB_PERSISTED_MARKER`` is in-process only), and
    inferring it from an identical tail alone would swallow a person's deliberate re-send — so the
    dispatcher must say so with ``tools.bot_relay.RESUME_UNANSWERED_TURN_ENV``, consumed (popped) here
    before the turn so tool subprocesses never inherit it. Which row counts as the unanswered DM, and
    how it is re-staged as ``_pending_cli_user_message``, is shared with the in-process peer-DM lane
    (``agent.session_persistence.adopt_unanswered_turn``, #115325).
    """
    from tools.bot_relay import RESUME_UNANSWERED_TURN_ENV

    if environ.pop(RESUME_UNANSWERED_TURN_ENV, None) != "1":
        return False
    from agent.session_persistence import adopt_unanswered_turn as _adopt_tail

    return _adopt_tail(getattr(cli, "conversation_history", None) or [], query, cli.agent)
