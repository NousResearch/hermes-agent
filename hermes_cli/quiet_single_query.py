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
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, MutableMapping

logger = logging.getLogger(__name__)

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


# ── requester-death policy for a spawner-owned delivery child ────────────────
#
# A delivery child is a DISPOSABLE one-shot whose output has exactly one consumer: the process
# that spawned it (the delivery runner / the gateway's relay handler). When that requester dies
# — the fleet kills a stuck delivery, the gateway is restarted, a cron tick is reaped — nothing
# reads the child again, yet the child keeps running: it holds the TARGET profile's
# ``state.db`` (+ its WAL), its own MCP server children, the one-shot exit linger, and it keeps
# spawning deliveries on its own account. On 2026-09-20 that orphan outlived its requester by
# 1h46m and reparented to the gateway, and every later delivery to that profile was refused
# with the mid-turn refusal text (not the turn-lock one).
#
# The policy (see ``website/docs/user-guide/bot-mode.md``): the child DETECTS the dead requester
# and ends its own turn. It does not linger, because the follow-up turns the linger exists to
# protect (#90879 / #113608) would have no consumer either. Ending normally is what releases the
# target profile's ``state.db`` and reaps the MCP children — the same teardown a clean exit runs.
#
# Armed ONLY when the spawner says so (REQUESTER_PID_ENV), so a person's ``hermes chat -Q`` is
# never affected. The requester pid is consumed (popped) before the turn, like
# TURN_REPORT_FILE_ENV, so nothing the turn spawns mistakes its own parent for the requester.
REQUESTER_PID_ENV = "HERMES_QUIET_REQUESTER_PID"
# Distinct from every other one-shot exit code: the turn was cut short because its requester
# exited, not because the turn failed. Also outside the kanban codes (75 rate-limit / 78
# terminal-provider), which describe a worker's own provider wall.
REQUESTER_DEATH_EXIT_CODE = 79
# Poll cadence and the bound on letting an interrupted turn unwind before the child finalizes.
REQUESTER_DEATH_POLL_SECONDS = 1.0
REQUESTER_DEATH_GRACE_SECONDS = 5.0


def peek_requester_pid(environ: MutableMapping[str, str] = os.environ) -> int | None:
    """The spawner's pid without consuming it (the active-session claim runs before the turn)."""
    return _parse_pid(environ.get(REQUESTER_PID_ENV))


def take_requester_pid(environ: MutableMapping[str, str] = os.environ) -> int | None:
    """Read and remove the spawner's pid so subprocesses started during the turn do not inherit it."""
    return _parse_pid(environ.pop(REQUESTER_PID_ENV, None))


def _parse_pid(raw: "str | None") -> int | None:
    try:
        pid = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def requester_pid_env(pid: int | None = None) -> dict[str, str]:
    """Child-env entry that arms the requester-death policy in a delivery child."""
    return {REQUESTER_PID_ENV: str(os.getpid() if pid is None else pid)}


def requester_alive(requester_pid: int, *, getppid=os.getppid, pid_exists=None) -> bool:
    """Whether ``requester_pid`` is still this child's living parent.

    Reparenting is the primary signal: the kernel moves a child to the nearest subreaper (the
    gateway) or init the moment its parent is REAPED, so ``getppid()`` is exact, instant and
    free (the same mechanism ``tui_gateway.slash_worker`` relies on). The pid probe only
    corroborates, for the window in which the requester is a zombie its own parent has not
    reaped yet — ``getppid()`` still names it there.
    """
    if getppid() != int(requester_pid):
        return False
    if pid_exists is None:
        try:
            from gateway.status import _pid_exists as pid_exists
        except Exception:  # pragma: no cover — no gateway module on a stripped install
            return True
    try:
        return bool(pid_exists(int(requester_pid)))
    except Exception:
        return True


def arm_requester_watchdog(
    requester_pid: int | None, on_requester_death: Callable[[], Any], *,
    poll_seconds: float | None = None,
    getppid=os.getppid, pid_exists=None,
) -> threading.Event:
    """End the enclosing one-shot when its requester dies; returns the stop Event.

    ``on_requester_death`` runs exactly once, on the watcher thread, as soon as the requester is
    gone — the caller's job there is to stop an in-flight turn (interrupt) and mark the run so
    the main thread exits through the normal finalize path. A no-op (never fires) without a
    requester pid, so an ordinary quiet one-shot is untouched.

    ``poll_seconds`` defaults to :data:`REQUESTER_DEATH_POLL_SECONDS` resolved at call time, so
    the cadence is tunable (tests, an operator leash) without reaching into the signature.
    """
    stop = threading.Event()
    pid = requester_pid
    if not pid:
        return stop
    interval = REQUESTER_DEATH_POLL_SECONDS if poll_seconds is None else float(poll_seconds)

    def _await_requester_death() -> None:
        while True:
            # Check BEFORE the first wait: a requester can already be gone when the child starts
            # (the spawner died between fork and the turn), and that case must not wait a cadence.
            if requester_alive(pid, getppid=getppid, pid_exists=pid_exists):
                if stop.wait(max(0.01, interval)):
                    return
                continue
            logger.warning(
                "delivery child %s: requester pid %s is gone — ending this run instead of "
                "outliving it (holding a target profile's state.db and MCP children with no "
                "consumer left for this turn)", os.getpid(), pid)
            try:
                on_requester_death()
            except Exception:  # a failed callback must not leave the orphan alive
                logger.debug("requester-death callback failed", exc_info=True)
            return

    threading.Thread(target=_await_requester_death, name=f"requester-death-{pid}",
                     daemon=True).start()
    return stop


# After the child reports its turn, a child with nothing to linger for exits at once; a spawner
# that needs only the outcome gives it that long so its real exit code and stream tails are
# booked instead of the report's summary.
REPORTED_TURN_EXIT_GRACE_SECONDS = 2.0


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
    deadline = time.monotonic() + timeout
    report = None
    while True:
        drain.join(timeout=exit_grace if report is not None and exit_grace is not None else 0.25)
        if not drain.is_alive():
            return subprocess.CompletedProcess(argv, proc.returncode, streams.get("out", ""), streams.get("err", ""))
        if report is not None and exit_grace is not None:
            break
        # Re-read while waiting for the cap: a follow-up turn rewrites the report with its answer.
        report = read_turn_report(report_path, proc.pid) or report
        if time.monotonic() >= deadline:
            if report is not None:
                break
            proc.kill()
            drain.join(timeout=5.0)
            # A killed child cannot run further, but the turn may have ENDED (and delivered)
            # in the window between the last report check and the kill landing. Re-read once:
            # a report that appeared means the turn completed — book it instead of
            # misreporting a delivered turn as a timeout (and never re-notifying).
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
