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
import sys
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


def write_turn_report(path: str | None, *, exit_code: int, error: str = "") -> None:
    """Atomically record ``{pid, exit_code, error}`` at *path*; a no-op without a path. Never raises:
    the report is the spawner's convenience, the turn itself is already persisted."""
    if not path:
        return
    record = {"pid": os.getpid(), "exit_code": int(exit_code), "error": str(error or "")}
    with contextlib.suppress(Exception):
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh)
        os.replace(tmp, path)


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
