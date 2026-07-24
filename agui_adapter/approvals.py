"""Parked-while-blocked approval registry for the AG-UI adapter.

A dangerous-command approval blocks the Hermes worker thread inside the
approval callback. Because AG-UI's native interrupt lifecycle ends the run at
the interrupt and resumes on a NEW request, the
blocked worker thread is *parked* (kept alive) in this registry keyed by
thread_id; the resume run re-attaches and resolves its decision so the command
runs inline. State exists ONLY while an approval is blocked. Single process
only (one uvicorn worker): the parked worker and its asyncio.Queue live on the
process's single event loop.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Process-env flags that route tools.approval.check_all_command_guards into its
# gateway/ask branch (see tools/approval.py::_is_gateway_approval_context and
# the ``is_ask`` check). Unlike the session *platform*, these are read straight
# from os.environ and are NOT overridable by the interactive contextvar the
# AG-UI worker installs. An AG-UI server uses the native interrupt/PARK approval
# path, so if either flag is inherited into this process every dangerous command
# would be silently diverted into the gateway fallback (submit_pending with no
# listener) and the interrupt would never fire — the same failure class as
# setting a session platform. They never legitimately apply to this surface.
_INTERRUPT_PREEMPTING_ENV = ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK")


def neutralize_interrupt_preempting_env() -> "list[str]":
    """Clear inherited gateway/ask env flags that would bypass the interrupt.

    Returns the names that were set (and have now been cleared) so the caller
    can log them. Safe to call once at app construction: an AG-UI process is
    definitionally an interrupt-approval surface, never a gateway/ask context.

    NOTE: this mutates the process-global ``os.environ`` (it pops the flags),
    which is a deliberate side effect — an embedder sharing one process with a
    gateway surface that legitimately sets these flags should not also build the
    AG-UI app in that process. ``create_app`` logs a WARNING naming what it
    cleared.
    """
    cleared = [k for k in _INTERRUPT_PREEMPTING_ENV if os.environ.get(k)]
    for k in cleared:
        os.environ.pop(k, None)
    return cleared

# Pushed onto a run's event queue (as the 2-tuple (PARK, interrupt)) to tell
# the CURRENT stream to stop draining and leave the worker parked; the stream
# uses its own run_id to build the RUN_FINISHED interrupt outcome around the
# carried Interrupt payload.
PARK = object()
# The worker turn is fully done; the stream emits its terminal RUN_FINISHED.
DONE = object()
# Pushed onto a run's queue (as the 2-tuple (ERROR, message)) by the worker on
# failure; BOTH the fresh and resume drain sites translate it into a RUN_ERROR.
ERROR = object()

_VALID_SCOPES = {"once", "session", "always"}


@dataclass(frozen=True)
class PendingApproval:
    """One pending approval (immutable). The ``decision`` future's *reference*
    is fixed at construction; the future itself is resolved to
    once/session/always/deny by whichever of the resume path or the timeout
    wins."""
    interrupt_id: str
    command: str
    description: str
    tool_call_id: Optional[str]
    allow_permanent: bool
    decision: "concurrent.futures.Future"


@dataclass(frozen=True)
class ParkedRun:
    thread_id: str
    queue: Any  # asyncio.Queue shared across the run-1 and resume streams
    pending: PendingApproval


_lock = threading.Lock()
_parked: dict[str, ParkedRun] = {}


def register(run: ParkedRun) -> bool:
    with _lock:
        if run.thread_id in _parked:
            return False
        _parked[run.thread_id] = run
        return True


def take(thread_id: str) -> Optional[ParkedRun]:
    with _lock:
        return _parked.pop(thread_id, None)


def discard(thread_id: str, expected: Optional[ParkedRun] = None) -> None:
    with _lock:
        if expected is None or _parked.get(thread_id) is expected:
            _parked.pop(thread_id, None)


def is_parked(thread_id: str) -> bool:
    with _lock:
        return thread_id in _parked


def resume_to_decision(resume_entries, interrupt_id: str) -> str:
    """Map AG-UI resume[] → Hermes decision. Fail closed to 'deny'."""
    for e in resume_entries or []:
        if getattr(e, "interrupt_id", None) != interrupt_id:
            continue
        if getattr(e, "status", None) != "resolved":
            # Anything other than an explicit "resolved" (e.g. "cancelled",
            # or an unexpected/future status) fails closed.
            return "deny"
        payload = getattr(e, "payload", None)
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("approved") is not True:
            return "deny"
        scope = payload.get("scope") or "once"
        return scope if scope in _VALID_SCOPES else "once"
    return "deny"


def approval_response_schema(allow_permanent: bool = True) -> dict:
    scopes = ["once", "session", "always"] if allow_permanent else ["once", "session"]
    return {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},
            "scope": {"type": "string", "enum": scopes},
        },
        "required": ["approved"],
    }


def expiry_iso(seconds: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))


def make_approval_callback(*, thread_id: str, emit, queue,
                           last_tool_call_id, new_id, timeout: float):
    """Build the Hermes approval callback for one AG-UI run.

    On a dangerous command it emits a (PARK, interrupt) marker so the DRAINING
    stream — which knows its own run_id — builds the native RUN_FINISHED
    interrupt outcome, then BLOCKS the worker thread on a decision future the
    resume run resolves. Returns the Hermes decision string. On timeout/error
    → 'deny' (silence != consent), unless a racing resume set the future first.
    """
    from ag_ui.core import Interrupt

    def _cb(command: str, description: str, *, allow_permanent: bool = True, **_) -> str:
        interrupt_id = new_id("int")
        decision: "concurrent.futures.Future" = concurrent.futures.Future()
        run = ParkedRun(
            thread_id=thread_id, queue=queue,
            pending=PendingApproval(interrupt_id, command, description,
                                    last_tool_call_id(), allow_permanent, decision),
        )
        if not register(run):
            logger.warning(
                "AG-UI: approval already pending for thread %s; denying concurrent request",
                thread_id,
            )
            return "deny"
        interrupt = Interrupt(
            id=interrupt_id,
            reason="tool_call",
            tool_call_id=last_tool_call_id(),
            message=f"{command}\n\n{description}",
            response_schema=approval_response_schema(allow_permanent),
            expires_at=expiry_iso(int(timeout)),
        )
        # Server-side audit of WHICH command was gated (the only such record on
        # this surface — core doesn't log the command on the interactive-approval
        # path). force=True makes this a hard safety boundary: the command is
        # redacted even if the operator globally disabled redaction
        # (security.redact_secrets: false), because a dangerous command's argv is
        # exactly where secrets appear and INFO logs may be shipped off-box. So no
        # credential can reach this log regardless of caller OR config.
        try:
            from agent.redact import redact_sensitive_text
            _cmd_preview = redact_sensitive_text(command, force=True)[:80]
        except Exception:  # noqa: BLE001 - never let logging break the gate
            _cmd_preview = "<redaction unavailable>"
        logger.info("AG-UI dangerous command awaiting approval (thread=%s): %s",
                    thread_id, _cmd_preview)
        emit((PARK, interrupt))
        try:
            return decision.result(timeout=timeout)
        except Exception:
            # Timeout or error: claim the future for "deny" so a racing resume that
            # then calls set_result() gets InvalidStateError and reports cleanly.
            claimed_deny = False
            if not decision.done():
                try:
                    decision.set_result("deny")
                    claimed_deny = True
                except Exception:
                    pass
            discard(thread_id, expected=run)
            if claimed_deny:
                # We won the race: nobody resumed within the timeout -> deny.
                logger.info("AG-UI approval timed out; denying (thread=%s)", thread_id)
            try:
                # Honor whoever set the future first: a resume-approval may
                # have landed microseconds before the timeout fired.
                return decision.result()
            except Exception:
                return "deny"

    return _cb
