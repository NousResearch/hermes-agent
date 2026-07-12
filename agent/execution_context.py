"""Context-local identity for agent execution authority.

Kanban workers are detached subprocesses, so ambient gateway/cron environment
markers are not a trustworthy description of who owns a tool call. The
dispatcher instead supplies a one-use launch ticket over a dedicated bootstrap
stream, bound to the exact active task run and worker PID. Worker startup
validates the handoff before ``AIAgent`` construction; exactly one agent
then consumes the process-local capability and binds it only while its
conversation runs.

The ContextVar is also important inside a worker process: delegated child
agents share the same process environment as the card owner, but they do not
own the card's approval grant.  ``bind_agent_execution_context`` therefore
downgrades any agent with ``_delegate_depth > 0`` to ``DELEGATE``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
from collections.abc import MutableMapping
from contextvars import ContextVar, Token
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional


class ExecutionRole(str, Enum):
    """Authority-bearing role of the agent whose turn is currently running."""

    DIRECT = "direct"
    KANBAN_OWNER = "kanban_owner"
    KANBAN_DELEGATE = "kanban_delegate"
    DELEGATE = "delegate"


_EXECUTION_ROLE: ContextVar[ExecutionRole] = ContextVar(
    "_EXECUTION_ROLE", default=ExecutionRole.DIRECT
)
_KANBAN_PAUSE_SECRET = secrets.token_bytes(32)
_KANBAN_PAUSE_TOKEN_FIELD = "_hermes_kanban_pause_token"
_KANBAN_OWNER_MARKER = "HERMES_KANBAN_SESSION"
_KANBAN_OWNER_NONCE = "HERMES_KANBAN_OWNER_BOOTSTRAP_NONCE"
_KANBAN_DELEGATE_MARKER = "HERMES_KANBAN_DELEGATE_SESSION"
_KANBAN_OWNER_LAUNCH_MARKER = "_HERMES_KANBAN_BOOTSTRAP_STDIN"
_KANBAN_OWNER_CLAIM_LOCK = threading.Lock()
_KANBAN_OWNER_LAUNCH_MAX_BYTES = 16 * 1024
_KANBAN_OWNER_LAUNCH_MAX_TTL_SECONDS = 60

# Process-local launch state. Only the trusted worker-startup path may move
# this from ``uninitialized`` to ``ready``. AIAgent construction consumes the
# ready state exactly once; environment mappings never participate in that
# transition.
_KANBAN_OWNER_LAUNCH_STATE = "uninitialized"


def _scrub_kanban_owner_launch_environment(
    source: MutableMapping[str, str],
) -> None:
    """Remove launch authority and leave only non-owner child identity."""

    owner_attempt = any(
        key in source
        for key in (
            _KANBAN_OWNER_LAUNCH_MARKER,
            _KANBAN_OWNER_MARKER,
            _KANBAN_OWNER_NONCE,
        )
    )
    source.pop(_KANBAN_OWNER_LAUNCH_MARKER, None)
    source.pop(_KANBAN_OWNER_MARKER, None)
    source.pop(_KANBAN_OWNER_NONCE, None)
    if owner_attempt:
        source[_KANBAN_DELEGATE_MARKER] = "1"


def _positive_ticket_int(value: Any) -> Optional[int]:
    """Return a positive JSON integer, rejecting bool and coercions."""

    if type(value) is not int or value <= 0:
        return None
    return value


def _nonempty_ticket_text(value: Any, *, max_length: int) -> Optional[str]:
    """Return one bounded, non-empty ticket string without outer whitespace."""

    if not isinstance(value, str) or len(value) > max_length:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def initialize_kanban_owner_launch_from_stream(stream: Any = None) -> bool:
    """Consume one dispatcher-to-worker launch ticket.

    This function is the only worker-side entry point that can install the
    process-local capability later consumed by :func:`execution_role_for_new_agent`.
    Merely setting Kanban environment variables, including the launch marker,
    never grants authority.

    The dispatcher sets ``_HERMES_KANBAN_BOOTSTRAP_STDIN=1`` and sends one
    UTF-8 JSON line through the worker's dedicated bootstrap stream. The line is
    capped at 16 KiB and must bind a short-lived token to the exact database,
    task, run, profile, claim, and worker PID. The selected board database is
    taken only from the validated ticket; no ``HERMES_KANBAN_DB`` fallback
    participates in owner selection.

    This is a correctness boundary inside Hermes's same-account trust envelope:
    it prevents ambient markers, explicit mappings, and supported child-launch
    paths from accidentally promoting an agent. It does not claim isolation
    from code already running as the same OS user and deliberately invoking
    private internals or rewriting the operator-owned board database.

    The initialization attempt is process-one-shot. It returns ``True`` only
    when the database atomically consumed the ticket and the in-process
    capability was installed. Every failure returns ``False`` and fails
    closed. Launch and legacy owner markers are scrubbed in all cases.
    """

    global _KANBAN_OWNER_LAUNCH_STATE

    source = os.environ
    launch_requested = source.get(_KANBAN_OWNER_LAUNCH_MARKER) == "1"
    with _KANBAN_OWNER_CLAIM_LOCK:
        if _KANBAN_OWNER_LAUNCH_STATE != "uninitialized":
            _scrub_kanban_owner_launch_environment(source)
            return False
        _KANBAN_OWNER_LAUNCH_STATE = (
            "loading" if launch_requested else "disabled"
        )

    if not launch_requested:
        _scrub_kanban_owner_launch_environment(source)
        return False

    accepted = False
    try:
        ticket_stream = stream
        if ticket_stream is None:
            ticket_stream = sys.stdin.buffer
        raw_line = ticket_stream.readline(_KANBAN_OWNER_LAUNCH_MAX_BYTES + 1)
        if isinstance(raw_line, str):
            raw_bytes = raw_line.encode("utf-8")
        elif isinstance(raw_line, (bytes, bytearray)):
            raw_bytes = bytes(raw_line)
        else:
            return False
        if not raw_bytes or len(raw_bytes) > _KANBAN_OWNER_LAUNCH_MAX_BYTES:
            return False
        try:
            ticket = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(ticket, dict) or type(ticket.get("v")) is not int:
            return False
        if ticket["v"] != 1:
            return False

        token = _nonempty_ticket_text(ticket.get("token"), max_length=512)
        db_path_raw = _nonempty_ticket_text(
            ticket.get("db_path"), max_length=16 * 1024
        )
        task_id = _nonempty_ticket_text(ticket.get("task_id"), max_length=512)
        profile = _nonempty_ticket_text(ticket.get("profile"), max_length=512)
        claim_lock = _nonempty_ticket_text(
            ticket.get("claim_lock"), max_length=1024
        )
        run_id = _positive_ticket_int(ticket.get("run_id"))
        worker_pid = _positive_ticket_int(ticket.get("worker_pid"))
        expires_at = _positive_ticket_int(ticket.get("expires_at"))
        if None in {
            token,
            db_path_raw,
            task_id,
            profile,
            claim_lock,
            run_id,
            worker_pid,
            expires_at,
        }:
            return False
        assert db_path_raw is not None
        db_path = Path(db_path_raw)
        if not db_path.is_absolute():
            return False
        if worker_pid != os.getpid():
            return False
        now = int(time.time())
        if not (
            now < expires_at <= now + _KANBAN_OWNER_LAUNCH_MAX_TTL_SECONDS
        ):
            return False

        from hermes_cli import kanban_db as kb

        # Deliberately pass the ticket path explicitly. Environment-selected
        # paths are ambient routing hints, not handoff validation.
        conn = kb.connect(db_path)
        try:
            consumed = bool(
                kb._consume_task_owner_bootstrap(
                    conn,
                    task_id=task_id,
                    run_id=run_id,
                    profile=profile,
                    claim_lock=claim_lock,
                    nonce=token,
                    worker_pid=worker_pid,
                    expires_at=expires_at,
                )
            )
        finally:
            conn.close()
        accepted = consumed
        return accepted
    except Exception:
        return False
    finally:
        with _KANBAN_OWNER_CLAIM_LOCK:
            if _KANBAN_OWNER_LAUNCH_STATE == "loading":
                _KANBAN_OWNER_LAUNCH_STATE = (
                    "ready" if accepted else "rejected"
                )
        _scrub_kanban_owner_launch_environment(source)


def _kanban_pause_claims(
    *,
    request_id: Any,
    task_id: Any,
    run_id: Any,
    profile: Any,
    display_target: Any,
    description: Any,
    outcome: Any,
) -> bytes:
    payload = {
        "description": str(description or ""),
        "display_target": str(display_target or ""),
        "outcome": str(outcome or "approval_pending"),
        "profile": str(profile or ""),
        "request_id": str(request_id or ""),
        "run_id": str(run_id or ""),
        "task_id": str(task_id or ""),
        "version": 1,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def issue_kanban_approval_pause_token(
    *,
    request_id: Any,
    task_id: Any,
    run_id: Any,
    profile: Any,
    display_target: Any,
    description: Any,
    outcome: Any = "approval_pending",
) -> str:
    """Sign one internal worker-halt marker bound to the exact owner run.

    Routable and unavailable outcomes attest a durable broker/card transition.
    ``approval_persistence_failed`` attests only that trusted approval code hit
    a fail-closed control path; it deliberately makes no claim that the card
    was parked or its worker slot was released.
    """

    claims = _kanban_pause_claims(
        request_id=request_id,
        task_id=task_id,
        run_id=run_id,
        profile=profile,
        display_target=display_target,
        description=description,
        outcome=outcome,
    )
    digest = hmac.new(_KANBAN_PAUSE_SECRET, claims, hashlib.sha256).hexdigest()
    return f"v1:{digest}"


def execution_role_from_environment(
    environ: Optional[Mapping[str, str]] = None,
) -> ExecutionRole:
    """Classify non-owner environment identity without granting authority.

    Owner authority is never derivable from a mapping alone; it requires the
    process-local capability installed by
    :func:`initialize_kanban_owner_launch_from_stream`. Any apparent owner
    bootstrap is therefore classified as a delegate here.
    """

    source = os.environ if environ is None else environ
    if source.get(_KANBAN_DELEGATE_MARKER) == "1":
        return ExecutionRole.KANBAN_DELEGATE
    if (
        _KANBAN_OWNER_LAUNCH_MARKER in source
        or _KANBAN_OWNER_MARKER in source
        or _KANBAN_OWNER_NONCE in source
    ):
        # Environment classification alone can never validate an owner.
        # Only the launch stream plus exact DB consumption may install the
        # process capability that returns KANBAN_OWNER.
        return ExecutionRole.KANBAN_DELEGATE
    return ExecutionRole.DIRECT


def execution_role_for_new_agent(
    environ: Optional[Mapping[str, str]] = None,
    *,
    claim_kanban_owner: bool = False,
) -> ExecutionRole:
    """Resolve authority for a newly-constructed agent.

    Owner authority comes only from the process-local capability installed by
    :func:`initialize_kanban_owner_launch_from_stream`. Supplying a mapping can
    classify an agent as a delegate, but can never consume that capability or
    grant owner authority. Only the CLI's explicitly identified primary worker
    construction may claim the installed capability; startup/plugin helpers
    and every later construction remain Kanban delegates.
    """

    global _KANBAN_OWNER_LAUNCH_STATE

    source = os.environ if environ is None else environ
    with _KANBAN_OWNER_CLAIM_LOCK:
        snapshot = dict(source)
        owner_attempt = (
            _KANBAN_OWNER_LAUNCH_MARKER in snapshot
            or _KANBAN_OWNER_MARKER in snapshot
            or _KANBAN_OWNER_NONCE in snapshot
        )
        if isinstance(source, MutableMapping):
            _scrub_kanban_owner_launch_environment(source)

        # Explicit mappings are data, never launch authority. In particular,
        # tests/plugins cannot pass an attacker-selected database and nonce to
        # consume the real process capability.
        if environ is not None:
            if owner_attempt:
                return ExecutionRole.KANBAN_DELEGATE
            return execution_role_from_environment(snapshot)

        if _KANBAN_OWNER_LAUNCH_STATE == "ready":
            if claim_kanban_owner:
                _KANBAN_OWNER_LAUNCH_STATE = "consumed"
                source[_KANBAN_DELEGATE_MARKER] = "1"
                return ExecutionRole.KANBAN_OWNER
            return ExecutionRole.KANBAN_DELEGATE
        if _KANBAN_OWNER_LAUNCH_STATE in {
            "loading",
            "rejected",
            "consumed",
        }:
            return ExecutionRole.KANBAN_DELEGATE

        if owner_attempt:
            return ExecutionRole.KANBAN_DELEGATE
        return execution_role_from_environment(snapshot)


def bind_agent_execution_context(agent: Any) -> Token:
    """Bind *agent*'s execution role for one conversation turn."""

    base_role = getattr(agent, "_execution_role", ExecutionRole.DIRECT)
    try:
        base_role = ExecutionRole(base_role)
    except (TypeError, ValueError):
        base_role = ExecutionRole.DIRECT

    if getattr(agent, "_delegate_depth", 0) > 0:
        role = (
            ExecutionRole.KANBAN_DELEGATE
            if base_role in {
                ExecutionRole.KANBAN_OWNER,
                ExecutionRole.KANBAN_DELEGATE,
            }
            else ExecutionRole.DELEGATE
        )
    else:
        role = base_role
    return _EXECUTION_ROLE.set(role)


def reset_agent_execution_context(token: Token) -> None:
    """Restore the execution role that preceded a conversation turn."""

    _EXECUTION_ROLE.reset(token)


def current_execution_role() -> ExecutionRole:
    """Return the role bound to the current context."""

    return _EXECUTION_ROLE.get()


def is_kanban_owner_context() -> bool:
    """Return whether this tool call belongs to the Kanban card owner.

    Approval code should use this helper instead of reading process-global
    environment variables.  ContextVars propagate into Hermes' concurrent
    tool workers, while remaining isolated from unrelated turns and delegates.
    """

    return _EXECUTION_ROLE.get() is ExecutionRole.KANBAN_OWNER


def is_kanban_delegate_context() -> bool:
    """Return whether a delegated child originated from a Kanban card owner.

    Such a child must neither consume the owner's one-use grant nor fall into
    the legacy unattended fail-open path. Approval policy uses this separate
    role to require the delegate's explicit callback or deny the action.
    """

    return _EXECUTION_ROLE.get() is ExecutionRole.KANBAN_DELEGATE


def kanban_approval_pending_metadata(result: Any) -> Optional[dict[str, Any]]:
    """Verify and normalize a durable-approval pause marker.

    Tool results are untrusted model/tool data. A remote MCP server or plugin
    can emit the same public JSON fields as Hermes, so those fields alone must
    never become agent-loop control flow. Only a card-owner context may halt,
    and the marker must carry the process-local capability issued by trusted
    approval code. Persisted outcomes attest that the broker atomically parked
    the task/run; ``approval_persistence_failed`` attests only a fail-closed
    process halt and must never be presented as a parked/released card. The
    capability also authenticates the redacted display metadata and outcome.
    """

    if not is_kanban_owner_context():
        return None

    payload: Any = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("status") != "kanban_approval_pending"
        or payload.get("kanban_approval_pending") is not True
    ):
        return None

    request_id = payload.get("request_id")
    display_target = payload.get("display_target") or ""
    description = payload.get("description") or "approval required"
    outcome = str(payload.get("outcome") or "approval_pending")
    supplied_token = payload.get(_KANBAN_PAUSE_TOKEN_FIELD)
    task_id = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    run_raw = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    profile = os.environ.get("HERMES_PROFILE", "").strip()
    if not isinstance(request_id, str) or not request_id.strip():
        return None
    if not isinstance(supplied_token, str):
        return None
    if outcome not in {
        "approval_pending",
        "approval_unavailable",
        "approval_persistence_failed",
    }:
        return None
    if not task_id or not run_raw or not profile:
        return None
    try:
        int(run_raw)
    except ValueError:
        return None
    expected_token = issue_kanban_approval_pause_token(
        request_id=request_id.strip(),
        task_id=task_id,
        run_id=run_raw,
        profile=profile,
        display_target=display_target,
        description=description,
        outcome=outcome,
    )
    if not hmac.compare_digest(supplied_token, expected_token):
        return None

    return {
        "approved": False,
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": request_id.strip(),
        "display_target": str(display_target),
        "description": str(description),
        "outcome": outcome,
        "error": "",
    }


__all__ = [
    "ExecutionRole",
    "bind_agent_execution_context",
    "current_execution_role",
    "execution_role_from_environment",
    "execution_role_for_new_agent",
    "initialize_kanban_owner_launch_from_stream",
    "issue_kanban_approval_pause_token",
    "is_kanban_delegate_context",
    "is_kanban_owner_context",
    "kanban_approval_pending_metadata",
    "reset_agent_execution_context",
]
