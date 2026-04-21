"""Finalizer gate validators for agent_bus tasks.

Implements Gate A (payload validation) and Gate B (transition check) per spec at
`~/wiki/operations/finalizer-gate-spec-v0.md`.

Pure validators — no DB access, no side effects. Caller (core.complete_task /
core.fail_task / core.keep_alive_task) decides how to act on the result.

Environment variables
---------------------
HERMES_FINALIZER_GATE:
    off       — validators callable but results ignored (legacy behaviour)
    advisory  — log warnings on violations, do not raise
    core      — raise on payload/transition violations (default)
    full      — core + session exit gate (reserved for Slice 5)

FINALIZER_KEEPALIVE_TIMEOUT_SEC:
    Default 1800 (30 minutes). Extension granted when keep_alive is called.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# -------- Gate modes --------
MODE_OFF = "off"
MODE_ADVISORY = "advisory"
MODE_CORE = "core"
MODE_FULL = "full"
VALID_MODES = {MODE_OFF, MODE_ADVISORY, MODE_CORE, MODE_FULL}
DEFAULT_MODE = MODE_CORE


def get_mode() -> str:
    """Return the current gate mode, defaulting to ``core``."""
    mode = os.environ.get("HERMES_FINALIZER_GATE", DEFAULT_MODE).lower()
    return mode if mode in VALID_MODES else DEFAULT_MODE


def is_enforcing() -> bool:
    """True when the gate should raise on violations."""
    return get_mode() in {MODE_CORE, MODE_FULL}


def keepalive_timeout_sec() -> int:
    """Seconds to extend deadline when keep_alive is called."""
    try:
        return max(60, int(os.environ.get("FINALIZER_KEEPALIVE_TIMEOUT_SEC", "1800")))
    except (TypeError, ValueError):
        return 1800


# -------- Outcomes & statuses --------
OUTCOME_DONE = "done"
OUTCOME_FAIL = "fail"
OUTCOME_KEEP_ALIVE = "keep-alive"
VALID_OUTCOMES = {OUTCOME_DONE, OUTCOME_FAIL, OUTCOME_KEEP_ALIVE}

STATUS_PENDING = "pending"
STATUS_ACK = "ack"
STATUS_PROGRESS = "progress"
STATUS_KEEP_ALIVE = "keep-alive"
STATUS_DONE = "done"
STATUS_FAIL = "fail"
STATUS_TIMEOUT = "timeout"

TERMINAL_STATUSES = {STATUS_DONE, STATUS_FAIL, STATUS_TIMEOUT}
NON_TERMINAL_ACTIONABLE = {STATUS_ACK, STATUS_PROGRESS, STATUS_KEEP_ALIVE}


# -------- Error codes --------
ERR_MISSING_FIELD = "MISSING_FIELD"
ERR_INVALID_OUTCOME = "INVALID_OUTCOME"
ERR_INVALID_TRANSITION = "INVALID_TRANSITION"
ERR_INVALID_TERMINAL_FLIP = "INVALID_TERMINAL_FLIP"
ERR_ALREADY_TERMINAL_IDENTICAL = "ALREADY_TERMINAL_IDENTICAL"  # soft ok


# -------- Validators --------
def validate_close_payload(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate the minimal shape for closing a task.

    Required: ``task_id`` (non-empty str), ``outcome`` (in VALID_OUTCOMES),
    ``summary`` (≥ 3 chars after strip).

    Returns (ok, error_code). ``error_code`` is ``None`` on success.
    """
    if not isinstance(payload, dict):
        return False, ERR_MISSING_FIELD

    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        return False, ERR_MISSING_FIELD

    outcome = payload.get("outcome")
    if outcome not in VALID_OUTCOMES:
        return False, ERR_INVALID_OUTCOME

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False, ERR_MISSING_FIELD

    return True, None


def validate_transition(current_status: str, new_outcome: str) -> Tuple[bool, Optional[str]]:
    """Check whether moving from ``current_status`` to ``new_outcome`` is legal.

    Rules
    -----
    - Non-terminal (ack / progress / keep-alive) -> done / fail / keep-alive: legal.
    - Pending -> done / fail: illegal (must ack first).
    - Pending -> keep-alive: illegal (must ack first).
    - Terminal -> same outcome: idempotent, returns (True, ERR_ALREADY_TERMINAL_IDENTICAL).
    - Terminal -> different terminal: illegal (INVALID_TERMINAL_FLIP).
    - Terminal -> keep-alive: illegal (INVALID_TRANSITION).
    """
    if new_outcome not in VALID_OUTCOMES:
        return False, ERR_INVALID_OUTCOME

    if current_status in TERMINAL_STATUSES:
        if current_status == new_outcome:
            return True, ERR_ALREADY_TERMINAL_IDENTICAL  # soft ok sentinel
        if new_outcome == OUTCOME_KEEP_ALIVE:
            return False, ERR_INVALID_TRANSITION
        return False, ERR_INVALID_TERMINAL_FLIP

    # Non-terminal source
    if current_status == STATUS_PENDING:
        return False, ERR_INVALID_TRANSITION

    if current_status in NON_TERMINAL_ACTIONABLE:
        return True, None

    # Unknown status — reject conservatively
    return False, ERR_INVALID_TRANSITION


def is_soft_ok(code: Optional[str]) -> bool:
    """True if the code represents a soft-ok (idempotent re-close)."""
    return code == ERR_ALREADY_TERMINAL_IDENTICAL


# -------- Helper for core.* callers --------
def enforce_close(
    payload: Dict[str, Any],
    current_status: str,
    *,
    logger=None,
) -> Tuple[str, Optional[str]]:
    """Run both gates and return (decision, code).

    decision is one of:
        - "proceed"     : payload + transition valid, caller should do the write
        - "idempotent"  : same-outcome re-close, caller should return existing task row
        - "reject"      : caller must raise ValueError(code) if enforcing

    In advisory / off mode, reject still returns the code but caller should
    still proceed (and log).
    """
    ok, code = validate_close_payload(payload)
    if not ok:
        if logger is not None:
            logger.warning("finalizer: payload rejected code=%s payload=%r", code, payload)
        return ("reject", code)

    ok, code = validate_transition(current_status, payload["outcome"])
    if ok and is_soft_ok(code):
        if logger is not None:
            logger.info(
                "finalizer: idempotent close task=%s outcome=%s",
                payload.get("task_id"), payload.get("outcome"),
            )
        return ("idempotent", code)
    if not ok:
        if logger is not None:
            logger.warning(
                "finalizer: transition rejected code=%s from=%s to=%s task=%s",
                code, current_status, payload.get("outcome"), payload.get("task_id"),
            )
        return ("reject", code)

    return ("proceed", None)


__all__ = [
    "MODE_OFF", "MODE_ADVISORY", "MODE_CORE", "MODE_FULL",
    "DEFAULT_MODE", "VALID_MODES",
    "OUTCOME_DONE", "OUTCOME_FAIL", "OUTCOME_KEEP_ALIVE", "VALID_OUTCOMES",
    "STATUS_PENDING", "STATUS_ACK", "STATUS_PROGRESS", "STATUS_KEEP_ALIVE",
    "STATUS_DONE", "STATUS_FAIL", "STATUS_TIMEOUT",
    "TERMINAL_STATUSES", "NON_TERMINAL_ACTIONABLE",
    "ERR_MISSING_FIELD", "ERR_INVALID_OUTCOME", "ERR_INVALID_TRANSITION",
    "ERR_INVALID_TERMINAL_FLIP", "ERR_ALREADY_TERMINAL_IDENTICAL",
    "get_mode", "is_enforcing", "keepalive_timeout_sec",
    "validate_close_payload", "validate_transition",
    "is_soft_ok", "enforce_close",
]
