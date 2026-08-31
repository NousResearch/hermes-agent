"""Kanban block -> assessor escalation plugin.

Subscribes to ``kanban_task_blocked`` and spawns the assessor agent
immediately, converting a block from a dead letter into the first hop of the
fleet escalation chain (worker -> assessor -> ... -> human for critical only).

Escalation target is canonical and pinned here so it is versioned and
testable: the first hop goes to Jobsy (the PM / board owner, who owns the
scope/AC triage mandate per his SOUL); runtime/environment/profile faults go
to Agent Smith (profile id ``default``). ``switch`` is deliberately NEVER an
escalation target — its only ownership is no_agent scheduled output, and it
must not assess or close build cards.

See plugin.yaml for the full rationale and guardrails.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["register"]

# Assessor routing for the FIRST hop. Defaults to Jobsy (the PM, who owns
# scope/AC assessment); runtime/environment/profile faults go to Agent Smith
# (profile id `default`). The assessor then escalates further if needed.
# These constants are the escalation-target contract — a regression test
# asserts Jobsy is the default and `switch` is never a target.
DEFAULT_ASSESSOR = "jobsy"
RUNTIME_ASSESSOR = "default"  # Agent Smith's profile id

# Signals that a block is a runtime/environment/profile fault (Smith's lane)
# rather than a scope/AC/product decision (Jobsy's lane). Matched case-
# insensitively against the block reason text.
_RUNTIME_MARKERS = (
    "unknown skill",
    "no module named",
    "interpreter",
    "python3",
    ".venv",
    "venv",
    "pytest",
    "symlink",
    "model",
    "provider",
    "credits",
    "http 402",
    "rate limit",
    "spawn",
    "crash",
    "exit code",
    "profile",
    "gateway",
    "db",
    "sqlite",
    "malformed",
    "dependencies",
    "pip",
    "tool_use",
    "missing",
    "not found",
)


def _hermes_bin() -> str:
    return os.path.expanduser("~/.hermes/hermes-agent/venv/bin/hermes")


def _board_db_path() -> str:
    # The kanban_* tools and the dispatcher all resolve the shared board to
    # ~/.hermes/kanban.db. If HERMES_KANBAN_DB is pinned (worker env), honour
    # it so the status check reads the same board the block was written to.
    return os.environ.get("HERMES_KANBAN_DB") or os.path.expanduser(
        "~/.hermes/kanban.db"
    )


def _is_truly_blocked(task_id: str) -> bool:
    """Return True iff the task really landed in `blocked` (not dependency→todo).

    The kanban_task_blocked hook also fires for `dependency` blocks, which are
    routed to `todo` and self-resume via parent-gating. Those must NOT trigger
    an assessor. Cheap read-only probe of the board.
    """
    try:
        con = sqlite3.connect(f"file:{_board_db_path()}?mode=ro", uri=True, timeout=10)
        try:
            row = con.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        finally:
            con.close()
        return bool(row) and row[0] == "blocked"
    except Exception as exc:  # noqa: BLE001 - never break the block txn
        logger.debug("kanban-block-escalator: status probe failed: %s", exc)
        return False


def _assessor_for(task_id: str, assignee: str | None, reason: str | None) -> str | None:
    """Choose the assessor for the first hop, or None to skip (loop guard)."""
    text = (reason or "").lower()
    assessor = RUNTIME_ASSESSOR if any(m in text for m in _RUNTIME_MARKERS) else DEFAULT_ASSESSOR

    # Loop guard: never escalate a card back to the profile that blocked it.
    # If the block came from the assessor itself, climb to the other tier.
    if assignee and assignee in (assessor,):
        return RUNTIME_ASSESSOR if assessor != RUNTIME_ASSESSOR else DEFAULT_ASSESSOR
    return assessor


def on_block(task_id: str = "", assignee: str | None = None, reason: str | None = None, **kwargs) -> None:
    """kanban_task_blocked callback. Fire-and-forget assessor trigger."""
    if not task_id:
        return
    if not _is_truly_blocked(task_id):
        return

    assessor = _assessor_for(task_id, assignee, reason)
    if not assessor:
        return

    prompt = (
        f"Assess the blocked kanban card {task_id} and unblock it if resolvable. "
        f"Block reason: {reason or '(none)'.strip()!r}. "
        "Classify it (scope/AC vs runtime/env) and either resolve-and-unblock, "
        "or escalate to the next agent up the chain. Only a genuinely critical "
        "block (missing creds, owner decision, money) stops at Richie."
    )

    try:
        # Fire-and-forget in a new session so the (short-lived) firing worker
        # process can exit without orphaning the assessor. start_new_session so
        # it survives parent exit; stdout/stderr to DEVNULL.
        subprocess.Popen(
            [_hermes_bin(), "-p", assessor, "--cli", "chat", "-q", prompt],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        logger.info(
            "kanban-block-escalator: escalated task %s to assessor %s",
            task_id, assessor,
        )
    except Exception as exc:  # noqa: BLE001 - never break the block txn
        logger.warning(
            "kanban-block-escalator: failed to spawn assessor for %s: %s",
            task_id, exc,
        )


def register(ctx) -> None:
    """Register the kanban_task_blocked lifecycle hook."""
    ctx.register_hook("kanban_task_blocked", on_block)