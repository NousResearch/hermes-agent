"""opportunistic-flagging plugin — auto-assess kanban tasks on completion.

Wires ``post_tool_call`` to detect ``kanban_complete`` calls and append
a self-assessment comment when any threshold is crossed:

- total token usage > 30 000
- retries > 3
- task duration > 60 s

All thresholds are advisory; the plugin never blocks completion.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

TOKEN_THRESHOLD = 30000
RETRY_THRESHOLD = 3
DURATION_THRESHOLD_SECONDS = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_session_tokens() -> Tuple[int, int]:
    """Return (input_tokens, output_tokens) from the most recent session in state.db."""
    try:
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "state.db"
        if not db_path.exists():
            return 0, 0
        sconn = sqlite3.connect(str(db_path))
        sconn.row_factory = sqlite3.Row
        try:
            row = sconn.execute(
                "SELECT input_tokens, output_tokens FROM sessions "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row:
                return int(row["input_tokens"] or 0), int(row["output_tokens"] or 0)
        finally:
            sconn.close()
    except Exception:
        logger.debug("opportunistic-flagging: failed to read session tokens", exc_info=True)
    return 0, 0


def _build_note(total_tokens: int, retries: int, duration_sec: int) -> Optional[str]:
    """Build a self-assessment comment if any threshold is crossed."""
    flags: list[str] = []
    if total_tokens > TOKEN_THRESHOLD:
        flags.append(f"used {total_tokens:,} tokens")
    if retries > RETRY_THRESHOLD:
        flags.append(f"required {retries} retries")
    if duration_sec > DURATION_THRESHOLD_SECONDS:
        mins, secs = divmod(duration_sec, 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        flags.append(f"took {time_str}")

    if not flags:
        return None

    if len(flags) == 1:
        note = f"Note: This task {flags[0]}."
    elif len(flags) == 2:
        note = f"Note: This task {flags[0]} and {flags[1]}."
    else:
        note = f"Note: This task {flags[0]}, {flags[1]}, and {flags[2]}."

    hints: list[str] = []
    if total_tokens > TOKEN_THRESHOLD:
        hints.append("consider chunking, summarizing, or using more targeted context")
    if retries > RETRY_THRESHOLD:
        hints.append("consider breaking into smaller steps or improving the initial prompt")
    if duration_sec > DURATION_THRESHOLD_SECONDS:
        hints.append("consider parallelizing work or using more specific subtasks")

    if hints:
        if len(hints) == 1:
            note += f" {hints[0].capitalize()}."
        elif len(hints) == 2:
            note += f" {hints[0].capitalize()} and {hints[1]}."
        else:
            note += f" {hints[0].capitalize()}, {hints[1]}, and {hints[2]}."

    return note


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    result: str = "",
    **_kw: Any,
) -> None:
    """Inspect kanban_complete calls and append flags when thresholds are crossed."""
    if tool_name != "kanban_complete":
        return

    # Only flag successful completions
    try:
        result_json = json.loads(result) if result else {}
    except Exception:
        return
    if not result_json.get("ok"):
        return

    task_id = (args or {}).get("task_id") if args else None
    if not task_id:
        return

    try:
        import hermes_cli.kanban_db as kb

        conn = kb.connect()
        try:
            run = kb.latest_run(conn, task_id)
            if not run or not run.ended_at:
                return

            total_runs = conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
            ).fetchone()[0]
            retries = max(0, total_runs - 1)

            input_tokens, output_tokens = _get_session_tokens()
            total_tokens = input_tokens + output_tokens

            duration_sec = run.ended_at - run.started_at

            note = _build_note(total_tokens, retries, duration_sec)
            if note:
                kb.add_comment(conn, task_id, author="opportunistic-flagging", body=note)
                logger.info(
                    "opportunistic-flagging: flagged %s (%s tokens, %s retries, %ss)",
                    task_id,
                    total_tokens,
                    retries,
                    duration_sec,
                )
        finally:
            conn.close()
    except Exception:
        logger.debug("opportunistic-flagging: failed to assess task", exc_info=True)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
