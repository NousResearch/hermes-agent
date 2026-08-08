"""Kanban goal-loop driver for the quiet single-query CLI path.

Extracted verbatim from cli.py (wave 1 godfile extraction, shard s5 cluster
c25). ``_run_kanban_goal_loop_q`` drives a dispatcher-spawned kanban
goal_mode worker through ``hermes_cli.goals.run_kanban_goal_loop``.
"""

import logging

logger = logging.getLogger(__name__)


def _run_kanban_goal_loop_q(cli: "HermesCLI", first_response: str) -> None:
    """Drive a kanban goal_mode worker through the Ralph-style goal loop.

    Called from the quiet single-query path AFTER the worker's first turn,
    only when ``HERMES_KANBAN_GOAL_MODE`` is set (dispatcher-spawned
    goal_mode card). Wires the worker's ``run_conversation`` and the kanban
    DB into ``goals.run_kanban_goal_loop``. All errors are swallowed by the
    caller — a broken goal loop must never wedge a worker, the dispatcher's
    claim TTL / crash detection is the backstop.
    """
    import os as _os

    task_id = (_os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return

    from hermes_cli import kanban_db as _kb
    from hermes_cli.goals import run_kanban_goal_loop as _run_loop, DEFAULT_MAX_TURNS as _DEF_TURNS

    # Resolve goal text from the card (title + body = the acceptance
    # criteria the judge evaluates against).
    conn = _kb.connect()
    try:
        task = _kb.get_task(conn, task_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if task is None:
        return

    goal_parts = [task.title or ""]
    if task.body:
        goal_parts.append(task.body)
    goal_text = "\n\n".join(p for p in goal_parts if p).strip()
    if not goal_text:
        return

    max_turns = task.goal_max_turns or _DEF_TURNS

    def _run_turn(prompt: str) -> str:
        result = cli.agent.run_conversation(
            user_message=prompt,
            conversation_history=cli.conversation_history,
        )
        # Keep session_id in sync if mid-run compression rotated it.
        if (
            getattr(cli.agent, "session_id", None)
            and cli.agent.session_id != cli.session_id
        ):
            cli.session_id = cli.agent.session_id
        resp = result.get("final_response", "") if isinstance(result, dict) else str(result)
        if resp:
            print(resp)
        return resp or ""

    def _task_status() -> "str | None":
        c = _kb.connect()
        try:
            t = _kb.get_task(c, task_id)
            return t.status if t is not None else None
        finally:
            try:
                c.close()
            except Exception:
                pass

    def _block(reason: str) -> None:
        c = _kb.connect()
        try:
            _kb.block_task(c, task_id, reason=reason)
        finally:
            try:
                c.close()
            except Exception:
                pass

    _run_loop(
        task_id=task_id,
        goal_text=goal_text,
        run_turn=_run_turn,
        task_status_fn=_task_status,
        block_fn=_block,
        max_turns=max_turns,
        first_response=first_response or "",
        log=lambda m: logger.info("%s", m),
    )
