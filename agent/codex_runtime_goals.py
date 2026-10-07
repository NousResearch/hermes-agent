"""Native Goal orchestration at the Codex runtime boundary, not a second agent loop."""
from __future__ import annotations


def run_app_server_work(agent, user_message, *, messages, **wire_options):
    from hermes_cli.config import load_config
    from hermes_cli.goals import load_goal
    from agent.codex_runtime import _persist_projected_messages, _record_codex_app_server_usage, _store_codex_thread_id
    cfg = load_config() or {}
    options = cfg.get("agent") or {}
    total = options.get("codex_turn_timeout", 600)
    idle = options.get("codex_idle_timeout", 1800)
    if (isinstance(total, bool) or isinstance(idle, bool) or not isinstance(total, (int, float))
            or not isinstance(idle, (int, float)) or total < 0 or idle <= 0):
        raise ValueError("agent.codex_turn_timeout must be >= 0; agent.codex_idle_timeout must be > 0")
    wire_options.update(turn_timeout=total, idle_timeout=idle)
    sid = getattr(agent, "session_id", None)
    state = load_goal(sid) if sid else None
    if state is None or state.runtime != "codex" or state.status != "active":
        return agent._codex_session.run_turn(user_input=user_message, **wire_options)
    from agent.transports.codex_app_server_goals import run_native_goal

    def commit_turn(turn, continuing):
        if not _persist_projected_messages(agent, turn, messages):
            raise RuntimeError("Native Goal turn was not durably mirrored; pausing instead of continuing")
        _store_codex_thread_id(agent, turn.thread_id)
        turn.projected_messages = []  # Already appended/flushed; aggregate must not insert them twice.
        if continuing:
            turn.usage_result = _record_codex_app_server_usage(agent, turn, messages=messages)
            turn.usage_recorded = True

    try:
        return run_native_goal(
            agent._codex_session, user_message, session_id=sid, state=state, on_turn=commit_turn,
            interrupt_requested=lambda: bool(getattr(agent, "_interrupt_requested", False)), **wire_options,
        )
    except Exception:
        from hermes_cli.codex_goals import pause_native_goal
        pause_native_goal(sid, state.goal_id, "Native Goal setup/transport failed; no automatic replay")
        raise


def native_resume_thread(agent):
    """A resumed native Goal stays on its budget-bearing thread even after /stop."""
    from hermes_cli.goals import load_goal
    sid = getattr(agent, "session_id", None)
    state = load_goal(sid) if sid else None
    if state is not None and state.runtime == "codex" and state.status == "active":
        return (state.native_goal or {}).get("threadId")
    return None
