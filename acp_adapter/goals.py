"""In-process ACP scheduling around the existing durable GoalManager.

The gateway path uses slash.exec and the owner's post-turn hook instead.
"""
from acp_adapter.config_options import command_body


def _manager(state):
    from hermes_cli.config import load_config
    from hermes_cli.goals import DEFAULT_MAX_TURNS, GoalManager, _get_session_db
    budget = (load_config().get('goals') or {}).get('max_turns', DEFAULT_MAX_TURNS)
    db = _get_session_db()
    target = db.get_compression_tip(state.session_id) if db is not None else state.session_id
    return GoalManager(target, default_max_turns=budget)


def run_command(state, arg):
    from hermes_cli.goal_command import dispatch_goal_command
    from hermes_cli.goals import last_user_message_content
    with state.goal_lock:
        result = dispatch_goal_command(_manager(state), command_body(arg), authorize_gate=lambda: None,
                                       last_user_message=last_user_message_content(state.history))
        with state.runtime_lock:
            if result.clear_pending or result.prompt:
                # Revoke even a prompt already detached by the drain. Publish this
                # before command success, atomically with respect to turn claims.
                state.goal_revision += 1
                state.pending_goal_prompt = None
            if result.prompt:
                state.pending_goal_prompt = result.prompt
        return result.output


def after_turn(state, response, *, interrupted):
    from hermes_cli.goals import count_active_delegations, gather_background_processes
    with state.goal_lock:
        mgr = _manager(state)
        if not mgr.is_active():
            return ''
        if interrupted:
            mgr.pause(reason='user-interrupted')
            with state.runtime_lock:
                state.pending_goal_prompt = None
            return mgr.status_line()
        with state.runtime_lock:
            if state.queued_prompts or state.pending_goal_prompt:
                return ''  # already-admitted human input takes priority
        if not response.strip():
            return ''
        decision = mgr.evaluate_after_turn(response, user_initiated=True,
            background_processes=gather_background_processes(owner_task_id=mgr.session_id),
            active_delegations=count_active_delegations(mgr.session_id))
        if state.cancel_event and state.cancel_event.is_set():
            mgr.pause(reason='user-interrupted')
            return mgr.status_line()
        if decision.get('should_continue'):
            with state.runtime_lock:
                state.pending_goal_prompt = decision['continuation_prompt']
        return decision.get('message') or ''
