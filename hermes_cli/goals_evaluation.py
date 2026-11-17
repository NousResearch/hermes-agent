"""Conditional persistence for a goal's synchronous gates/judge evaluation."""


class SupersededGoal(Exception):
    """A newer user control owns the persisted goal."""


def save_evaluation(manager):
    from hermes_cli import goals
    from hermes_cli.goals_waits import state_token
    db = goals._get_session_db()
    if db is None:
        raise SupersededGoal()
    expected = manager._evaluation_token
    def write(conn):
        from hermes_cli.goals_waits import observed_token, read_state
        current = read_state(conn, manager.session_id)
        if (current is None or observed_token(current) != expected
                or getattr(current, '_control_incarnation', None) != manager._evaluation_incarnation):
            raise SupersededGoal()
        from hermes_cli.goals_waits import persist_state
        return persist_state(conn, manager.session_id, manager.state)
    incarnation = db._execute_write(write)
    manager.state._control_incarnation = incarnation
    manager._evaluation_incarnation = incarnation
    manager._evaluation_token = state_token(manager.state)


def evaluate(manager, last_response, **kwargs):
    from hermes_cli.goals import _decision
    manager._evaluation_token = manager._persisted_token
    manager._evaluation_incarnation = getattr(manager.state, '_control_incarnation', None)
    try:
        from hermes_cli.goals_workers import release_completion
        release_completion(manager)
        return manager._evaluate_after_turn(last_response, **kwargs)
    except SupersededGoal:
        manager.refresh()
        return _decision(manager.state.status if manager.state else None, False, None,
                         'superseded', 'goal changed during evaluation', '')
    finally:
        manager._evaluation_token = None
        manager._evaluation_incarnation = None
