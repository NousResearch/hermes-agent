from hermes_cli.config import DEFAULT_CONFIG


def test_creator_session_wake_is_opt_in_by_default():
    assert DEFAULT_CONFIG["kanban"]["wake_creator_session"] is False
