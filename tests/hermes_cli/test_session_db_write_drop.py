"""Regression contracts for SessionDB-unavailable goal writes."""

import logging
from unittest.mock import patch


def test_save_goal_warns_and_returns_when_db_unavailable():
    from hermes_cli.goals import GoalState, save_goal

    state = GoalState(goal="write-drop-contract")

    with (
        patch(
            "hermes_cli.goals._get_session_db",
            return_value=None,
        ) as get_db,
        patch(
            "hermes_cli.goals._warn_dropped_write"
        ) as warn,
        patch.object(
            GoalState,
            "to_json",
            autospec=True,
        ) as serialize,
    ):
        save_goal(
            "write-drop-session",
            state,
        )

    get_db.assert_called_once_with()

    warn.assert_called_once_with(
        "GoalManager",
        "goal",
        "write-drop-session",
    )

    serialize.assert_not_called()


def test_warn_dropped_write_emits_warning_level(caplog):
    from hermes_cli.goals import _warn_dropped_write

    caplog.clear()

    with caplog.at_level(
        logging.WARNING,
        logger="hermes_cli.goals",
    ):
        _warn_dropped_write(
            "GoalManager",
            "goal",
            "write-drop-warning-session",
        )

    warnings = [
        record
        for record in caplog.records
        if (
            record.name == "hermes_cli.goals"
            and record.levelno == logging.WARNING
        )
    ]

    assert warnings
