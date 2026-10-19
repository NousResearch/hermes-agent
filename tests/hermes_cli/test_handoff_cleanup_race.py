"""Regression tests for #88234 — CLI cleanup must NOT finalize a session that
was handed off to the gateway.

The handoff flow re-binds the CLI session_id to a gateway session_key via
``switch_session``, which reopens the session row.  The CLI then exits and
``_run_cleanup`` fires ``_notify_session_finalize`` on that same session_id.
The resulting ``end_session`` call sets ``end_reason`` on a row the gateway
just reopened and is actively writing to — the handoff leg vanishes from
session history and ``session_search`` cannot find it.

The fix adds a module-level ``_handed_off_session_ids`` set (mirroring the
existing ``_single_query_finalize_attempted_session_ids`` pattern).
``_handle_handoff_command`` registers the session_id when the handoff
completes, and ``_should_emit_cleanup_session_finalize`` /
``_emit_interrupted_session_end`` check the set before firing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _reset_cli_globals(cli_mod):
    """Reset the module-level globals the cleanup path checks."""
    cli_mod._cleanup_done = False
    cli_mod._cleanup_in_progress = False
    cli_mod._single_query_finalize_attempted_session_ids.clear()
    cli_mod._handed_off_session_ids.clear()
    cli_mod._active_agent_ref = None


def test_handed_off_session_skips_cleanup_finalize():
    """_should_emit_cleanup_session_finalize returns False for a handed-off session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)
    cli_mod._handed_off_session_ids.add("handoff-session-123")

    assert cli_mod._should_emit_cleanup_session_finalize("handoff-session-123") is False


def test_normal_session_still_finalizes():
    """_should_emit_cleanup_session_finalize returns True for a non-handed-off session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)
    cli_mod._single_query_finalize_attempted_session_ids.add("other-session")

    assert cli_mod._should_emit_cleanup_session_finalize("normal-session") is True
    assert cli_mod._should_emit_cleanup_session_finalize("other-session") is False


def test_interrupted_session_end_skipped_for_handed_off():
    """_emit_interrupted_session_end returns early for a handed-off session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)
    cli_mod._handed_off_session_ids.add("handoff-session-456")

    agent = MagicMock()
    agent.session_id = "handoff-session-456"
    cli_mod._active_agent_ref = agent

    cli_mock = MagicMock()
    cli_mock.agent = agent
    cli_mock.session_id = "handoff-session-456"

    with patch("hermes_cli.lifecycle.invoke_hook") as mock_hook:
        cli_mod._emit_interrupted_session_end(cli_mock, reason="keyboard_interrupt")

    # on_session_end hook must NOT fire for a handed-off session
    mock_hook.assert_not_called()


def test_interrupted_session_end_fires_for_normal():
    """_emit_interrupted_session_end fires for a normal (non-handed-off) session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)

    agent = MagicMock()
    agent.session_id = "normal-session-789"
    agent._current_task_id = ""
    agent._current_turn_id = ""
    agent._current_api_request_id = ""
    agent.model = "test-model"
    agent.platform = "cli"
    cli_mod._active_agent_ref = agent

    cli_mock = MagicMock()
    cli_mock.agent = agent
    cli_mock.session_id = "normal-session-789"

    with patch("hermes_cli.lifecycle.invoke_hook") as mock_hook:
        cli_mod._emit_interrupted_session_end(cli_mock, reason="keyboard_interrupt")

    mock_hook.assert_called_once()


def test_cleanup_does_not_finalize_handed_off_session():
    """_run_cleanup must not call finalize_session for a handed-off session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)
    cli_mod._handed_off_session_ids.add("handoff-session-abc")

    agent = MagicMock()
    agent.session_id = "handoff-session-abc"
    agent._session_messages = []
    cli_mod._active_agent_ref = agent

    with (
        patch("hermes_cli.lifecycle.finalize_session") as mock_finalize,
        patch("hermes_cli.plugins.invoke_hook"),
    ):
        cli_mod._run_cleanup()

    mock_finalize.assert_not_called()


def test_cleanup_finalizes_normal_session():
    """_run_cleanup DOES call finalize_session for a normal session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)

    agent = MagicMock()
    agent.session_id = "normal-session-def"
    agent._session_messages = []
    cli_mod._active_agent_ref = agent

    with (
        patch("hermes_cli.lifecycle.finalize_session") as mock_finalize,
        patch("hermes_cli.plugins.invoke_hook"),
    ):
        cli_mod._run_cleanup()

    mock_finalize.assert_called_once()


def test_single_query_finalize_skipped_for_handed_off():
    """_notify_single_query_session_finalize must not fire for a handed-off session."""
    import cli as cli_mod

    _reset_cli_globals(cli_mod)
    cli_mod._handed_off_session_ids.add("handoff-session-single")

    agent = MagicMock()
    agent.session_id = "handoff-session-single"
    agent.platform = "cli"

    cli_mock = MagicMock()
    cli_mock.agent = agent
    cli_mock.session_id = "handoff-session-single"

    with patch("hermes_cli.lifecycle.finalize_session") as mock_finalize:
        cli_mod._notify_single_query_session_finalize(cli_mock)

    mock_finalize.assert_not_called()
