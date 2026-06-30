from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from hermes_cli.plugins import VALID_HOOKS, PluginManager
from cli import HermesCLI


def test_session_hooks_in_valid_hooks():
    """Verify session lifecycle hooks are registered as valid hooks."""
    assert "on_session_finalize" in VALID_HOOKS
    assert "on_session_reset" in VALID_HOOKS
    assert "on_session_resume" in VALID_HOOKS


@patch("hermes_cli.plugins.invoke_hook")
def test_session_finalize_on_reset(mock_invoke_hook):
    """Verify on_session_finalize fires when /new or /reset is used."""
    cli = HermesCLI()
    cli.agent = MagicMock()
    cli.agent.session_id = "test-session-id"

    # Simulate /new command which triggers on_session_finalize for the old session
    cli.new_session(silent=True)

    # Check if on_session_finalize was called for the old session
    assert any(
        c.args == ("on_session_finalize",)
        and c.kwargs["session_id"] == "test-session-id"
        and c.kwargs["platform"] == "cli"
        for c in mock_invoke_hook.call_args_list
    )
    # Check if on_session_reset was called for the new session
    assert any(
        c.args == ("on_session_reset",)
        and c.kwargs["session_id"] == cli.session_id
        and c.kwargs["platform"] == "cli"
        for c in mock_invoke_hook.call_args_list
    )


@patch("hermes_cli.plugins.invoke_hook")
def test_session_finalize_on_cleanup(mock_invoke_hook):
    """Verify on_session_finalize fires during CLI exit cleanup."""
    import cli as cli_mod

    mock_agent = MagicMock()
    mock_agent.session_id = "cleanup-session-id"
    cli_mod._active_agent_ref = mock_agent
    cli_mod._cleanup_done = False

    cli_mod._run_cleanup()

    assert any(
        c.args == ("on_session_finalize",)
        and c.kwargs["session_id"] == "cleanup-session-id"
        and c.kwargs["platform"] == "cli"
        and c.kwargs["reason"] == "shutdown"
        for c in mock_invoke_hook.call_args_list
    )


@patch("hermes_cli.plugins.invoke_hook")
def test_interrupted_session_end_helper_emits_observer_shape(mock_invoke_hook):
    """Verify quiet single-query interruption emits a correlated session end."""
    import cli as cli_mod

    mock_agent = MagicMock()
    mock_agent.session_id = "agent-session-id"
    mock_agent.model = "test-model"
    mock_agent.platform = "cli"
    mock_agent._current_task_id = "task-1"
    mock_agent._current_turn_id = "turn-1"
    mock_agent._current_api_request_id = "api-1"
    cli = SimpleNamespace(agent=mock_agent, session_id="cli-session-id")

    cli_mod._emit_interrupted_session_end(cli, reason="keyboard_interrupt")

    mock_agent.interrupt.assert_called_once_with("keyboard interrupt")
    assert cli.session_id == "agent-session-id"
    mock_invoke_hook.assert_called_once()
    call = mock_invoke_hook.call_args
    assert call.args == ("on_session_end",)
    assert call.kwargs["session_id"] == "agent-session-id"
    assert call.kwargs["task_id"] == "task-1"
    assert call.kwargs["turn_id"] == "turn-1"
    assert call.kwargs["api_request_id"] == "api-1"
    assert call.kwargs["completed"] is False
    assert call.kwargs["interrupted"] is True
    assert call.kwargs["reason"] == "keyboard_interrupt"


@patch("hermes_cli.plugins.invoke_hook")
def test_hook_errors_are_caught(mock_invoke_hook):
    """Verify hook exceptions are caught and don't crash the agent."""
    mgr = PluginManager()

    # Register a hook that raises
    def bad_callback(**kwargs):
        raise Exception("Hook failed")

    mgr._hooks["on_session_finalize"] = [bad_callback]

    # This should not raise
    results = mgr.invoke_hook("on_session_finalize", session_id="test", platform="cli")
    assert results == []
