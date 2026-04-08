import pytest
from unittest.mock import MagicMock, patch
from hermes_cli.plugins import VALID_HOOKS, PluginManager
import os
import shutil
import tempfile
from cli import HermesCLI


def test_session_hooks_in_valid_hooks():
    """Verify on_session_finalize and on_session_reset are registered as valid hooks."""
    assert "on_session_finalize" in VALID_HOOKS
    assert "on_session_reset" in VALID_HOOKS


@patch("hermes_cli.plugins.invoke_hook")
def test_session_finalize_on_reset(mock_invoke_hook):
    """Verify on_session_finalize fires when /new or /reset is used."""
    cli = HermesCLI()
    cli.agent = MagicMock()
    cli.agent.session_id = "test-session-id"

    # Simulate /new command which triggers on_session_finalize for the old session
    cli.new_session(silent=True)

    # Check if on_session_finalize was called for the old session
    mock_invoke_hook.assert_any_call(
        "on_session_finalize", session_id="test-session-id", platform="cli"
    )
    # Check if on_session_reset was called for the new session
    mock_invoke_hook.assert_any_call(
        "on_session_reset", session_id=cli.session_id, platform="cli"
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

    mock_invoke_hook.assert_any_call(
        "on_session_finalize", session_id="cleanup-session-id", platform="cli"
    )


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
