"""Regression tests for #15165 (CLI sibling site) — CLI exit cleanup must
forward the agent's conversation transcript to ``shutdown_memory_provider``
so memory providers' ``on_session_end`` hooks see the real messages.

Before the fix, ``_run_cleanup`` called
``shutdown_memory_provider(getattr(agent, 'conversation_history', None) or [])``.
``AIAgent`` has no ``conversation_history`` attribute — so the ``or []``
branch always fired and providers got an empty list on CLI exit. This
mirrors the gateway bug fixed in the same commit (gateway/run.py uses
``_session_messages``, which IS set on ``AIAgent``).

The fix reads ``_session_messages`` (same attribute the gateway path uses)
with an ``isinstance(..., list)`` guard so MagicMock-based agents in
other tests keep their existing no-arg behaviour.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_forwards_session_messages(mock_invoke_hook):
    """_run_cleanup forwards a populated ``_session_messages`` list."""
    import cli as cli_mod

    transcript = [
        {"role": "user", "content": "remember my dog is named Biscuit"},
        {"role": "assistant", "content": "Got it — Biscuit."},
    ]

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = transcript

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with(transcript)


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_empty_list_still_forwarded(mock_invoke_hook):
    """An agent that initialised but ran no turns has an empty list.
    Forwarding it (rather than falling through) matches the gateway-side
    behaviour and is explicit to providers."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = []

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with([])


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_non_list_attribute_falls_back_to_no_arg(mock_invoke_hook):
    """A MagicMock agent auto-synthesises ``_session_messages`` as a
    nested MagicMock. ``isinstance(mock, list)`` is False, so we fall
    back to the no-arg path rather than passing a garbage value to
    providers expecting ``List[Dict]``.  This keeps existing CLI test
    suites that use bare ``MagicMock()`` agents green."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    # No explicit _session_messages — MagicMock synthesises one on access.

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once_with()


@patch("hermes_cli.plugins.invoke_hook")
def test_cleanup_provider_exception_is_swallowed(mock_invoke_hook):
    """A raising ``shutdown_memory_provider`` must not crash CLI exit."""
    import cli as cli_mod

    agent = MagicMock()
    agent.session_id = "cli-session-id"
    agent._session_messages = [{"role": "user", "content": "x"}]
    agent.shutdown_memory_provider.side_effect = RuntimeError("boom")

    cli_mod._active_agent_ref = agent
    cli_mod._cleanup_done = False
    try:
        cli_mod._run_cleanup()  # must not raise
    finally:
        cli_mod._active_agent_ref = None
        cli_mod._cleanup_done = False

    agent.shutdown_memory_provider.assert_called_once()
