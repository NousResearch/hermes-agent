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


def test_cli_close_persists_agent_session_messages_before_end_session():
    """CLI shutdown flushes live agent messages before closing the session."""
    import cli as cli_mod

    transcript = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "partial answer"},
    ]
    conversation_history = [{"role": "user", "content": "long task"}]

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = conversation_history
    cli.session_id = "old-session"
    agent = MagicMock()
    agent.session_id = "live-session"
    agent._session_messages = transcript
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_called_once_with(transcript)
    assert cli.session_id == "live-session"


def test_cli_close_persist_falls_back_to_conversation_history():
    """Bare MagicMock agents do not provide a real _session_messages list."""
    import cli as cli_mod

    conversation_history = [{"role": "user", "content": "saved from cli"}]
    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = conversation_history
    cli.session_id = "session-id"
    agent = MagicMock()
    agent.session_id = "session-id"
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_called_once_with(conversation_history)


def test_cli_close_persist_skips_empty_transcripts():
    """Do not create empty session writes for idle CLI startup/shutdown."""
    import cli as cli_mod

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = []
    cli.session_id = "session-id"
    agent = MagicMock()
    agent.session_id = "session-id"
    agent._session_messages = []
    cli.agent = agent

    cli._persist_active_session_before_close()

    agent._persist_session.assert_not_called()


def test_cli_close_persist_real_db_survives_history_alias(tmp_path, monkeypatch):
    """CLI close safety-net must persist even when history aliases messages.

    In the real CLI, ``conversation_history`` and ``agent._session_messages`` can
    point at the same live list during interrupted shutdown.  Passing that list
    as ``conversation_history`` makes ``_flush_messages_to_session_db`` treat
    every message as already durable and write zero rows.  The close safety-net
    should use marker-based dedup instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    import cli as cli_mod
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cli-close-alias"
    db.create_session(session_id=session_id, source="cli")

    transcript = [
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "partial answer"},
    ]

    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent._session_messages = transcript
    agent._last_flushed_db_idx = 0
    agent._flushed_db_message_ids = set()
    agent._flushed_db_message_session_id = None
    agent._persist_disabled = False
    agent._cached_system_prompt = None
    agent._session_init_model_config = None
    agent._parent_session_id = None
    agent._session_json_enabled = False

    cli = object.__new__(cli_mod.HermesCLI)
    cli.conversation_history = transcript
    cli.session_id = "old-session"
    cli.agent = agent

    assert db.get_messages_as_conversation(session_id) == []

    cli._persist_active_session_before_close()

    stored = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in stored] == ["long task", "partial answer"]
    assert cli.session_id == session_id
