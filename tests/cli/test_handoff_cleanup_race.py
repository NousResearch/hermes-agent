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

import pytest


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


@pytest.mark.parametrize("handed_off", [False, True])
@pytest.mark.parametrize("delete_on_exit", [False, True])
def test_interactive_shutdown_preserves_gateway_owned_session(
    tmp_path, monkeypatch, handed_off, delete_on_exit,
):
    """Transferred transcripts survive TUI exit; ordinary close/delete still works."""
    import cli as cli_mod
    from hermes_state import SessionDB

    session_id = "interactive-handoff"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_mod, "_handed_off_session_ids", set())
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id=session_id, source="cli")
        db.append_message(session_id, "user", "original request")
        if handed_off:
            assert db.request_handoff(session_id, "telegram")
            assert db.claim_handoff(session_id)
            db.record_gateway_session_peer(
                session_id, source="telegram", session_key="agent:main:telegram:dm:test",
                chat_id="test", chat_type="dm",
            )
            db.append_message(session_id, "assistant", "gateway now owns this session")
            db.complete_handoff(session_id)
        before = db.get_messages(session_id)
        instance = MagicMock()
        instance.agent.session_id = session_id
        instance.session_id = session_id
        instance._session_db = db
        instance._HANDOFF_PENDING_TIMEOUT = 1
        if handed_off:
            assert cli_mod.HermesCLI._handoff_wait(instance, "telegram", "Transferred chat") is False
        instance._agent_running = True
        instance._voice_recorder = None
        instance._delete_session_on_exit = delete_on_exit
        instance._persist_active_session_before_close.side_effect = lambda: db.append_message(
            session_id, "user", "CLI close snapshot",
        )
        with (
            patch.object(cli_mod, "_run_cleanup") as cleanup,
            patch.object(cli_mod, "_invoke_interrupted_session_end") as end_hook,
            patch.object(cli_mod, "request_hard_interrupt"),
            patch("tools.voice_mode.cleanup_temp_recordings"),
        ):
            cli_mod.HermesCLI._tui_shutdown(instance)

        cleanup.assert_called_once_with()
        instance._release_active_session.assert_called_once_with()
        if handed_off:
            instance._persist_active_session_before_close.assert_not_called()
            instance._discard_session_if_empty.assert_not_called()
            end_hook.assert_not_called()
            assert db.get_session(session_id)["ended_at"] is None
            assert db.get_messages(session_id) == before
        else:
            instance._persist_active_session_before_close.assert_called_once_with()
            end_hook.assert_called_once()
            if delete_on_exit:
                assert db.get_session(session_id) is None
            else:
                assert db.get_session(session_id)["end_reason"] == "cli_close"
                instance._discard_session_if_empty.assert_called_once_with(session_id)
    finally:
        db.close()


@pytest.mark.parametrize("handed_off", [False, True])
def test_cleanup_releases_memory_without_ending_transferred_session(monkeypatch, handed_off):
    """Handoff releases local providers without extracting a stale session-end snapshot."""
    import cli as cli_mod

    from agent.memory_manager import MemoryManager
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "memory-handoff"
    agent._session_messages = [{"role": "user", "content": "CLI snapshot"}]
    agent._memory_manager = MemoryManager()
    provider = MagicMock()
    agent._memory_manager._providers = [provider]
    agent.context_compressor = MagicMock()
    monkeypatch.setattr(cli_mod, "_handed_off_session_ids", {agent.session_id} if handed_off else set())
    monkeypatch.setattr(cli_mod, "_active_agent_ref", agent)
    monkeypatch.setattr(cli_mod, "_cleanup_done", False)
    monkeypatch.setattr(cli_mod, "_cleanup_in_progress", False)
    monkeypatch.setattr(cli_mod, "_single_query_finalize_attempted_session_ids", set())
    resource_cleanup = MagicMock()
    monkeypatch.setattr(cli_mod, "_CLEANUP_STEPS", (("_stop_cli_wake_word", Exception),))
    monkeypatch.setattr(cli_mod, "_stop_cli_wake_word", resource_cleanup)
    monkeypatch.setattr(cli_mod, "_arm_exit_watchdog", MagicMock())
    monkeypatch.setattr(cli_mod, "_reset_terminal_input_modes_on_exit", MagicMock())
    with patch.object(cli_mod, "_notify_session_finalize") as finalize:
        cli_mod._run_cleanup()
    resource_cleanup.assert_called_once_with()
    if handed_off:
        finalize.assert_not_called()
        provider.on_session_end.assert_not_called()
        agent.context_compressor.on_session_end.assert_not_called()
    else:
        finalize.assert_called_once()
        provider.on_session_end.assert_called_once_with(agent._session_messages)
        agent.context_compressor.on_session_end.assert_called_once_with(agent.session_id, agent._session_messages)
    provider.shutdown.assert_called_once_with()
    assert agent._memory_provider_shutdown is True
