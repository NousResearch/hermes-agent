"""Regression for #140258: bootstrap belongs to a persisted first human turn."""

import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.session_transcript import TranscriptReadError
from gateway.turn_lease import TurnLeaseTimeoutError
from hermes_constants import get_hermes_home


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    skill = get_hermes_home() / "skills" / "bootstrap-contract"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text(
        "---\nname: bootstrap-contract\ndescription: Session bootstrap contract\n---\n"
        "BOOTSTRAP-INSTRUCTIONS: preserve the channel's operating contract.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "offline")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner._session_db = None  # gateway-owned transcript writes
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._rehydrate_prompt_pins = AsyncMock()
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._hmwa_run_session_hygiene = AsyncMock(side_effect=lambda *args: args[4])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._hmwa_deliver_auto_reset_notice = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text,
    )
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, None)
    runner._delivery_adapter_for = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._clear_conversation_scope = Mock()
    runner._evict_cached_agent = Mock()
    source = SessionSource(platform=Platform.SLACK, chat_id="bootstrap", user_id="human")
    return runner, source


async def _prepare(runner, source, *, internal=False, synthetic=False):
    entry = runner.session_store.get_or_create_session(source, touch_activity=not internal)
    event = MessageEvent(text="Check the channel contract", source=source,
                         auto_skill="bootstrap-contract", internal=internal)
    if synthetic:
        event = runner._synthetic_prompt_event(source, "Continue the background goal")
    prepared, _tokens = await runner._hmwa_prepare_turn(
        event, source, entry, entry.session_key, entry.session_key, 1,
    )
    return event, entry, prepared


async def _persist(runner, event, entry, prepared, *, failed=False):
    await runner._hmwa_persist_turn_transcript(
        event=event, source=event.source, session_entry=entry, session_key=entry.session_key,
        prepared=prepared, agent_result={"agent_persisted": False, "failed": failed},
        agent_messages=[], response="Completed", agent_failed_early=failed,
        hidden_reasoning_incomplete=False, is_context_overflow_failure=False,
    )


def _starts(runner):
    return [call for call in runner.hooks.emit.await_args_list if call.args[0] == "session:start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["new", "manual-reset", "auto-reset"])
@pytest.mark.parametrize("boundary", [
    "history", "text", "lease", "internal", "synthetic", "agent-failure",
    "transcript-write", "buffered-write", "routing-write",
])
async def test_bootstrap_survives_abort_then_commits_once(bootstrap, monkeypatch, origin, boundary):
    runner, source = bootstrap
    store = runner.session_store
    entry = store.get_or_create_session(source)
    if origin == "manual-reset":
        entry = store.reset_session(entry.session_key)
    elif origin == "auto-reset":
        store.suspend_session(entry.session_key)
        entry = store.get_or_create_session(source)

    with monkeypatch.context() as fault:
        if boundary == "history":
            fault.setattr(store, "load_transcript", Mock(side_effect=TranscriptReadError(entry.session_id)))
        elif boundary == "text":
            fault.setattr(runner, "_prepare_profile_scoped_inbound_message_text", AsyncMock(return_value=None))
        elif boundary == "lease":
            error = TurnLeaseTimeoutError(entry.session_id, owner_key=entry.session_key, generation=1, wait_seconds=0)
            fault.setattr(runner, "_hmwa_acquire_turn_lease", AsyncMock(side_effect=error))
        if boundary == "lease":
            with pytest.raises(TurnLeaseTimeoutError):
                await _prepare(runner, source)
        else:
            event, entry, prepared = await _prepare(
                runner, source, internal=boundary == "internal", synthetic=boundary == "synthetic",
            )
            if boundary == "history":
                assert isinstance(prepared, str)
            elif boundary == "text":
                assert prepared is None
            else:
                assert isinstance(prepared, runner._PreparedTurn)
                if boundary == "transcript-write":
                    fault.setattr(store, "append_to_transcript", Mock(side_effect=OSError("write unavailable")))
                elif boundary == "buffered-write":
                    db = store._db_for_session_id(entry.session_id)
                    fault.setattr(db, "append_message", Mock(side_effect=sqlite3.OperationalError("database is locked")))
                elif boundary == "routing-write":
                    save = store._save_entry

                    def reject_completion(*args, **kwargs):
                        if kwargs.get("entry_data", {}).get("is_fresh_reset") is False:
                            raise OSError("routing write unavailable")
                        return save(*args, **kwargs)

                    fault.setattr(store, "_save_entry", reject_completion)
                if boundary in {"transcript-write", "routing-write"}:
                    with pytest.raises(OSError):
                        await _persist(runner, event, entry, prepared)
                else:
                    await _persist(runner, event, entry, prepared, failed=boundary == "agent-failure")

    assert entry.is_fresh_reset, "an uncommitted turn spent bootstrap"
    if boundary in {"history", "text", "lease", "internal", "synthetic"}:
        assert not _starts(runner), "a refused or synthetic turn emitted session:start"
    # Re-open the real routing store to prove the pending state survived persistence.
    reopened = SessionStore(store.sessions_dir, runner.config)
    assert reopened.get_or_create_session(source).is_fresh_reset
    runner.hooks.emit.reset_mock()
    event, entry, prepared = await _prepare(runner, source)
    assert "BOOTSTRAP-INSTRUCTIONS" in prepared.message_text
    assert prepared.title_user_message == "Check the channel contract"
    assert len(_starts(runner)) == 1
    await _persist(runner, event, entry, prepared)
    assert not entry.is_fresh_reset
    assert store.has_input_owner(entry.session_id, prepared.persistence_owner)
    restarted = SessionStore(store.sessions_dir, runner.config)
    assert not restarted.get_or_create_session(source).is_fresh_reset
    runner.session_store = restarted
    runner.hooks.emit.reset_mock()
    _event, _entry, later = await _prepare(runner, source)
    assert "BOOTSTRAP-INSTRUCTIONS" not in later.message_text
    assert any("BOOTSTRAP-INSTRUCTIONS" in str(row.get("content")) for row in later.history)
    assert not _starts(runner)
    if origin == "auto-reset":
        runner._clear_conversation_scope.assert_called_once()
        runner._evict_cached_agent.assert_called_once()


@pytest.mark.asyncio
async def test_stale_completion_cannot_consume_reset_bootstrap(bootstrap):
    runner, source = bootstrap
    event, old_entry, prepared = await _prepare(runner, source)
    replacement = runner.session_store.reset_session(old_entry.session_key)
    await _persist(runner, event, old_entry, prepared)
    assert replacement.is_fresh_reset
    reopened = SessionStore(runner.session_store.sessions_dir, runner.config)
    assert reopened.get_or_create_session(source).is_fresh_reset
    assert not runner.session_store.complete_session_bootstrap(old_entry.session_key, old_entry.session_id)


@pytest.mark.asyncio
async def test_lease_waiter_rechecks_committed_bootstrap(bootstrap):
    runner, source = bootstrap
    first, entry, prepared = await _prepare(runner, source)
    waiting, release = asyncio.Event(), asyncio.Event()

    async def acquire(*args):
        waiting.set()
        await release.wait()

    runner._hmwa_acquire_turn_lease = acquire
    waiter = asyncio.create_task(_prepare(runner, source))
    try:
        await asyncio.wait_for(waiting.wait(), 2)
        await _persist(runner, first, entry, prepared)
    finally:
        release.set()
        _event, _entry, following = await asyncio.wait_for(waiter, 2)
    assert "BOOTSTRAP-INSTRUCTIONS" not in following.message_text
    assert len(_starts(runner)) == 1


@pytest.mark.asyncio
async def test_api_client_bootstrap_uses_private_execution_binding(bootstrap):
    from gateway.session_api_turn import api_execution

    runner, source = bootstrap
    source.platform = Platform.API_SERVER
    token = api_execution.set({})  # the canonical API drain's trusted, process-local binding
    try:
        event, entry, prepared = await _prepare(runner, source, internal=True)
        assert "BOOTSTRAP-INSTRUCTIONS" in prepared.message_text
        assert len(_starts(runner)) == 1
        await _persist(runner, event, entry, prepared)
        assert not entry.is_fresh_reset
    finally:
        api_execution.reset(token)
