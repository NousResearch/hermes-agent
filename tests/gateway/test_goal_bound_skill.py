"""Bound skills survive non-generating goal commands before the first turn (#130316)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_constants import get_hermes_home


@pytest.fixture
def bound_skill_runner():
    home = get_hermes_home()
    skill = home / "skills" / "incident-check"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text(
        "---\nname: incident-check\ndescription: Check reported incidents\n---\n"
        "BOUND_SKILL_WITNESS: inspect the incident history before answering.\n",
        encoding="utf-8",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(home / "sessions", runner.config)
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._set_session_env = lambda context: {}
    runner._clear_session_env = lambda tokens: None
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._delivery_adapter_for = lambda source: None
    runner._bind_adapter_run_generation = lambda *args: None
    runner._clear_goal_pending_continuations = lambda *args: None

    async def in_context(fn, *args):
        return fn(*args)

    async def hygiene(event, source, entry, key, history, *args):
        return history

    async def inbound_text(*, event, **kwargs):
        return event.text

    runner._run_in_executor_with_context = in_context
    runner._hmwa_run_session_hygiene = hygiene
    runner._prepare_profile_scoped_inbound_message_text = inbound_text
    source = SessionSource(platform=Platform.DISCORD, chat_id="incidents", user_id="operator")
    key = runner.session_store._generate_session_key(source)
    queued = []
    runner._adapter_and_key_for = lambda event: (object(), key)
    runner._enqueue_fifo = lambda key, event, adapter: queued.append(event)
    try:
        yield runner, source, key, queued
    finally:
        runner.session_store.close_all_db_handles()


async def _prepare(runner, source, key, event):
    entry = await runner.async_session_store.get_or_create_session(source)
    assert entry.updated_at != entry.created_at
    prepared, _ = await runner._hmwa_prepare_turn(event, source, entry, key, key, 1)
    assert isinstance(prepared, runner._PreparedTurn)
    return entry, prepared


def _record_turn(runner, entry, prepared):
    runner.session_store.append_to_transcript(
        entry.session_id, {"role": "user", "content": prepared.persist_user_message},
    )
    runner.session_store.append_to_transcript(
        entry.session_id, {"role": "assistant", "content": "The incident is resolved."},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_goal_kickoff_loads_bound_skill_once(bound_skill_runner, reset):
    runner, source, key, queued = bound_skill_runner
    if reset:
        runner.session_store.get_or_create_session(source)
        runner.session_store.reset_session(key)
    command = MessageEvent(text="/goal summarize incidents", source=source, auto_skill="incident-check")
    await runner._handle_goal_command(command)
    assert len(queued) == 1
    entry, prepared = await _prepare(runner, source, key, queued.pop())
    assert "BOUND_SKILL_WITNESS" in prepared.message_text
    _record_turn(runner, entry, prepared)

    await runner._handle_goal_command(
        MessageEvent(text="/goal resume", source=source, auto_skill="incident-check"),
    )
    continuation = queued.pop()
    assert continuation.auto_skill is None
    _, continued = await _prepare(runner, source, key, continuation)
    assert "BOUND_SKILL_WITNESS" not in continued.message_text
    assert sum("BOUND_SKILL_WITNESS" in str(m.get("content")) for m in continued.history) == 1


@pytest.mark.asyncio
async def test_goal_status_does_not_spend_first_turn_skill_loading(bound_skill_runner):
    runner, source, key, queued = bound_skill_runner
    await runner._handle_goal_command(MessageEvent(text="/goal status", source=source))
    assert queued == []
    event = MessageEvent(text="summarize incidents", source=source, auto_skill="incident-check")
    entry, prepared = await _prepare(runner, source, key, event)
    assert "BOUND_SKILL_WITNESS" in prepared.message_text
    assert prepared.title_user_message == "summarize incidents"
    _record_turn(runner, entry, prepared)

    later = MessageEvent(text="what changed?", source=source, auto_skill="incident-check")
    _, continued = await _prepare(runner, source, key, later)
    assert "BOUND_SKILL_WITNESS" not in continued.message_text

    # A lossy hygiene result is not proof of an unused session.
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    trimmed = MessageEvent(text="check once more", source=source, auto_skill="incident-check")
    _, prepared = await _prepare(runner, source, key, trimmed)
    assert "BOUND_SKILL_WITNESS" not in prepared.message_text

    # An unavailable transcript must fail closed before any skill is injected.
    from gateway.session_transcript import TranscriptReadError

    runner.async_session_store.load_transcript = AsyncMock(side_effect=TranscriptReadError(entry.session_id))
    runner._hmwa_auto_load_skills = Mock(wraps=runner._hmwa_auto_load_skills)
    unreadable = MessageEvent(text="check again", source=source, auto_skill="incident-check")
    reply, _ = await runner._hmwa_prepare_turn(unreadable, source, entry, key, key, 1)
    assert isinstance(reply, str)
    runner._hmwa_auto_load_skills.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_first_contact_notes_only_for_human_turns(bound_skill_runner, monkeypatch, internal):
    runner, source, key, _ = bound_skill_runner
    source.chat_type = "dm"
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    runner._hmwa_first_contact_notes = GatewayRunner._hmwa_first_contact_notes.__get__(runner)
    runner._deliver_platform_notice = AsyncMock()
    await runner._handle_goal_command(MessageEvent(text="/goal status", source=source))
    event = MessageEvent(
        text="summarize incidents", source=source, internal=internal,
        auto_skill=None if internal else "incident-check",
    )

    _, prepared = await _prepare(runner, source, key, event)

    # A plugin can open an empty session without inviting human onboarding. The
    # first human turn still gets both its bound skill and the first-contact notes.
    assert ("BOUND_SKILL_WITNESS" in prepared.message_text) is (not internal)
    assert bool(runner._consume_pending_turn_sidecar_notes(key)) is (not internal)
    assert runner._deliver_platform_notice.await_count == int(not internal)
