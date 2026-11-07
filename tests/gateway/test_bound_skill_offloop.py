"""Bound-skill preparation must leave the gateway responsive (#140256)."""

import asyncio
from datetime import datetime
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_context import get_session_env
from hermes_constants import (
    get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
)


@pytest.fixture
def prepared_lane(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._hmwa_open_session = AsyncMock(return_value=(False, True))
    runner._pinned_session_context_prompt = lambda *args, **kwargs: ""
    runner._hmwa_acquire_turn_lease = AsyncMock()
    runner._mark_durable_active_turn = AsyncMock()
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, load_transcript=AsyncMock(return_value=[]),
    )
    runner._hmwa_run_session_hygiene = AsyncMock(return_value=[])
    runner._hmwa_first_contact_notes = AsyncMock()
    runner._voice_channel_sidecar_note = lambda *args: None
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(
        side_effect=lambda **kwargs: kwargs["event"].text,
    )
    runner._hmwa_apply_message_timestamp = lambda event, text: (text, text, None)
    runner._bind_adapter_run_generation = lambda *args: None
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel", user_id="user")
    event = MessageEvent(text="original request", source=source, auto_skill="fixture")
    entry = SessionEntry(
        session_id="bound-skill", session_key="resolved-key",
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1),
    )
    quick_key = "admission-key"
    generation = runner._begin_session_run_generation(quick_key)
    yield runner, event, entry, quick_key, generation
    runner._shutdown_executor()


@pytest.mark.asyncio
async def test_bound_skill_load_keeps_loop_responsive_and_profile_context(
    prepared_lane, tmp_path, monkeypatch,
):
    runner, event, entry, key, generation = prepared_lane
    loop = asyncio.get_running_loop()
    release = threading.Event()
    observed = []

    def load(name, **kwargs):
        loop.call_soon_threadsafe(release.set)
        responsive = release.wait(2)
        observed.append((responsive, get_hermes_home(), get_session_env("HERMES_SESSION_KEY")))
        return ({"name": name, "content": "skill instructions"}, tmp_path, name)

    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", lambda *args: "skill instructions")
    token = set_hermes_home_override(tmp_path)
    try:
        prepared, env_tokens = await runner._hmwa_prepare_turn(
            event, event.source, entry, entry.session_key, key, generation,
        )
        runner._clear_session_env(env_tokens)
    finally:
        reset_hermes_home_override(token)
    assert observed[0][0], "bound-skill loading blocked an unrelated loop callback"
    assert observed[0][1] == tmp_path
    assert observed[0][2], "session context must propagate into skill loading"
    assert prepared.message_text == "skill instructions\n\noriginal request"
    assert prepared.title_user_message == "original request"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_obsolete_skill_worker_cannot_mutate_event_or_continue_preparation(
    prepared_lane, tmp_path, monkeypatch, cancel,
):
    runner, event, entry, key, generation = prepared_lane
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = asyncio.Event()
    restored = []

    def load(name, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        if not release.wait(5):
            raise RuntimeError("fixture release timed out")
        return ({"name": name}, tmp_path, name)

    def build(*args):
        loop.call_soon_threadsafe(finished.set)
        return "late instructions"

    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", build)

    async def prepare():
        before = get_session_env("HERMES_SESSION_KEY")
        try:
            return await runner._hmwa_prepare_turn(
                event, event.source, entry, entry.session_key, key, generation,
            )
        finally:
            restored.append(get_session_env("HERMES_SESSION_KEY") == before)

    task = asyncio.create_task(prepare())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            runner._begin_session_run_generation(key)
        release.set()
        await asyncio.wait_for(finished.wait(), 5)
        if not cancel:
            result, _ = await asyncio.wait_for(task, 5)
            assert result is None
        # Drain the worker after its last callback, not just the callback itself.
        await asyncio.to_thread(runner._executor.shutdown, wait=True)
    finally:
        release.set()
        if not task.done():
            task.cancel()
    assert restored == [True]
    assert event.text == "original request"
    runner._hmwa_acquire_turn_lease.assert_not_called()
    runner._mark_durable_active_turn.assert_not_called()
    runner._async_session_store.load_transcript.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "empty", "error", "ordered", "ongoing"])
async def test_bound_skill_fallback_and_first_turn_contract(prepared_lane, tmp_path, monkeypatch, mode):
    runner, event, entry, key, generation = prepared_lane
    calls = []
    event.auto_skill = ["first", "second"]
    if mode == "ongoing":
        runner._hmwa_open_session.return_value = (False, False)

    def load(name, **kwargs):
        calls.append(name)
        if mode == "error":
            raise OSError("synthetic read failure")
        if mode == "missing":
            return None
        return ({"name": name}, tmp_path, name)

    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", lambda skill, *args: (
        "" if mode == "empty" else skill["name"]
    ))
    prepared, tokens = await runner._hmwa_prepare_turn(
        event, event.source, entry, entry.session_key, key, generation,
    )
    runner._clear_session_env(tokens)
    expected = "first\n\nsecond\n\noriginal request" if mode == "ordered" else "original request"
    assert prepared.message_text == expected
    assert prepared.title_user_message == "original request"
    if mode == "ongoing":
        assert calls == []


@pytest.mark.asyncio
async def test_real_bound_skill_is_loaded_from_the_routed_home(prepared_lane, tmp_path):
    runner, event, entry, key, generation = prepared_lane
    home = tmp_path / "routed"
    skill = home / "skills" / "fixture"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: fixture\ndescription: Synthetic scoped skill.\n---\n\nRouted fixture instructions.\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(home)
    try:
        prepared, tokens = await runner._hmwa_prepare_turn(
            event, event.source, entry, entry.session_key, key, generation,
        )
        runner._clear_session_env(tokens)
    finally:
        reset_hermes_home_override(token)
    assert "Routed fixture instructions." in prepared.message_text
    assert prepared.message_text.endswith("original request")
    assert prepared.title_user_message == "original request"


@pytest.mark.asyncio
async def test_executor_refusal_restores_session_context(prepared_lane):
    runner, event, entry, key, generation = prepared_lane
    runner._executor_closing = True
    before = get_session_env("HERMES_SESSION_KEY")
    with pytest.raises(RuntimeError, match="executor unavailable"):
        await runner._hmwa_prepare_turn(event, event.source, entry, entry.session_key, key, generation)
    assert get_session_env("HERMES_SESSION_KEY") == before
    assert event.text == "original request"
    runner._hmwa_acquire_turn_lease.assert_not_called()
