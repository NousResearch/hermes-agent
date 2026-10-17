"""Bound skills must preserve loop liveness, profile scope and turn ownership (#140256)."""

import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.session_context import get_session_env
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def _turn(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", dict)
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
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
    runner._delivery_adapter_for = lambda source: None

    async def inbound_text(**kwargs):
        return kwargs["event"].text

    runner._prepare_profile_scoped_inbound_message_text = inbound_text
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel", user_id="user", profile="owner")
    entry = SessionEntry(session_id="skill-session", session_key="agent:owner:discord:channel",
                         created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
    event = MessageEvent(text="original request", source=source, auto_skill=["alpha", "beta"])
    generation = runner._begin_session_run_generation(entry.session_key)
    return runner, event, entry, generation


def _skills(home):
    for name in ("alpha", "beta"):
        folder = home / "skills" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test skill\ndeps: [fixture-tool]\n---\n{name}-instructions\n",
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_bound_skill_dependencies_and_preprocessing_leave_loop_responsive(tmp_path, monkeypatch):
    from agent import skill_commands

    runner, event, entry, generation = _turn(monkeypatch)
    home = tmp_path / "owner"
    _skills(home)
    loop, loop_thread = asyncio.get_running_loop(), threading.get_ident()
    release, finished, ticked = threading.Event(), threading.Event(), asyncio.Event()
    observations, responsive, preprocessing = [], [], []
    original_preprocess = skill_commands.preprocess_skill_content

    def tick():
        responsive.append(not finished.is_set())
        release.set()
        ticked.set()

    def ensure(dep):
        observations.append((dep, get_hermes_home(), get_session_env("HERMES_SESSION_KEY"), threading.get_ident()))
        loop.call_soon_threadsafe(tick)
        release.wait(5)
        finished.set()

    def preprocess(*args, **kwargs):
        preprocessing.append((get_hermes_home(), threading.get_ident()))
        return original_preprocess(*args, **kwargs)

    monkeypatch.setattr("pm.ensure", ensure)
    monkeypatch.setattr(skill_commands, "preprocess_skill_content", preprocess)
    home_token = set_hermes_home_override(home)
    tokens = []
    try:
        prepared, tokens = await runner._hmwa_prepare_turn(
            event, event.source, entry, entry.session_key, entry.session_key, generation,
        )
        await asyncio.wait_for(ticked.wait(), 5)
        assert responsive[0], "dependency setup blocked an unrelated event-loop callback"
        assert len(observations) == len(preprocessing) == 2
        assert all(dep == "fixture-tool" and seen_home == home and key == entry.session_key
                   and thread != loop_thread for dep, seen_home, key, thread in observations)
        assert all(seen_home == home and thread != loop_thread for seen_home, thread in preprocessing)
        assert prepared.title_user_message == "original request"
        assert event.text.index("alpha-instructions") < event.text.index("beta-instructions")
        assert event.text.endswith("original request")
        assert "alpha-instructions" in prepared.message_text
    finally:
        release.set()
        runner._clear_session_env(tokens)
        reset_hermes_home_override(home_token)
        runner._get_executor().shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["stop", "replace", "cancel"])
async def test_displaced_bound_skill_load_cannot_mutate_event_or_resume_turn(monkeypatch, action):
    runner, event, entry, generation = _turn(monkeypatch)
    event.auto_skill = "alpha"
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    release = threading.Event()

    def displace():
        runner._invalidate_session_run_generation(entry.session_key, reason="stop during skill load")
        if action == "replace":
            runner._begin_session_run_generation(entry.session_key)
        if action == "cancel":
            task.cancel()
        release.set()

    def load(name, **kwargs):
        loop.call_soon_threadsafe(displace)
        release.wait(5)
        return {"name": name, "content": f"{name}-instructions"}, None, name

    monkeypatch.setattr("agent.skill_commands._load_skill_payload", load)
    monkeypatch.setattr("agent.skill_commands._build_skill_message", lambda payload, *args: payload["content"])
    previous_key = get_session_env("HERMES_SESSION_KEY")
    tokens = []
    try:
        if action == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await runner._hmwa_prepare_turn(
                    event, event.source, entry, entry.session_key, entry.session_key, generation,
                )
        else:
            prepared, tokens = await runner._hmwa_prepare_turn(
                event, event.source, entry, entry.session_key, entry.session_key, generation,
            )
            assert prepared is None
        runner._get_executor().shutdown(wait=True)
        assert event.text == "original request"
        runner._hmwa_acquire_turn_lease.assert_not_awaited()
        runner._mark_durable_active_turn.assert_not_awaited()
        assert get_session_env("HERMES_SESSION_KEY") == previous_key
    finally:
        release.set()
        if tokens and get_session_env("HERMES_SESSION_KEY") != previous_key:
            runner._clear_session_env(tokens)
        runner._get_executor().shutdown(wait=True)
