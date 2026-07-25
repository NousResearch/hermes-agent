"""Regression coverage for Discord triggering-message routing metadata.

The current Discord message id is model-only routing context.
It must reach the model when Discord tools are loaded without becoming user-authored transcript content.
"""

import base64
import sys
import types
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import SessionDB
from run_agent import AIAgent


SESSION_KEY = "agent:main:discord:channel:channel-123:user-456"
MESSAGE_ID = "synthetic-discord-message-id"
USER_TEXT = "hello world"
ROUTING_NOTE = (
    f"[Triggering message id: `{MESSAGE_ID}` \u2014 use as `message_id` "
    "for reply/react/pin via the discord tools.]"
)
ROUTING_PREFIX = f"{ROUTING_NOTE}\n\n"


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        chat_type="group",
        user_id="user-456",
    )


def _event() -> MessageEvent:
    return MessageEvent(
        text=USER_TEXT,
        source=_source(),
        message_id=MESSAGE_ID,
    )


def _runner(monkeypatch, tmp_path) -> gateway_run.GatewayRunner:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = gateway_run.GatewayRunner(
        GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(enabled=True, token="fake"),
            }
        )
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    # _run_agent reads this mapping directly to detect resume-pending turns.
    runner.session_store._entries = {}
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id="session-discord-routing",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    runner._async_session_store = MagicMock()
    runner._async_session_store._store = runner.session_store
    runner._async_session_store.get_or_create_session = AsyncMock(
        return_value=runner.session_store.get_or_create_session.return_value
    )
    runner._async_session_store.load_transcript = AsyncMock(return_value=[])
    runner._async_session_store.has_any_sessions = AsyncMock(return_value=True)
    runner._async_session_store.append_to_transcript = AsyncMock()
    runner._async_session_store.update_session = AsyncMock()

    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "fake"},
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.parametrize("routed_tools_loaded", [True, False])
def test_discord_tool_gate_uses_routed_profile_scope(
    monkeypatch,
    tmp_path,
    routed_tools_loaded,
):
    runner = object.__new__(gateway_run.GatewayRunner)
    setattr(runner, "config", GatewayConfig(multiplex_profiles=True))
    setattr(
        runner,
        "_resolve_profile_home_for_source",
        lambda source: tmp_path / "routed",
    )
    active_home = None

    @contextmanager
    def fake_scope(home):
        nonlocal active_home
        previous = active_home
        active_home = home
        try:
            yield
        finally:
            active_home = previous

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(
        "gateway.session._discord_tools_loaded",
        lambda: (
            routed_tools_loaded
            if active_home == tmp_path / "routed"
            else not routed_tools_loaded
        ),
    )

    assert runner._discord_tools_loaded_for_source(_source()) is routed_tools_loaded
    assert active_home is None


def _persist_turn(
    tmp_path,
    *,
    content: Any,
    api_content: Any | None,
    override: Any,
):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="session-1", source="discord")
        agent = object.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "session-1"
        agent._last_flushed_db_idx = 0
        agent._flushed_db_message_ids = set()
        agent._flushed_db_message_session_id = None
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = override
        agent._persist_user_message_timestamp = None
        agent._persist_disabled = False
        agent._session_persist_lock = None

        message = {"role": "user", "content": content}
        if api_content is not None:
            message["api_content"] = api_content
        agent._flush_messages_to_session_db([message], [])
        return (
            db.get_messages("session-1"),
            db.get_messages_as_conversation("session-1"),
        )
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("discord_tools_loaded", [True, False])
async def test_triggering_message_metadata_is_model_only(
    monkeypatch,
    tmp_path,
    discord_tools_loaded,
):
    runner = _runner(monkeypatch, tmp_path)
    captured = {}

    async def fake_run_agent(*, message, session_key, **kwargs):
        notes = runner._consume_pending_turn_sidecar_notes(session_key)
        captured.update(message=message, notes=notes, kwargs=kwargs)
        return {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)

    with patch(
        "gateway.session._discord_tools_loaded",
        return_value=discord_tools_loaded,
    ):
        await runner._handle_message_with_agent(
            _event(),
            _source(),
            SESSION_KEY,
            1,
        )

    assert captured["kwargs"]["persist_user_message"] == USER_TEXT
    assert captured["kwargs"]["transient_user_message_prefix"] == (
        ROUTING_PREFIX if discord_tools_loaded else None
    )

    notes = captured["notes"]
    assert notes == []
    model_content = captured["message"]
    assert model_content == USER_TEXT

    rows, replay_history = _persist_turn(
        tmp_path,
        content=model_content,
        api_content=None,
        override=captured["kwargs"]["persist_user_message"],
    )
    user_rows = [row for row in rows if row["role"] == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["content"] == USER_TEXT
    assert user_rows[0].get("api_content") is None
    assert "api_content" not in replay_history[0]
    replayed_content = replay_history[0]["content"]
    assert MESSAGE_ID not in replayed_content


@pytest.mark.asyncio
async def test_timestamp_failure_keeps_routing_prefix_out_of_canonical_message(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    captured = {}

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    monkeypatch.setattr(
        "hermes_time.get_timezone",
        MagicMock(side_effect=RuntimeError("synthetic timezone failure")),
    )
    with patch("gateway.session._discord_tools_loaded", return_value=True):
        await runner._handle_message_with_agent(
            _event(),
            _source(),
            SESSION_KEY,
            1,
        )

    assert captured["message"] == USER_TEXT
    assert captured["persist_user_message"] == USER_TEXT
    assert captured["transient_user_message_prefix"] == ROUTING_PREFIX
    assert MESSAGE_ID not in captured["message"]
    assert MESSAGE_ID not in captured["persist_user_message"]

    rows, replay_history = _persist_turn(
        tmp_path,
        content=captured["message"],
        api_content=None,
        override=captured["persist_user_message"],
    )
    assert MESSAGE_ID not in str(rows)
    assert MESSAGE_ID not in str(replay_history)


@pytest.mark.asyncio
async def test_captionless_turn_preserves_explicit_empty_persistence_override(
    monkeypatch,
    tmp_path,
):
    runner = _runner(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(
        runner,
        "_prepare_profile_scoped_inbound_message_text",
        AsyncMock(return_value=""),
    )

    async def fake_run_agent(**kwargs):
        captured.update(kwargs)
        return {
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    monkeypatch.setattr(runner, "_run_agent", fake_run_agent)
    with patch("gateway.session._discord_tools_loaded", return_value=True):
        await runner._handle_message_with_agent(
            _event(),
            _source(),
            SESSION_KEY,
            1,
        )

    assert captured["persist_user_message"] == ""
    assert captured["transient_user_message_prefix"] == ROUTING_PREFIX
    assert captured["message"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("caption", [USER_TEXT, ""])
async def test_native_image_keeps_routing_note_out_of_persisted_content(
    monkeypatch,
    tmp_path,
    caption,
):
    runner = _runner(monkeypatch, tmp_path)
    native_session_key = f"{SESSION_KEY}:native"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "/x8AAusB9Y9ZP14AAAAASUVORK5CYII="
        )
    )
    runner._pending_native_image_paths_by_session[native_session_key] = [str(image_path)]
    captured = {}

    fake_agent = MagicMock()
    fake_agent.provider = "openai"
    fake_agent.model = "test-model"
    fake_agent.run_conversation.side_effect = lambda message, **kwargs: (
        captured.update(message=message, kwargs=kwargs)
        or {
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "ok"},
            ],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda *_args, **_kwargs: "test-model")
    with patch("run_agent.AIAgent", return_value=fake_agent):
        await runner._run_agent(
            message=caption,
            context_prompt="",
            history=[],
            source=_source(),
            session_id="session-discord-routing",
            session_key=native_session_key,
            persist_user_message=caption,
            transient_user_message_prefix=ROUTING_PREFIX,
        )

    assert "message" in captured
    model_content = captured["message"]
    persist_content = captured["kwargs"]["persist_user_message"]
    assert captured["kwargs"]["transient_user_message_prefix"] == ROUTING_PREFIX
    assert isinstance(model_content, list)
    assert ROUTING_NOTE not in model_content[0]["text"]
    assert isinstance(persist_content, list)
    expected_clean_text = caption or "What do you see in this image?"
    assert persist_content[0]["text"].startswith(expected_clean_text)
    assert ROUTING_NOTE not in persist_content[0]["text"]
    assert persist_content[1] == model_content[1]

    rows, replay_history = _persist_turn(
        tmp_path,
        content=model_content,
        api_content=None,
        override=persist_content,
    )
    user_rows = [row for row in rows if row["role"] == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["content"].startswith(expected_clean_text)
    assert ROUTING_NOTE not in user_rows[0]["content"]
    assert user_rows[0].get("api_content") is None
    assert "api_content" not in replay_history[0]
    assert ROUTING_NOTE not in replay_history[0]["content"]


@pytest.mark.asyncio
async def test_aborted_turn_cannot_leak_routing_note_to_next_turn(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)

    async def abort_run(**_kwargs):
        raise RuntimeError("synthetic abort")

    monkeypatch.setattr(runner, "_run_agent", abort_run)
    with patch("gateway.session._discord_tools_loaded", return_value=True):
        await runner._handle_message_with_agent(
            _event(),
            _source(),
            SESSION_KEY,
            1,
        )

    assert runner._consume_pending_turn_sidecar_notes(SESSION_KEY) == []
    assert runner._release_turn_lease(SESSION_KEY, 1)

    captured = {}

    async def capture_run(**kwargs):
        captured.update(kwargs)
        return {
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }

    monkeypatch.setattr(runner, "_run_agent", capture_run)
    with patch("gateway.session._discord_tools_loaded", return_value=False):
        await runner._handle_message_with_agent(
            _event(),
            _source(),
            SESSION_KEY,
            2,
        )

    assert captured["message"] == USER_TEXT
    assert MESSAGE_ID not in captured["message"]
