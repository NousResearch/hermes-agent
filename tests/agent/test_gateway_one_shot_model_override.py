import types

import pytest

from gateway.platforms.base import MessageEvent, MessageType, Platform, SessionSource
from gateway.run import GatewayRunner
from gateway.slash_commands import GatewaySlashCommandsMixin


class _DummySlashRunner(GatewaySlashCommandsMixin):
    pass


class _FakeSessionStore:
    def _generate_session_key(self, source):
        return f"{source.platform.value}:{source.chat_id}:{source.user_id}"


def _make_source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_name="chat",
        chat_type="dm",
        user_id="user-1",
        user_name="user",
    )


def _make_command_event(text: str, chat_id: str = "chat-1") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=_make_source(chat_id),
    )


def _make_runner() -> _DummySlashRunner:
    runner = object.__new__(_DummySlashRunner)
    runner.adapters = {}
    runner.session_store = _FakeSessionStore()
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_one_shot_model_overrides = {}
    runner._normalize_source_for_session_key = types.MethodType(
        GatewayRunner._normalize_source_for_session_key, runner
    )
    runner._session_key_for_source = types.MethodType(
        GatewayRunner._session_key_for_source, runner
    )
    runner._canonical_session_key_for_source = types.MethodType(
        GatewayRunner._canonical_session_key_for_source, runner
    )
    runner._resolve_turn_agent_config = types.MethodType(
        GatewayRunner._resolve_turn_agent_config, runner
    )
    runner._pop_one_shot_model_override = types.MethodType(
        GatewayRunner._pop_one_shot_model_override, runner
    )
    runner._thread_metadata_for_source = lambda source, anchor: None
    runner._reply_anchor_for_event = lambda event: None
    runner._evict_cached_agent = lambda session_key: None
    runner._agent_cache_lock = None
    runner._agent_cache = None
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_model_auto_clears_pending_override():
    runner = _make_runner()
    session_key = runner._canonical_session_key_for_source(_make_source("chat-1"))
    runner._pending_one_shot_model_overrides[session_key] = {"model": "gpt-5.4"}

    result = await runner._handle_model_command(_make_command_event("/model auto"))

    assert result == "Автоматический выбор модели включён."
    assert session_key not in runner._pending_one_shot_model_overrides


@pytest.mark.asyncio
async def test_model_strong_sets_pending_override_for_session_only():
    runner = _make_runner()

    result = await runner._handle_model_command(_make_command_event("/model strong", chat_id="chat-1"))

    key1 = runner._canonical_session_key_for_source(_make_source("chat-1"))
    key2 = runner._canonical_session_key_for_source(_make_source("chat-2"))
    assert result == "Сильная модель gpt-5.4 выбрана для следующей задачи. После неё вернусь в auto."
    assert runner._pending_one_shot_model_overrides[key1]["model"] == "gpt-5.4"
    assert key2 not in runner._pending_one_shot_model_overrides


@pytest.mark.asyncio
async def test_model_gpt54_sets_pending_override(monkeypatch):
    from gateway import run as gateway_run

    runner = _make_runner()

    class _Result:
        success = True
        new_model = "gpt-5.4"
        target_provider = "openai-api"
        api_key = "redacted"
        base_url = "http://localhost:20128/v1"
        api_mode = "chat_completions"
        provider_label = "OpenAI"
        model_info = None
        warning_message = None

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"model": {}})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: _Result())

    result = await runner._handle_model_command(_make_command_event("/model gpt-5.4"))

    key = runner._canonical_session_key_for_source(_make_source("chat-1"))
    assert result == "Следующая задача будет выполнена с gpt-5.4. После неё вернусь в auto."
    assert runner._pending_one_shot_model_overrides[key]["model"] == "gpt-5.4"


def test_pending_override_clears_after_successful_completion():
    runner = object.__new__(GatewayRunner)
    runner._pending_one_shot_model_overrides = {"session-1": {"model": "gpt-5.4"}}

    current = runner._pop_one_shot_model_override("session-1", success=True)

    assert current == {"model": "gpt-5.4"}
    assert "session-1" not in runner._pending_one_shot_model_overrides


def test_pending_override_remains_after_provider_failure():
    runner = object.__new__(GatewayRunner)
    runner._pending_one_shot_model_overrides = {"session-1": {"model": "gpt-5.4"}}

    current = runner._pop_one_shot_model_override("session-1", success=False)

    assert current == {"model": "gpt-5.4"}
    assert runner._pending_one_shot_model_overrides["session-1"]["model"] == "gpt-5.4"


@pytest.mark.asyncio
async def test_command_to_next_turn_handoff_uses_canonical_key_and_clears_after_success():
    runner = _make_runner()

    command_result = await runner._handle_model_command(_make_command_event("/model strong", chat_id="chat-1"))
    assert command_result == "Сильная модель gpt-5.4 выбрана для следующей задачи. После неё вернусь в auto."

    same_chat_source = _make_source("chat-1")
    session_key = runner._canonical_session_key_for_source(same_chat_source)
    assert runner._pending_one_shot_model_overrides[session_key]["model"] == "gpt-5.4"

    def build_effective_route(source, message):
        canonical_key = runner._canonical_session_key_for_source(source)
        model = "gpt-5.4-mini"
        runtime_kwargs = {
            "provider": "openai-api",
            "base_url": "http://localhost:20128/v1",
            "api_key": "redacted",
            "api_mode": "chat_completions",
        }
        pending = runner._pending_one_shot_model_overrides.get(canonical_key)
        if pending and pending.get("model") and not runtime_kwargs.get("model"):
            runtime_kwargs = dict(runtime_kwargs)
            runtime_kwargs["model"] = pending["model"]
        route = runner._resolve_turn_agent_config(message, model, runtime_kwargs)
        return canonical_key, route

    first_key, first_route = build_effective_route(same_chat_source, "OVERRIDE_OK")
    assert first_key == session_key
    assert first_route["model"] == "gpt-5.4"

    runner._pop_one_shot_model_override(session_key, success=True)
    assert session_key not in runner._pending_one_shot_model_overrides

    _, second_route = build_effective_route(same_chat_source, "AUTO_MINI_OK")
    assert second_route["model"] == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_different_chat_does_not_receive_override():
    runner = _make_runner()

    await runner._handle_model_command(_make_command_event("/model strong", chat_id="chat-1"))

    key1 = runner._canonical_session_key_for_source(_make_source("chat-1"))
    key2 = runner._canonical_session_key_for_source(_make_source("chat-2"))

    runtime_kwargs = {
        "provider": "openai-api",
        "base_url": "http://localhost:20128/v1",
        "api_key": "redacted",
        "api_mode": "chat_completions",
    }
    pending = runner._pending_one_shot_model_overrides.get(key2)
    if pending and pending.get("model") and not runtime_kwargs.get("model"):
        runtime_kwargs = dict(runtime_kwargs)
        runtime_kwargs["model"] = pending["model"]
    route = runner._resolve_turn_agent_config("AUTO_MINI_OK", "gpt-5.4-mini", runtime_kwargs)

    assert runner._pending_one_shot_model_overrides[key1]["model"] == "gpt-5.4"
    assert key2 not in runner._pending_one_shot_model_overrides
    assert route["model"] == "gpt-5.4-mini"


@pytest.mark.asyncio
async def test_failure_before_final_response_keeps_pending_override_for_same_chat():
    runner = _make_runner()

    await runner._handle_model_command(_make_command_event("/model strong", chat_id="chat-1"))
    session_key = runner._canonical_session_key_for_source(_make_source("chat-1"))

    runtime_kwargs = {
        "provider": "openai-api",
        "base_url": "http://localhost:20128/v1",
        "api_key": "redacted",
        "api_mode": "chat_completions",
    }
    pending = runner._pending_one_shot_model_overrides.get(session_key)
    if pending and pending.get("model") and not runtime_kwargs.get("model"):
        runtime_kwargs = dict(runtime_kwargs)
        runtime_kwargs["model"] = pending["model"]
    route = runner._resolve_turn_agent_config("OVERRIDE_OK", "gpt-5.4-mini", runtime_kwargs)

    assert route["model"] == "gpt-5.4"
    runner._pop_one_shot_model_override(session_key, success=False)
    assert runner._pending_one_shot_model_overrides[session_key]["model"] == "gpt-5.4"
