from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(platform: Platform, text: str = "hello") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(platform: Platform) -> tuple[object, SessionEntry]:
    from gateway.run import GatewayRunner

    source = _make_source(platform)
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=platform,
        chat_type="dm",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {platform: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 10,
            "input_tokens": 20,
            "output_tokens": 5,
            "model": "openai/test-model",
        }
    )
    return runner, session_entry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "expected_command"),
    [
        (Platform.SLACK, "/hermes sethome"),
        (Platform.TELEGRAM, "/sethome"),
    ],
)
async def test_home_channel_onboarding_uses_platform_specific_sethome_command(
    monkeypatch, platform: Platform, expected_command: str
):
    import gateway.run as gateway_run

    runner, session_entry = _make_runner(platform)
    monkeypatch.delenv(f"{platform.value.upper()}_HOME_CHANNEL", raising=False)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event(platform))

    assert result == "ok"
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=10,
    )
    onboarding_text = runner.adapters[platform].send.await_args.args[1]
    assert f"Type {expected_command}" in onboarding_text
    if platform == Platform.SLACK:
        assert "Type /sethome to make this chat your home channel" not in onboarding_text
    else:
        assert "/hermes sethome" not in onboarding_text
