"""Post-handler auto-TTS runs in the routed profile's secret scope under a multiplexed gateway.

The base adapter synthesizes the voice reply to a voice note after the message handler returned,
when the turn's scope is already reset. A provider setting read there through the profile secret
scope (Gemini's ``GEMINI_BASE_URL``, a plugin provider's key in ``is_available``) raised
``UnscopedSecretError`` and the voice reply was silently dropped (#145151).
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.base import BasePlatformAdapter, SendResult
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _multiplexed_runner(profile_home: Path) -> SimpleNamespace:
    """The runner surface the adapter's delivery path reads, with the REAL scope resolvers."""
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True),
                             _resolve_profile_home_for_source=lambda _source: profile_home)
    for name in ("_profile_scope_key_for_source", "_async_profile_scope_for_source",
                 "_media_delivery_scope_for_source"):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner))
    return runner


@pytest.mark.asyncio
async def test_voice_note_auto_tts_reads_the_routed_profiles_settings(tmp_path, monkeypatch):
    from hermes_agent.agent import secret_scope

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("PROBE_TTS_BASE_URL=http://coder-tts.local/v1\n", encoding="utf-8")
    monkeypatch.setenv("PROBE_TTS_BASE_URL", "http://launch-profile.local/v1")
    secret_scope.set_multiplex_active(True)

    adapter = _Adapter()
    adapter.gateway_runner = _multiplexed_runner(profile_home)
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="reply text"))
    event = MessageEvent(text="hello", message_type=MessageType.VOICE, message_id="voice-1",
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm"))
    seen = []

    def provider_reads_its_setting():
        from hermes_agent.hermes_cli.config import get_env_value
        seen.append(get_env_value("PROBE_TTS_BASE_URL"))  # raises UnscopedSecretError with no scope
        return True

    def fake_tts(*, text, output_path=None):
        provider_reads_its_setting()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"OggS")
        return json.dumps({"success": True, "file_path": output_path})

    with patch("hermes_agent.tools.tts_tool.check_tts_requirements", side_effect=provider_reads_its_setting), \
            patch("hermes_agent.tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
        await adapter._process_message_background(event, build_session_key(event.source))

    # Availability check and synthesis both read the ROUTED profile's value, never the launch env.
    assert seen == ["http://coder-tts.local/v1", "http://coder-tts.local/v1"]
    adapter.play_tts.assert_awaited_once()
    assert adapter.play_tts.await_args.kwargs["audio_path"].endswith(".ogg")
