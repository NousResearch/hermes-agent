"""``send_transcript_echo``: the adapter hook behind ``stt.echo_transcripts``."""

from types import SimpleNamespace

import pytest

from agent.i18n import t
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _TextAdapter(BasePlatformAdapter):
    """Keeps the default hook; records what reaches ``send``."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="m1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _NativeEchoAdapter(_TextAdapter):
    """Renders transcripts itself."""

    def __init__(self):
        super().__init__()
        self.echoes = []

    async def send_transcript_echo(self, chat_id, transcript, metadata=None):
        self.echoes.append((chat_id, transcript, metadata))
        return SendResult(success=True, message_id="e1")


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_echo_transcripts=True)
    return runner


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")


@pytest.mark.asyncio
async def test_default_hook_sends_the_localized_echo_line_through_send():
    adapter = _TextAdapter()
    meta = {"thread_id": "7"}

    await _runner()._echo_stt_transcripts(adapter, _source(), ["turn on the lights"], metadata=meta)

    assert adapter.sent == [("12345", t("gateway.voice.transcript_echo_short", text="turn on the lights"), meta)]


@pytest.mark.asyncio
async def test_override_receives_raw_transcripts_and_send_is_not_used():
    adapter = _NativeEchoAdapter()
    meta = {"thread_id": "7"}

    await _runner()._echo_stt_transcripts(adapter, _source(), ["turn on the lights", "and the fan"], metadata=meta)

    assert adapter.echoes == [("12345", "turn on the lights", meta), ("12345", "and the fan", meta)]
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_pending_voice_reaches_the_override_once_per_transcript():
    adapter = _NativeEchoAdapter()
    runner = _runner()
    event = SimpleNamespace()

    await runner._echo_pending_stt_transcripts_once(event, adapter, _source(), ["first"])
    await runner._echo_pending_stt_transcripts_once(event, adapter, _source(), ["first", "second"])

    assert [transcript for _chat, transcript, _meta in adapter.echoes] == ["first", "second"]


@pytest.mark.asyncio
async def test_failing_override_does_not_propagate():
    class _BrokenEchoAdapter(_TextAdapter):
        async def send_transcript_echo(self, chat_id, transcript, metadata=None):
            raise RuntimeError("display offline")

    adapter = _BrokenEchoAdapter()

    await _runner()._echo_stt_transcripts(adapter, _source(), ["hello"])

    assert adapter.sent == []
