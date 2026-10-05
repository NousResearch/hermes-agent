"""A pre_gateway_dispatch "skip" drop must not surface as a successful turn (#133475).

The skip was reported upward as a ``None`` response, and the adapter lifecycle scored
``not bool(response)`` as success: platforms rendered the same 👀→✅ ack as a turn that
ran and answered. The drop now travels on the event as ``_hermes_pre_gateway_skip`` and
completes as ``ProcessingOutcome.SKIPPED`` — adapters retract the in-progress marker and
add no verdict.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class RecordingAdapter(BasePlatformAdapter):
    """Records the processing-complete outcome it is driven through."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.completed = []

    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def stop_typing(self, chat_id):
        return None

    async def on_processing_complete(self, event, outcome):
        self.completed.append(outcome)


def _event(**attrs):
    chat_id = attrs.pop("chat_id", "chat-1")
    message_id = attrs.pop("message_id", "m-1")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )
    for key, value in attrs.items():
        setattr(event, key, value)
    return event


def _wire(adapter):
    adapter._start_typing_refresh = lambda *a: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *a: None
    adapter._message_handler = AsyncMock(return_value=None)


@pytest.mark.asyncio
async def test_hook_skipped_message_completes_as_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = RecordingAdapter()
    _wire(adapter)
    event = _event(_hermes_pre_gateway_skip=True)

    await adapter._process_message_background(event, "session")

    assert adapter.completed == [ProcessingOutcome.SKIPPED]


@pytest.mark.asyncio
async def test_empty_streamed_response_still_completes_as_success(
    tmp_path, monkeypatch
):
    """Behavior lock: only a hook drop is SKIPPED — a normal streamed/queued turn whose
    handler legitimately returns None keeps reporting success."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = RecordingAdapter()
    _wire(adapter)
    event = _event()

    await adapter._process_message_background(event, "session")

    assert adapter.completed == [ProcessingOutcome.SUCCESS]


@pytest.mark.asyncio
async def test_admit_event_marks_the_original_event_on_skip(monkeypatch):
    """``_hm_admit_event`` returns None on a skip hook AND marks the event object the
    adapter still holds — the marker must survive the local rebinding of ``event``."""
    import hermes_cli.lifecycle as lifecycle
    from gateway.run_inbound import GatewayInboundMixin

    async def fake_ainvoke_hook(hook_name, **kwargs):
        if hook_name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin says drop"}]
        return []

    monkeypatch.setattr(lifecycle, "ainvoke_hook", fake_ainvoke_hook)

    runner = object.__new__(GatewayInboundMixin)
    runner._scale_to_zero_note_real_inbound = lambda: None
    event = _event()

    admitted = await runner._hm_admit_event(event)

    assert admitted is None
    assert getattr(event, "_hermes_pre_gateway_skip", False) is True


@pytest.mark.asyncio
async def test_matrix_skipped_retracts_eyes_without_verdict():
    """SKIPPED on Matrix retracts the 👀 and adds neither ✅ nor ❌."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="syt_test_token",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@bot:example.org",
            },
        )
    )
    adapter._reactions_enabled = True
    adapter._reaction_redaction_delay_seconds = 0.01
    adapter._pending_reactions = {("!room:ex", "$msg1"): "$eyes_reaction_123"}
    adapter._redact_reaction = AsyncMock(return_value=True)
    adapter._send_reaction = AsyncMock(return_value="$next_reaction")

    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.MATRIX, chat_id="!room:ex", chat_type="dm"
        ),
        raw_message={},
        message_id="$msg1",
    )
    await adapter.on_processing_complete(event, ProcessingOutcome.SKIPPED)

    adapter._send_reaction.assert_not_awaited()
    await asyncio.sleep(0.03)
    adapter._redact_reaction.assert_awaited_once_with(
        "!room:ex", "$eyes_reaction_123", "processing complete"
    )


@pytest.mark.asyncio
async def test_telegram_skipped_clears_reactions(monkeypatch):
    """SKIPPED on Telegram clears the in-progress reaction (no 👍/👎 verdict)."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._bot = AsyncMock()
    event = _event(chat_id="123", message_id="456")

    await adapter.on_processing_complete(event, ProcessingOutcome.SKIPPED)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


def test_discord_skipped_message_is_not_redispatched(monkeypatch, tmp_path):
    """SKIPPED is ledgered as handled, so missed-message backfill does not re-dispatch a
    message a hook dropped on purpose (re-dispatch would just be dropped again)."""
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake-token"))
    raw = type("R", (), {"id": 91})()
    event = MessageEvent(
        text="dropped by hook",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="91", chat_type="dm"),
        raw_message=raw,
        message_id="91",
    )

    adapter._record_discord_processing_start(event, emoji_ack=False)
    adapter._record_discord_processing_complete(event, ProcessingOutcome.SKIPPED)

    assert adapter._discord_message_is_persistently_complete("91") is True
