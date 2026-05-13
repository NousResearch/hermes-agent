"""Tests for Telegram topic/thread routing fallbacks.

Supergroup forum topics route with ``message_thread_id``. Hermes-created
private DM topic lanes are different: live Telegram testing showed they only
stay in the expected lane when sends include both the private topic
``message_thread_id`` and a ``reply_to_message_id`` anchor to the triggering
user message. If either anchor is unavailable or rejected, the adapter must
avoid retrying with a partial topic route that can render outside the lane.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SendResult,
    _reply_anchor_for_event,
    _thread_metadata_for_source,
)
from gateway.session import build_session_key


# ── Fake telegram.error hierarchy ──────────────────────────────────────
# Mirrors the real python-telegram-bot hierarchy:
#   BadRequest → NetworkError → TelegramError → Exception


class FakeNetworkError(Exception):
    pass


class FakeBadRequest(FakeNetworkError):
    pass


class FakeTimedOut(FakeNetworkError):
    pass


class FakeRetryAfter(Exception):
    def __init__(self, seconds):
        super().__init__(f"Retry after {seconds}")
        self.retry_after = seconds


# Build a fake telegram module tree so the adapter's internal imports work
class _FakeInlineKeyboardButton:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data
        self.kwargs = kwargs


class _FakeInlineKeyboardMarkup:
    def __init__(self, inline_keyboard):
        self.inline_keyboard = inline_keyboard


class _FakeInputMediaPhoto:
    def __init__(self, media, caption=None, **kwargs):
        self.media = media
        self.caption = caption
        self.kwargs = kwargs


_fake_telegram = types.ModuleType("telegram")
_fake_telegram.Update = object
_fake_telegram.Bot = object
_fake_telegram.Message = object
_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram.InputMediaPhoto = _FakeInputMediaPhoto
_fake_telegram_error = types.ModuleType("telegram.error")
_fake_telegram_error.NetworkError = FakeNetworkError
_fake_telegram_error.BadRequest = FakeBadRequest
_fake_telegram_error.TimedOut = FakeTimedOut
_fake_telegram.error = _fake_telegram_error
_fake_telegram_constants = types.ModuleType("telegram.constants")
_fake_telegram_constants.ParseMode = SimpleNamespace(
    MARKDOWN_V2="MarkdownV2",
    MARKDOWN="Markdown",
    HTML="HTML",
)
_fake_telegram_constants.ChatType = SimpleNamespace(
    GROUP="group",
    SUPERGROUP="supergroup",
    CHANNEL="channel",
    PRIVATE="private",
)
_fake_telegram.constants = _fake_telegram_constants
_fake_telegram_ext = types.ModuleType("telegram.ext")
_fake_telegram_ext.Application = object
_fake_telegram_ext.CommandHandler = object
_fake_telegram_ext.CallbackQueryHandler = object
_fake_telegram_ext.MessageHandler = object
_fake_telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
_fake_telegram_ext.filters = object
_fake_telegram_request = types.ModuleType("telegram.request")
_fake_telegram_request.HTTPXRequest = object


@pytest.fixture(autouse=True)
def _inject_fake_telegram(monkeypatch):
    """Inject fake telegram modules so the adapter can import from them."""
    monkeypatch.setitem(sys.modules, "telegram", _fake_telegram)
    monkeypatch.setitem(sys.modules, "telegram.error", _fake_telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.constants", _fake_telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.ext", _fake_telegram_ext)
    monkeypatch.setitem(sys.modules, "telegram.request", _fake_telegram_request)


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    adapter._fallback_ips = []
    adapter._polling_conflict_count = 0
    adapter._polling_network_error_count = 0
    adapter._polling_error_callback_ref = None
    adapter.platform = Platform.TELEGRAM
    return adapter


def test_forum_general_topic_without_message_thread_id_keeps_thread_context():
    """Forum General-topic messages should keep synthetic thread context."""
    from gateway.platforms import telegram as telegram_mod

    adapter = _make_adapter()
    message = SimpleNamespace(
        text="hello from General",
        caption=None,
        chat=SimpleNamespace(
            id=-100123,
            type=telegram_mod.ChatType.SUPERGROUP,
            is_forum=True,
            title="Forum group",
        ),
        from_user=SimpleNamespace(id=456, full_name="Alice"),
        message_thread_id=None,
        reply_to_message=None,
        message_id=10,
        date=None,
    )

    event = adapter._build_message_event(message, msg_type=SimpleNamespace(value="text"))

    assert event.source.chat_id == "-100123"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "1"


@pytest.mark.asyncio
async def test_send_omits_general_topic_thread_id():
    """Telegram sends to forum General should omit message_thread_id=1."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="-100123",
        content="test message",
        metadata={"thread_id": "1"},
    )

    assert result.success is True
    assert len(call_log) == 1
    assert call_log[0]["chat_id"] == -100123
    assert call_log[0]["text"] == "test message"
    assert call_log[0]["reply_to_message_id"] is None
    assert call_log[0]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_send_typing_preserves_general_topic_thread_id():
    """Typing for forum General must send message_thread_id=1, not None.

    Asymmetric with _message_thread_id_for_send: sendMessage rejects
    message_thread_id=1, but sendChatAction needs it to scope the typing
    bubble to the General topic. Omitting it (message_thread_id=None) hides
    the bubble from the General-topic view entirely.

    Regression guard for the d5357f816 refactor that mapped "1" → None in
    the typing resolver and silently killed typing indicators in every
    forum-group General topic.
    """
    adapter = _make_adapter()
    call_log = []

    async def mock_send_chat_action(**kwargs):
        call_log.append(dict(kwargs))

    adapter._bot = SimpleNamespace(send_chat_action=mock_send_chat_action)

    await adapter.send_typing("-100123", metadata={"thread_id": "1"})

    assert call_log == [
        {"chat_id": -100123, "action": "typing", "message_thread_id": 1},
    ]


@pytest.mark.asyncio
async def test_send_typing_does_not_fall_back_to_root_for_dm_topic():
    """Typing failures in DM topics should not show an indicator in All Messages."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_chat_action(**kwargs):
        call_log.append(dict(kwargs))
        raise FakeBadRequest("Message thread not found")

    adapter._bot = SimpleNamespace(send_chat_action=mock_send_chat_action)

    await adapter.send_typing("12345", metadata={"thread_id": "22182"})

    assert call_log == [
        {"chat_id": 12345, "action": "typing", "message_thread_id": 22182},
    ]


@pytest.mark.asyncio
async def test_send_typing_skips_api_call_for_dm_topic_reply_fallback():
    """Hermes-created DM topic lanes have no working Bot API typing route.

    ``send_chat_action`` only accepts ``message_thread_id``, which Telegram's
    Bot API 10.0 rejects for these lanes — the call would silently fail and
    log a "thread not found" warning every typing tick (every 2s). Skipping
    the call entirely keeps logs clean while preserving the user-visible
    behavior (no typing indicator either way for these lanes).
    """
    adapter = _make_adapter()
    call_log = []

    async def mock_send_chat_action(**kwargs):
        call_log.append(dict(kwargs))

    adapter._bot = SimpleNamespace(send_chat_action=mock_send_chat_action)

    await adapter.send_typing(
        "12345",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert call_log == []


@pytest.mark.asyncio
async def test_send_retries_without_thread_on_thread_not_found():
    """When message_thread_id causes 'thread not found', retry without it."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        tid = kwargs.get("message_thread_id")
        if tid is not None:
            raise FakeBadRequest("Message thread not found")
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    assert result.message_id == "42"
    # First call has thread_id, second call retries without
    assert len(call_log) == 2
    assert call_log[0]["message_thread_id"] == 99999
    assert call_log[1]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_send_private_dm_topic_uses_direct_messages_topic_id():
    """Private Telegram topics route sends via direct_messages_topic_id."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        metadata={"thread_id": "99999", "direct_messages_topic_id": "99999"},
    )

    assert result.success is True
    assert call_log[0]["message_thread_id"] is None
    assert call_log[0]["direct_messages_topic_id"] == 99999


def test_base_gateway_metadata_marks_telegram_dm_topics_as_reply_fallback():
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_type="dm",
        thread_id="20189",
    )

    metadata = _thread_metadata_for_source(source, "462")

    assert metadata == {
        "thread_id": "20189",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
    }


def test_base_gateway_replies_to_triggering_message_for_telegram_dm_topic():
    """Private DM topic lanes should anchor replies to the active user message."""
    event = SimpleNamespace(
        message_id="463",
        reply_to_message_id="462",
        source=SimpleNamespace(
            platform=Platform.TELEGRAM,
            chat_type="dm",
            thread_id="20189",
        ),
    )

    assert _reply_anchor_for_event(event) == "463"


@pytest.mark.asyncio
async def test_gateway_runner_busy_ack_replies_to_triggering_message_for_telegram_dm_topic(monkeypatch, tmp_path):
    """GatewayRunner's duplicate thread metadata must match the base helper."""
    from gateway import run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    GatewayRunner = gateway_run.GatewayRunner

    class BusyAdapter:
        def __init__(self):
            self._pending_messages = {}
            self.calls = []

        async def _send_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return SendResult(success=True, message_id="ack-1")

    class BusyAgent:
        def interrupt(self, _text):
            return None

        def get_activity_summary(self):
            return {}

    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id="20197",
        user_id="user-1",
    )
    event = MessageEvent(
        text="busy follow-up",
        message_type=MessageType.TEXT,
        source=source,
        message_id="463",
        reply_to_message_id="462",
    )
    session_key = build_session_key(source)
    adapter = BusyAdapter()

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._running_agents = {session_key: BusyAgent()}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._is_user_authorized = lambda _source: True

    assert await runner._handle_active_session_busy_message(event, session_key) is True

    assert adapter.calls
    assert adapter.calls[0]["reply_to"] == "463"
    assert adapter.calls[0]["metadata"] == {
        "thread_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "463",
    }


@pytest.mark.asyncio
async def test_send_uses_reply_fallback_for_hermes_dm_topics():
    """Hermes-created Telegram DM topics route with thread id plus reply anchor."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(kwargs)
        return SimpleNamespace(message_id=777)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        reply_to="462",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_send_uses_metadata_reply_fallback_for_streaming_dm_topics():
    """Metadata-only sends still stay in Hermes-created Telegram DM topics."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(kwargs)
        return SimpleNamespace(message_id=778)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="streamed text",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_send_reply_fallback_applies_to_every_chunk_for_dm_topics():
    """Long Telegram DM-topic fallback sends must anchor every chunk."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=len(call_log))

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="A" * 5000,
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert len(call_log) > 1
    assert all(call["reply_to_message_id"] == 462 for call in call_log)
    assert all(call["message_thread_id"] == 20197 for call in call_log)
    assert all("direct_messages_topic_id" not in call for call in call_log)


@pytest.mark.asyncio
async def test_send_model_picker_uses_metadata_reply_fallback_for_dm_topics():
    """Inline keyboard sends also consume the metadata reply fallback."""
    adapter = _make_adapter()
    adapter._model_picker_state = {}
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(kwargs)
        return SimpleNamespace(message_id=779)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send_model_picker(
        chat_id="123",
        providers=[{"name": "OpenAI", "slug": "openai", "models": [], "total_models": 0}],
        current_model="gpt-test",
        current_provider="openai",
        session_key="telegram:123:20197",
        on_model_selected=lambda *_: None,
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_send_dm_topic_fallback_without_anchor_does_not_crash():
    """DM-topic fallback without an anchor must not use message_thread_id alone."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=780)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="source-only send",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[0]
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_send_dm_topic_reply_not_found_retry_drops_thread_id():
    """If Telegram deletes the reply anchor, private-topic retry must drop thread id too."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise FakeBadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=781)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="anchor disappeared",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[1]
    assert "direct_messages_topic_id" not in call_log[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "bot_method_name", "path_kw", "filename", "payload"),
    [
        ("send_image_file", "send_photo", "image_path", "photo.png", b"png-data"),
        ("send_document", "send_document", "file_path", "report.txt", b"report-data"),
        ("send_video", "send_video", "video_path", "clip.mp4", b"video-data"),
        ("send_voice", "send_voice", "audio_path", "clip.ogg", b"ogg-data"),
        ("send_voice", "send_audio", "audio_path", "clip.mp3", b"mp3-data"),
    ],
)
async def test_native_media_dm_topic_reply_not_found_retry_drops_thread_id(
    tmp_path,
    method_name,
    bot_method_name,
    path_kw,
    filename,
    payload,
):
    adapter = _make_adapter()
    media_path = tmp_path / filename
    media_path.write_bytes(payload)
    call_log = []

    async def mock_send_media(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise FakeBadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=782)

    adapter._bot = SimpleNamespace(**{bot_method_name: mock_send_media})

    result = await getattr(adapter, method_name)(
        chat_id="123",
        **{path_kw: str(media_path)},
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[1]
    assert "direct_messages_topic_id" not in call_log[1]


@pytest.mark.asyncio
async def test_animation_dm_topic_reply_not_found_retry_drops_thread_id():
    adapter = _make_adapter()
    call_log = []

    async def mock_send_animation(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise FakeBadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=786)

    adapter._bot = SimpleNamespace(send_animation=mock_send_animation)

    result = await adapter.send_animation(
        chat_id="123",
        animation_url="https://example.com/anim.gif",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[1]
    assert "direct_messages_topic_id" not in call_log[1]


@pytest.mark.asyncio
async def test_media_group_dm_topic_reply_not_found_retry_drops_thread_id(tmp_path):
    adapter = _make_adapter()
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"png-data")
    call_log = []

    async def mock_send_media_group(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise FakeBadRequest("Message to be replied not found")
        return [SimpleNamespace(message_id=783)]

    adapter._bot = SimpleNamespace(send_media_group=mock_send_media_group)

    await adapter.send_multiple_images(
        chat_id="123",
        images=[(f"file://{image_path}", "caption")],
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[1]
    assert "direct_messages_topic_id" not in call_log[1]


@pytest.mark.asyncio
async def test_send_image_url_dm_topic_reply_not_found_retry_drops_thread_id(monkeypatch):
    adapter = _make_adapter()
    call_log = []

    async def mock_send_photo(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise FakeBadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=784)

    adapter._bot = SimpleNamespace(send_photo=mock_send_photo)
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: True)

    result = await adapter.send_image(
        chat_id="123",
        image_url="https://example.com/photo.png",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[1]
    assert "direct_messages_topic_id" not in call_log[1]


@pytest.mark.asyncio
async def test_send_image_upload_dm_topic_reply_not_found_retry_drops_thread_id(monkeypatch):
    adapter = _make_adapter()
    call_log = []

    async def mock_send_photo(**kwargs):
        call_log.append(dict(kwargs))
        if len(call_log) == 1:
            raise RuntimeError("URL is too large")
        if len(call_log) == 2:
            raise FakeBadRequest("Message to be replied not found")
        return SimpleNamespace(message_id=785)

    class _FakeResponse:
        content = b"image-data"

        def raise_for_status(self):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, _url):
            return _FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(AsyncClient=_FakeAsyncClient),
    )
    adapter._bot = SimpleNamespace(send_photo=mock_send_photo)
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: True)

    result = await adapter.send_image(
        chat_id="123",
        image_url="https://example.com/photo.png",
        metadata={
            "thread_id": "20197",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": "462",
        },
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] == 462
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[1]["reply_to_message_id"] == 462
    assert call_log[1]["message_thread_id"] == 20197
    assert call_log[2]["reply_to_message_id"] is None
    assert "message_thread_id" not in call_log[2]
    assert "direct_messages_topic_id" not in call_log[2]


@pytest.mark.asyncio
async def test_slash_confirm_private_topic_callback_followup_sends_thread_and_reply(monkeypatch):
    adapter = _make_adapter()
    adapter._slash_confirm_state = {"confirm-1": "session-1"}
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=9001)

    async def resolve(_session_key, _confirm_id, _choice):
        return "done"

    from tools import slash_confirm

    monkeypatch.setattr(slash_confirm, "resolve", resolve)
    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    class Query:
        data = "sc:once:confirm-1"
        from_user = SimpleNamespace(id=42, first_name="Alice")
        message = SimpleNamespace(
            chat_id=12345,
            chat=SimpleNamespace(type=_fake_telegram_constants.ChatType.PRIVATE),
            message_thread_id=20197,
            message_id=462,
        )

        async def answer(self, **kwargs):
            return None

        async def edit_message_text(self, **kwargs):
            return None

    await adapter._handle_callback_query(SimpleNamespace(callback_query=Query()), SimpleNamespace())

    assert call_log
    assert call_log[0]["message_thread_id"] == 20197
    assert call_log[0]["reply_to_message_id"] == 462


@pytest.mark.asyncio
async def test_slash_confirm_forum_callback_followup_keeps_existing_thread_behavior(monkeypatch):
    adapter = _make_adapter()
    adapter._slash_confirm_state = {"confirm-1": "session-1"}
    adapter._is_callback_user_authorized = lambda *args, **kwargs: True
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=9001)

    async def resolve(_session_key, _confirm_id, _choice):
        return "done"

    from tools import slash_confirm

    monkeypatch.setattr(slash_confirm, "resolve", resolve)
    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    class Query:
        data = "sc:once:confirm-1"
        from_user = SimpleNamespace(id=42, first_name="Alice")
        message = SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type=_fake_telegram_constants.ChatType.SUPERGROUP),
            message_thread_id=20197,
            message_id=462,
        )

        async def answer(self, **kwargs):
            return None

        async def edit_message_text(self, **kwargs):
            return None

    await adapter._handle_callback_query(SimpleNamespace(callback_query=Query()), SimpleNamespace())

    assert call_log
    assert call_log[0]["message_thread_id"] == 20197
    assert "reply_to_message_id" not in call_log[0]
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_base_send_image_fallback_preserves_metadata():
    """Base image fallback should pass metadata through instead of referencing kwargs."""
    from gateway.platforms.base import BasePlatformAdapter

    class _ConcreteBaseAdapter(BasePlatformAdapter):
        async def connect(self):
            return True

        async def disconnect(self):
            return None

        async def send(self, **kwargs):
            call_log.append(kwargs)
            return SendResult(success=True, message_id="781")

        async def get_chat_info(self, chat_id):
            return None

    call_log = []
    adapter = _ConcreteBaseAdapter(Platform.TELEGRAM, None)
    metadata = {"thread_id": "20197"}

    result = await adapter.send_image(
        chat_id="123",
        image_url="https://example.invalid/image.png",
        metadata=metadata,
    )

    assert result.success is True
    assert call_log[0]["metadata"] is metadata


@pytest.mark.asyncio
async def test_send_raises_on_other_bad_request():
    """Non-thread BadRequest errors should NOT be retried — they fail immediately."""
    adapter = _make_adapter()

    async def mock_send_message(**kwargs):
        raise FakeBadRequest("Chat not found")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
        metadata={"thread_id": "99999"},
    )

    assert result.success is False
    assert "Chat not found" in result.error


@pytest.mark.asyncio
async def test_send_without_thread_id_unaffected():
    """Normal sends without thread_id should work as before."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=100)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is True
    assert len(call_log) == 1
    assert call_log[0]["message_thread_id"] is None


@pytest.mark.asyncio
async def test_send_retries_network_errors_normally():
    """Real transient network errors (not BadRequest) should still be retried."""
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        if attempt[0] < 3:
            raise FakeNetworkError("Connection reset")
        return SimpleNamespace(message_id=200)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is True
    assert attempt[0] == 3  # Two retries then success


@pytest.mark.asyncio
async def test_send_does_not_retry_timeout():
    """TimedOut (subclass of NetworkError) should NOT be retried in send().

    The request may have already been delivered to the user — retrying
    would send duplicate messages.
    """
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        raise FakeTimedOut("Timed out waiting for Telegram response")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="test message",
    )

    assert result.success is False
    assert "Timed out" in result.error
    # CRITICAL: only 1 attempt — no retry for TimedOut
    assert attempt[0] == 1


@pytest.mark.asyncio
async def test_thread_fallback_only_fires_once():
    """After clearing thread_id, subsequent chunks should also use None."""
    adapter = _make_adapter()

    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        tid = kwargs.get("message_thread_id")
        if tid is not None:
            raise FakeBadRequest("Message thread not found")
        return SimpleNamespace(message_id=42)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    # Send a long message that gets split into chunks
    long_msg = "A" * 5000  # Exceeds Telegram's 4096 limit
    result = await adapter.send(
        chat_id="123",
        content=long_msg,
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    # First chunk: attempt with thread → fail → retry without → succeed
    # Second chunk: should use thread_id=None directly (effective_thread_id
    # was cleared per-chunk but the metadata doesn't change between chunks)
    # The key point: the message was delivered despite the invalid thread


@pytest.mark.asyncio
async def test_send_retries_retry_after_errors():
    """Telegram flood control should back off and retry instead of failing fast."""
    adapter = _make_adapter()

    attempt = [0]

    async def mock_send_message(**kwargs):
        attempt[0] += 1
        if attempt[0] == 1:
            raise FakeRetryAfter(2)
        return SimpleNamespace(message_id=300)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(chat_id="123", content="test message")

    assert result.success is True
    assert result.message_id == "300"
    assert attempt[0] == 2
