"""Tests for Telegram topic/thread routing fallbacks.

Supergroup forum topics route with ``message_thread_id``. Hermes-created
private DM topic lanes are different: live Telegram testing showed they only
stay in the expected lane when sends include both the private topic
``message_thread_id`` and a ``reply_to_message_id`` anchor to the triggering
user message. If either anchor is unavailable or rejected, the adapter must
avoid retrying with a partial topic route that can render outside the lane.
"""

import sys
import socket
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
_fake_telegram_ext.InlineQueryHandler = object
_fake_telegram_ext.MessageHandler = object
_fake_telegram_ext.TypeHandler = object
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
    from plugins.platforms.telegram.adapter import TelegramAdapter

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
    adapter._pending_guest_queries = {}
    adapter._guest_only_chats = set()
    adapter._guest_reply_buffer = {}
    adapter._guest_inline_message_ids = {}
    adapter._seen_guest_update_ids = set()
    adapter._last_guest_update_id = 0
    adapter.platform = Platform.TELEGRAM
    return adapter


def test_non_forum_group_reply_thread_id_does_not_fork_session_key():
    """Reply-derived thread ids in ordinary groups must not create topic lanes."""
    import plugins.platforms.telegram.adapter as telegram_mod

    adapter = _make_adapter()
    message = SimpleNamespace(
        text="Done",
        caption=None,
        chat=SimpleNamespace(
            id=-100123,
            type=telegram_mod.ChatType.SUPERGROUP,
            is_forum=False,
            title="Regular group",
        ),
        from_user=SimpleNamespace(id=456, full_name="Alice"),
        message_thread_id=461,
        is_topic_message=False,
        reply_to_message=SimpleNamespace(
            message_id=460,
            text="Please complete the CAPTCHA/login, then reply done.",
            caption=None,
        ),
        message_id=462,
        date=None,
    )

    event = adapter._build_message_event(message, msg_type=MessageType.TEXT)

    assert event.source.chat_id == "-100123"
    assert event.source.chat_type == "group"
    assert event.source.thread_id is None
    assert build_session_key(event.source) == "agent:main:telegram:group:-100123:456"


def test_forum_group_topic_message_preserves_thread_session_key():
    """Real Telegram forum-topic messages should still route by topic id."""
    import plugins.platforms.telegram.adapter as telegram_mod

    adapter = _make_adapter()
    message = SimpleNamespace(
        text="hello from topic",
        caption=None,
        chat=SimpleNamespace(
            id=-100123,
            type=telegram_mod.ChatType.SUPERGROUP,
            is_forum=True,
            title="Forum group",
        ),
        from_user=SimpleNamespace(id=456, full_name="Alice"),
        message_thread_id=17585,
        is_topic_message=True,
        reply_to_message=None,
        message_id=10,
        date=None,
    )

    event = adapter._build_message_event(message, msg_type=MessageType.TEXT)

    assert event.source.chat_id == "-100123"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "17585"
    assert build_session_key(event.source) == "agent:main:telegram:group:-100123:17585"


def test_forum_general_topic_without_message_thread_id_keeps_thread_context():
    """Forum General-topic messages should keep synthetic thread context."""
    import plugins.platforms.telegram.adapter as telegram_mod

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
async def test_private_chat_explicit_thread_id_uses_message_thread_id_without_anchor():
    """Cron-resolved private-chat forum topics route by message_thread_id."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=270454)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="775566675",
        content="cron topic delivery",
        metadata={"thread_id": "270453"},
    )

    assert result.success is True
    assert call_log[0]["reply_to_message_id"] is None
    assert call_log[0]["message_thread_id"] == 270453
    assert "direct_messages_topic_id" not in call_log[0]


@pytest.mark.asyncio
async def test_private_dm_topic_reply_fallback_without_anchor_fails_loud():
    """Anchor-required DM topic fallback must not silently send elsewhere."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        return SimpleNamespace(message_id=270454)

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="775566675",
        content="missing anchor",
        metadata={
            "thread_id": "270453",
            "telegram_dm_topic_reply_fallback": True,
        },
    )

    assert result.success is False
    assert result.retryable is False
    assert result.error == adapter._dm_topic_missing_anchor_error()
    assert call_log == []


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
        "direct_messages_topic_id": "20189",
        "telegram_reply_to_message_id": "462",
    }


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
        "direct_messages_topic_id": "20197",
        "telegram_reply_to_message_id": "463",
    }


@pytest.mark.asyncio
async def test_created_private_topic_thread_not_found_fails_without_root_fallback():
    """Created private-topic sends must not retry into All Messages on stale thread IDs."""
    adapter = _make_adapter()
    call_log = []

    async def mock_send_message(**kwargs):
        call_log.append(dict(kwargs))
        raise FakeBadRequest("Message thread not found")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)

    result = await adapter.send(
        chat_id="123",
        content="created topic message",
        metadata={
            "thread_id": "32343",
            "telegram_dm_topic_created_for_send": True,
        },
    )

    assert result.success is False
    assert "thread not found" in str(result.error).lower()
    assert len(call_log) == 1
    assert call_log[0]["message_thread_id"] == 32343


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

    adapter._bot = SimpleNamespace(send_photo=mock_send_photo)
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: True)
    monkeypatch.setattr(
        url_safety,
        "create_ssrf_safe_async_client",
        lambda **_kwargs: _FakeAsyncClient(),
    )

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
async def test_send_image_upload_fallback_blocks_connect_time_rebind(monkeypatch):
    import httpcore
    from httpcore._backends.auto import AutoBackend
    from gateway.platforms.base import BasePlatformAdapter

    adapter = _make_adapter()
    adapter._bot = SimpleNamespace(
        send_photo=AsyncMock(side_effect=RuntimeError("force URL upload fallback"))
    )

    for proxy_var in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        monkeypatch.delenv(proxy_var, raising=False)

    answers = iter(("93.184.216.34", "169.254.169.254"))

    def fake_getaddrinfo(_host, port, *_args, **_kwargs):
        ip = next(answers)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    connect_attempts = []

    async def fake_connect_tcp(
        _self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        connect_attempts.append((host, port))
        raise httpcore.ConnectError("stop before network")

    async def fake_base_send_image(*_args, **_kwargs):
        return SendResult(success=False, error="fallback")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(AutoBackend, "connect_tcp", fake_connect_tcp)
    monkeypatch.setattr(BasePlatformAdapter, "send_image", fake_base_send_image)

    await adapter.send_image(
        chat_id="123",
        image_url="http://rebind.example/photo.png",
    )

    assert connect_attempts == []


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
        async def connect(self, *, is_reconnect: bool = False):
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
        chat_id="-100123",
        content=long_msg,
        metadata={"thread_id": "99999"},
    )

    assert result.success is True
    # First chunk: attempt with thread → fail → retry without → succeed
    # Second chunk: should use thread_id=None directly (effective_thread_id
    # was cleared per-chunk but the metadata doesn't change between chunks)
    # The key point: the message was delivered despite the invalid thread


