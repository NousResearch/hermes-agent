"""Regression tests for Telegram forum-topic media routing.

These tests lock two invariants that together explain and fix the reported bug
where photos and files sent to a bound group forum topic never reached the
agent while plain text in the same topic worked:

1. Routing symmetry: text, photos, and document/image files sent to the same
   forum topic resolve the same ``thread_id`` and session key, so media always
   routes to the same session as text in that topic.

2. Active-session bypass: a forum topic that already has an active session
   accepts follow-up messages (text and media, even without a caption mention)
   without a fresh @mention, matching Slack thread behavior. Idle topics still
   require a mention, so the bot stays quiet on unrelated group chatter.

The mention gate, not the routing layer, was dropping uncaptioned media; these
tests cover both the gate decision and the end-to-end media path.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed (mirrors test_telegram_documents)
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.session import SessionSource, build_session_key  # noqa: E402


GROUP_ID = -1003917904618  # "D4J Powerhouse" supergroup id from the report
TOPIC_ID = 14              # "Samson" forum topic id from the report
OTHER_TOPIC_ID = 99
BOT_USERNAME = "hermes_bot"


# ---------------------------------------------------------------------------
# Fakes / builders
# ---------------------------------------------------------------------------

class _FakeSessionStore:
    """Minimal session store exposing the read surface the gate touches.

    Carries a ``config`` with the session-isolation flags, mirroring the real
    SessionStore: the gate reads these to build a lookup key that matches how
    the store keyed the entry.
    """

    def __init__(self, keys=(), *, group_sessions_per_user=True, thread_sessions_per_user=False):
        self._entries = set(keys)
        self.ensure_loaded_called = False
        self.config = SimpleNamespace(
            group_sessions_per_user=group_sessions_per_user,
            thread_sessions_per_user=thread_sessions_per_user,
        )

    def _ensure_loaded(self):
        self.ensure_loaded_called = True


class _RaisingSessionStore:
    """Session store whose load step raises, to prove the gate fails safe."""

    _entries: set = set()

    def _ensure_loaded(self):
        raise RuntimeError("store unavailable")


def _make_adapter(*, require_mention=True, session_keys=(), session_store="fake"):
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "require_mention": require_mention,
            "free_response_chats": [],
            "mention_patterns": [],
            "allowed_topics": [],
            "allowed_chats": [],
            "group_allowed_chats": [],
        },
    )
    adapter = TelegramAdapter(config)
    adapter._bot = SimpleNamespace(id=999, username=BOT_USERNAME)
    if session_store == "fake":
        adapter._session_store = _FakeSessionStore(session_keys)
    elif session_store == "raising":
        adapter._session_store = _RaisingSessionStore()
    else:
        adapter._session_store = session_store
    # The empty-allowlist callback auth path is fail-closed; bypass it so the
    # trigger logic under test runs for fake senders.
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    return adapter


def _forum_message(
    *,
    text=None,
    caption=None,
    thread_id=TOPIC_ID,
    chat_id=GROUP_ID,
    from_user_id=111,
    entities=None,
    caption_entities=None,
    message_id=42,
):
    """A message posted into a supergroup forum topic."""
    return SimpleNamespace(
        message_id=message_id,
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=thread_id,
        is_topic_message=thread_id is not None,
        chat=SimpleNamespace(
            id=chat_id,
            type="supergroup",
            title="D4J Powerhouse",
            is_forum=True,
        ),
        from_user=SimpleNamespace(
            id=from_user_id,
            full_name="Alice Example",
            first_name="Alice",
        ),
        reply_to_message=None,
        quote=None,
        date=None,
    )


def _dm_message(*, caption=None, from_user_id=111):
    return SimpleNamespace(
        message_id=43,
        text=caption,
        caption=caption,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(
            id=from_user_id,
            type="private",
            title=None,
            full_name="Alice Example",
            is_forum=False,
        ),
        from_user=SimpleNamespace(
            id=from_user_id,
            full_name="Alice Example",
            first_name="Alice",
        ),
        reply_to_message=None,
        quote=None,
        date=None,
    )


def _topic_session_key(
    *,
    thread_id=TOPIC_ID,
    chat_id=GROUP_ID,
    user_id=111,
    group_sessions_per_user=True,
    thread_sessions_per_user=False,
):
    """The session key the store would use for a forum-topic message.

    Defaults match the store defaults; pass explicit flags to mirror a store
    configured with non-default isolation.
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=str(chat_id),
        chat_type="group",
        user_id=str(user_id),
        thread_id=str(thread_id),
    )
    return build_session_key(
        source,
        group_sessions_per_user=group_sessions_per_user,
        thread_sessions_per_user=thread_sessions_per_user,
    )


def _mention_entity(text, mention="@" + BOT_USERNAME):
    offset = text.index(mention)
    return SimpleNamespace(type="mention", offset=offset, length=len(mention))


def _file_obj(data=b"image-bytes", file_path="photos/file.jpg"):
    f = AsyncMock()
    f.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    f.file_path = file_path
    return f


def _photo_size(file_obj=None):
    return SimpleNamespace(get_file=AsyncMock(return_value=file_obj or _file_obj()))


# ---------------------------------------------------------------------------
# 1. Routing symmetry: text / photo / image-document resolve the same session
# ---------------------------------------------------------------------------

def test_text_photo_and_document_resolve_same_topic_thread_id():
    adapter = _make_adapter()

    text_event = adapter._build_message_event(_forum_message(text="hi"), MessageType.TEXT)
    photo_event = adapter._build_message_event(_forum_message(caption=None), MessageType.PHOTO)
    doc_event = adapter._build_message_event(_forum_message(caption="see file"), MessageType.DOCUMENT)

    assert text_event.source.thread_id == str(TOPIC_ID)
    assert photo_event.source.thread_id == str(TOPIC_ID)
    assert doc_event.source.thread_id == str(TOPIC_ID)


def test_text_photo_and_document_build_identical_session_key():
    adapter = _make_adapter()

    def key(event):
        return build_session_key(
            event.source,
            group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
        )

    text_key = key(adapter._build_message_event(_forum_message(text="hi"), MessageType.TEXT))
    photo_key = key(adapter._build_message_event(_forum_message(), MessageType.PHOTO))
    doc_key = key(adapter._build_message_event(_forum_message(caption="x"), MessageType.DOCUMENT))

    assert text_key == photo_key == doc_key
    # And it is the topic-scoped key, not the group-root key.
    assert str(TOPIC_ID) in text_key


def test_dm_photo_routes_to_dm_session_unchanged():
    adapter = _make_adapter()
    event = adapter._build_message_event(_dm_message(), MessageType.PHOTO)
    assert event.source.chat_type == "dm"
    assert event.source.thread_id is None


# ---------------------------------------------------------------------------
# 2. Forum-topic thread-id helper (single source of truth)
# ---------------------------------------------------------------------------

def test_effective_message_thread_id_helper():
    adapter = _make_adapter()
    # Real forum topic message.
    assert adapter._effective_message_thread_id(_forum_message(text="hi")) == str(TOPIC_ID)
    # Forum group with no explicit topic falls back to the General topic.
    no_topic = _forum_message(text="hi", thread_id=None)
    assert adapter._effective_message_thread_id(no_topic) == adapter._GENERAL_TOPIC_THREAD_ID
    # Plain (non-forum) group reply anchor is dropped.
    plain = _forum_message(text="hi")
    plain.is_topic_message = False
    plain.chat.is_forum = False
    assert adapter._effective_message_thread_id(plain) is None


# ---------------------------------------------------------------------------
# 3. Active-session bypass at the gate (the fix)
# ---------------------------------------------------------------------------

def test_uncaptioned_media_dropped_in_topic_without_active_session():
    """Baseline: with require_mention and no active session, captionless media
    is not processed (this is the reported drop)."""
    adapter = _make_adapter(require_mention=True, session_keys=())
    # No caption, no mention.
    assert adapter._should_process_message(_forum_message(caption=None)) is False
    assert adapter._should_process_message(_forum_message(text=None)) is False


def test_uncaptioned_media_processed_when_topic_has_active_session():
    adapter = _make_adapter(require_mention=True)
    adapter._session_store = _FakeSessionStore({_topic_session_key()})

    # Uncaptioned photo / document / text all pass once the topic has a session.
    assert adapter._should_process_message(_forum_message(caption=None)) is True
    assert adapter._should_process_message(_forum_message(text=None)) is True
    assert adapter._session_store.ensure_loaded_called is True


def test_active_session_accepts_every_media_shape():
    """The gate is media-type agnostic: once the topic has a session, a photo
    with a plain caption (no mention), an uncaptioned photo, and a document all
    pass. Without a session each is still dropped under require_mention."""
    active = _make_adapter(require_mention=True)
    active._session_store = _FakeSessionStore({_topic_session_key()})

    captioned_photo = _forum_message(caption="here is the screenshot")
    uncaptioned_photo = _forum_message(caption=None)
    document = _forum_message(caption=None)  # a file is gated identically

    assert active._should_process_message(captioned_photo) is True
    assert active._should_process_message(uncaptioned_photo) is True
    assert active._should_process_message(document) is True

    idle = _make_adapter(require_mention=True, session_keys=())
    assert idle._should_process_message(_forum_message(caption="here is the screenshot")) is False
    assert idle._should_process_message(_forum_message(caption=None)) is False


def test_active_session_bypass_is_scoped_to_the_matching_topic():
    adapter = _make_adapter(require_mention=True)
    # Only TOPIC_ID has an active session.
    adapter._session_store = _FakeSessionStore({_topic_session_key()})

    # A captionless message in a different topic is still dropped.
    other = _forum_message(caption=None, thread_id=OTHER_TOPIC_ID)
    assert adapter._should_process_message(other) is False


def test_mention_still_processed_without_active_session():
    """The active-session bypass is additive: an explicit mention still works
    even when no session exists yet (starting a conversation)."""
    adapter = _make_adapter(require_mention=True, session_keys=())
    text = "hey @" + BOT_USERNAME + " look"
    msg = _forum_message(text=text, entities=[_mention_entity(text)])
    assert adapter._should_process_message(msg) is True


def test_active_session_lookup_uses_store_config_isolation_flags():
    """The lookup key must use the store's isolation flags, not the adapter's
    PlatformConfig.extra (which never carries them). With
    thread_sessions_per_user=True the store keys the entry WITH the user id, so
    a gate that read the adapter extra (defaulting the flag to False) would
    build a user-less key and miss the entry."""
    adapter = _make_adapter(require_mention=True)
    user_id = 111
    keyed = _topic_session_key(user_id=user_id, thread_sessions_per_user=True)
    adapter._session_store = _FakeSessionStore({keyed}, thread_sessions_per_user=True)

    msg = _forum_message(caption=None, from_user_id=user_id)
    assert adapter._should_process_message(msg) is True

    # A different user's uncaptioned message is dropped: per-user keying means
    # only the user with the active session bypasses the mention requirement.
    other_user = _forum_message(caption=None, from_user_id=222)
    assert adapter._should_process_message(other_user) is False


def test_gate_fails_safe_when_session_store_missing_or_raising():
    no_store = _make_adapter(require_mention=True, session_store=None)
    assert no_store._should_process_message(_forum_message(caption=None)) is False

    raising = _make_adapter(require_mention=True, session_store="raising")
    assert raising._should_process_message(_forum_message(caption=None)) is False


def test_active_session_helper_ignores_non_forum_group_messages():
    adapter = _make_adapter(require_mention=True)
    adapter._session_store = _FakeSessionStore({_topic_session_key()})
    plain = _forum_message(caption=None)
    plain.is_topic_message = False
    plain.chat.is_forum = False
    # No real topic thread -> helper returns False even if a key happens to exist.
    assert adapter._forum_topic_has_active_session(plain) is False


# ---------------------------------------------------------------------------
# 4. End-to-end media path: caption preserved, attachment attached, topic kept
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_captioned_topic_photo_preserves_caption_and_attachment():
    adapter = _make_adapter(require_mention=True)
    # Caption mentions the bot so the message passes the gate on its own.
    caption = "@" + BOT_USERNAME + " look at this"
    msg = _forum_message(
        caption=caption,
        caption_entities=[_mention_entity(caption)],
    )
    msg.photo = [_photo_size()]
    msg.video = msg.audio = msg.voice = msg.sticker = msg.document = None
    msg.media_group_id = None
    update = SimpleNamespace(message=msg, update_id=1)

    captured = []
    adapter._enqueue_photo_event = lambda key, event: captured.append((key, event))

    with patch(
        "plugins.platforms.telegram.adapter.cache_image_from_bytes",
        return_value="/cache/user-photo.jpg",
    ):
        await adapter._handle_media_message(update, None)

    assert captured, "captioned topic photo should be enqueued, not dropped"
    _key, event = captured[0]
    assert event.media_urls == ["/cache/user-photo.jpg"]
    assert event.media_types == ["image/jpg"]
    assert "look at this" in (event.text or "")
    assert event.source.thread_id == str(TOPIC_ID)


@pytest.mark.asyncio
async def test_uncaptioned_topic_photo_reaches_pipeline_when_session_active():
    adapter = _make_adapter(require_mention=True)
    adapter._session_store = _FakeSessionStore({_topic_session_key()})

    msg = _forum_message(caption=None)
    msg.photo = [_photo_size()]
    msg.video = msg.audio = msg.voice = msg.sticker = msg.document = None
    msg.media_group_id = None
    update = SimpleNamespace(message=msg, update_id=2)

    captured = []
    adapter._enqueue_photo_event = lambda key, event: captured.append((key, event))

    with patch(
        "plugins.platforms.telegram.adapter.cache_image_from_bytes",
        return_value="/cache/user-photo.jpg",
    ):
        await adapter._handle_media_message(update, None)

    assert captured, "uncaptioned photo in an active-session topic must not be dropped"
    _key, event = captured[0]
    assert event.media_urls == ["/cache/user-photo.jpg"]
    assert event.source.thread_id == str(TOPIC_ID)


@pytest.mark.asyncio
async def test_uncaptioned_topic_document_reaches_pipeline_when_session_active():
    adapter = _make_adapter(require_mention=True)
    adapter._session_store = _FakeSessionStore({_topic_session_key()})
    adapter.handle_message = AsyncMock()

    file_obj = AsyncMock()
    file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"%PDF-1.4 ..."))
    file_obj.file_path = "documents/report.pdf"
    document = MagicMock()
    document.file_name = "report.pdf"
    document.mime_type = "application/pdf"
    document.file_size = 2048
    document.get_file = AsyncMock(return_value=file_obj)

    msg = _forum_message(caption=None)
    msg.photo = msg.video = msg.audio = msg.voice = msg.sticker = None
    msg.document = document
    msg.media_group_id = None
    update = SimpleNamespace(message=msg, update_id=3)

    with patch(
        "plugins.platforms.telegram.adapter.cache_document_from_bytes",
        return_value="/cache/report.pdf",
    ):
        await adapter._handle_media_message(update, None)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/cache/report.pdf"]
    assert event.media_types == ["application/pdf"]
    assert event.source.thread_id == str(TOPIC_ID)


# ---------------------------------------------------------------------------
# 5. Album / media-group keeps the topic thread and coalesces into one event
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_topic_album_coalesces_into_one_event_keeping_thread():
    adapter = _make_adapter(require_mention=True)
    adapter.handle_message = AsyncMock()

    ev1 = adapter._build_message_event(_forum_message(caption="album"), MessageType.PHOTO)
    ev1.media_urls = ["/cache/a.jpg"]
    ev1.media_types = ["image/jpg"]
    ev2 = adapter._build_message_event(_forum_message(), MessageType.PHOTO)
    ev2.media_urls = ["/cache/b.jpg"]
    ev2.media_types = ["image/jpg"]

    gid = "group-1"
    await adapter._queue_media_group_event(gid, ev1)
    await adapter._queue_media_group_event(gid, ev2)

    merged = adapter._media_group_events[gid]
    assert merged.media_urls == ["/cache/a.jpg", "/cache/b.jpg"]
    assert merged.source.thread_id == str(TOPIC_ID)

    # Cancel the pending flush task so the test does not leak a timer.
    task = adapter._media_group_tasks.get(gid)
    if task:
        task.cancel()
