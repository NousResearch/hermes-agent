"""A short Telegram question and the photo sent right after it reach the agent as one turn (#33270).

The text batch flushes after a short quiet period, while the photo is still downloading and then
waits in its own burst batch, so the agent used to answer the question without the image. Driven
through the real ``_handle_text_message`` and ``_handle_media_message`` in production order: the
photo's handler starts once the text's has returned, and its download outlasts the text delay.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_documents import _make_file_obj, _make_message, _make_photo
from tests.waiting import async_wait_until


def _update(msg):
    return SimpleNamespace(update_id=1, message=msg, effective_message=msg, channel_post=None,
                           business_message=None, edited_business_message=None)


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t"))
    adapter.handle_message = AsyncMock()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    adapter._media_batch_delay_seconds = 0.1
    return adapter


def _text(text: str, user_id: int = 1):
    msg = _make_message()
    msg.text, msg.caption, msg.from_user.id = text, None, user_id
    return msg


def _slow_photo(release: asyncio.Event, user_id: int = 1):
    file_obj = _make_file_obj(b"post")

    async def download():
        await release.wait()
        return bytearray(b"post")
    file_obj.download_as_bytearray = AsyncMock(side_effect=download)
    msg = _make_message(photo=[_make_photo(file_obj)])
    msg.from_user.id = user_id
    return msg


async def _text_then_photo(adapter, *, photo_user: int):
    release = asyncio.Event()
    with patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", new=AsyncMock(return_value="/tmp/post.jpg")):
        await adapter._handle_text_message(_update(_text("Is this a real study?")), MagicMock())
        photo = asyncio.create_task(adapter._handle_media_message(_update(_slow_photo(release, photo_user)), MagicMock()))
        await asyncio.sleep(0.6)  # health: allow HX013 -- a download outlasting the 0.18 s text delay IS the race
        release.set()
        await photo
        await async_wait_until(lambda: any(c.args[0].media_urls for c in adapter.handle_message.await_args_list), timeout=5)
    return [c.args[0] for c in adapter.handle_message.await_args_list]


@pytest.mark.asyncio
async def test_question_then_photo_is_one_turn_with_the_image():
    events = await _text_then_photo(_adapter(), photo_user=1)
    assert len(events) == 1
    assert events[0].media_urls == ["/tmp/post.jpg"]
    assert events[0].text == "Is this a real study?"


@pytest.mark.asyncio
async def test_another_senders_photo_does_not_take_the_question():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _msg: True
    events = await _text_then_photo(adapter, photo_user=2)
    assert [e.text for e in events] == ["Is this a real study?", ""]
    assert [e.media_urls for e in events] == [[], ["/tmp/post.jpg"]]


@pytest.mark.asyncio
async def test_observed_group_question_then_photo_is_one_turn():
    """An observed group moves the sender off ``source.user_id``; the photo still claims the question."""
    adapter = _adapter()
    adapter._is_group_chat = lambda _msg: True
    adapter._gate_or_observe = lambda *_a, **_kw: True
    adapter._should_process_message = lambda *_a, **_kw: True
    adapter._telegram_observe_unmentioned_group_messages = lambda: True
    adapter._telegram_observe_allowed_chats = lambda: {"100"}
    events = await _text_then_photo(adapter, photo_user=1)
    assert len(events) == 1
    assert events[0].media_urls == ["/tmp/post.jpg"]
    assert "Is this a real study?" in events[0].text


@pytest.mark.asyncio
async def test_photo_in_another_topic_does_not_hold_the_question():
    """Same user, question in topic A, photo in topic B: the question flushes on time, not after B's download."""
    adapter = _adapter()
    release = asyncio.Event()
    question, photo_msg = _text("Is this a real study?"), _slow_photo(release)
    question.message_thread_id, photo_msg.message_thread_id = 10, 20
    with patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", new=AsyncMock(return_value="/tmp/post.jpg")):
        await adapter._handle_text_message(_update(question), MagicMock())
        photo = asyncio.create_task(adapter._handle_media_message(_update(photo_msg), MagicMock()))
        try:
            await async_wait_until(lambda: adapter.handle_message.await_count >= 1, timeout=2)
        finally:
            release.set()
            await photo
        first = adapter.handle_message.await_args_list[0].args[0]
        assert (first.text, first.media_urls, first.source.thread_id) == ("Is this a real study?", [], "10")
        await async_wait_until(lambda: adapter.handle_message.await_count >= 2, timeout=5)
    assert adapter.handle_message.await_args_list[1].args[0].media_urls == ["/tmp/post.jpg"]


@pytest.mark.asyncio
async def test_observe_only_photo_does_not_hold_the_question():
    """A photo that only reaches the observed transcript (not addressed to the bot) never joins a
    turn, so it must not hold the sender's question while it downloads."""
    adapter = _adapter()
    release = asyncio.Event()
    adapter._gate_or_observe = lambda *_a, **_kw: True  # the question addresses the bot
    adapter._should_process_message = lambda *_a, **_kw: False  # the photo does not
    adapter._should_observe_unmentioned_group_message = lambda _msg: True
    adapter._observe_unmentioned_group_message = MagicMock()

    async def slow_observe_cache(*_a, **_kw):
        await release.wait()
    adapter._cache_observed_media = slow_observe_cache
    await adapter._handle_text_message(_update(_text("Is this a real study?")), MagicMock())
    photo = asyncio.create_task(adapter._handle_media_message(_update(_slow_photo(release)), MagicMock()))
    try:
        await async_wait_until(lambda: adapter.handle_message.await_count >= 1, timeout=2)
    finally:
        release.set()
        await photo
    assert adapter.handle_message.await_args_list[0].args[0].text == "Is this a real study?"


@pytest.mark.asyncio
@pytest.mark.parametrize("album", [False, True], ids=["single", "album"])
async def test_recovered_question_keeps_its_topic_when_a_newer_topic_binds_during_debounce(tmp_path, album):
    """Topic-mode DM: the threadless question is recovered into topic 10. The photo absorbs it, and
    while the photo debounces another topic (20) becomes the newest binding. The merged turn must
    still run in topic 10, where the question was routed, not re-derive topic 20 at dispatch."""
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key
    from hermes_state import SessionDB
    from tests.gateway.test_telegram_topic_mode import _make_runner

    db = SessionDB(db_path=tmp_path / "state.db")
    db.enable_telegram_topic_mode(chat_id="100", user_id="1")

    def bind(thread_id):
        db.create_session(session_id=f"s-{thread_id}", source="telegram", user_id="1")
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="100", user_id="1", chat_type="dm", thread_id=thread_id)
        db.bind_telegram_topic(chat_id="100", thread_id=thread_id, user_id="1",
                               session_key=build_session_key(source), session_id=f"s-{thread_id}")
    bind("10")
    adapter = _adapter()
    del adapter.handle_message  # real Telegram + base ingress, up to the session-processing seam
    adapter.set_message_handler(AsyncMock())
    adapter.set_topic_recovery_fn(_make_runner(session_db=db)._recover_telegram_topic_thread_id)
    adapter._get_dm_topic_info = lambda *_a: {}
    routed = []
    adapter._start_session_processing = lambda event, _key: routed.append((event.text, event.source.thread_id, event.media_urls)) or True
    release = asyncio.Event()
    photo_msg = _slow_photo(release)
    photo_msg.media_group_id = "album-1" if album else None
    try:
        with patch("plugins.platforms.telegram.adapter.cache_image_from_bytes_async", new=AsyncMock(return_value="/tmp/post.jpg")):
            await adapter._handle_text_message(_update(_text("Is this a real study?")), MagicMock())
            photo = asyncio.create_task(adapter._handle_media_message(_update(photo_msg), MagicMock()))
            await asyncio.sleep(0.4)  # health: allow HX013 -- a download outlasting the 0.18 s text delay IS the race
            release.set()
            await photo
            bind("20")  # another topic's turn refreshes the newest binding while the photo debounces
            await async_wait_until(lambda: routed, timeout=5)
    finally:
        db.close()
    assert routed == [("Is this a real study?", "10", ["/tmp/post.jpg"])]
