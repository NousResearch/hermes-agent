"""Tests for multi-sender attribution in merge_pending_message_event.

When the gateway is busy and queues messages for the next turn, multiple
participants in a shared-session group can each contribute messages that get
merged into a single follow-up turn. The merge MUST preserve sender
attribution per segment, otherwise the agent will misattribute later messages
to whoever sent the first one.

These tests reproduce the bug described in the 玉子燒廚房 (Tamagoyaki Kitchen)
scenario, where 玉青's nutrition logs and 子家's nutrition logs were getting
merged into a single text and the agent could not tell who said what.
"""

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    merge_pending_message_event,
)
from gateway.session import SessionSource, build_session_key


def _src(user_id: str, user_name: str) -> SessionSource:
    """Helper: shared group source (same chat, different participants)."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1003964576906",
        chat_type="group",
        user_id=user_id,
        user_name=user_name,
    )


def _key() -> str:
    # Shared session: same key regardless of user_id.
    return build_session_key(
        _src("anyone", "anyone"),
        group_sessions_per_user=False,
    )


# ---------------------------------------------------------------------------
# Multi-sender TEXT merge
# ---------------------------------------------------------------------------


def test_text_merge_preserves_each_sender_when_senders_differ():
    """玉青 sends, then 子家 sends: merged text must mark 子家's segment."""
    pending = {}
    yuqing = MessageEvent(
        text="米粉跟醬包各2",
        message_type=MessageType.TEXT,
        source=_src("1285712441", "玉青"),
    )
    zijia = MessageEvent(
        text="水煮蛋一顆",
        message_type=MessageType.TEXT,
        source=_src("8682281996", "子家"),
    )

    merge_pending_message_event(pending, _key(), yuqing, merge_text=True)
    merge_pending_message_event(pending, _key(), zijia, merge_text=True)

    merged = pending[_key()]
    # 子家's segment must carry [子家] prefix so the agent does not mistake
    # it for 玉青's message.
    assert "[子家] 水煮蛋一顆" in merged.text
    assert "米粉跟醬包各2" in merged.text


def test_text_merge_does_not_double_prefix_same_sender():
    """If the same sender sends two follow-ups, do not insert a sender prefix.

    The shared-session prefix is added once at the top by
    _prepare_inbound_message_text; merge should not add its own prefix when
    sender is unchanged.
    """
    pending = {}
    src = _src("1285712441", "玉青")
    first = MessageEvent(text="A", message_type=MessageType.TEXT, source=src)
    second = MessageEvent(text="B", message_type=MessageType.TEXT, source=src)

    merge_pending_message_event(pending, _key(), first, merge_text=True)
    merge_pending_message_event(pending, _key(), second, merge_text=True)

    merged = pending[_key()]
    # Same sender → plain newline join, no [玉青] prefix inserted by merge.
    assert merged.text == "A\nB"


def test_text_merge_three_alternating_senders():
    """玉青, 子家, 玉青 again: each segment after the first carries its sender."""
    pending = {}
    a1 = MessageEvent(text="早餐燕麥", message_type=MessageType.TEXT, source=_src("1", "玉青"))
    b = MessageEvent(text="咖啡一杯", message_type=MessageType.TEXT, source=_src("2", "子家"))
    a2 = MessageEvent(text="加蘋果", message_type=MessageType.TEXT, source=_src("1", "玉青"))

    merge_pending_message_event(pending, _key(), a1, merge_text=True)
    merge_pending_message_event(pending, _key(), b, merge_text=True)
    merge_pending_message_event(pending, _key(), a2, merge_text=True)

    merged = pending[_key()]
    assert "早餐燕麥" in merged.text
    assert "[子家] 咖啡一杯" in merged.text
    # When sender flips back, prefix again so it's clear who's talking.
    assert "[玉青] 加蘋果" in merged.text


# ---------------------------------------------------------------------------
# Multi-sender PHOTO merge
# ---------------------------------------------------------------------------


def test_photo_burst_merge_with_different_senders_marks_caption():
    """Photo from 玉青 then photo from 子家: caption must mark 子家's caption."""
    pending = {}
    yuqing_photo = MessageEvent(
        text="便當1",
        message_type=MessageType.PHOTO,
        source=_src("1285712441", "玉青"),
        media_urls=["/tmp/img_yuqing.jpg"],
        media_types=["image/jpeg"],
    )
    zijia_photo = MessageEvent(
        text="便當2",
        message_type=MessageType.PHOTO,
        source=_src("8682281996", "子家"),
        media_urls=["/tmp/img_zijia.jpg"],
        media_types=["image/jpeg"],
    )

    merge_pending_message_event(pending, _key(), yuqing_photo, merge_text=True)
    merge_pending_message_event(pending, _key(), zijia_photo, merge_text=True)

    merged = pending[_key()]
    assert merged.message_type == MessageType.PHOTO
    # Both images carried over.
    assert "/tmp/img_yuqing.jpg" in merged.media_urls
    assert "/tmp/img_zijia.jpg" in merged.media_urls
    # 子家's caption must be marked.
    assert "[子家] 便當2" in merged.text


def test_text_then_photo_from_different_sender_marks_photo_caption():
    """玉青 sends text, then 子家 sends a photo: photo caption must mark 子家."""
    pending = {}
    text_event = MessageEvent(
        text="幫我看",
        message_type=MessageType.TEXT,
        source=_src("1285712441", "玉青"),
    )
    photo_event = MessageEvent(
        text="這張",
        message_type=MessageType.PHOTO,
        source=_src("8682281996", "子家"),
        media_urls=["/tmp/zijia.jpg"],
        media_types=["image/jpeg"],
    )

    merge_pending_message_event(pending, _key(), text_event, merge_text=True)
    merge_pending_message_event(pending, _key(), photo_event, merge_text=True)

    merged = pending[_key()]
    assert merged.message_type == MessageType.PHOTO
    assert "幫我看" in merged.text
    assert "[子家] 這張" in merged.text
