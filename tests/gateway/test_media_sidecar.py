"""Tests for media sidecar JSON files.

When the Telegram adapter caches a photo/voice/document, it writes a JSON
sidecar next to the cached file recording who sent it. This lets the agent
later answer "who sent that photo?" by reading the sidecar — Telegram does
not write any sender metadata into the media bytes themselves and our cache
filenames are random uuid hashes.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import write_media_sidecar, read_media_sidecar


# --- pure helpers ------------------------------------------------------------


def test_write_and_read_sidecar_roundtrip(tmp_path):
    media = tmp_path / "img_abc.jpg"
    media.write_bytes(b"\xff\xd8\xff")  # not used by the helper, just realistic

    sidecar = write_media_sidecar(
        str(media),
        {
            "platform": "telegram",
            "from_id": "8682281996",
            "from_name": "TZ Chia Tseng",
            "message_id": 42,
            "chat_id": "-1003964576906",
        },
    )
    assert sidecar == f"{media}.json"
    assert os.path.exists(sidecar)

    payload = read_media_sidecar(str(media))
    assert payload is not None
    assert payload["from_id"] == "8682281996"
    assert payload["from_name"] == "TZ Chia Tseng"
    assert payload["message_id"] == 42
    assert payload["chat_id"] == "-1003964576906"
    assert "saved_at" in payload  # auto-stamped


def test_read_sidecar_returns_none_when_missing(tmp_path):
    assert read_media_sidecar(str(tmp_path / "nonexistent.jpg")) is None


def test_write_sidecar_handles_datetime_objects(tmp_path):
    media = tmp_path / "img_x.jpg"
    media.write_bytes(b"\xff\xd8\xff")
    write_media_sidecar(
        str(media),
        {"date": dt.datetime(2026, 4, 27, 12, 51, tzinfo=dt.timezone.utc)},
    )
    payload = read_media_sidecar(str(media))
    assert payload is not None
    # default=str fallback handles non-serialisable types
    assert "2026-04-27" in payload["date"]


# --- integration: TelegramAdapter._record_media_sender -----------------------


def _make_fake_message(
    *,
    msg_id=99,
    user_id=8682281996,
    user_name="TZ Chia Tseng",
    username="tz_chia",
    chat_id=-1003964576906,
    chat_type="supergroup",
    chat_title="玉子燒廚房",
    thread_id=None,
    caption=None,
    text=None,
    date=None,
):
    from_user = SimpleNamespace(
        id=user_id,
        full_name=user_name,
        first_name=user_name.split()[0] if user_name else "",
        username=username,
        is_bot=False,
    )
    chat = SimpleNamespace(id=chat_id, type=chat_type, title=chat_title, username=None)
    return SimpleNamespace(
        message_id=msg_id,
        from_user=from_user,
        chat=chat,
        message_thread_id=thread_id,
        date=date or dt.datetime(2026, 4, 27, 12, 51, tzinfo=dt.timezone.utc),
        caption=caption,
        text=text,
    )


def _make_adapter():
    """Build a TelegramAdapter without running its full __init__."""
    from gateway.platforms.telegram import TelegramAdapter
    return object.__new__(TelegramAdapter)


def test_record_media_sender_writes_full_metadata(tmp_path):
    adapter = _make_adapter()
    media = tmp_path / "img_xyz.jpg"
    media.write_bytes(b"\xff\xd8\xff")

    message = _make_fake_message(caption="my bento")
    adapter._record_media_sender(message, str(media))

    payload = read_media_sidecar(str(media))
    assert payload["platform"] == "telegram"
    assert payload["from_id"] == "8682281996"
    assert payload["from_name"] == "TZ Chia Tseng"
    assert payload["from_username"] == "tz_chia"
    assert payload["from_is_bot"] is False
    assert payload["chat_id"] == "-1003964576906"
    assert payload["chat_type"] == "supergroup"
    assert payload["chat_title"] == "玉子燒廚房"
    assert payload["message_id"] == 99
    assert "2026-04-27" in payload["date"]
    assert payload["caption"] == "my bento"
    assert payload["is_quoted_reply"] is False


def test_record_media_sender_marks_quoted_reply(tmp_path):
    adapter = _make_adapter()
    media = tmp_path / "img_q.jpg"
    media.write_bytes(b"\xff\xd8\xff")

    quoted_msg = _make_fake_message(
        msg_id=42,
        user_id=1285712441,
        user_name="王玉青",
        username="wang_yuching",
        text="here's lunch",
    )
    adapter._record_media_sender(quoted_msg, str(media), is_quoted_reply=True)

    payload = read_media_sidecar(str(media))
    assert payload["is_quoted_reply"] is True
    assert payload["from_id"] == "1285712441"
    assert payload["from_name"] == "王玉青"
    assert payload["quoted_text"] == "here's lunch"


def test_record_media_sender_swallows_errors(tmp_path):
    """No exception should escape even if the message has no usable fields."""
    adapter = _make_adapter()
    media = tmp_path / "img_e.jpg"
    media.write_bytes(b"\xff\xd8\xff")

    # Empty SimpleNamespace — most getattr() calls return None
    bare_message = SimpleNamespace()
    adapter._record_media_sender(bare_message, str(media))  # must not raise

    payload = read_media_sidecar(str(media))
    assert payload is not None
    assert payload["platform"] == "telegram"


def test_record_media_sender_noop_on_empty_path(tmp_path):
    adapter = _make_adapter()
    msg = _make_fake_message()
    # Should not raise, should not write anything
    adapter._record_media_sender(msg, "")
    adapter._record_media_sender(msg, None)  # type: ignore[arg-type]


def test_record_media_sender_noop_on_no_message(tmp_path):
    adapter = _make_adapter()
    media = tmp_path / "img_n.jpg"
    media.write_bytes(b"\xff\xd8\xff")
    adapter._record_media_sender(None, str(media))  # type: ignore[arg-type]
    assert not (tmp_path / "img_n.jpg.json").exists()
