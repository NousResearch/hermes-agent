"""Telegram album photos land in timestamped albums/<ts>-<key>/ folders."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.platforms import base as base
from plugins.platforms.telegram.adapter import TelegramAdapter

JPEG = b"\xff\xd8\xff fake jpeg data"


@pytest.fixture
def image_cache(tmp_path, monkeypatch):
    d = tmp_path / "img"
    d.mkdir()
    monkeypatch.setattr(base, "IMAGE_CACHE_DIR", d)
    base.clear_inbound_album_dirs()
    yield d
    base.clear_inbound_album_dirs()


def test_same_album_key_shares_folder(image_cache):
    p1 = base.cache_image_from_bytes(JPEG, ".jpg", album_key="mg111")
    p2 = base.cache_image_from_bytes(JPEG, ".jpg", album_key="mg111")
    a = Path(p1)
    b = Path(p2)
    assert a.parent == b.parent
    assert a.parent.parent.name == "albums"
    assert a.name == "photo-01.jpg"
    assert b.name == "photo-02.jpg"
    assert a.parent.name.endswith("-mg111")


def test_different_keys_do_not_mix(image_cache):
    p1 = base.cache_image_from_bytes(JPEG, ".jpg", album_key="mgAAA")
    p2 = base.cache_image_from_bytes(JPEG, ".jpg", album_key="mgBBB")
    assert Path(p1).parent != Path(p2).parent
    assert Path(p1).name == "photo-01.jpg"
    assert Path(p2).name == "photo-01.jpg"


def test_no_album_key_still_flat_img(image_cache):
    p = base.cache_image_from_bytes(JPEG, ".jpg")
    assert Path(p).parent == image_cache
    assert Path(p).name.startswith("img_")


def test_cleanup_drops_stale_album_dirs(image_cache):
    p = base.cache_image_from_bytes(JPEG, ".jpg", album_key="mgOLD")
    album = Path(p).parent
    stale = time.time() - 48 * 3600
    album.touch()
    import os
    os.utime(album, (stale, stale))
    removed = base._cleanup_cache_dir(image_cache, max_age_hours=24)
    assert removed >= 1
    assert not album.exists()


def test_telegram_album_key_prefers_media_group():
    # chat id is part of the key: message_ids are unique per chat, not
    # globally (PR #115282 review)
    msg = SimpleNamespace(media_group_id="998877", message_id=12, chat_id="-100123")
    assert TelegramAdapter._telegram_photo_album_key(msg) == "mg-100123-998877"
    single = SimpleNamespace(media_group_id=None, message_id=45, chat_id="-100123")
    assert TelegramAdapter._telegram_photo_album_key(single) == "msg-100123-45"


def test_telegram_album_key_same_message_id_different_chats_never_collide():
    from types import SimpleNamespace as NS
    a = TelegramAdapter._telegram_photo_album_key(NS(media_group_id=None, message_id=45, chat_id="-100111"))
    b = TelegramAdapter._telegram_photo_album_key(NS(media_group_id=None, message_id=45, chat_id="-100222"))
    assert a != b
