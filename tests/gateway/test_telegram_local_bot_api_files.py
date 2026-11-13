"""Inbound files from a ``--local`` telegram-bot-api whose storage is mounted elsewhere on this host (#55698).

A ``--local`` server answers getFile with ITS OWN absolute path (``/var/lib/telegram-bot-api/<token>/
videos/file_7.MOV``) and serves nothing over HTTP. When Hermes runs in another container/host, PTB
cannot stat that path, so it prefixes ``base_file_url`` and GETs it: 404 ``Not Found: method not
found``. ``extra.local_bot_api_root`` names where that storage is mounted here; the file is read from
``<root>/<token>/<rest>`` (or ``<root>/<rest>``) with two stats, no directory walk.
"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_media_fetch import local_bot_api_path

TOKEN = "123456:AAFakeToken_abc-DEF"
DATA = b"local-mov-bytes"


@pytest.fixture(autouse=True)
def _cache_dirs(tmp_path, monkeypatch):
    for name in ("DOCUMENT_CACHE_DIR", "VIDEO_CACHE_DIR", "AUDIO_CACHE_DIR", "IMAGE_CACHE_DIR"):
        monkeypatch.setattr(f"hermes_agent.gateway.platforms.base.{name}", tmp_path / name.lower())


def _adapter(extra):
    a = TelegramAdapter(PlatformConfig(enabled=True, token=TOKEN, extra=extra))
    a.handle_message = AsyncMock()
    a._is_callback_user_authorized = lambda user_id, **_kw: True
    return a


def _message(file_path, *, as_document):
    file_obj = SimpleNamespace(file_path=file_path, download_as_bytearray=AsyncMock(
        side_effect=RuntimeError("Not Found: method not found")))
    media = MagicMock(file_size=len(DATA), mime_type="video/mp4", file_name="IMG_5109.MOV")
    media.get_file = AsyncMock(return_value=file_obj)
    msg = MagicMock(message_id=42, text="", caption=None, date=None, media_group_id=None, message_thread_id=None,
                    photo=None, video=None, video_note=None, audio=None, voice=None, sticker=None, document=None)
    msg.chat = MagicMock(id=100, type="private", title=None, full_name="Test User")
    msg.from_user = MagicMock(id=1, full_name="Test User")
    msg.reply_text = AsyncMock()
    setattr(msg, "document" if as_document else "video", media)
    return msg, file_obj


def _stored(root, kind="videos"):
    path = root / TOKEN / kind / "file_7.MOV"
    path.parent.mkdir(parents=True)
    path.write_bytes(DATA)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("as_document", [False, True], ids=["video", "video-document"])
@pytest.mark.parametrize("reported", [
    "/var/lib/telegram-bot-api/{token}/{kind}/file_7.MOV",
    "http://127.0.0.1:8081/file/bot{token}//var/lib/telegram-bot-api/{token}/{kind}/file_7.MOV",
], ids=["container-path", "url-embedded"])
async def test_local_server_file_is_read_from_the_mounted_root(tmp_path, as_document, reported):
    kind = "documents" if as_document else "videos"
    _stored(tmp_path / "mount", kind)
    adapter = _adapter({"base_url": "http://127.0.0.1:8081/bot", "local_mode": True,
                        "local_bot_api_root": str(tmp_path / "mount")})
    msg, file_obj = _message(reported.format(token=TOKEN, kind=kind), as_document=as_document)

    await adapter._handle_media_message(MagicMock(message=msg, effective_message=msg), MagicMock())

    event = adapter.handle_message.call_args[0][0]
    assert event.message_type == MessageType.VIDEO
    assert Path(event.media_urls[0]).read_bytes() == DATA
    file_obj.download_as_bytearray.assert_not_awaited()  # no HTTP GET the --local server 404s
    msg.reply_text.assert_not_awaited()  # no "Couldn't download" reply


@pytest.mark.asyncio
async def test_without_the_root_the_download_path_is_unchanged(tmp_path):
    """No ``local_bot_api_root``: the PTB download runs as before (and its failure is surfaced)."""
    _stored(tmp_path / "mount")
    adapter = _adapter({"base_url": "http://127.0.0.1:8081/bot", "local_mode": True})
    msg, file_obj = _message(f"/var/lib/telegram-bot-api/{TOKEN}/videos/file_7.MOV", as_document=False)

    await adapter._handle_media_message(MagicMock(message=msg, effective_message=msg), MagicMock())

    file_obj.download_as_bytearray.assert_awaited()
    msg.reply_text.assert_awaited_once()


def test_mapping_bounds(tmp_path):
    root = tmp_path / "mount"
    stored = _stored(root)
    (root / "videos").mkdir()
    (root / "videos" / "flat.MOV").write_bytes(DATA)
    secret = tmp_path / "secret.txt"
    secret.write_text("x")

    assert local_bot_api_path(f"/srv/tg/{TOKEN}/videos/file_7.MOV", str(root)) == stored
    # a root mounted at the token directory itself (``<root>/<rest>``)
    assert local_bot_api_path(f"/var/lib/telegram-bot-api/{TOKEN}/videos/flat.MOV", str(root)) == root / "videos" / "flat.MOV"
    # a relative public-API path has no token segment; a missing file maps to nothing
    assert local_bot_api_path("videos/file_7.MOV", str(root)) is None
    assert local_bot_api_path(f"/var/lib/telegram-bot-api/{TOKEN}/videos/none.MOV", str(root)) is None
    # ``..`` never escapes the root
    assert local_bot_api_path(f"/x/{TOKEN}/../../secret.txt", str(root)) is None
    assert os.path.exists(secret)
