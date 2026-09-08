"""Real cache-file bounds and event-loop responsiveness at hosted Telegram ingress."""
import asyncio
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.hosted_room_attachments import MAX_ATTACHMENT_BYTES
from plugins.platforms.telegram.hosted_room_ingress import capture_media


@pytest.mark.parametrize("size,grows", [
    (0, False), (MAX_ATTACHMENT_BYTES, False), (MAX_ATTACHMENT_BYTES + 1, False),
    (2 * 1024**3, False), (MAX_ATTACHMENT_BYTES, True),
])
def test_capture_bounds_before_read_and_hash_without_blocking_loop(tmp_path, monkeypatch, size, grows):
    path = tmp_path / "payload.txt"
    with path.open("wb") as handle:
        handle.truncate(size)
    cached = SimpleNamespace(path=str(path), kind="file", media_type="text/plain", display_name=path.name)
    adapter = SimpleNamespace(_max_doc_bytes=2 * 1024**3,
                              _download_observed_media=AsyncMock(return_value=("ok", cached)))
    original_open = Path.open
    original_hash, original_stat = hashlib.sha256, os.fstat
    opened, release = threading.Event(), threading.Event()
    reads = []
    main_thread = threading.get_ident()

    def tracked_hash(data):
        assert threading.get_ident() != main_thread, "hashing blocked polling loop"
        assert 0 < len(data) <= MAX_ATTACHMENT_BYTES
        return original_hash(data)

    def growing_stat(fd):
        result = original_stat(fd)
        if grows:
            with original_open(path, "ab") as handle:
                handle.write(b"x")
        return result

    class TrackedFile:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, count=-1):
            reads.append(count)
            assert 0 < size <= MAX_ATTACHMENT_BYTES, "overbound cache must not be read"
            assert threading.get_ident() != main_thread, "cache read blocked polling loop"
            assert 0 < count <= MAX_ATTACHMENT_BYTES + 1
            return self.handle.read(count)

    def tracked_open(self, *args, **kwargs):
        if self != path:
            return original_open(self, *args, **kwargs)
        opened.set()
        assert threading.get_ident() != main_thread, "filesystem open blocked polling loop"
        assert release.wait(5), "event loop failed to release filesystem worker"
        return TrackedFile(original_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(hashlib, "sha256", tracked_hash)
    monkeypatch.setattr(os, "fstat", growing_stat)

    async def exercise():
        task = asyncio.create_task(capture_media(adapter, SimpleNamespace(chat_id=-99, message_id=1)))
        try:
            assert await asyncio.to_thread(opened.wait, 5)
        finally:
            release.set()
        return await task

    attachments, rejected = asyncio.run(exercise())
    if 0 < size <= MAX_ATTACHMENT_BYTES and not grows:
        assert rejected is None
        assert attachments[0]["size"] == size
        assert attachments[0]["sha256"] == original_hash(bytes(size)).hexdigest()
        assert attachments[0]["upload_id"] == "telegram:-99:1:0"
        assert reads
    else:
        assert attachments == []
        assert rejected == {"reason": "oversized"}
        assert bool(reads) is grows


@pytest.mark.parametrize("local_mode", [False, True])
@pytest.mark.parametrize("declared,actual,limit,reason", [
    (MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_BYTES, 2 * 1024**3, None),
    (MAX_ATTACHMENT_BYTES + 1, 0, 2 * 1024**3, "oversized"),
    (2 * 1024**3, 0, 2 * 1024**3, "oversized"),
    (1, MAX_ATTACHMENT_BYTES + 1, 2 * 1024**3, "oversized"),
    (2, 2, 1, "oversized"),
])
def test_native_download_honors_hosted_and_adapter_caps_before_caching(
        tmp_path, monkeypatch, declared, actual, limit, reason, local_mode):
    pytest.importorskip("telegram")
    from telegram import File
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:local-test", extra={"local_mode": local_mode}))
    adapter._max_doc_bytes = limit
    if local_mode:
        local_path = tmp_path / "bot-api.txt"
        with local_path.open("wb") as handle:
            handle.truncate(actual)
        file_obj = File("file-id", "unique-id", file_size=declared, file_path=str(local_path))
        original_open, original_read = Path.open, Path.read_bytes
        main_thread = threading.get_ident()

        def checked_open(path, *args, **kwargs):
            if path == local_path:
                assert threading.get_ident() != main_thread, "local Bot API file read blocked polling"
            return original_open(path, *args, **kwargs)

        def checked_read(path):
            assert path != local_path, "local Bot API file read must be bounded"
            return original_read(path)

        monkeypatch.setattr(Path, "open", checked_open)
        monkeypatch.setattr(Path, "read_bytes", checked_read)
    else:
        file_obj = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(actual)),
                                   file_path="payload.txt")
    source = SimpleNamespace(file_size=declared, get_file=AsyncMock(return_value=file_obj))
    monkeypatch.setattr(adapter, "_observed_media_source", lambda msg: (source, "payload.txt", "text/plain", "file"))
    attachments, rejected = asyncio.run(capture_media(adapter, SimpleNamespace(chat_id=-99, message_id=1)))
    if reason:
        assert attachments == []
        assert rejected == {"reason": reason}
    else:
        assert rejected is None
        assert attachments[0]["size"] == actual
        assert Path(attachments[0]["path"]).read_bytes() == bytes(actual)
    assert source.get_file.await_count == int(0 < declared <= min(limit, MAX_ATTACHMENT_BYTES))
    if reason:
        assert not list(tmp_path.rglob("doc_*")), "rejected bytes must not enter the native cache"


@pytest.mark.parametrize("status", ["oversized", "failed", "unreadable"])
def test_owner_media_rejection_is_durable_and_never_admits_caption_alone(tmp_path, monkeypatch, status):
    pytest.importorskip("telegram")
    from telegram.ext import ApplicationHandlerStop
    from plugins.platforms.telegram import hosted_room_ingress as ingress
    from plugins.platforms.telegram import hosted_room_transport as transport
    from gateway.hosted_rooms import create_room

    binding = {"enabled": True, "room_id": "media-room", "chat_id": -99, "owner_id": 42,
               "queue_db": str(tmp_path / "queue.db"), "control_profile": "impl",
               "bots": {"impl": {"id": 1, "username": "impl_bot"},
                        "research": {"id": 2, "username": "research_bot"}}}
    monkeypatch.setattr(ingress, "hosted_room_binding_path", lambda: tmp_path / "binding.json")
    monkeypatch.setattr(ingress, "load_binding", lambda *args, **kwargs: binding)
    adapter = SimpleNamespace(_max_doc_bytes=2 * 1024**3,
                              _download_observed_media=AsyncMock(return_value=(status, None)))
    application = Mock()
    assert ingress.wire(application, adapter)
    incoming = application.add_handler.call_args.args[0].callback

    async def exercise():
        for identity, is_bot in ((666, False), (42, True), (42, False)):
            message = SimpleNamespace(chat_id=-99, message_id=1, text=None, caption="keep this with its file",
                                      reply_to_message=None, message_thread_id=None)
            update = SimpleNamespace(effective_message=message, edited_message=None,
                                     effective_user=SimpleNamespace(id=identity, is_bot=is_bot))
            with pytest.raises(ApplicationHandlerStop):
                await incoming(update, None)

    asyncio.run(exercise())
    assert adapter._download_observed_media.await_count == 1
    service = SimpleNamespace(db_path=tmp_path / "room.db", send=Mock(), put_attachment=Mock())
    create_room(service.db_path, room_id=binding["room_id"], name="Media", authority_gateway_id="owner",
                members=[{"member_id": p, "profile": p, "handle": p} for p in binding["bots"]])
    item = transport.Transport(service, binding)
    item.ingest()
    with sqlite3.connect(binding["queue_db"]) as db:
        rows = db.execute("SELECT user_id,text,media_json,state,result FROM inbox").fetchall()
    assert len(rows) == 1
    owner, text, media, state, result = rows[0]
    assert owner == 42 and text == "keep this with its file"
    assert json.loads(media)["rejected"] == {"reason": status}
    assert state == "rejected" and status in json.loads(result)["error"]
    service.send.assert_not_called()
    service.put_attachment.assert_not_called()
