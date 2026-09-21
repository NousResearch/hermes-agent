"""Phase-2B generic document-batching cross-platform regression tests.

Hermetic: no hardcoded paths (uses ``tmp_path``), no stale ``bot_token=``,
no duplicate standalone-429 transport (that lives in
``tests/gateway/test_discord_standalone_media_batching.py``).

Proves:
  (a) Base default ``send_multiple_documents`` sends each doc individually.
  (b) Scheduler ``_send_media_via_adapter`` routes documents (one or many)
      to the generic batch method while voice/image/video keep individual paths.
  (c) Discord override chunks 26 → [10, 10, 6], forum routing works,
      chunk-failure per-file fallback works.
  (d) Phase-1 bare-path tuple contract still reaches the batch method.
"""
import asyncio
import os
import sys
import threading
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ── discord.py stub (may not be installed in test venv) ────────────────────

import plugins.platforms.discord.adapter as _adapter_mod

if _adapter_mod.discord is None:
    class _StubFile:
        def __init__(self, fp, filename=None):
            self.fp = fp
            self.filename = filename

    _adapter_mod.discord = types.SimpleNamespace(File=_StubFile)

from gateway.platforms.base import BasePlatformAdapter, SendResult  # noqa: E402
from cron.scheduler import _send_media_via_adapter  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


# ── Fakes ──────────────────────────────────────────────────────────────────


class _CountingAdapter(BasePlatformAdapter):
    """Minimal adapter counting send_document / send_multiple_documents /
    send_voice / send_image_file / send_video calls.

    Inherits ``send_multiple_documents`` from Base (per-document fallback)
    so we can verify the cross-platform default contract."""

    def __init__(self):
        self.doc_calls = []
        self.multi_calls = []
        self.voice_calls = []
        self.image_calls = []
        self.video_calls = []

    async def send(self, chat_id="", content="", reply_to=None, metadata=None, **kw):
        return SendResult(success=True)

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kw):
        self.doc_calls.append(file_path)
        return SendResult(success=True)

    async def send_multiple_documents(self, chat_id, documents, metadata=None,
                                       human_delay=0.0):
        self.multi_calls.append(list(documents))
        await BasePlatformAdapter.send_multiple_documents(
            self, chat_id, documents, metadata=metadata, human_delay=human_delay,
        )

    async def send_voice(self, chat_id, audio_path, metadata=None, **kw):
        self.voice_calls.append(audio_path)
        return SendResult(success=True)

    async def send_image_file(self, chat_id, image_path, metadata=None, **kw):
        self.image_calls.append(image_path)
        return SendResult(success=True)

    async def send_video(self, chat_id, video_path, metadata=None, **kw):
        self.video_calls.append(video_path)
        return SendResult(success=True)

    async def connect(self, **kw):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "dm"}

    @property
    def name(self):
        return "test-counting"


class _FakeChannel:
    """Captures send/create_thread calls, enforces Discord's 10-attachment cap."""

    def __init__(self):
        self.sends = []
        self.is_forum = False
        self.created_threads = []

    async def send(self, content=None, files=None, **kw):
        n = len(files) if files else 0
        if n > 10:
            raise RuntimeError(f"Max 10 attachments per message, got {n}")
        self.sends.append((content, n))

    async def create_thread(self, **kw):
        files = kw.get("files")
        n = len(files) if files else 0
        self.created_threads.append((kw.get("name", ""), n))
        return types.SimpleNamespace(id=999, message=types.SimpleNamespace(id="m1"))


class _FakeClient:
    def __init__(self, channel=None):
        self._channel = channel or _FakeChannel()

    def get_channel(self, _id):
        return self._channel

    async def fetch_channel(self, _id):
        return self._channel


def _make_discord_adapter(tmp_path, n=26):
    """Build a DiscordAdapter with a fake client and n real .html files."""
    files = []
    for i in range(n):
        p = tmp_path / f"article-{i:02d}.html"
        p.write_text(f"<html>article {i}</html>")
        files.append((str(p), ""))
    ad = DiscordAdapter.__new__(DiscordAdapter)
    ad._client = _FakeClient()
    ad._is_forum_parent = lambda channel: ad._client._channel.is_forum
    ad.platform = types.SimpleNamespace(value="discord")
    return ad, files


def _make_running_loop():
    """Create a running event loop on a daemon thread for safe_schedule_threadsafe."""
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return loop, t


def _stop_loop(loop, t):
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)
    loop.close()


def _run_via_adapter(adapter, chat_id, media_files, loop, job=None):
    _send_media_via_adapter(
        adapter=adapter,
        chat_id=chat_id,
        media_files=media_files,
        metadata=None,
        loop=loop,
        job=job or {"id": "test-job"},
    )


# ── (a) Base default: per-document fallback ────────────────────────────────


@pytest.mark.asyncio
async def test_base_default_sends_each_doc_individually(tmp_path):
    """Inheriting Base.send_multiple_documents sends each doc via send_document."""
    files = [(str(tmp_path / f"doc-{i}.md"), "") for i in range(3)]
    for fp, _ in files:
        open(fp, "w").write("doc")

    ad = _CountingAdapter()
    # Call Base directly (bypassing the _CountingAdapter override).
    await BasePlatformAdapter.send_multiple_documents(ad, "123", files, human_delay=0.0)

    assert len(ad.doc_calls) == 3
    for fp, _ in files:
        assert fp in ad.doc_calls


@pytest.mark.asyncio
async def test_base_default_skips_missing_files(tmp_path):
    real = tmp_path / "real.txt"
    real.write_text("ok")
    files = [(str(real), ""), ("/nonexistent/skip-me.txt", "")]

    ad = _CountingAdapter()
    await BasePlatformAdapter.send_multiple_documents(ad, "123", files, human_delay=0.0)

    assert ad.doc_calls == [str(real)]


# ── (b) Scheduler routing: documents → batch, others → individual ─────────


def test_scheduler_routes_many_documents_to_batch(tmp_path):
    """_send_media_via_adapter routes 5 documents to send_multiple_documents."""
    docs = []
    for i in range(5):
        p = tmp_path / f"report-{i}.html"
        p.write_text(f"<html>report {i}</html>")
        docs.append((str(p), False))

    ad = _CountingAdapter()
    loop, t = _make_running_loop()
    try:
        _run_via_adapter(ad, "123", docs, loop)
    finally:
        _stop_loop(loop, t)

    assert len(ad.multi_calls) == 1, f"expected 1 batch call, got {len(ad.multi_calls)}"
    assert len(ad.multi_calls[0]) == 5, "batch should contain all 5 documents"
    assert ad.doc_calls == [fp for fp, _ in docs], "base fallback should send all 5"
    assert ad.voice_calls == []
    assert ad.image_calls == []
    assert ad.video_calls == []


def test_scheduler_routes_single_document_to_batch(tmp_path):
    """Even a single document goes through send_multiple_documents."""
    p = tmp_path / "single.html"
    p.write_text("<html>single</html>")

    ad = _CountingAdapter()
    loop, t = _make_running_loop()
    try:
        _run_via_adapter(ad, "123", [(str(p), False)], loop)
    finally:
        _stop_loop(loop, t)

    assert len(ad.multi_calls) == 1
    assert ad.doc_calls == [str(p)]


def test_scheduler_routes_voice_image_video_individually(tmp_path):
    """Voice/image/video must NOT be batched — they use individual methods."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18")
    doc = tmp_path / "note.html"
    doc.write_text("<html>note</html>")

    media = [(str(audio), True), (str(img), False), (str(video), False), (str(doc), False)]

    ad = _CountingAdapter()
    loop, t = _make_running_loop()
    try:
        _run_via_adapter(ad, "123", media, loop)
    finally:
        _stop_loop(loop, t)

    assert ad.voice_calls == [str(audio)], f"voice: {ad.voice_calls}"
    assert ad.image_calls == [str(img)], f"image: {ad.image_calls}"
    assert ad.video_calls == [str(video)], f"video: {ad.video_calls}"
    # Document still goes through the batch path → base fallback → send_document.
    assert len(ad.multi_calls) == 1
    assert ad.doc_calls == [str(doc)]


def test_scheduler_mixed_media_partitions_correctly(tmp_path):
    """Mixed media: voice+image+doc → voice/image individual, doc batched."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\x00")
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG")
    doc1 = tmp_path / "a.html"
    doc1.write_text("a")
    doc2 = tmp_path / "b.html"
    doc2.write_text("b")

    media = [(str(audio), True), (str(img), False), (str(doc1), False), (str(doc2), False)]

    ad = _CountingAdapter()
    loop, t = _make_running_loop()
    try:
        _run_via_adapter(ad, "123", media, loop)
    finally:
        _stop_loop(loop, t)

    assert ad.voice_calls == [str(audio)]
    assert ad.image_calls == [str(img)]
    assert ad.video_calls == []
    assert len(ad.multi_calls) == 1
    assert len(ad.multi_calls[0]) == 2
    assert set(ad.doc_calls) == {str(doc1), str(doc2)}


# ── (c) Discord override: chunking, forum, fallback ───────────────────────


@pytest.mark.asyncio
async def test_discord_chunks_26_into_10_10_6(tmp_path):
    ad, files = _make_discord_adapter(tmp_path, 26)
    await ad.send_multiple_documents("123", files, human_delay=0.0)
    ch = ad._client._channel
    assert len(ch.sends) == 3
    assert [c for _, c in ch.sends] == [10, 10, 6]
    assert all(c <= 10 for _, c in ch.sends)


@pytest.mark.asyncio
async def test_discord_skips_missing_files(tmp_path):
    ad, files = _make_discord_adapter(tmp_path, 3)
    files.append(("/nonexistent/skip.html", ""))
    await ad.send_multiple_documents("123", files, human_delay=0.0)
    ch = ad._client._channel
    assert len(ch.sends) == 1
    assert ch.sends[0][1] == 3


@pytest.mark.asyncio
async def test_discord_forum_uses_create_thread(tmp_path):
    ad, files = _make_discord_adapter(tmp_path, 5)
    ad._client._channel.is_forum = True
    await ad.send_multiple_documents("123", files, human_delay=0.0)
    ch = ad._client._channel
    assert len(ch.created_threads) == 1
    assert ch.created_threads[0][1] == 5
    assert len(ch.sends) == 0


@pytest.mark.asyncio
async def test_discord_chunk_failure_per_file_fallback(tmp_path):
    ad, files = _make_discord_adapter(tmp_path, 3)

    original_send = ad._client._channel.send
    call_count = [0]

    async def failing_send(content=None, files=None, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated Discord API error")
        return await original_send(content=content, files=files, **kw)

    ad._client._channel.send = failing_send

    fallback_calls = []

    async def fake_send_file(chat_id, file_path, caption=None, file_name=None):
        fallback_calls.append(file_path)
        return SendResult(success=True)

    ad._send_file_attachment = fake_send_file

    await ad.send_multiple_documents("123", files, human_delay=0.0)

    assert len(fallback_calls) == 3, f"expected 3 per-file fallbacks, got {len(fallback_calls)}"


# ── (d) Phase-1 bare-path tuple contract ───────────────────────────────────


def test_bare_path_tuple_reaches_batch_method(tmp_path):
    """Bare-path tuples (path, False) from Phase-1 coercion reach
    send_multiple_documents through _send_media_via_adapter — no crash."""
    p = tmp_path / "report.html"
    p.write_text("<html>report</html>")

    ad = _CountingAdapter()
    loop, t = _make_running_loop()
    try:
        _run_via_adapter(ad, "123", [(str(p), False)], loop)
    finally:
        _stop_loop(loop, t)

    assert len(ad.multi_calls) == 1, "bare-path tuple must reach send_multiple_documents"
    assert ad.multi_calls[0] == [(str(p), False)]
    assert ad.doc_calls == [str(p)], "base fallback must deliver the file"
