import asyncio
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock

    class _FakeFile:
        """Mimics discord.py File close enough for upload tests: stores the source handle
        (path-based -> BufferedReader via ``open()``, BytesIO stays a BytesIO) and exposes
        ``fp``/``filename`` the way discord.File does."""
        def __init__(self, fp, *, filename=None):
            if isinstance(fp, (str, Path)):
                fp = open(fp, "rb")
            self.fp = fp
            self.filename = filename or os.path.basename(getattr(fp, "name", "") or "file")
            self._closed = False

        def read(self, *a):
            return self.fp.read(*a) if not self._closed else b""

        def close(self):
            if not self._closed:
                self.fp.close()
                self._closed = True

        def seekable(self):
            return True

        def seek(self, *a):
            self.fp.seek(*a)
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    discord_mod.File = _FakeFile
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_send_rejects_whitespace_and_records_failed_final_reply(
    caplog, monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("DISCORD_MISSED_MESSAGE_BACKFILL", "true")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    channel = SimpleNamespace(send=AsyncMock())
    get_channel = MagicMock(return_value=channel)
    adapter._client = SimpleNamespace(
        get_channel=get_channel,
        fetch_channel=AsyncMock(),
    )
    with caplog.at_level("WARNING"):
        result = await adapter.send(
            "555",
            "  \n\t ",
            reply_to="123",
            metadata={"notify": True},
        )

    assert result.success is False
    assert result.error == "Refusing to send empty message"
    get_channel.assert_not_called()
    channel.send.assert_not_awaited()
    row = adapter._with_discord_recovery_db(
        lambda conn: conn.execute(
            "SELECT status, replied, outage_response, response_message_id "
            "FROM discord_messages WHERE message_id='123'"
        ).fetchone()
    )
    assert tuple(row) == ("failed", 0, 0, None)
    assert "Dropped empty message to chat=555" in caplog.text


def _voice_adapter(reference_obj, *, native_result=None, native_error=None):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    channel = SimpleNamespace(
        id=555,
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(return_value=SimpleNamespace(id=888)),
    )
    request = AsyncMock(return_value=native_result or {"id": "777"})
    if native_error is not None:
        request.side_effect = native_error
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=request),
    )
    return adapter, channel, request


def _native_voice_payload(request):
    form = request.await_args.kwargs["form"]
    payload = next(part["value"] for part in form if part["name"] == "payload_json")
    return json.loads(payload)


@pytest.mark.asyncio
async def test_send_retries_without_reference_when_reply_target_is_deleted():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    sent_msgs = [SimpleNamespace(id=1001), SimpleNamespace(id=1002)]
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 10008): Unknown Message"
            )
        return sent_msgs[len(send_calls) - 2]

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    long_text = "A" * (adapter.MAX_MESSAGE_LENGTH + 50)
    result = await adapter.send("555", long_text, reply_to="99")

    assert result.success is True
    assert result.message_id == "1001"
    # ids-only reference: the fetch is gone entirely — the retry happens
    # on the send-side 10008, not a fetch failure
    assert channel.fetch_message.await_count == 0
    assert channel.send.await_count == 3
    # the reference is constructed from ids, not fetched + to_reference()
    _discord_mod.MessageReference.assert_any_call(
        message_id=99, channel_id=None, guild_id=None,
        fail_if_not_exists=False)
    assert send_calls[0]["reference"] is _discord_mod.MessageReference.return_value
    assert send_calls[1]["reference"] is None
    assert send_calls[2]["reference"] is None


# ---------------------------------------------------------------------------
# Forum channel tests
# ---------------------------------------------------------------------------

import discord as _discord_mod




# ---------------------------------------------------------------------------
# Forum follow-up chunk failure reporting + media on forum paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forum_post_file_creates_thread_with_attachment():
    """_forum_post_file routes file-bearing sends to create_thread with file kwarg."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread_ch = SimpleNamespace(id=777, send=AsyncMock())
    thread = SimpleNamespace(
        id=777,
        message=SimpleNamespace(
            id=800,
            attachments=[SimpleNamespace(filename="photo.png")],
        ),
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(return_value=thread)

    # discord.File is a real class; build a MagicMock that looks like one
    fake_file = SimpleNamespace(filename="photo.png")

    result = await adapter._forum_post_file(
        forum_channel,
        content="here is a photo",
        file=fake_file,
    )

    assert result.success is True
    assert result.message_id == "800"
    forum_channel.create_thread.assert_awaited_once()
    call_kwargs = forum_channel.create_thread.await_args.kwargs
    assert call_kwargs["file"] is fake_file
    assert call_kwargs["content"] == "here is a photo"
    # Thread name derived from content's first line
    assert call_kwargs["name"] == "here is a photo"


@pytest.mark.asyncio
async def test_forum_post_file_fails_when_starter_has_no_attachments():
    """Forum create_thread can succeed yet return an attachmentless starter (#66797)."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread = SimpleNamespace(
        id=7,
        message=SimpleNamespace(id=8, attachments=[]),
        thread=SimpleNamespace(id=7, send=AsyncMock()),
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.create_thread = AsyncMock(return_value=thread)

    fake_file = SimpleNamespace(filename="clip.mp4")
    result = await adapter._forum_post_file(
        forum_channel,
        content="video clip",
        files=[fake_file],
    )

    assert result.success is False
    assert "no files" in (result.error or "").lower()
    forum_channel.create_thread.assert_awaited_once()


# ---------------------------------------------------------------------------
# Typing indicator task lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_restartable_after_error():
    """After a typing error, send_typing should start a new task (not blocked by stale entry)."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._typing_tasks = {}

    # First call fails
    adapter._client.http.request = AsyncMock(side_effect=Exception("503"))
    await adapter.send_typing("12345")
    await asyncio.sleep(0.1)

    # Second call should work
    adapter._client.http.request = AsyncMock()
    await adapter.send_typing("12345")

    assert "12345" in adapter._typing_tasks, \
        "Should restart typing after previous failure"


# ---------------------------------------------------------------------------
# #66797 — outbound MEDIA video must reach channel.send as a real attachment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_video_uses_path_based_files_kwarg(tmp_path, monkeypatch):
    """Regression for #66797: video MEDIA delivery must use path-based
    ``discord.File`` via ``files=[...]`` (same pattern as image batching).

    The previous open-handle + singular ``file=`` form could return a successful
    message with zero attachments after an earlier image batch on the same
    channel — silent drop from the user's perspective.
    """
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42fake")

    captured = {}

    class _FakeFile:
        def __init__(self, fp, filename=None, **kwargs):
            captured["fp"] = fp
            captured["filename"] = filename

    monkeypatch.setattr(discord_platform.discord, "File", _FakeFile)

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent_msg = SimpleNamespace(
        id=4242,
        attachments=[SimpleNamespace(filename="clip.mp4", url="https://cdn.example/clip.mp4")],
    )
    channel = SimpleNamespace(
        send=AsyncMock(return_value=sent_msg),
        type=0,
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: False)

    result = await adapter.send_video("555", str(video))

    assert result.success is True
    assert result.message_id == "4242"
    assert captured["fp"] == str(video)
    assert captured["filename"] == "clip.mp4"
    channel.send.assert_awaited_once()
    send_kwargs = channel.send.await_args.kwargs
    assert send_kwargs.get("file") is None
    assert isinstance(send_kwargs.get("files"), list) and len(send_kwargs["files"]) == 1


@pytest.mark.asyncio
async def test_send_video_fails_loud_when_message_has_no_attachments(tmp_path, monkeypatch):
    """If Discord accepts the message but attaches nothing, fail loud (#66797)."""
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    monkeypatch.setattr(
        discord_platform.discord,
        "File",
        lambda fp, filename=None, **kwargs: SimpleNamespace(fp=fp, filename=filename),
    )

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    # Message id present, but no attachments — the silent-drop failure mode.
    sent_msg = SimpleNamespace(id=99, attachments=[])
    channel = SimpleNamespace(send=AsyncMock(return_value=sent_msg), type=0)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: False)

    result = await adapter.send_video("555", str(video))

    assert result.success is False
    assert "no files" in (result.error or "").lower()
    channel.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_video_missing_file_fails_fast_without_touching_channel():
    """A missing MEDIA path must fail loud before any Discord I/O (#66797).

    The pre-flight ``os.path.isfile`` guard turns a would-be crash inside
    ``discord.File`` into an actionable ``File not found`` result, and must
    short-circuit before the channel is ever resolved.
    """
    def _boom(*_args, **_kwargs):
        raise AssertionError("channel must not be resolved for a missing file")

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(get_channel=_boom, fetch_channel=AsyncMock(side_effect=_boom))

    result = await adapter.send_video("555", "/no/such/clip.mp4")

    assert result.success is False
    assert "not found" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_send_file_attachment_forum_uses_files_kwarg(tmp_path, monkeypatch):
    """Forum-parent delivery must also route the path-based file through the
    plural ``files=[...]`` kwarg (#66797), so the create_thread starter message
    carries the attachment rather than silently dropping it."""
    import plugins.platforms.discord.adapter as discord_platform

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")

    monkeypatch.setattr(
        discord_platform.discord,
        "File",
        lambda fp, filename=None, **kwargs: SimpleNamespace(fp=fp, filename=filename),
    )

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    created_thread = SimpleNamespace(
        id=7,
        message=SimpleNamespace(
            id=8,
            attachments=[SimpleNamespace(filename="clip.mp4")],
        ),
    )
    forum_channel = SimpleNamespace(
        id=7,
        create_thread=AsyncMock(return_value=created_thread),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )
    monkeypatch.setattr(adapter, "_is_forum_parent", lambda _ch: True)

    result = await adapter.send_video("555", str(video))

    assert result.success is True
    forum_channel.create_thread.assert_awaited_once()
    thread_kwargs = forum_channel.create_thread.await_args.kwargs
    assert thread_kwargs.get("file") is None
    assert isinstance(thread_kwargs.get("files"), list) and len(thread_kwargs["files"]) == 1


# ---------------------------------------------------------------------------
# Upload-size preflight (#50846 / #52698)
# ---------------------------------------------------------------------------


def _preflight_adapter(channel):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]
    adapter._client = SimpleNamespace(get_channel=lambda _cid: channel, fetch_channel=AsyncMock())
    return adapter


def _fake_getsize(monkeypatch, oversized: Path, size: int):
    original = os.path.getsize
    monkeypatch.setattr(
        os.path, "getsize",
        lambda path: size if str(path) == str(oversized) else original(path),
    )


def test_discord_upload_limit_uses_guild_filesize_limit():
    from plugins.platforms.discord.adapter_media import (
        _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES,
        DiscordMediaMixin,
    )

    limit_for = DiscordMediaMixin._discord_upload_limit_bytes
    boosted = SimpleNamespace(guild=SimpleNamespace(filesize_limit=50 * 1024 * 1024))
    # discord.py 2.7.1 still reports 10 MiB for unboosted guilds; the platform default is
    # 20 MiB since Sep 3 2026, so a stale library constant must never lower the preflight.
    stale = SimpleNamespace(guild=SimpleNamespace(filesize_limit=_DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES // 2))

    assert limit_for(boosted) == 50 * 1024 * 1024
    assert limit_for(stale) == _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES
    assert limit_for(SimpleNamespace(guild=None)) == _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES
    assert limit_for(SimpleNamespace(guild=SimpleNamespace(filesize_limit=0))) == _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path_kw, name", [
    ("send_video", "video_path", "clip.mp4"),
    ("send_document", "file_path", "report.pdf"),
    ("send_voice", "audio_path", "note.ogg"),
])
async def test_oversized_upload_rejected_before_send(tmp_path, monkeypatch, method, path_kw, name):
    """Oversized local files never reach channel.send(file(s)=...) (#50846): the caller gets an
    actionable error (name, size, limit), the user a notice, and the base fallback never runs."""
    from plugins.platforms.discord.adapter_media import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    big = tmp_path / name
    big.write_bytes(b"x")
    _fake_getsize(monkeypatch, big, _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1)

    async def base_must_not_run(*_a, **_k):
        raise AssertionError("base adapter fallback must not run for a preflight reject")

    monkeypatch.setattr(f"gateway.platforms.base.BasePlatformAdapter.{method}", base_must_not_run)
    send = AsyncMock(return_value=SimpleNamespace(id=999))
    http = SimpleNamespace(request=AsyncMock(side_effect=AssertionError("raw upload must not run")))
    adapter = _preflight_adapter(SimpleNamespace(id=555, guild=None, send=send))
    adapter._client.http = http

    result = await getattr(adapter, method)("555", **{path_kw: str(big)})

    assert result.success is False
    assert "too large" in result.error.lower()
    assert name in result.error and "20.0 MB" in result.error and "limit 20 MB" in result.error
    assert send.await_count == 1
    kwargs = send.await_args.kwargs
    assert "file" not in kwargs and "files" not in kwargs
    assert "Could not attach" in kwargs["content"] and name in kwargs["content"]


@pytest.mark.asyncio
async def test_send_video_under_guild_boost_limit_uploads(tmp_path, monkeypatch):
    """A boosted guild's higher cap is honored: a file over the default but under the guild
    limit is uploaded, not rejected."""
    from plugins.platforms.discord.adapter_media import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    video = tmp_path / "ok.mp4"
    video.write_bytes(b"fake-video-bytes")
    _fake_getsize(monkeypatch, video, _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1)
    sent_msg = SimpleNamespace(id=42, attachments=[SimpleNamespace(filename="ok.mp4", url="https://cdn/ok.mp4")])
    send = AsyncMock(return_value=sent_msg)
    adapter = _preflight_adapter(
        SimpleNamespace(id=777, guild=SimpleNamespace(filesize_limit=50 * 1024 * 1024), send=send))

    result = await adapter.send_video("777", str(video))

    assert result.success is True and result.message_id == "42"
    assert send.await_count == 1 and send.await_args.kwargs.get("files")


@pytest.mark.asyncio
async def test_send_multiple_images_skips_oversized_local_file(tmp_path, monkeypatch):
    """Sibling site of #50846: an oversized local image used to 413 the whole chunk and dump
    its siblings into the per-image fallback. It is skipped up front, the rest of the chunk is
    delivered with a notice appended; an all-oversized chunk still sends the notice alone."""
    from plugins.platforms.discord.adapter_media import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    small = tmp_path / "small.png"
    small.write_bytes(b"ok")
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * (_DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1))

    def _sent(*, content=None, files=None, **kw):
        return SimpleNamespace(id=7, attachments=[SimpleNamespace(filename=f.filename) for f in files or []])

    send = AsyncMock(side_effect=_sent)
    adapter = _preflight_adapter(SimpleNamespace(id=9, guild=None, send=send))

    mixed = await adapter.send_multiple_images("9", [(f"file://{small}", ""), (f"file://{big}", "")])
    assert mixed.success is True
    kwargs = send.await_args.kwargs
    assert len(kwargs["files"]) == 1  # only the small image made it
    assert "big.png" in kwargs["content"] and "exceeds" in kwargs["content"]

    send.reset_mock()
    only_big = await adapter.send_multiple_images("9", [(f"file://{big}", "")])
    assert only_big.success is False
    assert send.await_count == 1
    kwargs = send.await_args.kwargs
    assert not kwargs.get("files") and "big.png" in kwargs["content"]


# ---------------------------------------------------------------------------
# Post-send attachment verification (#66797 follow-up: verify, retry, top up,
# notify instead of trusting the POST)
# ---------------------------------------------------------------------------


def _chunk_channel(send_side_effect, warn: bool = True):
    channel = SimpleNamespace(id=9, guild=None, send=AsyncMock(side_effect=send_side_effect))
    adapter = _preflight_adapter(channel)
    adapter.warning_notifications_enabled = lambda *a, **k: warn  # type: ignore[method-assign]
    return adapter, channel


def _image(tmp_path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"fakepng")
    return p


@pytest.mark.asyncio
async def test_image_chunk_full_drop_retried(tmp_path, monkeypatch):
    """A zero-attachment response is the #66797 silent drop: retry once, succeed, no notice."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    img = _image(tmp_path, "cat.png")

    calls = {"n": 0}

    def _flaky(*, content=None, files=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:  # first attempt: the silent drop
            return SimpleNamespace(id=1, attachments=[])
        return SimpleNamespace(id=2, attachments=[SimpleNamespace(filename=f.filename) for f in files or []])

    adapter, channel = _chunk_channel(_flaky)
    result = await adapter.send_multiple_images("9", [(f"file://{img}", "cute cat")])

    assert result.success is True
    assert channel.send.await_count == 2
    assert channel.send.await_args_list[0].kwargs.get("content") == "cute cat"
    assert channel.send.await_args_list[1].kwargs.get("content") is None  # caption not duplicated


@pytest.mark.asyncio
async def test_image_chunk_partial_delivery_topped_up(tmp_path, monkeypatch):
    """Partial attachments: top up only the missing file in a follow-up message."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    cat, dog = _image(tmp_path, "cat.png"), _image(tmp_path, "dog.png")

    calls = {"n": 0}

    def _partial(*, content=None, files=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:  # only cat lands
            return SimpleNamespace(id=1, attachments=[SimpleNamespace(filename="cat.png")])
        return SimpleNamespace(id=2, attachments=[SimpleNamespace(filename=f.filename) for f in files or []])

    adapter, channel = _chunk_channel(_partial)
    result = await adapter.send_multiple_images("9", [(f"file://{cat}", ""), (f"file://{dog}", "")])

    assert result.success is True
    assert channel.send.await_count == 2
    first_files = channel.send.await_args_list[0].kwargs["files"]
    second_files = channel.send.await_args_list[1].kwargs["files"]
    assert len(first_files) == 2 and len(second_files) == 1
    assert second_files[0].filename == "dog.png"
    assert channel.send.await_args_list[1].kwargs.get("content") is None


@pytest.mark.asyncio
async def test_image_chunk_persistent_drop_fails_loud(tmp_path, monkeypatch):
    """Two zero-attachment responses: the batch fails and notifies (no silent drop)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    img = _image(tmp_path, "cat.png")

    def _dead(*, content=None, files=None, **kw):
        return SimpleNamespace(id=1, attachments=[])

    adapter, channel = _chunk_channel(_dead)
    result = await adapter.send_multiple_images("9", [(f"file://{img}", "cap")])

    assert result.success is False
    assert channel.send.await_count == 3  # attempt 1, retry, failure notice
    notice = channel.send.await_args_list[2].kwargs.get("content")
    assert notice and "attachment" in notice.lower() and "cat.png" in notice


@pytest.mark.asyncio
async def test_image_chunk_persistent_drop_quiet_mode(tmp_path, monkeypatch):
    """warning_notifications off: the batch still fails but stays quiet (text already landed)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    img = _image(tmp_path, "cat.png")

    def _dead(*, content=None, files=None, **kw):
        return SimpleNamespace(id=1, attachments=[])

    adapter, channel = _chunk_channel(_dead, warn=False)
    result = await adapter.send_multiple_images("9", [(f"file://{img}", "cap")])

    assert result.success is False
    assert channel.send.await_count == 2  # attempt + retry, no failure notice


@pytest.mark.asyncio
async def test_url_images_staged_as_temp_files(tmp_path, monkeypatch):
    """URL images must ride as path-based Files (open-handle BytesIOs can race the encoder)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    captured_files = []

    def _sent(*, content=None, files=None, **kw):
        captured_files.extend(files or [])
        return SimpleNamespace(id=1, attachments=[SimpleNamespace(filename=f.filename) for f in files or []])

    adapter, _channel = _chunk_channel(_sent)

    async def _fake_download(session, url, **kwargs):
        return 200, b"fakejpeg", {"content-type": "image/jpeg"}

    monkeypatch.setattr(
        "plugins.platforms.discord.adapter._read_url_image_with_redirect_guard", _fake_download,
    )

    result = await adapter.send_multiple_images("9", [("https://example.com/cat.jpg", "")])
    assert result.success is True
    assert len(captured_files) == 1
    staged = captured_files[0]
    # Path-based staging (an open BytesIO handle can race the multipart encoder):
    # with the real discord.File, fp is a file object whose .name is a temp path;
    # with a MagicMock-backed File, the first positional arg was the path string.
    assert getattr(staged, "filename", None) == "image_0.jpg"
    fp = getattr(staged, "fp", None)
    assert not isinstance(fp, io.BytesIO)
    if isinstance(fp, (str, Path)):
        assert "hermes_discord_images_" in str(fp)
    elif isinstance(fp, io.IOBase):
        assert isinstance(fp, io.BufferedReader) and "hermes_discord_images_" in fp.name


# ---------------------------------------------------------------------------
# _send_single_file_verified: the single URL-media path (send_image / send_video /
# send_animation) also verifies its attachment instead of trusting the POST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_file_verified_retries_drop_then_succeeds(tmp_path, monkeypatch):
    """A zero-attachment single-file send is retried once; the next landing succeeds. No user
    notice is sent (the caller decides the fallback)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    img = _image(tmp_path, "cat.png")

    calls = {"n": 0}

    def _flaky(*, content=None, file=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return SimpleNamespace(id=1, attachments=[])
        return SimpleNamespace(id=2, attachments=[SimpleNamespace(filename="cat.png")])

    channel = SimpleNamespace(id=9, guild=None, send=AsyncMock(side_effect=_flaky))
    adapter = _preflight_adapter(channel)
    ok = await adapter._send_single_file_verified(channel, content="cap", path=str(img), filename="cat.png")
    assert ok is True
    assert channel.send.await_count == 2


@pytest.mark.asyncio
async def test_single_file_verified_persistent_drop_falls_through(tmp_path, monkeypatch):
    """Two zero-attachment single-file sends: returns False so the caller can fall back to the URL."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    img = _image(tmp_path, "cat.png")

    def _dead(*, content=None, file=None, **kw):
        return SimpleNamespace(id=1, attachments=[])

    channel = SimpleNamespace(id=9, guild=None, send=AsyncMock(side_effect=_dead))
    adapter = _preflight_adapter(channel)
    ok = await adapter._send_single_file_verified(channel, content="cap", path=str(img), filename="cat.png")
    assert ok is False
    assert channel.send.await_count == 2  # attempt + retry, no extra user-notice send
