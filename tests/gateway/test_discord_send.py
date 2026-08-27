import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
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

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


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

import discord as _discord_mod  # noqa: E402 — imported after _ensure_discord_mock


class TestIsForumParent:
    def test_none_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        assert adapter._is_forum_parent(None) is False

    def test_forum_channel_class_instance(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        forum_cls = getattr(_discord_mod, "ForumChannel", None)
        if forum_cls is None:
            # Re-create a type for the mock
            forum_cls = type("ForumChannel", (), {})
            _discord_mod.ForumChannel = forum_cls
        ch = forum_cls()
        assert adapter._is_forum_parent(ch) is True


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


def test_discord_upload_limit_uses_guild_filesize_limit():
    from plugins.platforms.discord.adapter import (
        DiscordAdapter,
        _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES,
    )

    guild_channel = SimpleNamespace(guild=SimpleNamespace(filesize_limit=50 * 1024 * 1024))
    dm_channel = SimpleNamespace(guild=None)
    no_limit_guild = SimpleNamespace(guild=SimpleNamespace(filesize_limit=0))

    assert DiscordAdapter._discord_upload_limit_bytes(guild_channel) == 50 * 1024 * 1024
    assert DiscordAdapter._discord_upload_limit_bytes(dm_channel) == _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES
    assert DiscordAdapter._discord_upload_limit_bytes(no_limit_guild) == _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES


@pytest.mark.asyncio
async def test_send_file_attachment_rejects_oversized_before_upload(tmp_path):
    """Oversized local files must not call channel.send(file=...) — issue #50846."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]

    from plugins.platforms.discord.adapter import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    big = tmp_path / "clip.mp4"
    big.write_bytes(b"x")

    send = AsyncMock(return_value=SimpleNamespace(id=999))
    channel = SimpleNamespace(id=555, guild=None, send=send)
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    original = os.path.getsize

    def fake_getsize(path):
        if str(path) == str(big):
            return _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1
        return original(path)

    os.path.getsize = fake_getsize
    try:
        result = await adapter._send_file_attachment("555", str(big))
    finally:
        os.path.getsize = original

    assert result.success is False
    assert "too large" in (result.error or "").lower()
    assert "clip.mp4" in (result.error or "")
    assert send.await_count == 1
    assert send.await_args is not None
    kwargs = send.await_args.kwargs
    assert "file" not in kwargs and "files" not in kwargs
    assert "Could not attach" in (kwargs.get("content") or "")


@pytest.mark.asyncio
async def test_send_video_respects_guild_filesize_limit(tmp_path):
    """Guild boost limit is honored; files under the higher cap still upload."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]

    video = tmp_path / "ok.mp4"
    video.write_bytes(b"fake-video-bytes")

    sent_msg = SimpleNamespace(
        id=42,
        attachments=[SimpleNamespace(filename="ok.mp4", url="https://cdn.example/ok.mp4")],
    )
    send = AsyncMock(return_value=sent_msg)
    channel = SimpleNamespace(
        id=777,
        guild=SimpleNamespace(filesize_limit=50 * 1024 * 1024),
        send=send,
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send_video("777", str(video))
    assert result.success is True
    assert result.message_id == "42"
    assert send.await_count == 1
    assert send.await_args is not None
    kwargs = send.await_args.kwargs
    assert kwargs.get("file") is not None or kwargs.get("files")


@pytest.mark.asyncio
async def test_send_video_oversized_skips_base_fallback(tmp_path, monkeypatch):
    """Oversized send_video returns failure without falling back to base adapter."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]

    from plugins.platforms.discord.adapter import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    video = tmp_path / "huge.mp4"
    video.write_bytes(b"x")

    send = AsyncMock(return_value=SimpleNamespace(id=1))
    channel = SimpleNamespace(id=1, guild=None, send=send)
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    monkeypatch.setattr(
        os.path,
        "getsize",
        lambda path: (
            _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 10
            if str(path) == str(video)
            else 0
        ),
    )

    base_called = {"yes": False}

    async def boom(*_a, **_k):
        base_called["yes"] = True
        raise AssertionError("base send_video must not run for preflight reject")

    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.send_video",
        boom,
    )

    result = await adapter.send_video("1", str(video))
    assert result.success is False
    assert "too large" in (result.error or "").lower()
    assert base_called["yes"] is False


@pytest.mark.asyncio
async def test_send_multiple_images_skips_oversized_local_file(tmp_path, monkeypatch):
    """Sibling site of #50846: batch image sends must preflight local files too.

    An oversized local image in a chunk previously went straight into
    channel.send(files=...), 413-ing the whole chunk and dumping its siblings
    into the per-image fallback. The oversized file must be skipped up front,
    the rest of the chunk delivered, and a notice appended to the message.
    """
    from plugins.platforms.discord.adapter import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]

    small = tmp_path / "small.png"
    small.write_bytes(b"ok")
    big = tmp_path / "big.png"
    big.write_bytes(b"x")

    send = AsyncMock(return_value=SimpleNamespace(id=7))
    channel = SimpleNamespace(id=9, guild=None, send=send)
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    original = os.path.getsize
    monkeypatch.setattr(
        os.path,
        "getsize",
        lambda path: (
            _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1
            if str(path) == str(big)
            else original(path)
        ),
    )

    await adapter.send_multiple_images(
        "9",
        [(f"file://{small}", ""), (f"file://{big}", "")],
    )

    assert send.await_count == 1
    kwargs = send.await_args.kwargs
    files = kwargs.get("files") or []
    assert len(files) == 1  # only the small image made it
    assert "big.png" in (kwargs.get("content") or "")
    assert "exceeds" in (kwargs.get("content") or "")


@pytest.mark.asyncio
async def test_send_multiple_images_all_oversized_sends_notice(tmp_path, monkeypatch):
    """When every image in the chunk is oversized, the user still gets a notice."""
    from plugins.platforms.discord.adapter import _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES

    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._is_forum_parent = lambda _ch: False  # type: ignore[method-assign]

    big = tmp_path / "only.png"
    big.write_bytes(b"x")

    send = AsyncMock(return_value=SimpleNamespace(id=8))
    channel = SimpleNamespace(id=10, guild=None, send=send)
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(),
    )

    monkeypatch.setattr(
        os.path,
        "getsize",
        lambda path: _DISCORD_DEFAULT_UPLOAD_LIMIT_BYTES + 1,
    )

    await adapter.send_multiple_images("10", [(f"file://{big}", "")])

    assert send.await_count == 1
    kwargs = send.await_args.kwargs
    assert not kwargs.get("files")
    assert "only.png" in (kwargs.get("content") or "")
