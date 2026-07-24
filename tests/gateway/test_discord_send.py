import asyncio
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
async def test_send_retries_without_reference_when_reply_target_is_system_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    sent_msg = SimpleNamespace(id=1234)
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        if len(send_calls) == 1:
            raise RuntimeError(
                "400 Bad Request (error code: 50035): Invalid Form Body\n"
                "In message_reference: Cannot reply to a system message"
            )
        return sent_msg

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    assert result.success is True
    assert result.message_id == "1234"
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 2
    ref_msg.to_reference.assert_called_once_with(fail_if_not_exists=False)
    assert send_calls[0]["reference"] is reference_obj
    assert send_calls[1]["reference"] is None


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
    assert channel.fetch_message.await_count == 1
    assert channel.send.await_count == 3
    ref_msg.to_reference.assert_called_once_with(fail_if_not_exists=False)
    assert send_calls[0]["reference"] is reference_obj
    assert send_calls[1]["reference"] is None
    assert send_calls[2]["reference"] is None


@pytest.mark.asyncio
async def test_send_does_not_retry_on_unrelated_errors():
    """Regression guard: errors unrelated to the reply reference (e.g. 50013
    Missing Permissions) must NOT trigger the no-reference retry path — they
    should propagate out of the per-chunk loop and surface as a failed
    SendResult so the caller sees the real problem instead of a silent retry.
    """
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    reference_obj = object()
    ref_msg = SimpleNamespace(id=99, to_reference=MagicMock(return_value=reference_obj))
    send_calls = []

    async def fake_send(*, content, reference=None):
        send_calls.append({"content": content, "reference": reference})
        raise RuntimeError(
            "403 Forbidden (error code: 50013): Missing Permissions"
        )

    channel = SimpleNamespace(
        fetch_message=AsyncMock(return_value=ref_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("555", "hello", reply_to="99")

    # Outer except in adapter.send() wraps propagated errors as SendResult.
    assert result.success is False
    assert "50013" in (result.error or "")
    # Only the first attempt happens — no reference-retry replay.
    assert channel.send.await_count == 1
    assert send_calls[0]["reference"] is reference_obj


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

    def test_type_value_15(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=15)
        assert adapter._is_forum_parent(ch) is True

    def test_regular_channel_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=0)
        assert adapter._is_forum_parent(ch) is False

    def test_thread_returns_false(self):
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
        ch = SimpleNamespace(type=11)  # public thread
        assert adapter._is_forum_parent(ch) is False


@pytest.mark.asyncio
async def test_send_to_forum_creates_thread_post():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    # thread object has no 'send' so _send_to_forum uses thread.thread
    thread_ch = SimpleNamespace(id=555, send=AsyncMock(return_value=SimpleNamespace(id=600)))
    thread = SimpleNamespace(
        id=555,
        message=SimpleNamespace(id=500),
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "Hello forum!")

    assert result.success is True
    assert result.message_id == "500"
    forum_channel.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_to_forum_sends_remaining_chunks():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    # Force a small max message length so the message splits
    adapter.MAX_MESSAGE_LENGTH = 20

    chunk_msg_1 = SimpleNamespace(id=500)
    chunk_msg_2 = SimpleNamespace(id=501)
    thread_ch = SimpleNamespace(
        id=555,
        send=AsyncMock(return_value=chunk_msg_2),
    )
    # thread object has no 'send' so _send_to_forum uses thread.thread
    thread = SimpleNamespace(
        id=555,
        message=chunk_msg_1,
        thread=thread_ch,
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "A" * 50)

    assert result.success is True
    assert result.message_id == "500"
    # Should have sent at least one follow-up chunk
    assert thread_ch.send.await_count >= 1


@pytest.mark.asyncio
async def test_send_to_forum_create_thread_failure():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(side_effect=Exception("rate limited"))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    result = await adapter.send("999", "Hello forum!")

    assert result.success is False
    assert "rate limited" in result.error



# ---------------------------------------------------------------------------
# Forum follow-up chunk failure reporting + media on forum paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_forum_follow_up_chunk_failures_collected_as_warnings():
    """Partial-send chunk failures surface in raw_response['warnings']."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.MAX_MESSAGE_LENGTH = 20

    chunk_msg_1 = SimpleNamespace(id=500)
    # Every follow-up chunk fails — we should collect a warning per failure
    thread_ch = SimpleNamespace(
        id=555,
        send=AsyncMock(side_effect=Exception("rate limited")),
    )
    thread = SimpleNamespace(id=555, message=chunk_msg_1, thread=thread_ch)
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.name = "ideas"
    forum_channel.create_thread = AsyncMock(return_value=thread)
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: forum_channel,
        fetch_channel=AsyncMock(),
    )

    # Long enough to produce multiple chunks
    result = await adapter.send("999", "A" * 60)

    # Starter message (first chunk) was delivered via create_thread, so send is
    # successful overall — but follow-up chunks all failed and are reported.
    assert result.success is True
    assert result.message_id == "500"
    warnings = (result.raw_response or {}).get("warnings") or []
    assert len(warnings) >= 1
    assert all("rate limited" in w for w in warnings)


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
async def test_forum_post_file_uses_filename_when_no_content():
    """Thread name falls back to file.filename when no content is provided."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    thread = SimpleNamespace(
        id=1,
        message=SimpleNamespace(
            id=2,
            attachments=[SimpleNamespace(filename="voice-message.ogg")],
        ),
        thread=SimpleNamespace(id=1, send=AsyncMock()),
    )
    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 10
    forum_channel.name = "forum"
    forum_channel.create_thread = AsyncMock(return_value=thread)

    fake_file = SimpleNamespace(filename="voice-message.ogg")
    result = await adapter._forum_post_file(forum_channel, content="", file=fake_file)

    assert result.success is True
    call_kwargs = forum_channel.create_thread.await_args.kwargs
    # Content was empty → thread name derived from filename
    assert call_kwargs["name"] == "voice-message.ogg"


@pytest.mark.asyncio
async def test_forum_post_file_creation_failure():
    """_forum_post_file returns a failed SendResult when create_thread raises."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))

    forum_channel = _discord_mod.ForumChannel()
    forum_channel.id = 999
    forum_channel.create_thread = AsyncMock(side_effect=Exception("missing perms"))

    result = await adapter._forum_post_file(
        forum_channel,
        content="hi",
        file=SimpleNamespace(filename="x.png"),
    )

    assert result.success is False
    assert "missing perms" in (result.error or "")


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
async def test_typing_task_removed_after_api_error():
    """When typing API call fails, stale task must be removed so typing can restart."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._client.http.request = AsyncMock(side_effect=Exception("rate limited"))
    adapter._typing_tasks = {}

    await adapter.send_typing("12345")
    await asyncio.sleep(0.1)

    assert "12345" not in adapter._typing_tasks, \
        "Stale task should be removed after API error"


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


@pytest.mark.asyncio
async def test_typing_stop_cleans_up():
    """stop_typing should remove the task from _typing_tasks."""
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = MagicMock()
    adapter._client.http = MagicMock()
    adapter._client.http.request = AsyncMock()
    adapter._typing_tasks = {}

    await adapter.send_typing("12345")
    assert "12345" in adapter._typing_tasks

    await adapter.stop_typing("12345")
    assert "12345" not in adapter._typing_tasks


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
async def test_deliver_media_from_response_routes_mp4_to_send_video(tmp_path, monkeypatch):
    """Streaming/post-stream dispatch must call send_video for MEDIA:.mp4."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.run import GatewayRunner

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-mp4")
    image = tmp_path / "figure.png"
    image.write_bytes(b"fake-png")

    # Allow delivery from tmp_path in non-strict mode (default).
    monkeypatch.chdir(tmp_path)

    adapter = SimpleNamespace(
        name="Discord",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="v")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="d")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="i")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="vid")),
        send_multiple_images=AsyncMock(),
    )
    event = SimpleNamespace(
        source=SimpleNamespace(
            platform="discord",
            chat_id="chat-1",
            thread_id=None,
        )
    )
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: {},
        _reply_anchor_for_event=lambda event: None,
    )
    response = (
        f"Here is the figure:\n\nMEDIA:{image}\n\n"
        f"And the clip:\n\nMEDIA:{video}\n"
    )

    await GatewayRunner._deliver_media_from_response(runner, response, event, adapter)

    adapter.send_video.assert_awaited_once()
    sent_path = adapter.send_video.await_args.kwargs["video_path"]
    assert Path(sent_path).resolve() == video.resolve()
    adapter.send_multiple_images.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_forum_send_video_fails_loud_when_starter_has_no_attachments(tmp_path, monkeypatch):
    """Forum-parent send_video must fail loud when the starter message drops attachments."""
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
        message=SimpleNamespace(id=8, attachments=[]),
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

    assert result.success is False
    assert "no files" in (result.error or "").lower()
    forum_channel.create_thread.assert_awaited_once()
