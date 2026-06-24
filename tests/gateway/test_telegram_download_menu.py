import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_module
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Button:
    def __init__(self, text, callback_data=None, **kwargs):
        self.text = text
        self.callback_data = callback_data


class _Markup:
    def __init__(self, rows):
        self.inline_keyboard = rows


def _adapter(monkeypatch):
    monkeypatch.setattr(telegram_module, "InlineKeyboardButton", _Button)
    monkeypatch.setattr(telegram_module, "InlineKeyboardMarkup", _Markup)
    return TelegramAdapter(PlatformConfig(enabled=False, token="test-token"))


def _texts(markup):
    return [[button.text for button in row] for row in markup.inline_keyboard]


class _Msg:
    def __init__(self, text, reply_to_message=None):
        self.text = text
        self.caption = None
        self.reply_to_message = reply_to_message
        self.reply_text = AsyncMock()
        self.chat_id = 12345
        self.message_id = 77
        self.chat = SimpleNamespace(type="private", id=12345, title=None, full_name="Tester Chat")
        self.from_user = SimpleNamespace(id="user-1", first_name="Tester", full_name="Tester")
        self.message_thread_id = None
        self.is_topic_message = False


class _Update:
    def __init__(self, text, reply_to_message=None):
        self.message = _Msg(text, reply_to_message=reply_to_message)
        self.update_id = 99


@pytest.mark.asyncio
async def test_download_command_is_gateway_registered():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("download")

    assert cmd is not None
    assert cmd.gateway_only is False


@pytest.mark.asyncio
async def test_download_command_is_intercepted_by_telegram_adapter(monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter._should_process_message = lambda *args, **kwargs: True
    adapter._ensure_forum_commands = AsyncMock()
    adapter._handle_download_command = AsyncMock()
    adapter.handle_message = AsyncMock()
    update = _Update("/download https://example.com/video")

    await adapter._handle_command(update, AsyncMock())

    adapter._handle_download_command.assert_awaited_once_with(update)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_url_command_opens_menu(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_build_message_event",
        lambda *args, **kwargs: SimpleNamespace(source=SimpleNamespace(thread_id=None), message_id="msg-1"),
    )
    update = _Update("/download https://example.com/video")

    await adapter._handle_download_command(update)

    assert len(adapter._download_menu_state) == 1
    text, kwargs = update.message.reply_text.call_args.args[0], update.message.reply_text.call_args.kwargs
    assert text.startswith("Download options")
    assert kwargs["disable_web_page_preview"] is True
    rows = _texts(kwargs["reply_markup"])
    assert rows[0] == ["* Audio", "Video", "Both"]
    assert rows[1] == ["Run", "Cancel"]


@pytest.mark.asyncio
async def test_download_reply_to_message_uses_url(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_build_message_event",
        lambda *args, **kwargs: SimpleNamespace(source=SimpleNamespace(thread_id=None), message_id="msg-1"),
    )
    replied = SimpleNamespace(text="Check this https://example.com/video", caption=None)
    update = _Update("/download", reply_to_message=replied)

    await adapter._handle_download_command(update)

    assert len(adapter._download_menu_state) == 1
    state = next(iter(adapter._download_menu_state.values()))
    assert state["url"] == "https://example.com/video"
    assert update.message.reply_text.await_count == 1


@pytest.mark.asyncio
async def test_download_callback_updates_mode_and_rerenders(monkeypatch):
    adapter = _adapter(monkeypatch)
    state_id = "1"
    adapter._download_menu_state[state_id] = {
        "url": "https://example.com/video",
        "mode": "audio",
        "event": SimpleNamespace(source=SimpleNamespace(thread_id=None), message_id="msg-1"),
    }

    query = AsyncMock()
    query.message = SimpleNamespace(chat_id=12345)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_download_callback(query, f"dl:mode:{state_id}:video")

    assert adapter._download_menu_state[state_id]["mode"] == "video"
    edited_text, kwargs = query.edit_message_text.call_args.args[0], query.edit_message_text.call_args.kwargs
    assert edited_text.startswith("Download options")
    assert "Mode: video" in edited_text
    assert kwargs["disable_web_page_preview"] is True
    rows = _texts(kwargs["reply_markup"])
    assert rows[0] == ["Audio", "* Video", "Both"]
    query.answer.assert_awaited()


@pytest.mark.asyncio
async def test_download_callback_cleans_up_failed_downloads(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    state_id = "1"
    event = SimpleNamespace(source=SimpleNamespace(thread_id=None), message_id="msg-1")
    adapter._download_menu_state[state_id] = {
        "url": "https://example.com/video",
        "mode": "audio",
        "event": event,
    }

    download_dir = tmp_path / "partial"
    download_dir.mkdir()

    fake_runner = SimpleNamespace(
        _download_media_from_url=AsyncMock(
            return_value={
                "ok": False,
                "error": "yt-dlp audio download failed",
                "download_dir": str(download_dir),
            }
        )
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: fake_runner)

    query = AsyncMock()
    query.message = SimpleNamespace(chat_id=12345)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_download_callback(query, f"dl:run:{state_id}")

    assert not download_dir.exists()
    query.edit_message_text.assert_awaited()
    assert state_id in adapter._download_menu_state


@pytest.mark.asyncio
async def test_download_callback_runs_and_sends_audio_and_video(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    state_id = "1"
    event = SimpleNamespace(source=SimpleNamespace(thread_id="topic-1"), message_id="msg-1")
    adapter._download_menu_state[state_id] = {
        "url": "https://example.com/video",
        "mode": "both",
        "event": event,
    }

    audio_dir = tmp_path / "audio"
    video_dir = tmp_path / "video"
    audio_dir.mkdir()
    video_dir.mkdir()
    audio_path = audio_dir / "audio.mp3"
    video_path = video_dir / "video.mp4"
    audio_path.write_bytes(b"audio-bytes")
    video_path.write_bytes(b"video-bytes")

    fake_runner = SimpleNamespace(
        _download_media_from_url=AsyncMock(
            return_value={
                "ok": True,
                "kind": "both",
                "downloads": [
                    {
                        "ok": True,
                        "kind": "audio",
                        "file_path": str(audio_path),
                        "download_dir": str(audio_dir),
                    },
                    {
                        "ok": True,
                        "kind": "video",
                        "file_path": str(video_path),
                        "download_dir": str(video_dir),
                    },
                ],
            }
        )
    )
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: fake_runner)

    adapter.send_voice = AsyncMock(return_value=SimpleNamespace(success=True, message_id="v1"))
    adapter.send_video = AsyncMock(return_value=SimpleNamespace(success=True, message_id="v2"))
    adapter.send_document = AsyncMock(return_value=SimpleNamespace(success=True, message_id="d1"))

    query = AsyncMock()
    query.message = SimpleNamespace(chat_id=12345)
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()

    await adapter._handle_download_callback(query, f"dl:run:{state_id}")

    fake_runner._download_media_from_url.assert_awaited_once_with("https://example.com/video", "both")
    adapter.send_voice.assert_awaited_once()
    adapter.send_video.assert_awaited_once()
    adapter.send_document.assert_not_awaited()
    assert not audio_dir.exists()
    assert not video_dir.exists()
    assert state_id not in adapter._download_menu_state
