"""Cron media delivery forwards the voice flag to ``send_voice`` (#132120): the adapter decides
voice bubble vs plain audio from ``is_voice``, matching the gateway dispatch."""
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler_delivery import _send_media_via_adapter


class TestSendMediaVoiceFlag:
    def _safe_media_path(self, tmp_path, monkeypatch, name, data=b"media"):
        root = tmp_path / "media-cache"
        media_file = root / name
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(data)
        monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
        return media_file.resolve()

    @staticmethod
    def _run_with_loop(adapter, chat_id, media_files, metadata, job):
        def fake_run_coro(coro, _loop):
            coro.close()
            completed = Future()
            completed.set_result(MagicMock(success=True))
            return completed

        with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            _send_media_via_adapter(adapter, chat_id, media_files, metadata, MagicMock(), job)

    def test_voice_flag_forwarded_to_send_voice(self, tmp_path, monkeypatch):
        """Issue #132120: a ``[[audio_as_voice]]`` media file must reach the adapter —
        Telegram decides voice-bubble vs sendAudio from the ``is_voice`` kwarg, so a
        cron .mp3 without the flag arrives as a music file, not a bubble."""
        adapter = MagicMock()
        adapter.platform = "telegram"
        adapter.send_voice = AsyncMock()
        voice_path = self._safe_media_path(tmp_path, monkeypatch, "voice.mp3")
        self._run_with_loop(adapter, "123", [(str(voice_path), True)], None, {"id": "j4"})
        adapter.send_voice.assert_called_once()
        assert adapter.send_voice.call_args.kwargs["is_voice"] is True

    def test_audio_attachment_without_voice_flag_forwards_false(self, tmp_path, monkeypatch):
        """A plain audio-ext attachment on Telegram also routes to send_voice (the sendAudio
        lane); it must forward ``is_voice=False`` so it is never turned into a bubble."""
        adapter = MagicMock()
        adapter.platform = "telegram"
        adapter.send_voice = AsyncMock()
        audio_path = self._safe_media_path(tmp_path, monkeypatch, "clip.mp3")
        self._run_with_loop(adapter, "123", [(str(audio_path), False)], None, {"id": "j5"})
        adapter.send_voice.assert_called_once()
        assert adapter.send_voice.call_args.kwargs["is_voice"] is False

    def test_non_voice_methods_do_not_receive_is_voice(self, tmp_path, monkeypatch):
        """Only the voice sender takes the flag; image sends must not see it."""
        adapter = MagicMock()
        adapter.platform = "telegram"
        adapter.send_image_file = AsyncMock()
        photo_path = self._safe_media_path(tmp_path, monkeypatch, "photo.jpg")
        self._run_with_loop(adapter, "123", [(str(photo_path), False)], None, {"id": "j6"})
        adapter.send_image_file.assert_called_once()
        assert "is_voice" not in adapter.send_image_file.call_args.kwargs
