"""Cron media-send timeout resolution and failure-reason formatting.

Covers salvaged fixes:

- PR #87965 (@AiwendilInTheWoods): an argument-less exception (notably
  TimeoutError from ``future.result(timeout=...)``) has an empty ``str()``,
  which used to render "failed to send media <path>: " with no reason at
  all — in both the log line and the delivery error recorded on the run.
- PR #87967 (@AiwendilInTheWoods): the per-attachment send timeout was a
  hardcoded 30s; large attachments (long TTS audio, big exports) failed on
  slow uplinks with no way to raise it. Now resolved via
  HERMES_CRON_MEDIA_SEND_TIMEOUT → cron.media_send_timeout_seconds → 300s.
- In-flight media confirmation timeouts must not be retried via standalone
  fallback when ``future.cancel()`` returns False (dispatch already started).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from cron.scheduler_script import _DEFAULT_MEDIA_SEND_TIMEOUT, _get_media_send_timeout
from cron.scheduler_delivery import _send_media_via_adapter


class TestMediaSendTimeoutResolution:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: {})
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT == 300

    def test_env_wins(self, monkeypatch):
        monkeypatch.setenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "45")
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": 900}},
        )
        assert _get_media_send_timeout() == 45

    def test_config_value(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": 900}},
        )
        assert _get_media_send_timeout() == 900

    @pytest.mark.parametrize("bad", ["abc", "-5", "0", ""])
    def test_invalid_env_falls_back(self, monkeypatch, bad):
        monkeypatch.setenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", bad)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: {})
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT

    def test_invalid_config_falls_back(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr(
            "cron.scheduler.load_config",
            lambda: {"cron": {"media_send_timeout_seconds": "nope"}},
        )
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT


class TestEmptyReasonFallback:
    def _run(self, tmp_path, monkeypatch, exc):
        """Drive _send_media_via_adapter into its generic except handler."""
        media = tmp_path / "clip.mp3"
        media.write_bytes(b"x")

        monkeypatch.setattr(
            "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
            staticmethod(lambda files, session_key="": [(str(media), False)]),
        )

        def boom(coro, loop):
            coro.close()
            raise exc

        monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", boom)

        class _Adapter:
            async def send_voice(self, **kw):  # pragma: no cover - never awaited
                pass

        errors = _send_media_via_adapter(
            _Adapter(), "C123", [(str(media), False)], None, loop=object(),
            job={"id": "job-x"},
        )
        assert len(errors) == 1
        return errors[0][2]

    def test_timeout_error_names_the_class(self, tmp_path, monkeypatch):
        # TimeoutError() has an empty str() — the recorded reason must not
        # be blank (the trailing-colon-nothing log from the field report).
        err = self._run(tmp_path, monkeypatch, TimeoutError())
        assert err.rstrip() != f"failed to send media {tmp_path / 'clip.mp3'}:"
        assert "TimeoutError" in err

    def test_exception_with_message_keeps_it(self, tmp_path, monkeypatch):
        err = self._run(tmp_path, monkeypatch, RuntimeError("bridge closed"))
        assert "bridge closed" in err


class TestMediaInflightFallbackList:
    """Standalone fallback must retry only cancel-proven-not-dispatched attachments."""

    def _media_path(self, tmp_path, monkeypatch, name):
        root = tmp_path / "media-cache"
        media_file = root / name
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"media")
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
            (root,),
        )
        return str(media_file.resolve())

    def test_cancel_fail_omits_file_from_standalone_fallback(self, tmp_path, monkeypatch):
        from concurrent.futures import Future
        from gateway.config import Platform

        ok_path = self._media_path(tmp_path, monkeypatch, "ok.jpg")
        inflight_path = self._media_path(tmp_path, monkeypatch, "inflight.jpg")
        content = f"brief\nMEDIA:{inflight_path}\nMEDIA:{ok_path}"

        adapter = MagicMock()
        adapter.send_image_file = AsyncMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="1"))

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = {}
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.get_home_channel.return_value = None

        loop = MagicMock()
        loop.is_running.return_value = True

        text_future = Future()
        text_future.set_result(MagicMock(success=True, message_id="1"))

        inflight_future = MagicMock()
        inflight_future.result.side_effect = TimeoutError("timed out")
        inflight_future.cancel.return_value = False

        ok_future = Future()
        ok_future.set_result(MagicMock(success=True))

        schedule_calls = []

        def fake_schedule(coro, _loop):
            coro.close()
            schedule_calls.append(1)
            if len(schedule_calls) == 1:
                return text_future
            if len(schedule_calls) == 2:
                return inflight_future
            return ok_future

        standalone_send = AsyncMock(return_value={"success": True})
        job = {
            "id": "media-inflight-fallback",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("agent.async_utils.safe_schedule_threadsafe", side_effect=fake_schedule), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(
                job, content, adapters={Platform.TELEGRAM: adapter}, loop=loop)

        assert result is None
        standalone_send.assert_not_awaited()
        assert len(schedule_calls) == 3, "text + two media sends should run"

    def test_cancel_success_includes_file_in_standalone_fallback(self, tmp_path, monkeypatch):
        from concurrent.futures import Future
        from gateway.config import Platform

        wedged_path = self._media_path(tmp_path, monkeypatch, "wedged.jpg")
        content = f"brief\nMEDIA:{wedged_path}"

        adapter = MagicMock()
        adapter.send_image_file = AsyncMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="1"))

        pconfig = MagicMock()
        pconfig.enabled = True
        pconfig.extra = {}
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        mock_cfg.get_home_channel.return_value = None

        loop = MagicMock()
        loop.is_running.return_value = True

        text_future = Future()
        text_future.set_result(MagicMock(success=True, message_id="1"))

        wedged_future = Future()
        wedged_future.result = MagicMock(side_effect=TimeoutError("timed out"))

        def fake_schedule(coro, _loop):
            coro.close()
            if not hasattr(adapter, "_text_scheduled"):
                adapter._text_scheduled = True
                return text_future
            return wedged_future

        standalone_calls = []

        async def fake_standalone(*args, **kwargs):
            standalone_calls.append(kwargs.get("media_files"))
            return {"success": True}

        job = {
            "id": "media-wedged-fallback",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }

        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("agent.async_utils.safe_schedule_threadsafe", side_effect=fake_schedule), \
             patch("tools.send_message_tool._send_to_platform", new=fake_standalone):
            result = _deliver_result(
                job, content, adapters={Platform.TELEGRAM: adapter}, loop=loop)

        assert result is None
        assert standalone_calls, "standalone fallback should run for wedged media"
        retried_paths = [path for path, _voice in standalone_calls[0]]
        assert wedged_path in retried_paths
