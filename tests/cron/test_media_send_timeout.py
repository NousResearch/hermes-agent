"""Cron media-send timeout resolution and failure-reason formatting.

Covers two salvaged fixes:

- PR #87965 (@AiwendilInTheWoods): an argument-less exception (notably
  TimeoutError from ``future.result(timeout=...)``) has an empty ``str()``,
  which used to render "failed to send media <path>: " with no reason at
  all — in both the log line and the delivery error recorded on the run.
- PR #87967 (@AiwendilInTheWoods): the per-attachment send timeout was a
  hardcoded 30s; large attachments (long TTS audio, big exports) failed on
  slow uplinks with no way to raise it. Now resolved via
  HERMES_CRON_MEDIA_SEND_TIMEOUT → cron.media_send_timeout_seconds → 300s.
"""

import dataclasses

import pytest

from cron.scheduler_script import _DEFAULT_MEDIA_SEND_TIMEOUT, _get_media_send_timeout
from cron.scheduler_delivery import (
    _TargetDelivery, _send_media_via_adapter, _standalone_send)


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
        return errors[0]

    def test_timeout_error_names_the_class(self, tmp_path, monkeypatch):
        # TimeoutError() has an empty str() — the recorded reason must not
        # be blank (the trailing-colon-nothing log from the field report).
        err = self._run(tmp_path, monkeypatch, TimeoutError())
        assert err.rstrip() != f"failed to send media {tmp_path / 'clip.mp3'}:"
        assert "TimeoutError" in err

    def test_exception_with_message_keeps_it(self, tmp_path, monkeypatch):
        err = self._run(tmp_path, monkeypatch, RuntimeError("bridge closed"))
        assert "bridge closed" in err


class TestStandaloneFallbackTimeout:
    """The standalone lane's RuntimeError fallback carries the same attachments.

    #88787 made only ``_send_media_via_adapter`` configurable. ``_standalone_send``
    retries in a thread when ``asyncio.run`` refuses inside a running loop, and that
    retry passes ``media_files`` straight through — so a large attachment still hit a
    hardcoded 30s there while the live lane honoured the configured budget.
    """

    @staticmethod
    def _drive(monkeypatch, media_files, config):
        """Force _standalone_send down its fallback; return the timeout it asked for."""
        recorded = {}

        class _Coro:
            def close(self):
                pass

        def _fake_send_to_platform(*a, **kw):
            return _Coro()

        def _refuses(coro):
            raise RuntimeError("asyncio.run() cannot be called from a running event loop")

        class _Future:
            def result(self, timeout=None):
                recorded["timeout"] = timeout
                return "sent"

        class _Pool:
            def __init__(self, *a, **kw):
                pass

            def submit(self, *a, **kw):
                return _Future()

            def shutdown(self, *a, **kw):
                pass

        monkeypatch.setattr("tools.send_message_tool._send_to_platform", _fake_send_to_platform)
        monkeypatch.setattr("cron.scheduler_delivery.asyncio.run", _refuses)
        monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", _Pool)
        monkeypatch.setattr("cron.scheduler._interpreter_shutting_down", lambda *a: False)
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr("cron.scheduler.load_config", lambda: config)

        # Build a real _TargetDelivery so ``where`` stays the dataclass property;
        # fill every required field with None and set only what this lane reads.
        t = _TargetDelivery(**{
            f.name: None for f in dataclasses.fields(_TargetDelivery)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        })
        t.job = {"id": "job-x"}
        t.platform_name = "matrix"
        t.chat_id = "!room:example.org"
        t.pconfig = {}
        result, err = _standalone_send(t, "briefing", media_files)
        assert (result, err) == ("sent", None)
        return recorded["timeout"]

    def test_media_uses_configured_timeout(self, monkeypatch):
        got = self._drive(
            monkeypatch, [("/tmp/briefing.mp3", True)],
            {"cron": {"media_send_timeout_seconds": 900}},
        )
        assert got == 900

    def test_media_uses_default_when_unconfigured(self, monkeypatch):
        got = self._drive(monkeypatch, [("/tmp/briefing.mp3", True)], {})
        assert got == _DEFAULT_MEDIA_SEND_TIMEOUT == 300

    def test_text_only_keeps_short_timeout(self, monkeypatch):
        # No attachment, no reason to wait five minutes on a hung text send.
        got = self._drive(monkeypatch, [], {"cron": {"media_send_timeout_seconds": 900}})
        assert got == 30
