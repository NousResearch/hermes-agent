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

import pytest

from cron.scheduler_script import _DEFAULT_MEDIA_SEND_TIMEOUT, _get_media_send_timeout
from cron.scheduler_delivery import _send_media_via_adapter


class TestMediaSendTimeoutResolution:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", raising=False)
        monkeypatch.setattr("cron.scheduler.load_config", dict)
        assert _get_media_send_timeout() == _DEFAULT_MEDIA_SEND_TIMEOUT

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
        monkeypatch.setattr("cron.scheduler.load_config", dict)
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

        monkeypatch.setattr("agent.async_utils.WithdrawableDispatch.schedule", boom)

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


@pytest.mark.parametrize("stage", ["unstarted", "started", "completed"])
def test_media_timeout_preserves_dispatch_and_fallback_contract(
    tmp_path, monkeypatch, stage
):
    import asyncio
    import threading
    from types import SimpleNamespace

    from cron import scheduler_delivery, scheduler_script
    from gateway.config import Platform

    media = tmp_path / "media.png"
    media.write_bytes(b"png")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    monkeypatch.setattr(scheduler_script, "_get_media_send_timeout", lambda: 2.0)
    loop = asyncio.new_event_loop()
    released = threading.Event()
    blocked = threading.Event()
    events = []

    def block_loop():
        blocked.set()
        assert released.wait(timeout=15)

    class Adapter:
        platform = Platform.TELEGRAM

        async def send_image_file(self, **kwargs):
            events.append("started")
            try:
                if stage == "started":
                    while not released.is_set():
                        await asyncio.sleep(0.01)
                events.append("finished")
                return SimpleNamespace(success=True)
            except asyncio.CancelledError:
                events.append("cancelled")
                raise

    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    if stage == "unstarted":
        loop.call_soon_threadsafe(block_loop)
        assert blocked.wait(timeout=5)
    target = scheduler_delivery._TargetDelivery(
        job={"id": "media-dispatch"},
        platform=Platform.TELEGRAM,
        platform_name="telegram",
        chat_id="chat",
        thread_id=None,
        transport=None,
        pconfig=None,
        runtime_adapter=Adapter(),
        target_adapters={},
        config=None,
        loop=loop,
        notify_delivery=False,
        origin={},
        origin_target=False,
        origin_user_id=None,
        is_dm_target=False,
        mirror_text="",
        mirror_this_target=False,
        in_channel_surface=False,
        inchannel_continuable=False,
        opened_thread_id=None,
    )
    try:
        missing = scheduler_delivery._live_send_media(
            target, {}, [(str(media), False)], [], []
        )
        released.set()

        async def settle():
            await asyncio.sleep(0.05)

        asyncio.run_coroutine_threadsafe(settle(), loop).result(timeout=5)
    finally:
        released.set()
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    expected = (
        ([(str(media), False)], [])
        if stage == "unstarted"
        else ([], ["started", "finished"])
    )
    assert (missing, events) == expected
