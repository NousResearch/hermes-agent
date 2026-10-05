"""Live-adapter confirmation timeout in ``_deliver_result``: an in-flight send is left running and
not duplicated, a never-started send falls back to standalone."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result


class _ConfirmationTimesOut:
    """The real ``run_coroutine_threadsafe`` future, except that ``result()`` raises
    ``TimeoutError`` as soon as ``moment`` is set instead of after the confirmation bound."""

    def __init__(self, future, moment):
        self._future = future
        self._moment = moment

    def result(self, timeout=None):
        assert self._moment.wait(5)
        raise TimeoutError

    def __getattr__(self, name):
        return getattr(self._future, name)


class TestDeliverResultTimeoutCancelsFuture:
    """When the live adapter's confirmation outlasts the wait, the outcome depends on whether the
    send had STARTED on the gateway loop. Started: it is in flight (a paced multi-chunk send can
    legitimately outlast the wait) — leave it running and skip the standalone fallback, which would
    duplicate it (#38922). Never started (wedged loop): nothing was sent, so fall through to
    standalone or the message is silently dropped. ``future.cancel()`` cannot tell the two apart:
    a run_coroutine_threadsafe future stays PENDING until done, so cancel() returns True mid-send
    and kills it — these tests drive a real loop for that reason.
    """

    def _deliver(self, monkeypatch, adapter, loop):
        from cron import scheduler_delivery
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_SECS", 0.3)
        job = {"id": "timeout-job", "deliver": "origin", "origin": {"platform": "telegram", "chat_id": "123"}}
        standalone_send = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=mock_cfg), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(job, "Hello world", adapters={Platform.TELEGRAM: adapter}, loop=loop)
        return result, standalone_send

    @pytest.mark.parametrize("late_error", [None, RuntimeError("late boom"), "unconfirmed"])
    def test_in_flight_send_outlasting_the_wait_keeps_running_and_is_not_duplicated(
            self, monkeypatch, caplog, late_error):
        import asyncio
        import logging
        import threading
        from cron import scheduler_delivery_live

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        events = []
        started, observed = threading.Event(), threading.Event()
        release = asyncio.Event()

        async def slow_send(chat_id, content, **_kw):
            events.append("started")
            started.set()
            await release.wait()  # held until the confirmation wait has timed out
            events.append("finished")
            if isinstance(late_error, Exception):
                raise late_error
            if late_error == "unconfirmed":
                return None  # passes the router, rejected by _confirm_adapter_delivery
            return MagicMock(success=True, message_id="m1", raw_response=None)

        real_schedule = asyncio.run_coroutine_threadsafe

        def schedule(coro, target_loop):
            return _ConfirmationTimesOut(real_schedule(coro, target_loop), started)

        real_observe = scheduler_delivery_live._observe_late_live_send

        def observe(*args):
            real_observe(*args)
            observed.set()

        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", schedule)
        monkeypatch.setattr(scheduler_delivery_live, "_observe_late_live_send", observe)
        adapter = MagicMock()
        adapter.send = slow_send
        update_job = MagicMock()
        monkeypatch.setattr("cron.jobs.update_job", update_job)
        try:
            with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
                result, standalone_send = self._deliver(monkeypatch, adapter, loop)
                loop.call_soon_threadsafe(release.set)
                assert observed.wait(5), "the late outcome of the in-flight send was never observed"
        finally:
            loop.call_soon_threadsafe(release.set)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            loop.close()
        assert result is None, f"expected the in-flight send to count as delivered, got {result!r}"
        # The late outcome is still observed: a send that fails or comes back unconfirmed is logged.
        assert ("failed after confirmation timeout" in caplog.text) == isinstance(late_error, Exception)
        assert ("returned an unconfirmed result" in caplog.text) == (late_error == "unconfirmed")
        standalone_send.assert_not_awaited()
        assert events == ["started", "finished"], "the in-flight send must not be cancelled mid-way"
        update_job.assert_called_once_with("timeout-job", {"last_delivery_unverified": ["telegram:123"]})

    def test_send_that_never_started_falls_back_to_standalone(self, monkeypatch):
        import asyncio
        import threading
        import time

        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        loop.call_soon_threadsafe(time.sleep, 0.8)  # wedge the running loop past the 0.3s wait
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=MagicMock(success=True))
        try:
            result, standalone_send = self._deliver(monkeypatch, adapter, loop)
            time.sleep(0.8)  # the loop un-wedges: the abandoned send must still never go out
        finally:
            loop.call_soon_threadsafe(loop.stop)
        assert result is None, f"standalone should have delivered, got {result!r}"
        standalone_send.assert_awaited_once()
        adapter.send.assert_not_awaited()
