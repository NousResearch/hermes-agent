"""Regression tests for the Telegram polling health-confirmation log (#90504).

Both reconnect paths end on ``health pending getUpdates progress`` and
``_record_polling_progress`` used to complete silently, so the log stream for
"reconnected and healthy" was byte-identical to "reconnected and hung". The
first confirmed getUpdates round-trip of each generation now emits an INFO
line, turning the pending line into a resolvable pair whose *absence* after a
reconnect is a reliable hung-poll signature.
"""

import asyncio
import logging
from types import SimpleNamespace
from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_teardown_started = False
    a._polling_progress_accepting = True
    a._polling_generation = 1
    a._polling_progress_event = asyncio.Event()
    a._polling_network_error_count = 2
    a._polling_conflict_count = 3
    a._polling_conflict_recovery_generation = None
    a._send_path_degraded = True
    return a


class TestPollingHealthConfirmation:




    def test_stale_generation_progress_stays_silent(self, caplog):
        """Progress from an abandoned generation must neither log nor set the
        current event (pre-existing guard, pinned here because the log line
        must inherit the same generation-scoping)."""
        a = _bare_adapter()
        a._polling_generation = 2
        a._polling_progress_event = asyncio.Event()
        with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
            a._record_polling_progress(1)
        assert not [rec for rec in caplog.records if rec.levelno == logging.INFO]
        assert not a._polling_progress_event.is_set()

    def test_degraded_health_confirmation_replays_failed_final_responses(self):
        async def exercise():
            a = _bare_adapter()
            a._running = True
            a.gateway_runner = type("Runner", (), {})()
            owned = set()
            a.gateway_runner._retain_background_task = lambda task: (owned.add(task), task.add_done_callback(owned.discard), task)[-1]
            calls = []

            async def redeliver(platform, *, profile=None):
                calls.append((platform, profile))

            a.gateway_runner._redeliver_failed_obligations_for_platform = redeliver
            a._write_runtime_status_safe = lambda *args, **kwargs: None
            a._owner_profile = "default"

            assert a._record_polling_progress(1) is True
            await asyncio.sleep(0)
            assert calls == [(Platform.TELEGRAM, "default")]
            assert a._send_path_degraded is False

        asyncio.run(exercise())

    def test_health_replay_task_is_gateway_owned_and_logs_failure(self, caplog):
        async def exercise():
            a = _bare_adapter()
            a._running = True
            owned = set()
            runner = type("Runner", (), {})()

            def retain(task):
                owned.add(task)
                task.add_done_callback(owned.discard)
                return task

            runner._retain_background_task = retain
            a.gateway_runner = runner

            async def redeliver(*_args, **_kwargs):
                raise RuntimeError("offline replay probe")

            runner._redeliver_failed_obligations_for_platform = redeliver
            a._write_runtime_status_safe = lambda *args, **kwargs: None
            a._owner_profile = "default"
            with caplog.at_level(logging.WARNING, logger="plugins.platforms.telegram.adapter"):
                assert a._record_polling_progress(1) is True
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            assert not owned
            assert any("delivery replay after polling recovery failed" in r.message for r in caplog.records)

        asyncio.run(exercise())

    def test_mark_failed_after_health_replays_once_on_same_healthy_adapter(self):
        async def exercise():
            class FakeAdapter(BasePlatformAdapter):
                async def connect(self, **kwargs): return True
                async def disconnect(self): return None
                async def get_chat_info(self, chat_id): return {}
                async def send(self, *args, **kwargs): return SendResult(success=False, error="send_path_degraded")

            adapter = FakeAdapter.__new__(FakeAdapter)
            delivery_adapter = FakeAdapter.__new__(FakeAdapter)
            delivery_adapter._send_path_degraded = False
            delivery_adapter._owner_profile = "default"
            calls = []
            runner = SimpleNamespace()

            async def redeliver(platform, *, profile=None):
                calls.append((platform, profile))

            runner._redeliver_failed_obligations_for_platform = redeliver
            adapter.gateway_runner = runner
            adapter._final_delivery_adapter = lambda source: delivery_adapter
            source = SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm", user_id="u1")
            event = MessageEvent(text="probe", source=source, message_id="m-probe")
            result = SendResult(success=False, error="send_path_degraded")
            await adapter._finalize_delivery_obligation("obligation-probe", result, event, delivery_adapter)
            assert calls == [(Platform.TELEGRAM, "default")]

        asyncio.run(exercise())
