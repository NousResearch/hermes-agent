"""Regression tests for #82232.

When the shutdown interrupts an in-flight cron job, the job's own worker
thread tries to deliver an "interrupted" notice — and loses, because
``_bounded_adapter_teardown`` has already closed the transport by the time it
gets there. The notice is dropped, and ``_consume_interrupted_flag`` discards
the resulting ``delivery_error`` with it, so the run's only trace is a line in
jobs.json.

The gateway now sends that notice itself in the post-interrupt phase, while
adapters are still connected — the same window
``_notify_active_sessions_of_shutdown`` uses for chat sessions, which never
saw cron work because cron runs outside ``_running_agents`` (#60432).
"""

from unittest.mock import patch

import pytest

from gateway.config import Platform
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture(autouse=True)
def _reset_cron_running_set():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()


def _telegram_job(job_id="be62d36a9914", name="daily-digest", chat_id="123456"):
    return {
        "id": job_id,
        "name": name,
        "deliver": f"telegram:{chat_id}",
    }


def _telegram_target(chat_id="123456"):
    return {"platform": "telegram", "chat_id": chat_id, "thread_id": None}


def _bind_notifier(runner):
    from gateway.run import GatewayRunner

    runner._notify_interrupted_cron_jobs = (
        GatewayRunner._notify_interrupted_cron_jobs.__get__(runner, GatewayRunner)
    )
    runner._thread_metadata_for_target = (
        GatewayRunner._thread_metadata_for_target.__get__(runner, GatewayRunner)
    )
    return runner


class TestNotifyInterruptedCronJobs:
    @pytest.mark.asyncio
    async def test_owner_is_told_the_run_was_killed(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert len(adapter.sent) == 1
        body = adapter.sent[0]
        assert "daily-digest" in body
        assert "interrupted" in body.lower()
        assert adapter.sent_calls[0][0] == "123456"

    @pytest.mark.asyncio
    async def test_says_restarting_when_restart_was_requested(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        runner._restart_requested = True
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            await runner._notify_interrupted_cron_jobs([job["id"]])

        assert "restarting" in adapter.sent[0]

    @pytest.mark.asyncio
    async def test_local_only_job_stays_silent(self):
        """deliver=local, and deliver=origin with no resolvable origin
        (#43014), resolve to zero targets and must not fall back to a home
        channel."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = {"id": "j1", "name": "local-job", "deliver": "local"}

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets", return_value=[]):
            sent = await runner._notify_interrupted_cron_jobs(["j1"])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_respects_platform_gateway_restart_notification_false(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_failure_deliver_local_suppresses_interrupt_notice(self):
        """Interrupted notices are failure-category engine status (NS-788):
        a job with failure_deliver='local' opted out of failure pings, and
        the shutdown notice must honor that. Real target resolution — no
        _resolve_delivery_targets patch — so the failure_deliver override is
        actually exercised."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = dict(_telegram_job(), failure_deliver="local")

        with patch("cron.jobs.get_job", return_value=job):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_failure_deliver_unset_notice_reaches_deliver_target(self):
        """Control for the suppress test: same job without failure_deliver,
        same real resolution path — the notice goes to the deliver target."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert adapter.sent_calls[0][0] == "123456"

    @pytest.mark.asyncio
    async def test_empty_job_list_is_a_noop(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)

        assert await runner._notify_interrupted_cron_jobs([]) == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_a_raising_adapter_cannot_block_shutdown(self):
        """Best-effort by construction: a wedged adapter must not propagate."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        async def _boom(*_a, **_kw):
            raise RuntimeError("transport already closed")

        adapter.send = _boom

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0

    @pytest.mark.asyncio
    async def test_duplicate_targets_send_once_per_job(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target(), _telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert len(adapter.sent) == 1


class TestShutdownDeliversNoticeBeforeDisconnect:
    @pytest.mark.asyncio
    async def test_notice_is_sent_while_the_adapter_is_still_connected(self, monkeypatch):
        """The whole point is ordering: a notice sent after teardown is lost,
        which is the bug."""
        import cron.scheduler as sched
        import tools.browser_tool as _bt
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt

        runner, adapter = make_restart_runner()
        runner._restart_drain_timeout = 0.01  # force the interrupt path
        sched._running_job_ids.add("be62d36a9914")

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(_bt, "cleanup_all_browsers", lambda: None)

        events: list[str] = []
        real_send = adapter.send

        async def _tracking_send(chat_id, content, reply_to=None, metadata=None):
            if "was interrupted" in content:
                events.append("cron_notice")
            return await real_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        async def _tracking_disconnect():
            events.append("disconnect")

        adapter.send = _tracking_send
        adapter.disconnect = _tracking_disconnect

        with patch("gateway.status.remove_pid_file"), \
             patch("gateway.status.write_runtime_status"), \
             patch("cron.scheduler.mark_job_run"), \
             patch("cron.jobs.get_job", return_value=_telegram_job()), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            await runner.stop()

        assert "cron_notice" in events, "interrupted-cron notice was never sent"
        assert "disconnect" in events
        assert events.index("cron_notice") < events.index("disconnect"), (
            f"notice sent after adapter teardown — it would be lost: {events}"
        )


class TestDeliveryErrorIsRecordedWhenTheNoticeCannotBeSent:
    def test_interrupted_run_records_delivery_error_without_mark_job_run(self):
        """``_consume_interrupted_flag`` short-circuits ``mark_job_run``,
        which used to discard ``delivery_error`` along with it. The recovery
        path must use ``update_job`` so the repeat counter and next_run_at
        bookkeeping that ``mark_job_run`` owns is not run twice for one run.
        """
        import inspect

        import cron.scheduler as sched

        src = inspect.getsource(sched._run_one_job_body)
        assert 'update_job(job["id"], {"last_delivery_error": delivery_error})' in src, (
            "interrupted runs must still persist the delivery failure"
        )
        # The recovery branch hangs off the interrupted-flag short-circuit,
        # not off a second mark_job_run call.
        assert "if interrupted:" in src and "if delivery_error:" in src
