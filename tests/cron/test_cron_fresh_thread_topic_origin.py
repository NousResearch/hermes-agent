"""``platforms.telegram.extra.cron_fresh_thread_per_run`` lets a topic-born continuable cron job
open a fresh thread per run (#76548). Off by default: such a job keeps delivering to its origin
topic, the guard b693bee100bd added. Drives the production ``_deliver_result`` with a live adapter
whose ``create_handoff_thread`` is the only thread primitive."""

from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform


def _deliver(extra):
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True, message_id="m1", raw_response=None)
    adapter.create_handoff_thread = AsyncMock(return_value="42")
    adapter._session_store = MagicMock()
    pconfig = MagicMock(enabled=True, extra=extra)
    cfg = MagicMock(platforms={Platform.TELEGRAM: pconfig})
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_coro(coro, _loop):
        import asyncio
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    job = {"id": "topic-job", "name": "Topic Brief", "deliver": "origin", "attach_to_session": True,
           "origin": {"platform": "telegram", "chat_id": "123456789", "user_id": "987654321",
                      "thread_id": "100", "chat_type": "dm"}}
    with patch("gateway.config.load_gateway_config", return_value=cfg), \
            patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
            patch("asyncio.run_coroutine_threadsafe", side_effect=run_coro), \
            patch("gateway.mirror.mirror_to_session", return_value=True):
        _deliver_result(job, "Brief content.", adapters={Platform.TELEGRAM: adapter}, loop=loop)
    sent_threads = [(c.kwargs.get("metadata") or {}).get("thread_id") for c in adapter.send.await_args_list]
    return adapter.create_handoff_thread.await_count, sent_threads


def test_opt_in_opens_a_fresh_thread_for_a_topic_born_job():
    created, sent = _deliver({"cron_fresh_thread_per_run": True})
    assert created == 1
    assert sent and set(sent) == {"42"}


@pytest.mark.parametrize("extra", [{}, {"cron_fresh_thread_per_run": False}, {"cron_fresh_thread_per_run": "yes"}])
def test_default_keeps_delivering_to_the_origin_topic(extra):
    created, sent = _deliver(extra)
    assert created == 0
    assert sent and set(sent) == {"100"}
