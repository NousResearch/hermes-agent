import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, patch

from cron.scheduler_delivery import _open_continuable_cron_thread


def test_results_from_same_job_open_distinct_content_named_threads():
    adapter = AsyncMock()
    adapter.create_handoff_thread.return_value = "123.456"

    def run_now(coro, loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    job = {"id": "j1", "name": "Claudia manager — stamps-community"}
    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=run_now):
        for headline in ("Search indexing restored", "Newsletter needs your approval"):
            assert _open_continuable_cron_thread(
                job, adapter, "C123", object(), result_text=f"## {headline}\n\nDetails",
            ) == "123.456"
            assert adapter.create_handoff_thread.call_args.args == ("C123", headline)


def test_empty_result_retains_honest_job_name():
    adapter = AsyncMock()

    def run_now(coro, loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future

    with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=run_now):
        _open_continuable_cron_thread({"name": "Daily brief"}, adapter, "C123", object())
    assert adapter.create_handoff_thread.call_args.args == ("C123", "Daily brief")
