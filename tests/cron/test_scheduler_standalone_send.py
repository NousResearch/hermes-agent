"""Standalone-send fallback in cron delivery must never orphan its send coroutine.

The thread-pool fallback used to build ``_send()`` as a ``submit()`` argument: when ``submit()``
itself raised (executor shutdown) or the worker's ``asyncio.run()`` failed, the coroutine was
left unawaited ("coroutine '_send' was never awaited").
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler import _deliver_result


def _email_cfg():
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.EMAIL: pconfig}
    return mock_cfg


def _coroutines_in(calls):
    return [arg for c in calls for arg in c.args if inspect.iscoroutine(arg)]


def test_rejected_submit_does_not_construct_fallback_coroutine():
    """An executor that rejects submit() must not be handed a pre-built send coroutine."""
    job = {"id": "rejected-submit-job", "deliver": "email:a@example.com"}

    with patch("gateway.config.load_gateway_config", return_value=_email_cfg()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run", side_effect=RuntimeError("no running loop")) as mock_run, \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
         patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls:
        mock_pool = MagicMock()
        mock_pool_cls.return_value = mock_pool
        mock_pool.submit.side_effect = RuntimeError("executor rejected submission")

        result = _deliver_result(job, "Report content")

    assert result is not None
    assert "executor rejected submission" in result
    assert _coroutines_in(mock_pool.submit.call_args_list) == []
    for coro in _coroutines_in(mock_run.call_args_list):
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


def test_worker_run_failure_closes_fallback_coroutine():
    """A worker-side asyncio.run() failure must close the coroutine it was given."""
    job = {"id": "worker-run-failure-job", "deliver": "email:a@example.com"}

    with patch("gateway.config.load_gateway_config", return_value=_email_cfg()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("asyncio.run", side_effect=RuntimeError("loop rejected")) as mock_run, \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})):
        result = _deliver_result(job, "Report content")

    assert result is not None
    assert "loop rejected" in result
    coros = _coroutines_in(mock_run.call_args_list)
    assert len(coros) == 2  # the direct attempt and the worker fallback
    for coro in coros:
        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
