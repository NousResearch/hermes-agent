"""Regression tests for the Feishu adapter's owned SDK executor.

Blocking Feishu SDK calls used to run on asyncio's shared default executor.
When that executor was torn down (agent thread exit / loop cleanup), every
subsequent send failed permanently with "Executor shutdown has been called"
and the gateway became a zombie. The adapter now owns its own
ThreadPoolExecutor and recreates it on demand if it has been shut down.

Covers: #10849
"""
import concurrent.futures

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


def _bare_adapter() -> FeishuAdapter:
    """A FeishuAdapter with only the executor fields wired (no __init__)."""
    adapter = object.__new__(FeishuAdapter)
    import threading

    adapter._sdk_executor_lock = threading.Lock()
    adapter._sdk_executor = None
    adapter._sdk_executor_closing = False
    return adapter


def test_get_executor_creates_pool():
    adapter = _bare_adapter()
    executor = adapter._get_sdk_executor()
    assert isinstance(executor, concurrent.futures.ThreadPoolExecutor)
    # Same instance returned while alive.
    assert adapter._get_sdk_executor() is executor
    adapter._shutdown_sdk_executor()


def test_get_executor_recreates_after_shutdown():
    """A shut-down pool must be transparently replaced — the #10849 recovery."""
    adapter = _bare_adapter()
    first = adapter._get_sdk_executor()
    first.shutdown(wait=True)
    assert getattr(first, "_shutdown", False) is True

    second = adapter._get_sdk_executor()
    assert second is not first
    assert getattr(second, "_shutdown", False) is False
    adapter._shutdown_sdk_executor()


def test_shutdown_clears_reference():
    adapter = _bare_adapter()
    adapter._get_sdk_executor()
    adapter._shutdown_sdk_executor()
    assert adapter._sdk_executor is None
    # Idempotent.
    adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
async def test_run_blocking_executes_on_owned_pool():
    adapter = _bare_adapter()
    captured = {}

    def _work(value):
        import threading

        captured["thread"] = threading.current_thread().name
        return value * 2

    result = await adapter._run_blocking(_work, 21)
    assert result == 42
    # Ran on the adapter-owned pool, not the default executor.
    assert captured["thread"].startswith("hermes-feishu-sdk")
    adapter._shutdown_sdk_executor()


@pytest.mark.asyncio
async def test_run_blocking_survives_pool_shutdown():
    """After the pool is shut down, _run_blocking transparently recovers."""
    adapter = _bare_adapter()
    assert await adapter._run_blocking(lambda: "first") == "first"

    adapter._shutdown_sdk_executor()

    # _shutdown set the closing flag, so this would now refuse — re-arm first
    # the way a reconnect does, then the next call rebuilds the pool.
    adapter._sdk_executor_closing = False
    assert await adapter._run_blocking(lambda: "second") == "second"
    adapter._shutdown_sdk_executor()


def test_closing_flag_refuses_resurrection():
    """A real disconnect/shutdown must NOT be resurrected by the recreate path."""
    adapter = _bare_adapter()
    adapter._get_sdk_executor()  # build a live pool
    adapter._shutdown_sdk_executor()  # real teardown sets _closing

    assert adapter._sdk_executor_closing is True
    with pytest.raises(RuntimeError, match="shutting down"):
        adapter._get_sdk_executor()


@pytest.mark.asyncio
async def test_reconnect_rearms_executor():
    """connect() clears the closing flag so a reconnect can use the pool again."""
    import threading

    adapter = object.__new__(FeishuAdapter)
    adapter._sdk_executor_lock = threading.Lock()
    adapter._sdk_executor = None
    adapter._sdk_executor_closing = True  # as if a prior disconnect ran

    # connect() bails early (no creds) but must still re-arm the executor.
    adapter._app_id = ""
    adapter._app_secret = ""
    ok = await adapter.connect()
    assert ok is False  # bailed on missing creds
    assert adapter._sdk_executor_closing is False
    # And now the executor is usable again.
    assert await adapter._run_blocking(lambda: "rearmed") == "rearmed"
    adapter._shutdown_sdk_executor()

