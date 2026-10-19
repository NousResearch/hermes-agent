"""Exercise native asyncio rejection inside a real running event loop."""
import asyncio
import gc
import warnings

import pytest

from cron.scheduler_delivery import _run_send_in_fresh_loop


def test_native_running_loop_rejection_closes_fresh_coroutine():
    async def run():
        event = asyncio.Event()
        with pytest.raises(RuntimeError, match='cannot be called from a running event loop'):
            _run_send_in_fresh_loop(event.wait)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        asyncio.run(run())
        gc.collect()
    assert not [item for item in caught if 'was never awaited' in str(item.message)]
