"""Admission remains true even if the downstream turn later fails or is cancelled."""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_plugin_message_injection import _entry, _runner


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "exception"])
async def test_receipt_tracks_adoption_not_eventual_task_completion(accepted, failure):
    ready = asyncio.Event()
    release = asyncio.Event()

    async def handle(event):
        event._gateway_accepted = accepted
        ready.set()
        await release.wait()
        raise RuntimeError("downstream turn failed")

    entry = _entry()
    runner = _runner(entry, SimpleNamespace(handle_message=handle))
    runner._gateway_loop = asyncio.get_running_loop()
    receipts = []
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="checkpoint", plugin_id="test-plugin",
        expected_session_id=entry.session_id, on_delivery=receipts.append)
    await asyncio.wait_for(ready.wait(), 2)
    tasks = list(runner._background_tasks)
    if failure == "cancel":
        for task in tasks:
            task.cancel()
    else:
        release.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    assert receipts == [accepted]
