"""Blocking receipt waiters must leave the gateway's storage executor available."""
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pytest

pytest_plugins = ["tests.gateway.test_plugin_submission"]
pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")


@pytest.mark.asyncio
async def test_concurrent_receipts_use_independent_waiters(submission, monkeypatch):
    from gateway.plugin_submission import service

    ctx, _, authority, _, identity = submission
    loop = asyncio.get_running_loop()
    previous = loop._default_executor
    storage = ThreadPoolExecutor(max_workers=1)
    waiters = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(storage)
    monkeypatch.setattr(service, "call", partial(service.call, timeout=2))

    async def invoke(method, *args, **kwargs):
        return await loop.run_in_executor(
            waiters, partial(contextvars.copy_context().run, method, *args, **kwargs)
        )

    try:
        requests = [{**identity, "request_id": f"concurrent-{index}"} for index in range(4)]
        admitted = await asyncio.wait_for(asyncio.gather(*(
            invoke(ctx.submit_message, "work", **request) for request in requests
        )), timeout=10)
        assert all(row["ok"] and row["status"] == "queued" for row in admitted), admitted
        observed = await asyncio.wait_for(asyncio.gather(*(
            invoke(ctx.submission_status, **request) for request in requests
        )), timeout=10)
        assert [row["admission_id"] for row in observed] == [row["admission_id"] for row in admitted]
        assert len(set(row["admission_id"] for row in admitted)) == 4
        rows = authority.persistence.admissions.list(session_id=identity["expected_session_id"])
        assert len([row for row in rows if row.payload.get("text") == "work"]) == 4
    finally:
        loop.set_default_executor(previous)
        waiters.shutdown(wait=False, cancel_futures=True)
        storage.shutdown(wait=False, cancel_futures=True)
