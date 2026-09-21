"""Deadline regressions for Hindsight capacity retries."""

import asyncio
import threading

import pytest

from plugins.memory.hindsight import HindsightMemoryProvider


def test_capacity_request_is_cancelled_when_provider_timeout_expires(monkeypatch):
    provider = HindsightMemoryProvider()
    provider._mode = "cloud"
    monkeypatch.setattr(provider, "_timeout", 0.05)
    monkeypatch.setattr(provider, "_client", object())
    cancelled = threading.Event()
    active = {}

    async def operation(_client):
        active["task"] = asyncio.current_task()
        active["loop"] = asyncio.get_running_loop()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        with pytest.raises(TimeoutError):
            provider._run_hindsight_operation(operation, capacity_retry=True)
        assert cancelled.wait(2), "Request remained active after the caller timed out"
    finally:
        if active:
            active["loop"].call_soon_threadsafe(active["task"].cancel)
            cancelled.wait(2)


@pytest.mark.asyncio
async def test_retry_does_not_start_after_wait_budget_expires(monkeypatch):
    from hindsight_client_api.exceptions import ApiException
    from plugins.memory.hindsight import retry

    clock = [0.0]
    calls = []
    error = ApiException(status=503)
    error.headers = {"Retry-After": "1"}

    async def operation():
        calls.append(clock[0])
        if len(calls) == 1:
            raise error
        return "late success"

    async def delayed_wakeup(_delay):
        clock[0] = 3.0

    monkeypatch.setattr(retry.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(retry.asyncio, "sleep", delayed_wakeup)
    with pytest.raises(ApiException) as caught:
        await retry.retry_capacity(operation, total_budget=2)
    assert caught.value is error
    assert calls == [0.0]


def test_queued_capacity_request_never_starts_after_caller_timeout(monkeypatch):
    import plugins.memory.hindsight as hindsight

    provider = HindsightMemoryProvider()
    provider._mode = "cloud"
    monkeypatch.setattr(provider, "_timeout", 0.05)
    monkeypatch.setattr(provider, "_client", object())
    blocked = threading.Event()
    release = threading.Event()
    calls = []

    def block_loop():
        blocked.set()
        release.wait(5)

    async def operation(_client):
        calls.append("started")
        return "late success"

    async def drain():
        for _ in range(5):
            await asyncio.sleep(0)

    loop = hindsight._get_loop()
    loop.call_soon_threadsafe(block_loop)
    assert blocked.wait(2)
    try:
        with pytest.raises(TimeoutError):
            provider._run_hindsight_operation(operation, capacity_retry=True)
    finally:
        release.set()
        asyncio.run_coroutine_threadsafe(drain(), loop).result(timeout=2)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["+1", "1.5", "1e-309", "١", "-1", "NaN", "Infinity"])
async def test_malformed_retry_after_uses_fallback(monkeypatch, header):
    from hindsight_client_api.exceptions import ApiException
    from plugins.memory.hindsight import retry

    failure = ApiException(status=503)
    failure.headers = {"Retry-After": header}
    calls = []
    delays = []

    async def operation():
        calls.append(1)
        if len(calls) == 1:
            raise failure
        return "recovered"

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(retry.random, "uniform", lambda _low, _high: 0.25)
    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    assert await retry.retry_capacity(operation) == "recovered"
    assert delays == [0.25]


def test_embedded_sdk_retries_are_not_wrapped_in_provider_retries(monkeypatch):
    from hindsight_client_api.exceptions import ApiException
    from plugins.memory.hindsight import retry

    provider = HindsightMemoryProvider()
    provider._mode = "local_embedded"
    monkeypatch.setattr(provider, "_client", object())
    failure = ApiException(status=503)
    attempts = []

    async def embedded_operation(_client):
        # The embedded wrapper owns its SDK and has no public retry setting.
        attempts.extend(["sdk attempt"] * 3)
        raise failure

    monkeypatch.setattr(retry.random, "uniform", lambda _low, _high: 0)
    with pytest.raises(ApiException):
        provider._run_hindsight_operation(embedded_operation, capacity_retry=True)
    assert len(attempts) == 3
