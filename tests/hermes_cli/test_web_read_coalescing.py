"""Behavioral coverage for pre-threadpool HTTP read admission."""
import asyncio
import threading
from functools import wraps

import pytest


async def wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


def async_test(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


def decorator():
    from hermes_cli import web_read_coalescing
    return web_read_coalescing.coalesced_read


@async_test
async def test_identical_inflight_calls_share_one_read():
    release = threading.Event()
    calls = []

    @decorator()
    def read(key=1):
        calls.append(key)
        assert release.wait(3)
        return key

    tasks = [asyncio.create_task(read()) for _ in range(12)]
    try:
        await wait_until(lambda: calls)
        await asyncio.sleep(0.05)
        assert calls == [1]
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == [1] * 12
    assert await read() == 1
    assert calls == [1, 1]


@async_test
async def test_differing_keys_wait_before_threadpool_admission():
    import anyio.to_thread
    release = threading.Event()
    calls = []

    @decorator()
    def read(key):
        calls.append(key)
        assert release.wait(3)
        return key

    limiter = anyio.to_thread.current_default_thread_limiter()
    tasks = [asyncio.create_task(read(key)) for key in range(8)]
    try:
        await wait_until(lambda: calls)
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        assert limiter.borrowed_tokens == 1
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == list(range(8))


@async_test
async def test_caller_cancellation_keeps_worker_and_admission():
    release = threading.Event()
    calls = []

    @decorator()
    def read(key):
        calls.append(key)
        assert release.wait(3)
        return key

    first = asyncio.create_task(read(1))
    tasks = []
    try:
        await wait_until(lambda: calls)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        tasks = [asyncio.create_task(read(1)), asyncio.create_task(read(2))]
        await asyncio.sleep(0.05)
        assert calls == [1]
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == [1, 2]
    assert calls == [1, 2]


@async_test
async def test_canonical_arguments_coalesce():
    release = threading.Event()
    calls = []

    @decorator()
    def read(key=1, *, limit=2):
        calls.append(key)
        assert release.wait(3)
        return (key, limit)

    tasks = [asyncio.create_task(read()), asyncio.create_task(read(1)),
             asyncio.create_task(read(limit=2, key=1))]
    try:
        await wait_until(lambda: calls)
        await asyncio.sleep(0.05)
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == [(1, 2)] * 3
    assert calls == [1]


@async_test
async def test_home_isolation_preserves_worker_context(monkeypatch):
    from contextvars import ContextVar
    from pathlib import Path
    from hermes_cli import web_read_coalescing
    home = ContextVar("test_read_home", default=Path("/first"))
    monkeypatch.setattr(web_read_coalescing, "get_hermes_home", home.get, raising=False)
    release = threading.Event()
    calls = []

    @decorator()
    def read():
        calls.append(str(home.get()))
        assert release.wait(3)
        return str(home.get())

    first = asyncio.create_task(read())
    home.set(Path("/second"))
    second = asyncio.create_task(read())
    try:
        await wait_until(lambda: calls)
        await asyncio.sleep(0.05)
    finally:
        release.set()
        results = await asyncio.gather(first, second)
    assert results == ["/first", "/second"]
    assert calls == results


def test_fastapi_can_resolve_postponed_annotations():
    import inspect
    from pathlib import Path
    from fastapi import FastAPI
    namespace = {"Path": Path}
    exec("from __future__ import annotations\ndef read(key: Path) -> dict:\n    return {'key': str(key)}", namespace)
    read = decorator()(namespace["read"])
    assert inspect.iscoroutinefunction(read)
    assert inspect.signature(read).parameters["key"].annotation is Path
    app = FastAPI()
    app.get("/read")(read)
    assert app.openapi()["paths"]["/read"]["get"]["parameters"][0]["name"] == "key"


@async_test
async def test_failed_shared_read_is_retried():
    release = threading.Event()
    calls = []

    @decorator()
    def read():
        calls.append(1)
        assert release.wait(3)
        if len(calls) == 1:
            raise ValueError("read failed")
        return "recovered"

    tasks = [asyncio.create_task(read()) for _ in range(5)]
    await wait_until(lambda: calls)
    await asyncio.sleep(0.05)
    release.set()
    errors = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(error, ValueError) for error in errors)
    assert calls == [1]
    assert await read() == "recovered"
    assert calls == [1, 1]


@async_test
async def test_abandoned_failure_is_observed_and_retryable():
    import gc
    release = threading.Event()
    calls = []
    errors = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda loop, context: errors.append(context))

    @decorator()
    def read():
        calls.append(1)
        assert release.wait(3)
        raise ValueError("abandoned read")

    caller = asyncio.create_task(read())
    try:
        await wait_until(lambda: calls)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
    finally:
        release.set()
    await asyncio.sleep(0.05)
    gc.collect()
    assert errors == []
    with pytest.raises(ValueError, match="abandoned read"):
        await read()
    assert calls == [1, 1]


@async_test
async def test_profiles_burst_leaves_threadpool_status_responsive(monkeypatch):
    import anyio.to_thread
    import httpx
    from fastapi import FastAPI
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers import profiles
    from starlette.concurrency import run_in_threadpool
    release = threading.Event()
    calls = []

    def slow_profiles():
        calls.append(1)
        assert release.wait(3)
        return ["example"]

    monkeypatch.setattr(profiles_mod, "list_profiles", slow_profiles)
    monkeypatch.setattr(profiles, "_profile_to_dict", lambda p: {"name": p})
    monkeypatch.setattr(profiles, "run_in_threadpool", run_in_threadpool)
    app = FastAPI()
    app.include_router(profiles.router)

    @app.get("/api/status")
    def status():
        return {"ok": True}

    limiter = anyio.to_thread.current_default_thread_limiter()
    previous = limiter.total_tokens
    limiter.total_tokens = 2
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            tasks = [asyncio.create_task(client.get("/api/profiles")) for _ in range(12)]
            try:
                await wait_until(lambda: calls)
                await asyncio.sleep(0.05)
                response = await asyncio.wait_for(client.get("/api/status"), 0.5)
                assert response.status_code == 200
                assert response.json() == {"ok": True}
                assert calls == [1]
                assert limiter.borrowed_tokens == 1
            finally:
                release.set()
                responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in responses)
            assert all(r.json() == {"profiles": [{"name": "example"}]} for r in responses)
    finally:
        limiter.total_tokens = previous


@async_test
async def test_profiles_retains_threadpool_monkeypatch_seam(monkeypatch):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers import profiles
    calls = []

    async def runner(func):
        calls.append("runner")
        return func()

    monkeypatch.setattr(profiles, "run_in_threadpool", runner)
    monkeypatch.setattr(profiles_mod, "list_profiles", lambda: ["example"])
    monkeypatch.setattr(profiles, "_profile_to_dict", lambda p: {"name": p})
    assert await profiles.list_profiles_endpoint() == {"profiles": [{"name": "example"}]}
    assert calls == ["runner"]


@async_test
async def test_container_arguments_are_canonical_and_type_safe():
    release = threading.Event()
    calls = []

    @decorator()
    def read(options):
        calls.append(options)
        assert release.wait(3)
        return options

    values = [{"a": [1], "b": {2, 3}}, {"b": {3, 2}, "a": [1]}, True, 1]
    tasks = [asyncio.create_task(read(value)) for value in values]
    try:
        await asyncio.sleep(0.05)
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert results == values
    assert len(calls) == 3
    assert type(results[2]) is bool
    assert type(results[3]) is int


def test_decorated_function_is_isolated_between_live_event_loops():
    from concurrent.futures import ThreadPoolExecutor
    rendezvous = threading.Barrier(2)

    @decorator()
    def read():
        rendezvous.wait(timeout=3)
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(asyncio.run, read()) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]
    assert len(set(results)) == 2


@async_test
async def test_profiles_failure_preserves_fallback_and_retries(monkeypatch):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers import profiles
    from starlette.concurrency import run_in_threadpool
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("unavailable")
        return ["recovered"]

    monkeypatch.setattr(profiles, "run_in_threadpool", run_in_threadpool)
    monkeypatch.setattr(profiles_mod, "list_profiles", read)
    monkeypatch.setattr(profiles, "_profile_to_dict", lambda p: {"name": p})
    monkeypatch.setattr(profiles, "_fallback_profile_dicts", lambda mod: [{"name": "fallback"}])
    assert await profiles.list_profiles_endpoint() == {"profiles": [{"name": "fallback"}]}
    assert await profiles.list_profiles_endpoint() == {"profiles": [{"name": "recovered"}]}
    assert calls == [1, 1]


@async_test
async def test_profiles_fallback_is_coalesced_off_event_loop(monkeypatch):
    from hermes_cli import profiles as profiles_mod
    from hermes_cli.web_routers import profiles
    from starlette.concurrency import run_in_threadpool
    event_loop_thread = threading.get_ident()
    fallback_threads = []
    expected = {"profiles": [{"name": "fallback"}]}

    def unavailable():
        raise OSError("unavailable")

    def fallback(_):
        fallback_threads.append(threading.get_ident())
        return expected["profiles"]

    monkeypatch.setattr(profiles, "run_in_threadpool", run_in_threadpool)
    monkeypatch.setattr(profiles_mod, "list_profiles", unavailable)
    monkeypatch.setattr(profiles, "_fallback_profile_dicts", fallback)
    results = await asyncio.gather(*(profiles.list_profiles_endpoint() for _ in range(8)))
    assert all(result == expected for result in results)
    assert event_loop_thread not in fallback_threads
    assert len(fallback_threads) == 1