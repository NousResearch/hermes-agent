"""GET /api/profiles coalesces a poll burst before AnyIO threadpool admission (#105179)."""
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
