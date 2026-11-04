"""HTTP admission regression: board bursts must not consume every worker."""
import asyncio
import importlib.util
import os
import sys
import threading
from pathlib import Path

import anyio
import httpx
import pytest
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from hermes_cli import kanban_db as kb

OK = 200
BOARD_PATH = "/api/plugins/kanban/board"
STATUS_PATH = "/probe"
BURST_SIZE = 8


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_board_burst_preserves_http_worker_capacity(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    plugin_path = Path(os.environ.get(
        "HERMES_TEST_KANBAN_PLUGIN",
        str(Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"),
    ))
    spec = importlib.util.spec_from_file_location("kanban_admission_test", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, plugin)
    spec.loader.exec_module(plugin)
    app = FastAPI()
    app.include_router(plugin.router, prefix="/api/plugins/kanban")

    @app.get(STATUS_PATH)
    async def probe():
        return await run_in_threadpool(lambda: {"ready": True})

    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    calls = 0
    original = kb.list_tasks

    def blocked_read(*args, **kwargs):
        nonlocal calls
        with count_lock:
            calls += 1
        entered.set()
        assert release.wait(10), "test did not release board read"
        return original(*args, **kwargs)

    monkeypatch.setattr(kb, "list_tasks", blocked_read)
    limiter = anyio.to_thread.current_default_thread_limiter()
    old_tokens = limiter.total_tokens
    limiter.total_tokens = 2
    requests = []
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            requests = [asyncio.create_task(client.get(BOARD_PATH)) for _ in range(BURST_SIZE)]
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            response = await asyncio.wait_for(client.get(STATUS_PATH), timeout=2)
            assert response.status_code == OK
            release.set()
            responses = await asyncio.gather(*requests)
            assert all(r.status_code == OK for r in responses)
            assert calls == 1
    finally:
        release.set()
        if requests:
            await asyncio.gather(*requests, return_exceptions=True)
        limiter.total_tokens = old_tokens