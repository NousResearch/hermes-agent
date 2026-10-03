"""The boot lifecycle record (sentinel fsync, unclean-exit state.db integrity check, exit-diag
append) must run off the event loop: it is reached from the async gateway start path."""

import ast
import asyncio
import threading
import time
from pathlib import Path

import pytest

from gateway import lifecycle_ledger

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_record_startup_async_runs_the_whole_record_on_a_worker_thread(monkeypatch):
    loop_thread = threading.get_ident()
    seen = {}

    def slow_record(home=None):
        seen["thread"] = threading.get_ident()
        seen["home"] = home
        time.sleep(0.3)  # stands in for fsync / PRAGMA quick_check
        return {"prior_pid": 1}

    monkeypatch.setattr(lifecycle_ledger, "record_startup", slow_record)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    t = asyncio.create_task(ticker())
    try:
        result = await lifecycle_ledger.record_startup_async(home=Path("/nonexistent"))
    finally:
        t.cancel()

    assert result == {"prior_pid": 1}
    assert seen["home"] == Path("/nonexistent")
    assert seen["thread"] != loop_thread
    assert ticks >= 10  # the loop kept running while the record blocked


@pytest.mark.asyncio
async def test_record_startup_async_never_raises(monkeypatch):
    def boom(home=None):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(lifecycle_ledger, "record_startup", boom)
    assert await lifecycle_ledger.record_startup_async() is None


def test_gateway_start_path_uses_the_async_record():
    """No gateway/run*.py module may call the blocking ``record_startup`` directly."""
    offenders = []
    for path in sorted((REPO / "gateway").glob("run*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if name == "record_startup":
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"blocking record_startup() called from the gateway start path: {offenders}"
