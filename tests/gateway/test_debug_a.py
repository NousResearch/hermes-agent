"""Debug: why does A never start in the pytest harness? (works in the standalone repro)"""
import asyncio, json, threading
import sys
sys.path.insert(0, "/home/clawd/hermes-agent-pr/tests/gateway")
sys.path.insert(0, "/home/clawd/hermes-agent-pr/tests")
sys.path.insert(0, "/home/clawd/hermes-agent-pr")

import pytest
from unittest.mock import patch
from test_session_api import _session_post, _create_session_app  # noqa


@pytest.mark.asyncio
async def test_debug_a_start(adapter, session_db):
    session_id = session_db.create_session("dbg", "api_server")
    run_started = threading.Event()
    allow_finish = threading.Event()

    async def fake_run(**kwargs):
        del kwargs
        run_started.set()
        await asyncio.to_thread(allow_finish.wait, 5)
        return {"final_response": "x", "session_id": session_id}, {"total_tokens": 1}

    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        a_task = asyncio.create_task(_session_post(adapter, session_id, {"message": "A"}))
        statuses_seen = []
        for _ in range(80):
            statuses_seen = [s.get("status") for s in adapter._run_statuses.values()]
            if run_started.is_set():
                break
            await asyncio.sleep(0.05)
        print("\nSTATUSES:", statuses_seen)
        print("run_started:", run_started.is_set())
        allow_finish.set()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await a_task
        assert run_started.is_set(), f"A never started; statuses={statuses_seen}"
