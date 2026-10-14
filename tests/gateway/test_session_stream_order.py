"""Session SSE callbacks must reach the wire before completion or failure."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_session_api import _create_session_app

pytest_plugins = ("tests.gateway.test_session_api",)


class CompletedWorkerPool(ThreadPoolExecutor):
    """Exercise a worker completing before run_in_executor wraps its future."""

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        # No delay/racing assertion: this is a legal fast-worker interleaving.
        # Retain exceptions on the Future, as the real executor does.
        future.exception(timeout=10)
        return future


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True], ids=["completed", "failed"])
@pytest.mark.parametrize("fast_worker", [False, True], ids=["normal", "already-finished"])
async def test_session_stream_orders_worker_events_before_terminal(
    adapter, session_db, monkeypatch, fast_worker, failed,
):
    session_id = session_db.create_session("ordered-stream", "api_server")

    def create_agent(**callbacks):
        class Agent:
            provider, model = "openai-codex", "gpt-5"
            session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

            def run_conversation(self, **kwargs):
                callbacks["interim_assistant_callback"]("Checking first.", already_streamed=False)
                callbacks["tool_progress_callback"]("tool.started", "terminal", "pwd", {})
                callbacks["tool_progress_callback"]("tool.completed", "terminal", "ok", {})
                callbacks["stream_delta_callback"]("Done.")
                if failed:
                    raise RuntimeError("scripted failure")
                return {"final_response": "Done.", "messages": [], "api_calls": 1}

        return Agent()

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    loop = asyncio.get_running_loop()
    run_in_executor = loop.run_in_executor
    with CompletedWorkerPool(max_workers=1) as pool:
        if fast_worker:
            monkeypatch.setattr(
                loop, "run_in_executor",
                lambda executor, fn, *args: run_in_executor(executor or pool, fn, *args),
            )
        async with TestClient(TestServer(_create_session_app(adapter))) as client:
            response = await client.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "go"})
            assert response.status == 200
            body = await response.text()

    events = []
    for block in body.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "data" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    terminal = ["error"] if failed else ["assistant.completed", "run.completed"]
    assert [name for name, _ in events] == [
        "run.started", "message.started", "assistant.commentary", "tool.started",
        "tool.completed", "assistant.delta", *terminal, "done",
    ], body
    assert [payload["seq"] for _, payload in events] == list(range(1, len(events) + 1))
    assert {payload["session_id"] for _, payload in events} == {session_id}
    assert len({payload["run_id"] for _, payload in events}) == 1
    assert events[2][1]["text"] == "Checking first."
    assert events[5][1]["delta"] == "Done."
    if not failed:
        assert events[6][1]["content"] == "Done."
