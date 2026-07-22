"""End-to-end JSON-RPC over the real Starlette app, in-process, no LLM.

Builds the actual A2A app (card + DefaultRequestHandler + BoundedTaskStore)
around an echo agent and drives a ``message/send`` request through it via the
Starlette test client — exercising the full transport path (routing, JSON-RPC
decode, executor, event consumption, task assembly).
"""

from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from starlette.testclient import TestClient

from plugins.platforms.a2a.card import build_agent_card
from plugins.platforms.a2a.executor import HermesAgentExecutor
from plugins.platforms.a2a.sessions import ContextSessionStore
from plugins.platforms.a2a.task_store import BoundedTaskStore


def _build_echo_client(fakes, agent_factory=None) -> TestClient:
    store = ContextSessionStore(agent_factory=agent_factory or fakes.FakeAgent)
    executor = HermesAgentExecutor(store)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=BoundedTaskStore(max_tasks=32, max_history_messages=16),
    )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await executor.aclose()

    app = A2AStarletteApplication(
        agent_card=build_agent_card("http://test/"),
        http_handler=handler,
    ).build(lifespan=lifespan)
    return TestClient(app)


def test_message_send_returns_completed_task_with_echo(fakes):
    request = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": "m1",
                "parts": [{"kind": "text", "text": "ping"}],
            }
        },
    }
    with _build_echo_client(fakes) as client:
        resp = client.post("/", json=request)
    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body, body
    result = body["result"]

    # message/send returns the final Task once it reaches a terminal state.
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"

    # The echo response is delivered as an artifact.
    text = result["artifacts"][0]["parts"][0]["text"]
    assert text == "echo: ping"
    # Sanity: the whole payload mentions the echo.
    assert "echo: ping" in json.dumps(body)


def test_nonblocking_cancel_remains_canceled_in_task_store(fakes):
    class BlockingAgent:
        def __init__(self):
            self.stream_delta_callback = None
            self.reasoning_callback = None
            self.tool_progress_callback = None
            self.step_callback = None
            self.thinking_callback = None
            self._interrupted = threading.Event()

        def run_conversation(self, **_kwargs):
            self._interrupted.wait(5)
            return {"final_response": None, "interrupted": True, "messages": []}

        def interrupt(self, _message=None):
            self._interrupted.set()

        def clear_interrupt(self):
            self._interrupted.clear()

    send_request = {
        "jsonrpc": "2.0",
        "id": "send",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": "m-cancel",
                "parts": [{"kind": "text", "text": "wait"}],
            },
            "configuration": {"blocking": False},
        },
    }

    with _build_echo_client(fakes, BlockingAgent) as client:
        send = client.post("/", json=send_request).json()
        assert "error" not in send, send
        task_id = send["result"]["id"]

        canceled = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "cancel",
                "method": "tasks/cancel",
                "params": {"id": task_id},
            },
        ).json()
        assert "error" not in canceled, canceled
        assert canceled["result"]["status"]["state"] == "canceled"

        fetched = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "get",
                "method": "tasks/get",
                "params": {"id": task_id},
            },
        ).json()
        assert "error" not in fetched, fetched
        assert fetched["result"]["status"]["state"] == "canceled"


def test_message_stream_emits_sse_artifact_and_terminal_status(fakes):
    request = {
        "jsonrpc": "2.0",
        "id": "stream",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "kind": "message",
                "messageId": "m-stream",
                "parts": [{"kind": "text", "text": "stream ping"}],
            }
        },
    }

    with _build_echo_client(fakes) as client:
        with client.stream("POST", "/", json=request) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payloads = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    results = [payload["result"] for payload in payloads if "result" in payload]
    assert any(result.get("kind") == "artifact-update" for result in results)
    assert any(
        result.get("kind") == "status-update"
        and result["status"]["state"] == "completed"
        for result in results
    )
