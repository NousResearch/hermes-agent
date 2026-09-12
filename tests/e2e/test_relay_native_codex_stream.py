"""Native OpenAI Responses streaming through Relay managed execution."""

from __future__ import annotations

import json
import threading
import time

import pytest


def _sse_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def test_codex_terminal_event_closes_managed_provider_stream(tmp_path, monkeypatch):
    """Relay must not wait for provider EOF after a complete Responses event."""
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("nemo_relay")
    openai = pytest.importorskip("openai")

    from agent import relay_runtime
    from run_agent import AIAgent

    class HeldOpenSseStream(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body
            self.closed = threading.Event()
            self.read_after_terminal = threading.Event()

        def __iter__(self):
            yield self.body
            self.read_after_terminal.set()
            self.closed.wait(5)

        def close(self) -> None:
            self.closed.set()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    message = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "msg_held_open",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "All done.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
    }
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp_held_open",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5-codex",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
        },
    }
    held_stream = HeldOpenSseStream(_sse_event(message) + _sse_event(completed))
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=held_stream,
            request=request,
        )

    client = openai.OpenAI(
        api_key="test-key",
        base_url="https://example.com/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    relay_runtime._reset_for_tests()
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        provider="test-provider",
        model="gpt-5-codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.session_id = "codex-relay-session"
    agent._interrupt_requested = False
    agent.client = client
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id=agent.session_id,
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="codex-relay-turn",
        task_id="codex-relay-task",
    )
    lease.host.retain_managed_execution("test.codex_relay")

    try:
        started = time.monotonic()
        result = agent._run_codex_stream(
            {
                "model": "gpt-5-codex",
                "instructions": "You are Hermes.",
                "input": [{"role": "user", "content": "Ping"}],
                "tools": None,
                "store": False,
            }
        )
        elapsed = time.monotonic() - started
    finally:
        lease.host.release_managed_execution("test.codex_relay")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()
        client.close()

    assert elapsed < 2.0
    assert len(requests) == 1
    assert result.status == "completed"
    assert result.id == "resp_held_open"
    assert result.output[0].content[0].text == "All done."
    assert result.usage.total_tokens == 16
    assert held_stream.closed.wait(1)
    assert not held_stream.read_after_terminal.is_set()
