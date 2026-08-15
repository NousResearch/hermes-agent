"""Local-wire E2E coverage for DeepSeek Responses native web search.

This uses the real OpenAI SDK against an in-process HTTP server. It verifies the
route, request body, streamed ``web_search_call`` normalization, and stateless
turn-N+1 replay without spending a DeepSeek API key.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest


class _DeepSeekResponsesHandler(BaseHTTPRequestHandler):
    captured_requests: list[dict] = []
    response_items_queue: list[list[dict]] = []

    def do_POST(self):  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)

        # AIAgent probes custom endpoints during context-window discovery (for
        # example POST /api/show). Those requests are not Responses API turns
        # and must not consume the queued SSE fixtures.
        if self.path != "/responses":
            body = b'{"error":"not found"}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        request_body = json.loads(raw_body.decode("utf-8"))
        type(self).captured_requests.append({"path": self.path, "body": request_body})
        items = type(self).response_items_queue.pop(0)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        sequence = 0
        for output_index, item in enumerate(items):
            self._send_event({
                "type": "response.output_item.done",
                "sequence_number": sequence,
                "output_index": output_index,
                "item": item,
            })
            sequence += 1

        self._send_event({
            "type": "response.completed",
            "sequence_number": sequence,
            "response": _completed_response(items, request_body["model"]),
        })

    def _send_event(self, event: dict) -> None:
        payload = json.dumps(event, separators=(",", ":"))
        self.wfile.write(f"event: {event['type']}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, *_args):
        pass


def _message_item(text: str, item_id: str) -> dict:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }


def _completed_response(items: list[dict], model: str) -> dict:
    return {
        "id": "resp_local",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model,
        "output": items,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt": None,
        "reasoning": {"effort": "medium", "summary": None},
        "service_tier": "default",
        "store": False,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [{"type": "web_search"}],
        "top_logprobs": 0,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        },
        "user": None,
        "metadata": {},
    }


@pytest.fixture
def deepseek_wire_server():
    _DeepSeekResponsesHandler.captured_requests = []
    _DeepSeekResponsesHandler.response_items_queue = [
        [
            {
                "id": "ws_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "latest DeepSeek release",
                    "sources": [{"type": "url", "url": "https://example.test/source"}],
                },
            },
            _message_item("DeepSeek released an update.", "msg_1"),
        ],
        [_message_item("The source is example.test.", "msg_2")],
    ]
    server = HTTPServer(("127.0.0.1", 0), _DeepSeekResponsesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, _DeepSeekResponsesHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _web_search_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_real_sdk_streams_and_replays_deepseek_web_search_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    deepseek_wire_server,
    model,
):
    from run_agent import AIAgent

    server, handler = deepseek_wire_server
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **_kwargs: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr(
        "agent.web_search_registry._read_config_key",
        lambda *path: "deepseek" if path == ("web", "search_backend") else None,
    )
    monkeypatch.setattr(
        "agent.web_search_registry.get_active_search_provider",
        lambda: SimpleNamespace(name="deepseek"),
    )

    agent = AIAgent(
        api_key="not-a-real-key",
        base_url=base_url,
        provider="deepseek",
        api_mode="chat_completions",  # stale input must be corrected by model
        model=model,
        max_iterations=1,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
    )
    assert agent.api_mode == "codex_responses"
    assert agent.base_url == base_url

    tools = [_web_search_tool()]
    first_messages = [{"role": "user", "content": "What is new?"}]
    first_kwargs = agent._get_transport().preflight_kwargs(
        agent._build_api_kwargs(first_messages, tools_for_api=tools)
    )
    first_response = agent._run_codex_stream(first_kwargs)
    normalized = agent._get_transport().normalize_response(first_response)

    assert normalized.content == "DeepSeek released an update."
    assert [item["type"] for item in normalized.codex_message_items] == [
        "web_search_call",
        "message",
    ]

    history = [
        *first_messages,
        {
            "role": "assistant",
            "content": normalized.content,
            "codex_message_items": normalized.codex_message_items,
        },
        {"role": "user", "content": "Which source?"},
    ]
    second_kwargs = agent._get_transport().preflight_kwargs(
        agent._build_api_kwargs(history, tools_for_api=tools)
    )
    second_response = agent._run_codex_stream(second_kwargs)
    second_normalized = agent._get_transport().normalize_response(second_response)
    assert second_normalized.content == "The source is example.test."

    assert [request["path"] for request in handler.captured_requests] == [
        "/responses",
        "/responses",
    ]
    first_body = handler.captured_requests[0]["body"]
    assert first_body["model"] == model
    assert first_body["tools"] == [{"type": "web_search"}]
    assert first_body["reasoning"] == {"effort": "high"}
    assert "prompt_cache_key" not in first_body
    assert "include" not in first_body
    assert "context_management" not in first_body

    replay_input = handler.captured_requests[1]["body"]["input"]
    replayed_search = next(
        item for item in replay_input if item.get("type") == "web_search_call"
    )
    assert replayed_search == {
        "id": "ws_1",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "latest DeepSeek release",
            "sources": [{"type": "url", "url": "https://example.test/source"}],
        },
    }
    assert [item.get("type", item.get("role")) for item in replay_input] == [
        "user",
        "web_search_call",
        "message",
        "user",
    ]
