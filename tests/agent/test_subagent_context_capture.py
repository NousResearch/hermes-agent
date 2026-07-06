from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hermes_state import SessionDB
from agent.subagent_context_artifacts import (
    create_subagent_context_artifact_pointer,
    get_subagent_context_artifact,
)


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list[dict] = []
    response_queue: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).captured_requests.append(request)
        response = type(self).response_queue.pop(0) if type(self).response_queue else _text_resp("captured ok")
        if request.get("stream") is True:
            content = response["choices"][0]["message"].get("content") or ""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs):
        pass


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


@pytest.fixture()
def capture_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="parent-capture", source="cli")

    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
        provider="openai-compat",
        model="test-model",
        max_iterations=3,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        platform="subagent",
        session_db=db,
        parent_session_id="parent-capture",
    )
    child_session_id = str(getattr(agent, "session_id"))
    create_subagent_context_artifact_pointer(
        child_session_id=child_session_id,
        parent_session_id="parent-capture",
        subagent_id="sa-capture",
        role="leaf",
        profile="coder",
        model="test-model",
        provider="openai-compat",
        api_mode="chat_completions",
        base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1",
        toolsets=[],
        session_db=db,
    )
    setattr(agent, "_subagent_context_ref", {
        "child_session_id": child_session_id,
        "parent_session_id": "parent-capture",
        "subagent_id": "sa-capture",
        "role": "leaf",
        "profile": "coder",
        "model": "test-model",
        "provider": "openai-compat",
        "api_mode": "chat_completions",
        "base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1",
        "toolsets": [],
    })
    try:
        yield agent, db, _MockHandler
    finally:
        srv.shutdown()
        thread.join(timeout=2)
        db.close()


def test_capture_records_ordered_messages_and_provider_request(capture_agent):
    agent, db, _handler = capture_agent

    result = agent.run_conversation(
        user_message="hello capture password=hunter2",
        conversation_history=[],
        task_id="capture-test",
    )

    assert result["final_response"] == "captured ok"
    child_session_id = str(getattr(agent, "session_id"))
    artifact = get_subagent_context_artifact(child_session_id, session_db=db)
    assert artifact["ok"] is True
    payload = artifact["artifact"]
    assert payload["raw_unredacted_by_viewer"] is True
    assert payload["child_session_id"] == child_session_id
    assert payload["subagent_id"] == "sa-capture"
    assert payload["capture_sequence"] == 1
    roles = [msg["role"] for msg in payload["canonical_messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert "password=hunter2" in payload["canonical_messages"][-1]["content"]
    assert payload["provider_request"]["messages"][-1]["content"] == "hello capture password=hunter2"
    assert "_db_persisted" not in payload["provider_request"]["messages"][-1]
    assert "messages" in payload["provider_request_keys"]


def test_capture_uses_rewritten_terminal_payload_after_execution_middleware(capture_agent, monkeypatch):
    agent, db, handler = capture_agent
    handler.response_queue.append(_text_resp("rewritten ok"))

    def fake_middleware(request, next_call, **context):
        rewritten = dict(request)
        rewritten["messages"] = list(request["messages"]) + [
            {"role": "user", "content": "middleware-added"}
        ]
        return next_call(rewritten)

    monkeypatch.setattr("hermes_cli.middleware.run_llm_execution_middleware", fake_middleware)

    result = agent.run_conversation(
        user_message="trigger middleware",
        conversation_history=[],
        task_id="capture-rewrite-test",
    )

    assert result["final_response"] == "rewritten ok"
    assert handler.captured_requests
    payload = get_subagent_context_artifact(str(getattr(agent, "session_id")), session_db=db)["artifact"]
    assert payload["capture_sequence"] == 1
    assert payload["provider_request"]["messages"][-1]["content"] == "middleware-added"
    assert payload["canonical_messages"][-1]["content"] == "trigger middleware"


def test_capture_failure_does_not_break_model_call(capture_agent, monkeypatch):
    agent, _db, _handler = capture_agent

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("agent.subagent_context_artifacts.update_subagent_context_artifact_capture", boom)

    result = agent.run_conversation(
        user_message="survive capture failure",
        conversation_history=[],
        task_id="capture-fail-open-test",
    )

    assert result["final_response"] == "captured ok"
