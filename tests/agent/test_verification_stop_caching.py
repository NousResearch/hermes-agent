import os
import json
import pytest
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock

class _MockHandler(BaseHTTPRequestHandler):
    captured_requests = []
    response_queue = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        
        is_stream = req.get("stream") is True
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = {
                "id": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "DONE"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            }
        
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            if tcs:
                for ti, tc in enumerate(tcs):
                    chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": ti, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tcs else "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass

def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }

@pytest.fixture()
def agent_env(monkeypatch):
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_caching_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    # Import fresh to ensure changes are picked up
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
        provider="openai-compat", model="test-model",
        max_iterations=10, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home

def test_run_conversation_verification_loop_caching(agent_env, monkeypatch):
    agent, handler = agent_env

    # Mock verify_on_stop checks to trigger the nudge on the 1st turn attempt
    monkeypatch.setattr("agent.verification_stop.verify_on_stop_enabled", lambda *args, **kwargs: True)
    
    def mock_build_nudge(*args, **kwargs):
        attempts = kwargs.get("attempts", 0)
        if attempts == 0:
            return "[System: You edited code, please run tests]"
        return None
        
    monkeypatch.setattr("agent.verification_stop.build_verify_on_stop_nudge", mock_build_nudge)
    
    # 1. First call tries to finish (triggers verification stop nudge)
    handler.response_queue.append(_text_resp("I have completed the task."))
    
    # 2. Second call returns the final verified text response
    handler.response_queue.append(_text_resp("Everything is verified and clean."))
    
    # Run the conversation loop
    res = agent.run_conversation("Please complete the task", conversation_history=[], task_id="t")
    
    # Assert successful finish
    assert res["completed"] is True
    assert "Everything is verified and clean." in res["final_response"]
    
    # Filter for completion requests containing messages
    completion_requests = [req for req in handler.captured_requests if "messages" in req]
    assert len(completion_requests) == 2
    
    # Verify the payload of the 2nd completion request (the one sent after the synthetic verification nudge)
    req2 = completion_requests[1]
    messages2 = req2["messages"]
    
    # Check that messages2 has alternating roles
    # system -> user -> assistant (blocked) -> user (nudge)
    roles = [m["role"] for m in messages2]
    for idx in range(len(roles) - 1):
        r1, r2 = roles[idx], roles[idx+1]
        if r1 == "system":
            assert r2 == "user"
        elif r1 == "user":
            assert r2 == "assistant"
        elif r1 == "assistant":
            assert r2 == "user"

    # CRITICAL: Assert that the private caching flag is NOT present in any message sent to the API
    for msg in messages2:
        assert "_verification_stop_synthetic" not in msg
        assert "finish_reason" not in msg

    # CRITICAL: Assert that the final persistent session messages DO NOT contain the synthetic scaffolding
    final_messages = agent._session_messages
    synthetic_messages = [m for m in final_messages if m.get("_verification_stop_synthetic")]
    assert len(synthetic_messages) == 0
    
    # Confirm the blocked assistant message is gone from the persistent log
    log_file = agent.logs_dir / f"session_{agent.session_id}.json"
    if log_file.exists():
        log_data = json.loads(log_file.read_text(encoding="utf-8"))
        synthetic_in_log = [m for m in log_data["messages"] if m.get("_verification_stop_synthetic")]
        assert len(synthetic_in_log) == 0

def test_verification_stop_marker_popping():
    # Test that _verification_stop_synthetic is popped from API messages
    api_msg = {
        "role": "assistant",
        "content": "done",
        "_verification_stop_synthetic": True,
        "finish_reason": "stop",
    }
    
    api_msg_copy = api_msg.copy()
    if "finish_reason" in api_msg_copy:
        api_msg_copy.pop("finish_reason")
    api_msg_copy.pop("_verification_stop_synthetic", None)
    
    assert "_verification_stop_synthetic" not in api_msg_copy
    assert "finish_reason" not in api_msg_copy
    assert api_msg_copy["role"] == "assistant"
    assert api_msg_copy["content"] == "done"

def test_persist_session_filters_synthetic_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    
    # Mock SQLite db
    mock_db = MagicMock()
    
    # Import fresh run_agent
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    # Create agent instance with dummy credentials to bypass provider checks
    agent = AIAgent(
        session_id="test_sess_1",
        api_key="test-key",
        base_url="http://127.0.0.1:8000/v1",
        provider="custom",
        model="test-model",
        skip_memory=True,
    )
    agent._session_db = mock_db
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path / "logs"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "blocked done", "_verification_stop_synthetic": True},
        {"role": "user", "content": "[System: verify]", "_verification_stop_synthetic": True},
    ]
    
    agent._persist_session(messages)
    
    # Verify DB appends: mock_db.append_message should ONLY receive "hi" user message
    mock_db.append_message.assert_called_once()
    args, kwargs = mock_db.append_message.call_args
    assert kwargs.get("role") == "user"
    assert kwargs.get("content") == "hi"
    
    # Verify JSON log content
    log_file = agent.logs_dir / "session_test_sess_1.json"
    assert log_file.exists()
    
    log_data = json.loads(log_file.read_text(encoding="utf-8"))
    assert len(log_data["messages"]) == 1
    assert log_data["messages"][0]["role"] == "user"
    assert log_data["messages"][0]["content"] == "hi"
