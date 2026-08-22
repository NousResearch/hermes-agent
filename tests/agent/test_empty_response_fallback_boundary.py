"""#77305: outer empty-response fallback must roll back both iteration gates.

The empty-response ladder sits directly in the OUTER iteration loop, not the
inner retry loop. Each empty response consumed one outer logical iteration
(``api_call_count += 1`` and one ``iteration_budget`` slot), so when the
retry budget is exhausted and the fallback is activated, the fallback still
has not been called. If the site does NOT roll back both gates before its
``continue``, then at the ``max_iterations`` boundary the while condition
(``api_call_count < max_iterations and iteration_budget.remaining > 0``)
goes false and the loop exits — the just-activated fallback is never called
(#77305, still live on this path on main).

This test drives ``AIAgent.run_conversation`` against an in-process mock
provider with ``max_iterations = 4``: four empty responses exhaust the
empty-response retry budget, the fourth activates the fallback, and the
fallback must be issued as a FIFTH physical request and complete normally
with one logical iteration counted (4 logical charges / 5 physical attempts).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
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

    def log_message(self, *a, **kw):  # silence default stderr
        pass


def _text_resp(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env(monkeypatch):
    """In-process mock provider + isolated HERMES_HOME + a real AIAgent with
    ``max_iterations = 4`` and a single configured fallback.

    ``empty_response_guard.deterministic_empty`` is stubbed to False so the
    full 4-empty retry ladder runs (two same-signature zero-output responses
    would otherwise make the streak deterministic and skip the remaining
    retries, collapsing the boundary we are testing).
    """
    from agent import empty_response_guard as guard

    monkeypatch.setattr(guard, "deterministic_empty", lambda *a, **k: False)

    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_77305_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            del sys.modules[mod]
    from run_agent import AIAgent

    base = f"http://127.0.0.1:{port}/v1"
    agent = AIAgent(
        api_key="test-key", base_url=base,
        provider="openai-compat", model="test-model",
        max_iterations=4, enabled_toolsets=[],
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
        fallback_model=[{"provider": "openai-compat", "model": "fallback-model",
                         "api_key": "test-key", "base_url": base}],
    )
    agent.valid_tool_names = {"terminal", "read_file", "write_file", "execute_code", "session_search"}

    # Both primary and fallback point at the same mock provider. Simulate a
    # fallback switch by rotating the model identity and advancing the chain
    # index — the boundary accounting under test (roll back both gates before
    # the empty-response fallback continue) is independent of provider
    # resolution mechanics.
    def _fake_fallback(reason=None):
        agent.model = "fallback-model"
        agent._fallback_index = 1
        return True

    monkeypatch.setattr(agent, "_try_activate_fallback", _fake_fallback)

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def test_empty_response_fallback_rolls_back_both_gates_at_boundary(agent_env):
    """Four empty primary responses exhaust the retry budget; the fallback is
    activated on the fourth and must run as a FIFTH physical request and
    complete normally, with 4 logical charges (not a premature exit)."""
    agent, handler = agent_env

    # 4 empty responses (consume the outer iterations / retry budget) then a
    # normal text response served to the fallback.
    handler.response_queue = [_text_resp("") for _ in range(4)] + [_text_resp("fallback ok")]

    result = agent.run_conversation("hi", conversation_history=[], task_id="t")

    assert result["completed"] is True, result.get("final_response")
    assert result["final_response"] == "fallback ok", result.get("final_response")
    assert result["api_calls"] == 4, result["api_calls"]

    # 4 empty primary requests + 1 fallback request. Count only actual
    # conversation submissions (payloads carrying `messages`) — a model
    # context-length probe against the mock may also register as a request.
    conversation = [r for r in handler.captured_requests if "messages" in r]
    assert len(conversation) == 5, len(conversation)
    # The last (fallback) request must carry the fallback model identity.
    assert handler.captured_requests[-1]["model"] == "fallback-model"


def test_no_fallback_terminal_exhaustion_not_refunded(agent_env, monkeypatch):
    """Without a fallback, exhausted empty responses terminate at \"(empty)\"
    and are NOT refunded — the rollback only fires inside the successful
    ``_try_activate_fallback`` branch, so the terminal path keeps all four
    logical charges and never re-enters the loop (no fifth request)."""
    agent, handler = agent_env
    agent._fallback_chain = []
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda reason=None: False)

    handler.response_queue = [_text_resp("") for _ in range(4)]

    result = agent.run_conversation("hi", conversation_history=[], task_id="t")

    # Terminal exhaustion: all 4 logical iterations consumed, no refund, and
    # no re-entry (no fifth request). The final assistant message carries the
    # "(empty)" terminal sentinel.
    assert result["api_calls"] == 4, result["api_calls"]
    assert any(
        m.get("role") == "assistant" and m.get("content") == "(empty)"
        for m in result.get("messages", [])
    )
    conversation = [r for r in handler.captured_requests if "messages" in r]
    assert len(conversation) == 4, len(conversation)
