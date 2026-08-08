"""Tests for the pre_llm_call block directive (Cowork-inspired inference veto).

A ``pre_llm_call`` hook may return ``{"action": "block", "message": "..."}``
to veto the turn BEFORE any provider request is made — inspired by Claude
Cowork / Claude Enterprise "inference hooks" (Aug 2026), where a policy layer
inspects every prompt pre-inference and returns an allow/deny verdict.

Covers: prologue directive extraction (block wins, message required, first
block wins, context still collected), the wire invariant (zero provider
requests on a blocked turn), message-role alternation of the persisted
transcript, and recovery on the following unblocked turn. Also covers the
shell-hook stdout translation for the new event shape.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_state import SessionDB

BLOCK_MSG = "⛔ Blocked by policy: prompt contains a customer SSN."


# ---------------------------------------------------------------------------
# Mock provider (counts every request that would have reached the model)
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        if req.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {"content": "MODEL-ANSWER"}, "finish_reason": None}]},
                {"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        resp = {
            "id": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "MODEL-ANSWER"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


@pytest.fixture()
def block_env():
    """Mock provider + isolated HERMES_HOME + a shared SessionDB.

    Yields ``(make_agent, handler, db, sid, hook_state)`` where
    ``hook_state["results"]`` is what the patched ``pre_llm_call`` hook
    returns on the next turn.
    """
    _MockHandler.captured_requests = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_prellm_block_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    from run_agent import AIAgent

    db = SessionDB(db_path=Path(test_home) / "state.db")
    sid = "sess-block"
    hook_state = {"results": []}

    def make_agent():
        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
            provider="openai-compat", model="test-model",
            max_iterations=10, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="cli",
            session_db=db, session_id=sid,
        )
        return agent

    try:
        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda hook, **kw: (
                list(hook_state["results"]) if hook == "pre_llm_call" else []
            ),
        ):
            yield make_agent, _MockHandler, db, sid, hook_state
    finally:
        srv.shutdown()
        db.close()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _chat_requests(handler) -> list:
    # The model context-length probe also hits the mock; keep only
    # chat-completions payloads.
    return [r for r in handler.captured_requests if "messages" in r]


# ---------------------------------------------------------------------------
# E2E: blocked turn never reaches the provider
# ---------------------------------------------------------------------------

class TestPreLlmBlock:
    def test_block_prevents_provider_request(self, block_env):
        make_agent, handler, db, sid, hook_state = block_env
        hook_state["results"] = [{"action": "block", "message": BLOCK_MSG}]
        agent = make_agent()

        result = agent.run_conversation("here is an SSN 123-45-6789", conversation_history=[], task_id="t")

        assert result["final_response"] == BLOCK_MSG
        assert result["turn_exit_reason"] == "blocked_by_plugin_pre_llm_call"
        assert result["api_calls"] == 0
        assert result["failed"] is False
        assert _chat_requests(handler) == []  # nothing reached the provider

    def test_blocked_turn_keeps_alternation_and_next_turn_recovers(self, block_env):
        make_agent, handler, db, sid, hook_state = block_env
        hook_state["results"] = [{"action": "block", "message": BLOCK_MSG}]
        agent = make_agent()
        result = agent.run_conversation("secret stuff", conversation_history=[], task_id="t")
        assert result["final_response"] == BLOCK_MSG

        # Persisted transcript stays a valid user→assistant alternation.
        rows = [r for r in db.get_messages(sid) if r["role"] in ("user", "assistant")]
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert rows[-1]["content"] == BLOCK_MSG

        # Next turn (hook allows): normal provider round-trip on a fresh agent
        # that reloads the blocked turn's history from the store.
        hook_state["results"] = []
        agent2 = make_agent()
        history = db.get_messages_as_conversation(sid)
        result2 = agent2.run_conversation("hello again", conversation_history=history, task_id="t")
        assert result2["final_response"] == "MODEL-ANSWER"
        reqs = _chat_requests(handler)
        assert len(reqs) == 1
        roles = [m["role"] for m in reqs[0]["messages"]]
        # No two identical non-tool roles adjacent (alternation invariant).
        for a, b in zip(roles, roles[1:]):
            if a == b:
                assert a == "tool"

    def test_block_without_message_is_ignored(self, block_env):
        make_agent, handler, db, sid, hook_state = block_env
        hook_state["results"] = [{"action": "block"}]
        agent = make_agent()
        result = agent.run_conversation("hi", conversation_history=[], task_id="t")
        assert result["final_response"] == "MODEL-ANSWER"
        assert len(_chat_requests(handler)) == 1

    def test_first_block_wins_and_context_hooks_unaffected(self, block_env):
        make_agent, handler, db, sid, hook_state = block_env
        hook_state["results"] = [
            {"context": "some recalled context"},
            {"action": "block", "message": "first"},
            {"action": "block", "message": "second"},
        ]
        agent = make_agent()
        result = agent.run_conversation("hi", conversation_history=[], task_id="t")
        assert result["final_response"] == "first"
        assert _chat_requests(handler) == []


# ---------------------------------------------------------------------------
# Prologue unit: directive extraction
# ---------------------------------------------------------------------------

class TestDirectiveExtraction:
    def test_context_still_collected_when_no_block(self, block_env):
        make_agent, handler, db, sid, hook_state = block_env
        hook_state["results"] = [{"context": "CTX-A"}, {"context": "CTX-B"}]
        agent = make_agent()
        result = agent.run_conversation("hi", conversation_history=[], task_id="t")
        assert result["final_response"] == "MODEL-ANSWER"
        sent = _chat_requests(handler)[0]
        user_msgs = [m for m in sent["messages"] if m["role"] == "user"]
        assert "CTX-A" in user_msgs[0]["content"]
        assert "CTX-B" in user_msgs[0]["content"]


# ---------------------------------------------------------------------------
# Shell-hook stdout translation
# ---------------------------------------------------------------------------

class TestShellHookParse:
    def test_pre_llm_call_block_hermes_shape(self):
        from agent.shell_hooks import _parse_response
        out = _parse_response("pre_llm_call", json.dumps({"action": "block", "message": "nope"}))
        assert out == {"action": "block", "message": "nope"}

    def test_pre_llm_call_block_claude_code_shape(self):
        from agent.shell_hooks import _parse_response
        out = _parse_response("pre_llm_call", json.dumps({"decision": "block", "reason": "nope"}))
        assert out == {"action": "block", "message": "nope"}

    def test_pre_llm_call_context_passthrough_unchanged(self):
        from agent.shell_hooks import _parse_response
        out = _parse_response("pre_llm_call", json.dumps({"context": "Today is Friday"}))
        assert out == {"context": "Today is Friday"}
