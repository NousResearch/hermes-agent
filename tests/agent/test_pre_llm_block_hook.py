"""``pre_llm_call`` block directive: a policy hook vetoes a turn before any inference request (#57364)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from agent import shell_hooks
from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON, PRE_LLM_BLOCK_WITHHELD_PROMPT
from hermes_state import SessionDB

BLOCK_MSG = "Blocked by policy: prompt contains a customer ID."
SECRET = "CUSTOMER-ID-4242"


class _Provider(BaseHTTPRequestHandler):
    chat_requests: list = []

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        if "messages" in req:
            type(self).chat_requests.append(req)
        if req.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in (({"role": "assistant", "content": "MODEL-ANSWER"}, None), ({}, "stop")):
                chunk = {"id": "m", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({
            "id": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "MODEL-ANSWER"}, "finish_reason": "stop"}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


@pytest.fixture()
def provider_url():
    _Provider.chat_requests = []
    srv = HTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_blocked_turn_never_reaches_the_provider_and_is_never_replayed(provider_url, tmp_path, monkeypatch):
    """Turn 1 is vetoed: zero chat requests, the block message is the reply and closes the
    transcript pair. Turn 2 (a fresh agent reloading the session) reaches the model with the
    withheld placeholder in place of the vetoed prompt, while the transcript keeps the original."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from run_agent import AIAgent

    db = SessionDB(db_path=tmp_path / "state.db")
    verdicts = {"pre_llm_call": [{"context": "IGNORED-ON-BLOCK"}, {"action": "block", "message": BLOCK_MSG}]}

    def make_agent():
        return AIAgent(
            api_key="test-key", base_url=provider_url, provider="openai-compat", model="test-model",
            max_iterations=4, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
            skip_memory=True, save_trajectories=False, platform="cli", session_db=db, session_id="s-block",
        )

    try:
        with patch("hermes_cli.plugins.invoke_hook", side_effect=lambda hook, **kw: list(verdicts.get(hook, []))):
            result = make_agent().run_conversation(f"look up {SECRET}", conversation_history=[], task_id="t")
            assert _Provider.chat_requests == []
            assert result["final_response"] == BLOCK_MSG
            assert result["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON
            assert result["api_calls"] == 0 and result["failed"] is False

            verdicts.clear()
            history = db.get_messages_as_conversation("s-block")
            assert [(m["role"], m["content"]) for m in history] == [
                ("user", f"look up {SECRET}"), ("assistant", BLOCK_MSG)]
            follow_up = make_agent().run_conversation("next question", conversation_history=history, task_id="t")
        assert follow_up["final_response"] == "MODEL-ANSWER"
        (sent,) = _Provider.chat_requests
        wire = json.dumps(sent["messages"])
        assert SECRET not in wire and PRE_LLM_BLOCK_WITHHELD_PROMPT in wire
    finally:
        db.close()


@pytest.mark.parametrize(("stdout", "expected"), [
    ('{"action": "block", "message": "no"}', {"action": "block", "message": "no"}),
    ('{"decision": "block", "reason": "no"}', {"action": "block", "message": "no"}),
    ('{"context": "today is Friday"}', {"context": "today is Friday"}),
])
def test_shell_hook_pre_llm_call_block_in_both_dialects(stdout, expected):
    assert shell_hooks._parse_response("pre_llm_call", stdout) == expected
