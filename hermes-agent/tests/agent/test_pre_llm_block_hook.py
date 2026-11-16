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


def _agent(provider_url, db):
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key", base_url=provider_url, provider="openai-compat", model="test-model",
        max_iterations=4, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
        skip_memory=True, save_trajectories=False, platform="cli", session_db=db, session_id="s-block",
    )


def _blocked_session(provider_url, db):
    """Run one vetoed turn on ``s-block``; return its result."""
    verdicts = [{"context": "IGNORED-ON-BLOCK"}, {"action": "block", "message": BLOCK_MSG}]
    with patch("hermes_cli.plugins.invoke_hook",
               side_effect=lambda hook, **kw: list(verdicts) if hook == "pre_llm_call" else []):
        return _agent(provider_url, db).run_conversation(f"look up {SECRET}", conversation_history=[], task_id="t")


def test_blocked_turn_never_reaches_the_provider_and_stores_only_the_placeholder(provider_url, tmp_path, monkeypatch):
    """Turn 1 is vetoed: zero chat requests, the block message is the reply and closes the
    transcript pair. The stored row's content is the placeholder (what session_search and every
    other content reader sees); the typed text is display-only metadata the transcript renders.
    Turn 2 (a fresh agent reloading the session) replays the placeholder, never the prompt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    from agent.message_metadata import withheld_prompt_text
    from tools.session_search_tool import session_search

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        result = _blocked_session(provider_url, db)
        assert _Provider.chat_requests == []
        assert result["final_response"] == BLOCK_MSG
        assert result["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON
        assert result["api_calls"] == 0 and result["failed"] is False

        history = db.get_messages_as_conversation("s-block")
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", PRE_LLM_BLOCK_WITHHELD_PROMPT), ("assistant", BLOCK_MSG)]
        assert withheld_prompt_text(history[0]) == f"look up {SECRET}"
        assert SECRET not in session_search(query="CUSTOMER", db=db, current_session_id="other")

        follow_up = _agent(provider_url, db).run_conversation("next question", conversation_history=history,
                                                              task_id="t")
        assert follow_up["final_response"] == "MODEL-ANSWER"
        (sent,) = _Provider.chat_requests
        wire = json.dumps(sent["messages"])
        assert SECRET not in wire and PRE_LLM_BLOCK_WITHHELD_PROMPT in wire
    finally:
        db.close()


def test_compaction_of_a_blocked_turn_never_sends_the_prompt(provider_url, tmp_path, monkeypatch):
    """A later compaction serializes the stored rows for the summarizer and quotes user turns in its
    handoff: neither the summary request nor the compacted history may carry the vetoed prompt."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"auxiliary:\n  compression: {{provider: custom, base_url: '{provider_url}', api_key: k, model: m}}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent.conversation_compression import compress_context

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _blocked_session(provider_url, db)
        blocked_pair = db.get_messages_as_conversation("s-block")
        filler = [{"role": role, "content": f"{role} {i} " + "x" * 400}
                  for i in range(8) for role in ("user", "assistant")]
        agent = _agent(provider_url, db)
        compressed, _ = compress_context(agent, filler[:12] + blocked_pair + filler, "sys",
                                         approx_tokens=150000, force=True)
        assert _Provider.chat_requests, "the summarizer was never called"
        assert SECRET not in json.dumps(_Provider.chat_requests)
        assert SECRET not in json.dumps([{k: m.get(k) for k in ("content", "api_content")} for m in compressed])
    finally:
        db.close()


def test_retry_and_undo_after_a_block_resubmit_the_typed_text(provider_url, tmp_path, monkeypatch):
    """/retry (CLI, TUI, gateway, ACP, Desktop) and /undo read a rewind's live view or receipt: both carry
    what the user typed, so the hook re-decides on it (0 model calls, blocked again), never the placeholder."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    import hermes_state_runtime as rt

    typed = f"look up {SECRET}"
    for name in ("receipt", "rewind"):
        db = SessionDB(db_path=tmp_path / f"{name}.db")
        try:
            _blocked_session(provider_url, db)
            if name == "receipt":
                epoch = rt.begin_runtime_epoch(db, instance_id="owner")
                receipt = rt.mutate_runtime_session(
                    db, epoch=epoch, principal_id="human", session_id="s-block", request_id="r", expected_revision=0,
                    expected_generation=0, operation="rewind", payload={"target_message_id": db.get_messages("s-block")[0]["id"]})
                assert receipt["target_message"]["content"] == typed
                continue
            history = db.get_messages_as_conversation("s-block")
            outcome = db.rewind_user_turn("s-block", -1, warm_history=history, require_retryable=True)
            assert outcome.live_text == typed and outcome.prefix == []
            dlp = lambda hook, **kw: ([{"action": "block", "message": BLOCK_MSG}] if hook == "pre_llm_call"
                                      and SECRET in kw["user_message"] else [])
            with patch("hermes_cli.plugins.invoke_hook", side_effect=dlp):
                again = _agent(provider_url, db).run_conversation(outcome.live_text, conversation_history=[], task_id="t")
            assert again["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON and _Provider.chat_requests == []
        finally:
            db.close()


@pytest.mark.parametrize(("stdout", "expected"), [
    ('{"action": "block", "message": "no"}', {"action": "block", "message": "no"}),
    ('{"decision": "block", "reason": "no"}', {"action": "block", "message": "no"}),
    ('{"context": "today is Friday"}', {"context": "today is Friday"}),
])
def test_shell_hook_pre_llm_call_block_in_both_dialects(stdout, expected):
    assert shell_hooks._parse_response("pre_llm_call", stdout) == expected
