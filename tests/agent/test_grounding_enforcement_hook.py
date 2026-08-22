"""Integration tests for the post-response grounding-enforcement hook.

A grounding-aware context engine may advertise an ``enforce_response``
capability (via a ``capabilities()`` method) so the conversation loop lets it
audit the TRULY FINAL assistant answer and, when ungrounded, return a
replacement. The hook lives at the final-response assembly point in
``agent.conversation_loop.run_conversation`` — past every continuation/nudge
gate (Codex ``incomplete`` ack-continuation, length-continuation join,
dropped-tool-call recovery, verify-on-stop, pre_verify, kanban) — so it only
ever sees the delivered result, never an intermediate fragment.

Unlike a unit test that mirrors the production block, these drive the real
``AIAgent.run_conversation`` against an in-process mock provider and assert on
the returned ``final_response`` and the persisted ``messages``, so they pin the
integration placement AND persistence (the two concerns the sweeper flagged as
unverified). The built-in ContextCompressor/LCM have no ``capabilities`` method,
so the hook is a no-op for them; these tests opt in a fake engine explicitly.
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
    # Set by the fixture before each request cycle.
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
            tcs = msg.get("tool_calls")
            fr = resp["choices"][0].get("finish_reason") or ("tool_calls" if tcs else "stop")
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
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": fr}]})
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

    def log_message(self, *a, **kw):  # silence the default stderr logging
        pass


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _length_resp(text: str) -> dict:
    """A truncated (finish_reason=length) partial that triggers a continuation."""
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _GroundingProbe:
    """Opt-in grounding behavior grafted onto the REAL ContextCompressor.

    We attach ``capabilities`` and ``enforce_response`` onto the live
    ``agent.context_compressor`` rather than replacing it, so the rest of the
    conversation loop (``protect_first_n``, ``update_from_response``, etc.)
    keeps working while the hook still sees an engine that advertises the
    capability. Records every call so tests can assert the hook fired exactly
    once, on the final text only.
    """

    def __init__(self):
        self.calls: list = []

    def capabilities(self):
        return {"enforce_response": True}

    def enforce_response(self, content, messages, model="", final=True):
        self.calls.append({"content": content, "model": model, "final": final})
        if "unsupported" in content:
            return {"action": "replace", "text": "I can't verify that from the record."}
        return {"action": "keep"}


def _graft_engine(agent, probe):
    """Bind a probe's grounding methods onto the agent's real compressor."""
    agent.context_compressor.capabilities = probe.capabilities
    agent.context_compressor.enforce_response = probe.enforce_response


def _graft_broken(agent):
    """Advertise the capability but raise inside enforce_response."""
    agent.context_compressor.capabilities = lambda: {"enforce_response": True}

    def _boom(*a, **k):
        raise RuntimeError("engine blew up")

    agent.context_compressor.enforce_response = _boom


@pytest.fixture()
def agent_env():
    """Spin up the mock provider + an isolated HERMES_HOME, yield (agent, helpers)."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_grounding_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    # Import fresh so the patched conversation_loop is exercised even when the
    # module was imported earlier in the same worker.
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
    agent.valid_tool_names = {"terminal", "read_file", "write_file"}

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _final_assistant_content(result) -> str:
    msgs = result.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content") or ""
    return ""


def test_ungrounded_final_answer_is_replaced_and_persisted(agent_env):
    """An ungrounded final answer is swapped for the engine's replacement in
    BOTH the returned ``final_response`` and the persisted transcript."""
    agent, handler = agent_env
    probe = _GroundingProbe()
    _graft_engine(agent, probe)
    handler.response_queue.append(_text_resp("here is an unsupported claim"))

    result = agent.run_conversation("what is X?", conversation_history=[], task_id="t")

    assert result["final_response"] == "I can't verify that from the record."
    assert _final_assistant_content(result) == "I can't verify that from the record."
    # The hook fired exactly once, on the final text, with final=True.
    assert len(probe.calls) == 1
    assert probe.calls[0]["content"] == "here is an unsupported claim"
    assert probe.calls[0]["final"] is True


def test_grounded_final_answer_is_kept(agent_env):
    """A grounded answer passes through untouched (keep verdict)."""
    agent, handler = agent_env
    probe = _GroundingProbe()
    _graft_engine(agent, probe)
    handler.response_queue.append(_text_resp("a well-grounded answer"))

    result = agent.run_conversation("what is X?", conversation_history=[], task_id="t")

    assert result["final_response"] == "a well-grounded answer"
    assert _final_assistant_content(result) == "a well-grounded answer"
    assert len(probe.calls) == 1


def test_engine_without_capabilities_is_noop(agent_env):
    """The built-in-style engine (no capabilities()) never triggers the hook.

    The stock ContextCompressor ships no ``capabilities`` method, so the hook
    stays dormant and the answer is delivered verbatim.
    """
    agent, handler = agent_env
    assert not hasattr(agent.context_compressor, "capabilities")
    handler.response_queue.append(_text_resp("unsupported but engine has no caps"))

    result = agent.run_conversation("what is X?", conversation_history=[], task_id="t")

    assert result["final_response"] == "unsupported but engine has no caps"


def test_broken_engine_never_breaks_the_turn(agent_env):
    """An engine that raises inside enforce_response must not fail the turn;
    the original answer is delivered unchanged."""
    agent, handler = agent_env
    _graft_broken(agent)
    handler.response_queue.append(_text_resp("original answer stands"))

    result = agent.run_conversation("what is X?", conversation_history=[], task_id="t")

    assert result["final_response"] == "original answer stands"
    assert not result.get("failed")


def test_tool_call_turns_are_not_audited(agent_env):
    """A turn that dispatches a tool then finishes must only audit the FINAL
    text response, never the intermediate tool-call turn."""
    agent, handler = agent_env
    probe = _GroundingProbe()
    _graft_engine(agent, probe)
    # Turn 1: a tool call (intermediate, must be skipped by the hook).
    handler.response_queue.append(_tc_resp("read_file", '{"path": "x"}'))
    # Turn 2: the final text answer (audited).
    handler.response_queue.append(_text_resp("final grounded summary"))

    result = agent.run_conversation("read x and summarize", conversation_history=[], task_id="t")

    assert result["final_response"] == "final grounded summary"
    # Exactly one audit — the final text — despite the earlier tool-call turn.
    assert len(probe.calls) == 1
    assert probe.calls[0]["content"] == "final grounded summary"


def test_length_continuation_audits_only_the_joined_final(agent_env):
    """A length-truncated response that continues must be audited only ONCE,
    on the fully joined final text — not on the intermediate fragment."""
    agent, handler = agent_env
    probe = _GroundingProbe()
    _graft_engine(agent, probe)
    # First response is truncated (finish_reason=length); the loop continues.
    handler.response_queue.append(_length_resp("first part "))
    # Continuation completes the answer.
    handler.response_queue.append(_text_resp("second part"))

    result = agent.run_conversation("write a long answer", conversation_history=[], task_id="t")

    # The engine saw exactly one final audit, and its content includes both
    # joined parts — i.e. it ran on the assembled final text, not the fragment.
    assert len(probe.calls) == 1
    audited = probe.calls[0]["content"]
    assert "first part" in audited and "second part" in audited
    assert probe.calls[0]["final"] is True
