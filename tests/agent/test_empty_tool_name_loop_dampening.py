"""Regression for #47967 — empty-name phantom tool calls.

Weak open models (mimo, nemotron-class) that see tool-call XML/JSON sitting in
file contents or tool output get *primed* and emit their own structured tool
calls that mimic the payload — usually with an empty/whitespace ``name``. Those
calls can't be fuzzy-repaired toward a real tool, so the dispatch loop returns an
error and the model retries. Before this fix, every empty-name error dumped the
full tool catalog back to the model, which fed the priming loop more names to
mimic and inflated context 3-4x across the retry budget.

The fix: a blank/whitespace-only tool name gets a terse anti-priming error that
tells the model in-context tool-call syntax is DATA, with NO catalog dump. A
genuinely-wrong-but-nonempty name (an actual typo) still gets the full catalog
so the model can self-correct.

These assert the *behavior contract* of the dispatch branch (what content goes
back to the model for each name shape), exercised end-to-end through
``AIAgent.run_conversation`` against an in-process mock provider — not a snapshot
of the message string.
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

    def log_message(self, *a, **kw):  # silence the default stderr logging
        pass


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


def _batch_tc_resp(calls: list[tuple[str, str]]) -> dict:
    """Multi-call batch response: calls = [(name, arguments), ...]."""
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": name, "arguments": args}}
                for i, (name, args) in enumerate(calls)
            ]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    """Spin up the mock provider + an isolated HERMES_HOME, yield (agent, helpers)."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_47967_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    # Import fresh so the dispatch logic is exercised even when the module was
    # imported earlier in the same worker.  We purge run_agent + agent.*/tools.*/hermes_*
    # EXCEPT for a curated set of modules that other test files hold direct class
    # references to.  agent.context_compressor binds call_llm at module level
    # via from agent.auxiliary_client import call_llm; if we purge and reimport
    # it, a new module object is created, but tests that did
    # from agent.context_compressor import ContextCompressor at their own
    # module load still reference the OLD class.  patch("agent.context_compressor.
    # call_llm") then patches the NEW module's binding, which the OLD class never
    # sees — the real call_llm runs, hits the network, and the fallback path is
    # never taken (confirmed contamination: test_ghost_skill_pruning lines 268/302).
    _PRESERVE_MODULES = {
        "agent.context_compressor",
        "agent.auxiliary_client",
    }
    for mod in list(sys.modules):
        if mod in _PRESERVE_MODULES:
            continue
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
    agent.valid_tool_names = {"terminal", "read_file", "write_file", "execute_code", "session_search"}

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        # Stop the async QueueListener and close file handlers that point at
        # the temporary HERMES_HOME/logs/ *before* we rmtree the directory.
        # AIAgent.__init__ calls hermes_logging.setup_logging(), which starts a
        # QueueListener whose RotatingFileHandlers write to
        # <test_home>/.hermes/logs/agent.log and errors.log.  If we delete the
        # directory while the listener is still alive, subsequent log records
        # hit FileNotFoundError on the missing parent and emit tracebacks.
        # _reset_queued_handlers() stops the listener (draining pending records),
        # closes the handlers, and clears the module-level state so a later
        # setup_logging() in the same interpreter starts clean.
        import hermes_logging
        hermes_logging._reset_queued_handlers()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _tool_results(handler) -> list[str]:
    out = []
    for req in handler.captured_requests:
        for m in req.get("messages", []):
            if m.get("role") == "tool":
                out.append(m.get("content", ""))
    return out


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t "])
def test_empty_tool_name_gets_terse_error_no_catalog(agent_env, blank):
    """A blank/whitespace tool name must NOT trigger a full tool-catalog dump."""
    agent, handler = agent_env
    handler.response_queue.append(_tc_resp(blank, "{}"))
    handler.response_queue.append(_text_resp("Recovered in plain text."))

    agent.run_conversation("read ./payload and report", conversation_history=[], task_id="t")

    joined = " ".join(_tool_results(handler))
    assert "tool name was empty" in joined
    # The whole point: do not feed the priming loop the catalog of names.
    assert "Available tools:" not in joined


def test_unknown_nonempty_name_keeps_catalog(agent_env):
    """A genuinely-wrong NONempty name still gets the catalog for self-correction."""
    agent, handler = agent_env
    handler.response_queue.append(_tc_resp("frobnicate_xyz", "{}"))
    handler.response_queue.append(_text_resp("ok plain text"))

    agent.run_conversation("do a thing", conversation_history=[], task_id="t")

    joined = " ".join(_tool_results(handler))
    assert "frobnicate_xyz" in joined
    assert "Available tools:" in joined
    assert "tool name was empty" not in joined


# ── Mixed batches: valid calls execute, invalid calls get error results ──
#
# Degrading models (observed with gpt-5.6 past ~350K input; jonny's July 2026
# report) emit batches like 6 named calls + 1 blank-name call. Before the fix,
# the whole turn was voided ("Skipped: another tool call in this turn used an
# invalid name") and three such batches halted the session as partial even
# though most of the model's work was coherent.


def test_mixed_batch_executes_valid_and_errors_blank(agent_env):
    """Valid siblings of a blank-name call must execute, not be skipped."""
    agent, handler = agent_env
    agent.valid_tool_names = agent.valid_tool_names | {"todo"}
    handler.response_queue.append(_batch_tc_resp([("todo", "{}"), ("", "{}")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    joined = " ".join(_tool_results(handler))
    # The blank call got the terse anti-priming error...
    assert "tool name was empty" in joined
    # ...the valid sibling was NOT punished...
    assert "Skipped: another tool call" not in joined
    # ...and actually executed (todo returns its list, not an error result).
    assert result.get("completed", False)


def test_mixed_batch_preserves_tool_call_result_pairing(agent_env):
    """Every emitted tool_call keeps a matching tool result (provider invariant)."""
    agent, handler = agent_env
    agent.valid_tool_names = agent.valid_tool_names | {"todo"}
    handler.response_queue.append(_batch_tc_resp([("todo", "{}"), ("", "{}")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    msgs = result["messages"]
    tc_ids = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            tc_ids.extend(tc["id"] for tc in m["tool_calls"])
    result_ids = [
        m.get("tool_call_id") or "" for m in msgs
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    # Both the valid and blank call must appear in the assistant message,
    # and each must have exactly one matching tool result.
    assert set(tc_ids) == {"call_0", "call_1"}
    assert sorted(result_ids) == sorted(tc_ids)


def test_mixed_batches_do_not_strike_out_session(agent_env):
    """4 consecutive mixed batches must not trip the 3-strike halt."""
    agent, handler = agent_env
    agent.valid_tool_names = agent.valid_tool_names | {"todo"}
    for _ in range(4):
        handler.response_queue.append(_batch_tc_resp([("todo", "{}"), ("", "{}")]))
    handler.response_queue.append(_text_resp("survived"))

    result = agent.run_conversation("keep going", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    assert not result.get("partial", False)
    assert "survived" in (result.get("final_response") or "")


def test_all_invalid_batch_still_strikes_out(agent_env):
    """A turn with NO valid call must still advance the 3-strike halt."""
    agent, handler = agent_env
    for _ in range(3):
        handler.response_queue.append(_batch_tc_resp([("", "{}"), ("  ", "{}")]))

    result = agent.run_conversation("degenerate", conversation_history=[], task_id="t")

    assert result.get("partial", False)
    assert "invalid tool call" in (result.get("error") or "")


def test_invalid_tool_exhaustion_closes_tool_tail(agent_env):
    """Invalid-tool 3-strike partial must not leave a durable tool→user tail (#48879 class).

    Retries <3 append assistant+error tool rows, so the transcript already ends
    on ``tool`` before the exhaustion early-return. That return must close the
    sequence (same contract as interrupt aborts) so the next user turn is not
    ``tool → user`` for strict providers.
    """
    agent, handler = agent_env
    for _ in range(3):
        handler.response_queue.append(_tc_resp("frobnicate_xyz", "{}"))

    result = agent.run_conversation("degenerate", conversation_history=[], task_id="t")

    assert result.get("partial", False)
    msgs = result.get("messages") or []
    assert msgs, "expected persisted conversation messages"
    assert msgs[-1].get("role") == "assistant"
    assert "invalid tool call" in (msgs[-1].get("content") or "").lower()


def test_mixed_batch_invalid_call_with_broken_json_does_not_retry_turn(agent_env):
    """Broken args on a never-executing invalid call must not trigger the JSON retry loop."""
    agent, handler = agent_env
    agent.valid_tool_names = agent.valid_tool_names | {"todo"}
    handler.response_queue.append(_batch_tc_resp([("todo", "{}"), ("", '{"unclosed')]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    # Exactly 2 chat API calls: the batch turn + the final answer. A JSON
    # retry would add a third identical request.
    chat_calls = [r for r in handler.captured_requests if "messages" in r]
    assert len(chat_calls) == 2

def test_agent_env_preserves_context_compressor_identity(agent_env):
    """Regression: the module purge must not replace agent.context_compressor.

    Other test files (test_ghost_skill_pruning, test_compressor_fallback_update,
    ...) import ContextCompressor at their module load time and later patch
    agent.context_compressor.call_llm.  If this fixture purges and reimports
    agent.context_compressor, a NEW module object is created, but those tests
    still hold the OLD ContextCompressor class — whose compress() calls the OLD
    module-level call_llm binding, not the patched one.  The patch silently
    misses and the real call_llm runs (network hit, no fallback).
    """
    import agent.context_compressor as cc
    # The module in sys.modules must be the same object that other test files
    # imported — not a fresh reimport created by the fixture's purge loop.
    assert sys.modules["agent.context_compressor"] is cc
    # call_llm must be a live attribute of the preserved module so that
    # patch("agent.context_compressor.call_llm") reaches ContextCompressor.
    assert hasattr(cc, "call_llm")


def test_agent_env_teardown_flushes_queue_listener():
    """Regression: the fixture must stop the async QueueListener before
    deleting the temporary HERMES_HOME so file handlers do not keep writing to
    deleted agent.log/errors.log paths (FileNotFoundError tracebacks).

    Drives the fixture generator manually to inspect post-teardown state.
    """
    import hermes_logging
    # agent_env is a @pytest.fixture-wrapped generator; call the underlying
    # generator function directly to control setup/teardown.
    original = agent_env.__wrapped__
    gen = original()
    next(gen)  # setup + yield (agent, handler)
    try:
        gen.send(None)  # trigger finally
    except StopIteration:
        pass
    # After teardown, the listener must be stopped and file handlers cleared.
    assert hermes_logging._queue_listener is None, \
        "QueueListener leaked after fixture teardown"
    assert len(hermes_logging._queued_file_handlers) == 0, \
        "file handlers leaked after fixture teardown"
