"""Regression for #84698 — invalid-JSON tool recovery keyed by call id.

Two parallel calls to the same tool (one valid JSON, one invalid) used to
collide: recovery matched by tool *name*, so the valid sibling inherited the
Invalid-JSON error and never executed, and the truncation scan could
false-positive on a complete sibling that merely shared the broken call's name.

The fix mirrors the mixed invalid-*name* batch path: error-result only the
broken calls, execute valid siblings, key everything by ``tool_call_id``.
Exercised end-to-end through ``AIAgent.run_conversation`` against an in-process
mock provider, asserting the message-shape contract (which ids got which
results, how many API calls were spent) rather than any message text snapshot.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

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
        # Non-stream only: the fixture disables streaming because streaming
        # assembly auto-repairs bad JSON before the validation path sees it.
        assert req.get("stream") is not True
        resp = type(self).response_queue.pop(0) if type(self).response_queue else _text_resp("DONE")
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def _batch_tc_resp(calls: list[tuple[str, str]]) -> dict:
    return {
        "id": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                        for i, (name, args) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    """Mock provider + isolated HERMES_HOME; yields (agent, handler, dispatched-names list)."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_84698_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    # Import fresh so the real validation path is exercised even when these
    # modules were imported earlier in the same worker; restored in finally.
    saved_modules = dict(sys.modules)
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
    agent.valid_tool_names = {"terminal", "read_file", "write_file", "execute_code", "session_search", "todo"}
    agent._disable_streaming = True

    # Dispatch spy at the round's single execution entry point: record every
    # (name, args) the tool round hands to execution, then run the real thing.
    dispatched: list[tuple[str, str]] = []
    _real_execute = agent._execute_tool_calls

    def _spy(assistant_message, *a, **kw):
        dispatched.extend((tc.function.name, tc.function.arguments) for tc in assistant_message.tool_calls)
        return _real_execute(assistant_message, *a, **kw)

    agent._execute_tool_calls = _spy  # type: ignore[method-assign]

    try:
        yield agent, _MockHandler, dispatched
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)
        sys.modules.clear()
        sys.modules.update(saved_modules)
        for _name, _mod in saved_modules.items():
            _parent, _, _child = _name.rpartition(".")
            if _parent and _parent in saved_modules:
                try:
                    setattr(saved_modules[_parent], _child, _mod)
                except Exception:
                    pass
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _tool_result_ids(messages) -> Counter:
    """Multiplicity of tool_call_id across tool-role results (lossless, unlike a dict)."""
    return Counter(
        m.get("tool_call_id") or ""
        for m in messages or []
        if isinstance(m, dict) and m.get("role") == "tool"
    )


def _tool_result_content(messages, tool_call_id: str) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
            return m.get("content") or ""
    raise AssertionError(f"no tool result for {tool_call_id}")


def _assistant_tool_call_ids(messages) -> Counter:
    return Counter(
        tc["id"]
        for m in messages or []
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        for tc in m["tool_calls"]
    )


def _chat_calls(handler) -> list:
    return [r for r in handler.captured_requests if "messages" in r]


def test_mixed_json_same_name_executes_valid_sibling(agent_env):
    """Valid same-name sibling executes; only the broken call gets Invalid JSON; no retry burn."""
    agent, handler, dispatched = agent_env
    handler.response_queue.append(_batch_tc_resp([("todo", "{bad}"), ("todo", "{}")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    msgs = result.get("messages")
    assert "Invalid JSON" in _tool_result_content(msgs, "call_0")
    ok = _tool_result_content(msgs, "call_1")
    assert "Invalid JSON" not in ok and not ok.startswith(("Error:", "Skipped:"))
    # The valid sibling really reached dispatch — and only it.
    assert dispatched == [("todo", "{}")]
    # No whole-turn JSON retry when a valid sibling exists: batch + final text.
    assert len(_chat_calls(handler)) == 2


def test_mixed_json_preserves_tool_call_result_pairing(agent_env):
    """Every emitted tool_call keeps exactly one result (no orphans, no duplicates)."""
    agent, handler, _ = agent_env
    handler.response_queue.append(_batch_tc_resp([("todo", "{bad}"), ("todo", "{}")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    msgs = result.get("messages")
    assert _assistant_tool_call_ids(msgs) == Counter({"call_0": 1, "call_1": 1})
    assert _tool_result_ids(msgs) == Counter({"call_0": 1, "call_1": 1})
    # The next outbound request carries the same pairing to the provider.
    final_req = _chat_calls(handler)[-1]["messages"]
    assert _assistant_tool_call_ids(final_req) == Counter({"call_0": 1, "call_1": 1})
    assert _tool_result_ids(final_req) == Counter({"call_0": 1, "call_1": 1})


def test_all_invalid_json_still_retries_then_recovers(agent_env):
    """All-broken batch keeps the retry policy, and recovery errors are attributed per call id."""
    agent, handler, dispatched = agent_env
    # Distinguishable decoder messages so per-id attribution is provable.
    bad_a, bad_b = "{bad}", '{"a" 1}'
    for _ in range(3):
        handler.response_queue.append(_batch_tc_resp([("todo", bad_a), ("todo", bad_b)]))
    handler.response_queue.append(_text_resp("recovered"))

    result = agent.run_conversation("retry path", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    # Initial attempt + 2 silent retries, then recovery inject + one more call to answer.
    assert len(_chat_calls(handler)) == 4
    assert dispatched == []
    msgs = result.get("messages")
    with pytest.raises(json.JSONDecodeError) as ea:
        json.loads(bad_a)
    with pytest.raises(json.JSONDecodeError) as eb:
        json.loads(bad_b)
    assert str(ea.value) != str(eb.value)
    assert str(ea.value) in _tool_result_content(msgs, "call_0")
    assert str(eb.value) in _tool_result_content(msgs, "call_1")
    assert _tool_result_ids(msgs) == Counter({"call_0": 1, "call_1": 1})


def test_truncation_scan_does_not_false_positive_on_same_name_valid_sibling(agent_env):
    """A complete same-name sibling must not be mistaken for truncated output.

    ``"0"`` is valid JSON that does not end in ``}``/``]``. Under name-keying the
    truncation scan saw a ``todo`` call failing the suffix heuristic and aborted the
    whole turn as "truncated". Keyed by id, only the genuinely broken call is scanned.
    """
    agent, handler, dispatched = agent_env
    handler.response_queue.append(_batch_tc_resp([("todo", "{bad}"), ("todo", "0")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    assert result.get("error") != "Response truncated due to output length limit"
    msgs = result.get("messages")
    assert "Invalid JSON" in _tool_result_content(msgs, "call_0")
    assert dispatched == [("todo", "0")]


def test_genuinely_truncated_call_still_hard_stops_despite_valid_sibling(agent_env):
    """Policy lock: a truncated-looking broken call refuses the whole turn even with a valid sibling."""
    agent, handler, dispatched = agent_env
    handler.response_queue.append(_batch_tc_resp([("todo", '{"a":'), ("todo", "{}")]))
    handler.response_queue.append(_text_resp("should not be reached"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("error") == "Response truncated due to output length limit"
    assert dispatched == []
    assert len(_chat_calls(handler)) == 1


@pytest.mark.parametrize("order", ["broken_first", "valid_first"])
def test_dedup_never_evicts_valid_sibling_or_orphans_a_result(agent_env, order):
    """Dedup runs between validation and dispatch; a broken call must never collapse onto
    a valid one, and a genuinely duplicated broken call must not leave an orphan id."""
    agent, handler, dispatched = agent_env
    calls = [("todo", "{bad}"), ("todo", "{bad}"), ("todo", "{}")]
    if order == "valid_first":
        calls.reverse()
    handler.response_queue.append(_batch_tc_resp(calls))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    assert result.get("completed", False)
    assert dispatched == [("todo", "{}")]
    msgs = result.get("messages")
    emitted = _assistant_tool_call_ids(msgs)
    # Whatever dedup kept, every emitted call has exactly one result and vice versa.
    assert set(emitted.values()) == {1}
    assert _tool_result_ids(msgs) == emitted
    assert len(emitted) == 2  # one duplicate {bad} collapsed, the valid call survived
