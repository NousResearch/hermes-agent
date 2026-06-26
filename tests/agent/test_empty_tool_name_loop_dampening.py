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
from types import ModuleType

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _restore_parent_module_attrs(modules: dict[str, ModuleType]) -> None:
    """Restore parent-package attributes after sys.modules surgery.

    Removing ``agent.*`` from ``sys.modules`` and importing a fresh graph mutates
    package attributes such as ``agent.file_safety``. Re-inserting the old module
    objects into ``sys.modules`` is not enough: later imports can otherwise see
    stale package attributes pointing at the fresh graph. Rebind every restored
    child to its restored parent package.
    """
    for name, module in modules.items():
        if "." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = modules.get(parent_name) or sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, module)


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

    # Import fresh so the patched conversation_loop is exercised even when the
    # module was imported earlier in the same worker. Preserve the old module
    # objects so this fixture does not leave stale top-level imports in later
    # tests pointing at modules whose names were deleted/re-imported.
    removed_modules = {}
    for mod in list(sys.modules):
        if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
            removed_modules[mod] = sys.modules.pop(mod)
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
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home
        for mod in list(sys.modules):
            if mod == "run_agent" or mod.startswith("agent.") or mod.startswith("tools.") or mod.startswith("hermes_"):
                sys.modules.pop(mod, None)
        sys.modules.update(removed_modules)
        _restore_parent_module_attrs(removed_modules)


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
