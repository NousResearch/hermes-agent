"""Focused seam tests for CL-R4-1 response content normalization.

Every probe here is a *runtime behavioral* check: it either exercises the
real ``run_conversation`` path against an in-process mock provider (call-site
seam, lifecycle-hook ordering) or calls the extracted helper directly
(normalization semantics, mutation timing, downstream ``.strip()`` safety,
import identity, monkeypatch transparency). No test reads source files,
parses ASTs, shells out to git/subprocess, or inspects repository state
(AGENTS.md: never read source code in tests).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

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


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    """Spin up the mock provider + an isolated HERMES_HOME, yield (agent, handler)."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_clr41_")
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


# ── Runtime normalization semantics (kept) ──────────────────────────────────


def test_normalize_assistant_content_cases_and_identity():
    from agent.response_normalization import normalize_assistant_content

    cases = [
        ("plain", "plain"),
        (None, None),
        ({"text": "preferred", "content": "ignored"}, "preferred"),
        ({"content": "fallback"}, "fallback"),
        ({"other": 1}, json.dumps({"other": 1})),
        (["a", {"type": "text", "text": "b"}, {"type": "image", "text": 3}, {"text": 4}, {"type": "image"}], "a\nb\n3\n4"),
        ([{"type": "image"}, 3, None], ""),
        (42, "42"),
    ]
    for raw, expected in cases:
        message = SimpleNamespace(content=raw)
        original = message
        assert normalize_assistant_content(message) is None
        assert message is original
        assert message.content == expected


def test_normalize_assistant_content_preserves_exceptions_and_mutation_timing(monkeypatch):
    from agent import response_normalization

    message = SimpleNamespace(content={"other": object()})
    with pytest.raises(TypeError):
        response_normalization.normalize_assistant_content(message)
    assert isinstance(message.content, dict)

    message = SimpleNamespace(content=[{"type": "text", "text": None}])
    with pytest.raises(TypeError):
        response_normalization.normalize_assistant_content(message)
    assert message.content == [{"type": "text", "text": None}]

    message = SimpleNamespace(content="already string")
    response_normalization.normalize_assistant_content(message)
    assert message.content.strip() == "already string"


def test_normalized_content_is_strip_safe_downstream():
    """Downstream consumers call ``.strip()`` on the normalized result."""
    from agent.response_normalization import normalize_assistant_content

    message = SimpleNamespace(content={"text": "  hello  "})
    normalize_assistant_content(message)
    # Normalization is verbatim (no strip inside the helper)...
    assert message.content == "  hello  "
    # ...so the downstream .strip() call is safe on the resulting str.
    assert message.content.strip() == "hello"

    message = SimpleNamespace(content=[{"type": "text", "text": "a"}, " b "])
    normalize_assistant_content(message)
    assert message.content == "a\n b "
    assert message.content.strip() == "a\n b"


# ── Runtime call-site seam + lifecycle-hook ordering probes ─────────────────


def test_call_site_runs_normalize_between_transport_and_post_api_hook(agent_env, monkeypatch):
    """Runtime ordering probe for the call-site seam.

    Runs the real ``run_conversation`` path against the in-process mock
    provider and spies on the transport's ``normalize_response``, the
    extracted ``normalize_assistant_content`` helper, and the lifecycle hook
    dispatch. Asserts the helper is invoked on the assistant message AFTER
    transport normalization and BEFORE the ``post_api_request`` hook — by
    runtime observation, not by parsing source.
    """
    import agent.conversation_loop as conversation_loop
    import agent.response_normalization as response_normalization
    from hermes_cli import lifecycle

    agent, handler = agent_env
    # The normalize_assistant_content call-site seam lives on the
    # non-streaming response path (transport normalize → helper → hook).
    # Force the non-streaming path so the probe exercises that exact seam.
    agent._disable_streaming = True
    transport = agent._get_transport()

    order = []
    real_normalize_response = transport.normalize_response
    real_normalize = response_normalization.normalize_assistant_content

    def spy_normalize_response(response, **kwargs):
        order.append("normalize_response")
        return real_normalize_response(response, **kwargs)

    def spy_normalize(message):
        order.append("normalize_assistant_content")
        return real_normalize(message)

    def spy_has_hook(name):
        return name == "post_api_request"

    def spy_invoke_hook(name, **kwargs):
        order.append(f"hook:{name}")
        return []

    monkeypatch.setattr(transport, "normalize_response", spy_normalize_response)
    monkeypatch.setattr(response_normalization, "normalize_assistant_content", spy_normalize)
    monkeypatch.setattr(lifecycle, "has_hook", spy_has_hook)
    monkeypatch.setattr(lifecycle, "invoke_hook", spy_invoke_hook)

    handler.response_queue = [_text_resp("hello world")]
    result = agent.run_conversation("hi", conversation_history=[], task_id="t")

    assert "normalize_response" in order
    assert "normalize_assistant_content" in order
    assert "hook:post_api_request" in order
    assert order.index("normalize_response") < order.index("normalize_assistant_content")
    assert order.index("normalize_assistant_content") < order.index("hook:post_api_request")
    assert result.get("final_response") == "hello world"


# ── Runtime import identity + monkeypatch transparency ───────────────────────


def test_run_conversation_import_identity_and_monkeypatch_transparency(monkeypatch):
    import agent.conversation_loop as conversation_loop
    from agent import model_metadata

    # Runtime import identity: the module attribute is the real callable.
    assert callable(conversation_loop.run_conversation)
    assert conversation_loop.run_conversation.__module__ == "agent.conversation_loop"

    # Extraction boundary: the loop re-exports the shared helpers by identity
    # and does not define the normalization helper itself.
    for name in ("save_context_length", "estimate_request_tokens_rough", "estimate_messages_tokens_rough", "conversation_history_after_compression"):
        assert hasattr(conversation_loop, name)
    assert conversation_loop.save_context_length is model_metadata.save_context_length
    assert not hasattr(conversation_loop, "normalize_assistant_content")

    # Monkeypatch transparency: patching the module attribute intercepts calls.
    calls = []

    def fake_run_conversation(*args, **kwargs):
        calls.append((args, kwargs))
        return {"final_response": "intercepted"}

    monkeypatch.setattr(conversation_loop, "run_conversation", fake_run_conversation)
    assert conversation_loop.run_conversation is fake_run_conversation
    result = conversation_loop.run_conversation("probe")
    assert result == {"final_response": "intercepted"}
    assert len(calls) == 1


def test_response_normalization_module_runtime_probe():
    """The extracted module is importable and exposes only the helper."""
    import agent.response_normalization as m

    assert callable(m.normalize_assistant_content)
    assert m.normalize_assistant_content.__module__ == "agent.response_normalization"
    # Runtime surface check (no source reading): the module's only public
    # symbols are the cheap ``json`` import and the extracted helper.
    public = {name for name in dir(m) if not name.startswith("_")}
    assert public == {"json", "normalize_assistant_content"}
