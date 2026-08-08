"""Loop-level behavior for Anthropic's ``pause_turn`` continuation.

``tests/agent/test_anthropic_web_server_tools.py`` covers the transport half:
``pause_turn`` survives ``map_finish_reason`` and normalization as its own stop
reason rather than collapsing into ``stop``. That is necessary but not
sufficient — the behavior that matters lives in ``agent/conversation_loop.py``,
which treats the pause as *provider continuation state* rather than a client
tool call:

* the exact assistant content is appended and replayed, with **no** synthetic
  user or tool message inserted (Anthropic requires the original blocks back);
* the paused turn is persisted, and a persistence failure must not abort the
  continuation — the turn is still recoverable from the live message list;
* continuations are bounded so a pathological upstream cannot silently consume
  the whole agent budget;
* on hitting that bound the fallback chain gets a chance, and only if no
  fallback exists does the turn end as an explicit partial failure.

These drive the real ``run_conversation`` against an in-process mock Anthropic
Messages endpoint, so the assertions are on observable turn behavior — request
count, replayed payloads, result shape — not on a snapshot of internals.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _is_messages_path(path: str) -> bool:
    """True only for the Messages create endpoint.

    ``/v1/messages/count_tokens`` deliberately does not match.
    """
    return path.rstrip("/").endswith("/v1/messages")


class _MockAnthropicHandler(BaseHTTPRequestHandler):
    """Minimal Anthropic Messages endpoint driven by a queued response list.

    Only ``/v1/messages`` draws from the queue. The agent also POSTs probe
    endpoints (``/api/show`` for model capabilities, ``/v1/messages/count_tokens``
    for request sizing); letting those consume queued responses would shift every
    test's script by one and make the suite pass or fail on probe timing rather
    than on loop behavior.
    """

    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append((self.path, req))

        if not _is_messages_path(self.path):
            self._reply({})
            return

        if type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            # Exhausting the script means the loop ran longer than the test
            # scripted for. Answer with a marker the assertions won't accept
            # rather than silently extending the conversation.
            resp = _text_resp("UNSCRIPTED")
        self._reply(resp)

    def _reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        """Silence per-request stderr noise."""
        return


def _pause_resp(query: str = "current weather") -> dict:
    """A server-tool turn Anthropic paused mid-flight."""
    return {
        "id": "msg_pause",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "stop_reason": "pause_turn",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": query},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _tool_use_resp() -> dict:
    """A client tool call — any non-pause turn that keeps the loop iterating.

    The tool is deliberately unregistered: dispatch answers with an error the
    loop feeds back to the model, which is enough to advance the turn without
    standing up a real toolset.
    """
    return {
        "id": "msg_tool",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "no_such_tool",
                "input": {},
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "msg_done",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


@pytest.fixture()
def agent_env():
    """Mock Anthropic endpoint + a real agent pointed at it; yields (agent, handler).

    ``HERMES_HOME`` isolation is not done here: conftest's autouse
    ``_hermetic_environment`` already redirects it to a per-test tempdir. Nor are
    modules purged from ``sys.modules`` — the runner gives every test file its own
    interpreter (see conftest's "Module-level state reset" note), so a purge buys
    nothing and actively breaks sibling files: re-importing ``agent.transports``
    builds a second, empty transport registry while modules imported earlier still
    hold the first, so ``get_transport`` starts returning None for them.
    """
    _MockAnthropicHandler.captured_requests = []
    _MockAnthropicHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockAnthropicHandler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()

    # The file log handler opens <HERMES_HOME>/logs eagerly and floods stderr with
    # FileNotFoundError tracebacks when it is missing.
    os.makedirs(os.path.join(os.environ["HERMES_HOME"], "logs"), exist_ok=True)

    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url=f"http://127.0.0.1:{port}",
        provider="anthropic",
        api_mode="anthropic_messages",
        model="claude-test",
        max_iterations=10,
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        platform="cli",
    )
    # Streaming would need an SSE mock; the pause branch is transport-shape
    # agnostic, so keep the mock endpoint plain JSON. Set dynamically because
    # the agent reads this flag with getattr and never declares it.
    setattr(agent, "_disable_streaming", True)  # noqa: B010

    try:
        yield agent, _MockAnthropicHandler
    finally:
        srv.shutdown()
        srv.server_close()  # shutdown() stops serving; the socket needs closing too
        thread.join(timeout=5)


def _message_calls(handler) -> list[dict]:
    """Bodies of the Messages-API creates, in order.

    Probe POSTs are excluded: counting them as turns would make the
    continuation-budget assertions silently wrong.
    """
    return [body for path, body in handler.captured_requests if _is_messages_path(path)]


def _roles(messages) -> list[str]:
    return [m.get("role") for m in messages if isinstance(m, dict)]


def test_pause_turn_continues_without_synthetic_user_or_tool_message(agent_env):
    """A paused turn is replayed as-is: no fabricated user/tool message."""
    agent, handler = agent_env
    handler.response_queue.append(_pause_resp())
    handler.response_queue.append(_text_resp("Finished after the pause."))

    result = agent.run_conversation("what is the weather", conversation_history=[], task_id="t")

    # The pause consumed one request and the continuation a second one.
    calls = _message_calls(handler)
    assert len(calls) == 2
    assert result["completed"] is True
    assert "Finished after the pause." in (result["final_response"] or "")

    # The continuation request must carry the paused assistant turn back and
    # must NOT invent a user/tool message to carry it.
    replayed = calls[1]["messages"]
    assert replayed[-1]["role"] == "assistant"
    assert _roles(replayed).count("user") == 1
    assert not any(m.get("role") == "tool" for m in replayed)

    # Anthropic requires the native blocks back verbatim — a paused turn
    # replayed as flattened text would be rejected upstream, and that is
    # exactly what a well-meaning "normalize everything to a string" change
    # would do. Pin the block, its id, and its input.
    blocks = replayed[-1]["content"]
    assert isinstance(blocks, list), blocks
    server_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "server_tool_use"]
    assert len(server_uses) == 1, blocks
    assert server_uses[0]["id"] == "srvtoolu_1"
    assert server_uses[0]["name"] == "web_search"
    assert server_uses[0]["input"] == {"query": "current weather"}


def test_pause_turn_persistence_failure_is_reported_and_not_fatal(agent_env, monkeypatch, caplog):
    """A failed flush is named at the pause site, not absorbed anonymously.

    The turn surviving is necessary but not distinguishing: an outer recovery
    handler would swallow the exception and finish the turn anyway. What the
    local ``try/except`` buys is a *diagnosable* failure — the operator sees
    which write failed and for which session, instead of a silent gap between
    a paused turn and its continuation.
    """
    agent, handler = agent_env
    handler.response_queue.append(_pause_resp())
    handler.response_queue.append(_text_resp("Survived a bad flush."))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("session db unavailable")

    monkeypatch.setattr(agent, "_flush_messages_to_session_db", _boom)

    with caplog.at_level("WARNING", logger="agent.conversation_loop"):
        result = agent.run_conversation("what is the weather", conversation_history=[], task_id="t")

    assert len(_message_calls(handler)) == 2
    assert result["completed"] is True
    assert "Survived a bad flush." in (result["final_response"] or "")

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("pause_turn persistence failed" in msg for msg in warnings), warnings
    assert any("session db unavailable" in msg for msg in warnings), warnings


def test_repeated_pause_turn_without_fallback_ends_as_partial(agent_env, monkeypatch):
    """The continuation budget is bounded and the turn fails loudly, not silently."""
    agent, handler = agent_env
    for _ in range(6):
        handler.response_queue.append(_pause_resp())

    monkeypatch.setattr(agent, "_has_pending_fallback", lambda *_a, **_k: False)
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda *_a, **_k: False)

    result = agent.run_conversation("what is the weather", conversation_history=[], task_id="t")

    # Bounded: it must not burn the whole max_iterations budget on pauses.
    assert len(_message_calls(handler)) == 3
    assert result["completed"] is False
    assert result["partial"] is True
    assert "pause_turn" in (result["error"] or "")


def test_repeated_pause_turn_activates_fallback_when_available(agent_env, monkeypatch):
    """Hitting the bound hands the turn to the fallback chain before giving up."""
    agent, handler = agent_env
    for _ in range(3):
        handler.response_queue.append(_pause_resp())
    handler.response_queue.append(_text_resp("Fallback answered."))

    activated = {"n": 0}

    def _activate(*_args, **_kwargs):
        activated["n"] += 1
        return True

    monkeypatch.setattr(agent, "_has_pending_fallback", lambda *_a, **_k: True)
    monkeypatch.setattr(agent, "_try_activate_fallback", _activate)

    result = agent.run_conversation("what is the weather", conversation_history=[], task_id="t")

    assert activated["n"] == 1
    assert result["completed"] is True
    assert "Fallback answered." in (result["final_response"] or "")


def test_pause_budget_is_per_pause_run_not_cumulative(agent_env, monkeypatch):
    """Progress clears the budget: pauses must not accumulate across a whole turn.

    The bound exists to catch an upstream stuck in a pause loop, not to cap how
    many times a long turn may legitimately pause. Two pauses, then real
    progress, then two more pauses is five API calls and four pauses — well over
    the limit of 3 cumulatively, but never three *consecutively*. Without the
    per-run reset this turn would die as a false positive.
    """
    agent, handler = agent_env
    handler.response_queue.extend([
        _pause_resp(),
        _pause_resp(),
        _tool_use_resp(),   # progress: breaks the pause run
        _pause_resp(),
        _pause_resp(),
        _text_resp("Finished after four pauses."),
    ])

    monkeypatch.setattr(agent, "_has_pending_fallback", lambda *_a, **_k: False)
    monkeypatch.setattr(agent, "_try_activate_fallback", lambda *_a, **_k: False)

    result = agent.run_conversation("q1", conversation_history=[], task_id="t1")

    assert len(_message_calls(handler)) == 6
    assert result["completed"] is True
    assert "Finished after four pauses." in (result["final_response"] or "")
