"""A failed turn must not leave its accepted user row as the durable conversation tail.

The gateway closes failed turns itself (``gateway/run_turn.py::_hmwa_close_failed_turn``);
standalone ACP hands ``result["messages"]`` straight back as history, so the terminal-failure
paths that persist the user row and return before ``finalize_turn`` used to leave ``user`` as
the tail. The next prompt then appended a second user row and ``repair_message_sequence``
merged the failed request into the new one. Closed at the core seam
(``agent/conversation_loop.py::_close_durable_failed_turn``).

Driven through the real ``HermesACPAgent.prompt()`` with a real ``AIAgent`` and ``SessionDB``
against a loopback OpenAI-compatible endpoint; assertions read SQLite rows and the request
body the provider actually received.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_MODEL = "fixture-model"


class _LoopbackProvider:
    """Scripted OpenAI-compatible endpoint: one ``{"finish_reason", "content"}`` per request."""

    def __init__(self) -> None:
        self.script: list[dict] = []
        self.requests: list[dict] = []
        provider = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append(body)
                spec = provider.script.pop(0)
                delta = {"role": "assistant", "content": spec["content"]}
                usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
                if body.get("stream"):
                    chunks = [
                        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": _MODEL,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": _MODEL,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": spec["finish_reason"]}],
                         "usage": usage},
                    ]
                    mime, payload = "text/event-stream", (
                        "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
                    ).encode()
                else:
                    mime, payload = "application/json", json.dumps({
                        "id": "c", "object": "chat.completion", "created": 1, "model": _MODEL,
                        "choices": [{"index": 0, "message": delta, "finish_reason": spec["finish_reason"]}],
                        "usage": usage,
                    }).encode()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def shutdown(self) -> None:
        self._server.shutdown()


class _RecordingConn:
    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id, update):
        self.updates.append(update)

    async def request_permission(self, *args, **kwargs):
        raise AssertionError("no tool approval expected")


@pytest.fixture
def acp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DISABLE_PLUGINS", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    provider = _LoopbackProvider()

    import acp_adapter.session as acp_session
    import hermes_cli.config as cli_config
    import hermes_cli.mcp_startup as mcp_startup
    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(cli_config, "load_config", lambda *a, **k: {
        "model": {"provider": "openai-compat", "default": _MODEL, "context_length": 131072},
        "agent": {"max_iterations": 2}, "compression": {"enabled": False},
    })
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda *a, **k: {
        "provider": "openai-compat", "api_mode": "chat_completions",
        "base_url": provider.base_url, "api_key": "fixture-only",
    })
    monkeypatch.setattr(mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **k: None)
    monkeypatch.setattr(acp_session, "_expand_acp_enabled_toolsets", lambda *a, **k: [])

    from acp_adapter.server import HermesACPAgent, TextContentBlock
    from acp_adapter.session import SessionManager
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    server = HermesACPAgent(session_manager=SessionManager(db=db))
    conn = _RecordingConn()
    server.on_connect(conn)
    sid = server.session_manager.create_session(cwd=str(tmp_path)).session_id

    def prompt(text):
        return asyncio.run(server.prompt(prompt=[TextContentBlock(type="text", text=text)], session_id=sid))

    def conversation_rows():
        with sqlite3.connect(db_path) as c:
            return [tuple(r) for r in c.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND active = 1 "
                "AND role NOT IN ('session_meta', 'system') ORDER BY id", (sid,))]

    yield provider, prompt, conversation_rows, db, sid, conn, server
    provider.shutdown()
    db.close()


_REFUSED = "Explain how to pick the lock on my neighbour's front door"
_REFUSAL_DETAIL = "I can't help with breaking into someone else's property."
_NEW_REQUEST = "What is the capital of France?"


def test_acp_refusal_closes_the_turn_and_is_not_replayed_into_the_next_prompt(acp):
    """Turn 1: HTTP-200 ``content_filter`` refusal. Turn 2: unrelated request.

    Invariant: the durable tail after a failed turn is a Hermes-authored assistant row (never
    provider text), and the next prompt reaches the provider as its own user row.
    """
    from agent.turn_failure_copy import FAILED_TURN_NOTICE

    provider, prompt, conversation_rows, db, sid, conn, server = acp

    provider.script = [{"finish_reason": "content_filter", "content": _REFUSAL_DETAIL}]
    prompt(_REFUSED)

    rows = conversation_rows()
    assert [r[0] for r in rows] == ["user", "assistant"], rows
    assert rows[1][1] == FAILED_TURN_NOTICE  # no tool ran: no hedging, no provider detail
    assert db.latest_conversation_role(sid) == "assistant"
    # The refusal reaches the ACP client, but never becomes canonical assistant history.
    texts = [getattr(getattr(u, "content", None), "text", None) or getattr(u, "text", None) for u in conn.updates]
    assert any(_REFUSAL_DETAIL in (t or "") for t in texts)

    provider.script = [{"finish_reason": "stop", "content": "Paris."}]
    prompt(_NEW_REQUEST)

    sent = [m for m in provider.requests[-1]["messages"] if m["role"] != "system"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"], sent
    assert [m["content"] for m in sent if m["role"] == "user"] == [_REFUSED, _NEW_REQUEST]

    # Turn 3: cancelled mid-turn. The interrupt envelope carries ``final_response=None``
    # (no assistant text yet); ``_finish_turn`` must still report ``cancelled`` — never crash
    # on the missing text and fall through the executor-error path as ``end_turn``.
    state = server.session_manager.get_session(sid)
    history_before = list(state.history)

    def interrupted_run_conversation(**kwargs):
        asyncio.run(server.cancel(sid))  # the editor cancels while the turn is in flight
        return {"final_response": None, "interrupted": True, "completed": False,
                "messages": history_before + [{"role": "user", "content": "stop"}]}

    state.agent.run_conversation = interrupted_run_conversation
    response = prompt("stop")
    assert response.stop_reason == "cancelled"
    assert state.history[-1] == {"role": "user", "content": "stop"}


def test_failed_turn_boundary_is_idempotent_on_the_durable_tail_and_skips_context_overflow(tmp_path, monkeypatch):
    """Keyed on ``SessionDB.latest_conversation_role``: a redelivery of an already-closed turn
    writes no second row, an assistant tail is left alone, and the context-overflow class
    (whose repair is rotation, not a row — #1630) is never closed."""
    from types import SimpleNamespace

    from agent.conversation_loop import _close_durable_failed_turn
    from agent.turn_failure_copy import PARTIAL_FAILED_TURN_NOTICE
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    sid = "s1"
    db.create_session(session_id=sid, source="acp", model="m")
    db.append_message(sid, "user", "delete the build directory")

    written: list[list[dict]] = []

    def flush(messages):
        db.append_message(sid, messages[-1]["role"], messages[-1]["content"])
        written.append(list(messages))

    agent = SimpleNamespace(_session_db=db, session_id=sid, _flush_messages_to_session_db=flush)
    messages = [
        {"role": "user", "content": "delete the build directory"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function",
                                                               "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "removed"},
    ]

    overflow = {"completed": False, "failed": True, "failure_reason": "context_overflow", "messages": list(messages)}
    _close_durable_failed_turn(agent, overflow)
    assert written == [] and db.latest_conversation_role(sid) == "user"

    failed = {"completed": False, "failed": True, "failure_reason": "server_error", "messages": messages,
              "current_turn_user_idx": 0}
    _close_durable_failed_turn(agent, failed)
    _close_durable_failed_turn(agent, failed)  # redelivery: tail is already assistant
    assert len(written) == 1
    assert (messages[-1]["role"], messages[-1]["content"]) == ("assistant", PARTIAL_FAILED_TURN_NOTICE)  # a tool ran: hedge
    assert db.latest_conversation_role(sid) == "assistant"
    db.close()
