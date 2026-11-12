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

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append(body)
                spec = provider.script.pop(0)
                # A safety refusal that arrives as an HTTP error (400 / provider APIError) rather than
                # an HTTP-200 finish_reason: return the error body with the flagged text so the agent
                # classifies it as content_policy_blocked on the exception path.
                if spec.get("status") and spec["status"] != 200:
                    payload = json.dumps({"error": {
                        "message": spec.get("error", "request flagged for possible cybersecurity risk"),
                        "type": "invalid_request_error", "code": None}}).encode()
                    self.send_response(spec["status"])
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
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
        self.base_url = "http://api.test.invalid/v1"
        self.proxy_url = f"http://127.0.0.1:{self._server.server_port}"

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
    provider = _LoopbackProvider()
    # This fake OpenAI-compatible server is not a local model route. Route the
    # reserved .invalid hostname through its explicit loopback proxy; if the
    # proxy is ignored, DNS fails instead of sending a test prompt externally.
    monkeypatch.setenv("HTTP_PROXY", provider.proxy_url)
    monkeypatch.setenv("http_proxy", provider.proxy_url)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,::1")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost,::1")

    # This test proxy only transports the remote-provider fixture to its loopback
    # test server; it does not serve a local model. Proxy-to-local-model routing
    # is covered by the admission classifier tests.
    from hermes_cli import local_gpu_admission_middleware
    import hermes_cli.middleware as middleware_facade
    classify_route = local_gpu_admission_middleware._is_local_llm_route

    def classify_test_route(provider_name, endpoint):
        if "api.test.invalid" in str(endpoint):
            return False
        return classify_route(provider_name, endpoint)

    monkeypatch.setattr(local_gpu_admission_middleware, "_is_local_llm_route", classify_test_route)
    monkeypatch.setattr(middleware_facade, "_is_local_llm_route", classify_test_route)

    import acp_adapter.session as acp_session
    import hermes_cli.config as cli_config
    from hermes_cli import mcp_startup
    from hermes_cli import runtime_provider

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


def test_acp_refusal_never_poisons_the_next_prompt(acp):
    """Turn 1: HTTP-200 ``content_filter`` refusal. Turn 2: unrelated request.

    Invariant: a safety-filter refusal is deterministic for the unchanged conversation (the filter
    judges the whole thread), so the refused turn is dropped from the live history and never rides
    into the next prompt. Otherwise every later message re-refuses and the chat locks up (the
    "Hermes Bot stuck sessions" report). The refusal still reaches the client live; transient
    failures keep their close-and-replay path (the server_error / rate_limit / timeout tests below),
    which this does not touch.
    """
    provider, prompt, _conversation_rows, _db, sid, conn, server = acp

    # An earlier accepted exchange, then the refused turn.
    provider.script = [{"finish_reason": "stop", "content": "Hello there."}]
    prompt("hello")

    provider.script = [{"finish_reason": "content_filter", "content": _REFUSAL_DETAIL}]
    prompt(_REFUSED)

    # The refusal still reaches the ACP client live, but never becomes canonical history.
    texts = [getattr(getattr(u, "content", None), "text", None) or getattr(u, "text", None) for u in conn.updates]
    assert any(_REFUSAL_DETAIL in (t or "") for t in texts)

    provider.script = [{"finish_reason": "stop", "content": "Paris."}]
    prompt(_NEW_REQUEST)

    # The next prompt reaches the provider with the prior accepted turn but WITHOUT the refused
    # request — the deterministic filter is never handed the thread that tripped it again.
    sent = [m for m in provider.requests[-1]["messages"] if m["role"] != "system"]
    assert [m["content"] for m in sent if m["role"] == "user"] == ["hello", _NEW_REQUEST]
    assert _REFUSED not in {m.get("content") for m in sent}

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


def test_acp_refusal_is_erased_durably_and_survives_a_reload(acp):
    """Finding #3: the refused turn must leave the DURABLE transcript too, not only live memory.

    Standalone ACP owns its ``SessionDB``, so the agent flushed the refused turn's user row before
    the refusal was known. If only the in-memory history is cleaned, a fresh session load (process
    restart / reopening the chat) restores the refused request and the chat re-locks. The owner
    erases the whole refused turn durably, so the reload is clean. The user still SAW the refusal
    live (asserted in the test above); here we assert the model-facing transcript no longer carries
    it.
    """
    provider, prompt, conversation_rows, db, sid, _conn, _server = acp

    provider.script = [{"finish_reason": "stop", "content": "Hello there."}]
    prompt("hello")
    provider.script = [{"finish_reason": "content_filter", "content": _REFUSAL_DETAIL}]
    prompt(_REFUSED)

    # The durable transcript holds only the prior accepted exchange — the refused turn is erased.
    assert conversation_rows() == [("user", "hello"), ("assistant", "Hello there.")]
    assert _REFUSED not in {content for _role, content in conversation_rows()}

    # A fresh SessionManager (reopen / restart) loads the clean transcript, never the refused ask.
    from acp_adapter.session import SessionManager

    reloaded = SessionManager(db=db).get_session(sid)
    assert reloaded is not None
    assert _REFUSED not in {m.get("content") for m in reloaded.history}
    assert reloaded.history[-1].get("content") == "Hello there."


def test_acp_exception_shaped_refusal_is_also_erased(acp):
    """Finding M1: some providers return a safety refusal as an HTTP 400 / APIError, not an HTTP-200
    ``finish_reason``. That exception path must recover identically — not persist the refused turn
    and hand back the pre-refusal history — or the exception shape re-poisons the next prompt while
    the user is told the message was left out.
    """
    provider, prompt, conversation_rows, _db, _sid, _conn, _server = acp

    provider.script = [{"finish_reason": "stop", "content": "Hello there."}]
    prompt("hello")
    # No fallback configured, so the 400 is terminal and classified content_policy_blocked.
    provider.script = [{"status": 400, "error": "request flagged for possible cybersecurity risk"}]
    prompt(_REFUSED)

    # Durable transcript: only the prior accepted exchange; the refused ask is gone.
    assert conversation_rows() == [("user", "hello"), ("assistant", "Hello there.")]

    # Next prompt reaches the provider without the refused request.
    provider.script = [{"finish_reason": "stop", "content": "Paris."}]
    prompt(_NEW_REQUEST)
    sent = [m for m in provider.requests[-1]["messages"] if m["role"] != "system"]
    assert [m["content"] for m in sent if m["role"] == "user"] == ["hello", _NEW_REQUEST]
    assert _REFUSED not in {m.get("content") for m in sent}


def test_acp_refused_turn_stays_in_the_display_tagged_content_refused(acp):
    """The refused ask leaves the MODEL transcript (active-only) but is RETAINED in the durable
    DISPLAY read, tagged content_refused — so a fresh ACP session load (reopen/restart) still shows
    the greyed "you asked X → refused" block even though the model never sees it again."""
    from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY

    provider, prompt, conversation_rows, db, sid, _conn, _server = acp

    provider.script = [{"finish_reason": "stop", "content": "Hello there."}]
    prompt("hello")
    provider.script = [{"finish_reason": "content_filter", "content": _REFUSAL_DETAIL}]
    prompt(_REFUSED)

    # MODEL read (active-only): the refused ask is gone.
    assert _REFUSED not in {content for _role, content in conversation_rows()}

    # DISPLAY read (include_compacted): the refused ask is kept, tagged, with the stored notice.
    display = [m for m in db.get_messages_as_conversation(sid, include_compacted=True)
               if m["role"] != "session_meta"]
    refused_row = display[-1]
    assert refused_row["content"] == _REFUSED
    assert refused_row["display_kind"] == CONTENT_REFUSED_DISPLAY_KIND
    assert CONTENT_REFUSED_METADATA_KEY in (refused_row.get("display_metadata") or {})


def test_acp_reload_replays_the_refused_ask_and_notice_to_the_editor(acp):
    """Finding #4486245780: reopening the editor must still SHOW the refused turn.

    The refused turn leaves the MODEL context (``state.history`` is active-only) but is RETAINED in
    the display projection. A reopened editor replays history via ``load_session``; if that replays
    only the clean model history the user's refused ask and notice vanish on every reopen even though
    SQLite keeps them. This drives a REAL ``load_session`` through a FRESH server (process-restart
    fidelity) sharing the same DB and asserts the replayed ``session/update`` stream — not just the
    DB rows — carries the refused ask plus the refusal notice, and still the prior clean exchange.
    """
    from agent.turn_failure_copy import CONTENT_REFUSED_NOTICE
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionManager

    provider, prompt, _rows, db, sid, _conn, server = acp

    provider.script = [{"finish_reason": "stop", "content": "Hello there."}]
    prompt("hello")
    provider.script = [{"finish_reason": "content_filter", "content": _REFUSAL_DETAIL}]
    prompt(_REFUSED)

    cwd = server.session_manager.get_session(sid).cwd

    # Reopen from a fresh process: a new server + manager on the same DB cold-restores the session,
    # then replays it to the editor.
    reopened_conn = _RecordingConn()
    reopened = HermesACPAgent(session_manager=SessionManager(db=db))
    reopened.on_connect(reopened_conn)
    resp = asyncio.run(reopened.load_session(cwd=cwd, session_id=sid))
    assert resp is not None

    texts = [getattr(getattr(u, "content", None), "text", None) or getattr(u, "text", None)
             for u in reopened_conn.updates]
    # The refused ask is shown again (model history alone would hide it).
    assert any(_REFUSED in (t or "") for t in texts), f"refused ask hidden on reload: {texts}"
    # The refusal notice rides along so the editor shows "you asked X -> refused".
    assert any(CONTENT_REFUSED_NOTICE in (t or "") for t in texts), f"refusal notice missing: {texts}"
    # The prior accepted exchange is still replayed.
    assert any((t or "") == "hello" for t in texts)
    assert any("Hello there." in (t or "") for t in texts)


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


def test_prompt_after_cancel_keeps_the_unanswered_request(acp):
    """An unanswered request survives the next ACP prompt and its replay prefix."""
    provider, prompt, _rows, _db, sid, _conn, server = acp
    state = server.session_manager.get_session(sid)
    before, real = list(state.history), state.agent.run_conversation
    state.agent.run_conversation = lambda **kw: {
        "final_response": None, "interrupted": True, "completed": False,
        "messages": before + [{"role": "user", "content": "deploy build 42"}],
    }
    try:
        prompt("deploy build 42")
    finally:
        state.agent.run_conversation = real

    provider.script = [
        {"finish_reason": "stop", "content": "done"},
        {"finish_reason": "stop", "content": "ok"},
    ]
    prompt("also run the smoke tests")
    prompt("third")
    turn2 = provider.requests[-2]["messages"]
    turn3 = provider.requests[-1]["messages"]
    assert turn3[:len(turn2)] == turn2
    assert any(
        msg["role"] == "user" and msg["content"] == "deploy build 42\n\nalso run the smoke tests"
        for msg in turn2
    )


def test_failed_turn_boundary_keeps_the_error_card_for_rehydration(tmp_path, monkeypatch):
    """The boundary row carries the failure's error text and ``error_surface`` so a client
    reopening the session after a restart redraws the error card, not only the notice."""
    from types import SimpleNamespace

    from agent.conversation_loop import _close_durable_failed_turn
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    sid = "s1"
    db.create_session(session_id=sid, source="acp", model="m")
    db.append_message(sid, "user", "say hello")

    def flush(messages):
        db.append_message(sid, messages[-1]["role"], messages[-1]["content"],
                          display_metadata=messages[-1].get("display_metadata"))

    agent = SimpleNamespace(_session_db=db, session_id=sid, _flush_messages_to_session_db=flush,
                            provider="opencode-go", model="deepseek-v4.1-flash")
    messages = [{"role": "user", "content": "say hello"}]
    _close_durable_failed_turn(agent, {"completed": False, "failed": True, "failure_reason": "timeout",
                                       "error": "Connection error.", "messages": messages})

    metadata = messages[-1]["display_metadata"]
    assert metadata["error"] == "Connection error."
    assert metadata["error_surface"]["code"] == "timeout"
    assert metadata["error_surface"]["provider"] == "opencode-go"
    stored = db.get_messages(sid)[-1]["display_metadata"]
    assert (stored if isinstance(stored, dict) else json.loads(stored))["error_surface"]["code"] == "timeout"
    db.close()


def test_failed_turn_after_a_tool_row_is_closed_with_its_error_card(tmp_path, monkeypatch):
    """A turn that fails after a tool already ran (an onboarding beat answered, then the provider
    429s) leaves a ``tool`` row as the durable tail. It is as open as a bare user tail: without the
    boundary a client reopening the session has no error card to draw for the turn."""
    from types import SimpleNamespace

    from agent.conversation_loop import _close_durable_failed_turn
    from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, PARTIAL_FAILED_TURN_NOTICE
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(tmp_path / "state.db")
    sid = "s1"
    db.create_session(session_id=sid, source="tui", model="m")
    call = {"id": "call_setup_beat_name", "type": "function",
            "function": {"name": "setup_beat", "arguments": '{"beat":"name"}'}}
    messages = [
        {"role": "user", "content": "opening"},
        {"role": "assistant", "content": "Hi!", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": '{"answer":{"name":"Austin"}}'},
    ]
    db.append_message(sid, "user", "opening")
    db.append_message(sid, "assistant", "Hi!", tool_calls=[call])
    db.append_message(sid, "tool", messages[-1]["content"], tool_call_id=call["id"])

    def flush(rows):
        last = rows[-1]
        db.append_message(sid, last["role"], last["content"], display_kind=last.get("display_kind"),
                          display_metadata=last.get("display_metadata"))

    agent = SimpleNamespace(_session_db=db, session_id=sid, _flush_messages_to_session_db=flush,
                            provider="nous", model="m")
    failed = {"completed": False, "failed": True, "failure_reason": "rate_limit",
              "error": "429 Too Many Requests", "messages": messages, "current_turn_user_idx": 0}
    _close_durable_failed_turn(agent, failed)
    _close_durable_failed_turn(agent, failed)  # redelivery: the tail is now the boundary

    stored = db.get_messages(sid)
    assert [row["role"] for row in stored] == ["user", "assistant", "tool", "assistant"]
    boundary = stored[-1]
    assert (boundary["content"], boundary["display_kind"]) == (PARTIAL_FAILED_TURN_NOTICE, FAILED_TURN_DISPLAY_KIND)
    metadata = boundary["display_metadata"]
    assert (metadata if isinstance(metadata, dict) else json.loads(metadata))["error_surface"]["code"] == "rate_limit"
    db.close()
