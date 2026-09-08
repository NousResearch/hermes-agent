"""Tests for the ``api_content`` sidecar ("persist what you send").

The first LLM call of every turn used to miss the provider prompt cache
because the bytes sent to the API diverged from the bytes replayed from the
persisted transcript: memory-prefetch / plugin context is injected into the
API copy of the current turn's user message only, and the persist
user-message override (#48677) writes cleaned content to the DB row. The fix
persists the EXACT sent content in a nullable ``messages.api_content`` column
and replays it verbatim (no sanitize, no strip).

Covers: SessionDB round-trip and auto-migration, the shared composition
helper, prologue stamping order, the flush-override sidecar, and the
end-to-end wire invariant (turn N+1 replays turn N's bytes) against an
in-process mock provider.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import types
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_manager import build_memory_context_block
from agent.turn_context import (
    VOLATILE_USER_CONTEXT_REPLAY_ID_KEY,
    _volatile_base_content_fingerprint,
    bind_volatile_user_context,
    build_api_messages,
    build_turn_context,
    copy_volatile_user_context_history,
    compose_user_api_content,
)
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# compose_user_api_content — the single source of the injection composition
# ---------------------------------------------------------------------------

class TestComposeUserApiContent:
    def test_none_when_nothing_to_inject(self):
        assert compose_user_api_content("hello", "", "") is None


    def test_composes_memory_block_and_plugin_context(self):
        out = compose_user_api_content("hello", "likes tea", "PLUGIN-CTX")
        fenced = build_memory_context_block("likes tea")
        assert out == "hello" + "\n\n" + fenced + "\n\n" + "PLUGIN-CTX"




# ---------------------------------------------------------------------------
# SessionDB: schema, round-trip, verbatim replay
# ---------------------------------------------------------------------------

class TestSessionDbSidecar:
    def _open(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        return db

    def test_round_trip_is_verbatim_not_sanitized(self, tmp_path):
        """api_content must bypass sanitize_context/strip on load — clean
        content loses the <memory-context> block and outer whitespace, the
        sidecar must not."""
        db = self._open(tmp_path)
        sent = "  hello\n\n<memory-context>\nrecalled\n</memory-context>\n"
        try:
            db.append_message("s1", "user", content="hello", api_content=sent)
            msgs = db.get_messages_as_conversation("s1")
            assert msgs[0]["content"] == "hello"
            assert msgs[0]["api_content"] == sent  # byte-for-byte
        finally:
            db.close()


    def test_get_messages_exposes_column(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="hello", api_content="hello+ctx")
            rows = db.get_messages("s1")
            assert rows[0]["api_content"] == "hello+ctx"
        finally:
            db.close()



class TestAutoMigration:
    def test_reconciliation_adds_api_content_column(self, tmp_path):
        """Opening a pre-sidecar DB adds the column declaratively (the
        SCHEMA_SQL diff in _reconcile_columns), no version-gated migration."""
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                parent_session_id TEXT,
                started_at REAL NOT NULL
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT
            );
        """)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "user", "old row", 1000.0),
        )
        conn.commit()
        conn.close()

        db = SessionDB(db_path=db_path)
        try:
            cols = {
                row[1]
                for row in db._conn.execute('PRAGMA table_info("messages")').fetchall()
            }
            assert "api_content" in cols
            # Old rows read back NULL (no key); new writes round-trip.
            db.append_message("s1", "user", content="new", api_content="new+ctx")
            msgs = db.get_messages_as_conversation("s1")
            assert "api_content" not in msgs[0]
            assert msgs[1]["api_content"] == "new+ctx"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Prologue stamping (build_turn_context)
# ---------------------------------------------------------------------------

class _FakeTodoStore:
    def has_items(self):
        return True


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches
    (mirrors tests/agent/test_turn_context.py)."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self._skip_mcp_refresh = True
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Captures the user message's api_content at persist time, proving
        # the stamp lands BEFORE the early persist writes the row.
        self.api_content_at_persist = "<unset>"

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, _history=None):
        self.api_content_at_persist = messages[-1].get("api_content")


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


class TestPrologueStamping:
    def test_stamps_api_content_from_plugin_context(self):
        agent = _FakeAgent()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent)
        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["content"] == "hello"  # clean content untouched
        assert msg["api_content"] == compose_user_api_content(
            "hello", ctx.ext_prefetch_cache, ctx.plugin_user_context
        )
        assert msg["api_content"] == "hello\n\nPLUGIN-CTX"
        # The early persist saw the stamped sidecar (written in one insert).
        assert agent.api_content_at_persist == "hello\n\nPLUGIN-CTX"

    def test_no_stamp_without_injections(self):
        agent = _FakeAgent()
        with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
            ctx = _build(agent)
        assert "api_content" not in ctx.messages[ctx.current_turn_user_idx]
        assert agent.api_content_at_persist is None

    def test_no_stamp_for_codex_app_server(self):
        """codex_app_server turns bypass the api_messages build, so the
        injected bytes are never sent — stamping would persist a lie."""
        agent = _FakeAgent()
        agent.api_mode = "codex_app_server"
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent)
        assert "api_content" not in ctx.messages[ctx.current_turn_user_idx]


# ---------------------------------------------------------------------------
# Flush: persist-override rows keep the sent bytes in the sidecar (#48677)
# ---------------------------------------------------------------------------

class TestFlushOverrideSidecar:
    def _make_agent(self, db, sid):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        return agent

    def test_override_moves_sent_bytes_to_sidecar(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-ov"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            live = "[gateway note] observed context\n\nactual question"
            messages = [{"role": "user", "content": live}]
            agent._persist_user_message_idx = 0
            agent._persist_user_message_override = "actual question"
            agent._persist_user_message_timestamp = None
            agent._flush_messages_to_session_db(messages, None)

            msgs = db.get_messages_as_conversation(sid)
            assert msgs[0]["content"] == "actual question"
            assert msgs[0]["api_content"] == live
            # The live dict is never mutated by the flush.
            assert messages[0]["content"] == live
        finally:
            db.close()

    def test_stamped_sidecar_wins_over_override_derivation(self, tmp_path):
        """When the prologue already stamped api_content (injections), the
        flush must keep those bytes — they are what actually went out."""
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-ov2"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            messages = [
                {
                    "role": "user",
                    "content": "live text",
                    "api_content": "live text\n\nPLUGIN-CTX",
                }
            ]
            agent._persist_user_message_idx = 0
            agent._persist_user_message_override = "clean text"
            agent._persist_user_message_timestamp = None
            agent._flush_messages_to_session_db(messages, None)

            msgs = db.get_messages_as_conversation(sid)
            assert msgs[0]["content"] == "clean text"
            assert msgs[0]["api_content"] == "live text\n\nPLUGIN-CTX"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# End-to-end wire invariant against an in-process mock provider
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
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

    def log_message(self, *a, **kw):
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
def wire_env():
    """Mock provider + isolated HERMES_HOME + a shared SessionDB.

    Yields (make_agent, handler, db, sid): ``make_agent()`` builds a fresh
    AIAgent bound to the shared DB/session, so a second call models a
    process-restart turn N+1 that reloads history from the store.
    """
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_api_content_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    from run_agent import AIAgent

    from pathlib import Path

    db = SessionDB(db_path=Path(test_home) / "state.db")
    sid = "sess-wire"

    def make_agent():
        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
            provider="openai-compat", model="test-model",
            max_iterations=10, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="cli",
            session_db=db, session_id=sid,
        )
        agent.valid_tool_names = {"read_file"}
        return agent

    try:
        with patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=lambda hook, **kw: (
                [{"context": "PLUGIN-CTX"}] if hook == "pre_llm_call" else []
            ),
        ):
            yield make_agent, _MockHandler, db, sid
    finally:
        srv.shutdown()
        db.close()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _chat_requests(handler) -> list:
    # The model context-length probe also hits the mock; keep only
    # chat-completions payloads.
    return [r for r in handler.captured_requests if "messages" in r]


def _user_messages(req: dict) -> list:
    return [m for m in req.get("messages", []) if m.get("role") == "user"]


class TestWireInvariant:
    def test_injection_sent_stamped_and_stable_within_turn(self, wire_env):
        """The current turn's user message goes out with the injected context,
        the sidecar equals the sent bytes exactly, the field never reaches the
        wire, and every pass within the turn sends identical bytes."""
        make_agent, handler, db, sid = wire_env
        agent = make_agent()
        # Two API calls in one turn: tool call, then final text.
        handler.response_queue.append(_tc_resp("read_file", '{"file_path": "/nonexistent-path"}'))
        handler.response_queue.append(_text_resp("done"))

        with patch("agent.turn_context._volatile_base_content_fingerprint") as fingerprint:
            agent.run_conversation("hello please", conversation_history=[], task_id="t")
        fingerprint.assert_not_called()

        reqs = _chat_requests(handler)
        assert len(reqs) == 2
        sent_1 = _user_messages(reqs[0])[0]["content"]
        sent_2 = _user_messages(reqs[1])[0]["content"]
        assert sent_1 == "hello please\n\nPLUGIN-CTX"
        assert sent_2 == sent_1  # repeated builds: identical bytes

        # The sidecar never reaches the provider.
        for req in reqs:
            for m in req.get("messages", []):
                assert "api_content" not in m

        # Persisted row: clean content + exact sent bytes in the sidecar.
        user_rows = [r for r in db.get_messages(sid) if r["role"] == "user"]
        assert user_rows[0]["content"] == "hello please"
        assert user_rows[0]["api_content"] == sent_1

    def test_next_turn_replays_previous_turn_bytes(self, wire_env):
        """The cache invariant: the serialized user message replayed in turn
        N+1 (history reloaded from the store) EQUALS the bytes turn N sent."""
        make_agent, handler, db, sid = wire_env

        # ── Turn N ──
        agent1 = make_agent()
        agent1.run_conversation("hello please", conversation_history=[], task_id="t1")
        turn_n_user = _user_messages(_chat_requests(handler)[0])[0]
        turn_n_bytes = json.dumps(turn_n_user, sort_keys=True)

        # ── Turn N+1: fresh agent, history reloaded from the store ──
        history = db.get_messages_as_conversation(sid)
        # The stored history carries the sidecar, not the injected content.
        assert history[0]["content"] == "hello please"
        assert history[0]["api_content"] == turn_n_user["content"]

        handler.captured_requests = []
        agent2 = make_agent()
        agent2.run_conversation(
            "second question", conversation_history=history, task_id="t2"
        )

        replayed = _user_messages(_chat_requests(handler)[0])[0]
        assert json.dumps(replayed, sort_keys=True) == turn_n_bytes

        # And the new current-turn message got its own injection + sidecar.
        current = _user_messages(_chat_requests(handler)[0])[-1]
        assert current["content"] == "second question\n\nPLUGIN-CTX"

    def test_volatile_context_replays_from_ram_without_entering_storage(self, wire_env):
        """A private turn snapshot remains byte-stable while only RAM retains it."""
        make_agent, handler, db, sid = wire_env
        agent = make_agent()
        snapshot = (
            "[Background Telegram location context]\n"
            "Latitude: 37.7749\nLongitude: -122.4194"
        )

        with bind_volatile_user_context(agent, snapshot, "tg-1"):
            first_result = agent.run_conversation(
                "where am I?",
                conversation_history=[],
                task_id="t1",
                persist_user_platform_id="tg-1",
            )
        first_wire = _user_messages(_chat_requests(handler)[0])[0]
        assert first_wire["content"] == f"where am I?\n\nPLUGIN-CTX\n\n{snapshot}"
        assert "37.7749" not in json.dumps(first_result["messages"])

        history = db.get_messages_as_conversation(sid)
        durable = json.dumps(history)
        assert "37.7749" not in durable
        assert history[0]["message_id"] == "tg-1"

        handler.captured_requests = []
        with bind_volatile_user_context(agent, None, "tg-2"):
            agent.run_conversation(
                "what should I bring?",
                conversation_history=history,
                task_id="t2",
                persist_user_platform_id="tg-2",
            )
        replayed = _user_messages(_chat_requests(handler)[0])[0]
        assert json.dumps(replayed, sort_keys=True) == json.dumps(first_wire, sort_keys=True)

        for request in _chat_requests(handler):
            for message in request.get("messages", []):
                assert "_volatile_user_context_replay_id" not in message

    def test_rewritten_row_does_not_inherit_volatile_context(self, wire_env):
        """Compression/rewind may change the prefix; an id alone cannot move coordinates."""
        make_agent, handler, db, sid = wire_env
        agent = make_agent()
        snapshot = "[Background Telegram location context]\nLatitude: 1.25\nLongitude: 2.5"
        with bind_volatile_user_context(agent, snapshot, "tg-rewritten"):
            agent.run_conversation(
                "original",
                conversation_history=[],
                task_id="t1",
                persist_user_platform_id="tg-rewritten",
            )

        history = db.get_messages_as_conversation(sid)
        history[0]["content"] = "[compressed summary]"
        history[0].pop("api_content", None)
        handler.captured_requests = []
        with bind_volatile_user_context(agent, None, "tg-2"):
            agent.run_conversation(
                "next",
                conversation_history=history,
                task_id="t2",
                persist_user_platform_id="tg-2",
            )
        replayed = _user_messages(_chat_requests(handler)[0])[0]["content"]
        assert replayed == "[compressed summary]"
        assert "Latitude" not in replayed

    def test_provider_error_summary_withholds_bound_volatile_context(self):
        from run_agent import AIAgent

        agent = types.SimpleNamespace()
        snapshot = (
            "[Background Telegram location context]\n"
            "Latitude: 37.7749\nLongitude: -122.4194"
        )
        error = RuntimeError(
            'provider echoed {"content":"Latitude: 37.7749\\nLongitude: -122.4194"}'
        )

        with bind_volatile_user_context(agent, snapshot, "tg-private"):
            summary = AIAgent._summarize_api_error(error)

        assert "37.7749" not in summary
        assert "-122.4194" not in summary
        assert summary == "Provider error details withheld for private-context turn"

    def test_provider_error_withholds_boundary_sliced_private_context(self):
        from run_agent import AIAgent

        agent = types.SimpleNamespace()
        snapshot = (
            "[Background Telegram location context]\n"
            "Latitude: 37.7749\nLongitude: -122.4194"
        )
        # A provider may truncate through the middle of a scalar, so exact literal
        # replacement is insufficient for any provider-controlled error detail.
        error = RuntimeError("x" * 498 + "37.77")

        with bind_volatile_user_context(agent, snapshot, "tg-private"):
            summary = AIAgent._summarize_api_error(error)
            cleaned = AIAgent._clean_error_message(agent, str(error))

        expected = "Provider error details withheld for private-context turn"
        assert summary == expected
        assert cleaned == expected

    def test_provider_retry_and_invalid_response_paths_withhold_remote_text(
        self, wire_env, caplog,
    ):
        from agent.turn_recovery import (
            compute_error_backoff,
            describe_invalid_response,
            validate_response_shape,
        )

        make_agent, _handler, _db, _sid = wire_env
        agent = make_agent()
        agent.api_mode = "codex_responses"
        agent._get_transport = lambda: types.SimpleNamespace(
            validate_response=lambda _response: False
        )
        response = types.SimpleNamespace(
            status="failed",
            error={"code": 400, "message": "provider slice 37.77 / -122.41"},
            model="remote-37.77",
        )
        error = RuntimeError("retry fragment 37.77 / -122.41")
        error.response = types.SimpleNamespace(headers={"Retry-After": "37.77"})
        error.body = {"error": {"retry_after": 122.41}}
        snapshot = "Latitude: 37.7749\nLongitude: -122.4194"

        with (
            bind_volatile_user_context(agent, snapshot, "tg-private"),
            patch("agent.retry_utils.jittered_backoff", return_value=0.25),
            caplog.at_level(logging.WARNING),
        ):
            invalid, details = validate_response_shape(agent, response)
            error_msg, provider_name, _ = describe_invalid_response(
                agent, response, 0.1
            )
            wait = compute_error_backoff(
                agent,
                error,
                retry_count=1,
                max_retries=3,
                is_rate_limited=False,
                is_zai_coding_overload=False,
                base_url=agent.base_url,
                model=agent.model,
            )

        expected = "Provider error details withheld for private-context turn"
        assert invalid is True
        assert details == [f"response.status=failed: {expected}"]
        assert error_msg == expected
        assert provider_name == agent.provider
        assert wait == 0.25
        assert expected in caplog.text
        assert "37.77" not in caplog.text
        assert "-122.41" not in caplog.text

    def test_provider_error_request_dump_is_skipped_for_volatile_turn(self, tmp_path):
        from agent.agent_runtime_helpers import dump_api_request_debug

        agent = types.SimpleNamespace(
            _vprint=MagicMock(),
            log_prefix="",
            logs_dir=tmp_path,
        )
        snapshot = "[Background Telegram location context]\nLatitude: 1.25\nLongitude: 2.5"

        with bind_volatile_user_context(agent, snapshot, "tg-private"):
            result = dump_api_request_debug(
                agent,
                {"messages": [{"role": "user", "content": snapshot}]},
                reason="provider_error",
            )

        assert result is None
        assert list(tmp_path.rglob("request_dump_*.json")) == []
        assert "skipped" in agent._vprint.call_args.args[0]

    def test_historical_ram_replay_also_blocks_dump_and_redacts_errors(
        self, wire_env, tmp_path,
    ):
        from agent.agent_runtime_helpers import dump_api_request_debug
        from run_agent import AIAgent

        make_agent, _handler, _db, _sid = wire_env
        agent = make_agent()
        snapshot = (
            "[Background Telegram location context]\n"
            "Latitude: 48.8566\nLongitude: 2.3522"
        )
        with bind_volatile_user_context(agent, snapshot, "tg-first"):
            agent.run_conversation(
                "where am I?",
                conversation_history=[],
                task_id="t1",
                persist_user_platform_id="tg-first",
            )

        # The next turn has no current snapshot, but its wire prefix can replay the
        # first turn's snapshot from RAM. Failure paths must keep treating it as private.
        agent.logs_dir = tmp_path
        with bind_volatile_user_context(agent, None, "tg-second"):
            result = dump_api_request_debug(
                agent,
                {"messages": [{"role": "user", "content": snapshot}]},
                reason="provider_error",
            )
            summary = AIAgent._summarize_api_error(
                RuntimeError(f"provider echoed {snapshot}")
            )

        assert result is None
        assert list(tmp_path.rglob("request_dump_*.json")) == []
        assert "48.8566" not in summary
        assert "2.3522" not in summary

    def test_volatile_history_cap_is_stable_across_tool_followup_calls(self, wire_env):
        make_agent, _handler, _db, _sid = wire_env
        agent = make_agent()
        messages = []
        retained = OrderedDict()
        for index in range(256):
            replay_id = f"tg-{index}"
            content = f"question {index}"
            messages.append(
                {
                    "role": "user",
                    "content": content,
                    VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: replay_id,
                }
            )
            retained[replay_id] = (
                f"Latitude: {index}.1\nLongitude: {index}.2",
                _volatile_base_content_fingerprint(content),
            )
        agent._volatile_user_context_history = retained
        previous, _ = build_api_messages(
            agent,
            messages,
            current_turn_user_idx=len(messages) - 1,
            ext_prefetch_cache="",
            plugin_user_context="",
            moa_config=None,
            active_system_prompt="",
        )
        messages.append(
            {
                "role": "user",
                "content": "current question",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-current",
            }
        )

        kwargs = dict(
            current_turn_user_idx=len(messages) - 1,
            ext_prefetch_cache="",
            plugin_user_context="",
            moa_config=None,
            active_system_prompt="",
        )
        with bind_volatile_user_context(
            agent,
            "Latitude: 999.1\nLongitude: 999.2",
            "tg-current",
        ):
            first, _ = build_api_messages(agent, messages, **kwargs)
            second, _ = build_api_messages(agent, messages, **kwargs)

        assert first == second
        # A bounded sidecar may decline this turn's new snapshot, but it must never
        # evict context for an active historical row and rewrite the cached prefix.
        assert first[:-1] == previous
        assert first[0]["content"].endswith("Latitude: 0.1\nLongitude: 0.2")
        assert first[-1]["content"] == "current question"
        assert len(agent._volatile_user_context_history) == 256

    def test_moa_excludes_current_and_historical_volatile_context(self, wire_env):
        make_agent, _handler, _db, _sid = wire_env
        agent = make_agent()
        old_context = "Latitude: 1.25\nLongitude: 2.5"
        current_context = "Latitude: 37.7749\nLongitude: -122.4194"
        messages = [
            {
                "role": "user",
                "content": "old question",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-old",
            },
            {"role": "assistant", "content": "old answer"},
            {
                "role": "user",
                "content": "current question",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-current",
            },
        ]
        agent._volatile_user_context_history = OrderedDict(
            {
                "tg-old": (
                    old_context,
                    _volatile_base_content_fingerprint("old question"),
                )
            }
        )

        with bind_volatile_user_context(
            agent, current_context, "tg-current"
        ):
            api_messages, _ = build_api_messages(
                agent,
                messages,
                current_turn_user_idx=2,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config={},
                active_system_prompt="",
            )

        serialized = json.dumps(api_messages)
        assert "Latitude" not in serialized
        assert "Longitude" not in serialized
        assert "37.7749" not in serialized
        assert "1.25" not in serialized


# ---------------------------------------------------------------------------
# Review fixes: re-anchoring, MoA, in-place compaction backfill, override
# guard, sanitize-divergence capture, max-iterations replay, replay cleanup
# ---------------------------------------------------------------------------

from agent.turn_context import reanchor_current_turn_user_idx


class TestReanchorCurrentTurnUserIdx:



    def test_minus_one_when_no_user_message(self):
        messages = [{"role": "assistant", "content": "a"}]
        assert reanchor_current_turn_user_idx(messages, "hello") == -1
        assert reanchor_current_turn_user_idx([], "hello") == -1

    def test_non_dict_entries_ignored(self):
        messages = ["junk", {"role": "user", "content": "hello"}, None]
        assert reanchor_current_turn_user_idx(messages, "hello") == 1


class TestPrologueMoaAndInPlaceBackfill:
    def test_no_stamp_for_moa_turns(self):
        """MoA appends per-call aggregated context to the API copy AFTER the
        composition — a stamped sidecar would persist bytes that never match
        the wire."""
        agent = _FakeAgent()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent, moa_active=True)
        assert "api_content" not in ctx.messages[ctx.current_turn_user_idx]

    def test_inplace_compaction_backfills_sidecar_into_db(self):
        """In-place preflight compaction inserts the current-turn user row
        BEFORE the stamp (archive_and_compact), and the crash persist
        identity-skips every compacted dict — the stamp must be pushed into
        the existing row directly."""
        agent = _FakeAgent()
        agent.compression_enabled = True
        agent._session_db = MagicMock()

        calls = {"n": 0}

        def _should_compress(_tokens):
            calls["n"] += 1
            return calls["n"] == 1  # compress once, then stop

        agent.context_compressor = types.SimpleNamespace(
            protect_first_n=0,
            protect_last_n=0,
            threshold_tokens=1,
            context_length=1000,
            last_prompt_tokens=-1,
            should_compress=_should_compress,
            should_defer_preflight_to_real_usage=lambda _t: False,
            get_active_compression_failure_cooldown=lambda: None,
        )

        def _compress(messages, _system, approx_tokens=None, task_id=None):
            # Emulate compress_context in in_place mode: archive_and_compact
            # already inserted these rows (api_content=NULL — the stamp has
            # not happened yet), fresh copies replace the live dicts.
            agent._last_compaction_in_place = True
            return (
                [
                    {"role": "assistant", "content": "compaction summary"},
                    dict(messages[-1]),  # surviving current-turn user copy
                ],
                "SYSTEM",
            )

        agent._compress_context = _compress

        big = "x" * 4000
        history = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
        ]
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent, conversation_history=history)

        msg = ctx.messages[ctx.current_turn_user_idx]
        assert msg["content"] == "hello"
        assert msg["api_content"] == "hello\n\nPLUGIN-CTX"
        agent._session_db.set_latest_user_api_content.assert_called_once_with(
            "sess-1", "hello", "hello\n\nPLUGIN-CTX"
        )


class TestSetLatestUserApiContent:
    def _open(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        return db

    def test_updates_newest_active_user_row(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="q1")
            db.append_message("s1", "assistant", content="a1")
            db.append_message("s1", "user", content="q2")
            assert db.set_latest_user_api_content("s1", "q2", "q2\n\nCTX") == 1
            msgs = db.get_messages_as_conversation("s1")
            assert "api_content" not in msgs[0]
            assert msgs[2]["api_content"] == "q2\n\nCTX"
        finally:
            db.close()

    def test_content_mismatch_writes_nothing(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="q1")
            assert db.set_latest_user_api_content("s1", "other", "other+CTX") == 0
            msgs = db.get_messages_as_conversation("s1")
            assert "api_content" not in msgs[0]
        finally:
            db.close()


class TestFlushCompressedSummaryOverrideGuard:
    def _make_agent(self, db, sid):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        return agent

    def test_override_skipped_for_compression_merged_row(self, tmp_path):
        """The #48677 override must not replace a compression-merged user
        message: that would silently drop the compaction summary from the
        durable clean transcript."""
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-merged"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            merged = "[prior context]\ncompaction summary\n\nactual question"
            messages = [
                {
                    "role": "user",
                    "content": merged,
                    COMPRESSED_SUMMARY_METADATA_KEY: True,
                }
            ]
            agent._persist_user_message_idx = 0
            agent._persist_user_message_override = "actual question"
            agent._persist_user_message_timestamp = None
            agent._flush_messages_to_session_db(messages, None)

            msgs = db.get_messages_as_conversation(sid)
            assert msgs[0]["content"] == merged  # summary survives
            assert "api_content" not in msgs[0]  # wire == row, no sidecar
        finally:
            db.close()

    def test_live_override_skipped_for_compression_merged_row(self, tmp_path):
        """Same invariant as the test above, for the in-memory path.

        ``finalize_turn`` calls ``_apply_persist_user_message_override`` and
        only then ``_persist_session``, so the live dict is rewritten BEFORE
        the DB-write guard above ever sees it. A compaction-merged row must
        survive: the summary is the whole pre-compaction history, and this
        list is the continuation history the next turn is built from.
        """
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-merged-live"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            merged = "[prior context]\ncompaction summary\n\nactual question"
            messages = [
                {
                    "role": "user",
                    "content": merged,
                    COMPRESSED_SUMMARY_METADATA_KEY: True,
                }
            ]
            agent._persist_user_message_idx = 0
            agent._persist_user_message_override = "actual question"
            agent._persist_user_message_timestamp = 1730000000

            agent._apply_persist_user_message_override(messages)

            assert messages[0]["content"] == merged, (
                "the compaction summary was erased from the continuation history"
            )
            # The paired timestamp override is unrelated and still applies.
            assert messages[0]["timestamp"] == 1730000000
        finally:
            db.close()

    def test_live_override_still_applies_without_merge_marker(self, tmp_path):
        """Negative control: an ordinary turn is still cleaned in place."""
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-plain-live"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            messages = [
                {
                    "role": "user",
                    "content": "[gateway note] observed\n\nactual question",
                }
            ]
            agent._persist_user_message_idx = 0
            agent._persist_user_message_override = "actual question"
            agent._persist_user_message_timestamp = None

            agent._apply_persist_user_message_override(messages)

            assert messages[0]["content"] == "actual question"
        finally:
            db.close()


class TestFlushSanitizeDivergenceCapture:
    def _make_agent(self, db, sid):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=db,
            session_id=sid,
        )
        agent._session_db_created = True
        return agent

    def test_user_content_sanitize_would_rewrite_is_captured(self, tmp_path):
        """get_messages_as_conversation strips <memory-context> fences on
        load; the sent bytes must survive in the sidecar so a reloaded
        session replays what was actually on the wire."""
        from agent.memory_manager import sanitize_context

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-fence"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            raw = "what does a literal <memory-context> tag do?"
            assert sanitize_context(raw) != raw  # test precondition
            messages = [
                {"role": "user", "content": raw},
                {"role": "assistant", "content": "it fences recalled memory <memory-context>"},
            ]
            agent._flush_messages_to_session_db(messages, None)

            msgs = db.get_messages_as_conversation(sid)
            assert msgs[0]["content"] == sanitize_context(raw).strip()
            assert msgs[0]["api_content"] == raw
            assert msgs[1]["api_content"] == (
                "it fences recalled memory <memory-context>"
            )
        finally:
            db.close()

    def test_plain_whitespace_padding_not_captured(self, tmp_path):
        """The api build strips every outgoing content string, so mere
        surrounding whitespace replays identically — no sidecar bloat."""
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-ws"
        db.create_session(session_id=sid, source="cli")
        try:
            agent = self._make_agent(db, sid)
            messages = [{"role": "user", "content": "  hello  \n"}]
            agent._flush_messages_to_session_db(messages, None)
            msgs = db.get_messages_as_conversation(sid)
            assert "api_content" not in msgs[0]
        finally:
            db.close()


class TestMaxIterationsSummaryReplay:
    def test_summary_request_substitutes_sidecar_bytes(self):
        """The forced-summary request must replay the same bytes every
        main-loop call sent — popping the sidecar without substituting sends
        CLEAN content and diverges the prefix at the earliest injected
        message, exactly when the context is largest."""
        from run_agent import AIAgent
        from agent.chat_completion_helpers import handle_max_iterations

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"

        captured = {}

        class _Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "RAW-RESPONSE"

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        transport = types.SimpleNamespace(
            normalize_response=lambda _r: types.SimpleNamespace(content="SUMMARY")
        )

        messages = [
            {"role": "user", "content": "q1", "api_content": "q1\n\nPLUGIN-CTX"},
            {"role": "assistant", "content": "a1"},
        ]
        with patch.object(
            agent, "_ensure_primary_openai_client", return_value=client
        ), patch.object(agent, "_get_transport", return_value=transport):
            out = handle_max_iterations(agent, messages, 5)

        assert out == "SUMMARY"
        sent_users = [
            m for m in captured["messages"] if m.get("role") == "user"
        ]
        assert sent_users[0]["content"] == "q1\n\nPLUGIN-CTX"
        for m in captured["messages"]:
            assert "api_content" not in m
        # The live history dict is never mutated.
        assert messages[0]["content"] == "q1"
        assert messages[0]["api_content"] == "q1\n\nPLUGIN-CTX"

    def test_summary_request_replays_ram_context_with_identical_prefix(self):
        """The direct summary call must reuse existing RAM suffixes, not recapture."""
        from agent.chat_completion_helpers import _iteration_summary_api_messages
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openai-compat",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        agent.ephemeral_system_prompt = None
        agent.prefill_messages = []
        snapshot = "Latitude: 37.7749\nLongitude: -122.4194"
        messages = [
            {
                "role": "user",
                "content": "where am I?",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-1",
            },
            {"role": "assistant", "content": "near the bay"},
        ]

        with bind_volatile_user_context(agent, snapshot, "tg-1"):
            normal, _ = build_api_messages(
                agent,
                messages,
                current_turn_user_idx=0,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="SYS",
            )
            summary = _iteration_summary_api_messages(
                agent,
                messages + [{"role": "user", "content": "Summarize the work."}],
            )

        assert summary[:-1] == normal
        assert summary[-1]["content"] == "Summarize the work."
        assert snapshot in summary[1]["content"]
        assert snapshot not in json.dumps(messages)

    def test_inline_moa_summary_does_not_replay_ram_context(self):
        """An exhausted inline-MoA turn must keep its no-location boundary."""
        from agent.chat_completion_helpers import _iteration_summary_api_messages
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openai-compat",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        agent.ephemeral_system_prompt = None
        agent.prefill_messages = []
        snapshot = "Latitude: 51.5074\nLongitude: -0.1278"
        messages = [
            {
                "role": "user",
                "content": "old question",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-old",
            },
            {"role": "assistant", "content": "old answer"},
            {
                "role": "user",
                "content": "compare approaches",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-current",
            },
        ]

        with bind_volatile_user_context(agent, snapshot, "tg-old"):
            build_api_messages(
                agent,
                messages[:2],
                current_turn_user_idx=0,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="SYS",
            )
        with bind_volatile_user_context(agent, snapshot, "tg-current"):
            build_api_messages(
                agent,
                messages,
                current_turn_user_idx=2,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="SYS",
            )

        summary = _iteration_summary_api_messages(
            agent,
            messages + [{"role": "user", "content": "Summarize the work."}],
            allow_volatile_replay=False,
        )

        assert snapshot not in json.dumps(summary)
        assert snapshot not in json.dumps(messages)

    def test_same_runtime_fork_copies_replay_without_new_snapshot(self):
        """A cache-parity fork gets historical bytes via an unaliased RAM copy."""
        from run_agent import AIAgent

        def make_agent():
            instance = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                provider="openai-compat",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            instance.ephemeral_system_prompt = None
            return instance

        parent = make_agent()
        fork = make_agent()
        snapshot = "Latitude: 48.8566\nLongitude: 2.3522"
        history = [
            {
                "role": "user",
                "content": "old question",
                VOLATILE_USER_CONTEXT_REPLAY_ID_KEY: "tg-old",
            },
            {"role": "assistant", "content": "old answer"},
        ]
        with bind_volatile_user_context(parent, snapshot, "tg-old"):
            parent_wire, _ = build_api_messages(
                parent,
                history,
                current_turn_user_idx=0,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="",
            )

        copy_volatile_user_context_history(parent, fork)
        assert fork._volatile_user_context_history is not parent._volatile_user_context_history
        fork_messages = history + [{"role": "user", "content": "synthetic fork prompt"}]
        with bind_volatile_user_context(fork, None, None):
            fork_wire, _ = build_api_messages(
                fork,
                fork_messages,
                current_turn_user_idx=2,
                ext_prefetch_cache="",
                plugin_user_context="",
                moa_config=None,
                active_system_prompt="",
            )

        assert fork_wire[:2] == parent_wire
        assert fork_wire[-1]["content"] == "synthetic fork prompt"


class TestSessionRowExistsBeforePreflightCompaction:
    """Moving the crash persist after prefetch/pre_llm_call (one write with
    the final sidecar) must NOT delay session-row creation: with PRAGMA
    foreign_keys=ON, in-place ``archive_and_compact`` inserts message rows
    referencing ``sessions(id)`` and rotation creates a child whose
    ``parent_session_id`` points at the current row — both run during
    preflight compression, which a fresh oversized first turn reaches
    before the delayed persist. Drives the real ``compress_context`` path
    against a real, empty SessionDB."""

    def _make_agent(self, db, sid, *, in_place):
        from run_agent import AIAgent

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="cli",
                quiet_mode=True,
                session_db=db,
                session_id=sid,
                skip_context_files=True,
                skip_memory=True,
            )
        # Fresh first turn: no session row has been created yet.
        assert not agent._session_db_created
        agent._cached_system_prompt = "SYSTEM"
        agent.compression_enabled = True
        agent.compression_in_place = in_place
        agent._compression_feasibility_checked = True

        calls = {"n": 0}

        def _should_compress(_tokens):
            calls["n"] += 1
            return calls["n"] == 1  # compress once, then stop

        seen = {}
        compacted = [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "hello"},
        ]

        def _compress(_messages, **_kwargs):
            # The ordering invariant under test: the row must already exist
            # when compression starts — the DB writes right after this call
            # (archive_and_compact / child create_session) reference it.
            seen["row_at_compress"] = db.get_session(sid)
            return compacted

        compressor = MagicMock()
        compressor.protect_first_n = 0
        compressor.protect_last_n = 0
        compressor.threshold_tokens = 1
        compressor.context_length = 1000
        compressor.last_prompt_tokens = -1
        compressor.should_compress = _should_compress
        compressor.should_defer_preflight_to_real_usage = lambda _t: False
        compressor.get_active_compression_failure_cooldown = lambda: None
        compressor.compress = _compress
        compressor.compression_count = 1
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_summary_auth_failure = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        return agent, seen

    def _oversized_history(self):
        big = "x" * 4000
        return [
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
        ]

    def test_in_place_first_turn_compaction_persists(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-fresh-inplace"
        try:
            agent, seen = self._make_agent(db, sid, in_place=True)
            with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
                ctx = _build(agent, conversation_history=self._oversized_history())

            # The row was created before compression started — without it the
            # FK on messages.session_id rejects the compacted rows and the
            # compaction write is silently lost.
            assert seen["row_at_compress"] is not None
            assert agent.session_id == sid  # no rotation
            assert agent._last_compaction_in_place is True
            contents = [m["content"] for m in db.get_messages(sid)]
            assert "[CONTEXT COMPACTION] summary" in contents
            # And the live context is the compacted set.
            assert ctx.messages[ctx.current_turn_user_idx]["content"] == "hello"
        finally:
            db.close()

    def test_rotation_first_turn_compaction_creates_child(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-fresh-rot"
        try:
            agent, seen = self._make_agent(db, sid, in_place=False)
            with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
                _build(agent, conversation_history=self._oversized_history())

            # The parent row existed before compression started — the child
            # INSERT's parent_session_id FK needs it; without it
            # create_session raises and the orphan-avoidance path rolls the
            # id back, so the rotation never happens.
            assert seen["row_at_compress"] is not None
            child = agent.session_id
            assert child != sid
            child_row = db.get_session(child)
            assert child_row is not None
            assert child_row["parent_session_id"] == sid
            parent_row = db.get_session(sid)
            assert parent_row is not None
            assert parent_row["end_reason"] == "compression"
        finally:
            db.close()


class TestStaleConfirmationRedactionDropsSidecar:
    def test_redaction_pops_api_content(self):
        """The expired-confirmation redaction rewrites content — replaying
        the sidecar verbatim would resend the dangerous bytes it just
        redacted."""
        from agent.replay_cleanup import strip_stale_dangerous_confirmations

        history = [
            {
                "role": "user",
                "content": "confirm forced restart",
                "api_content": "confirm forced restart\n\nPLUGIN-CTX",
                "timestamp": 1000.0,
            }
        ]
        cleaned = strip_stale_dangerous_confirmations(
            history, now=1000.0 + 999999.0
        )
        assert cleaned[0]["content"] != "confirm forced restart"
        assert "api_content" not in cleaned[0]
