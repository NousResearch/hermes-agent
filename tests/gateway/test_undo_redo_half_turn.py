import ast
import asyncio
import inspect
import threading
import textwrap
from collections import OrderedDict

import pytest

import hermes_undo
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_state import SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    session_store._db = db
    hermes_undo._session_db = db
    hermes_undo.clear_state()
    yield session_store
    db.close()
    hermes_undo.clear_state()


def _source(chat_id="chat-1"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")


def _event(text, source):
    return MessageEvent(text=text, source=source, message_id="m1")


def _seed(db, sid):
    u1 = db.append_message(sid, "user", "q1")
    a1 = db.append_message(sid, "assistant", "a1")
    u2 = db.append_message(sid, "user", "q2")
    a2 = db.append_message(sid, "assistant", "a2")
    return u1, a1, u2, a2


def _runner(store):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    return runner


def test_cold_restart_redo_branches_touch_zero_rows(store):
    src = _source("restart")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    before = store._db.get_messages(entry.session_id, include_inactive=True)

    store.rewind_session(entry.session_id, 1)
    after_undo = store._db.get_messages(entry.session_id, include_inactive=True)
    hermes_undo.clear_state(entry.session_id)
    cold = store.restore_session(entry.session_id, 1)

    assert cold["reactivated_count"] == 0
    assert cold["message"] == "nothing to redo (redo history doesn't survive a restart)"
    assert store._db.get_messages(entry.session_id, include_inactive=True) == after_undo

    fresh = store.get_or_create_session(_source("never-undone"))
    _seed(store._db, fresh.session_id)
    fresh_before = store._db.get_messages(fresh.session_id, include_inactive=True)
    hermes_undo.clear_state(fresh.session_id)
    bare = store.restore_session(fresh.session_id, 1)

    assert bare["reactivated_count"] == 0
    assert bare["message"] == "nothing to redo"
    assert store._db.get_messages(fresh.session_id, include_inactive=True) == fresh_before
    assert before != after_undo


def test_no_active0_reconstruction_in_gateway_redo_paths():
    positive = ast.parse(
        "def planted(db):\n"
        "    return db.get_messages('sid', include_inactive=True)\n"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_messages"
        and any(
            kw.arg == "include_inactive" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in node.keywords
        )
        for node in ast.walk(positive)
    )

    for func in (SessionStore.restore_session, GatewayRunner._handle_redo_command):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        offenders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_messages"
            and any(
                kw.arg == "include_inactive"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
        ]
        assert offenders == []


def test_cached_agent_evicted_after_undo_and_after_redo(store):
    src = _source("evict")
    entry = store.get_or_create_session(src)
    _seed(store._db, entry.session_id)
    runner = _runner(store)
    cache_key = build_session_key(src)
    runner._agent_cache[cache_key] = ("stale-before-undo", "sig")

    undo_msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))

    assert "Undid" in undo_msg or "Removed" in undo_msg
    assert cache_key not in runner._agent_cache
    assert store.load_transcript(entry.session_id) == store._db.get_messages_as_conversation(entry.session_id)

    runner._agent_cache[cache_key] = ("stale-before-redo", "sig")
    redo_msg = asyncio.run(runner._handle_redo_command(_event("/redo", src)))

    assert "Redid" in redo_msg
    assert cache_key not in runner._agent_cache
    assert store.load_transcript(entry.session_id) == store._db.get_messages_as_conversation(entry.session_id)


def test_null_content_tail_confirmation_does_not_stringify_none(store):
    src = _source("null-tail")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "run tool")
    assistant = db.append_message(
        entry.session_id,
        "assistant",
        None,
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
    )
    db.append_message(entry.session_id, "tool", "ok", tool_call_id="call-1")
    runner = _runner(store)

    undo = store.rewind_session(entry.session_id, 1)
    assert set(undo["rewound_ids"]) == {assistant, assistant + 1}
    msg = asyncio.run(runner._handle_redo_command(_event("/redo", src)))

    assert "Redid" in msg
    assert "None" not in msg
