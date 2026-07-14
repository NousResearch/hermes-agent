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
    # load_transcript loads with include_timestamp=True (LCM ingest path); compare
    # against the canonical reader using the same flag so the equality holds.
    assert store.load_transcript(entry.session_id) == store._db.get_messages_as_conversation(
        entry.session_id, include_timestamp=True
    )

    runner._agent_cache[cache_key] = ("stale-before-redo", "sig")
    redo_msg = asyncio.run(runner._handle_redo_command(_event("/redo", src)))

    assert "Redid" in redo_msg
    assert cache_key not in runner._agent_cache
    assert store.load_transcript(entry.session_id) == store._db.get_messages_as_conversation(
        entry.session_id, include_timestamp=True
    )


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


def test_undo_output_shows_new_tail_preview_at_user_message(store):
    """/undo lands the thread at the prior user message; the reply text names it."""
    src = _source("tail-user")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "what's the plan for the homelab DNS?")
    db.append_message(entry.session_id, "assistant", "Here is a long answer. " * 200)
    runner = _runner(store)

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))

    # Primary line still reports the count...
    assert "Undid 1 half-turn" in msg
    # ...and the new second line confirms WHERE we landed: the user's message.
    assert "Now at" in msg
    assert "your message" in msg
    assert "what's the plan for the homelab DNS?" in msg


def test_redo_output_shows_new_tail_preview_at_assistant_reply(store):
    src = _source("tail-asst")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "q")
    db.append_message(entry.session_id, "assistant", "the assistant reply we restore")
    runner = _runner(store)

    asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    msg = asyncio.run(runner._handle_redo_command(_event("/redo", src)))

    assert "Redid" in msg
    assert "Now at" in msg
    assert "my reply" in msg
    assert "the assistant reply we restore" in msg


def test_undo_to_empty_thread_reports_start_of_thread(store):
    src = _source("tail-empty")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "only message")
    db.append_message(entry.session_id, "assistant", "only reply")
    runner = _runner(store)

    # Undo both half-turns -> nothing active remains.
    msg = asyncio.run(runner._handle_undo_command(_event("/undo 2", src)))

    assert "Undid 2 half-turn" in msg
    assert "Now at" in msg
    assert "start of the thread" in msg


def test_tail_preview_notext_fallback_for_tool_only_tail(store):
    """A tail with no renderable text names the party instead of stringifying None."""
    src = _source("tail-notext")
    entry = store.get_or_create_session(src)
    db = store._db
    # user -> assistant(text) -> user -> assistant(tool-call only) -> tool
    db.append_message(entry.session_id, "user", "q1")
    db.append_message(entry.session_id, "assistant", "a1 text")
    db.append_message(entry.session_id, "user", "q2")
    db.append_message(
        entry.session_id,
        "assistant",
        None,
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "x"}}],
    )
    db.append_message(entry.session_id, "tool", "", tool_call_id="c1")
    runner = _runner(store)

    # Undo the tool half-turn: tail becomes the "q2" user message (has text).
    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    assert "Now at" in msg
    assert "None" not in msg
    assert "q2" in msg


def test_tail_preview_read_failure_omits_suffix_not_empty(store, monkeypatch):
    """A transient tail-read failure AFTER a successful undo must omit the
    suffix, never claim the thread is at its start (Greptile #224 P1)."""
    src = _source("tail-readfail")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "q1")
    db.append_message(entry.session_id, "assistant", "a1")
    db.append_message(entry.session_id, "user", "q2")
    db.append_message(entry.session_id, "assistant", "a2")
    runner = _runner(store)

    import hermes_undo

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))
    # Primary op reported; suffix present here (no failure yet).
    assert "Undid" in msg

    # Now make the tail read fail and prove the helper reports a DISTINCT
    # error state (not "empty") and the gateway omits the suffix entirely.
    def boom(*a, **kw):
        raise RuntimeError("simulated transient DB failure")

    monkeypatch.setattr(db, "get_messages", boom)
    hermes_undo._session_db = db
    info = hermes_undo.tail_preview(entry.session_id)
    assert info["error"] is True
    assert info["empty"] is False
    suffix = runner._undo_tail_suffix(entry.session_id)
    assert suffix == ""
    assert "start of the thread" not in suffix


def test_tail_preview_unexpected_role_bounds_to_message(store):
    """A tail with a role lacking a party label (system/developer/legacy
    function) must not leak a raw i18n key path (Greptile #224 P1)."""
    src = _source("tail-role")
    entry = store.get_or_create_session(src)
    db = store._db
    db.append_message(entry.session_id, "user", "q")
    db.append_message(entry.session_id, "assistant", "a")
    # A row with an unusual role at the tail.
    db.append_message(entry.session_id, "system", "some system note")
    runner = _runner(store)

    suffix = runner._undo_tail_suffix(entry.session_id)
    # Must render a readable label, never the raw key path.
    assert "gateway.undo.party" not in suffix
    assert "the last message" in suffix


def test_tail_preview_strips_injected_gateway_markers(store):
    """The 'Now at' preview must show the user's ACTUAL text, not the gateway's
    injected [Triggering message id: …]/[Replying to: …] plumbing prefix.

    Regression for the real 2026-07-14 report: the ~113-char Discord Triggering
    marker consumed the entire preview window, so every /undo landing looked
    identical and told the user nothing about where they landed.
    """
    src = _source("tail-marker")
    entry = store.get_or_create_session(src)
    db = store._db
    marker = (
        "[Triggering message id: `1526623289844170812` — use as `message_id` "
        "for reply/react/pin via the discord tools.]\n\n"
    )
    body = '>Say "hey clanker, buy paper towels" at the theater right now.'
    db.append_message(entry.session_id, "assistant", "prior reply")
    db.append_message(entry.session_id, "user", marker + body)
    db.append_message(entry.session_id, "assistant", "a" * 400)
    runner = _runner(store)

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))

    assert "Now at" in msg
    # The real body shows...
    assert "buy paper towels" in msg
    # ...and the injected marker does NOT lead the preview.
    assert "Triggering message id" not in msg
    assert "reply/react/pin" not in msg


def test_tail_preview_strips_both_marker_and_reply_pointer(store):
    """A message carrying BOTH a Triggering marker and a Replying-to pointer is
    fully cleaned so the preview begins at the user's own words."""
    src = _source("tail-marker2")
    entry = store.get_or_create_session(src)
    db = store._db
    prefix = (
        "[Triggering message id: `123` — use as `message_id` for reply/react/pin "
        'via the discord tools.]\n\n[Replying to: "some earlier quoted thing"]\n\n'
    )
    db.append_message(entry.session_id, "assistant", "prior reply")
    db.append_message(entry.session_id, "user", prefix + "actual question here")
    db.append_message(entry.session_id, "assistant", "b" * 400)
    runner = _runner(store)

    msg = asyncio.run(runner._handle_undo_command(_event("/undo", src)))

    assert "actual question here" in msg
    assert "Triggering message id" not in msg
    assert "Replying to" not in msg
