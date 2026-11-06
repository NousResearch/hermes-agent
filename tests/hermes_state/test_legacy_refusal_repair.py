"""One-time repair of chats locked by a safety refusal stored before #149708.

Builds before #149708 kept a refused turn in the model context and closed it with a ``failed_turn``
boundary whose error card says ``content_policy_blocked`` (the exact rows below were read back from a
real ``hermes chat`` run of the pre-#149708 tree against an Anthropic-shaped ``stop_reason=refusal``
provider). The safety filter judges the whole thread, so that chat refused every later message. A
store opened by a current build takes those turns out of the model context once, keeps the asks on
screen as greyed refused rows, and leaves alone every chat it cannot prove locked.
"""
import json

from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND
from hermes_state import SessionDB
from hermes_state_legacy_refusal import LEGACY_REFUSAL_REPAIR_DONE

_REFUSAL_CARD = {
    "error": "content_policy_blocked: model declined (content_filter)",
    "error_surface": {"layer": "provider", "code": "content_policy_blocked", "retryable": False,
                      "provider": "anthropic", "model": "claude", "provider_label": "Anthropic"},
}
_BOUNDARY = "Your request was not processed. Send it again if you still want me to carry it out."


def _refused_turn(db, sid, ask):
    db.append_message(sid, "user", ask)
    db.append_message(sid, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)


def _as_legacy_store(db):
    """A store a pre-#149708 build wrote: the one-time repair has never run on it."""
    db._conn.execute("DELETE FROM state_meta WHERE key = ?", (LEGACY_REFUSAL_REPAIR_DONE,))
    db.close()


def _model(db, sid):
    return [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid, repair_alternation=True)]


def _display(db, sid):
    return [(m["role"], m["content"], m.get("display_kind"))
            for m in db.get_messages_as_conversation(sid, include_compacted=True)]


def test_locked_chat_answers_again_and_unlocked_chats_are_untouched(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("locked", "cli")
    db.append_message("locked", "user", "hello")
    db.append_message("locked", "assistant", "hi")
    _refused_turn(db, "locked", "the blocked ask")
    _refused_turn(db, "locked", "an innocent follow-up")  # refused too: the thread still held the ask
    # Control: the chat answered after its refusal, so later turns were built on it and it was not locked.
    db.create_session("answered", "cli")
    _refused_turn(db, "answered", "the blocked ask")
    db.append_message("answered", "user", "something else")
    db.append_message("answered", "assistant", "sure")
    untouched = _model(db, "answered")
    _as_legacy_store(db)

    db = SessionDB(tmp_path / "state.db")
    try:
        assert _model(db, "locked") == [("user", "hello"), ("assistant", "hi")]
        assert _display(db, "locked") == [
            ("user", "hello", None), ("assistant", "hi", None),
            ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND),
            ("user", "an innocent follow-up", CONTENT_REFUSED_DISPLAY_KIND),
        ]
        assert _model(db, "answered") == untouched
        assert db.get_meta(LEGACY_REFUSAL_REPAIR_DONE) == "1"
        # Once per store: a refusal-shaped row written later is never rescanned.
        _refused_turn(db, "answered", "a later ask")
        before = _model(db, "answered")
    finally:
        db.close()
    db = SessionDB(tmp_path / "state.db")
    try:
        assert _model(db, "answered") == before
    finally:
        db.close()


def test_refusal_folded_into_a_summary_is_left_alone_and_a_busy_chat_waits(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("compacted", "cli")
    _refused_turn(db, "compacted", "the blocked ask")
    # Bot Chat /new -> /compact: the refused turn was archived into a summary that still restates it.
    db._conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE session_id = 'compacted'")
    db.append_message("compacted", "user", "[CONTEXT COMPACTION] the user asked the blocked ask")
    db.append_message("compacted", "assistant", "noted")
    _refused_turn(db, "compacted", "an innocent follow-up")
    summary_view = _model(db, "compacted")
    db.create_session("busy", "cli")
    _refused_turn(db, "busy", "the blocked ask")
    assert db.acquire_session_turn_lease("busy", "pid=1:turn=live", ttl_seconds=300.0)
    _as_legacy_store(db)

    db = SessionDB(tmp_path / "state.db")
    try:
        # Dropping rows cannot unlock a chat whose summary carries the refusal: it is not touched.
        assert _model(db, "compacted") == summary_view
        # A turn owns the busy chat right now: the repair waits instead of rewriting under it.
        assert _model(db, "busy") == [("user", "the blocked ask"), ("assistant", _BOUNDARY)]
        assert db.get_meta(LEGACY_REFUSAL_REPAIR_DONE) is None
        assert json.loads(db.get_meta("legacy_refusal_repair_pending")) == ["busy"]
        db.release_session_turn_lease("busy", "pid=1:turn=live")
    finally:
        db.close()
    db = SessionDB(tmp_path / "state.db")
    try:
        assert _model(db, "busy") == []
        assert db.get_meta(LEGACY_REFUSAL_REPAIR_DONE) == "1"
    finally:
        db.close()


def test_subagent_and_cron_runs_keep_their_record(tmp_path):
    """A delegated run's one "turn" is the whole run: retaining it would hollow out the record and
    unlock nothing. Delegated and cron runs keep it by source even after an answered exchange; a chat
    never answered before a refused run that used tools (an ACP task run) keeps it by shape. A Bot Chat
    refused at its very first plain message is still unlocked: there /new becomes /compact."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("parent", "cli")
    db.append_message("parent", "user", "go")
    db.append_message("parent", "assistant", "ok")
    runs = {"sub": ("subagent", "parent"), "gw_child": ("telegram", "parent"), "job": ("cron", None),
            "task_chat": ("local", None)}
    for sid, (source, parent) in runs.items():
        db.create_session(sid, source, parent_session_id=parent)
        if sid != "task_chat":  # reads like a locked chat: only its source or lineage keeps it
            db.append_message(sid, "user", "a first goal")
            db.append_message(sid, "assistant", "done")
        db.append_message(sid, "user", "the delegated goal")
        for i in range(3):
            db.append_message(sid, "assistant", "", tool_calls=[
                {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
            db.append_message(sid, "tool", f"out{i}", tool_call_id=f"c{i}")
        db.append_message(sid, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)
        db.end_session(sid, "agent_close")
    before = {sid: (db.get_session(sid)["message_count"], _model(db, sid)) for sid in runs}
    db.create_session("first_ask", "local")
    _refused_turn(db, "first_ask", "the blocked ask")
    _refused_turn(db, "first_ask", "an innocent follow-up")
    _as_legacy_store(db)

    db = SessionDB(tmp_path / "state.db")
    try:
        assert {sid: (db.get_session(sid)["message_count"], _model(db, sid)) for sid in runs} == before
        assert _model(db, "first_ask") == []
        assert _display(db, "first_ask") == [("user", "the blocked ask", "content_refused"),
                                             ("user", "an innocent follow-up", "content_refused")]
        assert db.get_meta(LEGACY_REFUSAL_REPAIR_DONE) == "1"
    finally:
        db.close()


def test_internal_turn_heads_leave_the_context_without_being_shown(tmp_path):
    """Only a row the user typed is shown as their refused message; a background-task notice, a
    process notice or a hidden row that started a refused turn leaves the context unshown."""
    db = SessionDB(tmp_path / "state.db")
    kinds = ("async_delegation_complete", "process_complete", "hidden")
    for kind in kinds:
        db.create_session(kind, "cli")
        db.append_message(kind, "user", "hello")
        db.append_message(kind, "assistant", "hi")
        db.append_message(kind, "user", "[internal] background work finished", display_kind=kind)
        db.append_message(kind, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)
        _refused_turn(db, kind, "an innocent follow-up")
    _as_legacy_store(db)

    db = SessionDB(tmp_path / "state.db")
    try:
        for kind in kinds:
            assert _model(db, kind) == [("user", "hello"), ("assistant", "hi")]
            refused = [m for m in _display(db, kind) if m[2] == CONTENT_REFUSED_DISPLAY_KIND]
            assert refused == [("user", "an innocent follow-up", CONTENT_REFUSED_DISPLAY_KIND)], kind
            # Same footing as /undo and #149708's retention: an open view's /undo sees the history moved.
            assert db.get_session(kind)["rewind_count"] == 1
    finally:
        db.close()


def test_rotated_refusal_is_left_alone_and_other_failures_stay_in_the_context(tmp_path):
    """A /compress rotation's summary restates the refused ask, so its continuation is left as it
    is, and so is the ended root; a failure that is not a refusal keeps its turn in the context."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "cli")
    db.append_message("root", "user", "hello")
    db.append_message("root", "assistant", "hi")
    _refused_turn(db, "root", "the blocked ask")
    db.end_session("root", "compression")
    db.create_session("tip", "cli", parent_session_id="root")
    db.append_message("tip", "user", "[CONTEXT COMPACTION] the user asked the blocked ask")
    db.append_message("tip", "assistant", "noted")
    _refused_turn(db, "tip", "an innocent follow-up")
    rotated = {sid: _model(db, sid) for sid in ("root", "tip")}
    db.create_session("mixed", "cli")
    db.append_message("mixed", "user", "hello")
    db.append_message("mixed", "assistant", "hi")
    db.append_message("mixed", "user", "a slow ask")
    db.append_message("mixed", "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata={
        "error": "timeout", "error_surface": {"layer": "provider", "code": "provider_timeout"}})
    _refused_turn(db, "mixed", "the blocked ask")
    _as_legacy_store(db)

    db = SessionDB(tmp_path / "state.db")
    try:
        assert {sid: _model(db, sid) for sid in ("root", "tip")} == rotated
        assert _model(db, "mixed") == [
            ("user", "hello"), ("assistant", "hi"), ("user", "a slow ask"), ("assistant", _BOUNDARY)]
        assert db.get_session("mixed")["message_count"] == 4
    finally:
        db.close()
