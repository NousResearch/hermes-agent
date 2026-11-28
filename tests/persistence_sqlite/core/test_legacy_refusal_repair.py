"""One-time repair of chats locked by a stored safety refusal.

Older builds kept a refused turn in the model context and closed it with a ``failed_turn`` boundary
whose error card says ``content_policy_blocked`` (the exact rows below were read back from a real
``hermes chat`` run of such a build against an Anthropic-shaped ``stop_reason=refusal`` provider). The
safety filter judges the whole thread, so that chat refused every later message. A store opened by a
current build takes those turns out of the model context once, keeps the asks on screen as greyed
refused rows, and leaves alone every chat it cannot prove locked.
"""
import json

from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND
from persistence.conversation.messages import ConversationRequest, TranscriptView
from persistence.sqlite.conversation._store.legacy_refusal import LEGACY_REFUSAL_REPAIR_DONE
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_read import conversation, session_row
from tests.fixtures.storage_seed import seed_messages, seed_session

_REFUSAL_CARD = {
    "error": "content_policy_blocked: model declined (content_filter)",
    "error_surface": {"layer": "provider", "code": "content_policy_blocked", "retryable": False,
                      "provider": "anthropic", "model": "claude", "provider_label": "Anthropic"},
}
_BOUNDARY = "Your request was not processed. Send it again if you still want me to carry it out."


def _open(home):
    return open_persistence(profile_scope(home=home))


def _append(db, sid, role, content, **fields):
    seed_messages(db, sid, {"role": role, "content": content, **fields})


def _refused_turn(db, sid, ask):
    _append(db, sid, "user", ask)
    _append(db, sid, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)


def _raw(db, sql, params=()):
    raw_holder(db)._execute_write(lambda conn: conn.execute(sql, params))


def _as_legacy_store(db):
    """A store an older build wrote: the one-time repair has never run on it."""
    _raw(db, "DELETE FROM state_meta WHERE key = ?", (LEGACY_REFUSAL_REPAIR_DONE,))
    db.close()


def _model(db, sid):
    return [(m["role"], m["content"]) for m in conversation(db, sid, ConversationRequest(repair_alternation=True))]


def _display(db, sid):
    return [(m["role"], m["content"], m.get("display_kind"))
            for m in conversation(db, sid, ConversationRequest(view=TranscriptView.DISPLAY))]


def test_locked_chat_answers_again_and_unlocked_chats_are_untouched(tmp_path):
    db = _open(tmp_path)
    seed_session(db, "locked", "cli")
    _append(db, "locked", "user", "hello")
    _append(db, "locked", "assistant", "hi")
    _refused_turn(db, "locked", "the blocked ask")
    _refused_turn(db, "locked", "an innocent follow-up")  # refused too: the thread still held the ask
    # Control: the chat answered after its refusal, so later turns were built on it and it was not locked.
    seed_session(db, "answered", "cli")
    _refused_turn(db, "answered", "the blocked ask")
    _append(db, "answered", "user", "something else")
    _append(db, "answered", "assistant", "sure")
    untouched = _model(db, "answered")
    _as_legacy_store(db)

    db = _open(tmp_path)
    try:
        assert _model(db, "locked") == [("user", "hello"), ("assistant", "hi")]
        assert _display(db, "locked") == [
            ("user", "hello", None), ("assistant", "hi", None),
            ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND),
            ("user", "an innocent follow-up", CONTENT_REFUSED_DISPLAY_KIND),
        ]
        assert _model(db, "answered") == untouched
        assert db.state_meta.get(LEGACY_REFUSAL_REPAIR_DONE) == "1"
        # Once per store: a refusal-shaped row written later is never rescanned.
        _refused_turn(db, "answered", "a later ask")
        before = _model(db, "answered")
    finally:
        db.close()
    db = _open(tmp_path)
    try:
        assert _model(db, "answered") == before
    finally:
        db.close()


def test_refusal_folded_into_a_summary_is_left_alone_and_a_busy_chat_waits(tmp_path):
    db = _open(tmp_path)
    seed_session(db, "compacted", "cli")
    _refused_turn(db, "compacted", "the blocked ask")
    # Bot Chat /new -> /compact: the refused turn was archived into a summary that still restates it.
    _raw(db, "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = 'compacted'")
    _append(db, "compacted", "user", "[CONTEXT COMPACTION] the user asked the blocked ask")
    _append(db, "compacted", "assistant", "noted")
    _refused_turn(db, "compacted", "an innocent follow-up")
    summary_view = _model(db, "compacted")
    seed_session(db, "busy", "cli")
    _refused_turn(db, "busy", "the blocked ask")
    assert db.coordination.try_acquire_turn_lease(session_id="busy", holder="pid=1:turn=live", ttl_seconds=300.0)
    _as_legacy_store(db)

    db = _open(tmp_path)
    try:
        # Dropping rows cannot unlock a chat whose summary carries the refusal: it is not touched.
        assert _model(db, "compacted") == summary_view
        # A turn owns the busy chat right now: the repair waits instead of rewriting under it.
        assert _model(db, "busy") == [("user", "the blocked ask"), ("assistant", _BOUNDARY)]
        assert db.state_meta.get(LEGACY_REFUSAL_REPAIR_DONE) is None
        assert json.loads(db.state_meta.get("legacy_refusal_repair_pending")) == ["busy"]
        db.coordination.release_turn_lease(session_id="busy", holder="pid=1:turn=live")
    finally:
        db.close()
    db = _open(tmp_path)
    try:
        assert _model(db, "busy") == []
        assert db.state_meta.get(LEGACY_REFUSAL_REPAIR_DONE) == "1"
    finally:
        db.close()


def test_subagent_and_cron_runs_keep_their_record(tmp_path):
    """A delegated run's one "turn" is the whole run: retaining it would hollow out the record and
    unlock nothing. Delegated and cron runs keep it by source even after an answered exchange; a chat
    never answered before a refused run that used tools (an ACP task run) keeps it by shape. A Bot Chat
    refused at its very first plain message is still unlocked: there /new becomes /compact, and so is
    an answered chat whose refused turn ran tools (only a never-answered tool run is a task run)."""
    db = _open(tmp_path)
    seed_session(db, "parent", "cli")
    _append(db, "parent", "user", "go")
    _append(db, "parent", "assistant", "ok")
    runs = {"sub": ("subagent", "parent"), "gw_child": ("telegram", "parent"), "job": ("cron", None),
            "task_chat": ("local", None)}
    for sid, (source, parent) in runs.items():
        seed_session(db, sid, source, parent_session_id=parent)
        if sid != "task_chat":  # reads like a locked chat: only its source or lineage keeps it
            _append(db, sid, "user", "a first goal")
            _append(db, sid, "assistant", "done")
        _append(db, sid, "user", "the delegated goal")
        for i in range(3):
            _append(db, sid, "assistant", "", tool_calls=[
                {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
            _append(db, sid, "tool", f"out{i}", tool_call_id=f"c{i}")
        _append(db, sid, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)
        db.sessions.end(sid, "agent_close")
    before = {sid: (session_row(db, sid)["message_count"], _model(db, sid)) for sid in runs}
    seed_session(db, "first_ask", "local")
    _refused_turn(db, "first_ask", "the blocked ask")
    _refused_turn(db, "first_ask", "an innocent follow-up")
    seed_session(db, "tool_chat", "local")
    _append(db, "tool_chat", "user", "hello")
    _append(db, "tool_chat", "assistant", "hi")
    _append(db, "tool_chat", "user", "the blocked ask")
    _append(db, "tool_chat", "assistant", "", tool_calls=[
        {"id": "t0", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    _append(db, "tool_chat", "tool", "out", tool_call_id="t0")
    _append(db, "tool_chat", "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)
    _as_legacy_store(db)

    db = _open(tmp_path)
    try:
        assert {sid: (session_row(db, sid)["message_count"], _model(db, sid)) for sid in runs} == before
        assert _model(db, "first_ask") == []
        assert _display(db, "first_ask") == [("user", "the blocked ask", "content_refused"),
                                             ("user", "an innocent follow-up", "content_refused")]
        assert _model(db, "tool_chat") == [("user", "hello"), ("assistant", "hi")]
        assert [m for m in _display(db, "tool_chat") if m[2] == CONTENT_REFUSED_DISPLAY_KIND] == [
            ("user", "the blocked ask", CONTENT_REFUSED_DISPLAY_KIND)]
        assert db.state_meta.get(LEGACY_REFUSAL_REPAIR_DONE) == "1"
    finally:
        db.close()


def test_internal_turn_heads_leave_the_context_without_being_shown(tmp_path):
    """Only a row the user typed is shown as their refused message; a background-task notice, a
    process notice or a hidden row that started a refused turn leaves the context unshown."""
    db = _open(tmp_path)
    kinds = ("async_delegation_complete", "process_complete", "hidden")
    for kind in kinds:
        seed_session(db, kind, "cli")
        _append(db, kind, "user", "hello")
        _append(db, kind, "assistant", "hi")
        _append(db, kind, "user", "[internal] background work finished", display_kind=kind)
        _append(db, kind, "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata=_REFUSAL_CARD)
        _refused_turn(db, kind, "an innocent follow-up")
    _as_legacy_store(db)

    db = _open(tmp_path)
    try:
        for kind in kinds:
            assert _model(db, kind) == [("user", "hello"), ("assistant", "hi")]
            refused = [m for m in _display(db, kind) if m[2] == CONTENT_REFUSED_DISPLAY_KIND]
            assert refused == [("user", "an innocent follow-up", CONTENT_REFUSED_DISPLAY_KIND)], kind
            # Same footing as /undo and the retention of a new refusal: an open view's /undo sees the history moved.
            assert session_row(db, kind)["rewind_count"] == 1
    finally:
        db.close()


def test_rotated_refusal_is_left_alone_and_other_failures_stay_in_the_context(tmp_path):
    """A /compress rotation's summary restates the refused ask, so its continuation is left as it
    is, and so is the ended root; a failure that is not a refusal keeps its turn in the context."""
    db = _open(tmp_path)
    seed_session(db, "root", "cli")
    _append(db, "root", "user", "hello")
    _append(db, "root", "assistant", "hi")
    _refused_turn(db, "root", "the blocked ask")
    db.sessions.end("root", "compression")
    seed_session(db, "tip", "cli", parent_session_id="root")
    _append(db, "tip", "user", "[CONTEXT COMPACTION] the user asked the blocked ask")
    _append(db, "tip", "assistant", "noted")
    _refused_turn(db, "tip", "an innocent follow-up")
    rotated = {sid: _model(db, sid) for sid in ("root", "tip")}
    seed_session(db, "mixed", "cli")
    _append(db, "mixed", "user", "hello")
    _append(db, "mixed", "assistant", "hi")
    _append(db, "mixed", "user", "a slow ask")
    _append(db, "mixed", "assistant", _BOUNDARY, display_kind="failed_turn", display_metadata={
        "error": "timeout", "error_surface": {"layer": "provider", "code": "provider_timeout"}})
    _refused_turn(db, "mixed", "the blocked ask")
    _as_legacy_store(db)

    db = _open(tmp_path)
    try:
        assert {sid: _model(db, sid) for sid in ("root", "tip")} == rotated
        assert _model(db, "mixed") == [
            ("user", "hello"), ("assistant", "hi"), ("user", "a slow ask"), ("assistant", _BOUNDARY)]
        assert session_row(db, "mixed")["message_count"] == 4
    finally:
        db.close()
