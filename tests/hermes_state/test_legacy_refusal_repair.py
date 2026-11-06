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
