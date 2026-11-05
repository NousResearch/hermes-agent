"""The owner-side durable retention of a safety-refused turn (``retain_refused_turn_for_display``).

Regression for the "Hermes Bot stuck sessions" report: a refusal is deterministic for the whole
conversation, so a refused turn left in the MODEL's context re-trips the filter on every later
message — including after a reload. CLI and ACP own a direct ``SessionDB``, so they retain the
refused turn owner-side after the turn: the whole turn (user row + any tool rounds + the refusal)
leaves the model read, but the user's own message stays VISIBLE in the display projection, tagged
``content_refused``, so a close+reopen still shows "you asked X → refused". This pins both halves.
"""
from agent.turn_failure_copy import CONTENT_REFUSED_DISPLAY_KIND, CONTENT_REFUSED_METADATA_KEY
from hermes_state import SessionDB
from hermes_state_rewind import retain_refused_turn_for_display

_NOTICE = "This message was refused by the model's safety filter — try a different query."


def _display(db, sid):
    return [(m["role"], m["content"], m.get("display_kind"))
            for m in db.get_messages_as_conversation(sid, include_compacted=True)
            if m["role"] != "session_meta"]


def _model(db, sid):
    return [(m["role"], m["content"])
            for m in db.get_messages_as_conversation(sid) if m["role"] != "session_meta"]


def test_refused_turn_leaves_the_model_but_stays_in_the_display(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        # An accepted exchange, then a refused turn that ran a tool before the refusal.
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        db.append_message("s", "user", "the request the filter blocked")
        db.append_message("s", "assistant", "", tool_calls=[{"id": "t1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message("s", "tool", "ran", tool_call_id="t1")

        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE) is True

        # MODEL read: the whole refused turn is gone — the ask, the tool round, everything after —
        # so the deterministic filter is never handed the thread that tripped it again.
        assert _model(db, "s") == [("user", "hello"), ("assistant", "hi there")]

        # DISPLAY read: the prior exchange, plus the user's refused ask kept visible and tagged
        # content_refused. The raw tool actions + refusal collapse out (one greyed notice renders).
        assert _display(db, "s") == [
            ("user", "hello", None),
            ("assistant", "hi there", None),
            ("user", "the request the filter blocked", CONTENT_REFUSED_DISPLAY_KIND),
        ]
        # The stored notice rides display_metadata so every surface renders one copy.
        refused = db.get_messages_as_conversation("s", include_compacted=True)[-1]
        assert refused["display_metadata"][CONTENT_REFUSED_METADATA_KEY]["notice"] == _NOTICE
    finally:
        db.close()


def test_refused_turn_survives_a_simulated_reopen(tmp_path):
    """A fresh SessionDB (process restart / reopening the chat) still shows the refused block and
    still excludes it from the model read — the retention is durable, not an in-memory artifact."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        db.append_message("s", "user", "the request the filter blocked")
        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE) is True
    finally:
        db.close()

    reopened = SessionDB(db_path)
    try:
        assert _model(reopened, "s") == [("user", "hello"), ("assistant", "hi there")]
        assert _display(reopened, "s")[-1] == (
            "user", "the request the filter blocked", CONTENT_REFUSED_DISPLAY_KIND)
    finally:
        reopened.close()


def test_retain_is_best_effort_and_a_no_op_without_a_target():
    # No DB / no session id: never raises, reports nothing retained.
    assert retain_refused_turn_for_display(None, "s", notice=_NOTICE) is False
    assert retain_refused_turn_for_display(object(), "", notice=_NOTICE) is False


def test_retain_reports_false_without_a_user_turn(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        # No user turn to retain -> plan raises RewindTargetUnavailableError; swallowed, returns False.
        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE) is False
    finally:
        db.close()
