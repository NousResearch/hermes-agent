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


def test_retention_targets_the_boundary_not_the_newest_turn(tmp_path):
    """746: a mid-turn redirect persists a SECOND user row inside the refused run. The newest user
    turn (the default ``-1`` guess) is the redirect, so retaining it would leave the ORIGINAL ask +
    its tool rounds live in the model and the deterministic filter re-trips forever. Passing the
    refused run's original durable boundary drops the WHOLE run; the original ask is the retained
    display row."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        boundary = db.append_message("s", "user", "the request the filter blocked")
        db.append_message("s", "assistant", "", tool_calls=[{"id": "t1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message("s", "tool", "ran", tool_call_id="t1")
        # A mid-turn redirect, persisted as its own user row within the still-unfinished turn.
        db.append_message("s", "user", "actually, do this instead")

        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE, boundary_row_id=boundary) is True

        # The WHOLE refused run (original ask + tool round + redirect) left the model read.
        assert _model(db, "s") == [("user", "hello"), ("assistant", "hi there")]
        # The ORIGINAL ask is the retained, tagged display row — not the redirect.
        assert _display(db, "s") == [
            ("user", "hello", None),
            ("assistant", "hi there", None),
            ("user", "the request the filter blocked", CONTENT_REFUSED_DISPLAY_KIND),
        ]
    finally:
        db.close()


def test_without_the_boundary_the_newest_turn_guess_leaves_the_run_live(tmp_path):
    """746 control: the pre-fix behaviour (no boundary handed in) rewinds only the newest user turn,
    so a mid-turn redirect leaves the original refused ask ACTIVE in the model — exactly the bug the
    boundary fix removes."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        db.append_message("s", "user", "the request the filter blocked")
        # A tool round runs, THEN the mid-turn redirect lands — so the two user rows are separate
        # turns (not merged by consecutive-user repair), reproducing the real steer-mid-turn shape.
        db.append_message("s", "assistant", "", tool_calls=[{"id": "t1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message("s", "tool", "ran", tool_call_id="t1")
        db.append_message("s", "user", "actually, do this instead")

        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE) is True
        # Without the boundary, the original refused ask is still in the model read (the stuck bug).
        assert ("user", "the request the filter blocked") in _model(db, "s")
    finally:
        db.close()


def test_a_bad_boundary_falls_back_to_the_newest_turn(tmp_path):
    """A boundary that no longer resolves (already rewound / not a user row) must not fail the
    retention — it falls back to the newest user turn best-effort."""
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        db.append_message("s", "user", "the request the filter blocked")
        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE, boundary_row_id=999999) is True
        assert _display(db, "s")[-1] == (
            "user", "the request the filter blocked", CONTENT_REFUSED_DISPLAY_KIND)
    finally:
        db.close()


def test_retention_of_a_carrier_keeps_the_refused_label_on_reload(tmp_path):
    """788: when the refused ask is embedded in a composite compaction carrier, the display
    projection unwraps the carrier to the live ask — but must KEEP the content_refused tag so a
    reopened chat still renders the greyed refused block instead of an ordinary user bubble."""
    from agent.compaction_display import project_compaction_message_for_display
    from agent.context_compressor import (
        HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER)

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    carrier = (f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
               f"{_SUMMARY_END_MARKER}\n\nthe refused ask")
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "older ask")
        db.append_message("s", "assistant", "", tool_calls=[{"id": "c1", "type": "function",
                          "function": {"name": "shell", "arguments": "{}"}}])
        db.append_message("s", "tool", "ok", tool_call_id="c1")
        boundary = db.append_message("s", "user", carrier)  # composite carrier head of the refused turn
        db.append_message("s", "assistant", "failed")
        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE, boundary_row_id=boundary) is True
    finally:
        db.close()

    reopened = SessionDB(db_path)
    try:
        rows = reopened.get_messages_as_conversation("s", include_compacted=True)
        refused_rows = [m for m in rows if m.get("display_kind") == CONTENT_REFUSED_DISPLAY_KIND]
        # Exactly one row carries the refused tag: the archived carrier (the handoff scaffold that
        # rewind inserted as the new active head is a separate, hidden row).
        assert len(refused_rows) == 1
        refused_row = refused_rows[0]
        projected = project_compaction_message_for_display(refused_row)
        # The unwrap keeps the refused tag (so the greyed block renders), not a plain user bubble.
        assert projected is not None
        assert projected.get("display_kind") == CONTENT_REFUSED_DISPLAY_KIND
    finally:
        reopened.close()


def test_retention_applies_both_halves_in_one_write(tmp_path):
    """713: the model-side soft-delete and the display re-surface land in ONE write transaction, so a
    reopen can never see the user's message removed from BOTH views (half-applied retention). A
    successful retain always leaves the row EXCLUDED from the model AND tagged in the display."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    try:
        db.create_session("s", "cli")
        db.append_message("s", "user", "hello")
        db.append_message("s", "assistant", "hi there")
        boundary = db.append_message("s", "user", "the request the filter blocked")
        assert retain_refused_turn_for_display(db, "s", notice=_NOTICE, boundary_row_id=boundary) is True
        # rewind_count incremented exactly once (a single committed rewind), not zero or twice.
        count = db._conn.execute("SELECT rewind_count FROM sessions WHERE id = ?", ("s",)).fetchone()[0]
        assert count == 1
    finally:
        db.close()

    reopened = SessionDB(db_path)
    try:
        # Both halves are durable together: gone from the model, present+tagged in the display.
        assert _model(reopened, "s") == [("user", "hello"), ("assistant", "hi there")]
        assert _display(reopened, "s")[-1] == (
            "user", "the request the filter blocked", CONTENT_REFUSED_DISPLAY_KIND)
    finally:
        reopened.close()
