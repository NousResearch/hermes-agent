"""(#111873) Scheduled composer messages — the durable store.

A scheduled message is a one-shot future user turn for ONE session: persisted with an absolute due
time, fired at most once, cancellable while pending. Same SessionDB ``state_meta`` base as
``/goal``, ``/loop`` and ``/heartbeat``, so the process that owns the session can fire it and a
restart finds it again.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a real state.db, mirroring the /heartbeat tick tests."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield root
    goals._DB_CACHE.clear()


def _schedule(session_id: str = "sess-1", *, text: str = "run the linter", in_seconds: float = 60.0):
    from hermes_cli.scheduled_messages import schedule_message

    return schedule_message(session_id, text, time.time() + in_seconds)


def test_schedule_does_not_fire_and_lands_in_the_pending_list(home):
    from hermes_cli.scheduled_messages import due_items, list_messages

    message = _schedule(text="summarise the repo")

    assert due_items("sess-1") == []                      # nothing is due yet
    pending = list_messages("sess-1")
    assert [item["id"] for item in pending] == [message.id]
    assert pending[0]["text"] == "summarise the repo"
    assert pending[0]["status"] == "pending" and pending[0]["overdue"] is False
    assert pending[0]["due_at"] == pytest.approx(message.due_at)


def test_due_time_is_the_trigger(home):
    from hermes_cli.scheduled_messages import due_items

    past = _schedule(text="late", in_seconds=-30)
    future = _schedule(text="later", in_seconds=600)

    due = due_items("sess-1")
    assert [item.text for item in due] == ["late"]         # ordered by due time, only what is due
    assert due[0].id == past.id and due[0].id != future.id


def test_claim_is_at_most_once(home):
    """The repeat-scan guard: the second scan of the same item must find nothing to dispatch."""
    from hermes_cli.scheduled_messages import abandon_fire, claim_due, list_messages

    message = _schedule(text="once only", in_seconds=-1)

    first = claim_due("sess-1", message.id)
    assert first is not None and first.text == "once only"
    assert claim_due("sess-1", message.id) is None         # already fired
    assert list_messages("sess-1") == []                   # gone from the pending view
    assert abandon_fire("sess-1", message.id) is True
    # An abandoned claim is a RETRY of a dispatch that never ran, not a second send: claimable
    # again — exactly once more.
    assert claim_due("sess-1", message.id) is not None
    assert claim_due("sess-1", message.id) is None


def test_cancelled_message_never_claims(home):
    from hermes_cli.scheduled_messages import cancel_message, claim_due, due_items, list_messages

    message = _schedule(text="do not send", in_seconds=-1)
    assert cancel_message("sess-1", message.id) is True

    assert list_messages("sess-1") == [] and due_items("sess-1") == []
    assert claim_due("sess-1", message.id) is None
    assert cancel_message("sess-1", message.id) is False, "cancel is not idempotent-on-fire: it reports"


def test_abandon_fire_puts_a_failed_dispatch_back(home):
    """A turn that never started must not consume the message (mirrors heartbeat.abandon_fire)."""
    from hermes_cli.scheduled_messages import abandon_fire, claim_due, due_items

    message = _schedule(text="retry me", in_seconds=-1)
    claim_due("sess-1", message.id)
    assert abandon_fire("sess-1", message.id) is True

    assert [item.id for item in due_items("sess-1")] == [message.id]


def test_survives_a_restart(home):
    """Restart = a new process reading the same state.db: the bytes must be on disk, keyed by the
    session, with the due time intact."""
    from hermes_cli.scheduled_messages import list_messages, schedule_message
    from hermes_state import SessionDB

    message = schedule_message("sess-restart", "survive me", time.time() + 120)

    db = SessionDB(db_path=home / "state.db", read_only=True)
    try:
        raw = db.get_meta("scheduled_message:sess-restart")
    finally:
        db.close()
    assert raw and message.id in raw and "survive me" in raw

    from hermes_cli import goals

    goals._DB_CACHE.clear()                                # drop the cached handle: fresh connection
    assert [item["id"] for item in list_messages("sess-restart")] == [message.id]


def test_sessions_do_not_see_each_others_messages(home):
    from hermes_cli.scheduled_messages import list_messages

    _schedule("sess-a", text="for a")
    _schedule("sess-b", text="for b")

    assert [item["text"] for item in list_messages("sess-a")] == ["for a"]
    assert [item["text"] for item in list_messages("sess-b")] == ["for b"]


@pytest.mark.parametrize("text,due_delta,expected", [
    ("", 60.0, "text is required"),
    ("   ", 60.0, "text is required"),
    ("hello", -3600.0, "due time is in the past"),
    ("x" * 8001, 60.0, "longer than"),
])
def test_validation_rejects_payloads_the_ui_should_have_caught(home, text, due_delta, expected):
    from hermes_cli.scheduled_messages import schedule_message

    with pytest.raises(ValueError, match=expected):
        schedule_message("sess-1", text, time.time() + due_delta)


def test_pending_cap_is_enforced(home):
    from hermes_cli.scheduled_messages import MAX_PENDING, schedule_message

    for index in range(MAX_PENDING):
        schedule_message("sess-1", f"message {index}", time.time() + 60)

    with pytest.raises(ValueError, match="too many pending"):
        schedule_message("sess-1", "one too many", time.time() + 60)
