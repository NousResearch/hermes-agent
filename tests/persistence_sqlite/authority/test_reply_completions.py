"""Durable reply order follows commits, including detached reports and profile isolation."""

import pytest

from persistence.conversation.sessions import NewSession
from persistence.core.errors import PermanentStoreError
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch


def _claim(store, epoch, session_id):
    store.sessions.create(NewSession(session_id=session_id, source="test"))
    store.admissions.admit(epoch=epoch, principal_id="person", session_id=session_id,
                           request_id="prompt", payload={"text": "Question"})
    return store.admissions.claim(epoch=epoch, session_id=session_id)


def _settle(store, epoch, admission, *, outcome="completed", response="A private reply"):
    store.admissions.settle(
        epoch=epoch, admission_id=admission.admission_id, generation=admission.generation,
        outcome=outcome, result={"result": {"final_response": response, "messages": []}},
    )


def test_reply_events_follow_completion_order_across_restart(backend, store, scope_b, clock):
    epoch = begin_runtime_epoch(store.holder(), instance_id="owner")
    slow = _claim(store, epoch, "slow")
    fast = _claim(store, epoch, "fast")
    failed = _claim(store, epoch, "failed")
    empty = _claim(store, epoch, "empty")

    _settle(store, epoch, fast)
    first, = store.admissions.reply_completions(after=0, limit=1)
    assert (first.session_id, first.admission_id, first.kind) == ("fast", fast.admission_id, "turn")
    _settle(store, epoch, failed, outcome="failed")
    _settle(store, epoch, empty, response=" ")
    _settle(store, epoch, slow)

    store.sessions.create(NewSession(session_id="reports", source="test"))
    metadata = {"completion_note": ["cron:job:run"]}
    note = store.messages.append_completion_note("reports", "report", "Private report", "internal_notification", metadata)
    assert store.messages.append_completion_note("reports", "report", "Private report", "internal_notification", metadata) == note
    store.messages.append_completion_note("reports", "diagnostic", "Private diagnostic", "internal_notification",
                                          {**metadata, "notification_category": "diagnostic"})

    other = backend.open(scope_b)
    other_epoch = begin_runtime_epoch(other.holder(), instance_id="other")
    _settle(other, other_epoch, _claim(other, other_epoch, "fast"))
    backend.reopen()
    remaining = store.admissions.reply_completions(after=first.sequence, limit=100)
    assert [(row.session_id, row.kind) for row in remaining] == [("slow", "turn"), ("reports", "note")]
    assert remaining[0].sequence < remaining[1].sequence
    assert "Private" not in repr([first, *remaining])
    assert [row.session_id for row in other.admissions.reply_completions(after=0, limit=100)] == ["fast"]

    clock.now += 25 * 3600
    assert store.admissions.reply_completions(after=0, limit=100) == []
    next_epoch = begin_runtime_epoch(store.holder(), instance_id="replacement")
    _settle(store, next_epoch, _claim(store, next_epoch, "new"))
    newest, = store.admissions.reply_completions(after=remaining[-1].sequence, limit=1)
    assert newest.session_id == "new"


def test_reply_event_failure_rolls_back_the_terminal_result(store):
    epoch = begin_runtime_epoch(store.holder(), instance_id="owner")
    claimed = _claim(store, epoch, "session")
    holder = store.holder()
    holder._conn.execute(
        "CREATE TRIGGER refuse_reply_event BEFORE INSERT ON session_reply_completions "
        "BEGIN SELECT RAISE(ABORT, 'reply fixture'); END"
    )
    with pytest.raises(PermanentStoreError, match="reply fixture"):
        _settle(store, epoch, claimed)
    assert store.admissions.get(admission_id=claimed.admission_id).status == "started"
    assert store.admissions.settled_result(admission_id=claimed.admission_id) is None
    assert store.admissions.reply_completions(after=0, limit=100) == []

    holder._conn.execute("DROP TRIGGER refuse_reply_event")
    _settle(store, epoch, claimed)
    event, = store.admissions.reply_completions(after=0, limit=100)
    assert event.admission_id == claimed.admission_id
