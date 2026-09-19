"""Behaviour contracts for the Hindsight retain journal (outbox.py).

See #88944: buffered retain turns lived only in an in-memory queue.Queue /
list, so a crash between "enqueue" and "sent" silently dropped them. These
tests exercise the durable journal in isolation, in-process; the actual
crash-survival claim (a killed OS process, a fresh process recovering its
work) is proven separately in test_hindsight_journal_processes.py.
"""
import pytest

from plugins.memory.hindsight.outbox import Outbox


@pytest.fixture
def outbox(tmp_path):
    return Outbox(tmp_path, "partition-a")


def test_put_survives_as_pending_before_any_claim(outbox):
    """A row that was journaled but never picked up by a writer must still be
    reported as durable work — this is the core "enqueue then die" case."""
    identity = outbox.put({"item": {"content": "hello"}}, document_id="doc-1")

    pending = outbox.pending()

    assert [row["id"] for row in pending] == [identity]
    assert pending[0]["state"] == "queued"


def test_claim_then_finish_done_is_no_longer_pending(outbox):
    identity = outbox.put({"item": {"content": "hello"}}, document_id="doc-1")

    payload = outbox.claim(identity)
    assert payload == {"item": {"content": "hello"}}
    outbox.finish(identity, "done")

    assert outbox.pending() == []


def test_claim_is_exclusive(outbox):
    """Two claimers racing for the same row (two writer threads, or a recovery
    pass overlapping a live writer) must not both get to send it."""
    identity = outbox.put({"item": {}}, document_id="doc-1")

    first = outbox.claim(identity)
    second = outbox.claim(identity)

    assert first is not None
    assert second is None


def test_finish_after_done_cannot_be_undone(outbox):
    """A late/duplicate finish() (e.g. a stray recovery re-processing an already
    acknowledged row) must not resurrect or reclassify confirmed work."""
    identity = outbox.put({"item": {}}, document_id="doc-1")
    outbox.claim(identity)
    outbox.finish(identity, "done")

    outbox.finish(identity, "quarantined")

    assert outbox.pending() == []


def test_recover_requeues_a_row_stuck_mid_send(outbox):
    """This is the crash scenario: claim() ran (state='sending') but finish()
    never did, because the process died in between. recover() must hand that
    row back out as queued so a fresh writer can retry it."""
    identity = outbox.put({"item": {"content": "x"}}, document_id="doc-1")
    outbox.claim(identity)  # simulates the pre-crash writer; no finish() follows

    requeued = outbox.recover()

    assert requeued == [identity]
    assert outbox.pending()[0]["state"] == "queued"
    # The payload itself must have survived the crash intact.
    assert outbox.claim(identity) == {"item": {"content": "x"}}


def test_recover_does_not_touch_already_queued_or_done_rows(outbox):
    queued_id = outbox.put({"item": {}}, document_id="doc-1")
    done_id = outbox.put({"item": {}}, document_id="doc-1")
    outbox.claim(done_id)
    outbox.finish(done_id, "done")

    requeued = outbox.recover()

    assert requeued == [queued_id]


def test_quarantine_is_not_auto_resubmitted_by_recover(outbox):
    """A row that failed with an ambiguous/explicit error is journaled as
    quarantined. recover() (an ordinary startup) must NOT put it back to work —
    the whole point of quarantine is that the server may already have this
    data, so silently resending it on every restart is unsafe."""
    identity = outbox.put({"item": {}}, document_id="doc-1")
    outbox.claim(identity)
    outbox.finish(identity, "quarantined")

    requeued = outbox.recover()

    assert requeued == []
    assert outbox.pending()[0]["state"] == "quarantined"


def test_requeue_moves_quarantined_row_back_to_queued(outbox):
    """Regression for the production incident described in the task: 63 rows
    stuck in quarantine forever because the reference implementation had no
    way out. requeue() gives an explicit path back to 'queued'."""
    identity = outbox.put({"item": {}}, document_id="doc-1")
    outbox.claim(identity)
    outbox.finish(identity, "quarantined")

    assert outbox.requeue(identity) is True

    assert outbox.pending()[0]["state"] == "queued"


def test_requeue_refuses_once_attempts_exhausted(outbox):
    """The escape hatch from quarantine is bounded: once MAX_ATTEMPTS sends
    have genuinely been made, requeue() refuses instead of allowing a silent
    retry loop forever."""
    identity = outbox.put({"item": {}}, document_id="doc-1")
    for _ in range(Outbox.MAX_ATTEMPTS):
        outbox.claim(identity)
        outbox.finish(identity, "quarantined")
        outbox.requeue(identity)

    assert outbox.requeue(identity) is False
    assert outbox.pending()[0]["state"] == "quarantined"


def test_requeue_refuses_unknown_or_non_quarantined_row(outbox):
    identity = outbox.put({"item": {}}, document_id="doc-1")  # still 'queued'

    assert outbox.requeue(identity) is False
    assert outbox.requeue("not-a-real-id") is False


def test_unknown_schema_version_is_rejected(tmp_path):
    """A journal file written by an unrelated/future schema must never be
    silently adopted (would corrupt or lose data)."""
    outbox = Outbox(tmp_path, "partition-b")
    import sqlite3
    db = sqlite3.connect(outbox.path)
    db.execute("PRAGMA user_version=99")
    db.commit()
    db.close()

    with pytest.raises(RuntimeError):
        Outbox(tmp_path, "partition-b")


def test_journal_file_is_private(tmp_path):
    """Retained conversation content is sensitive; the journal must not be
    world/group readable."""
    import stat
    outbox = Outbox(tmp_path, "partition-c")

    mode = stat.S_IMODE(outbox.path.stat().st_mode)

    assert mode == 0o600
