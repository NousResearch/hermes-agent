"""Contract cases for the delivery-obligation verbs of ``CoordinationRepository``.

The clock is pinned through ``time.time``; the owner probe and the retry deadlines arrive as data (``dead_owners``, ``due``), so no case needs a dead process.
"""

from __future__ import annotations

import pytest

from persistence.authority.coordination import (
    OBLIGATION_FLOOD_MARKER,
    OBLIGATION_RECONNECTED_MARKER,
    ClaimedObligation,
    FailedObligationRow,
    OwnerStamp,
    RecordedOwner,
    RetryCandidate,
    RetryDue,
    flood_not_before,
    is_flood_error,
)

NOW = 1_800_000_000.0
OLD = OwnerStamp(pid=11, started_at=111)
ME = OwnerStamp(pid=22, started_at=222)
DEAD = frozenset({RecordedOwner(11, 111), RecordedOwner(None, None)})


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("time.time", lambda: now[0])
    return now


def record(coord, oid: str, *, owner: OwnerStamp = OLD, platform: str = "telegram", profile=None, content: str = "reply") -> None:
    coord.record_delivery_obligation(obligation_id=oid, session_key="sk-" + oid, platform=platform, chat_id="42", thread_id=None,
                                     content=content, adapter_profile=profile, owner=owner)


def test_flood_policy_helpers():
    assert is_flood_error("flood_control:12") and is_flood_error("Flood control exceeded. Retry in 7 seconds")
    assert not is_flood_error("send_path_degraded") and not is_flood_error(None)
    assert flood_not_before(100.0, "flood_control:12") == 112.0
    assert flood_not_before(100.0, "flood control exceeded, retry in 3.5") == 103.5
    assert flood_not_before(None, "flood_control:junk") == 60.0


def test_owners_and_failed_rows_are_read_by_exact_stamp(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    record(coord, "o-pending")
    record(coord, "o-mine", owner=ME, profile=" bot ")
    coord.mark_obligation_failed(obligation_id="o-mine", error="x" * 600)
    record(coord, "o-done", owner=OwnerStamp(pid=33, started_at=None))
    coord.mark_obligation_delivered(obligation_id="o-done")
    # CONTRACT: owners of pending, attempting and failed rows, NULLs kept; delivered rows do not count.
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(11, 111), RecordedOwner(22, 222)})
    # CONTRACT: this owner's failed rows; an unknown start time reads nothing; the error is cut to 500 characters; the profile is stripped.
    assert coord.list_owned_failed_obligations(owner=ME) == (FailedObligationRow(platform="telegram", adapter_profile="bot", updated_at=NOW, last_error="x" * 500, attempts=0, created_at=NOW),)
    assert coord.list_owned_failed_obligations(owner=OwnerStamp(pid=22, started_at=None)) == ()
    assert coord.list_runtime_retry_candidates("telegram", profile="bot", owner=ME) == (RetryCandidate(obligation_id="o-mine", updated_at=NOW, last_error="x" * 500, attempts=0),)
    assert coord.list_runtime_retry_candidates("telegram", owner=ME) == ()
    assert coord.list_runtime_retry_candidates("telegram", profile="bot", owner=OwnerStamp(pid=22, started_at=None)) == ()


def test_crash_left_reply_is_unowned_attempting_and_idempotent(backend, scope_a, clock):
    # CONTRACT: inserted unowned and attempting unless the same session key and content was ledgered since ``since``.
    coord = backend.open(scope_a).coordination
    record(coord, "o-ledgered", content="hello")
    coord.record_crash_left_reply(obligation_id="o-crash-1", session_key="sk-o-ledgered", platform="telegram", chat_id="42", thread_id=None, content="hello", since=NOW - 1)
    coord.record_crash_left_reply(obligation_id="o-crash-2", session_key="sk-new", platform="telegram", chat_id="42", thread_id="7", content="hello", since=NOW - 1)
    coord.record_crash_left_reply(obligation_id="o-crash-2", session_key="sk-new", platform="telegram", chat_id="42", thread_id="7", content="other", since=NOW + 1)
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(11, 111), RecordedOwner(None, None)})
    claimed = coord.sweep_recoverable_obligations(owner=ME, dead_owners=frozenset({RecordedOwner(None, None)}))
    assert [(c.obligation_id, c.thread_id, c.content, c.needs_marker) for c in claimed] == [("o-crash-2", "7", "hello", True)]


def test_release_runtime_claim_refunds_only_this_owners_attempting_row(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    record(coord, "o-run", owner=ME)
    coord.mark_obligation_attempting(obligation_id="o-run")
    assert coord.release_runtime_obligation_claim(obligation_id="o-run", error="adapter gone", owner=OwnerStamp(pid=22, started_at=None)) is False
    assert coord.release_runtime_obligation_claim(obligation_id="o-run", error="adapter gone", owner=OLD) is False
    assert coord.release_runtime_obligation_claim(obligation_id="o-run", error="adapter gone", owner=ME) is True
    assert coord.list_owned_failed_obligations(owner=ME)[0].last_error == "adapter gone"
    assert coord.release_runtime_obligation_claim(obligation_id="o-run", owner=ME) is False


def test_recoverable_sweep_claims_dead_owners_rows(backend, scope_a, clock):
    # CONTRACT: ``sweep_recoverable_obligations`` with the probe as data: a live owner's row is skipped; a never-claimed pending row is claimed plainly; others carry the marker; the claim moves to attempting, +1 attempt, stamped to the sweeper.
    coord = backend.open(scope_a).coordination
    record(coord, "o-plain")
    record(coord, "o-failed")
    coord.mark_obligation_failed(obligation_id="o-failed", error="boom")
    record(coord, "o-live", owner=ME)
    claimed = {c.obligation_id: c for c in coord.sweep_recoverable_obligations(owner=ME, dead_owners=DEAD)}
    assert set(claimed) == {"o-plain", "o-failed"}
    assert claimed["o-plain"] == ClaimedObligation(obligation_id="o-plain", session_key="sk-o-plain", platform="telegram", chat_id="42", thread_id=None,
                                                   content="reply", profile="default", attempts=1, needs_marker=False)
    assert (claimed["o-failed"].needs_marker, claimed["o-failed"].marker, claimed["o-failed"].attempts) == (True, None, 1)
    assert coord.list_obligation_owners() == frozenset({RecordedOwner(22, 222)})
    # The claimed rows are attempting under the sweeper now: a second sweep by another process sees a live owner.
    assert coord.sweep_recoverable_obligations(owner=OLD, dead_owners=frozenset({RecordedOwner(11, 111)})) == ()


def test_recoverable_sweep_abandons_exhausted_and_filters_targets(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    record(coord, "o-stale")
    clock[0] += 24 * 3600 + 1
    record(coord, "o-slack", platform="slack")
    record(coord, "o-bot", profile="bot")
    # CONTRACT: rows older than 24 h are abandoned (not claimed); the platform and (platform, profile) filters skip without spending an attempt.
    assert coord.sweep_recoverable_obligations(owner=ME, dead_owners=DEAD, deliverable_platforms=frozenset({"telegram"}),
                                               deliverable_targets=frozenset({("telegram", "default")})) == ()
    assert RecordedOwner(11, 111) in coord.list_obligation_owners()
    claimed = coord.sweep_recoverable_obligations(owner=ME, dead_owners=DEAD)
    assert sorted(c.obligation_id for c in claimed) == ["o-bot", "o-slack"]
    assert [c.profile for c in claimed if c.obligation_id == "o-bot"] == ["bot"]


def test_recoverable_sweep_abandons_spent_budgets(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    record(coord, "o-spent")
    for _ in range(3):
        coord.mark_obligation_failed(obligation_id="o-spent", error="boom")
        (claimed,) = coord.sweep_recoverable_obligations(owner=OLD, dead_owners=frozenset({RecordedOwner(11, 111)}))
        assert claimed.obligation_id == "o-spent"
    coord.mark_obligation_failed(obligation_id="o-spent", error="boom")
    assert coord.sweep_recoverable_obligations(owner=ME, dead_owners=DEAD) == ()
    assert coord.list_obligation_owners() == frozenset()


def test_recoverable_sweep_adopts_a_flood_row_inside_its_wait(backend, scope_a, clock):
    # CONTRACT: a flood-refused row inside its wait is adopted (owner re-stamped, no attempt spent) and returned with its deadline; past the wait it is claimed with the flood marker and its error cleared.
    coord = backend.open(scope_a).coordination
    record(coord, "o-flood")
    coord.mark_obligation_failed(obligation_id="o-flood", error="flood_control:30")
    (adopted,) = coord.sweep_recoverable_obligations(owner=ME, dead_owners=DEAD)
    assert (adopted.adopted, adopted.not_before, adopted.attempts, adopted.needs_marker) == (True, NOW + 30, 0, False)
    assert coord.list_owned_failed_obligations(owner=ME)[0].last_error == "flood_control:30"
    clock[0] += 31
    (claimed,) = coord.sweep_recoverable_obligations(owner=OLD, dead_owners=frozenset({RecordedOwner(22, 222)}))
    assert (claimed.adopted, claimed.marker, claimed.attempts) == (False, OBLIGATION_FLOOD_MARKER, 1)


def test_runtime_sweep_claims_due_unchanged_rows_of_this_owner(backend, scope_a, clock):
    # CONTRACT: ``sweep_failed_obligations_for_runtime`` over caller-computed deadlines: an unknown start time returns (); each row must still match its candidate; not yet due is skipped; claims carry the reconnect (or flood) marker and the pre-claim error.
    coord = backend.open(scope_a).coordination
    for oid, error in (("o-due", "send_path_degraded"), ("o-later", "boom"), ("o-flood", "flood_control:5"), ("o-moved", "boom")):
        record(coord, oid, owner=ME)
        coord.mark_obligation_failed(obligation_id=oid, error=error)
    candidates = {c.obligation_id: c for c in coord.list_runtime_retry_candidates("telegram", owner=ME)}
    due = [RetryDue(obligation_id=oid, updated_at=c.updated_at, last_error=c.last_error, attempts=c.attempts,
                    not_before=NOW + (60 if oid == "o-later" else 0)) for oid, c in candidates.items()]
    clock[0] += 1
    coord.mark_obligation_failed(obligation_id="o-moved", error="boom")
    assert coord.sweep_failed_obligations_for_runtime("telegram", owner=OwnerStamp(pid=22, started_at=None), due=due) == ()
    claimed = {c.obligation_id: c for c in coord.sweep_failed_obligations_for_runtime("telegram", owner=ME, due=due)}
    assert set(claimed) == {"o-due", "o-flood"}
    assert (claimed["o-due"].marker, claimed["o-due"].runtime_recovery, claimed["o-due"].last_error, claimed["o-due"].attempts) == (
        OBLIGATION_RECONNECTED_MARKER, True, "send_path_degraded", 1)
    assert claimed["o-flood"].marker == OBLIGATION_FLOOD_MARKER
    assert {row.last_error for row in coord.list_owned_failed_obligations(owner=ME)} == {"boom"}
