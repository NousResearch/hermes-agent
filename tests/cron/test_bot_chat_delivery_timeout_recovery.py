"""A raise AFTER the RPC left the socket must not read as a delivery failure.

The producer computes a deterministic delivery id (sha256 over source home, job id, run
id, target home) and reads that receipt BEFORE dialing. The authority commits the same
receipt when it ADMITS the turn, which can be minutes after a 30 s ``rpc`` budget lapses,
so the exception branch is a three-way decision, not a binary one:

* a record for THIS payload → its status is the disposition (``settled`` → delivered,
  ``queued``/``claimed`` → completion unverified, ``ambiguous`` → the detail branch);
* a record for ANOTHER payload → the id belongs to something else: a definite failure;
* no record after the grace → ``unknown``: the caller keeps that out of ``delivery_errors``
  and books it as unverified, because the payload may still be queued in front of the owner;
* a raise from before the dial (or the helper's own ``PreDialError``) → a definite failure.

The cause string must never publish as an empty tail: ``str(TimeoutError())`` is ``''`` on
CPython, so ``f"... unverified: {exc}"`` used to name no cause at all.
"""
import threading
import time

import pytest

GRACE = 0.05


def _install_authority(monkeypatch, mailbox, *, write_receipt=True, status="queued",
                       message=None, raises=None, category=None, write_after=None):
    """Stand in for the profile authority: optionally admit the turn, then lose the answer."""

    def fake_deliver(home, owner, message_, *, delivery_id, notification_category="result"):
        record = {
            "delivery_id": delivery_id,
            "id": delivery_id,
            "status": status,
            "message": message_ if message is None else message,
            "created_at": 1,
            "sequence": 1,
        }
        if category is not None:
            record["notification_category"] = category

        def commit():
            if write_after:
                time.sleep(write_after)
            with mailbox._locked(home) as root:
                mailbox._write(root / f"{delivery_id}.json", record)

        if write_receipt:
            if write_after:
                # A late admit: the raise happens now, the authority commits the receipt a moment
                # later — the shape of the live incident (attempt 00:23:17, owner turn 00:24:50).
                threading.Thread(target=commit, daemon=True).start()
            else:
                commit()
        raise raises if raises is not None else TimeoutError()  # str(TimeoutError()) == ""

    monkeypatch.setattr(mailbox, "find_canonical_live_owner",
                        lambda home: {"session_id": "bot"})
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", fake_deliver)
    # The grace is a wall-clock wait: pin it so the corpus runs in milliseconds. Tolerant of its
    # absence ON PURPOSE: run against pre-fix HEAD the corpus must fail on the BEHAVIOUR it asserts,
    # never on a missing fixture attribute, or the polarity receipt proves only that the test
    # harness knows the new symbol.
    from cron import scheduler_delivery
    if hasattr(scheduler_delivery, "BOT_CHAT_ADMISSION_GRACE_SECONDS"):
        monkeypatch.setattr(scheduler_delivery, "BOT_CHAT_ADMISSION_GRACE_SECONDS", GRACE)


def _deliver(job):
    from cron.scheduler_delivery import _deliver_to_bot_chat
    return _deliver_to_bot_chat(job, "the payload", "")


def _receipts(job) -> dict:
    return job.get("_bot_chat_delivery_receipts") or {}


def test_timeout_after_admission_is_not_a_delivery_failure(tmp_path, monkeypatch):
    """Admission survives a lost answer: the job carries a queued receipt, and the caller's
    ``delivery_errors`` rule therefore keeps it out of ``delivery_failed``."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox)

    job = {"id": "j-timeout", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert outcome is not None, "an unverified admission is not a settled reply"
    assert _receipts(job)["bot-chat:(own)"]["status"] == "queued"
    assert "unverified: " not in outcome, "the empty-cause tail must never ship"
    assert "completion unverified; do not resend" in outcome


def test_timeout_without_a_receipt_is_unknown_not_a_failure(tmp_path, monkeypatch):
    """Nothing on disk after the grace is not proof of non-delivery: the authority may still be
    mid-turn. The job carries ``unknown`` so the caller books it as unverified, not as failed."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, write_receipt=False)

    job = {"id": "j-unknown", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert _receipts(job)["bot-chat:(own)"]["status"] == "unknown"
    assert outcome is not None, "unknown is still an unverified status string, never None"
    assert outcome.startswith("bot-chat:(own) unknown (receipt "), outcome
    assert "admission unconfirmed: TimeoutError" in outcome, "the class must be named"
    assert not outcome.endswith("unverified: "), "an empty tail is never acceptable"


def test_pre_dial_failure_stays_definite(tmp_path, monkeypatch):
    """``PreDialError`` means nothing reached the authority: that is a real delivery failure."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, write_receipt=False,
                       raises=mailbox.PreDialError("no session file lock"))

    job = {"id": "j-predial", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert _receipts(job) == {}, "a definite failure invents no receipt"
    assert outcome is not None
    assert outcome.endswith("unverified: failed before the dial: PreDialError: no session file lock"), outcome


def test_settled_receipt_recovered_after_a_raise_reads_as_delivered(tmp_path, monkeypatch):
    """The owner settled inside the grace: no error at all."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, status="settled")

    job = {"id": "j-settled", "name": "n", "execution_id": "r1"}
    assert _deliver(job) is None
    assert _receipts(job)["bot-chat:(own)"]["status"] == "settled"


def test_a_receipt_for_a_different_payload_is_never_recovered(tmp_path, monkeypatch):
    """The delivery id is deterministic per run; a record holding another payload is a
    different admission and must not launder this failure."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, message="some other payload")

    job = {"id": "j-mismatch", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert _receipts(job) == {}
    assert outcome is not None
    assert "another payload owns this delivery id" in outcome, outcome
    assert "unverified: " in outcome


def test_a_diagnostic_receipt_does_not_admit_a_result_payload(tmp_path, monkeypatch):
    """Same text, different category = different admission. The success path compares both, so
    the recovery must too, or a muted notice would vouch for a result that never went out."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, category="diagnostic")

    job = {"id": "j-category", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert _receipts(job) == {}
    assert outcome is not None
    assert "another payload owns this delivery id" in outcome, outcome


def test_a_receipt_that_appears_late_is_still_recognised(tmp_path, monkeypatch):
    """The owner admits after the raise, not before: the grace must still find that record.

    This is the live shape — a 30 s rpc budget lapsing while the owner is mid-turn. An instant
    re-read (design v1) sees an empty store here and calls a delivered payload a failure.
    """
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, write_after=0.2)
    from cron import scheduler_delivery
    monkeypatch.setattr(scheduler_delivery, "BOT_CHAT_ADMISSION_GRACE_SECONDS", 1.0)

    job = {"id": "j-late", "name": "n", "execution_id": "r1"}
    started = time.monotonic()
    outcome = _deliver(job)
    waited = time.monotonic() - started

    assert _receipts(job)["bot-chat:(own)"]["status"] == "queued", "a late admit is not a failure"
    assert outcome is not None
    assert "completion unverified; do not resend" in outcome
    assert waited >= 0.2, f"the grace returned before the receipt existed ({waited:.2f}s)"


def test_the_grace_waits_instead_of_reading_once(tmp_path, monkeypatch):
    """Nothing on disk yet is PENDING, not terminal: the branch must spend its grace waiting.

    ``await_delivery`` returns None on the first read when no record exists (it serves callers that
    already own a receipt), so using it here made the grace a no-op — measured at 0.00 s by
    scripts/probe-late-admit.py. This case pins the wait itself, not just the verdict.
    """
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, write_receipt=False)
    from cron import scheduler_delivery
    monkeypatch.setattr(scheduler_delivery, "BOT_CHAT_ADMISSION_GRACE_SECONDS", 0.3)

    job = {"id": "j-wait", "name": "n", "execution_id": "r1"}
    started = time.monotonic()
    outcome = _deliver(job)
    waited = time.monotonic() - started

    assert _receipts(job)["bot-chat:(own)"]["status"] == "unknown"
    assert outcome is not None
    assert waited >= 0.25, f"the grace was a no-op ({waited:.2f}s for a 0.3s budget)"


def test_an_unreadable_receipt_is_unknown_not_a_failure(tmp_path, monkeypatch):
    """A receipt the grace could not read is a blind spot, not proof of non-delivery: the class
    of the read error is logged, and the job still reports ``unknown``."""
    from tools import bot_live_delivery as mailbox

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_authority(monkeypatch, mailbox, write_receipt=False)

    calls = {"n": 0}

    def unreadable(home, delivery_id):
        # The branch reads this store once BEFORE dialing too; that read must succeed, or the case
        # tests the pre-dial failure path instead of the blind grace it is about.
        calls["n"] += 1
        if calls["n"] == 1:
            return
        raise OSError("receipt store unreadable")

    monkeypatch.setattr(mailbox, "read_delivery_result", unreadable)

    job = {"id": "j-blind", "name": "n", "execution_id": "r1"}
    outcome = _deliver(job)

    assert _receipts(job)["bot-chat:(own)"]["status"] == "unknown"
    assert outcome is not None
    assert "admission unconfirmed" in outcome, outcome
