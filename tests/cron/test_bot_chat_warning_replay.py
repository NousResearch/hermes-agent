"""Warning policy cannot change the identity or outcome of a Bot Chat handoff."""
import json
from unittest.mock import Mock

import pytest

from cron import bot_chat_delivery as pending
from cron import scheduler_delivery as delivery
from hermes_cli.active_sessions import try_acquire_active_session
from hermes_state import SessionDB
from tools import bot_live_delivery as mailbox


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="chat", source="tui")
    db.set_session_title("chat", "Bot Chat")
    leases = []
    run = Mock(side_effect=AssertionError("must not launch a second owner"))
    monkeypatch.setattr(delivery, "_run_bot_chat_turn", run)

    def acquire(live):
        lease, refusal = try_acquire_active_session(
            session_id="chat", surface="desktop" if live else "cli", config={},
            registry_home=tmp_path,
            metadata={"bot_live_delivery_consumer": True, "live_session_id": "live"} if live else {},
        )
        assert refusal is None
        leases.append(lease)
        return lease

    yield tmp_path, acquire, run
    for lease in leases:
        lease.release()
    db.close()


def policy(home, suppress):
    (home / "config.yaml").write_text(
        f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}\n"
    )


def receipt_key(job):
    return job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["delivery_id"]


@pytest.mark.parametrize("live,first_failure,suppress_retry", [
    (False, True, True), (True, False, True), (True, True, False),
])
def test_admitted_identity_rejects_category_changing_retry(owner, live, first_failure, suppress_retry):
    home, acquire, run = owner
    acquire(live)
    job = {"id": "job", "execution_id": "run"}
    assert "queued" in delivery._deliver_to_bot_chat(job, "same payload", "", for_failure=first_failure)
    key = receipt_key(job)
    before = mailbox.read_delivery_result(home, key) if live else pending.read_pending(key)
    policy(home, suppress_retry)
    result = delivery._deliver_to_bot_chat(job, "same payload", "", for_failure=not first_failure)
    assert result and "different payload" in result
    assert not job.get("_notification_all_targets_suppressed")
    after = mailbox.read_delivery_result(home, key) if live else pending.read_pending(key)
    assert after == before
    run.assert_not_called()


@pytest.mark.parametrize("live,status", [
    (False, "queued"), (False, "settled"), (True, "claimed"), (True, "ambiguous"),
])
def test_admitted_outcome_survives_suppressed_retry(owner, live, status):
    home, acquire, run = owner
    acquire(live)
    job = {"id": "job", "execution_id": "run"}
    assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
    key = receipt_key(job)
    if live:
        if status != "queued":
            mailbox.claim_pending_delivery(home, mailbox.find_canonical_live_owner(home))
        if status not in ("queued", "claimed"):
            mailbox.complete_delivery(home, key, status=status, reply="retained result")
        before = mailbox.read_delivery_result(home, key)
    else:
        path = home / "cron" / "bot_chat_pending" / f"{key}.json"
        record = pending.read_pending(key)
        record["status"] = status
        path.write_text(json.dumps(record))
        before = pending.read_pending(key)
    policy(home, True)
    result = delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
    if not live and status == "queued":
        assert result is None
        assert pending.read_pending(key)["status"] == "suppressed"
        assert job.get("_notification_all_targets_suppressed")
    else:
        assert result is None if status == "settled" else status in result
        assert not job.get("_notification_all_targets_suppressed")
        assert (mailbox.read_delivery_result(home, key) if live else pending.read_pending(key)) == before
    run.assert_not_called()


@pytest.mark.parametrize("live", [False, True])
def test_suppressed_admission_is_durable_across_policy_changes(owner, live):
    home, acquire, run = owner
    acquire(live)
    policy(home, True)
    job = {"id": "job", "execution_id": "run"}
    assert delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True) is None
    assert job.get("_notification_all_targets_suppressed")
    policy(home, False)
    assert delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True) is None
    assert job.get("_notification_all_targets_suppressed")
    key = receipt_key(job)
    assert pending.read_pending(key)["status"] == "suppressed"
    assert pending.read_pending(key)["content"] == "diagnostic"
    assert mailbox.read_delivery_result(home, key) is None
    pending.drain()
    run.assert_not_called()


@pytest.mark.parametrize("owner_suppressed", [False, True])
def test_named_bot_chat_uses_recipient_policy_not_launch_home(owner, monkeypatch, owner_suppressed):
    home, acquire, run = owner
    acquire(True)
    launch = home / "launch"
    launch.mkdir()
    policy(home, owner_suppressed)
    policy(launch, not owner_suppressed)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: home)
    job = {"id": "job", "execution_id": "recipient-run"}
    result = delivery._deliver_to_bot_chat(job, "diagnostic", "recipient", for_failure=True)
    if owner_suppressed:
        assert result is None and job.get("_notification_all_targets_suppressed")
    else:
        assert result and "queued" in result
        key = job["_bot_chat_delivery_receipts"]["bot-chat:recipient"]["delivery_id"]
        assert mailbox.read_delivery_result(home, key)["notification_category"] == "diagnostic"
        assert mailbox.read_delivery_result(launch, key) is None
    # Return to the launch home's own policy, proving recipient lookup did not leak.
    own = {"id": "own", "execution_id": "own-run"}
    if not owner_suppressed:
        assert delivery._deliver_to_bot_chat(own, "diagnostic", "", for_failure=True) is None
        assert own.get("_notification_all_targets_suppressed")
    run.assert_not_called()


def test_deferred_diagnostic_transfers_category_to_real_live_mailbox(owner):
    home, acquire, run = owner
    cli = acquire(False)
    job = {"id": "job", "execution_id": "run"}
    assert "queued" in delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True)
    key = receipt_key(job)
    cli.release()
    acquire(True)
    pending.drain()
    assert pending.read_pending(key)["status"] == "transferred"
    record = mailbox.claim_pending_delivery(home, mailbox.find_canonical_live_owner(home))
    assert record["delivery_id"] == key
    assert record["notification_category"] == "diagnostic"
    assert "diagnostic" in record["message"]
    mailbox.complete_delivery(home, key, status="settled", reply="handled")
    policy(home, True)
    assert delivery._deliver_to_bot_chat(job, "diagnostic", "", for_failure=True) is None
    assert not job.get("_notification_all_targets_suppressed")
    pending.drain()
    assert mailbox.claim_pending_delivery(home, mailbox.find_canonical_live_owner(home)) is None
    run.assert_not_called()
