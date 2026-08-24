"""Cron admission must not become a second writer or a completed-delivery claim."""
from pathlib import Path
from unittest.mock import Mock

from cron import scheduler_delivery as delivery
from tools import bot_live_delivery as mailbox


def test_live_delivery_retry_keeps_receipt_across_owner_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = tmp_path / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(source))
    subprocess_run = Mock(side_effect=AssertionError("live owner must not spawn CLI"))
    monkeypatch.setattr(delivery.subprocess, "run", subprocess_run)
    from hermes_cli.profiles import get_profile_dir

    for profile, home in [("", source), ("research", get_profile_dir("research"))]:
        owner = dict(profile_home=str(home.resolve()), session_id="bot", lease_id="lease",
                     live_session_id="live")
        discovery = Mock(return_value=owner)
        monkeypatch.setattr(mailbox, "find_canonical_live_owner", discovery)
        job = dict(id="digest", name="Digest", execution_id="first-run")
        pending = delivery._deliver_to_bot_chat(job, "payload", profile)
        assert pending and "queued" in pending
        records = list((home / "runtime/bot_live_delivery").glob("*.json"))
        assert len(records) == 1
        key = records[0].stem
        record = mailbox.read_delivery_result(home, key)
        assert record and record["message"].endswith("payload")
        discovery.side_effect = AssertionError("receipt must precede discovery")
        assert delivery._deliver_to_bot_chat(dict(job), "payload", profile) == pending
        mailbox.claim_pending_delivery(home, owner)
        mailbox.complete_delivery(home, key, status="ambiguous", error="owner died")
        outcome = delivery._deliver_to_bot_chat(dict(job), "payload", profile)
        assert outcome and "ambiguous" in outcome
        discovery.side_effect = None
        next_job = dict(job, execution_id="next-run")
        outcome = delivery._deliver_to_bot_chat(next_job, "payload", profile)
        assert outcome and "queued" in outcome
        assert len(list((home / "runtime/bot_live_delivery").glob("*.json"))) == 2
    subprocess_run.assert_not_called()


def test_result_records_pending_until_terminal_receipt(tmp_path, monkeypatch):
    from cron import jobs
    from gateway import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("_HERMES_CRON_EXTERNAL_WORKER", raising=False)
    owner = dict(profile_home=str(tmp_path.resolve()), session_id="bot", lease_id="lease",
                 live_session_id="live")
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)
    monkeypatch.setattr(delivery._sched, "load_config", lambda: {})
    monkeypatch.setattr(config, "load_gateway_config", lambda: None)
    monkeypatch.setattr(delivery.subprocess, "run", Mock(side_effect=AssertionError("CLI")))
    updates = []
    monkeypatch.setattr(jobs, "update_job", lambda key, values: updates.append(values))
    job = dict(id="digest", execution_id="run", deliver="bot-chat")
    error = delivery._deliver_result(job, "payload")
    assert error is None
    queued = updates[-1]["last_delivery_queued"]
    assert queued and next(iter(queued.values()))["status"] == "queued"
    assert delivery._sched._classify_delivery_outcome(
        delivery_error=error, delivery_queued=queued, should_deliver=True, unresolved_origin=False,
        normalized_deliver="bot-chat", incident_acked=False, success=True) == "queued"
    record = mailbox.claim_pending_delivery(tmp_path, owner)
    assert record is not None
    mailbox.complete_delivery(tmp_path, record["delivery_id"], status="settled", reply="done")
    assert delivery._deliver_result(job, "payload") is None
    assert updates[-1]["last_delivery_queued"] is None
