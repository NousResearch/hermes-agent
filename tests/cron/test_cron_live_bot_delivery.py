"""Cron admission must not become a second writer or a completed-delivery claim."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

from cron import scheduler_delivery as delivery
from tools import bot_live_delivery as mailbox


class _FakeAuthority:
    """The profile authority's canonical deliver door: one admission per stable
    id, exact retries inspect the committed state, the owner may settle it."""

    def __init__(self):
        self.records = {}

    def __call__(self, home, params):
        record = self.records.get(params["id"])
        if record is None:
            record = self.records[params["id"]] = dict(
                status="queued", delivery_id=params["id"], profile_home=str(Path(home).resolve()),
                session_id="local-bot", message=params["message"], admission_id="adm-" + params["id"][:8],
                reply="", error="", reason="")
            with mailbox._locked(home) as root:
                mailbox._write(root / f"{params['id']}.json", record)
        elif record["message"] != params["message"]:
            raise ValueError("admission_conflict")
        return dict(record)


def _owner(home):
    return dict(profile_home=str(Path(home).resolve()), session_id="local-bot", canonical=True,
                lease_id="authority-1", live_session_id="local-bot")


def test_live_delivery_retry_keeps_receipt_across_owner_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = tmp_path / "custom-home"
    monkeypatch.setenv("HERMES_HOME", str(source))
    subprocess_run = Mock(side_effect=AssertionError("live owner must not spawn CLI"))
    monkeypatch.setattr(subprocess, "run", subprocess_run)
    from hermes_cli.profiles import get_profile_dir

    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    for profile, home in [("", source), ("research", get_profile_dir("research"))]:
        discovery = Mock(return_value=_owner(home))
        monkeypatch.setattr(mailbox, "find_canonical_live_owner", discovery)
        job = dict(id="digest", name="Digest", execution_id="first-run")
        admitted_before = len(authority.records)
        pending = delivery._deliver_to_bot_chat(job, "payload", profile)
        assert pending and "queued" in pending
        records = list((home / "runtime/bot_live_delivery").glob("*.json"))
        assert len(records) == 1
        key = records[0].stem
        record = mailbox.read_delivery_result(home, key)
        assert record and record["message"].endswith("payload")
        # Same execution id -> same stable delivery id, no second admission.
        discovery.side_effect = AssertionError("receipt must precede discovery")
        assert delivery._deliver_to_bot_chat(dict(job), "payload", profile) == pending
        assert key in authority.records and len(authority.records) == admitted_before + 1
        # The owner dies mid-execution: the authority reports the admission ambiguous.
        authority.records[key].update(status="ambiguous", error="owner died")
        outcome = delivery._deliver_to_bot_chat(dict(job), "payload", profile)
        assert outcome and "ambiguous" in outcome and "owner died" in outcome
        # Only a NEW execution mints a new id.
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
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: _owner(tmp_path))
    authority = _FakeAuthority()
    monkeypatch.setattr(mailbox, "authority_delivery", authority)
    monkeypatch.setattr(delivery._sched, "load_config", lambda: {})
    monkeypatch.setattr(config, "load_gateway_config", lambda: None)
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("CLI")))
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
    (key,) = authority.records
    assert next(iter(queued.values()))["delivery_id"] == key
    authority.records[key].update(status="settled", reply="done")
    assert delivery._deliver_result(job, "payload") is None
    assert updates[-1]["last_delivery_queued"] is None

