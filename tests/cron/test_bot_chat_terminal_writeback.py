"""Read-only transport reconciliation against isolated stores; never dispatch a job."""
import hashlib
import json
from unittest.mock import Mock

import pytest

from cron import executions, jobs, delivery_outcome
from cron.scheduler_delivery import bot_chat_message
from tools import bot_live_delivery as mailbox


@pytest.fixture
def pending(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron/jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron/output")
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron/executions.db")
    from cron import delivery_queue
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", home / "cron/deliveries.db")
    send = Mock(side_effect=AssertionError("reconciliation must never send"))
    monkeypatch.setattr(mailbox, "deliver_to_live_owner", send)
    monkeypatch.setattr(mailbox, "authority_delivery", send)
    monkeypatch.setattr(mailbox, "find_canonical_live_owner", send)
    job = dict(id="sample-job", name="Scheduled report", no_agent=True, enabled=True,
               deliver="bot-chat", last_status="delivery_queued")
    run = executions.create_execution(job["id"], source="test")
    executions.finish_execution(run["id"], success=True, delivery_outcome="queued")
    key = hashlib.sha256(json.dumps([str(home.resolve()), job["id"], run["id"],
                                    str(home.resolve())], separators=(",", ":")).encode()).hexdigest()
    message = bot_chat_message(job, "Synthetic report body")
    queued = dict(status="queued", delivery_id=key, execution_id=run["id"],
                  profile_home=str(home.resolve()), notification_category="result",
                  message_sha256=hashlib.sha256(message.encode()).hexdigest())
    job["last_delivery_queued"] = {"bot-chat:(own)": queued}
    jobs.save_jobs([job])
    record = dict(delivery_id=key, profile_home=str(home.resolve()), admission_id="admission",
                  session_id="bot", message=message, status="settled", reply="done")
    root = home / "runtime/bot_live_delivery"
    root.mkdir(parents=True)
    path = root / f"{key}.json"
    path.write_text(json.dumps(record))
    yield home, job, run, record, path
    send.assert_not_called()


def test_settled_no_agent_receipt_updates_both_stores_idempotently(pending):
    _, job, run, _, _ = pending
    assert delivery_outcome.settle_queued_delivery(job["id"], run["id"])
    assert executions.get_execution(run["id"])["delivery_outcome"] == "delivered"
    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "ok"
    assert saved.get("last_delivery_error") is None
    assert not saved.get("last_delivery_queued")
    delivery_outcome.settle_queued_delivery(job["id"], run["id"])
    assert jobs.get_job(job["id"])["last_status"] == "ok"


def test_producer_pins_envelope_for_later_receipt_validation(pending):
    _, job, run, record, path = pending
    record["status"] = "queued"
    path.write_text(json.dumps(record))
    from cron.scheduler_delivery import _deliver_to_bot_chat, _record_delivery_verification
    job["execution_id"] = run["id"]
    assert "queued" in _deliver_to_bot_chat(job, "Synthetic report body", "")
    assert job["_bot_chat_delivery_receipts"]["bot-chat:(own)"]["execution_id"] == run["id"]
    _record_delivery_verification(job, [])
    queued = jobs.get_job(job["id"])["last_delivery_queued"]["bot-chat:(own)"]
    assert queued["execution_id"] == run["id"]
    assert queued["profile_home"] == record["profile_home"]
    assert queued["notification_category"] == "result"
    assert queued["message_sha256"] == hashlib.sha256(record["message"].encode()).hexdigest()


@pytest.mark.parametrize("fault", [None, "wrong_payload", "partial_envelope", "missing_output", "wrong_header"])
def test_legacy_no_agent_receipt_uses_unique_run_output(pending, fault):
    from datetime import datetime
    from cron.scheduler_prompt import _job_doc_header
    home, job, run, _, _ = pending
    job["last_delivery_queued"]["bot-chat:(own)"] = {
        "status": "queued", "delivery_id": job["last_delivery_queued"]["bot-chat:(own)"]["delivery_id"]}
    if fault == "partial_envelope":
        job["last_delivery_queued"]["bot-chat:(own)"]["execution_id"] = "wrong-execution"
    jobs.save_jobs([job])
    stamp = datetime.fromisoformat(executions.get_execution(run["id"])["finished_at"])
    root = home / "cron/output/sample-job"
    root.mkdir(parents=True)
    output = _job_doc_header("Scheduled report", "wrong" if fault == "wrong_header" else "sample-job",
                             stamp.strftime("%Y-%m-%d %H:%M:%S"), "no_agent (script)")
    body = "wrong payload" if fault == "wrong_payload" else "Synthetic report body"
    if fault != "missing_output":
        (root / (stamp.strftime("%Y-%m-%d_%H-%M-%S") + ".md")).write_text(output + "\n---\n\n" + body + "\n")
    assert delivery_outcome.settle_queued_delivery("sample-job", run["id"]) is (fault is None)
    assert jobs.get_job("sample-job")["last_status"] == ("delivery_queued" if fault else "ok")


@pytest.mark.parametrize("status", ["failed", "cancelled", "ambiguous"])
def test_negative_terminal_receipts_surface_without_resend(pending, status):
    _, _, run, record, path = pending
    record.update(status=status, error="owner refused")
    path.write_text(json.dumps(record))
    assert delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert executions.get_execution(run["id"])["delivery_outcome"] == "failed"
    assert jobs.get_job("sample-job")["last_status"] == "delivery_failed"
    assert status in jobs.get_job("sample-job")["last_delivery_error"]


@pytest.mark.parametrize("change", ["missing", "queued", "claimed", "profile", "payload", "id", "category"])
def test_unverified_receipt_leaves_queued(pending, change):
    _, _, run, record, path = pending
    if change == "missing":
        path.unlink()
    else:
        updates = {"queued": {"status": "queued"}, "claimed": {"status": "claimed"},
                   "profile": {"profile_home": "/wrong/profile"}, "payload": {"message": "wrong"},
                   "id": {"delivery_id": "0" * 64}, "category": {"notification_category": "diagnostic"}}
        record.update(updates[change])
        path.write_text(json.dumps(record))
    assert not delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert executions.get_execution(run["id"])["delivery_outcome"] == "queued"
    assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"


def test_old_receipt_cannot_overwrite_newer_running_attempt(pending):
    _, _, run, _, _ = pending
    newer = executions.create_execution("sample-job", source="newer")
    assert delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"
    assert executions.get_execution(newer["id"])["delivery_outcome"] is None


def test_tick_reconciles_after_restart_without_sending(pending, monkeypatch):
    import importlib
    from cron import scheduler_tick, scheduler_authority, scheduler
    _, _, run, _, _ = pending
    importlib.reload(delivery_outcome)
    monkeypatch.setattr(scheduler_authority, "reconcile_pending", lambda **kw: None)
    monkeypatch.setattr("cron.bot_chat_legacy.drain_legacy_pending", lambda: None)
    monkeypatch.setattr(scheduler, "_maybe_reap_dead_owners", lambda: None)
    scheduler_tick._pre_dispatch_housekeeping(scheduler, headless=True)
    assert executions.get_execution(run["id"])["delivery_outcome"] == "delivered"
    assert jobs.get_job("sample-job")["last_status"] == "ok"


def test_settlement_before_bookkeeping_is_not_lost(pending, monkeypatch):
    from types import SimpleNamespace
    from cron import scheduler, scheduler_bookkeeping
    _, job, run, _, _ = pending
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET status='running', delivery_outcome=NULL, finished_at=NULL WHERE id=?", (run["id"],))
    # No terminal ledger yet: an early callback must not mutate the job.
    assert not delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    mark = Mock(return_value=True)
    monkeypatch.setattr(scheduler, "mark_job_run", mark)
    monkeypatch.setattr(scheduler, "self_removal_delivery_allowed", lambda _: False)
    d = SimpleNamespace(job=job, success=True, error=None, delivery_error=None,
                        should_deliver=True, blocked_config=False, unresolved_origin=False,
                        incident_acked=False, failure_incident_id=None)
    assert scheduler_bookkeeping.finish_completed_run(d, None, run["id"])
    assert mark.call_args.kwargs["execution_id"] == run["id"]
    assert executions.get_execution(run["id"])["delivery_outcome"] == "delivered"
    assert jobs.get_job("sample-job")["last_status"] == "ok"


@pytest.mark.parametrize("phase", ["after_ledger", "before_receipt"])
def test_restart_repairs_partial_write_or_later_receipt(pending, phase):
    import importlib
    _, _, run, record, path = pending
    if phase == "after_ledger":
        executions.settle_delivery_outcome(run["id"], "delivered")
    else:
        record["status"] = "claimed"
        path.write_text(json.dumps(record))
        delivery_outcome.reconcile_bot_chat_deliveries()
        assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"
        record["status"] = "settled"
        path.write_text(json.dumps(record))
    importlib.reload(delivery_outcome)
    delivery_outcome.reconcile_bot_chat_deliveries()
    assert jobs.get_job("sample-job")["last_status"] == "ok"
    assert executions.get_execution(run["id"])["delivery_outcome"] == "delivered"


@pytest.mark.parametrize("field,value", [("execution_id", "wrong-run"), ("profile_home", "/wrong"),
                                         ("delivery_id", "0" * 64), ("message_sha256", "0" * 64)])
def test_wrong_producer_identity_is_not_settled(pending, field, value):
    _, job, run, _, _ = pending
    job["last_delivery_queued"]["bot-chat:(own)"][field] = value
    jobs.save_jobs([job])
    assert not delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"


def test_new_fire_claim_blocks_stale_status_write(pending):
    _, job, run, _, _ = pending
    job["canonical_completions"] = [run["id"]]
    job["fire_claim"] = {"by": "new-owner"}
    jobs.save_jobs([job])
    delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"


def test_legacy_missing_output_is_unverified(pending):
    _, job, run, _, _ = pending
    expected = job["last_delivery_queued"]["bot-chat:(own)"]
    job["last_delivery_queued"]["bot-chat:(own)"] = {"status": "queued", "delivery_id": expected["delivery_id"]}
    jobs.save_jobs([job])
    assert not delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert executions.get_execution(run["id"])["delivery_outcome"] == "queued"


def test_envelope_cannot_change_between_validation_and_job_write(pending, monkeypatch):
    _, job, run, _, _ = pending
    job["canonical_completions"] = [run["id"]]
    jobs.save_jobs([job])
    settle = executions.settle_delivery_outcome
    def race(execution_id, outcome):
        result = settle(execution_id, outcome)
        changed = jobs.get_job("sample-job")
        changed["last_delivery_queued"]["bot-chat:(own)"]["message_sha256"] = "0" * 64
        jobs.save_jobs([changed])
        return result
    monkeypatch.setattr(executions, "settle_delivery_outcome", race)
    delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert jobs.get_job("sample-job")["last_status"] == "delivery_queued"


def test_bot_settlement_does_not_clear_another_delivery_failure(pending):
    _, job, run, _, _ = pending
    job.update(last_status="delivery_failed", last_delivery_error="other target failed",
               canonical_completions=[run["id"]])
    jobs.save_jobs([job])
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET delivery_outcome='failed' WHERE id=?", (run["id"],))
    delivery_outcome.reconcile_bot_chat_deliveries()
    assert jobs.get_job("sample-job")["last_delivery_error"] == "other target failed"


def test_multiple_targets_wait_for_all_receipts(pending, monkeypatch):
    home, job, run, record, _ = pending
    other = home.parent / "other-profile"
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: other)
    key = hashlib.sha256(json.dumps([str(home.resolve()), "sample-job", run["id"], str(other.resolve())],
                                   separators=(",", ":")).encode()).hexdigest()
    expected = dict(job["last_delivery_queued"]["bot-chat:(own)"], profile_home=str(other.resolve()), delivery_id=key)
    job["last_delivery_queued"]["bot-chat:other"] = expected
    jobs.save_jobs([job])
    assert not delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    root = other / "runtime/bot_live_delivery"
    root.mkdir(parents=True)
    second = dict(record, delivery_id=key, profile_home=str(other.resolve()), status="ambiguous")
    (root / f"{key}.json").write_text(json.dumps(second))
    assert delivery_outcome.settle_queued_delivery("sample-job", run["id"])
    assert jobs.get_job("sample-job")["last_status"] == "delivery_failed"
