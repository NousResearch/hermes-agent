"""Queue admission is not delivery; late suppression updates exact projections."""
import json

import pytest
from cron import executions, incidents, jobs, scheduler, delivery_queue
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.mark.parametrize("setting", [None, False, True])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_external_timeout_then_drain_reconciles_exact_failed_execution(tmp_path, monkeypatch, setting, exit_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {"cron": {"wrap_response": False, "preflight": False}}
    if setting is not None:
        cfg["display"] = {"suppress_warning_notifications": setting}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    sent = []
    async def send(platform, pconfig, chat_id, text, **kwargs):
        sent.append(text)
        return {"success": True, "message_id": "recorded-send"}
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "failure.sh").write_text("#!/bin/sh\nprintf 'retained stdout\\n'\nprintf 'diagnostic stderr\\n' >&2\nexit 7\n")
    script = scripts / "failure.sh"
    script.write_text(script.read_text().replace("exit 7", f"exit {exit_code}"))
    job = jobs.create_job(prompt=None, schedule="every 1h", script="failure.sh", no_agent=True, deliver="telegram:fixture")
    execution = executions.create_execution(job["id"], source="test-external")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    wait = delivery_queue.enqueue_and_wait
    monkeypatch.setattr(delivery_queue, "enqueue_and_wait",
        lambda execution_id, job, content, for_failure=False: wait(execution_id, job, content, for_failure=for_failure, timeout=0))
    scheduler.run_one_job(job)
    before = executions.get_execution(execution["id"])
    assert delivery_queue.get_status(execution["id"])["status"] == "pending"
    assert not sent
    assert before["status"] == ("failed" if exit_code else "completed")
    assert before["delivery_outcome"] == "queued"
    if exit_code:
        incident = next(i for i in incidents.list_incidents() if i["job_id"] == job["id"])
        assert incident["state"] == "detected"
    assert scheduler.drain_delivery_queue({}, None) == 1
    assert scheduler.drain_delivery_queue({}, None) == 0
    after = executions.get_execution(execution["id"])
    muted = setting is True and exit_code != 0
    expected = "suppressed" if muted else "delivered"
    assert delivery_queue.get_status(execution["id"])["status"] == expected
    assert after["delivery_outcome"] == expected
    assert after["status"] == before["status"]
    assert after["error"] == before["error"]
    assert len(sent) == (0 if muted else 1)
    if exit_code:
        incident = next(i for i in incidents.list_incidents() if i["job_id"] == job["id"])
        assert incident["state"] == ("detected" if muted else "alerted")
    outputs = list((tmp_path / "cron" / "output" / job["id"]).glob("*.md"))
    assert outputs and "retained stdout" in outputs[0].read_text()
    assert not jobs.get_job(job["id"]).get("last_delivery_queued")



@pytest.mark.parametrize("sibling", ["none", "delivered", "ambiguous"])
def test_bot_chat_late_suppression_reconciles_after_reopen(tmp_path, monkeypatch, sibling):
    from cron import bot_chat_delivery
    from hermes_state import SessionDB
    from hermes_cli.active_sessions import try_acquire_active_session
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: false}\n")
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="bot", source="cli")
    db.set_session_title("bot", "Bot Chat")
    lease, refusal = try_acquire_active_session(session_id="bot", surface="cli", config={}, registry_home=tmp_path)
    assert refusal is None
    try:
        monkeypatch.setattr(scheduler, "run_job", lambda *a, **k: (False, "raw evidence", "", "failure evidence"))
        job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver="bot-chat")
        scheduler.run_one_job(job)
        before = executions.latest_execution(job["id"])
        assert before["delivery_outcome"] == "queued"
        key = next(iter(jobs.get_job(job["id"])["last_delivery_queued"].values()))["delivery_id"]
        if sibling != "none":
            # A sibling's durable disposition must survive this target's suppression.
            with executions._transaction() as conn:
                manifest = json.loads(conn.execute("SELECT delivery_manifest FROM executions WHERE id=?", (before["id"],)).fetchone()[0])
                manifest["delivered"] = sibling == "delivered"
                manifest["unverified"] = ["ambiguous sibling"] if sibling == "ambiguous" else []
                conn.execute("UPDATE executions SET delivery_manifest=? WHERE id=?", (json.dumps(manifest), before["id"]))
        (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}\n")
        bot_chat_delivery.drain()
        bot_chat_delivery.drain()
        assert bot_chat_delivery.read_pending(key)["status"] == "suppressed"
        after = executions.get_execution(before["id"])
        assert after["delivery_outcome"] == {"none": "suppressed", "delivered": "delivered", "ambiguous": "unknown"}[sibling]
        assert after["error"] == before["error"]
        assert after["finished_at"] == before["finished_at"]
        assert not jobs.get_job(job["id"]).get("last_delivery_queued")
    finally:
        lease.release()
        db.close()


def _queued_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver="telegram:test")
    execution = executions.create_execution(job["id"], source="test")
    job["execution_id"] = execution["id"]
    jobs.bind_delivery_execution(job["id"], execution["id"])
    incident_id, _ = incidents.upsert_incident(job["id"], "raw failure")
    executions.bind_delivery_incident(execution["id"], incident_id)
    executions.record_delivery_manifest(execution["id"], {"external": True})
    delivery_queue.enqueue(execution["id"], job, "diagnostic", for_failure=True)
    jobs.update_delivery_projection(job["id"], execution["id"], {"last_delivery_queued": {"gateway": {"status": "queued"}}})
    executions.finish_execution(execution["id"], success=False, error="raw failure", delivery_outcome="queued")
    return job, execution, incident_id


@pytest.mark.parametrize("terminal", ["delivered", "suppressed", "failed", "unknown"])
@pytest.mark.parametrize("newer", [False, True])
def test_restart_repairs_exact_execution_without_replay(tmp_path, monkeypatch, terminal, newer):
    import os
    import subprocess
    import sys
    job, execution, incident_id = _queued_execution(tmp_path, monkeypatch)
    before = executions.get_execution(execution["id"])
    if newer:
        latest = executions.create_execution(job["id"], source="newer")
        jobs.bind_delivery_execution(job["id"], latest["id"])
        jobs.update_delivery_projection(job["id"], latest["id"], {
            "last_delivery_queued": {"newer-target": {"status": "queued"}},
            "last_delivery_unverified": ["newer-ambiguity"], "last_delivery_error": "newer-error"})
    saved = jobs.get_job(job["id"])
    # Kill-window simulation: persist terminal receipt, omit projection callback.
    monkeypatch.setattr(executions, "reconcile_delivery_projections", lambda: None)
    if terminal == "unknown":
        delivery_queue.claim_next()
        delivery_queue._terminalize_wait_timeout(execution["id"])
    else:
        def send(queued_job, *args):
            queued_job["_notification_all_targets_suppressed"] = terminal == "suppressed"
            return "wire failure" if terminal == "failed" else None
        assert delivery_queue.drain(send) == 1
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "queued"
    # All connections reopened in a fresh interpreter. A terminal receipt never sends.
    code = "from cron.delivery_queue import drain; assert drain(lambda *a: (_ for _ in ()).throw(AssertionError('REPLAY'))) == 0"
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True)
    after = executions.get_execution(execution["id"])
    assert after["delivery_outcome"] == terminal
    assert {k: after[k] for k in ("status", "error", "finished_at")} == {k: before[k] for k in ("status", "error", "finished_at")}
    assert incidents.get_incident(incident_id)["state"] == ("alerted" if terminal == "delivered" else "detected")
    if newer:
        assert jobs.get_job(job["id"]) == saved
    else:
        assert not jobs.get_job(job["id"])["last_delivery_queued"]


def test_external_claimed_timeout_reports_unknown_immediately(tmp_path, monkeypatch):
    from cron import scheduler_delivery
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **k: (False, "raw", "", "failed script"))
    job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver="telegram:test")
    execution = executions.create_execution(job["id"], source="test")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    wait = delivery_queue.enqueue_and_wait
    def claim_then_wait(key, job, content, *, for_failure=False):
        delivery_queue.enqueue(key, job, content, for_failure=for_failure)
        assert delivery_queue.claim_next()
        return wait(key, job, content, for_failure=for_failure, timeout=0)
    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", claim_then_wait)
    scheduler.run_one_job(job)
    assert delivery_queue.get_status(execution["id"])["status"] == "unknown"
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "unknown"


def test_late_gateway_result_cannot_override_unknown_send(tmp_path, monkeypatch):
    job, execution, _ = _queued_execution(tmp_path, monkeypatch)
    assert delivery_queue.claim_next()
    delivery_queue._terminalize_wait_timeout(execution["id"])
    # The late sender finishes target bookkeeping after the worker fenced the claim.
    executions.record_delivery_manifest(execution["id"], {"bot": {}, "delivered": True})
    assert not delivery_queue._finish(execution["id"], error=None)
    delivery_queue.drain(lambda *a: pytest.fail("must not replay"))
    assert delivery_queue.get_status(execution["id"])["status"] == "unknown"
    assert executions.get_execution(execution["id"])["delivery_outcome"] == "unknown"



def test_direct_unverified_native_send_retains_legacy_outcome(tmp_path, monkeypatch):
    import asyncio
    from concurrent.futures import Future
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from gateway.platforms.base import SendResult
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: false}")
    monkeypatch.setattr(scheduler, "run_job", lambda *a, **k: (True, "raw", "requested result", None))
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    async def send(*args, **kwargs):
        return SendResult(success=True)
    router = SimpleNamespace(_deliver_to_platform=send)
    monkeypatch.setattr("gateway.delivery.DeliveryRouter", lambda *a, **k: router)
    def run_coro(coro, loop):
        future = Future()
        future.set_result(asyncio.run(coro))
        return future
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", run_coro)
    job = jobs.create_job(prompt="fixture", schedule="every 1h", deliver="telegram:test")
    loop = MagicMock()
    loop.is_running.return_value = True
    scheduler.run_one_job(job, adapters={Platform.TELEGRAM: MagicMock()}, loop=loop)
    assert executions.latest_execution(job["id"])["delivery_outcome"] == "delivered"
    assert jobs.get_job(job["id"])["last_delivery_unverified"]
