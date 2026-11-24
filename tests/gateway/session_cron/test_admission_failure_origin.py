"""Executor origin survives the actual committed cron admission boundary."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.storage_handles import storage_handle


@pytest.mark.asyncio
async def test_monitor_origin_survives_committed_admission_and_recovery(tmp_path, monkeypatch):
    from cron import jobs, scheduler
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority, admission_row
    from gateway.session_contract import SessionRef

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {}, "platform_toolsets": {"cli": []}})
    monkeypatch.setattr(run, "_profile_runtime_scope", lambda home: nullcontext())
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda config: [])
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "source.sh").write_text("echo 'HTTP 429 SYNTHETIC_SOURCE_DETAIL' >&2\nexit 1\n")
    with jobs.use_cron_store(tmp_path / "cron"):
        job = jobs.create_job(prompt="watch", model="synthetic-model", schedule="every 1h",
                              monitor_script="source.sh", deliver="local")
        db = storage_handle(tmp_path)
        runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / "sessions", GatewayConfig()),
                                 adapters={})
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", persistence=db,
                                     epoch=db.runtime.begin_epoch(instance_id="test"))
        runner.session_authority = authority
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)
        params = {"job_id": job["id"], "request_id": "origin-fire", "extra_prompt": None}
        try:
            receipt = await session_cron.operation(authority, "submit", params)
            sid = receipt["session_id"]
            row = admission_row(db.admissions.claim(epoch=authority.epoch, session_id=sid))
            policy = next(iter(runner.adapters.values())).policies[sid]
            with pytest.raises(RuntimeError, match="429"):
                await session_cron.execute(authority, SessionRef(str(tmp_path), sid), row, policy)
            completion = authority.pending_results[row["admission_id"]]
            assert completion["result"]["cron_job_flags"].get("_failure_origin") == "monitor_source"
            db.admissions.settle(epoch=authority.epoch, admission_id=row["admission_id"],
                                 generation=row["generation"], outcome="failed", result=completion)
            state = await session_cron.operation(authority, "status", receipt)
            recovered = await session_cron.operation(authority, "recover", params)
            assert state["job_flags"]["_failure_origin"] == recovered["job_flags"]["_failure_origin"] == "monitor_source"
            session_cron.bind_owner(authority)
            try:
                working = dict(job)
                result = await asyncio.to_thread(scheduler.run_job, working, execution_id="origin-fire")
            finally:
                session_cron.unbind_owner(authority)
            assert result[0] is False
            assert working["_failure_origin"] == "monitor_source"
            notice = scheduler._summarize_cron_failure_for_delivery(working, result[3])
            assert "monitored source" in notice
            assert "SYNTHETIC_SOURCE_DETAIL" not in notice
            assert "_failure_origin" not in jobs.get_job(job["id"])
        finally:
            db.close()


@pytest.mark.parametrize("bad_origin", [[], {}, 7])
def test_invalid_origin_keeps_legacy_delivery(monkeypatch, bad_origin):
    from cron import scheduler
    monkeypatch.setattr(scheduler, "load_config", dict)
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda config: [])
    notice = scheduler._summarize_cron_failure_for_delivery(
        {"id": "legacy", "_failure_origin": bad_origin}, "some opaque legacy error")
    assert "some opaque legacy error" in notice


def test_source_error_cannot_impersonate_scheduler_inactivity(monkeypatch):
    from cron import scheduler
    monkeypatch.setattr(scheduler, "load_config", dict)
    notice = scheduler._summarize_cron_failure_for_delivery(
        {"id": "source", "_failure_origin": "monitor_source"}, "idle for 600s (limit 300s)")
    assert "monitored source" in notice
    assert "stalled" not in notice


def test_independent_script_copy_does_not_claim_no_model_was_called(monkeypatch):
    from cron import scheduler
    monkeypatch.setattr(scheduler, "load_config", dict)
    notice = scheduler._summarize_cron_failure_for_delivery(
        {"id": "collector", "_failure_origin": "script_execution"}, "script called its own LLM: HTTP 503")
    assert "No model was invoked" not in notice
    assert "agent stage" in notice
