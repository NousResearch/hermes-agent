"""Pre-agent cron gates settle their admitted session without losing retry identity (#143785)."""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("gate,reason", [
    ("unchanged", "cron_skipped"),
    ("source_error", "cron_preflight_failed"),
    ("wake_false", "cron_skipped"),
    ("preflight", "cron_preflight_failed"),
])
def test_pre_agent_exit_ends_session_and_preserves_retry(tmp_path, monkeypatch, gate, reason):
    from cron import jobs, monitor, scheduler
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = {"id": "job", "prompt": "Check the source", "model": "test-model"}
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    if gate in {"unchanged", "source_error"}:
        job.update(monitor_script="monitor.py",
                   monitor_state={"last_output_hash": monitor.hash_monitor_output("stable")})
        body = 'print("stable")' if gate == "unchanged" else 'raise RuntimeError("source unavailable")'
        (scripts / "monitor.py").write_text(body, encoding="utf-8")
    elif gate == "wake_false":
        job["script"] = "wake.py"
        (scripts / "wake.py").write_text('print(\'{"wakeAgent": false}\')', encoding="utf-8")
    else:
        monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *args:
                            scheduler._CronAgentSetup(blocked=(False, "blocked", "", "preflight refused")))

    monkeypatch.setattr(jobs, "get_job", lambda jid: dict(job))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {}, "platform_toolsets": {"cli": []}})
    monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *args, **kwargs: pytest.fail("Agent must not start"))
    with SessionDB(tmp_path / "state.db") as db:
        runner = SimpleNamespace(_draining=False, adapters={},
                                 session_store=SessionStore(tmp_path / "sessions", GatewayConfig()))
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id="test", db=db,
                                     epoch=begin_runtime_epoch(db, instance_id="test"))
        runner.session_authority = authority
        monkeypatch.setattr(authority, "_schedule", lambda ref: None)

        async def probe():
            params = {"job_id": "job", "request_id": "fire", "extra_prompt": None}
            receipt = await session_cron.operation(authority, "submit", params)
            sid = receipt["session_id"]
            row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
            adapter = next(iter(runner.adapters.values()))
            ref = SessionRef(authority.profile_id, sid)
            if reason == "cron_preflight_failed":
                with pytest.raises(RuntimeError):
                    await session_cron.execute(authority, ref, row, adapter.policies[sid])
            else:
                assert await session_cron.execute(authority, ref, row, adapter.policies[sid]) == scheduler.SILENT_MARKER
            session = db.get_session(sid)
            assert session["end_reason"] == reason
            assert session["ended_at"] is not None
            assert session["message_count"] == 0
            result = authority.pending_results[row["admission_id"]]
            outcome = "completed" if result["result"]["completed"] else "failed"
            settle_session_input(db, epoch=authority.epoch, admission_id=row["admission_id"],
                                 generation=row["generation"], outcome=outcome, result=result)
            assert await session_cron.operation(authority, "submit", params) == receipt
            recovered = await session_cron.operation(authority, "recover", params)
            assert recovered["status"] == "terminal"
            assert recovered["result"] == result["result"]["cron_result"]
            assert db.get_session(sid)["ended_at"] == session["ended_at"]

        asyncio.run(probe())
