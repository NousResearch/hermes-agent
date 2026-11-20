"""Receipt reconciliation must precede reclaimed-owner alerts in the real tick path."""

import json
from contextlib import asynccontextmanager

from cron import executions, incidents, jobs, scheduler, scheduler_authority


def test_tick_settles_receipt_before_alerting_other_dead_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "_owner_is_live", lambda *_args: False)
    monkeypatch.setattr(scheduler, "_last_dead_owner_reap_at", {})
    monkeypatch.setattr(scheduler, "get_due_jobs", list)
    monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)
    monkeypatch.setattr(scheduler, "_maybe_run_worktree_maintenance", lambda: None)
    deliveries = []

    def deliver(job, content, **kwargs):
        deliveries.append((job["id"], kwargs))

    monkeypatch.setattr(scheduler, "_deliver_result", deliver)
    adapters, loop = object(), object()

    with jobs.use_cron_store(tmp_path / "cron"):
        receipted_job = jobs.create_job(prompt="done", schedule="every 1h", deliver="local")
        orphan_job = jobs.create_job(
            prompt="lost", schedule="every 1h", deliver="local",
            failure_deliver="slack:D0ALERTS",
        )
        receipted = executions.create_execution(receipted_job["id"], source="test")
        orphan = executions.create_execution(orphan_job["id"], source="test")
        with executions._transaction() as conn:
            conn.execute("UPDATE executions SET process_id='departed-owner', pid=999999")
        params = {
            "job_id": receipted_job["id"], "request_id": receipted["id"],
            "extra_prompt": None,
        }
        journal = scheduler_authority.journal_path(receipted_job["id"], receipted["id"])
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({"params": params, "receipt": None}))

        class Peer:
            async def rpc(self, method, **kwargs):
                assert method == "cron.recover"
                assert kwargs == params
                return {
                    "status": "terminal", "job": receipted_job,
                    "result": [True, "document", "answer", None],
                }

        @asynccontextmanager
        async def connected():
            yield Peer()

        monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", connected)

        assert scheduler.tick(verbose=False, adapters=adapters, loop=loop) == 0

        assert executions.get_execution(receipted["id"])["status"] == "completed"
        assert not journal.exists()
        recovered = executions.get_execution(orphan["id"])
        assert recovered["status"] == "unknown"
        assert recovered["delivery_outcome"] == "delivered"
        assert {item["job_id"] for item in incidents.list_incidents()} == {orphan_job["id"]}
        alerts = [(job_id, context) for job_id, context in deliveries if context.get("for_failure")]
        assert len(alerts) == 1
        assert alerts[0] == (
            orphan_job["id"], {"adapters": adapters, "loop": loop, "for_failure": True},
        )
