"""Creation is either armed normally or durably paused before registration."""
import json

from cron import jobs
from tools import cronjob_tools


def test_paused_creation_is_inert_until_operator_action(tmp_path, monkeypatch, make_cron_provider):
    registered, writes = [], []
    provider = make_cron_provider(register_job=lambda job: registered.append(job["id"]))
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: provider)
    save = jobs.save_jobs

    def observe(rows):
        writes.append(json.loads(json.dumps(rows)))
        return save(rows)

    monkeypatch.setattr(jobs, "save_jobs", observe)
    with jobs.use_cron_store(tmp_path / "cron"):
        def create(**kwargs):
            return json.loads(cronjob_tools.registry.dispatch("cronjob_manage", {
                "action": "create", "schedule": "every 1h", "prompt": "canary", **kwargs}))

        active = create()
        assert active["success"] and active["job"]["enabled"]
        paused = create(paused=True)
        assert paused["success"], paused
        job_id = paused["job_id"]
        first = next(row for rows in writes for row in rows if row["id"] == job_id)
        assert first["enabled"] is False and first["state"] == "paused"
        assert first["paused_at"] and first["paused_reason"] and first["next_run_at"] is None
        assert registered == [active["job_id"]]
        assert job_id not in {row["id"] for row in jobs.get_due_jobs()}
        assert jobs.claim_job_for_fire(job_id) is False
        resumed = jobs.resume_job(job_id)
        assert resumed["enabled"] and resumed["next_run_at"] and resumed["paused_reason"] is None
        forced = create(paused=True, paused_reason="canary review")
        assert jobs.claim_job_for_fire(forced["job_id"], force=True) is True


def test_invalid_creation_is_rejected_without_writes(tmp_path):
    with jobs.use_cron_store(tmp_path / "cron"):
        for flags in ({"paused": "yes"}, {"paused": None}, {"paused": 1},
                      {"paused_reason": "orphan"}, {"paused": True, "paused_reason": 123}):
            result = json.loads(cronjob_tools.registry.dispatch("cronjob_manage", {
                "action": "create", "schedule": "every 1h", "prompt": "invalid", **flags}))
            assert result["success"] is False, (flags, result)
            assert jobs.load_jobs() == []
        from types import SimpleNamespace
        from hermes_cli.main import cmd_cron

        args = SimpleNamespace(cron_command="create", schedule="every 1h", prompt="invalid",
                               paused=False, paused_reason="orphan")
        assert cmd_cron(args) == 1
        assert jobs.load_jobs() == []
