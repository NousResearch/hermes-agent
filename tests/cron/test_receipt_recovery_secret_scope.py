"""Receipt recovery binds and restores the firing home's secret scope."""
import json

from agent.secret_scope import current_secret_scope, current_secret_scope_home
from cron import jobs, scheduler, scheduler_authority


def test_recovered_receipt_delivery_has_firing_scope(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.env_loader.hydrate_profile_secret_sources", lambda home: None)
    monkeypatch.setattr("tui_gateway.launch_profile_policy.owner_secret_scope", lambda home: {"EXAMPLE_TOKEN": "fixture"})
    monkeypatch.setattr("cron.scheduler_provider.routed_profile_fire", lambda: True)
    before = current_secret_scope()
    observed = []
    with jobs.use_cron_store(home / "cron"):
        job = jobs.create_job(prompt="report", schedule="every 1h", deliver="local")
        params = {"job_id": job["id"], "request_id": "receipt-fixture", "extra_prompt": None}
        journal = scheduler_authority.journal_path(job["id"], params["request_id"])
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({"params": params, "job": job,
                                      "refusal": [True, "output", "answer", None]}))
        def deliver(*args, **kwargs):
            observed.append((dict(current_secret_scope() or {}), current_secret_scope_home()))
        monkeypatch.setattr(scheduler, "_save_compose_deliver", deliver)
        monkeypatch.setattr("cron.scheduler_bookkeeping.finish_completed_run", lambda *a, **k: False)
        monkeypatch.setattr("cron.scheduler_bookkeeping.fail_empty_response", lambda *a, **k: None)
        scheduler_authority.reconcile_pending(allow_connect=False)
    assert observed == [({"EXAMPLE_TOKEN": "fixture"}, str(home))]
    assert current_secret_scope() is before
