"""#83197/#83557: scope spans real owned cron execution and its current delivery handoff."""
from pathlib import Path

from agent import secret_scope
from cron import scheduler
from cron.jobs import create_job
from cron.scheduler_provider import _profile_cron_scope


def test_owned_fire_keeps_profile_credentials_through_delivery_queue_handoff(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "reviewer"
    home.mkdir(parents=True)
    (home / ".env").write_text("TELEGRAM_BOT_TOKEN=reviewer-token\n", encoding="utf-8")
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    seen = {}

    def run(job, **kwargs):
        seen["run"] = secret_scope.current_secret_scope()
        assert secret_scope.get_secret("TELEGRAM_BOT_TOKEN") == "reviewer-token"
        return True, "ran", "job report text", None

    def enqueue(execution_id, job, content, **kwargs):
        seen["delivery"] = secret_scope.current_secret_scope()
        assert secret_scope.get_secret("TELEGRAM_BOT_TOKEN") == "reviewer-token"
        assert content == "job report text"
        return {"status": "pending"}

    # Replace only transport/agent work: job creation, ledger ownership and scope are real.
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr(scheduler, "run_job", run)
    monkeypatch.setattr("cron.delivery_queue.enqueue", enqueue)
    prior = secret_scope.current_secret_scope()
    with _profile_cron_scope(home):
        outer_scope = secret_scope.current_secret_scope()
        job = create_job("fixture prompt", "every 1h", deliver="telegram:12345")
        assert scheduler.run_one_job(job) is True
        assert seen["run"] is not None
        assert seen["delivery"] == seen["run"]
        assert secret_scope.current_secret_scope() == outer_scope
    assert secret_scope.current_secret_scope() is prior
