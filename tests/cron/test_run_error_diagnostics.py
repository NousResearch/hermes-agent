"""Failure detail belongs in the local audit, not the delivery summary."""
from cron import jobs, scheduler


def test_run_error_persists_redacted_cause_but_returns_summary(tmp_path, monkeypatch):
    def fail_setup(*args):
        try:
            raise ConnectionError("https://user:password@localhost/api?token=secret-token")
        except ConnectionError as cause:
            raise RuntimeError("Connection error.") from cause

    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", fail_setup)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="Check status", schedule="every 1h", model="local-probe", deliver="local")
        success, output, response, error = scheduler.run_job(job)
        saved = jobs.save_job_output(job["id"], output).read_text(encoding="utf-8")
        assert not success and response == "" and error == "RuntimeError: Connection error."
        assert "Traceback (most recent call last)" in saved and "fail_setup" in saved
        assert "ConnectionError" in saved and "localhost" in saved
        assert "password" not in saved and "secret-token" not in saved


def test_list_exposes_run_error_and_clears_it_after_success(tmp_path, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from tools.cronjob_job_args import _format_job

    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="Check status", schedule="every 1h", model="local-probe", deliver="local")
        reason = "RuntimeError: https://user:password@localhost/api?token=secret-token"
        jobs.mark_job_run(job["id"], False, error=reason)
        displayed = _format_job(jobs.get_job(job["id"]))
        assert displayed["last_error"].startswith("RuntimeError:") and "localhost" in displayed["last_error"]
        assert "password" not in displayed["last_error"] and "secret-token" not in displayed["last_error"]
        assert displayed["last_delivery_error"] is None and displayed["last_fire_error"] is None
        CLICommandsMixin._cron_list(object(), "list", {"all": False})
        console = capsys.readouterr().out
        assert f"error: {displayed['last_error']}" in console
        assert "password" not in console and "secret-token" not in console
        jobs.mark_job_run(job["id"], True)
        assert _format_job(jobs.get_job(job["id"]))["last_error"] is None
