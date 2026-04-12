"""Tests for hermes_cli.work_orders surfaces."""

from argparse import Namespace

from cron.jobs import create_job, get_job
from hermes_cli.work_orders import capture_work_orders_slash_output, work_orders_command
from tools.execution_work_orders import enqueue_execution_work_order
from tools.execution_work_orders_tool import RUNNER_JOB_NAME



def _setup_cron(monkeypatch, tmp_path):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    return cron_dir


class TestWorkOrdersCliCommand:
    def test_list_enqueue_cancel(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        enqueue_execution_work_order(
            goal="Count lines",
            command="cd /workspace && wc -l tools/delegate_tool.py | awk '{print $1}'",
            now=1.0,
        )

        rc = work_orders_command(Namespace(work_orders_command="list", limit=10, status=None, work_order_id=None))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Execution Work Orders" in out
        assert "Count lines" in out
        assert "queued" in out

    def test_install_status_remove(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        _setup_cron(monkeypatch, tmp_path)

        rc = work_orders_command(
            Namespace(
                work_orders_command="install",
                schedule="every 2h",
                limit=7,
                reclaim_limit=9,
                claim_ttl_seconds=600.0,
                model="gpt-5.4",
                provider="openai-codex",
                base_url=None,
            )
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Created execution work-order runner job" in out

        status_rc = work_orders_command(Namespace(work_orders_command="status"))
        assert status_rc == 0
        out = capsys.readouterr().out
        assert "Installed jobs: 1" in out
        assert "Run limit:         7" in out

        remove_rc = work_orders_command(Namespace(work_orders_command="remove"))
        assert remove_rc == 0
        out = capsys.readouterr().out
        assert "Removed execution work-order runner job" in out

    def test_slash_status_install_remove(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        _setup_cron(monkeypatch, tmp_path)

        before = capture_work_orders_slash_output("/workorders status")
        assert "Installed jobs: 0" in before

        install_out = capture_work_orders_slash_output("/workorders install --schedule 'every 1h' --limit 11 --reclaim-limit 12 --claim-ttl-seconds 240")
        assert "Created execution work-order runner job" in install_out

        after = capture_work_orders_slash_output("/workorders status")
        assert "Installed jobs: 1" in after
        assert "Run limit:         11" in after
        assert "Reclaim limit:     12" in after

        remove_out = capture_work_orders_slash_output("/workorders remove")
        assert "Removed execution work-order runner job" in remove_out

    def test_same_name_unrelated_job_survives_workorders_surface(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.execution_work_orders.get_hermes_home", lambda: tmp_path)
        _setup_cron(monkeypatch, tmp_path)

        unrelated = create_job(
            prompt="plain unrelated cron job",
            schedule="every 4h",
            name=RUNNER_JOB_NAME,
            deliver="local",
        )

        status_out = capture_work_orders_slash_output("/workorders status")
        assert "Installed jobs: 0" in status_out

        remove_out = capture_work_orders_slash_output("/workorders remove")
        assert "No execution work-order runner job was installed" in remove_out
        assert get_job(unrelated["id"]) is not None
