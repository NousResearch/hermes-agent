"""``hermes cron runs --json``: stable machine-readable execution history (#118072)."""

from __future__ import annotations

import json

from hermes_cli.cron import cron_runs
from hermes_cli.subcommands.cron import build_cron_parser


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_cron_runs_json_prints_array_of_records(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    claimed = executions.create_execution("job-1", source="builtin")
    executions.finish_execution(claimed["id"], success=True)

    cron_runs(job_id="job-1", json_output=True)

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["job_id"] == "job-1"
    assert payload[0]["status"] == "completed"
    # Never the human "No cron execution attempts recorded." prose.
    assert "No cron execution attempts recorded." not in out


def test_cron_runs_json_empty_is_bracket_not_prose(monkeypatch, tmp_path, capsys):
    _point_ledger(monkeypatch, tmp_path)

    cron_runs(job_id="no-such-job", json_output=True)

    out = capsys.readouterr().out
    assert json.loads(out) == []
    assert "No cron execution attempts recorded." not in out


def test_cron_runs_json_flag_is_parsed():
    import argparse

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_cron_parser(subparsers, cmd_cron=lambda a: None)

    args = parser.parse_args(["cron", "runs", "some-job", "--json"])
    assert args.json is True
