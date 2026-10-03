import argparse
import json

from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser


def test_doctor_parser_and_failure_exit(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=cron_command)
    args = parser.parse_args(["cron", "doctor", "--json", "--check-provider"])
    calls = []

    def inspect(**kwargs):
        calls.append(kwargs)
        return [{"id": "test", "issues": [{"code": "provider_unavailable", "message": "expired"}]}]

    monkeypatch.setattr("cron.health.inspect_jobs", inspect)
    assert args.func(args) == 1
    assert json.loads(capsys.readouterr().out)["healthy"] is False
    assert calls == [{"check_provider": True}]


def test_doctor_empty_json_is_healthy(monkeypatch, capsys):
    from hermes_cli.cron import cron_doctor

    monkeypatch.setattr("cron.health.inspect_jobs", lambda **kwargs: [])
    assert cron_doctor(as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {"healthy": True, "jobs": []}
