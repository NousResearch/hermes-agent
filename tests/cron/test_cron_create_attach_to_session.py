"""`hermes cron create --attach-to-session` CLI flag.

The cronjob() tool and jobs.json already support attach_to_session (a
continuable-session delivery), but there was no `hermes cron create` flag for
it -- it could only be set from the model tool or by hand-editing jobs.json.
This wires the flag through the create parser and dispatch.
"""

from __future__ import annotations

import argparse

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def _create_parser():
    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    build_cron_parser(sub, cmd_cron=lambda args: 0)
    return parser


def test_cli_create_parser_accepts_attach_to_session():
    args = _create_parser().parse_args(
        ["cron", "create", "every day at 08:00", "hi", "--attach-to-session"]
    )
    assert args.attach_to_session is True


def test_cli_create_defaults_attach_to_session_to_none():
    """Absent flag = None, so create_job leaves the key off (byte-identical)."""
    args = _create_parser().parse_args(["cron", "create", "every 5m", "hi"])
    assert args.attach_to_session is None


def test_cli_create_forwards_attach_to_session(hermes_env):
    """End-to-end: the flag lands on the persisted job record."""
    from hermes_cli.cron import cron_create
    from cron.jobs import load_jobs

    args = _create_parser().parse_args(
        ["cron", "create", "every day at 08:00", "run it",
         "--attach-to-session", "--deliver", "local"]
    )
    rc = cron_create(args)
    assert rc == 0

    jobs = load_jobs()
    assert len(jobs) == 1
    assert jobs[0]["attach_to_session"] is True
