"""CLI wiring for ``hermes webapp setup`` (guided headless publishing)."""
import argparse
from types import SimpleNamespace

import hermes_cli.webapp_launcher as launcher
from hermes_cli.subcommands.webapp import build_webapp_parser


def _parser():
    root = argparse.ArgumentParser()
    build_webapp_parser(root.add_subparsers(dest="cmd"), cmd_webapp=lambda args: 0)
    return root


def test_parser_accepts_setup_command():
    args = _parser().parse_args(["webapp", "setup"])
    assert args.command == "setup"


def test_parser_without_command_keeps_legacy_flags():
    args = _parser().parse_args(["webapp", "--status"])
    assert args.command is None and args.status is True
    args = _parser().parse_args(["webapp"])
    assert args.command is None


def test_parser_rejects_unknown_command():
    import pytest
    with pytest.raises(SystemExit):
        _parser().parse_args(["webapp", "bogus"])


def test_cmd_webapp_dispatches_setup_to_flow(monkeypatch):
    called = []
    monkeypatch.setattr(launcher, "_deleted_profile_refusal", lambda home: None)
    monkeypatch.setattr("hermes_cli.webapp_publish.flow.interactive_setup",
                        lambda: called.append(True))
    rc = launcher.cmd_webapp(SimpleNamespace(command="setup"))
    assert rc == 0 and called == [True]


def test_cmd_webapp_without_setup_does_not_run_flow(monkeypatch):
    called = []
    monkeypatch.setattr(launcher, "_deleted_profile_refusal", lambda home: None)
    monkeypatch.setattr("hermes_cli.webapp_publish.flow.interactive_setup",
                        lambda: called.append(True))
    monkeypatch.setattr(launcher, "_gateway_home", lambda home: "/tmp/fake")
    monkeypatch.setattr(launcher, "_status", lambda target: 0)
    rc = launcher.cmd_webapp(SimpleNamespace(command=None, status=True))
    assert rc == 0 and called == []