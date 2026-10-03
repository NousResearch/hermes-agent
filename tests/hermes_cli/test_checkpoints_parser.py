"""Regression coverage for profile-aware checkpoint help text."""

import argparse


def test_checkpoints_help_uses_active_hermes_home(monkeypatch):
    import hermes_constants
    from hermes_cli.subcommands.checkpoints import build_checkpoints_parser

    monkeypatch.setattr(hermes_constants, "display_hermes_home", lambda: "~/.hermes/profiles/coder")

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_checkpoints_parser(subparsers)

    help_text = next(action.help for action in subparsers._choices_actions if action.dest == "checkpoints")

    assert help_text == "Inspect / prune / clear ~/.hermes/profiles/coder/checkpoints/"
