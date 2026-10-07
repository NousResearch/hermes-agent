"""Tests for shared-metrics configuration discovery and setup."""

from __future__ import annotations

import argparse

from rabbit_cli.config import DEFAULT_CONFIG
from rabbit_cli.setup import setup_telemetry
from rabbit_cli.subcommands.setup import build_setup_parser


def test_shared_metrics_are_registered_disabled_by_default():
    assert DEFAULT_CONFIG["telemetry"]["shared_metrics"]["enabled"] is False


def test_setup_telemetry_enables_shared_metrics(monkeypatch):
    config = {}
    monkeypatch.setattr(
        "rabbit_cli.setup.prompt_yes_no",
        lambda _question, default: not default,
    )

    setup_telemetry(config)

    assert config["telemetry"]["shared_metrics"]["enabled"] is True


def test_disabling_collection_persists(monkeypatch):
    """`rabbit tools` -> disable shared metrics writes enabled=false into config."""
    monkeypatch.setattr(
        "rabbit_cli.setup.prompt_yes_no", lambda _question, default: False
    )
    config = {"telemetry": {"shared_metrics": {"enabled": True}}}

    setup_telemetry(config)

    assert config["telemetry"]["shared_metrics"]["enabled"] is False


def test_setup_parser_accepts_telemetry_section():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    handler = object()
    build_setup_parser(subparsers, cmd_setup=handler)

    args = parser.parse_args(["setup", "telemetry"])

    assert args.section == "telemetry"
    assert args.func is handler


def test_no_answer_survives_the_callers_later_config_save(monkeypatch):
    """A "no" equals the shipped defaults; saved only through ``save_config`` it was stripped, so
    the profile read undecided and every surface (Desktop strip, CLI offer) asked again."""
    from rabbit_cli.config import load_config, read_raw_config, save_config

    monkeypatch.setattr("rabbit_cli.setup.prompt_yes_no", lambda _question, default: False)
    config = load_config()
    setup_telemetry(config)
    save_config(config)  # what the wizard and `rabbit tools` do afterwards

    assert read_raw_config()["telemetry"]["shared_metrics"] == {"enabled": False}


def test_chat_offer_asks_an_undecided_profile_once(monkeypatch):
    from rabbit_cli.config import read_raw_config
    from rabbit_cli.observability import shared_metrics_consent as consent

    asked = []
    # Answer the first choice: "Collect locally only".
    monkeypatch.setattr("rabbit_cli.curses_ui.curses_radiolist", lambda *a, **k: asked.append(a) or 0)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("RABBIT_DESKTOP", raising=False)
    monkeypatch.delenv("RABBIT_NONINTERACTIVE", raising=False)

    consent.offer_consent_before_chat(argparse.Namespace(query="hi"))  # no person to ask
    assert asked == []
    consent.offer_consent_before_chat(argparse.Namespace())
    consent.offer_consent_before_chat(argparse.Namespace())

    assert len(asked) == 1
    assert consent.consent_state(read_raw_config()) == {"enabled": True, "decided": True}
