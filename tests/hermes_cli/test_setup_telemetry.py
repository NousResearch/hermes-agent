"""Tests for shared-metrics configuration discovery and setup."""

from __future__ import annotations

import argparse

from hermes_cli.config import DEFAULT_CONFIG
from hermes_cli.setup import setup_telemetry
from hermes_cli.subcommands.setup import build_setup_parser


def test_shared_metrics_are_registered_disabled_by_default():
    assert DEFAULT_CONFIG["telemetry"]["shared_metrics"]["enabled"] is False


def test_setup_telemetry_enables_shared_metrics(monkeypatch):
    config = {}
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no",
        lambda _question, default: not default,
    )

    setup_telemetry(config)

    assert config["telemetry"]["shared_metrics"]["enabled"] is True


def test_disabling_collection_closes_the_send_consent_window(monkeypatch, tmp_path):
    """`hermes tools` -> disable shared metrics must withdraw send consent.

    The not-enabled branch returned early without recording anything, so the
    consent window stayed open and re-enabling later would release every
    package collected in between.
    """
    from hermes_cli.observability.shared_metrics import SharedMetricsStore
    from hermes_cli.observability.shared_metrics_sender import (
        reconcile_send_consent,
    )
    from hermes_cli.sqlite_util import write_txn

    store = SharedMetricsStore(
        database_path=tmp_path / "m.db", outbox_directory=tmp_path / "o"
    )
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics.SharedMetricsStore",
        lambda *a, **k: store,
    )

    # The user had consented; now they turn collection off entirely.
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no", lambda _question, default: False
    )
    config = {"telemetry": {"shared_metrics": {"enabled": True, "send": True}}}
    # Consent was granted earlier, so a window is open — that is precisely
    # the state whose closure must be recorded.
    with store._connection() as connection:
        with write_txn(connection):
            reconcile_send_consent(connection, True)

    setup_telemetry(config)

    assert config["telemetry"]["shared_metrics"]["enabled"] is False
    assert config["telemetry"]["shared_metrics"]["send"] is False
    with store._connection() as connection:
        open_windows = connection.execute(
            "SELECT COUNT(*) FROM send_consent_windows WHERE closed_at IS NULL"
        ).fetchone()[0]
    assert open_windows == 0, (
        "disabling collection left the send consent window open"
    )


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
    from hermes_cli.config import load_config, read_raw_config, save_config

    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda _question, default: False)
    config = load_config()
    setup_telemetry(config)
    save_config(config)  # what the wizard and `hermes tools` do afterwards

    assert read_raw_config()["telemetry"]["shared_metrics"] == {"enabled": False, "send": False, "offer_version": 2}


def test_chat_offer_asks_an_undecided_profile_once(monkeypatch):
    from hermes_cli.config import read_raw_config
    from hermes_cli.observability import shared_metrics_consent as consent

    asked = []
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: asked.append(a) or 1)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)

    consent.offer_consent_before_chat(argparse.Namespace(query="hi"))  # no person to ask
    assert asked == []
    consent.offer_consent_before_chat(argparse.Namespace())
    consent.offer_consent_before_chat(argparse.Namespace())

    assert len(asked) == 1
    assert consent.consent_state(read_raw_config()) == {"enabled": True, "send": False, "decided": True, "reask": False}


def test_offer_drops_keys_typed_before_it_appeared(monkeypatch):
    """An Enter typed while Hermes booted answered "No thanks" before the offer was on screen."""
    from hermes_cli.observability import shared_metrics_consent as consent

    events = []
    monkeypatch.setattr("hermes_cli.curses_ui.flush_stdin", lambda: events.append("flush"))
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: events.append("ask") or -1)

    consent.offer_consent()

    assert events == ["flush", "ask"]


def test_unseen_no_thanks_is_asked_once_more_and_then_settled(monkeypatch):
    """A pre-fix "No thanks" is re-offered with the reason; Esc on the re-ask keeps it for good,
    while a pre-fix opt-in was a deliberate pick and is never re-asked."""
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.observability import shared_metrics_consent as consent

    shown = []
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: shown.append(k["description"]) or -1)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)

    save_config({"telemetry": {"shared_metrics": {"enabled": True, "send": False}}}, strip_defaults=False)
    consent.offer_consent_before_chat(argparse.Namespace())
    assert shown == []

    save_config({"telemetry": {"shared_metrics": {"enabled": False, "send": False}}}, strip_defaults=False)
    consent.offer_consent_before_chat(argparse.Namespace())
    consent.offer_consent_before_chat(argparse.Namespace())

    assert len(shown) == 1 and shown[0].startswith(consent._REASK_NOTE)
    assert consent.consent_state(read_raw_config()) == {"enabled": False, "send": False, "decided": True, "reask": False}


def test_every_cli_promise_that_pre_opt_in_data_stays_local_names_the_install_note():
    """Invariant: the fresh-install note is the one record kept from before the consent answer and
    counted after it, so each CLI surface stating the pre-opt-in promise names that exception, and
    the explainer still fits an 80-column terminal (print_info indents by two)."""
    from hermes_cli.observability.shared_metrics_consent import _OFFER_DESCRIPTION
    from hermes_cli.setup import _SEND_CONSENT_EXPLAINER

    explainer = " ".join(_SEND_CONSENT_EXPLAINER)
    for text in (explainer, " ".join(_OFFER_DESCRIPTION.split())):
        assert "before you opt in" in text and "install note" in text.split("before you opt in")[0], text
    assert max(len(line) for line in _SEND_CONSENT_EXPLAINER) + 2 <= 80
