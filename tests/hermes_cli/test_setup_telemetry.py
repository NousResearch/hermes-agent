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
    from hermes_cli.observability.shared_metrics_sender import SEND_REVOKED_KEY

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
    # Consent was granted earlier, so a window is already open — that is
    # precisely the state whose closure must be recorded.
    from hermes_cli.sqlite_util import write_txn
    from hermes_cli.observability.shared_metrics_sender import opt_in_period

    with store._connection() as connection:
        with write_txn(connection):
            opt_in_period(connection)

    setup_telemetry(config)

    assert config["telemetry"]["shared_metrics"]["enabled"] is False
    assert config["telemetry"]["shared_metrics"]["send"] is False
    with store._connection() as connection:
        row = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = ?", (SEND_REVOKED_KEY,)
        ).fetchone()
    assert row is not None and row[0] == "1", (
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
