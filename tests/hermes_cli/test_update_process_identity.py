"""Updater integration for runtime-owned process ledger identity."""

from __future__ import annotations

from unittest.mock import patch

from runtime import process_identity as pi


def _entry(pid, purpose, spawner_pid):
    return {
        "pid": pid,
        "create_time": float(pid),
        "purpose": purpose,
        "install": "test-install",
        "spawner_pid": spawner_pid,
        "spawner_create": float(spawner_pid),
        "registered_at": 0.0,
        "argv": "",
    }


def _holders(*pids):
    return [(pid, "python.exe", f"python.exe -m hermes_cli.main serve --pid {pid}") for pid in pids]


def test_updater_reaps_only_ledger_proven_orphans():
    from hermes_cli import main as cli_main

    entries = [
        _entry(200, "serve", 700),
        _entry(201, "serve", 500),
        _entry(202, "chat", 700),
        _entry(203, "mcp-helper", 700),
    ]
    with (
        patch.object(pi, "ledger_entries", return_value=entries),
        patch.object(pi, "spawner_is_dead", side_effect=lambda entry: entry["spawner_pid"] == 700),
    ):
        assert cli_main._ledger_reapable_backend_pids(_holders(200, 201, 202, 203)) == [200, 203]


def test_updater_ledger_rung_never_raises():
    from hermes_cli import main as cli_main

    with patch.object(pi, "ledger_entries", side_effect=RuntimeError("boom")):
        assert cli_main._ledger_reapable_backend_pids(_holders(200)) == []
