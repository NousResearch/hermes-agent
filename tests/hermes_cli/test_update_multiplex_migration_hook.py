"""``hermes update`` runs the multiplex auto-migration hook only after a verified-healthy fleet restart.

Reuses the mocked-git ``cmd_update`` harness from ``test_update_fleet_restart_pending`` (network and
restart stubbed). The hook itself is faked to a recorder: what is under test is its placement — it
runs on the success path and is skipped when the fleet verification exits 1.
"""

from __future__ import annotations

import pytest

from hermes_cli import main as hermes_main
from hermes_cli import gateway_migrate

from tests.hermes_cli.test_update_fleet_restart_pending import (
    _make_head_moved_side_effect, _patch_update_deps, _update_args,
)


@pytest.fixture
def hook_calls(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gateway_migrate, "maybe_auto_migrate_after_update", lambda: calls.append("hook"))
    return calls


def test_successful_update_runs_multiplex_hook_once(monkeypatch, tmp_path, hook_calls):
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())
    hermes_main.cmd_update(_update_args())
    assert hook_calls == ["hook"]


def test_incomplete_fleet_verification_skips_multiplex_hook(monkeypatch, tmp_path, hook_calls):
    """A fleet that may still run stale code is never migrated on top of the failure."""
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())
    monkeypatch.setattr("hermes_cli.update_receipt.print_fleet_version_matrix", lambda rows: True)
    with pytest.raises(SystemExit) as exc:
        hermes_main.cmd_update(_update_args())
    assert exc.value.code == 1
    assert hook_calls == []
