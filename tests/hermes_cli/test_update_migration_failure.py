"""Automatic migration must finish before updater success bookkeeping."""
from unittest.mock import Mock

import pytest

from hermes_cli import gateway_migrate, update_cmd, update_cmd_fleet, update_receipt


def test_failed_migration_writes_partial_receipt_and_keeps_pending_marker(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(update_cmd_fleet, '_print_legacy_units_warning', lambda: None)
    monkeypatch.setattr(update_cmd, '_finish_dashboard_update_cleanup', lambda *a, **kw: None)
    monkeypatch.setattr(update_cmd, '_surviving_pre_update_serve_runtimes', lambda *a: [])
    monkeypatch.setattr(update_cmd_fleet, '_collect_fleet_snapshot', lambda *a: [])
    monkeypatch.setattr(update_receipt, 'print_fleet_version_matrix', lambda *a: False)
    finalize = Mock()
    monkeypatch.setattr(update_receipt, 'finalize_update_receipt', finalize)
    monkeypatch.setattr(gateway_migrate, 'maybe_auto_migrate_after_update', Mock(side_effect=RuntimeError('migration failed')))
    marker = tmp_path / 'fleet_restart_pending'
    marker.write_text('pending')
    restart = update_cmd_fleet._GatewayRestartOutcome(False, [], [], [], [], [], [], set())
    with pytest.raises(SystemExit) as exc:
        update_cmd_fleet._verify_fleet_after_update(restart, _pre_update_plan=None, _windows_gateway_resume=None,
                                                   node_failures=[], update_complete=True)
    assert exc.value.code == 1
    assert finalize.call_args.args == ('partial',)
    assert marker.exists()
