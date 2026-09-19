"""Post-RESTART state.db integrity verification (#110007).

The maintenance-phase guard (``_verify_and_restore_state_dbs_post_update``) runs
while the OLD gateway may still hold the DB; both reported production corruptions
were born in the drain→restart handoff and passed that earlier guard. The sweep
added here runs on a fresh connection AFTER ``_restart_gateway_fleet_after_update``,
records per-home pass/fail in the update receipt, and reuses the snapshot-restore
remediation instead of adopting a corrupt DB silently.
"""

import sqlite3
from pathlib import Path

import pytest

import hermes_cli.update_receipt as update_receipt
from hermes_cli.update_cmd_maint import _verify_state_dbs_after_fleet_restart


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _make_valid_state_db(home: Path) -> Path:
    state = home / "state.db"
    conn = sqlite3.connect(state)
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    return state


def _make_corrupt_state_db(home: Path) -> Path:
    """A file that fails integrity_check but keeps SQLite header magic: truncate a
    valid database mid-page so the header passes and the page walk fails."""
    state = _make_valid_state_db(home)
    data = state.read_bytes()
    # Keep the 100-byte header + a torn body: the header check passes, PRAGMA
    # integrity_check fails (invalid page geometry / truncated file).
    state.write_bytes(data[: len(data) // 3])
    return state


class TestPostRestartSweep:
    def test_valid_home_records_ok_step(self, _isolated_hermes_home, monkeypatch):
        _make_valid_state_db(_isolated_hermes_home)
        recorded = []
        monkeypatch.setattr(
            "hermes_cli.update_receipt.record_step",
            lambda name, ok, detail="": recorded.append((name, ok, detail)),
        )
        from hermes_cli.update_cmd import _record_update_step
        monkeypatch.setattr(
            "hermes_cli.update_cmd._record_update_step",
            lambda step, ok, detail="": recorded.append((step, ok, detail)),
        )
        # No receipt active; the sweep still runs and records via the patched step fn.
        _verify_state_dbs_after_fleet_restart()
        assert any(step == "post_restart_state_db_integrity" and ok for step, ok, _ in recorded)

    def test_corrupt_home_records_failure_and_attempts_restore(
        self, _isolated_hermes_home, monkeypatch, capsys
    ):
        _make_corrupt_state_db(_isolated_hermes_home)
        recorded = []
        from hermes_cli.update_cmd import _record_update_step
        monkeypatch.setattr(
            "hermes_cli.update_cmd._record_update_step",
            lambda step, ok, detail="": recorded.append((step, ok, detail)),
        )
        restore_calls = []
        monkeypatch.setattr(
            "hermes_cli.update_cmd_maint._verify_and_restore_one_state_db",
            lambda home, label: restore_calls.append((home, label)),
        )
        _verify_state_dbs_after_fleet_restart()
        # A failed integrity result is recorded as a failed receipt step.
        assert any(
            step == "post_restart_state_db_integrity" and not ok for step, ok, _ in recorded
        )
        # And the snapshot-restore remediation ran for the corrupt home (never adopted silently).
        assert restore_calls and restore_calls[0][1] == "default home"
        # A clear warning was printed.
        assert "post-restart integrity check" in capsys.readouterr().out

    def test_missing_state_db_is_not_a_failure(self, _isolated_hermes_home, monkeypatch):
        # No state.db in the home: nothing to verify, no step recorded, no crash.
        from hermes_cli.update_cmd import _record_update_step
        recorded = []
        monkeypatch.setattr(
            "hermes_cli.update_cmd._record_update_step",
            lambda step, ok, detail="": recorded.append((step, ok, detail)),
        )
        _verify_state_dbs_after_fleet_restart()
        assert recorded == []

    def test_guard_never_raises_on_verifier_error(self, _isolated_hermes_home, monkeypatch):
        _make_valid_state_db(_isolated_hermes_home)
        from hermes_cli.backup import verify_sqlite_integrity

        def _boom(path, **kw):
            raise RuntimeError("verifier exploded")

        monkeypatch.setattr("hermes_cli.backup.verify_sqlite_integrity", _boom)
        # Must record the failure via receipt and return, not raise.
        _verify_state_dbs_after_fleet_restart()

    def test_wired_into_fleet_verification_after_version_matrix(self, monkeypatch, _isolated_hermes_home):
        """The sweep must run inside _verify_fleet_after_update BEFORE the receipt
        finalizes, so results persist in the finalized receipt (#110007)."""
        from hermes_cli import update_cmd_fleet

        called_after_matrix = []
        order = []

        def _fake_verify(restart, **kw):
            order.append("fleet_verify")
            return []

        monkeypatch.setattr(update_cmd_fleet, "_collect_fleet_snapshot", _fake_verify)
        monkeypatch.setattr(
            "hermes_cli.update_receipt.print_fleet_version_matrix", lambda rows: False
        )
        sweep = []

        def _fake_sweep():
            order.append("integrity_sweep")

        monkeypatch.setattr(
            "hermes_cli.update_cmd_maint._verify_state_dbs_after_fleet_restart", _fake_sweep
        )
        finalize_order = []

        def _fake_finalize(outcome, fleet=None, **kw):
            finalize_order.append(outcome)
            order.append("receipt_finalize")
            return Path("receipt.json")

        monkeypatch.setattr(
            "hermes_cli.update_receipt.finalize_update_receipt", _fake_finalize
        )

        class _Restart:
            restarted_services = []
            relaunched_profiles = []
            externally_supervised_profiles = []
            killed_pids = []
            failed_or_stale_units = []
            incomplete = False

            def fleet_probe_signals(self):
                return None, []

        # Stub every earlier phase to no-ops. _finish_dashboard_update_cleanup and
        # _surviving_pre_update_serve_runtimes are imported INSIDE the function from
        # hermes_cli.update_cmd, so patch them at their source module.
        import hermes_cli.update_cmd as update_cmd
        monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
        monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: None)
        monkeypatch.setattr(update_cmd, "_warn_stale_serve_runtimes", lambda rows: None)
        monkeypatch.setattr(update_cmd, "_fleet_probe_expected_runtimes", lambda *a, **k: None)
        # get_hermes_home must still return a real Path: the tail of the function
        # clears the fleet-restart-pending marker under it.
        monkeypatch.setattr(
            update_cmd, "get_hermes_home", lambda: Path(_isolated_hermes_home)
        )

        update_cmd_fleet._verify_fleet_after_update(
            _Restart(), _pre_update_plan=None, _windows_gateway_resume=None,
            node_failures=[], update_complete=True,
        )
        assert order.index("integrity_sweep") < order.index("receipt_finalize")
