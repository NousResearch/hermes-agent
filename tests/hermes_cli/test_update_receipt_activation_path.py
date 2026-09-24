"""The ACTIVATION path (fleet restart → verify) must write the receipt the no-op path writes.

Issue #112558 finding 1: one real `hermes update` pulled 2204 commits, synced deps, built the
web UI and restarted the supervised gateway — and left NO receipt in
``~/.hermes/logs/update_receipts`` while the immediately following no-op run did. The write is
swallowed twice (``finalize_update_receipt``'s own ``except``, then
``update_cmd_common._best_effort``), so a failure is invisible and unrecoverable, and the
activation run's exit code (partial → 1) never reached disk because the inner finalize happens
before the ``sys.exit``.

Covers:
- the activation path (``_verify_fleet_after_update``) persists success / partial receipts,
  with the exit code and a ``latest.json`` pointer
- a failed write is WARNING-visible and leaves a durable ``*.json.failed`` record
- the ``_best_effort`` wrapper around the finalize is not DEBUG-only
"""

import json
import logging

import pytest

import hermes_cli.update_receipt as ur
from hermes_cli import update_cmd, update_cmd_fleet


@pytest.fixture()
def receipt_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME for receipt writes (same shape as test_update_receipt.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    # ``_receipt_dir`` resolves through ``hermes_constants.get_hermes_home`` (env var), not
    # ``hermes_cli.config`` — patch where production reads.
    monkeypatch.setenv("HERMES_HOME", str(home))
    ur._current = None
    yield home
    ur._current = None


@pytest.fixture()
def quiet_verify(monkeypatch):
    """Neutralize everything ``_verify_fleet_after_update`` does besides receipt bookkeeping."""
    monkeypatch.setattr(update_cmd_fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd, "_warn_stale_serve_runtimes", lambda rows: None)
    monkeypatch.setattr(update_cmd, "_fleet_probe_expected_runtimes", lambda *a, **k: False)
    monkeypatch.setattr(update_cmd_fleet, "_collect_fleet_snapshot", lambda restart, expected: [])
    monkeypatch.setattr(
        "hermes_cli.gateway_migrate.maybe_auto_migrate_after_update", lambda: None, raising=False
    )
    return None


def _restart_outcome(**overrides):
    kwargs = dict(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[],
        killed_pids=set(),
    )
    kwargs.update(overrides)
    return update_cmd_fleet._GatewayRestartOutcome(**kwargs)


def _latest(receipt_home):
    path = receipt_home / "logs" / "update_receipts" / "latest.json"
    return json.loads(path.read_text(encoding="utf-8")), path


class TestActivationPathReceipt:
    def test_success_is_persisted_with_exit_code(self, receipt_home, quiet_verify):
        restart = _restart_outcome(restarted_services=["hermes-gateway"])
        ur.begin_update_receipt()
        restart.record_receipt()  # what the restart phase does before verification
        update_cmd_fleet._verify_fleet_after_update(
            restart, _pre_update_plan=None, _windows_gateway_resume=None,
            node_failures=[], update_complete=True,
        )

        receipts = list((receipt_home / "logs" / "update_receipts").glob("update_*.json"))
        assert len(receipts) == 1, "the activation path must persist exactly one receipt"
        payload = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert payload["outcome"] == "success"
        assert payload["exit_code"] == 0
        assert payload["gateway_restart"]["restarted_services"] == ["hermes-gateway"]
        latest, latest_path = _latest(receipt_home)
        assert latest_path.is_file()
        assert latest == payload, "latest.json must point at the receipt just written"

    def test_partial_persists_exit_code_before_sys_exit(self, receipt_home, quiet_verify):
        """A stale fleet exits 1 — but the receipt has to land on disk FIRST (the field run
        left neither a receipt nor any record of the exit code)."""
        ur.begin_update_receipt()
        with pytest.raises(SystemExit) as exc:
            update_cmd_fleet._verify_fleet_after_update(
                _restart_outcome(incomplete=True, failed_or_stale_units=["hermes-gateway"]),
                _pre_update_plan=None, _windows_gateway_resume=None,
                node_failures=[], update_complete=True,
            )
        assert exc.value.code == 1

        receipts = list((receipt_home / "logs" / "update_receipts").glob("update_*.json"))
        assert len(receipts) == 1
        payload = json.loads(receipts[0].read_text(encoding="utf-8"))
        assert payload["outcome"] == "partial"
        assert payload["exit_code"] == 1
        latest, _ = _latest(receipt_home)
        assert latest["outcome"] == "partial" and latest["exit_code"] == 1


class TestWriteFailureIsVisible:
    def test_unresolvable_receipt_home_warns_instead_of_debug(
        self, receipt_home, monkeypatch, caplog
    ):
        """The field shape: writing the receipt blew up on the way to disk. That used to be
        swallowed at DEBUG."""
        def boom():
            raise ImportError("cannot import name 'file_signature' from 'utils'")

        monkeypatch.setattr(ur, "_receipt_dir", boom)
        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "2204 commits")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.update_receipt"):
            assert ur.finalize_update_receipt("partial") is None

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "a failed receipt write must not be a DEBUG-only event"
        assert "file_signature" in caplog.text

    def test_failed_write_leaves_a_queryable_record(self, receipt_home, monkeypatch, caplog):
        """A write that dies mid-flight must leave something to read afterwards."""
        def boom(path, body):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(ur, "_write_receipt_body", boom, raising=False)
        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "2204 commits")

        with caplog.at_level(logging.WARNING, logger="hermes_cli.update_receipt"):
            assert ur.finalize_update_receipt("partial") is None

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] or caplog.text, (
            "the failure must be logged above DEBUG"
        )
        sidecars = list((receipt_home / "logs" / "update_receipts").glob("*.json.failed"))
        assert len(sidecars) == 1, "the failed receipt must be recoverable from disk"
        payload = json.loads(sidecars[0].read_text(encoding="utf-8"))
        assert payload["outcome"] == "partial"
        assert payload["exit_code"] == 1
        assert payload["steps"][0]["name"] == "git_pull"
        assert "No space left" in payload["receipt_error"]
        # A broken run must not be advertised as the current state.
        assert not (receipt_home / "logs" / "update_receipts" / "latest.json").exists()

    def test_outer_best_effort_layer_is_not_debug_only(self, monkeypatch, caplog):
        """`update_cmd._finalize_receipt` wraps the call in `_best_effort` (DEBUG before)."""
        def boom(status):
            raise OSError("receipt directory unwritable")

        monkeypatch.setattr(ur, "finalize_update_receipt", boom)
        with caplog.at_level(logging.WARNING, logger="hermes_cli.update_cmd"):
            update_cmd._finalize_receipt("partial", "Update receipt finalize failed: %s")

        assert [r for r in caplog.records if r.levelno >= logging.WARNING], caplog.text
