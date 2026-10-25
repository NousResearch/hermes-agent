"""Maintenance must refuse before opening a writer and never report failed work as success.

All stores/paths are synthetic; optimizer and holder outcomes are injected at their
production seams. Subprocess tests exercise the actual main() exit-code adapter.
"""

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import sessions_cmd as sc


def _run(args):
    try:
        return sc.cmd_sessions(args)
    except KeyboardInterrupt:
        pytest.fail("maintenance interruption must return the explicit CLI status 130")


def _args(action="optimize-storage", **overrides):
    values = dict(sessions_action=action, force=False, dry_run=False, yes=True, no_vacuum=False)
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def maintenance(monkeypatch, tmp_path):
    import hermes_state
    import hermes_state_holders
    from hermes_cli import sessions_owner

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = Mock(db_path=tmp_path / "state.db")
    db.db_path.write_bytes(b"synthetic size fixture")
    db.fts_optimize_available.side_effect = [True, False]
    db.optimize_fts_storage.return_value = {"ok": True, "vacuumed": True}
    db.vacuum.return_value = 0  # zero indexes is valid; it is not a failure boolean
    db.logical_size_bytes.return_value = 0
    ctor = Mock(return_value=db)
    scan = Mock(return_value=None)
    monkeypatch.setattr(hermes_state, "SessionDB", ctor)
    monkeypatch.setattr(hermes_state_holders, "held_store_refusal", scan)
    monkeypatch.setattr(sessions_owner, "live_owner", lambda: None)
    monkeypatch.setattr(sc.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**9))
    return SimpleNamespace(db=db, ctor=ctor, scan=scan)


@pytest.mark.parametrize("action", ["optimize", "optimize-storage", "prune"])
def test_refusal_precedes_any_writer_initialization(maintenance, action, capsys):
    maintenance.scan.return_value = "synthetic holder refusal"
    handler = Mock()
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sc._DB_HANDLERS, action, handler)
        assert _run(_args(action)) == 1
    maintenance.ctor.assert_not_called()
    handler.assert_not_called()
    assert "synthetic holder refusal" in capsys.readouterr().out


@pytest.mark.parametrize("action", ["optimize", "optimize-storage", "prune"])
def test_second_scan_catches_holder_appearing_during_open(maintenance, action, monkeypatch):
    events = []
    maintenance.scan.side_effect = lambda *_a, **_k: events.append("scan") or (
        "new holder" if events.count("scan") == 2 else None
    )
    maintenance.ctor.side_effect = lambda **_k: events.append("open") or maintenance.db
    handler = Mock()
    monkeypatch.setitem(sc._DB_HANDLERS, action, handler)
    assert _run(_args(action)) == 1
    assert events == ["scan", "open", "scan"]
    handler.assert_not_called()
    maintenance.db.close.assert_called_once()


def test_scan_exception_fails_closed_before_open(maintenance, capsys):
    maintenance.scan.side_effect = OSError("synthetic scan failure")
    assert _run(_args()) == 1
    maintenance.ctor.assert_not_called()
    assert "synthetic scan failure" in capsys.readouterr().out


def test_force_retains_explicit_override(maintenance, monkeypatch):
    handler = Mock(return_value=0)
    monkeypatch.setitem(sc._DB_HANDLERS, "optimize", handler)
    assert _run(_args("optimize", force=True)) == 0
    maintenance.scan.assert_not_called()
    maintenance.ctor.assert_called_once_with(db_path=maintenance.db.db_path, read_only=False)
    handler.assert_called_once()


def test_prune_preview_opens_read_only_without_holder_override(maintenance, monkeypatch):
    handler = Mock(return_value=0)
    monkeypatch.setitem(sc._DB_HANDLERS, "prune", handler)
    assert _run(_args("prune", dry_run=True)) == 0
    maintenance.ctor.assert_called_once_with(db_path=maintenance.db.db_path, read_only=True)
    maintenance.scan.assert_not_called()


def test_owner_prune_stays_read_only(maintenance, monkeypatch):
    from hermes_cli import sessions_owner
    monkeypatch.setattr(sessions_owner, "live_owner", lambda: object())
    monkeypatch.setitem(sc._DB_HANDLERS, "prune", Mock(return_value=0))
    assert _run(_args("prune")) == 0
    maintenance.ctor.assert_called_once_with(db_path=maintenance.db.db_path, read_only=True)
    maintenance.scan.assert_not_called()


@pytest.mark.parametrize("result", [
    {"ok": False, "reason": "backfill_incomplete"},
    {"ok": True, "vacuumed": False},
    {"ok": True, "vacuumed": None},
    {"ok": True},
    {"ok": "yes", "vacuumed": True},
    None,
])
def test_storage_failure_returns_nonzero_without_success_banner(maintenance, result, capsys):
    maintenance.db.optimize_fts_storage.return_value = result
    assert _run(_args()) == 1
    assert "✓ Search index optimized." not in capsys.readouterr().out
    maintenance.db.close.assert_called_once()


@pytest.mark.parametrize("failure", [RuntimeError("synthetic optimizer failure"), KeyboardInterrupt()])
def test_storage_exception_exit_codes(maintenance, failure, capsys):
    maintenance.db.optimize_fts_storage.side_effect = failure
    assert _run(_args()) == (130 if isinstance(failure, KeyboardInterrupt) else 1)
    output = capsys.readouterr().out
    assert "✓ Search index optimized." not in output
    assert "No data was lost" not in output  # exceptions do not establish this postcondition
    maintenance.db.close.assert_called_once()


@pytest.mark.parametrize("postcondition", [True, None, 0, "", RuntimeError("synthetic verification failure")])
def test_storage_requires_completed_postcondition(maintenance, postcondition, capsys):
    maintenance.db.fts_optimize_available.side_effect = [True, postcondition]
    assert _run(_args()) == 1
    assert "✓ Search index optimized." not in capsys.readouterr().out
    maintenance.db.close.assert_called_once()


@pytest.mark.parametrize("no_vacuum", [False, True])
def test_storage_verified_success_is_explicit(maintenance, no_vacuum, capsys):
    maintenance.db.optimize_fts_storage.return_value = {
        "ok": True, "vacuumed": None if no_vacuum else True,
    }
    assert _run(_args(no_vacuum=no_vacuum)) == 0
    assert maintenance.db.fts_optimize_available.call_count == 2
    assert maintenance.db.optimize_fts_storage.call_args.kwargs["vacuum"] is not no_vacuum
    assert "✓ Search index optimized." in capsys.readouterr().out


def test_storage_no_work_is_explicit_noop(maintenance):
    maintenance.db.fts_optimize_available.side_effect = [False]
    assert _run(_args()) == 0
    maintenance.db.optimize_fts_storage.assert_not_called()


@pytest.mark.parametrize("unknown", [None, 0, ""])
def test_storage_unknown_availability_is_not_success(maintenance, unknown):
    maintenance.db.fts_optimize_available.side_effect = [unknown]
    assert _run(_args()) == 1
    maintenance.db.optimize_fts_storage.assert_not_called()


def test_storage_insufficient_disk_is_failure(maintenance, monkeypatch):
    monkeypatch.setattr(sc.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert _run(_args()) == 1
    maintenance.db.optimize_fts_storage.assert_not_called()


@pytest.mark.parametrize("answer", ["n", EOFError(), KeyboardInterrupt()])
def test_prompt_cancellation_does_not_claim_completion(maintenance, monkeypatch, answer, capsys):
    monkeypatch.setattr("builtins.input", Mock(side_effect=answer) if isinstance(answer, BaseException)
                        else Mock(return_value=answer))
    assert _run(_args(yes=False)) == (130 if isinstance(answer, KeyboardInterrupt) else 0)
    maintenance.db.optimize_fts_storage.assert_not_called()
    assert "✓ Search index optimized." not in capsys.readouterr().out


def test_storage_rechecks_holder_after_confirmation(maintenance, monkeypatch):
    def confirm(_prompt):
        maintenance.scan.return_value = "holder arrived while waiting for confirmation"
        return "yes"
    monkeypatch.setattr("builtins.input", confirm)
    assert _run(_args(yes=False)) == 1
    maintenance.db.optimize_fts_storage.assert_not_called()


@pytest.mark.parametrize("outcome", [RuntimeError("synthetic vacuum failure"), False, None])
def test_optimize_failure_is_nonzero(maintenance, outcome, capsys):
    if isinstance(outcome, Exception):
        maintenance.db.vacuum.side_effect = outcome
    else:
        maintenance.db.vacuum.return_value = outcome
    assert _run(_args("optimize")) == 1
    assert "Optimized " not in capsys.readouterr().out


def test_optimize_zero_indexes_is_explicit_success(maintenance):
    assert _run(_args("optimize")) == 0


@pytest.mark.parametrize("mode, expected", [
    ("refused", 1), ("exception", 1), ("postcondition", 1), ("vacuum", 1),
    ("success", 0), ("cancel", 0), ("interrupt", 130),
])
def test_main_propagates_maintenance_exit_code(tmp_path, mode, expected):
    # Use main(), not a test-only `sys.exit(cmd_sessions(...) or 0)` adapter.
    script = '''
import sys
from pathlib import Path
from unittest.mock import Mock
import hermes_state
import hermes_state_holders
from hermes_cli import sessions_cmd as sc
mode = sys.argv[1]
db = Mock(db_path=Path(__import__('os').environ['HERMES_HOME']) / 'state.db')
db.fts_optimize_available.side_effect = [True, mode == 'postcondition']
db.optimize_fts_storage.return_value = {'ok': mode != 'refused', 'vacuumed': mode != 'vacuum'}
db.logical_size_bytes.return_value = 0
if mode == 'exception':
    db.optimize_fts_storage.side_effect = RuntimeError('synthetic failure')
if mode == 'interrupt':
    db.optimize_fts_storage.side_effect = KeyboardInterrupt()
hermes_state.SessionDB = Mock(return_value=db)
hermes_state_holders.held_store_refusal = Mock(return_value=None)
import hermes_cli.main as cli
sys.argv = ['hermes', 'sessions', 'optimize-storage'] + ([] if mode == 'cancel' else ['--yes'])
cli.main()
'''
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(home))
    proc = subprocess.run([sys.executable, "-c", script, mode], input="n\n", text=True,
                          capture_output=True, env=env, timeout=60, check=False,
                          cwd=Path(__file__).resolve().parents[2])
    assert proc.returncode == expected, proc.stdout + proc.stderr
    if expected != 0 or mode == "cancel":
        assert "✓ Search index optimized." not in proc.stdout
    assert not (home / "state.db").exists()
