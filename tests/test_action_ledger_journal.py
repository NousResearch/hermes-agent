from __future__ import annotations

from datetime import datetime, timezone
import importlib
from multiprocessing import Process, Queue
from pathlib import Path

import pytest

ledger_module = importlib.import_module("hermes_cli.signal_coo." + "action_ledger")
ActionLedger = ledger_module.ActionLedger
LedgerMigrationHoldError = ledger_module.LedgerMigrationHoldError


NOW = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)


def _append_action(path: str, index: int, queue: Queue) -> None:
    try:
        ledger = ActionLedger(Path(path))
        record = ledger.add_action(scope="EA", summary=f"item {index}", now=NOW)
        queue.put(record.handle)
    except Exception as exc:  # pragma: no cover - surfaced by parent assertion
        queue.put(f"ERROR: {type(exc).__name__}: {exc}")
        raise


def test_jsonl_journal_appends_changed_records_and_rebuilds_snapshot(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "actions.jsonl")
    record = ledger.add_action(scope="EA", summary="Draft a reply", now=NOW)
    record.status = "approval_required"
    record.resolution_history.append(
        {
            "at": "2026-07-05T12:01:00Z",
            "status": "approval_requested",
            "reason": "Operator requested explicit approval.",
        }
    )

    ledger.save([record])

    assert len((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    loaded = ledger.load()
    assert len(loaded) == 1
    assert loaded[0].status == "approval_required"
    assert loaded[0].resolution_history[-1]["status"] == "approval_requested"
    assert ledger.snapshot_path.exists()

    snapshot_text = ledger.snapshot_path.read_text(encoding="utf-8")
    ledger.snapshot_path.write_text(snapshot_text.replace("approval_required", "hand_edited"), encoding="utf-8")
    assert ledger.load()[0].status == "approval_required"


def test_torben_json_path_redirects_to_existing_journal(tmp_path: Path) -> None:
    journal = tmp_path / "torben-action-ledger.jsonl"
    ActionLedger(journal).add_action(scope="EA", summary="journal source", now=NOW)

    loaded = ActionLedger(tmp_path / "torben-action-ledger.json").load()

    assert len(loaded) == 1
    assert loaded[0].summary == "journal source"


def test_journal_hold_fails_loudly_before_writing(tmp_path: Path) -> None:
    ledger = ActionLedger(tmp_path / "torben-action-ledger.jsonl")
    ledger.hold_path.write_text("migration in progress\n", encoding="utf-8")

    with pytest.raises(LedgerMigrationHoldError):
        ledger.add_action(scope="EA", summary="blocked during migration", now=NOW)

    assert not ledger.path.exists()
    ledger.hold_path.unlink()
    ledger.add_action(scope="EA", summary="retried after migration", now=NOW)
    assert len(ledger.load()) == 1


def test_journal_concurrent_appends_do_not_lose_updates(tmp_path: Path) -> None:
    journal = tmp_path / "actions.jsonl"
    queue: Queue = Queue()
    processes = [Process(target=_append_action, args=(str(journal), index, queue)) for index in range(8)]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    handles = [queue.get(timeout=1) for _ in processes]
    assert not any(str(handle).startswith("ERROR:") for handle in handles)
    assert len(set(handles)) == len(processes)
    assert len(ActionLedger(journal).load()) == len(processes)
    assert len(journal.read_text(encoding="utf-8").splitlines()) == len(processes)
