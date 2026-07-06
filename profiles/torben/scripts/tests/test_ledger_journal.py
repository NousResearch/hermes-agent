from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
NOW = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
LEDGER_STEM = "torben-action-ledger"


def _uv_python(args: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    return subprocess.run(
        ["uv", "run", "python", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _legacy_record(handle: str, *, status: str) -> dict:
    return {
        "handle": handle,
        "scope": "EA",
        "summary": f"Summary for {handle}",
        "evidence_ids": ["evidence-1"],
        "allowed_next_actions": ["approve", "discard"],
        "status": status,
        "risk_class": "medium",
        "outbound_message_id": None,
        "created_at": "2026-07-05T11:00:00Z",
        "expires_at": None,
        "user_visible_summary": f"Visible {handle}",
        "executor_state": {"mutation_type": "gmail_hygiene"},
        "resolution_history": [],
    }


def test_migration_demotes_approved_records_and_writes_report(tmp_path: Path) -> None:
    legacy = tmp_path / f"{LEDGER_STEM}.json"
    legacy.write_text(
        json.dumps(
            [
                _legacy_record("EA-20260705-001", status="approved"),
                _legacy_record("EA-20260705-002", status="approval_required"),
                _legacy_record("GTM-20260705-001", status="staged"),
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    journal = tmp_path / f"{LEDGER_STEM}.jsonl"
    report_path = tmp_path / "torben-ledger-migration-report.json"

    result = _uv_python(
        [
            str(SCRIPTS_DIR / "torben_ledger_migrate.py"),
            "--legacy",
            str(legacy),
            "--journal",
            str(journal),
            "--report",
            str(report_path),
            "--apply",
            "--json",
        ]
    )
    report = json.loads(result.stdout)

    assert report["status"] == "migrated"
    assert report["legacy_record_count"] == 3
    assert report["journal_record_count"] == 3
    assert report["record_loss"] == 0
    assert report["approved_demoted"] == 1
    assert report["approved_remaining"] == 0
    assert report_path.exists()
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 3

    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    migrated = {record["handle"]: record for record in records}
    assert migrated["EA-20260705-001"]["status"] == "approval_required"
    assert migrated["EA-20260705-001"]["resolution_history"][-1]["previous_status"] == "approved"
    assert (tmp_path / f"{LEDGER_STEM}.snapshot.json").exists()


def test_journal_hold_fails_then_retry_appends_once(tmp_path: Path) -> None:
    code = r'''
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from hermes_cli.signal_coo.action_ledger import ActionLedger, LedgerMigrationHoldError

ledger = ActionLedger(Path(sys.argv[1]))
now = datetime(2026, 7, 5, 12, 0, tzinfo=timezone.utc)
ledger.hold_path.write_text("migration in progress\n", encoding="utf-8")
try:
    ledger.add_action(scope="EA", summary="queued by caller retry", now=now)
except LedgerMigrationHoldError:
    pass
else:
    raise SystemExit("write unexpectedly succeeded during hold")
ledger.hold_path.unlink()
record = ledger.add_action(scope="EA", summary="queued by caller retry", now=now)
print(json.dumps({"handle": record.handle, "records": len(ledger.load()), "lines": len(ledger.path.read_text(encoding="utf-8").splitlines())}))
'''
    result = _uv_python(["-c", code, str(tmp_path / f"{LEDGER_STEM}.jsonl")])
    payload = json.loads(result.stdout)

    assert payload == {"handle": "EA-20260705-001", "records": 1, "lines": 1}
