from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("HERMES_REPO_ROOT", "/Users/ericfreeman/.hermes/hermes-agent"))
NOW = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)


def _uv_python(code: str, *args: str) -> dict:
    env = dict(os.environ)
    env["HERMES_REPO_ROOT"] = str(REPO_ROOT)
    result = subprocess.run(
        ["uv", "run", "python", "-c", code, str(SCRIPTS_DIR), *args],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return json.loads(result.stdout)


def test_cron_and_agent_mutations_append_same_schema(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    code = r'''
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from torben_mutation_spine import list_mutations, record_mutation

ledger = Path(sys.argv[2])
now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
record_mutation(
    ledger_path=ledger,
    category="gmail_archive",
    executor="cron",
    risk_class="low",
    summary="Archive stale receipt",
    undo_pointer="gmail:restore-inbox:thread-1",
    surface="gmail",
    now=now,
)
record_mutation(
    ledger_path=ledger,
    category="calendar_edit",
    executor="agent",
    risk_class="medium",
    summary="Create private busy block",
    undo_pointer="calendar:delete:block-1",
    surface="google-calendar",
    now=now,
)
print(json.dumps({"records": list_mutations(ledger_path=ledger)}, sort_keys=True))
'''
    payload = _uv_python(code, str(ledger))

    records = payload["records"]
    assert [record["executor"] for record in records] == ["cron", "agent"]
    assert {tuple(sorted(record.keys())) for record in records} == {
        ("category", "executor", "handle", "risk_class", "status", "surface", "undo")
    }
    assert records[0]["undo"] == "gmail:restore-inbox:thread-1"
    assert records[1]["risk_class"] == "medium"


def test_no_undo_surface_cannot_reach_auto_within_caps(tmp_path: Path) -> None:
    code = r'''
import json
import sys
from pathlib import Path

scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from torben_mutation_spine import evaluate_promotion_undo_gate

print(json.dumps({
    "blocked": evaluate_promotion_undo_gate(category="payment_adjacent", target_rung="auto_within_caps"),
    "allowed_lower": evaluate_promotion_undo_gate(category="payment_adjacent", target_rung="approve_each"),
    "allowed_undo": evaluate_promotion_undo_gate(category="gmail_trash", target_rung="auto_within_caps"),
}, sort_keys=True))
'''
    payload = _uv_python(code)

    assert payload["blocked"]["status"] == "refused"
    assert payload["blocked"]["reason"] == "surface_has_no_real_undo"
    assert payload["allowed_lower"]["status"] == "allowed"
    assert payload["allowed_undo"]["status"] == "allowed"


def test_cli_records_mutation_and_lists_view(tmp_path: Path) -> None:
    ledger = tmp_path / "torben-action-ledger.jsonl"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPTS_DIR / "torben_mutation_spine.py"),
            "--ledger",
            str(ledger),
            "--json",
            "record",
            "--category",
            "gmail_archive",
            "--executor",
            "cron",
            "--risk-class",
            "low",
            "--summary",
            "Archive stale receipt",
            "--undo",
            "gmail:restore-inbox:thread-2",
            "--surface",
            "gmail",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, "HERMES_REPO_ROOT": str(REPO_ROOT)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    record_payload = json.loads(result.stdout)

    listed = _uv_python(
        r'''
import json
import sys
from pathlib import Path
scripts_dir = Path(sys.argv[1])
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from torben_mutation_spine import list_mutations
print(json.dumps({"records": list_mutations(ledger_path=Path(sys.argv[2]))}, sort_keys=True))
''',
        str(ledger),
    )

    assert record_payload["record"]["executor_state"]["schema"] == "torben.mutation-spine.v1"
    assert listed["records"][0]["executor"] == "cron"
