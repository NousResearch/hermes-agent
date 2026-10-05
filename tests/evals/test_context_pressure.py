from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shlex

from evals.context_pressure.runner import run_one
from evals.context_pressure.runner import _format_command
from evals.context_pressure.tasks import (
    EVIDENCE_COUNT,
    create_distributed_evidence_workspace,
    validate_distributed_evidence,
)


def _write_valid_artifacts(workspace: Path) -> None:
    rows = []
    for path in sorted((workspace / "evidence").glob("*.txt")):
        evidence_id = path.stem
        index = int(evidence_id)
        rows.append({
            "id": evidence_id,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "status": "UNVERIFIED" if index % 7 == 0 else "VERIFIED",
            "observation": f"Processed {evidence_id}",
        })
    assert len(rows) == EVIDENCE_COUNT
    (workspace / "INSPECTION_LEDGER.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )
    distractors = ", ".join(f"{index:03d} unverified" for index in range(7, 51, 7))
    (workspace / "LONG_REPORT.md").write_text(
        "combined conclusion: ANCHOR=ORBIT-7, CONTROL=MIDDLE-42, "
        "CHECKSUM=LATE-91. All evidence was processed. "
        f"Rejected distractors: {distractors}.\n",
        encoding="utf-8",
    )


def test_distributed_evidence_requires_complete_ledger(tmp_path: Path) -> None:
    create_distributed_evidence_workspace(tmp_path)
    _write_valid_artifacts(tmp_path)

    result = validate_distributed_evidence(tmp_path)

    assert result.passed
    assert all(result.checks.values())

    ledger = json.loads((tmp_path / "INSPECTION_LEDGER.json").read_text())
    ledger.pop()
    (tmp_path / "INSPECTION_LEDGER.json").write_text(json.dumps(ledger))
    result = validate_distributed_evidence(tmp_path)
    assert not result.passed
    assert not result.checks["ledger_shape"]


def test_distributed_evidence_rejects_tampered_hash_and_distractor(
    tmp_path: Path,
) -> None:
    create_distributed_evidence_workspace(tmp_path)
    _write_valid_artifacts(tmp_path)

    ledger = json.loads((tmp_path / "INSPECTION_LEDGER.json").read_text())
    ledger[0]["sha256"] = "0" * 64
    (tmp_path / "INSPECTION_LEDGER.json").write_text(json.dumps(ledger))
    report = (tmp_path / "LONG_REPORT.md").read_text()
    (tmp_path / "LONG_REPORT.md").write_text(
        report.replace("007 unverified", "007 verified")
    )

    result = validate_distributed_evidence(tmp_path)

    assert not result.passed
    assert not result.checks["ledger_hashes"]
    assert not result.checks["distractors_rejected"]


def test_runner_preserves_timeout_classification(tmp_path: Path) -> None:
    result = run_one(
        arm="sleepy",
        command_template='{python} -c "import time; time.sleep(1)"',
        repetition=1,
        timeout=0.05,
        model=None,
        provider=None,
        config=None,
        root_output=tmp_path,
    )

    assert result["timed_out"] is True
    assert result["return_code"] == 124
    assert result["validated"] is False
    assert Path(result["result_file"]).is_file()


def test_runner_quotes_multiword_command_values() -> None:
    command = _format_command(
        "{python} -c {prompt}",
        {
            "python": shlex.quote("/usr/bin/python3"),
            "prompt": shlex.quote("a multi-word prompt"),
        },
    )

    assert command == ["/usr/bin/python3", "-c", "a multi-word prompt"]
