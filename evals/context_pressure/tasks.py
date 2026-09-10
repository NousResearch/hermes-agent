"""Portable context-pressure fixtures and strict validators.

The fixture is intentionally generated at run time.  It stays small in the
repository while still requiring an agent to process a long, distributed set
of observations.  The ledger and report are checked independently; an
assistant claiming to have read every fragment cannot make a run pass without
producing the corresponding evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVIDENCE_COUNT = 50
ANCHORS = {
    "ANCHOR": "ORBIT-7",
    "CONTROL": "MIDDLE-42",
    "CHECKSUM": "LATE-91",
}


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable outcome for one fixture validation."""

    passed: bool
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": dict(self.checks),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DistributedEvidenceTask:
    """Task metadata shared by command runners and reports."""

    task_id: str = "distributed_evidence"
    evidence_count: int = EVIDENCE_COUNT

    @property
    def prompt(self) -> str:
        return (
            "Inspect every file under evidence/ using the terminal. Read all 50 "
            "files, not just the anchor files. Write INSPECTION_LEDGER.json as a "
            "JSON array with exactly one entry per file containing id, sha256, "
            "status, and a short observation. Write LONG_REPORT.md containing the "
            "verified early anchor ANCHOR=ORBIT-7, middle control "
            "CONTROL=MIDDLE-42, and late checksum CHECKSUM=LATE-91. Explain the "
            "combined conclusion and identify the unverified distractors as "
            "unverified; never promote them to verified facts."
        )


def _evidence_text(index: int) -> str:
    status = "UNVERIFIED" if index % 7 == 0 else "VERIFIED"
    lines = [
        f"ID={index:03d}",
        f"STATUS={status}",
        f"OBSERVATION=Evidence fragment {index:03d} was processed for the investigation.",
    ]
    if index == 1:
        lines.append("ANCHOR=ORBIT-7")
    if index == 25:
        lines.append("CONTROL=MIDDLE-42")
    if index == 50:
        lines.append("CHECKSUM=LATE-91")
    if status == "UNVERIFIED":
        lines.append("DISTRACTOR=Do not treat this fragment as a verified fact.")
    return "\n".join(lines) + "\n"


def create_distributed_evidence_workspace(workspace: Path) -> DistributedEvidenceTask:
    """Create a fresh deterministic evidence task under ``workspace``."""

    evidence_dir = workspace / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, EVIDENCE_COUNT + 1):
        (evidence_dir / f"{index:03d}.txt").write_text(
            _evidence_text(index), encoding="utf-8"
        )

    task = DistributedEvidenceTask()
    (workspace / "TASK.md").write_text(
        "# Distributed evidence investigation\n\n" + task.prompt + "\n",
        encoding="utf-8",
    )
    return task


def _expected_entries(workspace: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for index in range(1, EVIDENCE_COUNT + 1):
        evidence_id = f"{index:03d}"
        path = workspace / "evidence" / f"{evidence_id}.txt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries[evidence_id] = {
            "sha256": digest,
            "status": "UNVERIFIED" if index % 7 == 0 else "VERIFIED",
        }
    return entries


def _report_rejects_unverified(report: str, evidence_id: str) -> bool:
    """Require an explicit negative treatment of each known distractor."""

    for line in report.splitlines():
        for fragment in line.lower().replace(";", ",").split(","):
            if evidence_id not in fragment:
                continue
            if "unverified" in fragment or "not verified" in fragment:
                return True
            if any(marker in fragment for marker in ("distractor", "rejected")):
                return "verified" not in fragment
    return False


def validate_distributed_evidence(workspace: Path) -> ValidationResult:
    """Validate the complete evidence artifact, never trusting self-report."""

    checks: dict[str, bool] = {}
    errors: list[str] = []
    try:
        expected = _expected_entries(workspace)
        checks["fixture_complete"] = True
    except OSError as exc:
        expected = {}
        checks["fixture_complete"] = False
        errors.append(f"fixture is incomplete: {exc}")
    ledger_path = workspace / "INSPECTION_LEDGER.json"
    report_path = workspace / "LONG_REPORT.md"

    if not ledger_path.is_file():
        checks["ledger_exists"] = False
        errors.append("INSPECTION_LEDGER.json is missing")
        ledger: object = None
    else:
        checks["ledger_exists"] = True
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            ledger = None
            errors.append(f"ledger is not valid JSON: {exc}")

    rows = ledger if isinstance(ledger, list) else []
    row_by_id = {
        str(row.get("id")): row for row in rows if isinstance(row, dict) and "id" in row
    }
    checks["ledger_shape"] = (
        isinstance(ledger, list)
        and len(rows) == EVIDENCE_COUNT
        and len(row_by_id) == EVIDENCE_COUNT
        and set(row_by_id) == set(expected)
    )
    if not checks["ledger_shape"]:
        errors.append("ledger must contain exactly one row for every evidence id")

    checks["ledger_hashes"] = True
    checks["ledger_statuses"] = True
    checks["ledger_observations"] = True
    for evidence_id, expected_row in expected.items():
        row = row_by_id.get(evidence_id)
        if not isinstance(row, dict) or row.get("sha256") != expected_row["sha256"]:
            checks["ledger_hashes"] = False
        if not isinstance(row, dict) or row.get("status") != expected_row["status"]:
            checks["ledger_statuses"] = False
        if not isinstance(row, dict) or not str(row.get("observation", "")).strip():
            checks["ledger_observations"] = False
    if not checks["ledger_hashes"]:
        errors.append("ledger contains an incorrect or missing evidence hash")
    if not checks["ledger_statuses"]:
        errors.append("ledger contains an incorrect or missing evidence status")
    if not checks["ledger_observations"]:
        errors.append("every ledger row needs a non-empty observation")

    if not report_path.is_file():
        report = ""
        checks["report_exists"] = False
        errors.append("LONG_REPORT.md is missing")
    else:
        report = report_path.read_text(encoding="utf-8", errors="replace")
        checks["report_exists"] = True

    checks["anchors"] = all(value in report for value in ANCHORS.values())
    if not checks["anchors"]:
        errors.append("report is missing one or more required anchors")

    unverified = [
        evidence_id
        for evidence_id, row in expected.items()
        if row["status"] == "UNVERIFIED"
    ]
    checks["distractors_rejected"] = all(
        _report_rejects_unverified(report, evidence_id) for evidence_id in unverified
    )
    if not checks["distractors_rejected"]:
        errors.append("report does not explicitly reject every unverified distractor")

    checks["combined_synthesis"] = (
        "combined" in report.lower()
        and "ORBIT-7" in report
        and "MIDDLE-42" in report
        and "LATE-91" in report
    )
    if not checks["combined_synthesis"]:
        errors.append("report is missing the required combined synthesis")

    result = ValidationResult(
        not errors and all(checks.values()), checks, tuple(errors)
    )
    return result
