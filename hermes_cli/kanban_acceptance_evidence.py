"""Deterministic evidence export for protected Kanban delivery runs.

The acceptance transaction remains the authority.  This module only projects
its durable database facts into a review/install receipt; it never completes,
archives, assigns, or otherwise mutates a task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCHEMA = "hermes.artifact-review-evidence/v1"
_TICKET = re.compile(r"[A-Z][A-Z0-9]+-[0-9]{4,}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _payload(row: Any) -> dict[str, Any]:
    value = row["payload"]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _validated_cfo_links(links: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("CFO usage links must be objects")
        ref = link.get("ref")
        receipt_sha256 = link.get("receipt_sha256")
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("each CFO usage link requires a non-empty ref")
        if not isinstance(receipt_sha256, str) or not _SHA256.fullmatch(receipt_sha256):
            raise ValueError("each CFO usage link requires a lowercase SHA-256 receipt digest")
        normalized.append({"ref": ref.strip(), "receipt_sha256": receipt_sha256})
    if not normalized:
        raise ValueError("protected delivery evidence requires at least one CFO usage link")
    return sorted(normalized, key=lambda item: (item["ref"], item["receipt_sha256"]))


def _revalidate_acceptance_receipt(receipt: dict[str, Any]) -> None:
    claimed = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not isinstance(claimed, str) or claimed != _digest(unsigned):
        raise ValueError("artifact acceptance receipt digest is missing or stale")
    for artifact in receipt.get("artifacts") or ():
        if not isinstance(artifact, dict):
            raise ValueError("artifact acceptance receipt contains malformed artifact evidence")
        path, expected = artifact.get("cas_path"), artifact.get("sha256")
        if not isinstance(path, str) or not isinstance(expected, str):
            raise ValueError("artifact acceptance receipt omits CAS path or digest")
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("accepted CAS artifact is missing or not a regular file")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected or candidate.stat().st_size != artifact.get("size"):
            raise ValueError("accepted CAS artifact no longer matches its receipt")


def build_artifact_review_evidence(
    conn: Any,
    task_id: str,
    *,
    ticket_ref: str,
    cfo_usage_links: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Return a content-addressed evidence projection for one accepted task.

    ``ticket_ref`` and CFO receipt references are supplied by the integration
    owner because Roots and the CFO ledger are separate authorities.  Their
    digests are retained verbatim; this exporter does not claim it queried or
    accepted either external system.
    """
    if not isinstance(ticket_ref, str) or not _TICKET.fullmatch(ticket_ref.strip()):
        raise ValueError("ticket_ref must be a canonical ticket reference")
    task = conn.execute(
        "SELECT id, status, completion_contract, assignee, current_run_id FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    if task is None:
        raise ValueError("task not found")
    if task["completion_contract"] != "artifact-review-v1" or task["status"] != "done":
        raise ValueError("task is not a completed artifact-review-v1 delivery")
    rows = conn.execute(
        "SELECT id, run_id, kind, payload FROM task_events WHERE task_id=? ORDER BY id",
        (task_id,),
    ).fetchall()
    acceptances = [row for row in rows if row["kind"] == "artifact_review_acceptance"]
    if len(acceptances) != 1:
        raise ValueError("completed protected task must have exactly one acceptance receipt")
    acceptance = _payload(acceptances[0])
    if acceptance.get("ok") is not True:
        raise ValueError("protected task has no accepted artifact receipt")
    _revalidate_acceptance_receipt(acceptance)

    runs = conn.execute(
        "SELECT id, profile, status, outcome FROM task_runs WHERE task_id=? ORDER BY id",
        (task_id,),
    ).fetchall()
    run_rows = [
        {"run_id": int(row["id"]), "profile": row["profile"],
         "status": row["status"], "outcome": row["outcome"]}
        for row in runs
    ]
    phases = []
    for row in rows:
        if row["kind"] not in {
            "assigned", "claimed", "review_requested", "changes_requested",
            "artifact_review_acceptance", "completed",
        }:
            continue
        payload = _payload(row)
        phases.append({
            "event_id": int(row["id"]),
            "run_id": row["run_id"],
            "kind": row["kind"],
            "actor": payload.get("assignee") or payload.get("implementer") or payload.get("reviewer"),
        })
    kinds = {item["kind"] for item in phases}
    if not {"claimed", "review_requested", "artifact_review_acceptance"}.issubset(kinds):
        raise ValueError("protected task is missing builder/reviewer lifecycle evidence")
    bindings = acceptance.get("bindings") or {}
    if bindings.get("task_id") != task_id or bindings.get("run_id") != acceptances[0]["run_id"]:
        raise ValueError("acceptance event and receipt run bindings disagree")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ticket_ref": ticket_ref.strip(),
        "task": {
            "task_id": task_id,
            "status": task["status"],
            "completion_contract": task["completion_contract"],
            "final_assignee": task["assignee"],
            "accepted_run_id": acceptances[0]["run_id"],
        },
        "runs": run_rows,
        "phases": phases,
        "repair_run_ids": [
            int(row["run_id"]) for row in rows
            if row["kind"] == "changes_requested" and row["run_id"] is not None
        ],
        "acceptance": {
            "receipt_sha256": acceptance["receipt_sha256"],
            "artifact_sha256": sorted(item["sha256"] for item in acceptance["artifacts"]),
            "reviewer": bindings.get("reviewer"),
            "review_profile": bindings.get("review_profile"),
            "source_revision": bindings.get("source_revision"),
        },
        "cfo_usage_links": _validated_cfo_links(cfo_usage_links),
        "external_authority_note": (
            "Ticket and CFO references are digest-bound provenance links; their authorities "
            "must be checked by the integration owner."
        ),
    }
    report["report_sha256"] = _digest(report)
    return report

