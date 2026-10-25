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


def _unverified_cfo_claims(links: Iterable[dict[str, str]]) -> list[dict[str, str]]:
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
        raise ValueError("delivery report requires at least one declared CFO usage link")
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


def _lifecycle_cycles(
    event_rows: list[Any], run_rows: list[Any], acceptance_row: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Reconstruct ordered builder/reviewer cycles from run-bound events."""
    runs = {int(row["id"]): row for row in run_rows}
    requests = [row for row in event_rows if row["kind"] == "review_requested"]
    if not requests:
        raise ValueError("protected task has no review handoff")
    cycles: list[dict[str, Any]] = []
    repairs: list[dict[str, int]] = []
    for index, request in enumerate(requests):
        request_payload = _payload(request)
        builder = request_payload.get("implementer")
        reviewer = request_payload.get("reviewer")
        builder_run_id = request["run_id"]
        if (
            not isinstance(builder, str) or not builder.strip()
            or not isinstance(reviewer, str) or not reviewer.strip()
            or builder.strip() == reviewer.strip()
            or builder_run_id is None
        ):
            raise ValueError("review handoff has missing or non-independent identities")
        builder, reviewer = builder.strip(), reviewer.strip()
        builder_run = runs.get(int(builder_run_id))
        if (
            builder_run is None or builder_run["profile"] != builder
            or builder_run["outcome"] != "review_requested"
        ):
            raise ValueError("review handoff is not bound to the builder run")
        builder_claims = [
            row for row in event_rows
            if row["kind"] == "claimed" and row["run_id"] == builder_run_id
            and int(row["id"]) < int(request["id"])
        ]
        if len(builder_claims) != 1:
            raise ValueError("builder run has missing or ambiguous claim evidence")

        boundary_id = int(requests[index + 1]["id"]) if index + 1 < len(requests) else int(acceptance_row["id"]) + 1
        reviewer_claims = []
        for row in event_rows:
            if row["kind"] != "claimed" or not (int(request["id"]) < int(row["id"]) < boundary_id):
                continue
            run_id = row["run_id"]
            run = runs.get(int(run_id)) if run_id is not None else None
            if run is not None and run["profile"] == reviewer and _payload(row).get("source_status") == "review":
                reviewer_claims.append(row)
        if len(reviewer_claims) != 1:
            raise ValueError("review handoff has missing or ambiguous reviewer claim evidence")
        reviewer_claim = reviewer_claims[0]
        reviewer_run_id = int(reviewer_claim["run_id"])

        dispositions = [
            row for row in event_rows
            if row["kind"] in {"changes_requested", "artifact_review_acceptance"}
            and row["run_id"] == reviewer_run_id
            and int(reviewer_claim["id"]) < int(row["id"]) < boundary_id
        ]
        if len(dispositions) != 1:
            raise ValueError("reviewer run has missing or ambiguous disposition evidence")
        disposition = dispositions[0]
        cycle: dict[str, Any] = {
            "builder": builder,
            "builder_run_id": int(builder_run_id),
            "builder_claim_event_id": int(builder_claims[0]["id"]),
            "review_requested_event_id": int(request["id"]),
            "reviewer": reviewer,
            "reviewer_run_id": reviewer_run_id,
            "reviewer_claim_event_id": int(reviewer_claim["id"]),
            "disposition": disposition["kind"],
            "disposition_event_id": int(disposition["id"]),
        }
        if disposition["kind"] == "changes_requested":
            change = _payload(disposition)
            if change.get("implementer") != builder or change.get("reviewer") != reviewer:
                raise ValueError("repair handoff identities differ from the review handoff")
            if index + 1 >= len(requests):
                raise ValueError("changes-requested review has no subsequent repair run")
            next_request = requests[index + 1]
            repair_run_id = next_request["run_id"]
            if repair_run_id is None:
                raise ValueError("repair handoff has no builder run")
            if _payload(next_request).get("implementer") != builder:
                raise ValueError("repair run is assigned to a different builder")
            next_claims = [
                row for row in event_rows
                if row["kind"] == "claimed" and row["run_id"] == repair_run_id
            ]
            if len(next_claims) != 1 or int(next_claims[0]["id"]) <= int(disposition["id"]):
                raise ValueError("repair builder claim is missing or precedes the rejection")
            cycle["repair_builder_run_id"] = int(repair_run_id)
            repairs.append({
                "rejected_reviewer_run_id": reviewer_run_id,
                "changes_requested_event_id": int(disposition["id"]),
                "repair_builder_run_id": int(repair_run_id),
            })
        elif (
            index != len(requests) - 1
            or int(disposition["id"]) != int(acceptance_row["id"])
        ):
            raise ValueError("acceptance is not the final review disposition")
        cycles.append(cycle)
    return cycles, repairs


def build_artifact_review_evidence(
    conn: Any,
    task_id: str,
    *,
    ticket_ref: str,
    cfo_usage_links: Iterable[dict[str, str]],
) -> dict[str, Any]:
    """Return a content-addressed evidence projection for one accepted task.

    ``ticket_ref`` and CFO receipt references are unverified caller claims.
    They are isolated under ``external_claims`` and explicitly barred from use
    as acceptance-gate evidence. Roots and the CFO ledger must be checked at
    their own authenticated boundaries.
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
    bindings = acceptance.get("bindings") or {}
    if bindings.get("task_id") != task_id or bindings.get("run_id") != acceptances[0]["run_id"]:
        raise ValueError("acceptance event and receipt run bindings disagree")
    cycles, repair_cycles = _lifecycle_cycles(rows, runs, acceptances[0])
    final_cycle = cycles[-1]
    if (
        bindings.get("owner") != final_cycle["builder"]
        or bindings.get("reviewer") != final_cycle["reviewer"]
        or bindings.get("run_id") != final_cycle["reviewer_run_id"]
    ):
        raise ValueError("acceptance identities differ from the final lifecycle cycle")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "task": {
            "task_id": task_id,
            "status": task["status"],
            "completion_contract": task["completion_contract"],
            "final_assignee": task["assignee"],
            "accepted_run_id": acceptances[0]["run_id"],
        },
        "delivery_cycles": cycles,
        "repair_cycles": repair_cycles,
        "acceptance": {
            "receipt_sha256": acceptance["receipt_sha256"],
            "artifact_sha256": sorted(item["sha256"] for item in acceptance["artifacts"]),
            "reviewer": bindings.get("reviewer"),
            "review_profile": bindings.get("review_profile"),
            "source_revision": bindings.get("source_revision"),
        },
        "external_claims": {
            "verified": False,
            "ticket_ref": ticket_ref.strip(),
            "cfo_usage_links": _unverified_cfo_claims(cfo_usage_links),
            "note": (
                "Caller-supplied labels only. No Roots ticket receipt or CFO receipt bytes were "
                "resolved or authenticated by this exporter."
            ),
        },
        "use_as_acceptance_gate": False,
    }
    report["report_sha256"] = _digest(report)
    return report
