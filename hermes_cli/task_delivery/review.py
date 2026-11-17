"""Pure transitions for explicitly managed reviews; callers persist them atomically.

Each report is a complete review of one work item's current candidate. Missing
findings resolve prior blockers but retain their evidence and attempt history.
The initial review and two repair/review cycles span at most three candidates;
renaming a finding cannot buy another cycle. Starting a new review epoch is an
explicit owner operation outside this module, never a side effect of a report.
"""
from __future__ import annotations

import copy
import json


class ReviewError(ValueError):
    """An invalid, conflicting or exhausted managed review transition."""


_CATEGORIES = frozenset({
    "regression", "release_blocker", "preexisting", "fixture_incompatibility",
    "contract_change", "suggestion",
})
_REPORT_FIELDS = frozenset({"request_id", "reviewer", "candidate_digest", "findings", "passed"})
_FINDING_FIELDS = frozenset({"issue_id", "category", "blocking", "evidence_ref", "message"})
_STATE_FIELDS = frozenset({"version", "status", "candidate_digest", "candidate_digests", "history", "issues"})


def _text(value: object, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ReviewError(f"{field} must be a nonempty string of at most {limit} characters")
    return value.strip()


def _fields(value: object, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, dict) or value.keys() != expected:
        raise ReviewError(f"{label} must contain exactly its required fields")


def _finding(value: object) -> dict:
    _fields(value, _FINDING_FIELDS, "finding")
    category = value["category"]
    if not isinstance(category, str) or category not in _CATEGORIES:
        raise ReviewError("finding category is not supported")
    if type(value["blocking"]) is not bool:
        raise ReviewError("finding blocking must be a boolean")
    return {
        "issue_id": _text(value["issue_id"], "issue_id", 128),
        "category": category,
        "blocking": value["blocking"],
        "evidence_ref": _text(value["evidence_ref"], "evidence_ref", 2048),
        "message": _text(value["message"], "message", 8192),
    }


def validate_review(report: dict) -> dict:
    """Return a detached canonical report, rejecting ambiguous verdicts."""
    _fields(report, _REPORT_FIELDS, "review report")
    if type(report["passed"]) is not bool:
        raise ReviewError("passed must be a boolean")
    raw_findings = report["findings"]
    if not isinstance(raw_findings, list) or len(raw_findings) > 128:
        raise ReviewError("findings must be an array of at most 128 findings")
    findings = [_finding(finding) for finding in raw_findings]
    identifiers = [finding["issue_id"] for finding in findings]
    if len(set(identifiers)) != len(identifiers):
        raise ReviewError("findings must not repeat an issue_id")
    if report["passed"] != (not any(finding["blocking"] for finding in findings)):
        raise ReviewError("passed must agree with the absence of blocking findings")
    return {
        "request_id": _text(report["request_id"], "request_id", 128),
        "reviewer": _text(report["reviewer"], "reviewer", 256),
        "candidate_digest": _text(report["candidate_digest"], "candidate_digest", 256),
        "findings": sorted(findings, key=lambda finding: finding["issue_id"]),
        "passed": report["passed"],
    }


def _empty_state() -> dict:
    return {"version": 1, "status": "unreviewed", "candidate_digest": None,
            "candidate_digests": [], "history": [], "issues": {}}


def _advance(state: dict, report: dict) -> None:
    if state["status"] == "needs_decision":
        raise ReviewError("review state needs_decision; start a new explicit review epoch")
    if len(state["history"]) >= 128:
        raise ReviewError("review history limit reached; start a new explicit review epoch")
    candidate = report["candidate_digest"]
    candidates = state["candidate_digests"]
    if candidate not in candidates:
        if len(candidates) >= 3:
            raise ReviewError("candidate review budget exhausted; start a new explicit review epoch")
        candidates.append(candidate)
    for issue in state["issues"].values():
        issue["resolved"] = True
    for finding in report["findings"]:
        issue = state["issues"].setdefault(finding["issue_id"], {"candidate_digests": []})
        issue.update(finding, reviewer=report["reviewer"], resolved=not finding["blocking"])
        if finding["blocking"] and candidate not in issue["candidate_digests"]:
            issue["candidate_digests"].append(candidate)
    state["history"].append(copy.deepcopy(report))
    state["candidate_digest"] = candidate
    state["status"] = ("passed" if report["passed"] else
                       "needs_decision" if len(candidates) >= 3 else "changes_requested")


def _validated_state(state: dict | None) -> dict:
    if state is None:
        return _empty_state()
    _fields(state, _STATE_FIELDS, "review state")
    history = state["history"]
    if type(state["version"]) is not int or state["version"] != 1:
        raise ReviewError("review state version is unsupported")
    if not isinstance(history, list) or not 1 <= len(history) <= 128:
        raise ReviewError("review state history must contain 1 to 128 reports")
    rebuilt, identifiers = _empty_state(), set()
    for raw in history:
        report = validate_review(raw)
        if report["request_id"] in identifiers:
            raise ReviewError("review state history repeats request_id")
        identifiers.add(report["request_id"])
        _advance(rebuilt, report)
    try:
        matches = json.dumps(state, sort_keys=True, allow_nan=False) == json.dumps(rebuilt, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ReviewError("review state is not JSON data") from exc
    if not matches:
        raise ReviewError("review state does not match its recorded history")
    return rebuilt


def apply_review(state: dict | None, report: dict, *, candidate_digest: str) -> dict:
    """Apply a current-candidate snapshot; exact retries return the current state.

    An old request replay never rolls back newer state. Request identity is
    checked before candidate freshness so a caller can safely retry a write
    whose response was lost, even after another candidate was installed.
    """
    normalized = validate_review(report)
    current = _validated_state(state)
    for previous in current["history"]:
        if previous["request_id"] == normalized["request_id"]:
            if previous != normalized:
                raise ReviewError("request_id already identifies a different review report")
            return current
    if normalized["candidate_digest"] != _text(candidate_digest, "candidate_digest", 256):
        raise ReviewError("review candidate does not match the current candidate")
    _advance(current, normalized)
    return current


def allow_repair(state: dict | None) -> bool:
    """Only a valid failed review with remaining budget permits automatic repair."""
    try:
        return _validated_state(state)["status"] == "changes_requested"
    except ReviewError:
        return False
