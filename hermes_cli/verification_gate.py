"""Deterministic completion verification used at the Kanban done boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CARD_TYPE_ARTIFACTS = {
    "pr-merged": "metadata.verification.artifact with type=pr-merged, url, merged=true, and merge_commit",
    "deployed": "metadata.verification.artifact with type=deployed, url, and a successful smoke_test",
    "document-committed": "metadata.verification.artifact with type=document-committed, path, and commit",
    "decision-recorded": "metadata.verification.artifact with type=decision-recorded, path, and decision",
}


def _failure(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _report(failures: list[dict[str, str]], checks: dict[str, bool]) -> dict[str, Any]:
    failures.sort(key=lambda item: item["code"])
    return {"ok": not failures, "failures": failures, "checks": dict(sorted(checks.items()))}


def _requires_pull_request(contract: str | None) -> bool:
    """``pr-merged``, ``OWNER/REPO`` and PR-URL contracts publish a PR; the rest do not."""
    return contract == "pr-merged" or contract not in (None, "local-only", *CARD_TYPE_ARTIFACTS)


def verify_completion(payload: Any, *, contract: str | None = None, full: bool = True) -> dict[str, Any]:
    """Return a stable, side-effect-free verification report.

    ``full=False`` checks only the artifact of a typed ``contract``; ``full=True`` adds the
    evidence/preflight/scope predicates, plus repository predicates for PR contracts.
    """
    failures: list[dict[str, str]] = []
    checks: dict[str, bool] = {}
    if not isinstance(payload, Mapping):
        if full:
            return {"ok": False, "failures": [_failure("invalid_input", "payload must be a JSON object")], "checks": {}}
        payload = {}

    task, repo = _mapping(payload.get("task")), _mapping(payload.get("repo"))
    preflight, scope = _mapping(payload.get("preflight")), _mapping(payload.get("scope"))

    if contract in CARD_TYPE_ARTIFACTS:
        artifact = _mapping(payload.get("artifact"))
        artifact_type = artifact.get("type") == contract
        checks["required_artifact"] = artifact_type
        if not artifact_type:
            failures.append(_failure("missing_artifact", f"required artifact: {CARD_TYPE_ARTIFACTS[contract]}"))
        else:
            required = {
                "pr-merged": ("url", "merged", "merge_commit"),
                "deployed": ("url", "smoke_test"),
                "document-committed": ("path", "commit"),
                "decision-recorded": ("path", "decision"),
            }[contract]
            missing = [key for key in required if not artifact.get(key)]
            smoke = _mapping(artifact.get("smoke_test"))
            if contract == "deployed" and (smoke.get("exit_code") != 0 or not smoke.get("command")):
                missing.append("smoke_test.exit_code=0")
            if missing:
                failures.append(_failure("incomplete_artifact", "artifact is missing: " + ", ".join(missing)))

    if not full:
        return _report(failures, checks)

    predicates = [
        ("clean_worktree", repo.get("dirty") is not True, "unsaved_work", "repository worktree is dirty"),
    ]
    if _requires_pull_request(contract):
        predicates += [
            ("commit", isinstance(repo.get("commit"), str) and bool(repo["commit"].strip()), "missing_commit", "a commit identifier is required"),
            ("pushed", repo.get("pushed") is True, "missing_push", "the commit must be pushed"),
            ("pull_request", isinstance(_mapping(repo.get("pr")).get("number"), int) and _mapping(repo.get("pr")).get("number") > 0 and _mapping(repo.get("pr")).get("state") in {"open", "merged"}, "missing_pr", "an open or merged pull request is required"),
            ("ci_green", _mapping(repo.get("ci")).get("status") in {"success", "passed", "green"}, "ci_not_green", "CI status must be success"),
            ("review", _mapping(repo.get("review")).get("approved") is True, "missing_review", "an approved review is required"),
        ]
    for check, passed, code, detail in predicates:
        checks[check] = passed
        if not passed:
            failures.append(_failure(code, detail))

    evidence = task.get("evidence")
    evidence_ok = isinstance(evidence, list) and bool(evidence) and all(
        isinstance(item, Mapping) and isinstance(item.get("command"), str) and bool(item["command"].strip())
        and isinstance(item.get("exit_code"), int) and item["exit_code"] == 0 for item in evidence
    )
    checks["structured_evidence"] = evidence_ok
    if not evidence_ok:
        failures.append(_failure("missing_evidence", "evidence must be a non-empty list of successful command records"))

    requested, available = preflight.get("requested_skills", []), preflight.get("available_skills", [])
    skills_ok = isinstance(requested, list) and isinstance(available, list) and not sorted({x for x in requested if x not in available})
    checks["skills_supported"] = skills_ok
    if not skills_ok:
        failures.append(_failure("unsupported_skill", "requested skills must be available"))

    review_checks, supported = preflight.get("review_checks", []), preflight.get("supported_flags", [])
    flags_ok = isinstance(review_checks, list) and isinstance(supported, list) and not sorted({x for x in review_checks if x not in supported})
    checks["review_checks_supported"] = flags_ok
    if not flags_ok:
        failures.append(_failure("invalid_check", "review checks must be supported"))

    copies, maximum = preflight.get("dependency_copies"), preflight.get("max_dependency_copies")
    budget_ok = isinstance(copies, int) and not isinstance(copies, bool) and isinstance(maximum, int) and not isinstance(maximum, bool) and copies <= maximum
    checks["dependency_copy_budget"] = budget_ok
    if not budget_ok:
        failures.append(_failure("workspace_budget", "dependency copy count exceeds configured budget"))

    secret_findings = scope.get("secret_findings")
    secrets_ok = isinstance(secret_findings, list) and not secret_findings
    checks["secrets_clear"] = secrets_ok
    if not secrets_ok:
        failures.append(_failure("secret_finding", "secret findings must be empty"))

    changed, roots = scope.get("changed_paths"), scope.get("allowed_roots")
    scope_ok = isinstance(changed, list) and isinstance(roots, list) and all(
        isinstance(path, str) and any(path.startswith(root) for root in roots if isinstance(root, str)) for path in changed
    )
    checks["scope"] = scope_ok
    if not scope_ok:
        failures.append(_failure("scope_violation", "changed paths must be under an allowed root"))

    return _report(failures, checks)
