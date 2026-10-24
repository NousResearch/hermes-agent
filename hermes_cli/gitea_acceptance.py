"""Fail-closed Gitea Actions acceptance, independent of writer-posted statuses.

Pure assessment of one pull request against a pinned policy. ``get(path)`` is an
authenticated, read-only GET against the forge's ``/api/v1`` returning parsed JSON;
the caller owns origin/auth binding and attests executor custody (``custody_ok``).
Commit statuses are never read: a writer that can post a status cannot forge a
completed Actions job on the pinned runner for the pinned workflow bytes.

A ``qualification_only`` policy may assess an open PR and its receipt says so; it
can never stand in for production. A production policy additionally requires the
PR to be merged by an allowlisted identity into a recorded merge commit, and an
independent, official approval pinned to the exact head.

The trusted gate command that the kanban transport runs prints
:func:`gate_receipt` as JSON (see ``hermes_cli/kanban_pr_acceptance_gitea.py``).
"""
from __future__ import annotations

import base64
import binascii
import hashlib
from datetime import datetime
import re
from urllib.parse import quote

HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
REPO = re.compile(r"(?!\.\.?/)[A-Za-z0-9_.-]+/(?!\.\.?$)[A-Za-z0-9_.-]+")
# Lowercase host, optional port, no userinfo/query/fragment: one spelling per origin.
ORIGIN = re.compile(r"https://(?!github\.com(?:[:/]|$))[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?(?::[1-9][0-9]{0,4})?")
PR_URL = re.compile(r"(%s)/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pulls/([1-9][0-9]{0,9})" % ORIGIN.pattern)

PAGE_SIZE = 50
MAX_RUN_PAGES = 40
MAX_REVIEW_PAGES = 20

_POLICY_KEYS = {"policy_id", "qualification_only", "repo", "repository_id", "base",
                "runner_ids", "mergers", "workflows"}
_OPTIONAL_POLICY_KEYS = {"reviewers"}


class EvidenceError(ValueError):
    """The forge's answer is incomplete, ambiguous or changed while reading."""


def parse_pr_url(value) -> tuple[str, str, int] | None:
    """``(origin, OWNER/REPO, number)`` for an exact Gitea PR URL, else None."""
    match = PR_URL.fullmatch(value) if isinstance(value, str) else None
    if not match or not REPO.fullmatch(match[2]):
        return None
    return match[1], match[2], int(match[3])


def _require(condition, message="invalid policy"):
    if not condition:
        raise ValueError(message)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def validate_policy(policy) -> dict:
    """Strict schema: unknown keys (a stray ``allow_qualification``) are refused."""
    _require(isinstance(policy, dict) and _POLICY_KEYS <= policy.keys()
             <= _POLICY_KEYS | _OPTIONAL_POLICY_KEYS)
    _require(_nonempty_str(policy["policy_id"]) and isinstance(policy["qualification_only"], bool))
    _require(isinstance(policy["repo"], str) and REPO.fullmatch(policy["repo"]))
    _require(_is_int(policy["repository_id"]) and _nonempty_str(policy["base"]))
    runners = policy["runner_ids"]
    _require(isinstance(runners, list) and runners and all(_is_int(r) for r in runners))
    mergers = policy["mergers"]
    _require(isinstance(mergers, list) and mergers and all(
        isinstance(m, dict) and m.keys() == {"login", "id"} and _nonempty_str(m["login"]) and _is_int(m["id"])
        for m in mergers))
    _require(len({m["login"] for m in mergers}) == len(mergers))
    reviewers = policy.get("reviewers")
    _require(reviewers is None or (isinstance(reviewers, list) and reviewers
                                   and all(_nonempty_str(r) for r in reviewers)))
    workflows = policy["workflows"]
    _require(isinstance(workflows, list) and workflows)
    for workflow in workflows:
        _require(isinstance(workflow, dict) and workflow.keys() == {"path", "sha256", "jobs"})
        _require(_nonempty_str(workflow["path"]) and isinstance(workflow["sha256"], str)
                 and _HEX64.fullmatch(workflow["sha256"]))
        jobs = workflow["jobs"]
        _require(isinstance(jobs, list) and jobs and all(
            isinstance(j, dict) and j.keys() == {"name", "context"}
            and _nonempty_str(j["name"]) and _nonempty_str(j["context"]) for j in jobs))
        _require(len({j["name"] for j in jobs}) == len(jobs))
    _require(len({w["path"].split("/")[-1] for w in workflows}) == len(workflows))
    contexts = [j["context"] for w in workflows for j in w["jobs"]]
    _require(len(set(contexts)) == len(contexts))
    return policy


def _protection_ok(protection: dict, policy: dict) -> bool:
    contexts = {j["context"] for w in policy["workflows"] for j in w["jobs"]}
    base_ok = (protection["enable_status_check"] is True
               and set(protection["status_check_contexts"]) == contexts
               and len(protection["status_check_contexts"]) == len(contexts)
               and protection["block_on_outdated_branch"] is True
               and protection["enable_merge_whitelist"] is True
               and set(protection["merge_whitelist_usernames"]) == {m["login"] for m in policy["mergers"]}
               and not protection["merge_whitelist_teams"]
               and protection["enable_push"] is False
               and protection["enable_force_push"] is False
               and protection["enable_bypass_allowlist"] is False
               and protection["block_admin_merge_override"] is True)
    if policy["qualification_only"]:
        return base_ok
    approvals = protection["required_approvals"]
    return (base_ok and isinstance(approvals, int) and approvals >= 1
            and protection["block_on_rejected_reviews"] is True)


def _list_runs(get, rp: str) -> list:
    runs, total = [], None
    for page in range(1, MAX_RUN_PAGES + 1):
        data = get(f"{rp}/actions/runs?limit={PAGE_SIZE}&page={page}")
        batch, count = data["workflow_runs"], data["total_count"]
        if not isinstance(batch, list) or not isinstance(count, int) or len(batch) > PAGE_SIZE:
            raise EvidenceError("malformed run page")
        if total is None:
            total = count
        elif count != total:
            raise EvidenceError("run enumeration changed while paging")
        if not batch and len(runs) < total:
            raise EvidenceError("run pages end before total_count")
        runs.extend(batch)
        if len(runs) >= total:
            break
    else:
        raise EvidenceError("run page limit exceeded")
    if len(runs) != total or len({r["id"] for r in runs}) != len(runs):
        raise EvidenceError("run enumeration duplicated or overflowed")
    return runs


def _list_reviews(get, rp: str, number: int) -> list:
    reviews = []
    for page in range(1, MAX_REVIEW_PAGES + 1):
        batch = get(f"{rp}/pulls/{number}/reviews?limit={PAGE_SIZE}&page={page}")
        if not isinstance(batch, list) or len(batch) > PAGE_SIZE:
            raise EvidenceError("malformed review page")
        reviews.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
    else:
        raise EvidenceError("review page limit exceeded")
    if len({r["id"] for r in reviews}) != len(reviews):
        raise EvidenceError("duplicate reviews")
    return reviews


def _workflow_bytes_ok(file: dict, expected_sha256: str) -> bool:
    if file.get("type") != "file" or file.get("encoding") != "base64":
        return False
    try:
        content = base64.b64decode(file["content"], validate=False)
    except (binascii.Error, TypeError):
        return False
    return hashlib.sha256(content).hexdigest() == expected_sha256


def _job_classification(job: dict, run: dict, sha: str, policy: dict) -> str:
    bound = (job.get("head_sha") == sha and job.get("run_id") == run["id"]
             and job.get("run_attempt") == run["run_attempt"]
             and job.get("runner_id") in policy["runner_ids"])
    if not bound:
        return "untrusted"
    if run["status"] != "completed" or job["status"] != "completed":
        return "pending"
    if run.get("conclusion") == "success" and job.get("conclusion") == "success":
        return "success"
    return "failure"


def _workflow_checks(get, rp, policy, number, sha, runs, receipt) -> list[str] | str:
    """Outcomes for every required job, or a terminal classification string."""
    outcomes = []
    for workflow in policy["workflows"]:
        file = get(f"{rp}/contents/{workflow['path']}?ref={sha}")
        if not _workflow_bytes_ok(file, workflow["sha256"]):
            return "workflow"
        run_path = workflow["path"].split("/")[-1] + f"@refs/pull/{number}/head"
        candidates = [r for r in runs if r["head_sha"] == sha
                      and r["repository"]["id"] == policy["repository_id"]
                      and r["event"] == "pull_request" and r["path"] == run_path
                      and any(p["number"] == number and p["head"]["sha"] == sha
                              and p["base"]["ref"] == policy["base"] for p in r["pull_requests"])]
        if not candidates:
            return "missing"
        run = max(candidates, key=lambda r: r["id"])
        jobs = get(f"{rp}/actions/runs/{run['id']}/jobs?limit={PAGE_SIZE}")
        listed = jobs["jobs"]
        if (not isinstance(listed, list) or len(listed) != jobs["total_count"]
                or len({j["id"] for j in listed}) != len(listed)):
            raise EvidenceError("incomplete jobs")
        for required in workflow["jobs"]:
            matches = [j for j in listed if j["name"] == required["name"]]
            if len(matches) > 1:
                raise EvidenceError("ambiguous job")
            if not matches:
                outcomes.append("missing")
                continue
            job = matches[0]
            classification = _job_classification(job, run, sha, policy)
            receipt["checks"].append({"name": required["context"], "id": job["id"], "run_id": run["id"],
                                      "run_attempt": run["run_attempt"], "runner_id": job.get("runner_id"),
                                      "head_sha": sha, "classification": classification,
                                      "completed_at": job.get("completed_at"),
                                      "run_completed_at": run.get("completed_at")})
            outcomes.append(classification)
    return outcomes


def _merge_provenance(pr: dict, policy: dict) -> dict | None:
    merged_by = pr.get("merged_by") or {}
    allowed = {(m["login"], m["id"]) for m in policy["mergers"]}
    commit = pr.get("merge_commit_sha")
    if (pr.get("merged") is not True or pr.get("state") != "closed"
            or (merged_by.get("login"), merged_by.get("id")) not in allowed
            or not isinstance(commit, str) or not HEX40.fullmatch(commit)
            or not _nonempty_str(pr.get("merged_at"))):
        return None
    return {"merged": True, "merge_commit_sha": commit, "merged_by": merged_by["login"],
            "merged_by_id": merged_by["id"], "merged_at": pr["merged_at"]}


def _timestamp(value):
    if not isinstance(value, str):
        raise EvidenceError("missing timestamp")
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise EvidenceError("timestamp needs timezone")
    return value


def _independent_approval(reviews: list, pr: dict, policy: dict, sha: str) -> dict | None:
    author = pr["user"]
    allowed = set(policy.get("reviewers") or [])
    latest: dict = {}
    for review in sorted(reviews, key=lambda r: r["id"]):
        # Comments and pending drafts do not revoke an earlier formal decision.
        if review["state"] in {"APPROVED", "REQUEST_CHANGES"}:
            latest[review["user"]["id"]] = review
    if any(r["state"] == "REQUEST_CHANGES" and not r["dismissed"] for r in latest.values()):
        return None
    for review in sorted(latest.values(), key=lambda r: r["id"], reverse=True):
        user = review["user"]
        if (review["state"] == "APPROVED" and review["commit_id"] == sha
                and review["stale"] is False and review["dismissed"] is False
                and review["official"] is True
                and user["id"] != author["id"] and user["login"] != author["login"]
                and (not allowed or user["login"] in allowed)
                and (not pr.get("merged") or _timestamp(review.get("submitted_at")) <= _timestamp(pr.get("merged_at")))):
            return {"id": review["id"], "reviewer": user["login"], "reviewer_id": user["id"],
                    "commit_id": sha, "submitted_at": review.get("submitted_at")}
    return None


def _pr_identity(pr: dict) -> tuple:
    merged_by = pr.get("merged_by") or {}
    return (pr["head"]["sha"], pr["base"]["ref"], pr["base"]["sha"], pr["state"], pr.get("merged"),
            pr.get("merge_commit_sha"), merged_by.get("id"), merged_by.get("login"))


def _state_classification(pr: dict, production: bool) -> str | None:
    if pr.get("merged") is True:
        return None
    if pr["state"] != "open":
        return "closed"
    return "unmerged" if production else None


def assess(get, policy: dict, number: int, *, expected_head: str | None = None,
           custody_ok: bool = False, require_merged: bool = True, require_review: bool = False) -> dict:
    """Receipt for PR ``number`` of ``policy['repo']``; ``ok`` only on complete, stable evidence."""
    policy = validate_policy(policy)
    production = not policy["qualification_only"] or require_review
    rp = "/repos/" + policy["repo"]
    receipt = {"ok": False, "classification": "missing", "checks": [], "head_sha": None,
               "repo": policy["repo"], "policy_id": policy["policy_id"],
               "qualification_only": policy["qualification_only"]}
    try:
        pr = get(f"{rp}/pulls/{number}")
        sha = pr["head"]["sha"]
        if pr["number"] != number or not isinstance(sha, str) or not HEX40.fullmatch(sha):
            raise EvidenceError("invalid PR identity")
        receipt["head_sha"] = sha
        if expected_head is not None and expected_head != sha:
            return dict(receipt, classification="stale")
        state = _state_classification(pr, production and require_merged)
        if state:
            return dict(receipt, classification=state)
        base_repo = pr["base"]["repo"]
        if (base_repo["id"] != policy["repository_id"] or base_repo["full_name"] != policy["repo"]
                or pr["base"]["ref"] != policy["base"]):
            return dict(receipt, classification="binding")
        protection_path = f"{rp}/branch_protections/{quote(policy['base'], safe='')}"
        protection = get(protection_path)
        if not _protection_ok(protection, {**policy, "qualification_only": not production}):
            return dict(receipt, classification="protection")
        runs = _list_runs(get, rp)
        outcomes = _workflow_checks(get, rp, policy, number, sha, runs, receipt)
        if isinstance(outcomes, str):
            return dict(receipt, classification=outcomes)
        reviews = None
        if production:
            merge = _merge_provenance(pr, policy)
            if merge is None and require_merged:
                return dict(receipt, classification="merge")
            if pr.get("merged") and any(
                    _timestamp(c.get("completed_at")) > _timestamp(pr.get("merged_at"))
                    or _timestamp(c.get("run_completed_at")) > _timestamp(pr.get("merged_at"))
                    for c in receipt["checks"]):
                return dict(receipt, classification="merge")
            reviews = _list_reviews(get, rp, number)
            review = _independent_approval(reviews, pr, policy, sha)
            if review is None:
                return dict(receipt, classification="review")
            receipt.update(review=review)
            if merge is not None:
                receipt["merge"] = merge
        # Readback after every page: evidence for an old head/base/merge is never transferable.
        if _pr_identity(get(f"{rp}/pulls/{number}")) != _pr_identity(pr):
            return dict(receipt, classification="stale")
        if get(protection_path) != protection:
            return dict(receipt, classification="protection")
        if production and _list_reviews(get, rp, number) != reviews:
            return dict(receipt, classification="stale")
        if _list_runs(get, rp) != runs:
            return dict(receipt, classification="stale")
        if not custody_ok:
            return dict(receipt, classification="custody")
        outcome = next((o for o in outcomes if o != "success"), "success" if outcomes else "missing")
        return dict(receipt, ok=outcome == "success", classification=outcome)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError):
        receipt.pop("merge", None)
        receipt.pop("review", None)
        return dict(receipt, classification="infra")


def gate_receipt(get, policy: dict, pr_url: str, origin: str, *, custody_ok: bool = False) -> dict:
    """Full receipt for a trusted gate command: binds pr_url/provider/origin to the assessment."""
    parsed = parse_pr_url(pr_url)
    base = {"ok": False, "classification": "binding", "checks": [], "head_sha": None,
            "pr_url": pr_url, "provider": "gitea", "origin": origin}
    if parsed is None or parsed[0] != origin or not isinstance(policy, dict) or parsed[1] != policy.get("repo"):
        return base
    try:
        result = assess(get, policy, parsed[2], custody_ok=custody_ok)
    except ValueError:
        return dict(base, classification="infra")
    return {**result, "pr_url": pr_url, "provider": "gitea", "origin": origin}
