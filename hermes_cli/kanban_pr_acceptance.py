"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.
"""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=30, check=True)
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def _validated_local_gate(local_gate: dict | None, sha: str, coordinator: str | None) -> dict | None:
    if not isinstance(local_gate, dict) or not isinstance(coordinator, str) or not coordinator.strip():
        return None
    if local_gate.get("head_sha") != sha or local_gate.get("result") != "passed":
        return None
    steps = local_gate.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    names = []
    for step in steps:
        if not isinstance(step, dict):
            return None
        name = step.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_-]{1,80}", name) or step.get("result") != "passed":
            return None
        names.append(name)
    if len(names) != len(set(names)):
        return None
    review = local_gate.get("independent_review")
    if not isinstance(review, dict) or review.get("status") != "approved":
        return None
    reviewer, review_task = review.get("reviewer"), review.get("task_id")
    if (review.get("head_sha") != sha or not isinstance(reviewer, str) or not reviewer.strip() or
            reviewer.casefold() == coordinator.casefold() or
            not isinstance(review_task, str) or not review_task.strip()):
        return None
    if local_gate.get("approved_by") != coordinator or local_gate.get("private_database") is not True:
        return None
    ci = local_gate.get("ci")
    if not isinstance(ci, dict) or ci.get("classification") not in {"not_required", "pre_runner_infrastructure"}:
        return None
    check_run_ids = ci.get("check_run_ids")
    if not isinstance(check_run_ids, list) or any(type(run_id) is not int or run_id < 1 for run_id in check_run_ids):
        return None
    if len(check_run_ids) != len(set(check_run_ids)):
        return None
    if ci["classification"] == "not_required" and check_run_ids:
        return None
    if ci["classification"] == "pre_runner_infrastructure" and not check_run_ids:
        return None
    return {
        "ci_classification": ci["classification"],
        "check_run_ids": set(check_run_ids),
        "reviewer": reviewer,
        "review_task_id": review_task,
        "reviewed_head_sha": sha,
    }


def _verified_pre_runner_failure(repo: str, check: dict, sha: str,
                                 gate: dict | None, cache: dict[int, bool]) -> bool:
    check_id = check.get("id")
    if (gate is None or gate["ci_classification"] != "pre_runner_infrastructure" or
            type(check_id) is not int or check_id not in gate["check_run_ids"]):
        return False
    if check_id in cache:
        return cache[check_id]
    if check.get("head_sha") != sha or check.get("status") != "completed" or check.get("conclusion") != "failure":
        cache[check_id] = False
        return False
    details_url = check.get("details_url") or check.get("html_url") or ""
    run_url = re.match(
        rf"^https://github\.com/{re.escape(repo)}/actions/runs/([1-9][0-9]*)(?:[/?#]|$)",
        str(details_url), re.IGNORECASE,
    )
    if not run_url:
        cache[check_id] = False
        return False
    pages = _api(f"repos/{repo}/actions/runs/{run_url.group(1)}/jobs?per_page=100", paginate=True)
    jobs = [job for page in pages for job in page.get("jobs", [])]
    total = pages[0].get("total_count") if pages else None
    if not pages or type(total) is not int or len({job.get("id") for job in jobs}) != total or not jobs:
        raise ValueError("Incomplete GitHub Actions job evidence")
    verified = all(job.get("conclusion") == "failure" and job.get("runner_id") in (None, 0)
                   and not job.get("steps") for job in jobs)
    cache[check_id] = verified
    return verified


def collect_acceptance(contract: str, published_pr: str | None, *,
                       local_gate: dict | None = None, coordinator: str | None = None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        pr = _api("graphql", query=query)["data"]["repository"]["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        rules = _api(f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        total_runs = pages[0]["total_count"] if pages else 0
        if len({r["id"] for r in runs}) != total_runs:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        gate = _validated_local_gate(local_gate, sha, coordinator)
        gate_ids = gate["check_run_ids"] if gate else set()
        pre_runner_cache: dict[int, bool] = {}
        verified_pre_runner_ids: set[int] = set()
        external_blockers = []
        if not required:
            for check in runs:
                if check.get("head_sha") != sha:
                    external_blockers.append("stale")
                    continue
                conclusion = check.get("conclusion")
                if check.get("status") == "completed" and conclusion in {"success", "skipped", "neutral"}:
                    continue
                if conclusion == "failure" and _verified_pre_runner_failure(repo, check, sha, gate, pre_runner_cache):
                    verified_pre_runner_ids.add(check["id"])
                    continue
                external_blockers.append(_classify(check, sha, conclusion, True))
            for status in statuses:
                if status.get("state") == "success":
                    continue
                external_blockers.append(_classify(status, sha, status.get("state"), False))
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, sha, outcome, is_run)
                if (is_run and outcome == "failure" and
                        _verified_pre_runner_failure(repo, check, sha, gate, pre_runner_cache)):
                    verified_pre_runner_ids.add(check["id"])
                    classification = "pre_runner_infrastructure"
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        gate_matches_ci = bool(gate) and (
            (gate["ci_classification"] == "not_required" and not verified_pre_runner_ids)
            or (gate["ci_classification"] == "pre_runner_infrastructure"
                and gate_ids == verified_pre_runner_ids)
        )
        if not required and not gate:
            receipt.update(classification="missing", detail="No repository-required checks are configured; provide a coordinator-approved local_gate receipt bound to this PR head.")
        else:
            # Re-read after all pages: old-head successes are never transferable.
            current = _api(f"repos/{repo}/pulls/{number}")
            if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
                receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
                return receipt
            non_pre_runner = [value for value in outcomes if value not in {"success", "pre_runner_infrastructure"}]
            if external_blockers:
                receipt["classification"] = external_blockers[0]
            elif not required:
                if gate_matches_ci:
                    receipt.update(classification="local_gate", local_gate_verified=True)
                else:
                    receipt.update(classification="infra", detail="Local gate CI disposition does not match fresh GitHub check-run evidence.")
            elif not non_pre_runner and any(value == "pre_runner_infrastructure" for value in outcomes):
                if gate_matches_ci:
                    receipt.update(classification="local_gate", local_gate_verified=True)
                else:
                    receipt.update(classification="infra", detail="Pre-run CI failure was not fully evidenced by the coordinator receipt.")
            elif not non_pre_runner:
                receipt["classification"] = "success"
            else:
                receipt["classification"] = non_pre_runner[0]
            receipt["ok"] = receipt["classification"] in {"success", "local_gate"}
        if receipt.get("local_gate_verified") and gate is not None:
            receipt["local_gate"] = {
                "head_sha": sha,
                "approved_by": coordinator,
                "reviewer": gate["reviewer"],
                "review_task_id": gate["review_task_id"],
                "reviewed_head_sha": gate["reviewed_head_sha"],
                "ci_classification": gate["ci_classification"],
                "check_run_ids": sorted(gate_ids),
            }
        receipt["external_checks"] = [
            {"id": check.get("id"), "name": check.get("name"), "head_sha": check.get("head_sha"),
             "classification": _classify(check, sha, check.get("conclusion"), True)}
            for check in runs
        ]
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
