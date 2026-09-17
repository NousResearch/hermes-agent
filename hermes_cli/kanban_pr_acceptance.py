"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.

One provider-side failure is not an infrastructure fault: GitHub answers
``repos/{repo}/rules/branches/{branch}`` on a private repository on a free plan
with ``403 Upgrade to GitHub Pro or make this repository public``. That endpoint
is the only way to enumerate repository-required checks, so the contract would
be unsatisfiable no matter what the work looks like. Only that exact plan
restriction degrades acceptance to ``verification_mode="local-only"``, and only
when the contract is still proven from evidence that does not need the gated
endpoint — see :func:`_degraded_acceptance`. Every other failure (401, generic
or scope-related 403, 404, 429, timeout/network/5xx, failing or pending CI,
diverged head, missing PR/commit) keeps the contract unaccepted.
"""
from __future__ import annotations

import json
import re
import subprocess
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{40}")
_HTTP_STATUS = re.compile(r"\(HTTP (\d{3})\)")
# Wording GitHub uses when an endpoint is gated by the repository's plan rather
# than by the caller's identity or token scopes. Generic 403s (permission
# denied, missing scopes, SSO, secondary rate limits) never match.
_CAPABILITY_MESSAGES = ("upgrade to github pro", "make this repository public")
# Observed check-run conclusions that carry no negative signal for a degraded
# contract; anything else (failure, cancelled, timed_out, action_required, ...)
# blocks the fallback.
_GREEN_CONCLUSIONS = {"success", "skipped", "neutral"}


class ApiError(RuntimeError):
    """A failed ``gh api`` call: HTTP status (``None`` when nothing was answered) and gh's short message."""

    def __init__(self, status: int | None, message: str):
        self.status = status
        self.message = " ".join((message or "").split())[:200]
        super().__init__(f"HTTP {status}: {self.message}" if status is not None else self.message)


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _failure(error: Exception) -> ApiError:
    """Turn a subprocess failure into a status + one-line message (never the whole stderr)."""
    text = (getattr(error, "stderr", None) or getattr(error, "output", None) or "")
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    detail = re.sub(r"^gh:\s*", "", lines[-1]) if lines else ""
    match = _HTTP_STATUS.search(detail)
    status = int(match[1]) if match else None
    return ApiError(status, _HTTP_STATUS.sub("", detail).strip())


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False):
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise _failure(error) from error
    value = json.loads(result.stdout)
    if isinstance(value, dict) and value.get("errors"):
        raise ValueError("GitHub returned incomplete GraphQL evidence")
    return value


def _capability_unavailable(error: ApiError) -> bool:
    """Whether a failure is unambiguously the plan/capability restriction, not auth or a transient fault."""
    if error.status != 403:
        return False
    lowered = error.message.lower()
    return any(fragment in lowered for fragment in _CAPABILITY_MESSAGES)


def _api_detail(error: ApiError) -> str:
    return f"HTTP {error.status}: {error.message}" if error.status is not None else "network/API failure (no HTTP status)"


def collect_acceptance(contract: str, published_pr: str | None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [], "verification_mode": "github-api",
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
        if not _SHA.fullmatch(sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = {(r["context"], (r.get("app") or {}).get("databaseId")) for r in protection.get("requiredStatusChecks", [])}
        rules_endpoint = f"repos/{repo}/rules/branches/{quote(branch, safe='')}"
        degraded = None
        try:
            rules = _api(rules_endpoint + "?per_page=100", paginate=True)
        except ApiError as error:
            if not _capability_unavailable(error):
                raise
            # The endpoint is unreadable for this repository's plan, not for
            # this work. Record the original status/message and keep collecting.
            degraded = {"check": rules_endpoint, "status": error.status, "message": error.message}
            rules = []
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required and degraded is None:
            receipt["detail"] = "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks."
            return receipt
        pages = _api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _api(f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
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
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        # Re-read after all pages: old-head successes are never transferable.
        current = _api(f"repos/{repo}/pulls/{number}")
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        if degraded is not None:
            return _degraded_acceptance(receipt, degraded, repo=repo, branch=branch,
                                        sha=sha, number=number, runs=runs, statuses=statuses,
                                        outcomes=outcomes, current=current)
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except (ApiError, OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh stderr (credentials/host details); the failed phase is actionable.
        receipt.update(classification="infra", detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.")
        return receipt


def _degraded_acceptance(receipt: dict, degraded: dict, *, repo: str, branch: str, sha: str,
                         number: int, runs: list, statuses: list, outcomes: list,
                         current: dict) -> dict:
    """Validate a contract whose required-check enumeration was plan-gated.

    Without the rules endpoint the declared required checks cannot be read, so
    the contract is proven from independent evidence instead: the PR is OPEN or
    MERGED at the exact collected head/base, that head commit exists on the
    remote, the head fast-forwards the base, and no observed check or legacy
    status on the head is failing or pending (with at least one green check).
    A merged PR additionally needs its merge commit on the remote. Anything
    less stays unaccepted — the degraded receipt records why.
    """
    evidence: list[dict] = []
    receipt.update(verification_mode="local-only",
                   degraded_reason="provider_capability",
                   degraded_check=degraded["check"],
                   degraded_status=degraded["status"],
                   degraded_message=degraded["message"],
                   alternative_evidence=evidence)

    def record(check: str, ok: bool, detail: str) -> None:
        evidence.append({"check": check, "ok": ok, "detail": detail})

    def reject(classification: str, detail: str) -> dict:
        receipt.update(classification=classification, detail=detail)
        return receipt

    # PR existence, state and exact base/head (already re-read under one call).
    state = current.get("state")
    record(f"repos/{repo}/pulls/{number}", True,
           f"state={state} merged={bool(current.get('merged'))} "
           f"head={current['head']['sha']} base={current['base']['ref']}")

    # The expected commit must exist on the remote under the collected head.
    commit_check = f"repos/{repo}/commits/{sha}"
    transport = False
    try:
        commit = _api(commit_check)
        commit_ok, commit_status = commit.get("sha") == sha, None
        commit_detail = "expected commit present on the remote" if commit_ok else "remote commit sha mismatch"
    except ApiError as error:
        commit_ok, commit_status, commit_detail = False, error.status, _api_detail(error)
        transport = error.status is None
    record(commit_check, commit_ok, commit_detail)
    if not commit_ok:
        return reject("infra" if transport else "missing",
                      "Expected commit is not verifiable on the remote; the provider-capability fallback cannot validate this contract.")

    # base...head must fast-forward: a diverged/behind head is a merge conflict, not a publication.
    compare_check = f"repos/{repo}/compare/{branch}...{sha}"
    transport = False
    try:
        comparison = _api(compare_check)
        state_name = comparison.get("status")
        compare_ok = state_name in {"ahead", "identical"} and bool((comparison.get("merge_base_commit") or {}).get("sha"))
        compare_detail = f"status={state_name}"
    except ApiError as error:
        compare_ok, compare_detail, transport = False, _api_detail(error), error.status is None
    record(compare_check, compare_ok, compare_detail)
    if not compare_ok:
        return reject("infra" if transport else "missing",
                      "Head does not fast-forward the base branch; the provider-capability fallback cannot validate this contract.")

    # Observed CI on the exact head: nothing failing or pending, at least one green.
    observed = [r for r in runs if r.get("head_sha", r.get("sha")) == sha]
    legacy = [s for s in statuses if s.get("sha") == sha]
    broken = [r for r in observed
              if r.get("status") != "completed" or r.get("conclusion") not in _GREEN_CONCLUSIONS]
    blocked_statuses = [s for s in legacy if s.get("state") != "success"]
    green = ([r for r in observed if r.get("conclusion") == "success"]
             or [s for s in legacy if s.get("state") == "success"])
    ci_class = "success"
    if broken or blocked_statuses:
        ci_class = "pending" if any(r.get("status") != "completed" for r in broken) else "failure"
        if any(s.get("state") == "pending" for s in blocked_statuses):
            ci_class = "pending"
    elif not green:
        ci_class = "missing"
    record(f"repos/{repo}/commits/{sha}/check-runs", ci_class == "success",
           f"{len(observed)} observed check run(s), {len(broken)} failing/pending, {len(green)} green")
    if outcomes:
        record("declared-required-contexts", all(x == "success" for x in outcomes),
               f"{len(outcomes)} declared required context(s) evaluated: {sorted(set(outcomes))}")
    if ci_class != "success" or any(x != "success" for x in outcomes):
        first = next((x for x in outcomes if x != "success"), ci_class)
        return reject(first if first in {"failure", "pending", "stale", "infra", "missing"} else "missing",
                      "Required checks are not provably successful on the expected head; the provider-capability fallback cannot validate this contract.")

    # A merged contract must show its merge commit on the remote.
    if current.get("merged"):
        merge_sha = current.get("merge_commit_sha")
        merge_check = f"repos/{repo}/pulls/{number}/merge-commit"
        if not (isinstance(merge_sha, str) and _SHA.fullmatch(merge_sha)):
            record(merge_check, False, f"merge_commit_sha={merge_sha!r}")
            return reject("missing", "Merged contract requires a merge commit on the remote; the provider-capability fallback cannot validate this contract.")
        try:
            landed = _api(f"repos/{repo}/commits/{merge_sha}")
            merge_ok = landed.get("sha") == merge_sha
            merge_detail = f"merge commit {merge_sha} present on the remote" if merge_ok else "merge commit sha mismatch"
            merge_transport = False
        except ApiError as error:
            merge_ok, merge_detail, merge_transport = False, _api_detail(error), error.status is None
        record(merge_check, merge_ok, merge_detail)
        if not merge_ok:
            return reject("infra" if merge_transport else "missing",
                          "Merge commit is not verifiable on the remote; the provider-capability fallback cannot validate this contract.")

    if any(not item["ok"] for item in evidence):
        return reject("missing", "Provider-capability fallback evidence is incomplete; the contract is not validated.")

    receipt.update(ok=True, classification="success",
                   detail=f"GitHub plan restriction on {degraded['check']} (HTTP {degraded['status']}: {degraded['message']}); "
                          "contract validated from alternative evidence: PR state, remote head commit, base...head comparison, "
                          "and successful observed checks.")
    return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
