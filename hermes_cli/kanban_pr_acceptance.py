"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.
The acceptance path is deliberately fail-closed: GitHub evidence is parsed
strictly, and provider diagnostics never become durable task data.
"""
from __future__ import annotations

from datetime import datetime
from datetime import timezone
import json
import re
import subprocess
from typing import Any
from urllib.parse import quote
from urllib.parse import urlsplit

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")
_SHA = re.compile(r"[0-9a-f]{40}")
_BRANCH = re.compile(r"[A-Za-z0-9._/-]+")
_HTTP_STATUS = re.compile(r"(?:HTTP|status(?:\s+code)?)\s*[: ]\s*([45][0-9]{2})", re.IGNORECASE)
_CHECK_CONCLUSIONS = frozenset({
    "action_required", "cancelled", "failure", "neutral", "skipped", "stale", "startup_failure", "success", "timed_out",
})
_STATUS_STATES = frozenset({"error", "failure", "pending", "success"})


class _GitHubApiError(subprocess.SubprocessError):
    """A redacted GitHub CLI failure with only its HTTP status retained."""

    def __init__(self, status_code: int | None = None):
        self.status_code = status_code
        super().__init__("GitHub API request failed")


class _AcceptanceRejected(ValueError):
    """A provider response was valid enough to classify, but cannot complete."""

    def __init__(self, classification: str, detail: str, evidence: dict[str, Any] | None = None):
        self.classification = classification
        self.detail = detail
        self.evidence = evidence or {}
        super().__init__(detail)


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _http_status(*values: str) -> int | None:
    for value in values:
        match = _HTTP_STATUS.search(value or "")
        if match:
            return int(match.group(1))
    return None


def _api_error_status(stderr: str, stdout: str) -> int | None:
    status = _http_status(stderr, stdout)
    if status is not None:
        return status
    try:
        value = json.loads(stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    raw_status = value.get("status") if isinstance(value, dict) else None
    return raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else None


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False) -> Any:
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        command += ["--paginate", "--slurp"]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.CalledProcessError as exc:
        raise _GitHubApiError(_api_error_status(str(exc.stderr), str(exc.stdout))) from None
    if result.returncode != 0:
        raise _GitHubApiError(_api_error_status(result.stderr, result.stdout))
    try:
        value = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("GitHub returned invalid JSON") from exc
    if isinstance(value, dict) and value.get("errors"):
        raise _GitHubApiError(_api_error_status(result.stderr, result.stdout))
    if isinstance(value, dict) and isinstance(value.get("status"), int) and value["status"] >= 400:
        raise _GitHubApiError(int(value["status"]))
    return value


def _graphql(query: str) -> dict[str, Any]:
    value = _api("graphql", query=query)
    if not isinstance(value, dict):
        raise ValueError("GitHub returned malformed GraphQL evidence")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Malformed {label}")
    return value


def _safe_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or hostname != "github.com" or parsed.username or parsed.password:
        return None
    return f"https://github.com{parsed.path}"


def _branch(value: Any) -> str:
    if not isinstance(value, str) or not value or value.startswith("-") or not _BRANCH.fullmatch(value):
        raise ValueError("Malformed target branch")
    return value


def _graphql_pr(owner: str, name: str, number: int) -> dict[str, Any]:
    query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
        baseRef{branchProtectionRule{id requiredStatusChecks{context app{databaseId}}}}}}}''' % (
        json.dumps(owner), json.dumps(name), number)
    value = _graphql(query)
    try:
        pull_request = value["data"]["repository"]["pullRequest"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Pull request unavailable") from exc
    if not isinstance(pull_request, dict):
        raise ValueError("Pull request unavailable")
    return pull_request


def _rest_pr(owner: str, name: str, number: int) -> dict[str, Any]:
    value = _api(f"repos/{owner}/{name}/pulls/{number}")
    if not isinstance(value, dict):
        raise ValueError("Malformed pull request evidence")
    return value


def _pr_snapshot(owner: str, name: str, number: int) -> dict[str, Any]:
    graphql_pr = _graphql_pr(owner, name, number)
    rest_pr = _rest_pr(owner, name, number)
    graph_head = _sha(graphql_pr.get("headRefOid"), "GraphQL candidate head")
    graph_branch = _branch(graphql_pr.get("baseRefName"))
    graph_state = graphql_pr.get("state")
    if not isinstance(graph_state, str):
        raise ValueError("Malformed GraphQL pull request state")
    graph_state = graph_state.upper()

    head = rest_pr.get("head")
    base = rest_pr.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise ValueError("Malformed pull request refs")
    rest_head = _sha(head.get("sha"), "REST candidate head")
    rest_branch = _branch(base.get("ref"))
    if graph_head != rest_head or graph_branch != rest_branch:
        raise _AcceptanceRejected(
            "stale",
            "PR head/base changed while collecting evidence; retry.",
            {"head_sha": graph_head, "base_branch": graph_branch},
        )

    rest_state = rest_pr.get("state")
    if not isinstance(rest_state, str):
        raise ValueError("Malformed REST pull request state")
    rest_state = rest_state.upper()
    merged = rest_pr.get("merged")
    if not isinstance(merged, bool):
        raise ValueError("Malformed REST merge state")
    if graph_state == "OPEN":
        if rest_state != "OPEN" or merged:
            raise _AcceptanceRejected(
                "stale",
                "PR state changed while collecting evidence; retry.",
                {"head_sha": graph_head, "base_branch": graph_branch},
            )
        base_sha = base.get("sha")
        if base_sha is not None:
            base_sha = _sha(base_sha, "REST base SHA")
        return {
            "graphql": graphql_pr,
            "rest": rest_pr,
            "head_sha": graph_head,
            "base_branch": graph_branch,
            "base_sha": base_sha,
            "state": "OPEN",
            "merge_commit_sha": None,
        }
    if graph_state == "MERGED" and (rest_state != "CLOSED" or not merged):
        raise _AcceptanceRejected(
            "stale",
            "PR state changed while collecting evidence; retry.",
            {"head_sha": graph_head, "base_branch": graph_branch},
        )
    if graph_state != "MERGED":
        raise _AcceptanceRejected(
            "closed",
            "PR is not reported as merged by GitHub.",
            {"head_sha": graph_head, "base_branch": graph_branch, "pr_state": graph_state},
        )
    base_sha = _sha(base.get("sha"), "REST base SHA")
    if "merge_commit_sha" not in rest_pr:
        raise ValueError("Malformed merge commit field")
    raw_merge_commit_sha = rest_pr["merge_commit_sha"]
    merge_commit_sha = (
        None
        if raw_merge_commit_sha is None
        else _sha(raw_merge_commit_sha, "merge commit SHA")
    )
    return {
        "graphql": graphql_pr,
        "rest": rest_pr,
        "head_sha": graph_head,
        "base_branch": graph_branch,
        "base_sha": base_sha,
        "state": "MERGED",
        "merge_commit_sha": merge_commit_sha,
    }


def _required_check(value: Any, app_key: str) -> tuple[str, int | None]:
    if not isinstance(value, dict) or not isinstance(value.get("context"), str) or not value["context"]:
        raise ValueError("Malformed required check")
    if app_key == "app":
        if "app" not in value:
            raise ValueError("Malformed required check app")
        app = value["app"]
        if app is not None and not isinstance(app, dict):
            raise ValueError("Malformed required check app")
        if app is not None and "databaseId" not in app:
            raise ValueError("Malformed required check app id")
        app_id = app["databaseId"] if app is not None else None
    else:
        if app_key not in value:
            raise ValueError("Malformed required check app id")
        app_id = value[app_key]
    if app_id is not None and (not isinstance(app_id, int) or isinstance(app_id, bool)):
        raise ValueError("Malformed required check app id")
    return value["context"], app_id


def _required_sort_key(item: tuple[str, int | None]) -> tuple[str, int, int]:
    return item[0], 0 if item[1] is None else 1, -1 if item[1] is None else item[1]


def _required_policy(pr: dict[str, Any], rules: list[dict[str, Any]], rules_source: str) -> tuple[set[tuple[str, int | None]], dict[str, Any]]:
    base_ref = pr.get("baseRef")
    if not isinstance(base_ref, dict) or "branchProtectionRule" not in base_ref:
        raise ValueError("Malformed base ref")
    protection = base_ref["branchProtectionRule"]
    if protection is None:
        classic_checks: list[Any] = []
        classic_count = 0
        classic_ids: list[Any] = []
    elif (not isinstance(protection, dict)
          or not isinstance(protection.get("requiredStatusChecks"), list)):
        raise ValueError("Malformed branch protection")
    else:
        classic_checks = protection["requiredStatusChecks"]
        classic_count = 1
        classic_id = protection.get("id")
        classic_ids = [classic_id] if isinstance(classic_id, (str, int)) and not isinstance(classic_id, bool) else []

    required = {_required_check(check, "app") for check in classic_checks}
    rule_required: set[tuple[str, int | None]] = set()
    rule_check_count = 0
    rule_ids: list[Any] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Malformed repository rule")
        rule_type = rule.get("type")
        if not isinstance(rule_type, str) or not rule_type:
            raise ValueError("Malformed repository rule type")
        rule_id = rule.get("id")
        if isinstance(rule_id, (str, int)) and not isinstance(rule_id, bool):
            rule_ids.append(rule_id)
        if rule_type.upper() != "REQUIRED_STATUS_CHECKS":
            continue
        parameters = rule.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("Malformed required-status-check rule")
        checks = parameters.get("required_status_checks", parameters.get("requiredStatusChecks"))
        if not isinstance(checks, list):
            raise ValueError("Malformed required-status-check rule")
        rule_check_count += len(checks)
        rule_required.update(_required_check(check, "integration_id") for check in checks)
    required.update(rule_required)
    ordered = sorted(required, key=_required_sort_key)
    policy = {
        "classic_object_count": classic_count,
        "classic_ids": classic_ids,
        "rules_object_count": len(rules),
        "rules_ids": rule_ids,
        "rules_source": rules_source,
        "classic_required_check_count": len(classic_checks),
        "rules_required_check_count": rule_check_count,
        "required_count": len(ordered),
    }
    return required, policy


def _rules_for_branch(repo: str, branch: str) -> tuple[list[dict[str, Any]], str]:
    value = _api(
        f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100",
        paginate=True,
    )
    if not isinstance(value, list):
        raise ValueError("Malformed repository rules pagination")
    rules: list[dict[str, Any]] = []
    for page in value:
        if not isinstance(page, list):
            raise ValueError("Malformed repository rules page")
        for rule in page:
            if not isinstance(rule, dict):
                raise ValueError("Malformed repository rule")
            rules.append(rule)
    return rules, "available"


def _rules_with_fallback(repo: str, branch: str) -> tuple[list[dict[str, Any]], str]:
    try:
        return _rules_for_branch(repo, branch)
    except _GitHubApiError as exc:
        # A plan-level Rules API 403 is the sole unavailable-endpoint case. All
        # other failures remain infrastructure failures instead of silently
        # widening the acceptance policy.
        if exc.status_code == 403:
            return [], "unavailable_http_403"
        raise


def _pages(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Malformed {label} pagination")
    return value


def _normalize_check_run(value: Any, sha: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Malformed check run")
    if not isinstance(value.get("id"), int) or isinstance(value["id"], bool):
        raise ValueError("Malformed check run id")
    name = value.get("name")
    head_sha = value.get("head_sha")
    status = value.get("status")
    if not isinstance(name, str) or not name or not isinstance(status, str):
        raise ValueError("Malformed check run")
    head_sha = _sha(head_sha, "check-run head SHA")
    if "app" not in value:
        raise ValueError("Malformed check run app")
    app = value["app"]
    if app is None:
        app_id = None
    elif isinstance(app, dict):
        if "id" not in app:
            raise ValueError("Malformed check run app id")
        app_id = app["id"]
        if app_id is not None and (not isinstance(app_id, int) or isinstance(app_id, bool)):
            raise ValueError("Malformed check run app id")
    else:
        raise ValueError("Malformed check run app")
    conclusion = value.get("conclusion")
    if conclusion is not None and not isinstance(conclusion, str):
        raise ValueError("Malformed check run conclusion")
    safe_conclusion = conclusion.lower() if conclusion and conclusion.lower() in _CHECK_CONCLUSIONS else None
    return {
        "id": value["id"],
        "name": name,
        "head_sha": head_sha,
        "status": status.lower(),
        "conclusion": safe_conclusion,
        "app_id": app_id,
        "html_url": _safe_url(value.get("html_url")),
    }


def _check_runs(value: Any, sha: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pages = _pages(value, "check-run")
    runs: list[dict[str, Any]] = []
    total_count: int | None = None
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("check_runs"), list):
            raise ValueError("Malformed check-run page")
        page_total = page.get("total_count")
        if not isinstance(page_total, int) or isinstance(page_total, bool) or page_total < 0:
            raise ValueError("Malformed check-run count")
        if total_count is None:
            total_count = page_total
        elif total_count != page_total:
            raise ValueError("Inconsistent check-run count")
        runs.extend(_normalize_check_run(run, sha) for run in page["check_runs"])
    total_count = total_count if total_count is not None else 0
    ids = [run["id"] for run in runs]
    if len(set(ids)) != len(ids) or len(ids) != total_count:
        raise ValueError("Incomplete check-run pagination")
    return runs, {"pages": len(pages), "count": len(runs), "total_count": total_count}


def _normalize_status(value: Any, sha: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Malformed legacy status")
    if not isinstance(value.get("id"), int) or isinstance(value["id"], bool):
        raise ValueError("Malformed legacy status id")
    context = value.get("context")
    state = value.get("state")
    if not isinstance(context, str) or not context or not isinstance(state, str):
        raise ValueError("Malformed legacy status")
    normalized_state = state.lower()
    return {
        "id": value["id"],
        "context": context,
        "sha": _sha(value.get("sha"), "legacy status SHA"),
        "state": normalized_state if normalized_state in _STATUS_STATES else None,
        "target_url": _safe_url(value.get("target_url")),
    }


def _statuses(value: Any, sha: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pages = _pages(value, "status")
    statuses: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("Malformed status page")
        statuses.extend(_normalize_status(status, sha) for status in page)
    ids = [status["id"] for status in statuses]
    if len(set(ids)) != len(ids):
        raise ValueError("Incomplete status pagination")
    return statuses, {"pages": len(pages), "count": len(statuses)}


def _classify(check: dict[str, Any], sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run:
        if check.get("status") != "completed":
            return "pending"
        return {
            "success": "success",
            "failure": "failure",
            "cancelled": "failure",
            "timed_out": "failure",
            "action_required": "failure",
            "neutral": "failure",
            "skipped": "failure",
            "stale": "failure",
            "startup_failure": "failure",
        }.get(outcome or "", "infra")
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(
        outcome or "", "infra"
    )


def _is_unpinned(app_id: int | None) -> bool:
    return app_id in (None, -1)


def _latest_statuses(statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for status in statuses:
        current = latest.get(status["context"])
        if current is None or status["id"] > current["id"]:
            latest[status["context"]] = status
    return list(latest.values())


def _outcome_worst(outcomes: list[str]) -> str:
    if not outcomes:
        return "missing"
    priority = {"infra": 0, "stale": 1, "failure": 2, "pending": 3, "missing": 4, "success": 5}
    return min(outcomes, key=lambda outcome: priority.get(outcome, 0))


def _safe_actor(metadata: dict[str, Any]) -> str:
    actor = metadata.get("actor")
    if isinstance(actor, str) and re.fullmatch(r"[A-Za-z0-9_.@-]{1,100}", actor):
        return actor
    return "kanban_complete"


def _reviewed_head(metadata: dict[str, Any]) -> str | None:
    review = metadata.get("review")
    if isinstance(review, dict):
        for key in ("reviewed_head_sha", "reviewed_head", "reviewed_sha", "head_sha"):
            if key in review:
                return review[key]
    for key in ("reviewed_head_sha", "reviewed_head", "reviewed_sha"):
        if key in metadata:
            return metadata[key]
    return None


def _approved_paths(metadata: dict[str, Any]) -> list[str]:
    raw: Any = None
    sources = [metadata]
    review = metadata.get("review")
    if isinstance(review, dict):
        sources.append(review)
    for source in sources:
        for key in ("approved_allowlist", "allowlist", "approved_paths", "path_allowlist", "scope_allowlist"):
            if key in source:
                raw = source[key]
                break
        if raw is not None:
            break
    if isinstance(raw, dict):
        raw = raw.get("paths")
    if not isinstance(raw, (list, tuple)):
        raise _AcceptanceRejected("missing", "An approved path allowlist is required for merged PR completion.")
    paths: list[str] = []
    for path in raw:
        if not isinstance(path, str) or not path.strip() or "\x00" in path:
            raise _AcceptanceRejected("infra", "The approved path allowlist is malformed.")
        if path in paths:
            raise _AcceptanceRejected("infra", "The approved path allowlist contains duplicate paths.")
        paths.append(path)
    return paths


def _file_paths(value: Any, label: str) -> tuple[list[str], int]:
    pages = _pages(value, label)
    paths: list[str] = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError(f"Malformed {label} page")
        for file_entry in page:
            if not isinstance(file_entry, dict) or not isinstance(file_entry.get("filename"), str):
                raise ValueError(f"Malformed {label} entry")
            filename = file_entry["filename"]
            if not filename or "\x00" in filename or filename in paths:
                raise ValueError(f"Malformed {label} path set")
            paths.append(filename)
    return paths, len(pages)


def _pr_files(repo: str, number: int) -> tuple[list[str], int]:
    return _file_paths(
        _api(f"repos/{repo}/pulls/{number}/files?per_page=100", paginate=True),
        "pull-request file",
    )


def _merge_commit(repo: str, merge_sha: str) -> tuple[list[str], list[str], int]:
    value = _api(f"repos/{repo}/commits/{merge_sha}?per_page=100", paginate=True)
    pages = _pages(value, "merge commit")
    if not pages:
        raise ValueError("Missing merge commit evidence")
    parents: list[str] | None = None
    files: list[str] = []
    for page in pages:
        if not isinstance(page, dict) or page.get("sha") != merge_sha:
            raise ValueError("Malformed merge commit evidence")
        raw_parents = page.get("parents")
        if not isinstance(raw_parents, list):
            raise ValueError("Malformed merge commit parents")
        page_parents = []
        for parent in raw_parents:
            if not isinstance(parent, dict):
                raise ValueError("Malformed merge commit parent")
            page_parents.append(_sha(parent.get("sha"), "merge commit parent SHA"))
        if parents is None:
            parents = page_parents
        elif parents != page_parents:
            raise ValueError("Inconsistent merge commit parents")
        page_files = page.get("files")
        if not isinstance(page_files, list):
            raise ValueError("Malformed merge commit files")
        for file_entry in page_files:
            if not isinstance(file_entry, dict) or not isinstance(file_entry.get("filename"), str):
                raise ValueError("Malformed merge commit file")
            filename = file_entry["filename"]
            if not filename or "\x00" in filename or filename in files:
                raise ValueError("Malformed merge commit path set")
            files.append(filename)
    return parents or [], files, len(pages)


def _run_git(arguments: list[str]):
    return subprocess.run(
        ["git", *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _target_reachability(branch: str, merge_sha: str) -> dict[str, Any]:
    target_ref = f"origin/{branch}"
    fetch_command = ["git", "fetch", "origin", branch]
    fetch = _run_git(fetch_command[1:])
    target: dict[str, Any] = {
        "ref": target_ref,
        "fetched_ref": target_ref,
        "fetch": {"command": " ".join(fetch_command), "outcome": "success" if fetch.returncode == 0 else "failure"},
        "reachable": False,
    }
    if fetch.returncode != 0:
        raise _AcceptanceRejected(
            "infra",
            "Target branch fetch failed; refresh origin and retry.",
            {"target": target},
        )
    target_sha_command = ["git", "rev-parse", "--verify", target_ref]
    target_sha_result = _run_git(target_sha_command[1:])
    if target_sha_result.returncode != 0:
        target["target_sha"] = {"command": " ".join(target_sha_command), "outcome": "failure"}
        raise _AcceptanceRejected("infra", "Fresh target SHA could not be verified; retry.", {"target": target})
    target["sha"] = _sha(target_sha_result.stdout.strip(), "fetched target SHA")
    target["target_sha"] = {"command": " ".join(target_sha_command), "outcome": "success"}
    reach_command = ["git", "merge-base", "--is-ancestor", merge_sha, target_ref]
    reach = _run_git(reach_command[1:])
    target["reachability"] = {
        "command": " ".join(reach_command),
        "outcome": "reachable" if reach.returncode == 0 else "unreachable",
    }
    target["reachable"] = reach.returncode == 0
    if reach.returncode == 1:
        raise _AcceptanceRejected("unreachable", "The merge commit is not reachable from freshly fetched target main.", {"target": target})
    if reach.returncode != 0:
        target["reachability"]["outcome"] = "failure"
        raise _AcceptanceRejected("infra", "Target reachability could not be verified; retry.", {"target": target})
    return target


def _policy_signature(required: set[tuple[str, int | None]], policy: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(sorted(required, key=_required_sort_key)),
        policy.get("classic_object_count"),
        tuple(sorted(policy.get("classic_ids", []), key=lambda value: (type(value).__name__, repr(value)))),
        policy.get("rules_object_count"),
        tuple(sorted(policy.get("rules_ids", []), key=lambda value: (type(value).__name__, repr(value)))),
        policy.get("rules_source"),
        policy.get("classic_required_check_count"),
        policy.get("rules_required_check_count"),
    )


def collect_acceptance(contract: str, published_pr: str | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, dict) else {}
    receipt: dict[str, Any] = {
        "ok": False,
        "classification": "missing",
        "head_sha": None,
        "candidate_sha": None,
        "base_sha": None,
        "base_branch": None,
        "pr_state": None,
        "pr_url": None,
        "reviewed_head_sha": None,
        "merge_commit_sha": None,
        "required": [],
        "checks": [],
        "actor": _safe_actor(metadata),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
        "Use kanban_block if human input is needed; receipts remain on the task event log.",
    }
    try:
        if not isinstance(contract, str):
            raise _AcceptanceRejected("missing", "A valid persisted completion contract is required for PR completion.")
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            raise _AcceptanceRejected(
                "missing",
                "Supply metadata.published_pr matching the persisted completion contract.",
            )
        repo, number_text = match[1], match[2]
        number = int(number_text)
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        initial = _pr_snapshot(owner, name, number)
        receipt.update(
            head_sha=initial["head_sha"],
            candidate_sha=initial["head_sha"],
            base_sha=initial["base_sha"],
            base_branch=initial["base_branch"],
            pr_state=initial["state"],
        )
        if initial["state"] == "OPEN":
            raise _AcceptanceRejected("open", "PR is still open; an open PR cannot satisfy Done.")

        reviewed_head = _reviewed_head(metadata)
        if reviewed_head is not None:
            reviewed_head = _sha(reviewed_head, "reviewed head SHA")
        receipt["reviewed_head_sha"] = reviewed_head
        if reviewed_head is None:
            raise _AcceptanceRejected("missing", "An exact independently reviewed head SHA is required for merged PR completion.")
        if reviewed_head != initial["head_sha"]:
            raise _AcceptanceRejected(
                "stale",
                "The merged PR head does not match the independently reviewed head; retry with the reviewed candidate.",
            )

        merge_sha = initial["merge_commit_sha"]
        receipt["merge_commit_sha"] = merge_sha
        if merge_sha is None:
            raise _AcceptanceRejected("missing", "GitHub reports a merged PR without a merge commit; retry after the merge is available.")
        allowlist = _approved_paths(metadata)

        rules, rules_source = _rules_with_fallback(repo, initial["base_branch"])
        required, policy = _required_policy(initial["graphql"], rules, rules_source)
        ordered_required = sorted(required, key=_required_sort_key)
        receipt["required"] = [{"context": context, "app_id": app_id} for context, app_id in ordered_required]
        receipt["policy"] = policy
        if not required:
            raise _AcceptanceRejected(
                "missing",
                "No repository-required checks are configured; explicitly use a local-only contract for non-CI tasks.",
            )

        runs, run_counts = _check_runs(
            _api(f"repos/{repo}/commits/{initial['head_sha']}/check-runs?per_page=100&filter=latest", paginate=True),
            initial["head_sha"],
        )
        statuses, status_counts = _statuses(
            _api(f"repos/{repo}/commits/{initial['head_sha']}/statuses?per_page=100", paginate=True),
            initial["head_sha"],
        )
        latest_statuses = _latest_statuses(statuses)
        receipt["check_counts"] = {"check_runs": run_counts, "statuses": status_counts}
        outcomes: list[str] = []
        for context, app_id in ordered_required:
            selected_runs = [
                run for run in runs
                if run["name"] == context and (_is_unpinned(app_id) or run["app_id"] == app_id)
            ]
            selected_statuses = [
                status for status in latest_statuses
                if _is_unpinned(app_id) and status["context"] == context
            ]
            selected: list[tuple[dict[str, Any], bool]] = [(run, True) for run in selected_runs]
            selected.extend((status, False) for status in selected_statuses)
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": initial["head_sha"]})
                continue
            context_outcomes: list[str] = []
            for check, is_run in selected:
                outcome = check.get("conclusion") if is_run else check.get("state")
                classification = _classify(check, initial["head_sha"], outcome, is_run)
                context_outcomes.append(classification)
                receipt["checks"].append({
                    "name": context,
                    "id": check["id"],
                    "type": "check_run" if is_run else "status",
                    "app_id": check.get("app_id"),
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification,
                    "conclusion": outcome,
                })
            outcomes.append(_outcome_worst(context_outcomes))
        check_classification = _outcome_worst(outcomes)
        if check_classification != "success":
            raise _AcceptanceRejected(
                check_classification,
                "Required checks are not all terminal green for the exact candidate head; retry after the checks settle.",
            )

        whole_pr, whole_pages = _pr_files(repo, number)
        merge_parents, merged_delta, delta_pages = _merge_commit(repo, merge_sha)
        declared_method = metadata.get("merge_method")
        review = metadata.get("review")
        if declared_method is None and isinstance(review, dict):
            declared_method = review.get("merge_method")
        if declared_method is not None and declared_method not in {"merge", "squash"}:
            raise _AcceptanceRejected("infra", "The merge method is unsupported; expected merge or squash.")
        if len(merge_parents) == 2:
            if declared_method == "squash" or merge_parents[1] != reviewed_head:
                raise _AcceptanceRejected("stale", "The merge commit does not contain the independently reviewed PR head as its source parent.")
            merge_method = "merge"
        elif len(merge_parents) == 1:
            if declared_method != "squash" or merge_parents[0] == reviewed_head:
                raise _AcceptanceRejected("stale", "The merge commit does not prove squash source semantics for the reviewed PR head.")
            merge_method = "squash"
        else:
            raise _AcceptanceRejected("infra", "The merge commit has unsupported parent semantics.")
        receipt["merge_method"] = merge_method
        receipt["merge_parents"] = merge_parents
        receipt["scope"] = {
            "approved": allowlist,
            "whole_pr": whole_pr,
            "merged_delta": merged_delta,
            "whole_pr_pages": whole_pages,
            "merged_delta_pages": delta_pages,
        }
        if set(whole_pr) != set(allowlist) or set(merged_delta) != set(allowlist):
            raise _AcceptanceRejected(
                "failure",
                "Merged whole-PR and delta paths do not exactly match the approved allowlist.",
            )

        receipt["target"] = _target_reachability(initial["base_branch"], merge_sha)

        try:
            final = _pr_snapshot(owner, name, number)
        except _AcceptanceRejected as exc:
            if exc.classification in {"open", "closed", "stale"}:
                raise _AcceptanceRejected("stale", "PR head/base/state changed while collecting evidence; retry.") from None
            raise
        if (
            final["state"] != "MERGED"
            or final["head_sha"] != initial["head_sha"]
            or final["base_branch"] != initial["base_branch"]
            or final["base_sha"] != initial["base_sha"]
            or final["merge_commit_sha"] != merge_sha
        ):
            raise _AcceptanceRejected("stale", "PR head/base/state changed while collecting evidence; retry.")
        final_rules, final_rules_source = _rules_with_fallback(repo, final["base_branch"])
        final_required, final_policy = _required_policy(final["graphql"], final_rules, final_rules_source)
        if _policy_signature(final_required, final_policy) != _policy_signature(required, policy):
            raise _AcceptanceRejected("stale", "Required-check policy changed while collecting evidence; retry.")

        receipt["ok"] = True
        receipt["classification"] = "success"
        return receipt
    except _AcceptanceRejected as exc:
        receipt.update(classification=exc.classification, detail=exc.detail)
        receipt.update(exc.evidence)
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        # Never persist gh/git stderr (credentials, host details, and tokens);
        # the failed phase is represented by a stable actionable classification.
        receipt.update(
            classification="infra",
            detail="GitHub acceptance evidence unavailable or incomplete; check gh authentication/API access and retry.",
        )
        return receipt
