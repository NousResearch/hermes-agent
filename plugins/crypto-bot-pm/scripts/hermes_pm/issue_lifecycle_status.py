#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

try:
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        load_read_token,
        redact_text,
        redact_url,
        repo_api_path,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        DEFAULT_GITEA_BASE_URL,
        DEFAULT_OWNER,
        DEFAULT_REPO,
        load_read_token,
        redact_text,
        redact_url,
        repo_api_path,
    )


ISSUE_LIFECYCLE_SCHEMA_VERSION = "hermes.pm.issue_lifecycle_status.v1"
DEFAULT_ISSUE_INDEX = 1
EXPECTED_PM_SEED_ISSUE_TITLE = (
    "[Hermes PM] Establish initial PM-managed backlog item"
)
EXPECTED_SAFE_LABELS = {
    "approval-required",
    "hermes-pm",
    "managed-project",
    "needs-triage",
    "pm-review",
    "proposal-only",
    "read-only",
}
PM_6B_BODY_MARKERS = (
    "Hermes PM Checkpoint 6B",
    "Hermes PM Checkpoint 6.",
    "first approval-gated forge-write rehearsal",
    "proposal-only backlog expansion",
)
CHECKPOINT_6_BODY_SNIPPETS = (
    "Created by Hermes PM Checkpoint 6.",
    "This is the first approval-gated forge-write rehearsal.",
    "Scope is project-management metadata only.",
    (
        "No labels, projects, comments, PRs, workflows, runners, deploys, "
        "runtime actions, branch-writer actions, financial actions, or secret "
        "access were approved."
    ),
    (
        "The Operator should review whether Hermes PM may continue toward "
        "proposal-only backlog expansion."
    ),
)

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "creates_issues": False,
    "creates_labels": False,
    "creates_comments": False,
    "mutates_projects": False,
    "runs_workflows": False,
    "starts_runners": False,
    "branch_writer_invoked": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}

Transport = Callable[[urllib.request.Request, int], tuple[int, bytes]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_transport(
    request: urllib.request.Request,
    timeout: int,
) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class IssueLifecycleReadClient:
    mutation_capability = False
    allowed_methods = ("GET", "HEAD")
    blocked_methods = ("POST", "PUT", "PATCH", "DELETE")

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GITEA_BASE_URL,
        token: str | None = None,
        timeout: int = 20,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._transport = transport or default_transport
        self.methods_used: list[str] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        return self.request("GET", path, params=params)

    def head(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        return self.request("HEAD", path, params=params)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        method = method.upper()
        if method not in self.allowed_methods:
            raise ValueError(
                "Issue lifecycle reader permits GET/HEAD only; "
                f"blocked {method}."
            )
        url = self._build_url(path, params=params)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        request = urllib.request.Request(url, headers=headers, method=method)
        self.methods_used.append(method)
        status, body = self._transport(request, self.timeout)
        if method == "HEAD" or not body:
            return status, {}
        try:
            return status, json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            decoded = body.decode("utf-8", "replace")
            return status, {"non_json_body": redact_text(decoded)}

    def _build_url(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{clean_path}"
        if query:
            url = f"{url}?{query}"
        return url


def issue_api_path(owner: str, repo: str, issue_index: Any) -> str:
    quoted = urllib.parse.quote(str(issue_index), safe="")
    return repo_api_path(owner, repo, f"/issues/{quoted}")


def _list_payload(payload: Any, key: str | None = None) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if key and isinstance(payload.get(key), list):
            return payload[key]
        for candidate in ("items", "data", "comments", "workflow_runs", "projects"):
            if isinstance(payload.get(candidate), list):
                return payload[candidate]
    return []


def _label_names(payload: Any) -> list[str]:
    labels = _list_payload(payload)
    names: list[str] = []
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
            if name:
                names.append(str(name))
        elif label:
            names.append(str(label))
    return sorted(dict.fromkeys(names))


def _assignee_names(payload: Any) -> list[str]:
    assignees = _list_payload(payload)
    names: list[str] = []
    for assignee in assignees:
        if isinstance(assignee, dict):
            name = assignee.get("login") or assignee.get("username") or assignee.get(
                "full_name"
            )
            if name:
                names.append(str(name))
        elif assignee:
            names.append(str(assignee))
    return sorted(dict.fromkeys(names))


def _milestone_title(payload: Any) -> str | None:
    if isinstance(payload, dict):
        title = payload.get("title")
        return str(title) if title else None
    if payload:
        return str(payload)
    return None


def _is_pull_request_issue(payload: dict[str, Any]) -> bool:
    pull_request = payload.get("pull_request")
    return isinstance(pull_request, dict) and bool(pull_request)


def _body_marker_present(body: str) -> bool:
    if not body:
        return False
    if any(marker in body for marker in PM_6B_BODY_MARKERS):
        return True
    return all(snippet in body for snippet in CHECKPOINT_6_BODY_SNIPPETS)


def _comments_summary(status: int | None, payload: Any) -> dict[str, Any]:
    comments = _list_payload(payload, key="comments")
    checked = status == 200
    return {
        "checked": checked,
        "endpoint_status": status,
        "state": (
            "absent"
            if checked and not comments
            else "present"
            if checked
            else "unknown"
        ),
        "count": len(comments) if checked else None,
        "only_read_checked": True,
    }


def _project_summary(status: int | None, payload: Any) -> dict[str, Any]:
    projects = _list_payload(payload, key="projects")
    unsupported = status in {404, 405, 501}
    return {
        "checked": status is not None,
        "endpoint_status": status,
        "state": (
            "unsupported"
            if unsupported
            else "empty"
            if status == 200 and not projects
            else "available"
            if status == 200
            else "unknown"
        ),
        "project_count": len(projects) if status == 200 else 0,
        "cards_mutated": False,
        "mutated": False,
    }


def _workflow_summary(status: int | None, payload: Any) -> dict[str, Any]:
    runs = _list_payload(payload, key="workflow_runs")
    unsupported = status in {404, 405, 501}
    return {
        "checked": status is not None,
        "endpoint_status": status,
        "state": (
            "unsupported"
            if unsupported
            else "empty"
            if status == 200 and not runs
            else "available"
            if status == 200
            else "unknown"
        ),
        "recent_run_count": len(runs) if status == 200 else None,
        "runs_workflows": False,
    }


def _runner_summary() -> dict[str, Any]:
    return {
        "checked": False,
        "state": "unknown",
        "online": "unknown",
        "safely_knowable": False,
        "starts_runners": False,
    }


def _safe_get(
    client: IssueLifecycleReadClient,
    path: str,
    warnings: list[str],
    *,
    params: dict[str, Any] | None = None,
    optional: bool = False,
) -> tuple[int | None, Any]:
    try:
        return client.get(path, params=params)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        message = f"GET {path} failed: {redact_text(str(exc))}"
        if optional:
            warnings.append(message)
            return None, {}
        raise


def _issue_url(
    *,
    base_url: str,
    owner: str,
    repo: str,
    issue_index: Any,
    payload: dict[str, Any] | None = None,
) -> str:
    if isinstance(payload, dict):
        html_url = payload.get("html_url")
        if html_url:
            return redact_url(str(html_url))
    return (
        f"{base_url.rstrip('/')}/{urllib.parse.quote(owner, safe='')}/"
        f"{urllib.parse.quote(repo, safe='')}/issues/"
        f"{urllib.parse.quote(str(issue_index), safe='')}"
    )


def _lifecycle_state(
    *,
    issue_exists: bool,
    title_matches: bool,
    issue_index_matches: bool,
    is_pull_request: bool,
    body_marker_present: bool,
    labels_ok: bool,
    milestone_absent: bool,
    assignees_absent: bool,
    state: str | None,
) -> str:
    if not issue_exists:
        return "missing"
    if not (
        title_matches
        and issue_index_matches
        and not is_pull_request
        and body_marker_present
        and labels_ok
        and milestone_absent
        and assignees_absent
    ):
        return "mismatch"
    if state == "open":
        return "open_pm_seed_issue"
    if state == "closed":
        return "closed_pm_seed_issue"
    return "unknown"


def summarize_seed_issue_from_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
) -> dict[str, Any]:
    issues = (
        snapshot.get("issues")
        if isinstance(snapshot, dict) and isinstance(snapshot.get("issues"), dict)
        else {}
    )
    candidates: list[dict[str, Any]] = []
    for key in ("open", "recently_closed", "closed", "all"):
        for item in issues.get(key) or []:
            if isinstance(item, dict):
                candidates.append(item)
    matches = [
        item
        for item in candidates
        if item.get("title") == expected_title
        and int(item.get("number") or -1) == issue_index
    ]
    title_matches = [
        item for item in candidates if item.get("title") == expected_title
    ]
    match = matches[0] if matches else None
    issue_url = None
    if isinstance(match, dict):
        issue_url = match.get("html_url")
    owner = snapshot.get("owner") if isinstance(snapshot, dict) else DEFAULT_OWNER
    repo = snapshot.get("repo") if isinstance(snapshot, dict) else DEFAULT_REPO
    base_url = (
        snapshot.get("gitea_base_url")
        if isinstance(snapshot, dict)
        else DEFAULT_GITEA_BASE_URL
    )
    if match and not issue_url:
        issue_url = _issue_url(
            base_url=str(base_url or DEFAULT_GITEA_BASE_URL),
            owner=str(owner or DEFAULT_OWNER),
            repo=str(repo or DEFAULT_REPO),
            issue_index=issue_index,
            payload=match,
        )
    exists = match is not None
    state = str(match.get("state") or "unknown") if match else None
    return {
        "issue_index": issue_index,
        "issue_url": issue_url,
        "title": expected_title,
        "exists": exists,
        "state": state,
        "lifecycle_state": (
            "open_pm_seed_issue"
            if exists and state == "open"
            else "closed_pm_seed_issue"
            if exists and state == "closed"
            else "missing"
        ),
        "duplicate_seed_issue_blocker": len(title_matches) > 1,
        "matching_issue_count": len(title_matches),
        "source": "gitea_snapshot",
    }


def capture_issue_lifecycle_status(
    *,
    base_url: str = DEFAULT_GITEA_BASE_URL,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    issue_index: int = DEFAULT_ISSUE_INDEX,
    expected_title: str = EXPECTED_PM_SEED_ISSUE_TITLE,
    timeout: int = 20,
    env: dict[str, str] | None = None,
    transport: Transport | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    token, token_env_name = load_read_token(env=env)
    client = IssueLifecycleReadClient(
        base_url=base_url,
        token=token,
        timeout=timeout,
        transport=transport,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    clean_base_url = base_url.rstrip("/")

    issue_status, issue_payload = _safe_get(
        client,
        issue_api_path(owner, repo, issue_index),
        warnings,
    )
    comments_status, comments_payload = _safe_get(
        client,
        f"{issue_api_path(owner, repo, issue_index)}/comments",
        warnings,
        params={"limit": 20},
        optional=True,
    )
    projects_status, projects_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/projects"),
        warnings,
        params={"limit": 20},
        optional=True,
    )
    workflows_status, workflows_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/actions/runs"),
        warnings,
        params={"limit": 20},
        optional=True,
    )

    issue_exists = issue_status == 200 and isinstance(issue_payload, dict)
    issue: dict[str, Any] = issue_payload if issue_exists else {}
    actual_index = issue.get("number")
    title = str(issue.get("title") or "") if issue_exists else ""
    state = str(issue.get("state") or "unknown") if issue_exists else None
    body = str(issue.get("body") or "") if issue_exists else ""
    labels = _label_names(issue.get("labels") if issue_exists else [])
    unexpected_labels = sorted(
        label for label in labels if label.lower() not in EXPECTED_SAFE_LABELS
    )
    assignees = _assignee_names(issue.get("assignees") if issue_exists else [])
    milestone = _milestone_title(issue.get("milestone")) if issue_exists else None
    is_pull_request = bool(issue_exists and _is_pull_request_issue(issue))
    body_marker = issue_exists and _body_marker_present(body)
    issue_index_matches = issue_exists and int(actual_index or -1) == issue_index
    title_matches = issue_exists and title == expected_title
    labels_ok = not unexpected_labels
    milestone_absent = milestone is None
    assignees_absent = not assignees
    comments = _comments_summary(comments_status, comments_payload)
    projects = _project_summary(projects_status, projects_payload)
    workflows = _workflow_summary(workflows_status, workflows_payload)
    runner = _runner_summary()

    if not issue_exists:
        blockers.append(f"Issue #{issue_index} was not readable by GET.")
    if issue_exists and not issue_index_matches:
        blockers.append(f"Issue index did not match expected #{issue_index}.")
    if issue_exists and not title_matches:
        blockers.append("Issue title does not match the expected PM seed issue.")
    if is_pull_request:
        blockers.append("Issue endpoint returned pull request metadata.")
    if issue_exists and not body_marker:
        blockers.append("Issue body is missing the PM-6B or known PM seed marker.")
    if unexpected_labels:
        blockers.append(
            "Issue has unexpected labels: " + ", ".join(unexpected_labels)
        )
    elif labels:
        warnings.append("Issue has safe labels attached; PM-6B originally used none.")
    if milestone is not None:
        blockers.append("Issue has a milestone; PM seed issue should not.")
    if assignees:
        warnings.append(
            "Issue has assignees; verify whether Gitea auto-populated them."
        )
    if comments.get("state") == "present":
        warnings.append("Issue comments are present; PM-7 did not mutate them.")
    if projects.get("state") in {"available"} and projects.get("project_count"):
        warnings.append("Project endpoint returned projects; PM-7 did not mutate them.")

    lifecycle_state = _lifecycle_state(
        issue_exists=issue_exists,
        title_matches=title_matches,
        issue_index_matches=issue_index_matches,
        is_pull_request=is_pull_request,
        body_marker_present=body_marker,
        labels_ok=labels_ok,
        milestone_absent=milestone_absent,
        assignees_absent=assignees_absent,
        state=state,
    )
    matches_expected = lifecycle_state in {
        "open_pm_seed_issue",
        "closed_pm_seed_issue",
    }
    methods = sorted(set(client.methods_used))
    if any(method not in {"GET", "HEAD"} for method in methods):
        blockers.append("A non-read Gitea HTTP method was observed.")

    return {
        "schema_version": ISSUE_LIFECYCLE_SCHEMA_VERSION,
        "observed_at": created_at or utc_now(),
        "gitea_base_url": redact_url(clean_base_url, token),
        "owner": owner,
        "repo": repo,
        "auth_used": bool(token),
        "auth_env_var_used": token_env_name if token else None,
        "token_value_exposed": False,
        "http_methods_used": methods,
        "issue_index": issue_index,
        "issue_url": _issue_url(
            base_url=clean_base_url,
            owner=owner,
            repo=repo,
            issue_index=issue_index,
            payload=issue,
        ),
        "title": title or expected_title,
        "expected_title": expected_title,
        "state": state,
        "created_at": issue.get("created_at") if issue_exists else None,
        "updated_at": issue.get("updated_at") if issue_exists else None,
        "labels": labels,
        "unexpected_labels": unexpected_labels,
        "milestone": milestone,
        "assignees": assignees,
        "is_pull_request": is_pull_request,
        "body_marker_present": bool(body_marker),
        "matches_expected_pm_issue": bool(matches_expected),
        "comments_summary": comments,
        "project_summary": projects,
        "workflow_summary": workflows,
        "runner_summary": runner,
        "lifecycle_state": lifecycle_state,
        "no_mutation_observed": not blockers
        or all("non-read Gitea HTTP method" not in item for item in blockers),
        "blockers": [redact_text(item, token) for item in blockers],
        "warnings": [redact_text(item, token) for item in warnings],
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def compact_lifecycle_summary(status: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(status, dict):
        return {
            "issue_index": DEFAULT_ISSUE_INDEX,
            "issue_url": None,
            "title": EXPECTED_PM_SEED_ISSUE_TITLE,
            "exists": False,
            "state": None,
            "lifecycle_state": "unknown",
            "duplicate_seed_issue_blocker": False,
            "matching_issue_count": 0,
            "source": "unknown",
        }
    return {
        "issue_index": status.get("issue_index") or DEFAULT_ISSUE_INDEX,
        "issue_url": status.get("issue_url"),
        "title": status.get("title") or status.get("expected_title"),
        "exists": status.get("lifecycle_state")
        in {"open_pm_seed_issue", "closed_pm_seed_issue"},
        "state": status.get("state"),
        "lifecycle_state": status.get("lifecycle_state") or "unknown",
        "duplicate_seed_issue_blocker": False,
        "matching_issue_count": 1
        if status.get("lifecycle_state")
        in {"open_pm_seed_issue", "closed_pm_seed_issue"}
        else 0,
        "source": "issue_lifecycle_status",
        "matches_expected_pm_issue": bool(status.get("matches_expected_pm_issue")),
    }


def format_issue_lifecycle_text(status: dict[str, Any]) -> str:
    comments = (
        status.get("comments_summary")
        if isinstance(status.get("comments_summary"), dict)
        else {}
    )
    projects = (
        status.get("project_summary")
        if isinstance(status.get("project_summary"), dict)
        else {}
    )
    workflows = (
        status.get("workflow_summary")
        if isinstance(status.get("workflow_summary"), dict)
        else {}
    )
    blockers = status.get("blockers") or []
    warnings = status.get("warnings") or []
    workflow_count = workflows.get("recent_run_count")
    lines = [
        "Hermes PM issue lifecycle status",
        f"Issue: #{status.get('issue_index') or '<unknown>'}",
        f"Title matched: {'yes' if status.get('matches_expected_pm_issue') else 'no'}",
        f"State: {status.get('lifecycle_state') or 'unknown'}",
        f"Comments: {comments.get('state') or 'unknown'}",
        f"Projects: {projects.get('state') or 'unknown'}",
        f"Workflow runs: {workflow_count if workflow_count is not None else 'unknown'}",
        "Gitea writes: no",
        "Workflow/runner/runtime/financial/secret actions: no",
    ]
    if blockers:
        lines.append(f"Blockers: {len(blockers)}")
        for blocker in blockers[:4]:
            lines.append(f"- {blocker}")
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")
        for warning in warnings[:3]:
            lines.append(f"- {warning}")
    return redact_text("\n".join(lines))


def _assert_non_action_schema() -> None:
    true_values = [key for key, value in NON_ACTION_BOOLEANS.items() if value]
    if true_values:  # pragma: no cover - import-time guard
        joined = ", ".join(true_values)
        raise RuntimeError(
            "Issue lifecycle non-action booleans must be false: "
            f"{joined}"
        )


_assert_non_action_schema()
