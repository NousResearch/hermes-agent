#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


SNAPSHOT_SCHEMA_VERSION = "hermes.pm.gitea_readonly_snapshot.v1"
DEFAULT_GITEA_BASE_URL = "http://127.0.0.1:3005"
DEFAULT_OWNER = "preston"
DEFAULT_REPO = "crypto_bot"
TOKEN_ENV_NAMES = ("GITEA_READ_TOKEN", "GITEA_TOKEN", "TEA_TOKEN")
ALLOWED_HTTP_METHODS = ("GET", "HEAD")
MUTATING_HTTP_METHODS = ("POST", "PUT", "PATCH", "DELETE")

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "creates_issues": False,
    "creates_prs": False,
    "comments": False,
    "edits_labels": False,
    "starts_workflows": False,
    "writes_files": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}


class ReadOnlyGiteaError(RuntimeError):
    """Raised for redacted read-only Gitea failures."""


Transport = Callable[[urllib.request.Request, int], tuple[int, bytes]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def redact_text(text: str, token: str | None = None) -> str:
    redacted = text
    if token:
        redacted = redacted.replace(token, "<redacted-token>")
    redacted = re.sub(
        r"(?i)(token\s+)[A-Za-z0-9._~+/=-]+",
        r"\1<redacted-token>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(authorization:\s*)[A-Za-z0-9._~+/=-]+",
        r"\1<redacted-token>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(access_token=)[^&\s]+",
        r"\1<redacted-token>",
        redacted,
    )
    return redacted


def redact_url(url: str, token: str | None = None) -> str:
    redacted = redact_text(url, token)
    parsed = urllib.parse.urlsplit(redacted)
    if parsed.username or parsed.password:
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        redacted = urllib.parse.urlunsplit(
            (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
        )
    return redacted


def default_transport(
    request: urllib.request.Request,
    timeout: int,
) -> tuple[int, bytes]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def load_read_token(
    *,
    env: dict[str, str] | None = None,
    no_token: bool = False,
) -> tuple[str | None, str | None]:
    if no_token:
        return None, None
    source = env if env is not None else os.environ
    for name in TOKEN_ENV_NAMES:
        value = source.get(name)
        if value:
            return value, name
    return None, None


class ReadOnlyGiteaClient:
    mutation_capability = False
    allowed_methods = ALLOWED_HTTP_METHODS
    blocked_methods = MUTATING_HTTP_METHODS

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

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        method = method.upper()
        if method not in ALLOWED_HTTP_METHODS:
            raise ReadOnlyGiteaError(
                "read-only Gitea client permits GET/HEAD only; "
                f"blocked {method}"
            )
        return self._request(method, path, params=params)

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    def head(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self.request("HEAD", path, params=params)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = self._build_url(path, params=params)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        request = urllib.request.Request(url, headers=headers, method=method)
        self.methods_used.append(method)
        try:
            _status, body = self._transport(request, self.timeout)
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            message = (
                f"HTTP {exc.code} from {redact_url(url, self._token)}: {details}"
            )
            raise ReadOnlyGiteaError(redact_text(message, self._token)) from exc
        except urllib.error.URLError as exc:
            message = (
                "connection error from "
                f"{redact_url(url, self._token)}: {exc.reason}"
            )
            raise ReadOnlyGiteaError(redact_text(message, self._token)) from exc
        except OSError as exc:
            message = f"transport error from {redact_url(url, self._token)}: {exc}"
            raise ReadOnlyGiteaError(redact_text(message, self._token)) from exc
        if method == "HEAD" or not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ReadOnlyGiteaError(
                f"non-JSON response from {redact_url(url, self._token)}"
            ) from exc

    def _build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        clean_path = path if path.startswith("/") else f"/{path}"
        query = urllib.parse.urlencode(params or {})
        url = f"{self.base_url}{clean_path}"
        if query:
            url = f"{url}?{query}"
        return url


def repo_api_path(owner: str, repo: str, suffix: str = "") -> str:
    quoted_owner = urllib.parse.quote(owner, safe="")
    quoted_repo = urllib.parse.quote(repo, safe="")
    return f"/api/v1/repos/{quoted_owner}/{quoted_repo}{suffix}"


def _safe_get(
    client: ReadOnlyGiteaClient,
    path: str,
    blockers: list[dict[str, str]],
    *,
    params: dict[str, Any] | None = None,
    optional: bool = False,
) -> Any:
    try:
        return client.get(path, params=params)
    except ReadOnlyGiteaError as exc:
        entry = {"endpoint": path, "error": str(exc)}
        if optional:
            entry["optional"] = "true"
        blockers.append(entry)
        return None


def _list_payload(payload: Any, key: str | None = None) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if key and isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    return []


def _safe_user_name(payload: dict[str, Any]) -> str | None:
    user = payload.get("user")
    if isinstance(user, dict):
        value = user.get("login") or user.get("username") or user.get("full_name")
        return str(value) if value else None
    return None


def _label_names(payload: dict[str, Any]) -> list[str]:
    labels = payload.get("labels") or []
    names = [label.get("name") for label in labels if isinstance(label, dict)]
    return sorted(str(name) for name in names if name)


def summarize_repository(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "full_name": payload.get("full_name"),
        "html_url": payload.get("html_url"),
        "clone_url": redact_url(str(payload.get("clone_url") or "")),
        "default_branch": payload.get("default_branch"),
        "private": payload.get("private"),
        "empty": payload.get("empty"),
        "archived": payload.get("archived"),
        "open_issues_count": payload.get("open_issues_count"),
        "stars_count": payload.get("stars_count"),
        "forks_count": payload.get("forks_count"),
        "updated_at": payload.get("updated_at"),
    }


def summarize_issue(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": payload.get("number"),
        "title": payload.get("title"),
        "state": payload.get("state"),
        "html_url": payload.get("html_url"),
        "user": _safe_user_name(payload),
        "labels": _label_names(payload),
        "milestone": _milestone_title(payload.get("milestone")),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "closed_at": payload.get("closed_at"),
    }


def _is_pull_request_issue(payload: dict[str, Any]) -> bool:
    pull_request = payload.get("pull_request")
    return isinstance(pull_request, dict) and bool(pull_request)


def summarize_pull_request(payload: dict[str, Any]) -> dict[str, Any]:
    head = payload.get("head") if isinstance(payload.get("head"), dict) else {}
    base = payload.get("base") if isinstance(payload.get("base"), dict) else {}
    merged = payload.get("merged")
    merged_at = payload.get("merged_at")
    return {
        "number": payload.get("number"),
        "title": payload.get("title"),
        "state": payload.get("state"),
        "html_url": payload.get("html_url"),
        "user": _safe_user_name(payload),
        "head": head.get("ref"),
        "base": base.get("ref"),
        "merged": merged,
        "merged_at": merged_at,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "closed_at": payload.get("closed_at"),
    }


def _milestone_title(payload: Any) -> str | None:
    if isinstance(payload, dict):
        title = payload.get("title")
        return str(title) if title else None
    return None


def summarize_label(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "name": payload.get("name"),
        "color": payload.get("color"),
        "description": payload.get("description"),
    }


def summarize_milestone(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "title": payload.get("title"),
        "state": payload.get("state"),
        "open_issues": payload.get("open_issues"),
        "closed_issues": payload.get("closed_issues"),
        "due_on": payload.get("due_on"),
        "updated_at": payload.get("updated_at"),
    }


def summarize_commit(payload: dict[str, Any]) -> dict[str, Any]:
    commit = payload.get("commit") if isinstance(payload.get("commit"), dict) else {}
    author = commit.get("author") if isinstance(commit.get("author"), dict) else {}
    return {
        "sha": payload.get("sha") or payload.get("id"),
        "html_url": payload.get("html_url"),
        "message": str(commit.get("message") or "").splitlines()[0][:160],
        "author_name": author.get("name"),
        "author_date": author.get("date"),
    }


def summarize_status(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": payload.get("context"),
        "state": payload.get("state"),
        "target_url": payload.get("target_url"),
        "description": payload.get("description"),
        "updated_at": payload.get("updated_at"),
    }


def summarize_workflow_run(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "name": payload.get("name") or payload.get("workflow_name"),
        "status": payload.get("status"),
        "conclusion": payload.get("conclusion"),
        "event": payload.get("event"),
        "head_branch": payload.get("head_branch"),
        "head_sha": payload.get("head_sha"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def summarize_project(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "title": payload.get("title") or payload.get("name"),
        "state": payload.get("state"),
        "html_url": payload.get("html_url"),
        "updated_at": payload.get("updated_at"),
    }


def _summarize_items(
    payload: Any,
    summarizer: Callable[[dict[str, Any]], Any],
) -> list:
    return [
        summarizer(item)
        for item in _list_payload(payload)
        if isinstance(item, dict)
    ]


def build_snapshot(
    *,
    client: ReadOnlyGiteaClient,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    auth_used: bool = False,
    token_env_name: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not auth_used:
        warnings.append(
            {
                "code": "token_absent",
                "message": (
                    "No read token env var was used; attempted unauthenticated "
                    "read-only requests."
                ),
            }
        )
    elif token_env_name:
        warnings.append(
            {
                "code": "token_env_used",
                "message": (
                    f"Read token env var {token_env_name} was used; "
                    "value redacted."
                ),
            }
        )

    repo_payload = _safe_get(client, repo_api_path(owner, repo), blockers)
    repository = summarize_repository(repo_payload)
    default_branch = repository.get("default_branch")

    branch: dict[str, Any] = {}
    branch_sha = None
    if default_branch:
        branch_payload = _safe_get(
            client,
            repo_api_path(
                owner,
                repo,
                f"/branches/{urllib.parse.quote(str(default_branch), safe='')}",
            ),
            blockers,
            optional=True,
        )
        if isinstance(branch_payload, dict):
            commit = branch_payload.get("commit")
            branch_sha = commit.get("id") if isinstance(commit, dict) else None
            branch = {"name": branch_payload.get("name"), "commit_sha": branch_sha}

    open_issues_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/issues"),
        blockers,
        params={"state": "open", "limit": limit},
    )
    closed_issues_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/issues"),
        blockers,
        params={"state": "closed", "limit": limit},
        optional=True,
    )
    open_issues = [
        summarize_issue(item)
        for item in _list_payload(open_issues_payload)
        if isinstance(item, dict) and not _is_pull_request_issue(item)
    ]
    closed_issues = [
        summarize_issue(item)
        for item in _list_payload(closed_issues_payload)
        if isinstance(item, dict) and not _is_pull_request_issue(item)
    ]

    open_pulls_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/pulls"),
        blockers,
        params={"state": "open", "limit": limit},
    )
    closed_pulls_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/pulls"),
        blockers,
        params={"state": "closed", "limit": limit},
        optional=True,
    )
    open_pulls = _summarize_items(open_pulls_payload, summarize_pull_request)
    closed_pulls = _summarize_items(closed_pulls_payload, summarize_pull_request)

    label_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/labels"),
        blockers,
        params={"limit": limit},
        optional=True,
    )
    milestone_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/milestones"),
        blockers,
        params={"state": "all", "limit": limit},
        optional=True,
    )
    commits_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/commits"),
        blockers,
        params={"limit": min(limit, 20)},
        optional=True,
    )
    commits = _summarize_items(commits_payload, summarize_commit)

    status_sha = branch_sha
    if status_sha is None and commits:
        status_sha = commits[0].get("sha")
    statuses: list[dict[str, Any]] = []
    combined_status: dict[str, Any] = {}
    if status_sha:
        statuses_payload = _safe_get(
            client,
            repo_api_path(
                owner,
                repo,
                f"/statuses/{urllib.parse.quote(str(status_sha), safe='')}",
            ),
            blockers,
            params={"limit": limit},
            optional=True,
        )
        statuses = _summarize_items(statuses_payload, summarize_status)
        combined_payload = _safe_get(
            client,
            repo_api_path(
                owner,
                repo,
                f"/commits/{urllib.parse.quote(str(status_sha), safe='')}/status",
            ),
            blockers,
            optional=True,
        )
        if isinstance(combined_payload, dict):
            combined_status = {
                "sha": combined_payload.get("sha"),
                "state": combined_payload.get("state"),
                "total_count": combined_payload.get("total_count"),
            }

    workflow_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/actions/runs"),
        blockers,
        params={"limit": min(limit, 20)},
        optional=True,
    )
    workflow_runs = [
        summarize_workflow_run(item)
        for item in _list_payload(workflow_payload, key="workflow_runs")
        if isinstance(item, dict)
    ]
    project_payload = _safe_get(
        client,
        repo_api_path(owner, repo, "/projects"),
        blockers,
        params={"limit": limit},
        optional=True,
    )

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "gitea_base_url": redact_url(client.base_url),
        "owner": owner,
        "repo": repo,
        "auth_used": bool(auth_used),
        "token_env_var_used": token_env_name if auth_used else None,
        "token_value_exposed": False,
        "http_methods_used": sorted(set(client.methods_used)),
        "repository": {
            **repository,
            "default_branch_detail": branch,
            "recent_commits": commits,
        },
        "issues": {
            "open": open_issues,
            "recently_closed": closed_issues,
            "open_count": len(open_issues),
            "recently_closed_count": len(closed_issues),
        },
        "pull_requests": {
            "open": open_pulls,
            "recently_closed_or_merged": closed_pulls,
            "open_count": len(open_pulls),
            "recently_closed_or_merged_count": len(closed_pulls),
        },
        "labels": _summarize_items(label_payload, summarize_label),
        "milestones": _summarize_items(milestone_payload, summarize_milestone),
        "checks": {
            "target_sha": status_sha,
            "statuses": statuses,
            "combined_status": combined_status,
        },
        "workflows": {
            "recent_runs": workflow_runs,
            "recent_run_count": len(workflow_runs),
        },
        "projects": _summarize_items(project_payload, summarize_project),
        "blockers": blockers,
        "warnings": warnings,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def capture_gitea_snapshot(
    *,
    base_url: str = DEFAULT_GITEA_BASE_URL,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    timeout: int = 20,
    no_token: bool = False,
    limit: int = 50,
    transport: Transport | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    token, token_env_name = load_read_token(env=env, no_token=no_token)
    client = ReadOnlyGiteaClient(
        base_url=base_url,
        token=token,
        timeout=timeout,
        transport=transport,
    )
    return build_snapshot(
        client=client,
        owner=owner,
        repo=repo,
        auth_used=bool(token),
        token_env_name=token_env_name,
        limit=limit,
    )
