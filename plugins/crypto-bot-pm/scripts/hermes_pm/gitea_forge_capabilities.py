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
        MUTATING_HTTP_METHODS,
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
        MUTATING_HTTP_METHODS,
        load_read_token,
        redact_text,
        redact_url,
        repo_api_path,
    )


FORGE_CAPABILITIES_SCHEMA_VERSION = "hermes.pm.gitea_forge_capabilities.v1"
ALLOWED_HTTP_METHODS = ("GET", "HEAD")

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "creates_issues": False,
    "creates_prs": False,
    "comments": False,
    "creates_labels": False,
    "mutates_projects": False,
    "starts_workflows": False,
    "starts_runners": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
}

Transport = Callable[[urllib.request.Request, int], tuple[int, bytes]]


class ForgeCapabilityProbeError(RuntimeError):
    """Raised when the read-only forge capability client is misused."""


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


class ForgeCapabilityReadOnlyClient:
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
    ) -> tuple[dict[str, Any], Any]:
        method = method.upper()
        if method not in ALLOWED_HTTP_METHODS:
            raise ForgeCapabilityProbeError(
                "forge capability client permits GET/HEAD only; "
                f"blocked {method}"
            )
        url = self._build_url(path, params=params)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        request = urllib.request.Request(url, headers=headers, method=method)
        self.methods_used.append(method)
        try:
            status, body = self._transport(request, self.timeout)
        except urllib.error.URLError as exc:
            probe = self._probe_entry(
                name="",
                method=method,
                path=path,
                status_code=None,
                error=f"connection error: {exc.reason}",
            )
            return probe, None
        except OSError as exc:
            probe = self._probe_entry(
                name="",
                method=method,
                path=path,
                status_code=None,
                error=f"transport error: {exc}",
            )
            return probe, None
        payload: Any = None
        error = ""
        if method != "HEAD" and body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                error = "non-JSON response"
        probe = self._probe_entry(
            name="",
            method=method,
            path=path,
            status_code=status,
            error=error,
        )
        return probe, payload

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        return self.request("GET", path, params=params)

    def _probe_entry(
        self,
        *,
        name: str,
        method: str,
        path: str,
        status_code: int | None,
        error: str = "",
    ) -> dict[str, Any]:
        available = bool(status_code is not None and 200 <= status_code < 300)
        return {
            "name": name,
            "method": method,
            "endpoint": path,
            "status_code": status_code,
            "available": available,
            "error": redact_text(error, self._token) if error else "",
            "called_write_api": False,
        }

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
        return redact_url(url, self._token)


def _status_to_support(probe: dict[str, Any]) -> bool | str:
    status_code = probe.get("status_code")
    if isinstance(status_code, int) and 200 <= status_code < 300:
        return True
    if status_code == 404:
        return False
    return "unknown"


def _first_issue_number(payload: Any) -> Any:
    if not isinstance(payload, list):
        return None
    for item in payload:
        if isinstance(item, dict) and "pull_request" not in item:
            number = item.get("number")
            if number is not None:
                return number
    return None


def _first_commit_sha(payload: Any) -> str | None:
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    if not isinstance(first, dict):
        return None
    value = first.get("sha") or first.get("id")
    return str(value) if value else None


def _named_probe(
    client: ForgeCapabilityReadOnlyClient,
    *,
    name: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    probe, payload = client.get(path, params=params)
    probe["name"] = name
    return probe, payload


def build_gitea_forge_capabilities(
    *,
    client: ForgeCapabilityReadOnlyClient,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    auth_used: bool = False,
    token_env_name: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    if not auth_used:
        warnings.append(
            {
                "code": "token_absent",
                "message": "Unauthenticated read-only endpoint probes were used.",
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

    repo_probe, _repo_payload = _named_probe(
        client,
        name="repo_metadata",
        path=repo_api_path(owner, repo),
    )
    probes.append(repo_probe)

    issues_probe, issues_payload = _named_probe(
        client,
        name="issues_list",
        path=repo_api_path(owner, repo, "/issues"),
        params={"state": "open", "limit": limit},
    )
    probes.append(issues_probe)

    labels_probe, _labels_payload = _named_probe(
        client,
        name="labels_list",
        path=repo_api_path(owner, repo, "/labels"),
        params={"limit": limit},
    )
    probes.append(labels_probe)

    milestones_probe, _milestones_payload = _named_probe(
        client,
        name="milestones_list",
        path=repo_api_path(owner, repo, "/milestones"),
        params={"state": "all", "limit": limit},
    )
    probes.append(milestones_probe)

    projects_probe, _projects_payload = _named_probe(
        client,
        name="projects_list",
        path=repo_api_path(owner, repo, "/projects"),
        params={"limit": limit},
    )
    probes.append(projects_probe)

    pulls_probe, _pulls_payload = _named_probe(
        client,
        name="pull_requests_list",
        path=repo_api_path(owner, repo, "/pulls"),
        params={"state": "open", "limit": limit},
    )
    probes.append(pulls_probe)

    issue_number = _first_issue_number(issues_payload)
    comments_support: bool | str = "unknown"
    if issue_number is not None:
        comments_probe, _comments_payload = _named_probe(
            client,
            name="issue_comments_list",
            path=repo_api_path(owner, repo, f"/issues/{issue_number}/comments"),
            params={"limit": min(limit, 10)},
        )
        probes.append(comments_probe)
        comments_support = _status_to_support(comments_probe)
    else:
        warnings.append(
            {
                "code": "comments_probe_skipped",
                "message": (
                    "No open issue number was available for read-only "
                    "comments probing."
                ),
            }
        )

    commits_probe, commits_payload = _named_probe(
        client,
        name="commits_list",
        path=repo_api_path(owner, repo, "/commits"),
        params={"limit": 1},
    )
    probes.append(commits_probe)
    status_support: bool | str = "unknown"
    commit_sha = _first_commit_sha(commits_payload)
    if commit_sha:
        statuses_probe, _statuses_payload = _named_probe(
            client,
            name="statuses_read",
            path=repo_api_path(
                owner,
                repo,
                f"/statuses/{urllib.parse.quote(commit_sha, safe='')}",
            ),
            params={"limit": min(limit, 10)},
        )
        probes.append(statuses_probe)
        combined_probe, _combined_payload = _named_probe(
            client,
            name="combined_status_read",
            path=repo_api_path(
                owner,
                repo,
                f"/commits/{urllib.parse.quote(commit_sha, safe='')}/status",
            ),
        )
        probes.append(combined_probe)
        if _status_to_support(statuses_probe) is True or _status_to_support(
            combined_probe
        ) is True:
            status_support = True
        elif statuses_probe.get("status_code") == 404 and combined_probe.get(
            "status_code"
        ) == 404:
            status_support = False
    else:
        warnings.append(
            {
                "code": "status_probe_skipped",
                "message": "No commit SHA was available for status endpoint probing.",
            }
        )

    for probe in probes:
        status_code = probe.get("status_code")
        if status_code not in (None, 200, 201, 202, 204):
            blockers.append(
                {
                    "endpoint": str(probe.get("endpoint") or ""),
                    "status_code": str(status_code),
                    "name": str(probe.get("name") or ""),
                }
            )

    return {
        "schema_version": FORGE_CAPABILITIES_SCHEMA_VERSION,
        "created_at": utc_now(),
        "gitea_base_url": redact_url(client.base_url),
        "owner": owner,
        "repo": repo,
        "auth_used": bool(auth_used),
        "token_env_var_used": token_env_name if auth_used else None,
        "token_value_exposed": False,
        "read_only": True,
        "permission_proof": False,
        "endpoint_shape_evidence_only": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "http_methods_used": sorted(set(client.methods_used)),
        "endpoint_probes": probes,
        "repo_metadata_supported": _status_to_support(repo_probe),
        "issues_write_preview_supported": _status_to_support(issues_probe),
        "labels_write_preview_supported": _status_to_support(labels_probe),
        "milestones_read_supported": _status_to_support(milestones_probe),
        "projects_write_preview_supported": _status_to_support(projects_probe),
        "comments_write_preview_supported": comments_support,
        "pull_requests_read_supported": _status_to_support(pulls_probe),
        "statuses_read_supported": status_support,
        "blockers": blockers,
        "warnings": warnings,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def capture_gitea_forge_capabilities(
    *,
    base_url: str = DEFAULT_GITEA_BASE_URL,
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
    timeout: int = 20,
    no_token: bool = False,
    limit: int = 20,
    transport: Transport | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    token, token_env_name = load_read_token(env=env, no_token=no_token)
    client = ForgeCapabilityReadOnlyClient(
        base_url=base_url,
        token=token,
        timeout=timeout,
        transport=transport,
    )
    return build_gitea_forge_capabilities(
        client=client,
        owner=owner,
        repo=repo,
        auth_used=bool(token),
        token_env_name=token_env_name,
        limit=limit,
    )
