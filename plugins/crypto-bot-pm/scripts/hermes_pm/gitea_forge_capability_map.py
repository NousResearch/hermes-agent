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


GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION = (
    "hermes.pm.gitea_forge_capability_map.v1"
)
ALLOWED_HTTP_METHODS = ("GET", "HEAD", "OPTIONS")

READY_CLASSIFICATIONS = {
    "endpoint_ready",
    "read_endpoint_ready_write_unproven",
}

PROJECT_OPERATION_TYPES = {"create_project_column", "create_project_card"}

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "mutation_executed": False,
    "creates_issues": False,
    "creates_labels": False,
    "mutates_projects": False,
    "comments": False,
    "starts_workflows": False,
    "starts_runners": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "branch_writer_invoked": False,
}

Transport = Callable[[urllib.request.Request, int], tuple[int, bytes]]


class ForgeCapabilityMapError(RuntimeError):
    """Raised when the read-only capability map client is misused."""


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


class GiteaForgeCapabilityMapClient:
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
            raise ForgeCapabilityMapError(
                "forge capability map client permits GET/HEAD/OPTIONS only; "
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
                params=params,
                status_code=None,
                error=f"connection error: {exc.reason}",
            )
            return probe, None
        except OSError as exc:
            probe = self._probe_entry(
                name="",
                method=method,
                path=path,
                params=params,
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
            params=params,
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

    def head(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        return self.request("HEAD", path, params=params)

    def options(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        return self.request("OPTIONS", path, params=params)

    def _probe_entry(
        self,
        *,
        name: str,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        status_code: int | None,
        error: str = "",
    ) -> dict[str, Any]:
        available = bool(status_code is not None and 200 <= status_code < 300)
        return {
            "name": name,
            "method": method,
            "endpoint": path,
            "params": params or {},
            "status_code": status_code,
            "available": available,
            "auth_used": bool(self._token),
            "token_exposed": False,
            "evidence_strength": "direct_read" if available else "unavailable",
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


def _quote(value: Any) -> str:
    return urllib.parse.quote(str(value), safe="")


def _named_probe(
    client: GiteaForgeCapabilityMapClient,
    *,
    name: str,
    path: str,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    probe, payload = client.get(path, params=params)
    probe["name"] = name
    return probe, payload


def _status_to_classification(status_code: Any) -> str:
    if isinstance(status_code, int) and 200 <= status_code < 300:
        return "read_endpoint_ready_write_unproven"
    if status_code in {401, 403}:
        return "auth_required"
    if status_code == 404:
        return "endpoint_not_found"
    if status_code is None:
        return "endpoint_unknown"
    return "endpoint_unknown"


def _evidence_strength(status_code: Any) -> str:
    if isinstance(status_code, int) and 200 <= status_code < 300:
        return "direct_read"
    if status_code == 404:
        return "unavailable"
    return "not_sampled" if status_code is None else "unavailable"


def _status_support(probe: dict[str, Any]) -> bool | str:
    classification = _status_to_classification(probe.get("status_code"))
    if classification in READY_CLASSIFICATIONS:
        return True
    if classification == "endpoint_not_found":
        return False
    return "unknown"


def _list_payload(payload: Any, key: str | None = None) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if key and isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    if isinstance(payload, dict):
        for fallback_key in ("items", "data", "projects"):
            if isinstance(payload.get(fallback_key), list):
                return payload[fallback_key]
    return []


def _first_issue_number(*payloads: Any) -> Any:
    for payload in payloads:
        for item in _list_payload(payload):
            if isinstance(item, dict) and "pull_request" not in item:
                number = item.get("number")
                if number is not None:
                    return number
    return None


def _first_commit_sha(payload: Any) -> str | None:
    for item in _list_payload(payload):
        if isinstance(item, dict):
            value = item.get("sha") or item.get("id")
            if value:
                return str(value)
    return None


def _first_project_id(*payloads: Any) -> Any:
    for payload in payloads:
        for item in _list_payload(payload):
            if isinstance(item, dict):
                value = item.get("id")
                if value is not None:
                    return value
    return None


def _first_column_id(*payloads: Any) -> Any:
    for payload in payloads:
        for item in _list_payload(payload):
            if isinstance(item, dict):
                value = item.get("id")
                if value is not None:
                    return value
    return None


def _write_endpoint(owner: str, repo: str, operation_type: str) -> dict[str, str]:
    base = repo_api_path(owner, repo)
    if operation_type == "create_issue":
        return {"method": "POST", "endpoint": f"{base}/issues"}
    if operation_type == "create_label":
        return {"method": "POST", "endpoint": f"{base}/labels"}
    if operation_type == "update_issue":
        return {
            "method": "POST",
            "endpoint": f"{base}/issues/{{issue_index}}/labels",
        }
    if operation_type == "comment_on_issue":
        return {
            "method": "POST",
            "endpoint": f"{base}/issues/{{issue_index}}/comments",
        }
    if operation_type == "create_project_column":
        return {
            "method": "POST",
            "endpoint": f"{base}/projects/{{project_id}}/columns",
        }
    if operation_type == "create_project_card":
        return {
            "method": "POST",
            "endpoint": (
                f"{base}/projects/{{project_id}}/columns/{{column_id}}/cards"
            ),
        }
    if operation_type == "request_pr_review":
        return {
            "method": "POST",
            "endpoint": f"{base}/pulls/{{pull_index}}/reviews",
        }
    return {"method": "POST", "endpoint": "<unsupported>"}


def _operation_entry(
    *,
    operation_type: str,
    probe: dict[str, Any] | None,
    owner: str,
    repo: str,
    classification: str | None = None,
    evidence_strength: str | None = None,
    blockers: list[str] | None = None,
    warnings: list[str] | None = None,
    endpoint: str | None = None,
    method: str = "GET",
) -> dict[str, Any]:
    status_code = probe.get("status_code") if probe else None
    resolved_classification = classification or _status_to_classification(status_code)
    resolved_endpoint = endpoint or (str(probe.get("endpoint")) if probe else "")
    resolved_method = str(probe.get("method") if probe else method)
    resolved_evidence = evidence_strength or _evidence_strength(status_code)
    entry_blockers = list(blockers or [])
    entry_warnings = list(warnings or [])
    if resolved_classification == "auth_required":
        entry_blockers.append("Read endpoint requires authentication.")
    elif resolved_classification == "endpoint_not_found":
        entry_blockers.append("Read endpoint returned 404.")
    elif resolved_classification == "endpoint_unknown":
        entry_blockers.append("Read endpoint capability is unknown.")
    if resolved_classification in READY_CLASSIFICATIONS:
        entry_warnings.append("Write permission is not proven by read evidence.")
    return {
        "operation_type": operation_type,
        "classification": resolved_classification,
        "status_code": status_code,
        "endpoint": resolved_endpoint,
        "method": resolved_method,
        "auth_used": bool(probe.get("auth_used")) if probe else False,
        "token_exposed": False,
        "evidence_strength": resolved_evidence,
        "required_future_write_endpoint": _write_endpoint(
            owner,
            repo,
            operation_type,
        ),
        "blockers": list(dict.fromkeys(entry_blockers)),
        "warnings": list(dict.fromkeys(entry_warnings)),
    }


def _project_classification(
    *,
    project_probe: dict[str, Any],
    columns_probe: dict[str, Any] | None,
    cards_probe: dict[str, Any] | None,
    operation_type: str,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    project_status = project_probe.get("status_code")
    if project_status in {401, 403}:
        return "auth_required", project_probe, ["Project endpoint requires auth."]
    if project_status == 404:
        return (
            "endpoint_not_found",
            project_probe,
            "Repository project endpoint returned 404.".splitlines(),
        )
    if not isinstance(project_status, int) or not 200 <= project_status < 300:
        return "endpoint_unknown", project_probe, ["Project endpoint is unknown."]
    if operation_type == "create_project_column":
        if columns_probe is None:
            return (
                "endpoint_unknown",
                project_probe,
                ["No project id was available for read-only column sampling."],
            )
        return (
            _status_to_classification(columns_probe.get("status_code")),
            columns_probe,
            [],
        )
    if cards_probe is None:
        return (
            "endpoint_unknown",
            columns_probe or project_probe,
            ["No project column id was available for read-only card sampling."],
        )
    return _status_to_classification(cards_probe.get("status_code")), cards_probe, []


def build_gitea_forge_capability_map(
    *,
    client: GiteaForgeCapabilityMapClient,
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
                    f"Read token env var {token_env_name} was used; value redacted."
                ),
            }
        )

    repo_probe, _repo_payload = _named_probe(
        client,
        name="repo_metadata",
        path=repo_api_path(owner, repo),
    )
    probes.append(repo_probe)

    branches_probe, _branches_payload = _named_probe(
        client,
        name="repo_branches",
        path=repo_api_path(owner, repo, "/branches"),
        params={"limit": limit},
    )
    probes.append(branches_probe)

    commits_probe, commits_payload = _named_probe(
        client,
        name="repo_commits",
        path=repo_api_path(owner, repo, "/commits"),
        params={"limit": min(limit, 10)},
    )
    probes.append(commits_probe)

    issues_probe, issues_payload = _named_probe(
        client,
        name="issues_list",
        path=repo_api_path(owner, repo, "/issues"),
        params={"limit": limit},
    )
    probes.append(issues_probe)

    issues_open_probe, issues_open_payload = _named_probe(
        client,
        name="issues_open",
        path=repo_api_path(owner, repo, "/issues"),
        params={"state": "open", "limit": limit},
    )
    probes.append(issues_open_probe)

    issues_closed_probe, issues_closed_payload = _named_probe(
        client,
        name="issues_closed",
        path=repo_api_path(owner, repo, "/issues"),
        params={"state": "closed", "limit": limit},
    )
    probes.append(issues_closed_probe)

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

    pulls_probe, _pulls_payload = _named_probe(
        client,
        name="pull_requests_list",
        path=repo_api_path(owner, repo, "/pulls"),
        params={"state": "open", "limit": limit},
    )
    probes.append(pulls_probe)

    issue_number = _first_issue_number(
        issues_payload,
        issues_open_payload,
        issues_closed_payload,
    )
    comments_probe: dict[str, Any] | None = None
    if issue_number is not None:
        comments_probe, _comments_payload = _named_probe(
            client,
            name="issue_comments_list",
            path=repo_api_path(owner, repo, f"/issues/{_quote(issue_number)}/comments"),
            params={"limit": min(limit, 10)},
        )
        probes.append(comments_probe)
    else:
        warnings.append(
            {
                "code": "comments_probe_skipped",
                "message": (
                    "No issue index was available; comments capability is "
                    "not_sampled rather than probed with a fake issue."
                ),
            }
        )

    commit_sha = _first_commit_sha(commits_payload)
    statuses_probe: dict[str, Any] | None = None
    combined_status_probe: dict[str, Any] | None = None
    commit_statuses_probe: dict[str, Any] | None = None
    if commit_sha:
        quoted_sha = _quote(commit_sha)
        statuses_probe, _statuses_payload = _named_probe(
            client,
            name="statuses_read",
            path=repo_api_path(owner, repo, f"/statuses/{quoted_sha}"),
            params={"limit": min(limit, 10)},
        )
        probes.append(statuses_probe)
        combined_status_probe, _combined_payload = _named_probe(
            client,
            name="combined_status_read",
            path=repo_api_path(owner, repo, f"/commits/{quoted_sha}/status"),
        )
        probes.append(combined_status_probe)
        commit_statuses_probe, _commit_statuses_payload = _named_probe(
            client,
            name="commit_statuses_read",
            path=repo_api_path(owner, repo, f"/commits/{quoted_sha}/statuses"),
            params={"limit": min(limit, 10)},
        )
        probes.append(commit_statuses_probe)
    else:
        warnings.append(
            {
                "code": "status_probe_skipped",
                "message": "No commit SHA was available for status endpoint probing.",
            }
        )

    projects_probe, projects_payload = _named_probe(
        client,
        name="repo_projects_list",
        path=repo_api_path(owner, repo, "/projects"),
        params={"limit": limit},
    )
    probes.append(projects_probe)

    user_probe, _user_payload = _named_probe(
        client,
        name="owner_user_metadata",
        path=f"/api/v1/users/{_quote(owner)}",
    )
    probes.append(user_probe)

    user_projects_probe, user_projects_payload = _named_probe(
        client,
        name="owner_user_projects_list",
        path=f"/api/v1/users/{_quote(owner)}/projects",
        params={"limit": limit},
    )
    probes.append(user_projects_probe)

    org_probe, _org_payload = _named_probe(
        client,
        name="owner_org_metadata",
        path=f"/api/v1/orgs/{_quote(owner)}",
    )
    probes.append(org_probe)

    org_projects_probe, org_projects_payload = _named_probe(
        client,
        name="owner_org_projects_list",
        path=f"/api/v1/orgs/{_quote(owner)}/projects",
        params={"limit": limit},
    )
    probes.append(org_projects_probe)

    project_id = _first_project_id(
        projects_payload,
        user_projects_payload,
        org_projects_payload,
    )
    repo_project_detail_probe: dict[str, Any] | None = None
    repo_columns_probe: dict[str, Any] | None = None
    global_columns_probe: dict[str, Any] | None = None
    repo_cards_probe: dict[str, Any] | None = None
    global_cards_probe: dict[str, Any] | None = None
    if project_id is not None:
        quoted_project = _quote(project_id)
        repo_project_detail_probe, _project_detail_payload = _named_probe(
            client,
            name="repo_project_detail",
            path=repo_api_path(owner, repo, f"/projects/{quoted_project}"),
        )
        probes.append(repo_project_detail_probe)
        repo_columns_probe, repo_columns_payload = _named_probe(
            client,
            name="repo_project_columns_list",
            path=repo_api_path(owner, repo, f"/projects/{quoted_project}/columns"),
            params={"limit": limit},
        )
        probes.append(repo_columns_probe)
        global_columns_probe, global_columns_payload = _named_probe(
            client,
            name="global_project_columns_list",
            path=f"/api/v1/projects/{quoted_project}/columns",
            params={"limit": limit},
        )
        probes.append(global_columns_probe)
        column_id = _first_column_id(repo_columns_payload, global_columns_payload)
        if column_id is not None:
            quoted_column = _quote(column_id)
            repo_cards_probe, _repo_cards_payload = _named_probe(
                client,
                name="repo_project_cards_list",
                path=repo_api_path(
                    owner,
                    repo,
                    (
                        f"/projects/{quoted_project}/columns/"
                        f"{quoted_column}/cards"
                    ),
                ),
                params={"limit": limit},
            )
            probes.append(repo_cards_probe)
            global_cards_probe, _global_cards_payload = _named_probe(
                client,
                name="global_project_cards_list",
                path=f"/api/v1/projects/columns/{quoted_column}/cards",
                params={"limit": limit},
            )
            probes.append(global_cards_probe)
        else:
            warnings.append(
                {
                    "code": "project_cards_probe_skipped",
                    "message": (
                        "A project was discoverable, but no column id was "
                        "available for read-only card sampling."
                    ),
                }
            )
    else:
        warnings.append(
            {
                "code": "project_columns_probe_skipped",
                "message": (
                    "No project id was discoverable; project columns/cards "
                    "were not sampled with fake ids."
                ),
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

    status_probe_for_operation = next(
        (
            probe
            for probe in (statuses_probe, combined_status_probe, commit_statuses_probe)
            if probe and probe.get("available")
        ),
        statuses_probe or combined_status_probe or commit_statuses_probe,
    )
    columns_probe = next(
        (
            probe
            for probe in (repo_columns_probe, global_columns_probe)
            if probe and probe.get("available")
        ),
        repo_columns_probe or global_columns_probe,
    )
    cards_probe = next(
        (
            probe
            for probe in (repo_cards_probe, global_cards_probe)
            if probe and probe.get("available")
        ),
        repo_cards_probe or global_cards_probe,
    )
    project_column_class, project_column_probe, project_column_blockers = (
        _project_classification(
            project_probe=projects_probe,
            columns_probe=columns_probe,
            cards_probe=cards_probe,
            operation_type="create_project_column",
        )
    )
    project_card_class, project_card_probe, project_card_blockers = (
        _project_classification(
            project_probe=projects_probe,
            columns_probe=columns_probe,
            cards_probe=cards_probe,
            operation_type="create_project_card",
        )
    )

    comment_entry = (
        _operation_entry(
            operation_type="comment_on_issue",
            probe=comments_probe,
            owner=owner,
            repo=repo,
        )
        if comments_probe
        else _operation_entry(
            operation_type="comment_on_issue",
            probe=None,
            owner=owner,
            repo=repo,
            classification="endpoint_unknown",
            evidence_strength="not_sampled",
            endpoint=repo_api_path(owner, repo, "/issues/{issue_index}/comments"),
            blockers=["No issue index was available for comments sampling."],
        )
    )
    operation_capabilities = {
        "create_issue": _operation_entry(
            operation_type="create_issue",
            probe=issues_probe,
            owner=owner,
            repo=repo,
        ),
        "create_label": _operation_entry(
            operation_type="create_label",
            probe=labels_probe,
            owner=owner,
            repo=repo,
        ),
        "update_issue": _operation_entry(
            operation_type="update_issue",
            probe=issues_probe,
            owner=owner,
            repo=repo,
            warnings=["Issue label update also depends on future label write scope."],
        ),
        "comment_on_issue": comment_entry,
        "create_project_column": _operation_entry(
            operation_type="create_project_column",
            probe=project_column_probe,
            owner=owner,
            repo=repo,
            classification=project_column_class,
            blockers=project_column_blockers,
        ),
        "create_project_card": _operation_entry(
            operation_type="create_project_card",
            probe=project_card_probe,
            owner=owner,
            repo=repo,
            classification=project_card_class,
            blockers=project_card_blockers,
        ),
        "request_pr_review": _operation_entry(
            operation_type="request_pr_review",
            probe=pulls_probe,
            owner=owner,
            repo=repo,
            warnings=["Review request write shape is unproven by pull listing."],
        ),
    }

    endpoint_ready_operation_types = sorted(
        operation_type
        for operation_type, detail in operation_capabilities.items()
        if detail.get("classification") in READY_CLASSIFICATIONS
        and not detail.get("blockers")
    )
    blocked_or_unknown_operation_types = sorted(
        operation_type
        for operation_type, detail in operation_capabilities.items()
        if operation_type not in endpoint_ready_operation_types
    )
    project_status = _status_support(projects_probe)
    statuses_support = (
        True
        if status_probe_for_operation and status_probe_for_operation.get("available")
        else False
        if status_probe_for_operation
        and status_probe_for_operation.get("status_code") == 404
        else "unknown"
    )
    result = {
        "schema_version": GITEA_FORGE_CAPABILITY_MAP_SCHEMA_VERSION,
        "created_at": utc_now(),
        "gitea_base_url": redact_url(client.base_url),
        "owner": owner,
        "repo": repo,
        "auth_used": bool(auth_used),
        "token_env_var_used": token_env_name if auth_used else None,
        "token_exposed": False,
        "token_value_exposed": False,
        "read_only": True,
        "permission_proof": False,
        "endpoint_shape_evidence_only": True,
        "calls_gitea_write_api": False,
        "mutation_executed": False,
        "allowed_http_methods": list(ALLOWED_HTTP_METHODS),
        "forbidden_http_methods": list(MUTATING_HTTP_METHODS),
        "http_methods_used": sorted(set(client.methods_used)),
        "endpoint_probes": probes,
        "operation_capabilities": operation_capabilities,
        "endpoint_ready_operation_types": endpoint_ready_operation_types,
        "blocked_or_unknown_operation_types": blocked_or_unknown_operation_types,
        "repo_metadata_supported": _status_support(repo_probe),
        "repo_branches_supported": _status_support(branches_probe),
        "repo_commits_supported": _status_support(commits_probe),
        "issues_read_supported": _status_support(issues_probe),
        "issues_open_read_supported": _status_support(issues_open_probe),
        "issues_closed_read_supported": _status_support(issues_closed_probe),
        "labels_read_supported": _status_support(labels_probe),
        "milestones_read_supported": _status_support(milestones_probe),
        "pull_requests_read_supported": _status_support(pulls_probe),
        "comments_read_supported": (
            _status_support(comments_probe) if comments_probe else "unknown"
        ),
        "statuses_read_supported": statuses_support,
        "projects_read_supported": project_status,
        "repo_project_id_sampled": project_id is not None,
        "project_columns_sampled": columns_probe is not None,
        "project_cards_sampled": cards_probe is not None,
        "project_endpoint_status": {
            "repo_projects_status_code": projects_probe.get("status_code"),
            "user_projects_status_code": user_projects_probe.get("status_code"),
            "org_projects_status_code": org_projects_probe.get("status_code"),
            "repo_project_columns_status_code": (
                columns_probe.get("status_code") if columns_probe else None
            ),
            "project_cards_status_code": (
                cards_probe.get("status_code") if cards_probe else None
            ),
            "classification": (
                "endpoint_not_found"
                if project_status is False
                else "read_endpoint_ready_write_unproven"
                if project_status is True
                else "endpoint_unknown"
            ),
            "issue_only_fallback_recommended": (
                project_status is not True
                or any(
                    operation_capabilities[operation_type]["classification"]
                    not in READY_CLASSIFICATIONS
                    for operation_type in PROJECT_OPERATION_TYPES
                )
            ),
        },
        "required_future_write_endpoints": {
            operation_type: detail["required_future_write_endpoint"]
            for operation_type, detail in operation_capabilities.items()
        },
        "blockers": blockers,
        "warnings": warnings,
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return result


def capture_gitea_forge_capability_map(
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
    client = GiteaForgeCapabilityMapClient(
        base_url=base_url,
        token=token,
        timeout=timeout,
        transport=transport,
    )
    return build_gitea_forge_capability_map(
        client=client,
        owner=owner,
        repo=repo,
        auth_used=bool(token),
        token_env_name=token_env_name,
        limit=limit,
    )
