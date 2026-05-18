#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

try:
    from scripts.hermes_pm.forge_issue_executor import (
        APPROVED_ISSUES_ENDPOINT,
        APPROVED_OWNER,
        APPROVED_REPO,
        CHECKPOINT_6_BODY_SNIPPETS,
        DEFAULT_GITEA_BASE_URL,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import (
        load_read_token,
        redact_text,
        repo_api_path,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_issue_executor import (  # type: ignore[no-redef]
        APPROVED_ISSUES_ENDPOINT,
        APPROVED_OWNER,
        APPROVED_REPO,
        CHECKPOINT_6_BODY_SNIPPETS,
        DEFAULT_GITEA_BASE_URL,
    )
    from gitea_readonly_snapshot import (  # type: ignore[no-redef]
        load_read_token,
        redact_text,
        repo_api_path,
    )


FORGE_ISSUE_ATTESTATION_SCHEMA_VERSION = (
    "hermes.pm.forge_issue_creation_attestation.v1"
)

NON_ACTION_BOOLEANS = {
    "calls_gitea_write_api": False,
    "mutation_executed": False,
    "labels_created": False,
    "project_mutated": False,
    "comments_created": False,
    "prs_created": False,
    "workflows_run": False,
    "runners_started": False,
    "branch_writer_invoked": False,
    "deploys": False,
    "runtime_actions": False,
    "financial_actions": False,
    "secret_access": False,
    "token_value_exposed": False,
}

SHA256_RE = re.compile(r"[0-9a-f]{64}")
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


class ForgeIssueAttestationReadClient:
    mutation_capability = False
    allowed_methods = ("GET",)

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
        url = self._build_url(path, params=params)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        self.methods_used.append("GET")
        status, body = self._transport(request, self.timeout)
        if not body:
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


def _bool_false(value: Any) -> bool:
    return value is False


def _list_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("comments", "items", "data", "projects"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def _issue_path(issue_index: Any) -> str:
    quoted = urllib.parse.quote(str(issue_index), safe="")
    return f"{APPROVED_ISSUES_ENDPOINT}/{quoted}"


def attest_forge_issue_creation(
    *,
    evidence: dict[str, Any],
    expected_title: str,
    env: dict[str, str] | None = None,
    timeout: int = 20,
    transport: Transport | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    base_url = str(evidence.get("gitea_base_url") or DEFAULT_GITEA_BASE_URL).rstrip("/")
    token, token_env = load_read_token(env=env)
    client = ForgeIssueAttestationReadClient(
        base_url=base_url,
        token=token,
        timeout=timeout,
        transport=transport,
    )
    blockers: list[str] = []
    warnings: list[str] = []
    issue_index = evidence.get("issue_index")
    operation_id = str(evidence.get("operation_id") or "")
    plan_sha = str(evidence.get("plan_sha256") or "")

    if evidence.get("owner") != APPROVED_OWNER:
        blockers.append("Evidence owner is not preston.")
    if evidence.get("repo") != APPROVED_REPO:
        blockers.append("Evidence repo is not crypto_bot.")
    if evidence.get("issue_title") != expected_title:
        blockers.append("Evidence issue title does not match expected title.")
    if not operation_id:
        blockers.append("Evidence operation_id is missing.")
    if not SHA256_RE.fullmatch(plan_sha):
        blockers.append("Evidence plan_sha256 is missing or invalid.")
    if evidence.get("mutation_executed") is not True:
        blockers.append("Evidence does not record a completed mutation.")
    if evidence.get("write_call_count") != 1:
        blockers.append("Evidence write_call_count must be exactly 1.")
    if evidence.get("write_endpoint") != APPROVED_ISSUES_ENDPOINT:
        blockers.append("Evidence write_endpoint must be the create-issue endpoint.")

    for field in (
        "labels_created",
        "projects_mutated",
        "comments_created",
        "workflows_run",
        "runners_started",
        "branch_writer_invoked",
        "deploys",
        "runtime_actions",
        "financial_actions",
        "secret_access",
        "token_value_exposed",
    ):
        if not _bool_false(evidence.get(field)):
            blockers.append(f"Evidence field {field} must be false.")

    issue_status: int | None = None
    issue_payload: Any = {}
    comments_status: int | None = None
    comments_payload: Any = {}
    projects_status: int | None = None
    projects_payload: Any = {}

    if issue_index is None:
        blockers.append("Evidence issue_index is missing.")
    else:
        issue_status, issue_payload = client.get(_issue_path(issue_index))
        comments_status, comments_payload = client.get(
            f"{_issue_path(issue_index)}/comments",
            params={"limit": 10},
        )
        projects_status, projects_payload = client.get(
            repo_api_path(APPROVED_OWNER, APPROVED_REPO, "/projects"),
            params={"limit": 10},
        )

    issue_ok = isinstance(issue_payload, dict) and issue_status == 200
    title_matched = issue_ok and issue_payload.get("title") == expected_title
    body = str(issue_payload.get("body") or "") if issue_ok else ""
    body_markers_present = issue_ok and all(
        snippet in body for snippet in CHECKPOINT_6_BODY_SNIPPETS
    )
    labels_empty = issue_ok and not bool(issue_payload.get("labels"))
    comments = _list_payload(comments_payload)
    comments_empty = comments_status == 200 and comments == []
    projects = _list_payload(projects_payload)
    project_absent_or_empty = projects_status == 404 or (
        projects_status == 200 and projects == []
    )

    if not issue_ok:
        blockers.append("Created issue was not readable by GET.")
    if issue_ok and issue_payload.get("repository"):
        repository = issue_payload.get("repository")
        if isinstance(repository, dict):
            full_name = repository.get("full_name")
            if full_name and full_name != f"{APPROVED_OWNER}/{APPROVED_REPO}":
                blockers.append("Issue repository full_name did not match evidence.")
    if not title_matched:
        blockers.append("Created issue title did not match expected title.")
    if not body_markers_present:
        blockers.append("Created issue body is missing Checkpoint 6 markers.")
    if not labels_empty:
        blockers.append("Created issue has labels.")
    if comments_status != 200:
        warnings.append("Issue comments could not be safely verified.")
    elif not comments_empty:
        blockers.append("Issue has comments; PM-6 approved none.")
    if not project_absent_or_empty:
        blockers.append("Project/card evidence was present or not safely excluded.")

    result = {
        "schema_version": FORGE_ISSUE_ATTESTATION_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "valid": not blockers,
        "evidence_id": evidence.get("evidence_id"),
        "issue_index": issue_index,
        "issue_url": evidence.get("issue_url"),
        "issue_title": evidence.get("issue_title"),
        "operation_id": operation_id,
        "plan_sha256": plan_sha,
        "approval_id": evidence.get("approval_id"),
        "gitea_base_url": base_url,
        "owner": evidence.get("owner"),
        "repo": evidence.get("repo"),
        "auth_used": bool(token),
        "auth_env_var_used": token_env if token else None,
        "http_methods_used": sorted(set(client.methods_used)),
        "checks": {
            "issue_exists": issue_ok,
            "issue_title_matched": bool(title_matched),
            "body_checkpoint_6_marker_present": bool(body_markers_present),
            "issue_repo_matched": evidence.get("owner") == APPROVED_OWNER
            and evidence.get("repo") == APPROVED_REPO,
            "labels_empty": bool(labels_empty),
            "comments_empty_if_verifiable": bool(comments_empty),
            "project_card_evidence_absent": bool(project_absent_or_empty),
            "operation_id_recorded": bool(operation_id),
            "plan_sha256_recorded": bool(SHA256_RE.fullmatch(plan_sha)),
            "no_workflows_or_runners_triggered_by_tool": (
                evidence.get("workflows_run") is False
                and evidence.get("runners_started") is False
            ),
        },
        "blockers": [redact_text(item) for item in blockers],
        "warnings": [redact_text(item) for item in warnings],
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }
    return result


def format_forge_issue_attestation_text(attestation: dict[str, Any]) -> str:
    lines = [
        "Hermes PM one-issue attestation",
        f"Valid: {'yes' if attestation.get('valid') else 'no'}",
        f"Issue: #{attestation.get('issue_index') or '<none>'}",
        f"Operation: {attestation.get('operation_id') or '<none>'}",
        f"Plan sha256: {attestation.get('plan_sha256') or '<none>'}",
        f"Methods: {', '.join(attestation.get('http_methods_used') or ['<none>'])}",
        "Gitea writes performed by attestation: no",
    ]
    blockers = attestation.get("blockers") or []
    if blockers:
        lines.append(f"Blockers: {len(blockers)}")
        for blocker in blockers[:4]:
            lines.append(f"- {blocker}")
    return redact_text("\n".join(lines))
