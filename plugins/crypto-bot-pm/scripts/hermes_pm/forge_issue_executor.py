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

try:
    from scripts.hermes_pm.forge_execution_gate import build_forge_execution_gate
    from scripts.hermes_pm.forge_write_plan import (
        redact_secret_values,
        sha256_payload,
    )
    from scripts.hermes_pm.gitea_readonly_snapshot import redact_text
    from scripts.hermes_pm.selected_candidate_execution_payload import (
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        SelectedCandidateExecutionPayloadError,
        validate_selected_candidate_execution_payload,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from forge_execution_gate import (  # type: ignore[no-redef]
        build_forge_execution_gate,
    )
    from forge_write_plan import (  # type: ignore[no-redef]
        redact_secret_values,
        sha256_payload,
    )
    from gitea_readonly_snapshot import redact_text  # type: ignore[no-redef]
    from selected_candidate_execution_payload import (  # type: ignore[no-redef]
        STABLE_EXECUTION_PAYLOAD_HASH_FIELD,
        SelectedCandidateExecutionPayloadError,
        validate_selected_candidate_execution_payload,
    )


FORGE_ISSUE_CREATION_RESULT_SCHEMA_VERSION = (
    "hermes.pm.forge_issue_creation_result.v1"
)
FORGE_ISSUE_CREATION_EVIDENCE_SCHEMA_VERSION = (
    "hermes.pm.forge_issue_creation_evidence.v1"
)

DEFAULT_GITEA_BASE_URL = "http://127.0.0.1:3005"
APPROVED_OWNER = "preston"
APPROVED_REPO = "crypto_bot"
APPROVED_PROJECT_ID = "crypto_bot"
APPROVED_OPERATION_TYPE = "create_issue"
APPROVED_ISSUES_ENDPOINT = "/api/v1/repos/preston/crypto_bot/issues"
WRITE_TOKEN_ENV_NAMES = ("GITEA_TOKEN", "TEA_TOKEN")

CHECKPOINT_6_ISSUE_TITLE = (
    "[Hermes PM] Establish initial PM-managed backlog item"
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

READY_CAPABILITY_CLASSIFICATIONS = {
    "endpoint_ready",
    "read_endpoint_ready_write_unproven",
}

FORBIDDEN_PAYLOAD_FIELDS = (
    "assignee",
    "assignees",
    "closed",
    "comments",
    "due_date",
    "deadline",
    "milestone",
    "project",
    "project_id",
    "ref",
)

TOKEN_LOOKING_PATTERNS = (
    re.compile(r"(?i)\b(token|secret|password|api[_-]?key)(\s*=\s*)[^\s,;]+"),
    re.compile(r"(?i)\bauthorization\s*:\s*bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(access_token=)[^&\s]+"),
)

NON_ACTION_BOOLEANS = {
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

Transport = Callable[[urllib.request.Request, int], tuple[int, bytes]]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def repo_issue_path(issue_index: Any) -> str:
    quoted = urllib.parse.quote(str(issue_index), safe="")
    return f"{APPROVED_ISSUES_ENDPOINT}/{quoted}"


def default_transport(
    request: urllib.request.Request,
    timeout: int,
) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class ForgeIssueExecutorError(RuntimeError):
    """Raised when the narrow issue executor is asked to exceed PM-6 scope."""


class GiteaIssueExecutorClient:
    mutation_capability = True
    allowed_write_endpoint = APPROVED_ISSUES_ENDPOINT

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: int = 20,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout
        self._transport = transport or default_transport
        self.write_calls: list[dict[str, str]] = []
        self.methods_used: list[str] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        self._ensure_get_path(path)
        return self._request("GET", path, params=params)

    def post_issue(self, payload: dict[str, Any]) -> tuple[int, Any]:
        return self._request("POST", APPROVED_ISSUES_ENDPOINT, payload=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[int, Any]:
        method = method.upper()
        if method == "POST" and path != APPROVED_ISSUES_ENDPOINT:
            raise ForgeIssueExecutorError(
                "PM-6 permits POST only to the create-issue endpoint."
            )
        if method != "POST" and method != "GET":
            raise ForgeIssueExecutorError(
                f"PM-6 issue executor blocks HTTP method {method}."
            )
        url = self._build_url(path, params=params)
        data = None
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        if payload is not None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        self.methods_used.append(method)
        if method == "POST":
            self.write_calls.append({"method": "POST", "endpoint": path})
        status, body = self._transport(request, self.timeout)
        decoded = self._decode_body(body)
        return status, decoded

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

    def _decode_body(self, body: bytes) -> Any:
        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {"non_json_body": redact_text(body.decode("utf-8", "replace"))}

    def _ensure_get_path(self, path: str) -> None:
        if path == APPROVED_ISSUES_ENDPOINT:
            return
        issue_prefix = f"{APPROVED_ISSUES_ENDPOINT}/"
        suffix = path.removeprefix(issue_prefix)
        if path.startswith(issue_prefix) and suffix.isdigit():
            return
        raise ForgeIssueExecutorError(
            "PM-6 issue executor permits GET only for issue preflight and "
            "issue verification."
        )


def _load_write_token(
    env: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    source = env if env is not None else os.environ
    for name in WRITE_TOKEN_ENV_NAMES:
        value = source.get(name)
        if value:
            return value, name
    return None, None


def _normalize_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _is_loopback_base_url(value: str) -> bool:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.username or parsed.password or parsed.path or parsed.query:
        return False
    host = parsed.hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def _safe_text(value: str) -> bool:
    if "\x00" in value:
        return False
    return not any(pattern.search(value) for pattern in TOKEN_LOOKING_PATTERNS)


def _redact_issue_text(value: str) -> str:
    redacted = value
    redacted = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1<redacted-token>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1<redacted-token>",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(token|secret|password|api[_-]?key)(\s*=\s*)[^\s,;]+",
        r"\1\2<redacted-token>",
        redacted,
    )
    return redact_text(redacted)


def _operation_by_id(
    plan: dict[str, Any],
    operation_id: str,
) -> dict[str, Any] | None:
    operations = plan.get("operations")
    if not isinstance(operations, list):
        return None
    for operation in operations:
        if (
            isinstance(operation, dict)
            and operation.get("operation_id") == operation_id
        ):
            return operation
    return None


def _selected_operation_ids(token: dict[str, Any]) -> list[str]:
    values = token.get("approved_operation_ids")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _payload_has_forbidden_fields(payload: dict[str, Any]) -> list[str]:
    offenders: list[str] = []
    if payload.get("labels"):
        offenders.append("labels")
    for field in FORBIDDEN_PAYLOAD_FIELDS:
        if payload.get(field):
            offenders.append(field)
    return sorted(dict.fromkeys(offenders))


def _validate_operation_payload(
    *,
    operation: dict[str, Any],
    expected_title: str,
    execution_payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if operation.get("operation_type") != APPROVED_OPERATION_TYPE:
        blockers.append("Selected operation type must be create_issue.")
    if operation.get("blocked"):
        blockers.append("Selected operation is blocked in the plan.")
    if operation.get("http_method_that_would_be_used") != "POST":
        blockers.append("Selected operation must preview POST.")
    if operation.get("expected_gitea_endpoint") != APPROVED_ISSUES_ENDPOINT:
        blockers.append("Selected operation endpoint must be the issue endpoint.")

    payload = operation.get("proposed_payload_redacted")
    if not isinstance(payload, dict):
        blockers.append("Selected operation payload must be an object.")
        return None, blockers

    title = str(payload.get("title") or "")
    body = str(payload.get("body") or "")
    if title != expected_title:
        blockers.append("Selected operation title does not match expected title.")
    if not body:
        blockers.append("Issue body is required.")
    if execution_payload is not None:
        try:
            frozen = validate_selected_candidate_execution_payload(execution_payload)
        except SelectedCandidateExecutionPayloadError as exc:
            blockers.append(f"Selected-candidate execution payload is invalid: {exc}")
        else:
            if operation.get("operation_id") != frozen["operation_id"]:
                blockers.append(
                    "Selected operation ID does not match the execution payload."
                )
            if title != frozen["issue_title"]:
                blockers.append(
                    "Selected operation title does not match execution payload."
                )
            if body != frozen["issue_body"]:
                blockers.append(
                    "Selected operation body does not match execution payload."
                )
    else:
        if title != CHECKPOINT_6_ISSUE_TITLE:
            blockers.append("Selected operation title is not the PM-6 issue title.")
        for snippet in CHECKPOINT_6_BODY_SNIPPETS:
            if snippet not in body:
                blockers.append(f"Issue body is missing required marker: {snippet}")
    forbidden_fields = _payload_has_forbidden_fields(payload)
    if forbidden_fields:
        blockers.append(
            "Selected issue payload includes forbidden fields: "
            + ", ".join(forbidden_fields)
        )
    if not _safe_text(title) or not _safe_text(body):
        blockers.append("Selected issue title/body contains token-looking text.")
    if len(title) > 255:
        blockers.append("Selected issue title is too long.")
    if len(body) > 10000:
        blockers.append("Selected issue body is too long.")
    return payload, blockers


def _capability_ready(capability_map: dict[str, Any]) -> tuple[bool, str]:
    capabilities = capability_map.get("operation_capabilities")
    if isinstance(capabilities, dict):
        detail = capabilities.get(APPROVED_OPERATION_TYPE)
        if not isinstance(detail, dict):
            return False, "Capability map lacks create_issue detail."
        classification = str(detail.get("classification") or "")
        blockers = detail.get("blockers") or []
        endpoint = detail.get("required_future_write_endpoint")
        if not endpoint:
            endpoint = {"endpoint": APPROVED_ISSUES_ENDPOINT}
        endpoint_value = (
            endpoint.get("endpoint") if isinstance(endpoint, dict) else None
        )
        ready = (
            classification in READY_CAPABILITY_CLASSIFICATIONS
            and not blockers
            and endpoint_value == APPROVED_ISSUES_ENDPOINT
        )
        if ready:
            return True, "create_issue capability has ready read evidence."
        return False, (
            "create_issue capability is not ready for PM-6 issue rehearsal."
        )
    ready_types = capability_map.get("endpoint_ready_operation_types") or []
    if APPROVED_OPERATION_TYPE in ready_types:
        return True, "create_issue is listed as endpoint-ready."
    return False, "Capability map does not mark create_issue endpoint-ready."


def _base_result(
    *,
    plan: dict[str, Any],
    token: dict[str, Any],
    operation_id: str,
    expected_plan_sha256: str,
    expected_title: str,
    base_url: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FORGE_ISSUE_CREATION_RESULT_SCHEMA_VERSION,
        "created_at": created_at or utc_now(),
        "created": False,
        "issue_index": None,
        "issue_url": None,
        "issue_title": expected_title,
        "operation_id": operation_id,
        "plan_sha256": expected_plan_sha256,
        "approval_id": str(token.get("approval_id") or ""),
        "gitea_base_url": base_url,
        "owner": plan.get("owner"),
        "repo": plan.get("repo"),
        "mutation_executed": False,
        "write_token_available": False,
        "auth_env_var_used": None,
        "temporary_token_created": False,
        "temporary_token_revoked": False,
        "gitea_write_calls": [],
        "blockers": [],
        "warnings": [],
        "duplicate_preflight": {
            "checked": False,
            "duplicate_found": False,
            "duplicate_issue_index": None,
            "duplicate_issue_url": None,
        },
        "verification": {
            "issue_get_performed": False,
            "issue_title_matched": False,
            "issue_labels_empty": False,
        },
        "non_action_booleans": dict(NON_ACTION_BOOLEANS),
    }


def _block(
    result: dict[str, Any],
    *reasons: str,
) -> dict[str, Any]:
    result["blockers"] = list(
        dict.fromkeys(
            [
                *result.get("blockers", []),
                *[_redact_issue_text(reason) for reason in reasons],
            ]
        )
    )
    return redact_secret_values(result)


def _parse_issue_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("issues", "items", "data"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    return []


def find_duplicate_issue(
    *,
    client: GiteaIssueExecutorClient,
    title: str,
    max_pages: int = 10,
    page_limit: int = 50,
) -> dict[str, Any]:
    for page in range(1, max_pages + 1):
        status, payload = client.get(
            APPROVED_ISSUES_ENDPOINT,
            params={"state": "all", "limit": page_limit, "page": page},
        )
        if status < 200 or status >= 300:
            return {
                "checked": True,
                "safe_to_create": False,
                "duplicate_found": False,
                "status_code": status,
                "blocker": f"Duplicate preflight GET failed with HTTP {status}.",
            }
        issues = _parse_issue_list(payload)
        for issue in issues:
            if "pull_request" in issue:
                continue
            if issue.get("title") == title:
                return {
                    "checked": True,
                    "safe_to_create": False,
                    "duplicate_found": True,
                    "duplicate_issue_index": issue.get("number"),
                    "duplicate_issue_url": issue.get("html_url"),
                }
        if len(issues) < page_limit:
            break
    return {
        "checked": True,
        "safe_to_create": True,
        "duplicate_found": False,
        "duplicate_issue_index": None,
        "duplicate_issue_url": None,
    }


def _validate_scope(
    *,
    plan: dict[str, Any],
    base_url: str,
    explicit_base_url: bool,
) -> list[str]:
    blockers: list[str] = []
    plan_base_url = _normalize_base_url(plan.get("gitea_base_url"))
    if explicit_base_url:
        if not _is_loopback_base_url(base_url):
            blockers.append("Explicit Gitea base URL must be loopback.")
    elif base_url != DEFAULT_GITEA_BASE_URL:
        blockers.append(
            f"Gitea base URL must default to {DEFAULT_GITEA_BASE_URL}."
        )
    if plan_base_url != base_url:
        blockers.append("Resolved Gitea base URL must match the plan.")
    if plan.get("project_id") != APPROVED_PROJECT_ID:
        blockers.append("Project id must be crypto_bot.")
    if plan.get("owner") != APPROVED_OWNER:
        blockers.append("Owner must be preston.")
    if plan.get("repo") != APPROVED_REPO:
        blockers.append("Repo must be crypto_bot.")
    return blockers


def execute_forge_issue_create(
    *,
    forge_write_plan: dict[str, Any],
    approval_token: dict[str, Any],
    gitea_capabilities: dict[str, Any],
    operation_id: str,
    expected_plan_sha256: str,
    expected_title: str,
    confirmation: bool,
    expected_execution_payload_sha256: str | None = None,
    execution_payload: dict[str, Any] | None = None,
    base_url: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 20,
    transport: Transport | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    plan = redact_secret_values(forge_write_plan)
    token = redact_secret_values(approval_token)
    capabilities = redact_secret_values(gitea_capabilities)
    resolved_base_url = _normalize_base_url(base_url or plan.get("gitea_base_url"))
    explicit_base_url = base_url is not None
    result = _base_result(
        plan=plan,
        token=token,
        operation_id=operation_id,
        expected_plan_sha256=expected_plan_sha256,
        expected_title=expected_title,
        base_url=resolved_base_url,
        created_at=created_at,
    )

    if not confirmation:
        return _block(result, "Explicit PM-6 issue creation confirmation is required.")

    plan_sha = sha256_payload(plan)
    result["plan_sha256"] = plan_sha
    stable_plan_hash = plan.get(STABLE_EXECUTION_PAYLOAD_HASH_FIELD)
    stable_required = bool(
        stable_plan_hash
        and (
            plan.get("requires_stable_execution_payload_sha256") is True
            or plan.get("approval_hash_kind") == STABLE_EXECUTION_PAYLOAD_HASH_FIELD
        )
    )
    result["stable_execution_payload_sha256"] = stable_plan_hash
    result["expected_execution_payload_sha256"] = expected_execution_payload_sha256
    if stable_required:
        if not expected_execution_payload_sha256:
            return _block(
                result,
                "Selected-candidate execution requires exact "
                "stable_execution_payload_sha256; whole-plan SHA-256 is only "
                "reference evidence.",
            )
        if expected_execution_payload_sha256 != stable_plan_hash:
            return _block(
                result,
                "Expected stable execution payload SHA-256 does not match the plan.",
            )
        if execution_payload is None:
            embedded_payload = plan.get("selected_candidate_execution_payload")
            execution_payload = (
                embedded_payload if isinstance(embedded_payload, dict) else None
            )
        if execution_payload is None:
            return _block(
                result,
                "Selected-candidate execution requires the frozen execution "
                "payload for exact title/body validation.",
            )
        try:
            frozen_payload = validate_selected_candidate_execution_payload(
                execution_payload
            )
        except SelectedCandidateExecutionPayloadError as exc:
            return _block(result, f"Frozen execution payload is invalid: {exc}")
        if frozen_payload[STABLE_EXECUTION_PAYLOAD_HASH_FIELD] != stable_plan_hash:
            return _block(
                result,
                "Frozen execution payload hash does not match the plan.",
            )
        if operation_id != frozen_payload["operation_id"]:
            return _block(
                result,
                "Requested operation ID does not match the frozen execution payload.",
            )
        if expected_title != frozen_payload["issue_title"]:
            return _block(
                result,
                "Expected title does not match the frozen execution payload.",
            )
    elif expected_plan_sha256 != plan_sha:
        return _block(result, "Expected plan SHA-256 does not match the plan.")

    scope_blockers = _validate_scope(
        plan=plan,
        base_url=resolved_base_url,
        explicit_base_url=explicit_base_url,
    )
    if scope_blockers:
        return _block(result, *scope_blockers)

    selected_ids = _selected_operation_ids(token)
    if selected_ids != [operation_id]:
        return _block(
            result,
            "Approval token must select exactly the requested operation ID.",
        )
    if token.get("allowed_operation_types") != [APPROVED_OPERATION_TYPE]:
        return _block(result, "Approval token must allow only create_issue.")
    if token.get("max_operations") != 1:
        return _block(result, "Approval token max_operations must be exactly 1.")

    validation_gate = build_forge_execution_gate(
        forge_write_plan=plan,
        approval_token=token,
        gitea_capabilities=capabilities,
    )
    if validation_gate.get("ready_for_future_execution") is not True:
        reasons = validation_gate.get("approval_validation", {}).get("reasons") or []
        if not reasons:
            reasons = ["Forge execution gate is not ready."]
        return _block(result, *[str(reason) for reason in reasons])
    approved_operations = validation_gate.get("approved_operations") or []
    if len(approved_operations) != 1:
        return _block(
            result,
            "Forge execution gate must approve exactly one operation.",
        )
    approved_operation = approved_operations[0]
    if approved_operation.get("operation_id") != operation_id:
        return _block(result, "Forge execution gate approved the wrong operation ID.")
    if approved_operation.get("operation_type") != APPROVED_OPERATION_TYPE:
        return _block(result, "Forge execution gate approved a non-issue operation.")

    capability_ready, capability_reason = _capability_ready(capabilities)
    if not capability_ready:
        return _block(result, capability_reason)

    operation = _operation_by_id(plan, operation_id)
    if operation is None:
        return _block(result, "Requested operation ID is not present in the plan.")
    payload, payload_blockers = _validate_operation_payload(
        operation=operation,
        expected_title=expected_title,
        execution_payload=execution_payload if stable_required else None,
    )
    if payload_blockers:
        return _block(result, *payload_blockers)
    if payload is None:
        return _block(result, "Selected operation payload is unavailable.")

    token_value, token_env_name = _load_write_token(env=env)
    result["write_token_available"] = bool(token_value)
    result["auth_env_var_used"] = token_env_name
    if not token_value:
        return _block(
            result,
            "No safe Gitea write token was available from GITEA_TOKEN or TEA_TOKEN.",
        )

    client = GiteaIssueExecutorClient(
        base_url=resolved_base_url,
        token=token_value,
        timeout=timeout,
        transport=transport,
    )
    try:
        duplicate = find_duplicate_issue(client=client, title=expected_title)
    except (ForgeIssueExecutorError, OSError, urllib.error.URLError) as exc:
        return _block(result, f"Duplicate preflight failed: {exc}")
    result["duplicate_preflight"] = duplicate
    if duplicate.get("duplicate_found"):
        return _block(
            result,
            "Duplicate issue with the exact PM-6 title already exists.",
        )
    if duplicate.get("safe_to_create") is not True:
        blocker = str(duplicate.get("blocker") or "Duplicate check failed.")
        return _block(result, blocker)

    issue_payload = {
        "title": str(payload.get("title") or ""),
        "body": str(payload.get("body") or ""),
    }
    try:
        status, created_payload = client.post_issue(issue_payload)
    except (ForgeIssueExecutorError, OSError, urllib.error.URLError) as exc:
        result["gitea_write_calls"] = list(client.write_calls)
        result["mutation_executed"] = bool(client.write_calls)
        return _block(result, f"Create issue POST failed without retry: {exc}")

    result["gitea_write_calls"] = list(client.write_calls)
    result["mutation_executed"] = bool(client.write_calls)
    if len(client.write_calls) != 1:
        return _block(result, "Executor attempted more than one Gitea write call.")
    if status not in {200, 201}:
        return _block(result, f"Create issue POST returned HTTP {status}; no retry.")
    if not isinstance(created_payload, dict):
        return _block(result, "Create issue response was not a JSON object; no retry.")

    issue_index = created_payload.get("number")
    if issue_index is None:
        return _block(result, "Create issue response did not include issue number.")
    result["issue_index"] = issue_index
    result["issue_url"] = created_payload.get("html_url")
    result["issue_title"] = created_payload.get("title") or expected_title

    verify_status, verify_payload = client.get(repo_issue_path(issue_index))
    verification = result["verification"]
    verification["issue_get_performed"] = True
    verification["issue_get_status_code"] = verify_status
    if verify_status != 200 or not isinstance(verify_payload, dict):
        return _block(result, "Created issue verification GET failed.")
    verification["issue_title_matched"] = verify_payload.get("title") == expected_title
    verification["issue_labels_empty"] = not bool(verify_payload.get("labels"))
    if not verification["issue_title_matched"]:
        return _block(result, "Created issue verification title mismatch.")
    if not verification["issue_labels_empty"]:
        return _block(result, "Created issue unexpectedly has labels.")

    result["created"] = True
    return redact_secret_values(result)


def build_issue_creation_evidence(
    result: dict[str, Any],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    seed = {
        "issue_index": result.get("issue_index"),
        "operation_id": result.get("operation_id"),
        "plan_sha256": result.get("plan_sha256"),
        "approval_id": result.get("approval_id"),
    }
    evidence_id = f"forge-issue-evidence-{sha256_payload(seed)[:16]}"
    return redact_secret_values(
        {
            "schema_version": FORGE_ISSUE_CREATION_EVIDENCE_SCHEMA_VERSION,
            "evidence_id": evidence_id,
            "created_at": created_at or utc_now(),
            "issue_index": result.get("issue_index"),
            "issue_url": result.get("issue_url"),
            "issue_title": result.get("issue_title"),
            "operation_id": result.get("operation_id"),
            "plan_sha256": result.get("plan_sha256"),
            "approval_id": result.get("approval_id"),
            "gitea_base_url": result.get("gitea_base_url"),
            "owner": result.get("owner"),
            "repo": result.get("repo"),
            "mutation_executed": bool(result.get("mutation_executed")),
            "write_call_count": len(result.get("gitea_write_calls") or []),
            "write_endpoint": APPROVED_ISSUES_ENDPOINT,
            "labels_created": False,
            "projects_mutated": False,
            "comments_created": False,
            "workflows_run": False,
            "runners_started": False,
            "branch_writer_invoked": False,
            "deploys": False,
            "runtime_actions": False,
            "financial_actions": False,
            "secret_access": False,
            "token_value_exposed": False,
        }
    )


def format_forge_issue_result_text(result: dict[str, Any]) -> str:
    write_calls = result.get("gitea_write_calls") or []
    lines = [
        "Hermes PM one-issue forge write",
        f"Created: {'yes' if result.get('created') else 'no'}",
        f"Operation: {result.get('operation_id') or '<none>'}",
        f"Plan sha256: {result.get('plan_sha256') or '<none>'}",
        f"Approval: {result.get('approval_id') or '<none>'}",
        f"Issue: #{result.get('issue_index') or '<none>'}",
        f"Write calls: {len(write_calls)}",
        f"Write endpoint: {APPROVED_ISSUES_ENDPOINT if write_calls else '<none>'}",
        (
            "Forbidden side effects: "
            "labels=no projects=no comments=no prs=no workflows=no runners=no "
            "deploys=no runtime=no financial=no secrets=no branch-writer=no"
        ),
    ]
    blockers = result.get("blockers") or []
    if blockers:
        lines.append(f"Blockers: {len(blockers)}")
        for blocker in blockers[:4]:
            lines.append(f"- {blocker}")
    return _redact_issue_text("\n".join(lines))
