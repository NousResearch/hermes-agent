"""Minimal local-first MCP bridge task store.

This module intentionally does not execute tasks. It validates task
contracts, records accepted/refused submissions under HERMES_HOME, and
returns stored status/result records for a future orchestrator to consume.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


REQUIRED_SUBMIT_FIELDS = (
    "title",
    "project",
    "mode",
    "task_contract",
    "allowed_actions",
    "forbidden_actions",
    "return_format",
)

SCOPE_FIELDS = ("repo_scope", "worktree_scope")

_UNCLEAR_CONTRACTS = {
    "do it",
    "do stuff",
    "fix it",
    "make changes",
    "whatever",
    "help",
}

_HARD_REFUSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "approval bypass or safety-disable requests are refused",
        re.compile(
            r"\b(bypass|skip|disable|turn\s+off|ignore)\s+"
            r"(?:the\s+)?(?:approval|approvals|safety|safeguards?|policy|gatekeeper)\b",
            re.I,
        ),
    ),
    ("run_shell is not exposed by this bridge", re.compile(r"\brun[_ -]?shell\b", re.I)),
    (
        "raw shell access is not exposed by this bridge",
        re.compile(
            r"\b(raw\s+shell|shell\s+access|direct\s+command\s+execution|run[_ -]?shell)\b|"
            r"\b(?:add|expose|provide|enable|create|publish|surface|make)\b.{0,80}"
            r"\b(raw\s+shell|shell\s+access|direct\s+command\s+execution|run[_ -]?shell)\b|"
            r"\b(raw\s+shell|shell\s+access|direct\s+command\s+execution)\b.{0,80}"
            r"\b(?:through|as|via|on|to)\s+(?:the\s+)?(?:bridge|mcp|public\s+mcp|workers?)\b",
            re.I,
        ),
    ),
    (
        "secret/env/token/auth file access is refused",
        re.compile(
            r"\b(read[_ -]?secret|secret[s]?|\.env|token[s]?|credential[s]?|"
            r"key\s+material|auth\s+file[s]?)\b",
            re.I,
        ),
    ),
    (
        "git_push is not exposed by this bridge",
        re.compile(r"\bgit[_-]push\b", re.I),
    ),
    (
        "shopify_import is not exposed by this bridge",
        re.compile(r"\bshopify[_ -]?import\b", re.I),
    ),
    (
        "docker_run is not exposed by this bridge",
        re.compile(r"\bdocker[_ -]?run\b|\bdocker\s+run\b", re.I),
    ),
    (
        "direct Codex calls are not exposed by this bridge",
        re.compile(r"\bcodex[_ -]?direct\b|\bdirect\s+codex\b", re.I),
    ),
    (
        "MCP tool changes are refused by this bridge",
        re.compile(
            r"\b(add|remove|install|delete|change|expose|publish|surface|create)\s+"
            r"(?:an?\s+)?(?:public\s+)?mcp\s+tools?\b|"
            r"\bmcp\s+tools?\s+(?:add|remove|install|delete|change|expose|publish|surface|create)s?\b",
            re.I,
        ),
    ),
)

_APPROVAL_REQUIRED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "local file writes require human approval before routing",
        re.compile(r"\b(write[_ -]?file|direct\s+write[_ -]?file|local\s+writes?)\b", re.I),
    ),
    (
        "destructive git operations require human approval before routing",
        re.compile(
            r"\bgit\s+(reset|clean|merge|rebase|switch|stash|tag|fetch|pull|push)\b|"
            r"\bforce\s+push\b|\bdestructive\s+git\b",
            re.I,
        ),
    ),
    (
        "Shopify writes require human approval before routing",
        re.compile(r"\bshopify\s+(write|mutation|update|import|delete)\b", re.I),
    ),
    (
        "PROD actions require human approval before routing",
        re.compile(
            r"\bprod(?:uction)?\s+(deploy|release|write|mutation|update|delete)\b|"
            r"\bdeploy\s+to\s+prod(?:uction)?\b",
            re.I,
        ),
    ),
    (
        "OpenAI API use requires human approval before routing",
        re.compile(r"\bopenai[_ -]?call\b|\bdirect\s+openai\b|\bopenai\s+api\b", re.I),
    ),
    (
        "service restart/reload requires human approval before routing",
        re.compile(r"\b(restart|reload|refresh)\s+(?:service[s]?|gateway|bridge|server[s]?|process(?:es)?)\b", re.I),
    ),
    (
        "real submit_task reproduction requires human approval before routing",
        re.compile(r"\bsubmit[_ -]?task\b", re.I),
    ),
)

_NETWORK_PATTERN = re.compile(
    r"\b(network|internet|external\s+api|curl|wget|http[s]?://)\b",
    re.I,
)
_EXACT_ENDPOINT_PATTERN = re.compile(
    r"\b(?:https?://(?:[A-Za-z0-9.-]+|localhost|127(?:\.\d{1,3}){3}|\[[0-9A-Fa-f:.]+\])"
    r"(?::\d{1,5})?(?:/[^\s,;]*)?|"
    r"(?:host|origin|endpoint)s?\s+(?:is|are|named|outside|against|for|to)?\s*"
    r"(?:[A-Za-z0-9.-]+|localhost|127(?:\.\d{1,3}){3})(?::\d{1,5})?)\b",
    re.I,
)
_BOUNDED_DIAGNOSTIC_PATTERN = re.compile(
    r"\b(read[-\s]?only|diagnos(?:e|tic)|health|status|inventory|list[_ -]?tools|"
    r"process\s+status|local/public\s+tunnel|exact\s+(?:host|origin|endpoint))\b",
    re.I,
)
_MUTATION_PATTERN = re.compile(
    r"\b(write|mutation|mutate|modify|delete|deploy|release|settings?\s+change|"
    r"tool\s+change|restart|reload|submit[_ -]?task)\b",
    re.I,
)
_TASK_ID_PATTERN = re.compile(r"\bmcp_[A-Za-z0-9_-]+\b")
_FULL_CODE_PATTERN = re.compile(
    r"\b(?:HERMES-BRIDGE-)?([0-9]{3,}[A-Z]{2,}(?:-[A-Z0-9]+)*)\b"
)
_LEADING_CODE_PATTERN = re.compile(
    r"\b(?:HERMES-BRIDGE-)?([0-9]{3,}[A-Z]{2,})(?:-[A-Z0-9-]+)?\b"
)
_BROAD_SCOPE_PATTERN = re.compile(
    r"\b(all|any|multiple|every)\s+(repo|repos|repository|repositories|worktree|worktrees)\b|"
    r"\bcross[-\s]?repo\b|\bentire\s+filesystem\b",
    re.I,
)
_INFRASTRUCTURE_ERROR_RESPONSE_PATTERN = re.compile(
    r"\bhttp\s*401\b|"
    r"\btoken[_\s-]invalidated\b|"
    r"\btoken[_\s-]revoked\b|"
    r"\binvalidated\s+oauth\b|"
    r"\bauthentication\s+token\s+invalidated\b|"
    r"\bprovider\s+authentication\s+failed\b|"
    r"\bno\s+credentials\s+stored\b|"
    r"\brefresh[_\s-]token[_\s-]reused\b",
    re.I,
)
_NEGATION_PATTERN = re.compile(
    r"\b(do\s+not|don't|dont|no|without|forbid(?:den)?|refuse|avoid|must\s+not|never)\b",
    re.I,
)
_PROHIBITION_FIELDS = {
    "forbidden_actions",
    "safety_statement",
    "risk_boundary",
    "risk_boundaries",
    "training_notes",
    "stop_conditions",
}
_POSITIVE_FIELDS = {
    "title",
    "project",
    "mode",
    "task_contract",
    "allowed_actions",
    "approvals",
    "allowed_commands",
    "commands",
    "repo_scope",
    "worktree_scope",
    "worker_selection_guidance",
    "expected_branch",
    "expected_head",
}


class MCPBridgeError(ValueError):
    """Raised when a bridge operation cannot be completed."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bridge_home() -> Path:
    """Return the bridge task-store home without changing global HERMES_HOME.

    The ChatGPT/API Playground bridge runs with a dedicated persistent home,
    while the live Discord gateway normally runs under the user's regular
    Hermes home. Mirror helpers must therefore be able to find the bridge
    inbox explicitly instead of relying only on the gateway process home.
    """
    explicit_home = os.getenv("HERMES_MCP_BRIDGE_HOME")
    if explicit_home:
        return Path(explicit_home).expanduser()

    home = get_hermes_home()
    if home.name == "mcp_bridge_playground_home":
        return home

    playground_home = home / "mcp_bridge_playground_home"
    if playground_home.exists():
        return playground_home

    return home


def _task_dir() -> Path:
    explicit_tasks_dir = os.getenv("HERMES_MCP_BRIDGE_TASKS_DIR")
    if explicit_tasks_dir:
        return Path(explicit_tasks_dir).expanduser()
    return _bridge_home() / "mcp_bridge_tasks"


def _record_path(task_id: str) -> Path:
    if not re.match(r"^mcp_[A-Za-z0-9_-]+$", task_id or ""):
        raise MCPBridgeError("invalid task_id")
    return _task_dir() / f"{task_id}.json"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _flatten(value: Any, *, skip_keys: set[str] | None = None) -> str:
    skip_keys = skip_keys or set()
    parts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in skip_keys:
                continue
            parts.append(str(key))
            parts.append(_flatten(item, skip_keys=skip_keys))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            parts.append(_flatten(item, skip_keys=skip_keys))
    elif value is not None:
        parts.append(str(value))
    return " ".join(p for p in parts if p)


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _missing_submit_fields(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_SUBMIT_FIELDS:
        if field == "forbidden_actions":
            if field not in payload or not isinstance(payload.get(field), list):
                missing.append(field)
        elif not _is_non_empty(payload.get(field)):
            missing.append(field)
    return missing


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 40):start]
    return bool(_NEGATION_PATTERN.search(window))


def _flatten_selected(payload: dict[str, Any], fields: set[str]) -> str:
    return _flatten({field: payload.get(field) for field in fields if field in payload})


def _flatten_positive_surface(payload: dict[str, Any]) -> str:
    positive_payload = {field: payload.get(field) for field in _POSITIVE_FIELDS if field in payload}
    return _flatten(positive_payload, skip_keys=_PROHIBITION_FIELDS).strip()


def _allows_bounded_read_only_network(payload: dict[str, Any], text: str) -> bool:
    """Permit exact-scope read-only health diagnostics, not broad network access."""
    positive_text = _flatten_positive_surface(payload)
    combined_text = f"{positive_text} {_flatten_selected(payload, _PROHIBITION_FIELDS)}"
    if not _EXACT_ENDPOINT_PATTERN.search(positive_text):
        return False
    if not _BOUNDED_DIAGNOSTIC_PATTERN.search(positive_text):
        return False

    for match in _MUTATION_PATTERN.finditer(text):
        if not _is_negated(text, match.start()):
            return False

    required_prohibitions = ("submit", "restart", "reload", "mutation")
    lowered = combined_text.lower()
    return all(word in lowered for word in required_prohibitions)


def _find_hard_refusal(payload: dict[str, Any]) -> str | None:
    missing = _missing_submit_fields(payload)
    if not any(_is_non_empty(payload.get(field)) for field in SCOPE_FIELDS):
        missing.append("repo_scope or worktree_scope")
    if missing:
        return f"missing required field(s): {', '.join(missing)}"

    contract = payload.get("task_contract")
    contract_text = _flatten(contract).strip()
    if not contract_text:
        return "unclear task contract: task_contract is empty"
    if contract_text.lower() in _UNCLEAR_CONTRACTS or len(contract_text) < 12:
        return "unclear task contract: provide concrete objective and boundaries"

    scope_text = _flatten({field: payload.get(field) for field in SCOPE_FIELDS})
    if _BROAD_SCOPE_PATTERN.search(scope_text):
        return "broad cross-repo mutation is refused; provide one repo_scope or worktree_scope"

    allowed_text = _flatten(payload.get("allowed_actions"))
    checked_text = _flatten_positive_surface(payload)
    for reason, pattern in _HARD_REFUSAL_PATTERNS:
        for match in pattern.finditer(allowed_text):
            if not _is_negated(allowed_text, match.start()):
                return reason
        for match in pattern.finditer(checked_text):
            if not _is_negated(checked_text, match.start()):
                return reason
    return None


def _find_approval_required(payload: dict[str, Any]) -> str | None:
    allowed_text = _flatten(payload.get("allowed_actions"))
    checked_text = _flatten_positive_surface(payload)

    for reason, pattern in _APPROVAL_REQUIRED_PATTERNS:
        for match in pattern.finditer(allowed_text):
            if not _is_negated(allowed_text, match.start()):
                return reason
        for match in pattern.finditer(checked_text):
            if not _is_negated(checked_text, match.start()):
                return reason

    for text in (allowed_text, checked_text):
        for match in _NETWORK_PATTERN.finditer(text):
            if not _is_negated(text, match.start()):
                if _allows_bounded_read_only_network(payload, text):
                    continue
                return "network access with unclear scope requires human approval before routing"
    return None


def _write_record(record: dict[str, Any]) -> None:
    directory = _task_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = _record_path(record["task_id"])
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_record(task_id: str) -> dict[str, Any]:
    path = _record_path(task_id)
    if not path.exists():
        raise MCPBridgeError(f"task {task_id} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_infrastructure_error_response(response: str) -> bool:
    """Return True for provider/auth failures that are not task reports."""
    return bool(_INFRASTRUCTURE_ERROR_RESPONSE_PATTERN.search(str(response or "")))


def submit_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and record a bridge task without executing it."""
    if not isinstance(payload, dict):
        raise MCPBridgeError("submit_task payload must be an object")

    task_id = "mcp_" + uuid.uuid4().hex
    created_at = _now_iso()
    refusal = _find_hard_refusal(payload)
    approval_required_reason = None if refusal else _find_approval_required(payload)
    status = "refused" if refusal else "approval_required" if approval_required_reason else "accepted"
    execution_state = "approval_required" if approval_required_reason else "not_executed"
    execution_message = (
        "Recorded for Hermes approval routing; no task runner was invoked."
        if approval_required_reason
        else "Recorded for future Hermes orchestration; no task runner was invoked."
    )
    record = {
        "task_id": task_id,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "project": _jsonable(payload.get("project")),
        "title": _jsonable(payload.get("title")),
        "mode": _jsonable(payload.get("mode")),
        "repo_scope": _jsonable(payload.get("repo_scope")),
        "worktree_scope": _jsonable(payload.get("worktree_scope")),
        "payload": _jsonable(payload),
        "refusal_reason": refusal,
        "approval_required_reason": approval_required_reason,
        "execution": {
            "state": execution_state,
            "message": execution_message,
        },
        "result": None,
    }
    _write_record(record)
    return {
        "ok": refusal is None,
        "task_id": task_id,
        "status": status,
        "refusal_reason": refusal,
        "approval_required_reason": approval_required_reason,
    }


def get_task_status(task_id: str) -> dict[str, Any]:
    record = _read_record(task_id)
    return {
        "ok": True,
        "task_id": record["task_id"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "refusal_reason": record.get("refusal_reason"),
        "approval_required_reason": record.get("approval_required_reason"),
        "execution": record.get("execution"),
    }


def get_task_result(task_id: str) -> dict[str, Any]:
    record = _read_record(task_id)
    return {
        "ok": True,
        "task_id": record["task_id"],
        "status": record["status"],
        "result": record.get("result"),
        "refusal_reason": record.get("refusal_reason"),
        "approval_required_reason": record.get("approval_required_reason"),
        "execution": record.get("execution"),
        "record": record,
    }


def update_task_lifecycle(
    task_id: str,
    *,
    status: str | None = None,
    execution_state: str | None = None,
    result: dict[str, Any] | None = None,
    execution_updates: dict[str, Any] | None = None,
) -> bool:
    """Update canonical task lifecycle fields without changing task metadata.

    This helper is intentionally internal to the bridge module and is not part
    of the MCP tool facade. It preserves the original contract/payload and any
    unknown record metadata by touching only status, result, execution, and
    updated_at. Refused records are terminal and cannot be promoted to
    completed by mirror flows.
    """
    record = _read_record(task_id)
    if record.get("status") == "refused" and status == "completed":
        return False

    updated = dict(record)
    if status is not None:
        updated["status"] = str(status)

    if result is not None:
        updated["result"] = _jsonable(result)

    if execution_state is not None or execution_updates:
        execution = updated.get("execution")
        if not isinstance(execution, dict):
            execution = {}
        else:
            execution = dict(execution)
        if execution_state is not None:
            execution["state"] = str(execution_state)
        if execution_updates:
            for key, value in execution_updates.items():
                execution[str(key)] = _jsonable(value)
        updated["execution"] = execution

    if updated == record:
        return True

    updated["updated_at"] = _now_iso()
    _write_record(updated)
    return True


def mirror_task_result(
    task_id: str,
    response: str,
    *,
    platform: str = "discord",
    source: str = "discord_gateway_final_response",
) -> bool:
    """Internally mirror a delivered gateway final response into a task record.

    This is deliberately not exposed as an MCP tool. It updates only the
    lifecycle/result fields so the submitted contract and metadata remain
    intact.
    """
    response = str(response or "").strip()
    if not response:
        return False
    if _is_infrastructure_error_response(response):
        return False

    result = {
        "source": str(source or "discord_gateway_final_response"),
        "platform": platform,
        "response": response,
    }
    return update_task_lifecycle(
        task_id,
        status="completed",
        execution_state="completed",
        result=result,
        execution_updates={
            "result_mirrored_by": "Hermes",
            "message": "Final Discord response mirrored to bridge record.",
        },
    )


def _matching_task_ids_for_code(recent: list[dict[str, Any]], code: str) -> list[str]:
    code = str(code or "").upper().strip()
    if not code:
        return []
    bridge_prefix = f"HERMES-BRIDGE-{code}"
    matches = [
        str(task.get("task_id") or "")
        for task in recent
        if str(task.get("title") or "").upper().startswith((code, bridge_prefix))
    ]
    return [task_id for task_id in matches if task_id]


def _resolve_unique_code_candidates(
    recent: list[dict[str, Any]], codes: list[str]
) -> tuple[str | None, bool]:
    candidates: set[str] = set()
    for code in codes:
        matches = _matching_task_ids_for_code(recent, code)
        if len(matches) == 1:
            candidates.add(matches[0])
        elif len(matches) > 1:
            return None, True

    if len(candidates) == 1:
        return next(iter(candidates)), False
    if len(candidates) > 1:
        return None, True
    return None, False


def resolve_task_id_from_text(text: str, *, limit: int = 100) -> str | None:
    """Resolve a bridge task reference from gateway-visible text.

    Exact ``mcp_...`` ids win. Otherwise, full work codes such as
    ``002DV-B2`` or ``HERMES-BRIDGE-002DW-A`` are tried before broad base
    codes such as ``002DV``. Code references resolve only when exactly one
    recent bridge task title starts with that code (or the canonical
    ``HERMES-BRIDGE-<code>`` prefix).
    """
    text = str(text or "")
    if not text.strip():
        return None

    exact_ids: list[str] = []
    for task_id in dict.fromkeys(_TASK_ID_PATTERN.findall(text)):
        try:
            _read_record(task_id)
            exact_ids.append(task_id)
        except MCPBridgeError:
            continue
    if len(exact_ids) == 1:
        return exact_ids[0]
    if len(exact_ids) > 1:
        return None

    full_codes = list(
        dict.fromkeys(match.group(1).upper() for match in _FULL_CODE_PATTERN.finditer(text))
    )
    broad_codes = list(
        dict.fromkeys(match.group(1).upper() for match in _LEADING_CODE_PATTERN.finditer(text))
    )
    if not full_codes and not broad_codes:
        return None

    try:
        recent = list_recent_tasks(limit=limit).get("tasks", [])
    except MCPBridgeError:
        return None

    specific_codes = [code for code in full_codes if "-" in code]
    resolved, ambiguous = _resolve_unique_code_candidates(recent, specific_codes)
    if resolved or ambiguous:
        return resolved

    resolved, ambiguous = _resolve_unique_code_candidates(recent, broad_codes)
    if resolved or ambiguous:
        return resolved
    return None


def list_recent_tasks(limit: int = 20) -> dict[str, Any]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise MCPBridgeError("limit must be an integer") from None
    if limit < 1 or limit > 100:
        raise MCPBridgeError("limit must be between 1 and 100")

    directory = _task_dir()
    records: list[dict[str, Any]] = []
    if directory.exists():
        for path in directory.glob("mcp_*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append(record)
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return {
        "ok": True,
        "tasks": [
            {
                "task_id": record.get("task_id"),
                "status": record.get("status"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "project": record.get("project"),
                "title": record.get("title"),
                "refusal_reason": record.get("refusal_reason"),
                "approval_required_reason": record.get("approval_required_reason"),
            }
            for record in records[:limit]
        ],
    }
