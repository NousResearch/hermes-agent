"""Local inert Goal Contract persistence for Mission Control.

Goal Contract source references are user-entered opaque strings. This module
stores them as text only; it does not resolve paths, expand ``~``, fetch URLs,
hash, preview, stat, or parse referenced artifacts.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_cli.mission_control import redact_text, redact_value


GOAL_CONTRACT_STATUSES = {"draft", "active", "archived"}
MAX_TEXT_CHARS = 100_000
MAX_LIST_ITEMS = 100
MAX_LIST_ITEM_CHARS = 4_000
_LOCK = threading.RLock()


class GoalContractError(ValueError):
    """Raised for invalid Goal Contract requests."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "goal-contracts"


def audit_path() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "state" / "mission-control" / "goal-contracts-audit.jsonl"


def _contract_path(contract_id: str) -> Path:
    if not re.fullmatch(r"contract_[0-9TZ]+_[a-f0-9]{12}", contract_id):
        raise GoalContractError("Invalid contract id")
    return state_dir() / f"{contract_id}.json"


def _new_contract_id(created_at: str) -> str:
    stamp = re.sub(r"[^0-9TZ]", "", created_at.replace("+00:00", "Z"))
    return f"contract_{stamp}_{secrets.token_hex(6)}"


def _bounded_text(value: Any, *, field: str, required: bool = False) -> str:
    if value is None:
        if required:
            raise GoalContractError(f"Missing required field: {field}")
        return ""
    text = str(value)
    if required and not text.strip():
        raise GoalContractError(f"Missing required field: {field}")
    return text[:MAX_TEXT_CHARS]


def _string_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise GoalContractError(f"{field} must be a list of strings")
    items: list[str] = []
    for item in value[:MAX_LIST_ITEMS]:
        if not isinstance(item, str):
            raise GoalContractError(f"{field} must be a list of strings")
        if item:
            items.append(item[:MAX_LIST_ITEM_CHARS])
    return items


def _status(value: Any, *, default: str = "draft") -> str:
    status = str(value or default)
    if status not in GOAL_CONTRACT_STATUSES:
        raise GoalContractError("status must be one of: draft, active, archived")
    return status


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(redact_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _append_audit(event: str, contract: dict[str, Any] | None = None, *, result: str = "ok") -> None:
    record = {
        "timestamp": _now_iso(),
        "event": event,
        "actor": "dashboard",
        "surface": "dashboard",
        "contract_id": (contract or {}).get("id"),
        "status": (contract or {}).get("status"),
        "trusted_for_execution": False,
        "inert_context_only": True,
        "result": redact_text(result),
    }
    path = audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(redact_value(record), sort_keys=True) + "\n")


def _summary(contract: dict[str, Any]) -> dict[str, Any]:
    return redact_value(
        {
            "id": contract["id"],
            "title": contract["title"],
            "objective": contract.get("objective", ""),
            "status": contract["status"],
            "success_criteria_count": len(contract.get("success_criteria") or []),
            "constraint_count": len(contract.get("constraints") or []),
            "source_ref_count": len(contract.get("source_refs") or []),
            "vocabulary_version": "g1",
            "linked_mission_brief_id": contract.get("linked_mission_brief_id"),
            "created_at": contract["created_at"],
            "updated_at": contract["updated_at"],
            "archived_at": contract.get("archived_at"),
            "trusted_for_execution": False,
            "inert_context_only": True,
        }
    )


def _read_contract_unlocked(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise GoalContractError("Goal Contract file is invalid")
    return data


def create_contract(data: dict[str, Any]) -> dict[str, Any]:
    created_at = _now_iso()
    status = _status(data.get("status"))
    archived_at = created_at if status == "archived" else None
    contract = {
        "id": _new_contract_id(created_at),
        "title": redact_text(_bounded_text(data.get("title"), field="title", required=True)),
        "objective": redact_text(_bounded_text(data.get("objective"), field="objective", required=True)),
        "status": status,
        "success_criteria": redact_value(_string_list(data.get("success_criteria"), field="success_criteria")),
        "constraints": redact_value(_string_list(data.get("constraints"), field="constraints")),
        "source_refs": redact_value(_string_list(data.get("source_refs"), field="source_refs")),
        "vocabulary_version": "g1",
        "linked_mission_brief_id": _bounded_text(data.get("linked_mission_brief_id"), field="linked_mission_brief_id"),
        "author": "dashboard",
        "created_at": created_at,
        "updated_at": created_at,
        "archived_at": archived_at,
        "trusted_for_execution": False,
        "inert_context_only": True,
    }
    if isinstance(data.get("metadata"), dict):
        contract["metadata"] = data["metadata"]
    with _LOCK:
        path = _contract_path(contract["id"])
        _atomic_write_json(path, contract)
        _append_audit("goal_contract_created", contract)
    return redact_value(contract)


def list_contracts(*, include_archived: bool = False) -> dict[str, Any]:
    with _LOCK:
        directory = state_dir()
        directory.mkdir(parents=True, exist_ok=True)
        contracts: list[dict[str, Any]] = []
        for path in sorted(directory.glob("contract_*.json")):
            try:
                contract = _read_contract_unlocked(path)
            except Exception:
                continue
            if include_archived or contract.get("status") != "archived":
                contracts.append(contract)
    contracts.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return {"items": [_summary(contract) for contract in contracts], "warnings": []}


def get_contract(contract_id: str) -> dict[str, Any]:
    with _LOCK:
        path = _contract_path(contract_id)
        try:
            contract = _read_contract_unlocked(path)
        except FileNotFoundError:
            raise
    return {"contract": redact_value(contract)}


def update_contract(contract_id: str, data: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        path = _contract_path(contract_id)
        try:
            contract = _read_contract_unlocked(path)
        except FileNotFoundError:
            raise
        if "title" in data:
            contract["title"] = redact_text(_bounded_text(data.get("title"), field="title", required=True))
        if "objective" in data:
            contract["objective"] = redact_text(
                _bounded_text(data.get("objective"), field="objective", required=True)
            )
        if "status" in data:
            status = _status(data.get("status"))
            contract["status"] = status
            contract["archived_at"] = _now_iso() if status == "archived" else None
        if "success_criteria" in data:
            contract["success_criteria"] = redact_value(
                _string_list(data.get("success_criteria"), field="success_criteria")
            )
        if "constraints" in data:
            contract["constraints"] = redact_value(_string_list(data.get("constraints"), field="constraints"))
        if "source_refs" in data:
            contract["source_refs"] = redact_value(_string_list(data.get("source_refs"), field="source_refs"))
        if "linked_mission_brief_id" in data:
            contract["linked_mission_brief_id"] = _bounded_text(
                data.get("linked_mission_brief_id"),
                field="linked_mission_brief_id",
            )
        if "metadata" in data and isinstance(data.get("metadata"), dict):
            contract["metadata"] = data["metadata"]
        contract["vocabulary_version"] = "g1"
        contract["author"] = "dashboard"
        contract["trusted_for_execution"] = False
        contract["inert_context_only"] = True
        contract["updated_at"] = _now_iso()
        _atomic_write_json(path, contract)
        _append_audit("goal_contract_updated", contract)
    return {"contract": redact_value(contract)}


def archive_contract(contract_id: str) -> dict[str, Any]:
    with _LOCK:
        path = _contract_path(contract_id)
        try:
            contract = _read_contract_unlocked(path)
        except FileNotFoundError:
            raise
        now = _now_iso()
        contract["status"] = "archived"
        contract["archived_at"] = now
        contract["updated_at"] = now
        contract["vocabulary_version"] = "g1"
        contract["author"] = "dashboard"
        contract["trusted_for_execution"] = False
        contract["inert_context_only"] = True
        _atomic_write_json(path, contract)
        _append_audit("goal_contract_archived", contract)
    return {"contract": redact_value(contract)}
