"""Paperclip/Fleet runtime context helpers for Hermes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
_INTERNAL_CRON_HINTS = {
    "backup",
    "cleanup",
    "diagnostic",
    "doctor",
    "health",
    "internal",
    "log rotation",
    "maintenance",
    "rotate",
}


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in _TRUE_VALUES


def paperclip_context_from_env() -> dict[str, str | None]:
    """Return Paperclip/Fleet context injected by the Fleet-issued env pack."""
    return {
        "fleet_runtime_id": os.getenv("FLEET_RUNTIME_ID") or os.getenv("CONTAINER_ID"),
        "paperclip_company_id": os.getenv("PAPERCLIP_COMPANY_ID"),
        "paperclip_node_id": os.getenv("PAPERCLIP_NODE_ID") or os.getenv("PAPERCLIP_AGENT_ID"),
        "paperclip_role": os.getenv("PAPERCLIP_ROLE"),
        "paperclip_title": os.getenv("PAPERCLIP_TITLE"),
        "paperclip_reports_to": os.getenv("PAPERCLIP_REPORTS_TO_AGENT_ID"),
    }


def paperclip_managed_routines_enabled() -> bool:
    """Return True when Paperclip/Fleet owns client-visible routine authority."""
    if _truthy_env("PAPERCLIP_MANAGED_ROUTINES"):
        return True
    return bool(os.getenv("PAPERCLIP_COMPANY_ID") and _truthy_env("PAPERCLIP_ROUTINES_MANAGED"))


def is_internal_maintenance_cron(*, name: str | None, prompt: str | None, deliver: str | None) -> bool:
    """Classify local cron creation that is still safe under Paperclip-managed authority."""
    text = f"{name or ''} {prompt or ''}".lower()
    if deliver and deliver not in {"local", "none"}:
        return False
    return any(hint in text for hint in _INTERNAL_CRON_HINTS)


def paperclip_cron_create_block_reason(
    *,
    name: str | None,
    prompt: str | None,
    deliver: str | None,
) -> str:
    """Return a Paperclip reroute reason for blocked cron creation, or empty string."""
    if not paperclip_managed_routines_enabled():
        return ""
    if _truthy_env("HERMES_ALLOW_CLIENT_VISIBLE_LOCAL_CRON"):
        return ""
    if is_internal_maintenance_cron(name=name, prompt=prompt, deliver=deliver):
        return ""
    context = paperclip_context_from_env()
    role = context.get("paperclip_title") or context.get("paperclip_role") or "runtime"
    return (
        "Paperclip/Fleet owns client-visible routine authority for this Hermes "
        f"{role}. Create or update the routine in Paperclip and mirror it through "
        "Fleet Manager instead of creating a local cron job."
    )


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def _routine_keys_from_contracts(document: Any) -> set[str]:
    if not isinstance(document, dict):
        return set()
    routines = document.get("routines")
    if not isinstance(routines, list):
        return set()
    keys = set()
    for routine in routines:
        if not isinstance(routine, dict):
            continue
        key = str(routine.get("routineKey") or routine.get("routine_key") or routine.get("id") or "").strip()
        if key:
            keys.add(key)
    return keys


def paperclip_runtime_diagnostics(
    *,
    hermes_home: Path,
    contracts_path: Path | None = None,
) -> dict[str, Any]:
    """Build Paperclip/Fleet diagnostics without mutating runtime state."""
    contracts = contracts_path or hermes_home / "routine-contracts.yaml"
    contract_body = contracts.read_bytes() if contracts.exists() else None
    contract_document: Any = None
    if contract_body is not None:
        try:
            import yaml  # type: ignore[import-untyped]

            contract_document = yaml.safe_load(contract_body.decode("utf-8")) or {}
        except Exception as exc:
            contract_document = {"error": str(exc)}
    routine_keys = _routine_keys_from_contracts(contract_document)
    jobs = _load_json(hermes_home / "cron" / "jobs.json")
    cron_jobs = jobs if isinstance(jobs, list) else []
    unmanaged_cron_jobs = []
    for job in cron_jobs:
        if not isinstance(job, dict):
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        key = str(metadata.get("paperclip_routine_key") or job.get("paperclip_routine_key") or "").strip()
        if key and key in routine_keys:
            continue
        unmanaged_cron_jobs.append(
            {
                "id": job.get("id"),
                "name": job.get("name"),
                "enabled": job.get("enabled", True),
                "paperclip_routine_key": key or None,
            }
        )

    return {
        "paperclip_managed_routines": paperclip_managed_routines_enabled(),
        "context": paperclip_context_from_env(),
        "routine_contract": {
            "path": str(contracts),
            "exists": contract_body is not None,
            "sha256": hashlib.sha256(contract_body).hexdigest() if contract_body is not None else None,
            "routine_keys": sorted(routine_keys),
        },
        "local_cron": {
            "jobs_path": str(hermes_home / "cron" / "jobs.json"),
            "job_count": len(cron_jobs),
            "not_backed_by_paperclip_routines": unmanaged_cron_jobs,
        },
    }
