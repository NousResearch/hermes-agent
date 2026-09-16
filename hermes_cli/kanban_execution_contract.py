"""Validation for immutable, task-bound terminal execution authority."""
from __future__ import annotations

from typing import Any

_ARGOCD_OPERATIONS = frozenset({"sync", "rollback", "terminate-op"})


def validate_execution_contract(value: Any) -> dict[str, Any] | None:
    """Validate a persisted contract; no contract preserves legacy behaviour.

    Contracts are intentionally declarative and narrow.  The initial core type
    authorizes a named ArgoCD application operation on one exact API endpoint.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("execution_contract must be an object")
    if set(value) != {"kind", "targets"} or value.get("kind") != "argocd":
        raise ValueError("execution_contract must be {kind: 'argocd', targets: [...]}")
    targets = value.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("execution_contract.targets must be a non-empty list")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"server", "application", "operation"}:
            raise ValueError("each execution_contract target requires server, application, operation")
        server = target["server"]
        application = target["application"]
        operation = target["operation"]
        if not all(isinstance(item, str) and item.strip() == item and item for item in (server, application, operation)):
            raise ValueError("execution_contract target values must be non-blank strings without surrounding whitespace")
        if operation not in _ARGOCD_OPERATIONS:
            raise ValueError(f"execution_contract operation must be one of {sorted(_ARGOCD_OPERATIONS)}")
        key = (server, application, operation)
        if key not in seen:
            normalized.append({"server": server, "application": application, "operation": operation})
            seen.add(key)
    return {"kind": "argocd", "targets": normalized}
