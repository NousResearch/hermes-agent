"""Shared predicates for writes scoped to one active Kanban lease."""

from __future__ import annotations

from typing import Any, Optional

LEASE_GUARD_UNSET = object()


def lease_guard_predicate(
    *,
    expected_run_id: Optional[int],
    expected_claim_lock: Any,
    expected_tenant: Any,
    expected_workspace_path: Any,
) -> tuple[str, tuple[Any, ...], bool]:
    """Return the SQL predicate, parameters, and whether the full guard is active.

    ``expected_run_id`` alone retains the existing run-scoped API. Supplying
    any newer identity field requires the whole lease identity, preventing a
    partial claim/workspace check from being mistaken for a complete binding.
    """
    identity = (expected_claim_lock, expected_tenant, expected_workspace_path)
    supplied = tuple(value is not LEASE_GUARD_UNSET for value in identity)
    if not any(supplied):
        if expected_run_id is None:
            return "", (), False
        try:
            run_id = int(expected_run_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected_run_id must be an integer") from exc
        return " AND current_run_id = ?", (run_id,), False

    if expected_run_id is None or not all(supplied):
        raise ValueError(
            "lease-scoped writes require expected_run_id, expected_claim_lock, "
            "expected_tenant, and expected_workspace_path"
        )
    try:
        run_id = int(expected_run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_run_id must be an integer") from exc
    claim_lock = str(expected_claim_lock).strip()
    tenant = None if expected_tenant is None else str(expected_tenant).strip()
    workspace_path = str(expected_workspace_path).strip()
    if not claim_lock or tenant == "" or not workspace_path:
        raise ValueError("lease-scoped identity guards cannot be empty")
    return (
        " AND current_run_id = ? AND claim_lock = ? "
        "AND tenant IS ? AND workspace_path = ?",
        (run_id, claim_lock, tenant, workspace_path),
        True,
    )
