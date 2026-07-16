"""Fail-closed handlers for the bounded planner toolset."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import re
from typing import Any

from pydantic import ValidationError

from .approval import ApprovalScope
from .models import PlanV1
from .schemas import ARTIFACT_TYPES, CONTEXT_QUERY_IDS
from .storage.artifacts import ArtifactSecurityError, ArtifactStore

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_PARAMETER_RULES = {
    "approval_status.v1": ({"approval_id"}, {"approval_id"}),
    "approvals_for_run.v1": ({"run_id"}, {"run_id", "limit"}),
}


class PlannerToolHandlers:
    """Execute planner operations only through injected bounded dependencies."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore | None,
        context_reader: Any | None,
        approval_service: Any | None,
    ):
        self._artifacts = artifact_store
        self._context_reader = context_reader
        self._approvals = approval_service

    async def context_read(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._context_reader is None:
            return {"ok": False, "error": "planner database runtime unavailable"}
        try:
            query_id, parameters = _validate_context_args(args)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid context query"}

        try:
            rows = await self._context_reader.read(query_id, parameters)
        except Exception:
            # Dependency errors can include SQL fragments or connection details.
            return {"ok": False, "error": "planner context read failed"}
        return {
            "ok": True,
            "query_id": query_id,
            "rows": rows,
            "row_count": len(rows),
        }

    async def artifact_write(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._artifacts is None:
            return {"ok": False, "error": "planner artifact runtime unavailable"}
        try:
            _require_exact_keys(args, {"artifact_type", "artifact_id", "payload"})
            artifact_type = args["artifact_type"]
            artifact_id = args["artifact_id"]
            payload = args["payload"]
            if artifact_type not in ARTIFACT_TYPES:
                raise ValueError("unsupported artifact type")
            if not isinstance(artifact_id, str) or not isinstance(payload, dict):
                raise TypeError("invalid artifact request")
        except (TypeError, ValidationError, ValueError):
            return {"ok": False, "error": "artifact write rejected"}
        try:
            receipt = await asyncio.to_thread(
                self._artifacts.persist,
                artifact_type,
                artifact_id,
                payload,
            )
        except (ArtifactSecurityError, FileExistsError, FileNotFoundError, OSError):
            return {"ok": False, "error": "artifact write failed"}

        return {"ok": True, "receipt": asdict(receipt)}

    async def approval_request(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._artifacts is None or self._approvals is None:
            return {"ok": False, "error": "planner database runtime unavailable"}
        try:
            _require_exact_keys(
                args,
                {"artifact_id", "expected_sha256", "scope"},
            )
            artifact_id = args["artifact_id"]
            expected_sha256 = args["expected_sha256"]
            if not isinstance(artifact_id, str):
                raise TypeError("invalid artifact identifier")
            if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
                expected_sha256
            ):
                raise ValueError("invalid expected hash")
            scope_value = args["scope"]
            scope = (
                scope_value
                if isinstance(scope_value, ApprovalScope)
                else ApprovalScope.model_validate(scope_value)
            )
            plan_bytes = await asyncio.to_thread(
                self._artifacts.read_verified,
                "plan",
                artifact_id,
                expected_sha256=expected_sha256,
            )
            plan = PlanV1.model_validate_json(plan_bytes, strict=True)
        except (
            ArtifactSecurityError,
            FileNotFoundError,
            OSError,
            TypeError,
            ValidationError,
            ValueError,
        ):
            return {"ok": False, "error": "approval request rejected"}

        if not _scope_matches_plan(scope, plan, expected_sha256):
            return {"ok": False, "error": "approval scope does not match plan"}

        summary = _fixed_approval_summary(scope)
        try:
            record = await self._approvals.request(scope, summary)
        except Exception:
            # Approval backend failures must not expose DSNs or SQL to the model.
            return {"ok": False, "error": "approval request failed"}
        if record is None:
            return {"ok": True, "approved": False}
        return {
            "ok": True,
            "approved": True,
            "approval": {
                "approval_id": record.approval_id,
                "scope_sha256": record.scope_sha256,
                "plan_sha256": record.plan_sha256,
                "verdict": record.verdict.value,
                "surface": record.surface,
                "created_at": record.created_at.isoformat(),
                "expires_at": record.expires_at.isoformat(),
                "max_executions": record.max_executions,
                "consumed_count": record.consumed_count,
            },
        }


def _require_exact_keys(args: Any, required: set[str]) -> None:
    if not isinstance(args, dict) or set(args) != required:
        raise ValueError("request fields do not match the closed schema")


def _validate_context_args(args: Any) -> tuple[str, dict[str, Any]]:
    _require_exact_keys(args, {"query_id", "parameters"})
    query_id = args["query_id"]
    parameters = args["parameters"]
    if query_id not in CONTEXT_QUERY_IDS or not isinstance(parameters, dict):
        raise ValueError("unknown context query")
    required, allowed = _CONTEXT_PARAMETER_RULES[query_id]
    if not required.issubset(parameters) or not set(parameters).issubset(allowed):
        raise ValueError("invalid query parameters")
    for key, value in parameters.items():
        if key == "limit":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 100
            ):
                raise ValueError("invalid row limit")
        elif not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError("invalid query identifier")
    return query_id, dict(parameters)


def _scope_matches_plan(
    scope: ApprovalScope,
    plan: PlanV1,
    expected_sha256: str,
) -> bool:
    if scope.expires_at <= datetime.now(UTC):
        return False
    if scope.plan_sha256 != expected_sha256 or scope.run_id != plan.run_id:
        return False
    if scope.budgets != plan.budgets:
        return False
    if scope.max_executions > plan.budgets.max_executions:
        return False
    grant = next(
        (
            candidate
            for candidate in plan.capabilities
            if candidate.capability is scope.capability
        ),
        None,
    )
    if grant is None:
        return False
    return tuple(scope.input_hashes) == tuple(
        sorted(set(grant.input_hashes))
    ) and tuple(scope.external_rights) == tuple(
        sorted(set(grant.external_rights), key=lambda item: item.value)
    )


def _fixed_approval_summary(scope: ApprovalScope) -> str:
    budgets = scope.budgets
    return (
        "Research protocol approval request\n"
        f"run_id={scope.run_id}\n"
        f"capability={scope.capability.value}\n"
        f"plan_sha256={scope.plan_sha256}\n"
        f"scope_sha256={scope.scope_sha256()}\n"
        f"expires_at={scope.expires_at.isoformat()}\n"
        f"max_executions={scope.max_executions}\n"
        "budgets="
        f"duration:{budgets.max_duration_seconds}s,"
        f"records:{budgets.max_records},"
        f"bytes:{budgets.max_bytes},"
        f"external_calls:{budgets.max_external_calls}\n"
        f"scope_json={scope.canonical_json()}"
    )


_runtime: PlannerToolHandlers | None = None
_artifact_available = False
_context_available = False
_approval_available = False


def configure_planner_runtime(
    runtime: PlannerToolHandlers | None,
    *,
    artifact_available: bool = False,
    context_available: bool = False,
    approval_available: bool = False,
) -> None:
    """Install or clear an explicitly constructed local planner runtime."""
    global _approval_available, _artifact_available, _context_available, _runtime
    _runtime = runtime
    _artifact_available = runtime is not None and artifact_available is True
    _context_available = runtime is not None and context_available is True
    _approval_available = runtime is not None and approval_available is True


async def _dispatch(method: str, args: dict[str, Any]) -> dict[str, Any]:
    if _runtime is None:
        return {
            "ok": False,
            "error": "research protocol planner runtime is not configured",
        }
    return await getattr(_runtime, method)(args)


async def plan_context_read(**kwargs: Any) -> dict[str, Any]:
    return await _dispatch("context_read", kwargs)


async def plan_artifact_write(**kwargs: Any) -> dict[str, Any]:
    return await _dispatch("artifact_write", kwargs)


async def plan_approval_request(**kwargs: Any) -> dict[str, Any]:
    return await _dispatch("approval_request", kwargs)


def check_planner_artifacts() -> bool:
    """Expose artifact writes only after an explicit safe root is configured."""
    return _runtime is not None and _artifact_available


def check_planner_context() -> bool:
    """Expose context reads only when the fixed reader authority exists."""
    return _runtime is not None and _context_available


def check_planner_approvals() -> bool:
    """Expose approvals only when both artifact and writer authority exist."""
    return _runtime is not None and _artifact_available and _approval_available


__all__ = [
    "PlannerToolHandlers",
    "check_planner_approvals",
    "check_planner_artifacts",
    "check_planner_context",
    "configure_planner_runtime",
    "plan_approval_request",
    "plan_artifact_write",
    "plan_context_read",
]
