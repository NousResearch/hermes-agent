"""Turn-scoped usage ledger stored through session_model_usage.task.

The existing task dimension is deliberately used for zero-usage metadata rows,
so this feature needs no SQLite schema change and does not mutate the real
per-model token/cost rows.
"""

from __future__ import annotations

import ast
import base64
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agent.usage_pricing import CanonicalUsage

WORKFLOW_TASK_PREFIX = "workflow-ledger:v1:"
_TOKEN_FIELDS = (
    ("input", "input_tokens"),
    ("output", "output_tokens"),
    ("cache_read", "cache_read_tokens"),
    ("cache_write", "cache_write_tokens"),
    ("reasoning", "reasoning_tokens"),
)


def encode_workflow_task(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return WORKFLOW_TASK_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_workflow_task(task: Any) -> dict[str, Any] | None:
    if not isinstance(task, str) or not task.startswith(WORKFLOW_TASK_PREFIX):
        return None
    encoded = task[len(WORKFLOW_TASK_PREFIX):]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("kind") == "workflow" else None


def model_decision_reason(model: str | None, *, explicit_override: bool) -> str:
    """Explain why this turn kept its active model instead of silently rerouting."""
    model_name = str(model or "UNKNOWN")
    if explicit_override:
        return f"user-selected model {model_name} retained"
    return (
        f"active session model {model_name} retained; "
        "automatic per-task routing unavailable"
    )


def normalize_reasoning_level(
    request_kwargs: dict[str, Any], *, fallback: Any = None
) -> str:
    """Extract the provider's configured reasoning level without object reprs."""
    value = request_kwargs.get("reasoning_effort") or request_kwargs.get("reasoning") or fallback
    if isinstance(value, str) and value.lstrip().startswith("{"):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass
    if isinstance(value, dict):
        value = value.get("effort") or value.get("level")
    return str(value or "UNKNOWN")


@dataclass
class _RouteUsage:
    model: str
    provider: str
    reasoning_level: str
    request_count: int = 0
    retries: int = 0
    runtime_seconds: float = 0.0
    attempts_by_request: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    responses: int = 0
    usage_responses: int = 0
    tokens: dict[str, int] = field(default_factory=lambda: {name: 0 for name, _ in _TOKEN_FIELDS})
    estimated_cost: Decimal = Decimal("0")
    estimated_cost_known: bool = True
    cost_statuses: set[str] = field(default_factory=set)
    actual_cost: Decimal = Decimal("0")
    actual_cost_known: bool = True

    def add_attempt(self, request_id: str, duration_seconds: float) -> None:
        previous = self.attempts_by_request[request_id]
        self.attempts_by_request[request_id] = previous + 1
        self.request_count += 1
        if previous:
            self.retries += 1
        if duration_seconds >= 0:
            self.runtime_seconds += duration_seconds

    def add_response(
        self,
        usage: CanonicalUsage | None,
        estimated_cost_usd: float | Decimal | None,
        actual_provider_cost_usd: float | Decimal | None,
        cost_status: str | None,
    ) -> None:
        self.responses += 1
        if usage is None:
            self.estimated_cost_known = False
            self.actual_cost_known = False
            return
        self.usage_responses += 1
        for label, attribute in _TOKEN_FIELDS:
            self.tokens[label] += int(getattr(usage, attribute) or 0)
        if estimated_cost_usd is None:
            self.estimated_cost_known = False
        else:
            self.estimated_cost += Decimal(str(estimated_cost_usd))
            self.cost_statuses.add(cost_status or "estimated")
        if actual_provider_cost_usd is None:
            self.actual_cost_known = False
        else:
            self.actual_cost += Decimal(str(actual_provider_cost_usd))

    def as_payload(self) -> dict[str, Any]:
        retries_or_unreported = self.retries > 0 or self.responses < len(self.attempts_by_request)
        usage_complete = (
            not retries_or_unreported
            and self.usage_responses == len(self.attempts_by_request)
        )
        cost_complete = usage_complete and self.estimated_cost_known
        actual_complete = usage_complete and self.actual_cost_known
        cost_status = (
            next(iter(self.cost_statuses))
            if cost_complete and len(self.cost_statuses) == 1
            else "UNKNOWN"
        )
        return {
            "model": self.model or "UNKNOWN",
            "provider": self.provider or "UNKNOWN",
            "reasoning_level": self.reasoning_level or "UNKNOWN",
            "requests": self.request_count,
            "retries": self.retries,
            "runtime_seconds": round(self.runtime_seconds, 6),
            "tokens": {
                key: self.tokens[key] if usage_complete else "UNKNOWN"
                for key, _ in _TOKEN_FIELDS
            },
            "estimated_cost_usd": (
                str(self.estimated_cost) if cost_complete else "UNKNOWN"
            ),
            "estimated_cost_status": cost_status,
            "actual_provider_cost_usd": (
                str(self.actual_cost) if actual_complete else "UNKNOWN"
            ),
        }


class WorkflowUsageTracker:
    """Accumulate request/usage data for exactly one conversation turn."""

    def __init__(self, workflow_id: str, *, started_at: float | None = None):
        self.workflow_id = str(workflow_id or "UNKNOWN")
        self.started_at = float(started_at if started_at is not None else time.time())
        self._routes: dict[tuple[str, str, str], _RouteUsage] = {}
        self._request_routes: dict[str, tuple[str, str, str]] = {}
        self._finished: dict[str, Any] | None = None
        self._escalation_reason: str | None = None

    def _route(self, model: str, provider: str, reasoning_level: str) -> _RouteUsage:
        key = (str(model or "UNKNOWN"), str(provider or "UNKNOWN"), str(reasoning_level or "UNKNOWN"))
        if key not in self._routes:
            self._routes[key] = _RouteUsage(*key)
        return self._routes[key]

    def record_attempt(
        self,
        *,
        request_id: str,
        model: str,
        provider: str,
        reasoning_level: str | None,
        duration_seconds: float,
    ) -> None:
        route_key = (str(model or "UNKNOWN"), str(provider or "UNKNOWN"), str(reasoning_level or "UNKNOWN"))
        self._request_routes[request_id] = route_key
        self._route(*route_key).add_attempt(request_id, float(duration_seconds))

    def record_response(
        self,
        *,
        request_id: str,
        usage: CanonicalUsage | None,
        estimated_cost_usd: float | Decimal | None,
        actual_provider_cost_usd: float | Decimal | None = None,
        cost_status: str | None = None,
    ) -> None:
        route_key = self._request_routes.get(request_id)
        if route_key is None:
            return
        self._route(*route_key).add_response(
            usage, estimated_cost_usd, actual_provider_cost_usd, cost_status
        )

    def note_model_escalation(self, reason: str) -> None:
        """Store an observed, reasoned model/provider route change."""
        if reason and reason != "NONE":
            self._escalation_reason = str(reason)

    def finish(
        self,
        *,
        outcome: str,
        ended_at: float | None = None,
        decision_reason: str = "UNKNOWN",
        escalation_reason: str | None = None,
        result_quality: str = "UNKNOWN",
        rework: str = "UNKNOWN",
    ) -> None:
        self._finished = {
            "kind": "workflow",
            "workflow_id": self.workflow_id,
            "started_at": self.started_at,
            "ended_at": float(ended_at if ended_at is not None else time.time()),
            "outcome": outcome or "UNKNOWN",
            "decision_reason": decision_reason or "UNKNOWN",
            "escalation_reason": escalation_reason or self._escalation_reason or "NONE",
            "result_quality": result_quality or "UNKNOWN",
            "rework": rework or "UNKNOWN",
        }

    def persist(self, db: Any, session_id: str | None) -> None:
        if not self._finished or not db or not session_id:
            return
        for route in self._routes.values():
            payload = {**self._finished, "route": route.as_payload()}
            db.record_auxiliary_usage(
                session_id,
                encode_workflow_task(payload),
                model=route.model,
                billing_provider=route.provider,
                api_call_count=0,
            )


def read_workflow_rows(rows: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    """Decode and group workflow ledger rows, ignoring malformed/duplicate rows."""
    grouped: dict[str, dict[str, Any]] = {}
    route_keys: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        payload = decode_workflow_task(row.get("task"))
        if not payload:
            continue
        workflow_id = str(payload.get("workflow_id") or "UNKNOWN")
        item = grouped.setdefault(workflow_id, {
            "workflow_id": workflow_id,
            "started_at": payload.get("started_at"),
            "ended_at": payload.get("ended_at"),
            "outcome": payload.get("outcome") or "UNKNOWN",
            "decision_reason": payload.get("decision_reason") or "UNKNOWN",
            "escalation_reason": payload.get("escalation_reason") or "UNKNOWN",
            "result_quality": payload.get("result_quality") or "UNKNOWN",
            "rework": payload.get("rework") or "UNKNOWN",
            "routes": [],
        })
        route = payload.get("route")
        if not isinstance(route, dict):
            continue
        route = {
            **route,
            "reasoning_level": normalize_reasoning_level(
                {}, fallback=route.get("reasoning_level")
            ),
        }
        route_key = (
            str(route.get("model") or "UNKNOWN"),
            str(route.get("provider") or "UNKNOWN"),
            str(route.get("reasoning_level") or "UNKNOWN"),
        )
        if route_key not in route_keys[workflow_id]:
            item["routes"].append(route)
            route_keys[workflow_id].add(route_key)
    result = []
    for item in grouped.values():
        item["routes"].sort(key=lambda route: (route["model"], route["provider"]))
        item["requests"] = sum(int(route.get("requests") or 0) for route in item["routes"])
        retries = [route.get("retries") for route in item["routes"]]
        item["retries"] = "UNKNOWN" if "UNKNOWN" in retries else sum(int(value or 0) for value in retries)
        item["runtime_seconds"] = round(
            sum(float(route.get("runtime_seconds") or 0) for route in item["routes"]), 6
        )
        started_at = item.get("started_at")
        ended_at = item.get("ended_at")
        item["wall_runtime_seconds"] = (
            round(max(0.0, float(ended_at) - float(started_at)), 6)
            if started_at is not None and ended_at is not None
            else "UNKNOWN"
        )
        result.append(item)
    result.sort(key=lambda item: float(item.get("ended_at") or 0), reverse=True)
    return result[:limit]
