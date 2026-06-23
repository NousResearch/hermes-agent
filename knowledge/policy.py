"""Configurable routing policy for durable knowledge writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from knowledge.types import KnowledgeDestination, RouteAction, RouteDecision


DEFAULT_STATIC_TYPES = frozenset({
    "runbook",
    "architecture",
    "decision",
    "postmortem",
    "research",
    "project_doc",
    "api_doc",
    "integration_doc",
    "deployment",
    "troubleshooting",
    "deep_summary",
    "knowledge",
})
DEFAULT_DYNAMIC_TYPES = frozenset({
    "preference",
    "user_preference",
    "fact",
    "lesson",
    "entity",
    "memory",
    "correction",
})
DEFAULT_EPHEMERAL_TYPES = frozenset({
    "task_log",
    "session_log",
    "transcript",
    "scratch",
    "temporary",
    "progress",
    "build_output",
    "test_output",
    "verification_marker",
})


@dataclass(frozen=True)
class RoutePolicy:
    static_types: frozenset[str] = DEFAULT_STATIC_TYPES
    dynamic_types: frozenset[str] = DEFAULT_DYNAMIC_TYPES
    ephemeral_types: frozenset[str] = DEFAULT_EPHEMERAL_TYPES
    default_static_backend: str = "siyuan"
    default_dynamic_backend: str = "hindsight"
    unknown_policy: str = "dry_run"
    duplicate_policy: str = "update_existing"
    extra_static_types: frozenset[str] = field(default_factory=frozenset)
    extra_dynamic_types: frozenset[str] = field(default_factory=frozenset)
    extra_ephemeral_types: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "RoutePolicy":
        router_cfg = ((config or {}).get("knowledge") or {}).get("router") or {}
        def _set(name: str) -> frozenset[str]:
            value = router_cfg.get(name) or []
            if isinstance(value, str):
                value = [p.strip() for p in value.split(",")]
            return frozenset(str(v).strip().lower().replace("-", "_") for v in value if str(v).strip())
        return cls(
            default_static_backend=router_cfg.get("default_static_backend") or "siyuan",
            default_dynamic_backend=router_cfg.get("default_dynamic_backend") or "hindsight",
            unknown_policy=router_cfg.get("unknown_policy") or "dry_run",
            duplicate_policy=router_cfg.get("duplicate_policy") or "update_existing",
            extra_static_types=_set("extra_static_types"),
            extra_dynamic_types=_set("extra_dynamic_types"),
            extra_ephemeral_types=_set("extra_ephemeral_types"),
        )

    def decide(self, content_type: str) -> RouteDecision:
        kind = (content_type or "").strip().lower().replace("-", "_")
        if kind in self.static_types or kind in self.extra_static_types:
            return RouteDecision(
                destination=KnowledgeDestination.STATIC_KNOWLEDGE,
                action=RouteAction.WRITE_DOCUMENT,
                reason="static_reference",
                requires_title=True,
                snapshot=True,
                content_type=kind,
                backend=self.default_static_backend,
            )
        if kind in self.dynamic_types or kind in self.extra_dynamic_types:
            return RouteDecision(
                destination=KnowledgeDestination.DYNAMIC_MEMORY,
                action=RouteAction.RETAIN_MEMORY,
                reason="dynamic_memory",
                content_type=kind,
                backend=self.default_dynamic_backend,
            )
        if kind in self.ephemeral_types or kind in self.extra_ephemeral_types:
            return RouteDecision(
                destination=KnowledgeDestination.NONE,
                action=RouteAction.SKIP,
                reason="ephemeral_noise",
                content_type=kind,
            )
        return RouteDecision(
            destination=KnowledgeDestination.REVIEW,
            action=RouteAction.DRY_RUN_ONLY,
            reason="unknown_content_type",
            content_type=kind,
        )
