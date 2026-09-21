"""Fail-closed resolver for benchmark-backed, profile-specific model routes.

This module deliberately does not pick winners.  It validates and consumes a
separately produced benchmark artifact so runtime configuration cannot silently
inherit a convenient global model for an unmeasured task surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REQUIRED_SURFACES = frozenset(
    {
        "curator",
        "reference_aggregator",
        "review",
        "independent_review",
        "review_comment",
        "primary",
        "retry",
        "fallback",
        "circuit_breaker_probe",
        "local_privacy_fallback",
        "context_admission",
        "context_packing",
        "compaction",
        "compression",
        "title_generation",
        "conversation_summary",
        "mcp_discovery",
        "tool_selection",
        "tool_result_interpretation",
        "approval",
        "approval_response",
        "skills_hub",
        "skill_selection",
        "skill_execution",
        "vision",
        "coding",
        "ci_audit",
        "ci_repair",
        "pr_repair",
        "merge_maintenance",
        "auto_merge",
        "cron",
        "kanban",
        "orchestration",
        "task_assignment",
        "new_conversation",
    }
)

_LOCAL_DESTINATIONS = frozenset({"local", "loopback"})
_MIN_CONTEXT_WINDOW = 65_536


class RouteCompilationError(ValueError):
    """The routing artifact is incomplete, unsafe, or internally invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    provider: str
    model: str
    reasoning_effort: str | None
    context_window: int
    max_output_tokens: int
    no_output_timeout_seconds: float
    total_timeout_seconds: float
    concurrency_weight: int
    privacy_class: str
    destination_class: str
    artifact_digest: str
    policy_digest: str
    benchmark_case_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "no_output_timeout_seconds": self.no_output_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "concurrency_weight": self.concurrency_weight,
            "privacy_class": self.privacy_class,
            "destination_class": self.destination_class,
            "artifact_digest": self.artifact_digest,
            "policy_digest": self.policy_digest,
            "benchmark_case_ids": list(self.benchmark_case_ids),
        }


@dataclass(frozen=True, slots=True)
class CompiledRoute:
    primary: ResolvedRoute
    privacy_fallback: ResolvedRoute

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.as_dict(),
            "privacy_fallback": self.privacy_fallback.as_dict(),
        }


CompiledRouteTable = dict[str, dict[str, CompiledRoute]]


def _digest(value: object, *, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise RouteCompilationError(f"{field} must be a SHA-256 hex digest")
    return text


def _positive_int(row: Mapping[str, Any], field: str, *, minimum: int = 1) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteCompilationError(f"{field} must be an integer >= {minimum}")
    return value


def _positive_float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RouteCompilationError(f"{field} must be positive")
    return float(value)


def _parse_route(
    row: object,
    *,
    artifact_digest: str,
    policy_digest: str,
    location: str,
) -> ResolvedRoute:
    if not isinstance(row, Mapping):
        raise RouteCompilationError(f"{location} must be a route mapping")
    provider = str(row.get("provider") or "").strip()
    model = str(row.get("model") or "").strip()
    if not provider or not model:
        raise RouteCompilationError(f"{location} requires provider and model")
    context_window = _positive_int(row, "context_window", minimum=_MIN_CONTEXT_WINDOW)
    max_output_tokens = _positive_int(row, "max_output_tokens")
    no_output_timeout = _positive_float(row, "no_output_timeout_seconds")
    total_timeout = _positive_float(row, "total_timeout_seconds")
    if total_timeout < no_output_timeout:
        raise RouteCompilationError(
            f"{location} total_timeout_seconds must cover no_output_timeout_seconds"
        )
    case_ids = row.get("benchmark_case_ids")
    if not isinstance(case_ids, Sequence) or isinstance(case_ids, (str, bytes)):
        raise RouteCompilationError(f"{location} benchmark_case_ids must be a non-empty list")
    normalized_cases = tuple(str(case_id).strip() for case_id in case_ids if str(case_id).strip())
    if not normalized_cases:
        raise RouteCompilationError(f"{location} benchmark_case_ids must be non-empty")
    destination = str(row.get("destination_class") or "").strip().lower()
    if not destination:
        raise RouteCompilationError(f"{location} requires destination_class")
    reasoning = row.get("reasoning_effort")
    if reasoning is not None and not isinstance(reasoning, str):
        raise RouteCompilationError(f"{location} reasoning_effort must be a string or null")
    return ResolvedRoute(
        provider=provider,
        model=model,
        reasoning_effort=reasoning,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        no_output_timeout_seconds=no_output_timeout,
        total_timeout_seconds=total_timeout,
        concurrency_weight=_positive_int(row, "concurrency_weight"),
        privacy_class=str(row.get("privacy_class") or "").strip(),
        destination_class=destination,
        artifact_digest=artifact_digest,
        policy_digest=policy_digest,
        benchmark_case_ids=normalized_cases,
    )


def compile_profile_routes(
    profiles: Sequence[str], artifact: Mapping[str, Any]
) -> CompiledRouteTable:
    """Validate an exhaustive artifact and return immutable route rows.

    A profile must have its own complete matrix.  There is intentionally no
    default-profile inheritance because shared model choice is not benchmark
    evidence for another profile.
    """

    artifact_digest = _digest(artifact.get("artifact_digest"), field="artifact_digest")
    policy_digest = _digest(artifact.get("policy_digest"), field="policy_digest")
    artifact_profiles = artifact.get("profiles")
    if not isinstance(artifact_profiles, Mapping):
        raise RouteCompilationError("profiles must be a mapping")

    compiled: CompiledRouteTable = {}
    normalized_profiles = tuple(dict.fromkeys(str(profile).strip() for profile in profiles))
    if not normalized_profiles or any(not profile for profile in normalized_profiles):
        raise RouteCompilationError("at least one non-empty profile is required")

    for profile in normalized_profiles:
        rows = artifact_profiles.get(profile)
        if not isinstance(rows, Mapping):
            raise RouteCompilationError(f"profile {profile!r} has no explicit route matrix")
        missing = sorted(REQUIRED_SURFACES - set(rows))
        extra = sorted(set(rows) - REQUIRED_SURFACES)
        if missing:
            raise RouteCompilationError(
                f"profile {profile!r} is missing required surfaces: {', '.join(missing)}"
            )
        if extra:
            raise RouteCompilationError(
                f"profile {profile!r} declares unknown surfaces: {', '.join(extra)}"
            )
        compiled_rows: dict[str, CompiledRoute] = {}
        for surface in sorted(REQUIRED_SURFACES):
            pair = rows[surface]
            if not isinstance(pair, Mapping):
                raise RouteCompilationError(f"{profile}.{surface} must be a route pair")
            primary = _parse_route(
                pair.get("primary"),
                artifact_digest=artifact_digest,
                policy_digest=policy_digest,
                location=f"{profile}.{surface}.primary",
            )
            fallback = _parse_route(
                pair.get("privacy_fallback"),
                artifact_digest=artifact_digest,
                policy_digest=policy_digest,
                location=f"{profile}.{surface}.privacy_fallback",
            )
            if fallback.destination_class not in _LOCAL_DESTINATIONS:
                raise RouteCompilationError(
                    f"{profile}.{surface}.privacy_fallback must use a local destination"
                )
            compiled_rows[surface] = CompiledRoute(primary=primary, privacy_fallback=fallback)
        compiled[profile] = compiled_rows
    return compiled


def resolve_route(
    compiled: Mapping[str, Mapping[str, CompiledRoute]],
    *,
    profile: str,
    surface: str,
    privacy: str,
    required_context: int,
) -> ResolvedRoute:
    """Resolve one compiled row while enforcing privacy and context admission."""

    try:
        pair = compiled[profile][surface]
    except KeyError as exc:
        raise RouteCompilationError(f"no compiled route for {profile}.{surface}") from exc
    route = pair.privacy_fallback if privacy.strip().lower() in {"private", "raw"} else pair.primary
    if required_context > route.context_window:
        raise RouteCompilationError(
            f"{profile}.{surface} required context {required_context} exceeds "
            f"route context window {route.context_window}"
        )
    return route
