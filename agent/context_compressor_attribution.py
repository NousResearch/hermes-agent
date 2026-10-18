"""One route snapshot and selector for compression failure diagnostics (#72636)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SummaryRoute:
    provider: str
    model: str
    base_url: str
    dispatched: bool = False
    configured: bool = True


def select_summary_route(compressor: Any) -> SummaryRoute:
    """A wire snapshot replaces the configured route atomically; static identity is last."""
    route = getattr(compressor, "_last_aux_route", None)
    if route is not None:
        return route
    return SummaryRoute(
        provider=getattr(compressor, "provider", "") or "auto",
        model=getattr(compressor, "summary_model", "") or getattr(compressor, "model", "") or "unknown",
        base_url=getattr(compressor, "base_url", "") or "unknown",
        configured=False,
    )


def _safe_route(provider: str | None, model: str | None, base_url: str | None, *, dispatched: bool) -> SummaryRoute:
    return SummaryRoute(
        provider=provider or "unknown", model=model or "unknown",
        base_url=str(base_url or "unknown").split("#", 1)[0].split("?", 1)[0],
        dispatched=dispatched,
    )


class AuxRouteAttributionMixin:
    """Keep diagnostics about the final attempt separate from sticky preservation flags."""

    def _init_aux_route_attribution(self) -> None:
        self._last_aux_route: SummaryRoute | None = None
        self._last_attempt_failure_class: str | None = None

    def _prepare_aux_route_attribution(self) -> None:
        self._last_aux_route = None
        self._last_attempt_failure_class = None

    def _record_configured_aux_route(self, provider, model, base_url) -> None:
        """Capture the existing resolver's result before client construction can fail."""
        if provider not in (None, "", "auto"):
            self._last_aux_route = _safe_route(provider, model, base_url, dispatched=False)

    def _record_aux_route(self, provider, model, base_url) -> None:
        """The latest physical attempt supersedes configured identity, including after Relay."""
        self._last_aux_route = _safe_route(provider, model, base_url, dispatched=True)

    def _aux_route_label(self) -> str:
        route = select_summary_route(self)
        return f"(provider={route.provider} model={route.model})"

    def _summary_failure_identity(self) -> tuple[str, str, str]:
        route = select_summary_route(self)
        return route.provider, route.model, route.base_url

    def _record_summary_access_failure(self, error: Exception) -> None:
        """Keep the legacy preservation flag while distinguishing billing from credentials."""
        from agent.context_compressor import _exc_status_code, _SUMMARY_PERMANENT_QUOTA_MARKERS

        self._last_summary_auth_failure = True
        quota = _exc_status_code(error) == 402 or any(
            marker in str(error).lower() for marker in _SUMMARY_PERMANENT_QUOTA_MARKERS
        )
        self._last_attempt_failure_class = "quota" if quota else "auth"

    def _classify_attempt_failure(self, failure_class: str) -> None:
        """Record this attempt's failure class unless a more specific one is already set.

        The generic branch (timeout, JSON decode, 5xx, ... — anything that is not auth
        and not a network stream close) is classified "other" so the abort diagnostic
        does not inherit a stale "auth" verdict from a prior attempt (#72636).
        """
        if self._last_attempt_failure_class is None:
            self._last_attempt_failure_class = failure_class

    def _record_summary_failure_policy(self, error: Exception) -> bool:
        """Retain an exhausted chain's access abort without relabeling its final error."""
        from agent.context_compressor import _is_summary_access_or_quota_error

        access_error = _is_summary_access_or_quota_error(error)
        if access_error:
            self._record_summary_access_failure(error)
        policy_error = getattr(error, "_auxiliary_preservation_error", None)
        policy_access_error = policy_error is not None and _is_summary_access_or_quota_error(policy_error)
        if policy_access_error:
            self._last_summary_auth_failure = True
        return access_error or policy_access_error
