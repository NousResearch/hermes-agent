"""Routing policy primitives for the opt-in provider gateway."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.error_classifier import FailoverReason
from provider_gateway.config import GatewayConfig, load_gateway_config


@dataclass(frozen=True)
class ProviderRouteCandidate:
    """One possible provider/model route."""

    provider: str
    model: str
    source: str
    base_url: str | None = None
    key_env: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ProviderGatewayPolicy:
    """Ordered route policy assembled from gateway config and Hermes fallback state."""

    enabled: bool
    routing_strategy: str
    candidates: tuple[ProviderRouteCandidate, ...]

    def next_after(
        self,
        provider: str,
        model: str,
        *,
        base_url: str | None = None,
    ) -> ProviderRouteCandidate | None:
        """Return the next candidate after the matching current route."""
        current_key = _candidate_key(
            provider=provider,
            model=model,
            base_url=base_url,
            ignore_base_url=base_url is None,
        )
        for index, candidate in enumerate(self.candidates):
            candidate_key = _candidate_key(
                provider=candidate.provider,
                model=candidate.model,
                base_url=candidate.base_url,
                ignore_base_url=base_url is None,
            )
            if candidate_key == current_key:
                next_index = index + 1
                if next_index < len(self.candidates):
                    return self.candidates[next_index]
                return None
        return self.candidates[0] if self.candidates else None


_GATEWAY_FALLBACK_REASONS = {
    FailoverReason.billing,
    FailoverReason.rate_limit,
    FailoverReason.overloaded,
    FailoverReason.server_error,
    FailoverReason.timeout,
    FailoverReason.model_not_found,
    FailoverReason.provider_policy_blocked,
    FailoverReason.unknown,
}


def build_gateway_policy(
    agent: Any,
    config: GatewayConfig | None = None,
) -> ProviderGatewayPolicy:
    """Build an ordered route policy without mutating the agent."""
    config = config if isinstance(config, GatewayConfig) else load_gateway_config()
    candidates: list[ProviderRouteCandidate] = []
    seen: set[tuple[str, str, str | None]] = set()

    primary_provider = _clean_text(getattr(agent, "provider", None))
    primary_model = _clean_text(getattr(agent, "model", None))
    primary_base_url = _clean_base_url(getattr(agent, "base_url", None))

    _append_candidate(
        candidates,
        seen,
        ProviderRouteCandidate(
            provider=primary_provider,
            model=primary_model,
            source="primary",
            base_url=primary_base_url,
        ),
    )

    if config.enabled:
        for model in config.fallback_models:
            _append_candidate(
                candidates,
                seen,
                ProviderRouteCandidate(
                    provider=primary_provider,
                    model=_clean_text(model),
                    source="provider_gateway.routing.fallback_models",
                    base_url=primary_base_url,
                ),
            )

        # Integrate auto-discovered local Ollama models
        try:
            from provider_gateway.runtime import get_discovered_ollama_models
            for lm in get_discovered_ollama_models():
                _append_candidate(
                    candidates,
                    seen,
                    ProviderRouteCandidate(
                        provider=lm["provider"],
                        model=lm["model"],
                        source="ollama_discovery",
                        base_url=lm["base_url"],
                    ),
                )
        except Exception as exc:
            logger.debug("Failed to append discovered Ollama models: %s", exc)

        for entry in getattr(agent, "_fallback_chain", []) or []:
            if not isinstance(entry, dict):
                continue
            fallback_provider = _clean_text(entry.get("provider"))
            fallback_base_url = _clean_base_url(entry.get("base_url"))
            if not fallback_base_url and fallback_provider == primary_provider:
                fallback_base_url = primary_base_url
            key_env = _clean_text(entry.get("key_env")) or _clean_text(
                entry.get("api_key_env")
            )
            _append_candidate(
                candidates,
                seen,
                ProviderRouteCandidate(
                    provider=fallback_provider,
                    model=_clean_text(entry.get("model")),
                    source="fallback_chain",
                    base_url=fallback_base_url,
                    key_env=key_env or None,
                    api_key=_clean_text(entry.get("api_key")) or None,
                ),
            )

    return ProviderGatewayPolicy(
        enabled=config.enabled,
        routing_strategy=config.routing_strategy,
        candidates=tuple(candidates),
    )


def should_consider_gateway_fallback(reason: FailoverReason | None) -> bool:
    """Return whether a failure reason should try another route."""
    if reason is None:
        return True
    return reason in _GATEWAY_FALLBACK_REASONS


def _append_candidate(
    candidates: list[ProviderRouteCandidate],
    seen: set[tuple[str, str, str | None]],
    candidate: ProviderRouteCandidate,
) -> None:
    if not candidate.provider or not candidate.model:
        return
    key = _candidate_key(
        provider=candidate.provider,
        model=candidate.model,
        base_url=candidate.base_url,
    )
    if key in seen:
        return
    seen.add(key)
    candidates.append(candidate)


def _candidate_key(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    ignore_base_url: bool = False,
) -> tuple[str, str, str | None]:
    return (
        _clean_text(provider).lower(),
        _clean_text(model),
        None if ignore_base_url else _clean_base_url(base_url),
    )


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _clean_base_url(value: Any) -> str | None:
    text = _clean_text(value).rstrip("/")
    return text or None
