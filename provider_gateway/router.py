"""Dynamic router for the opt-in provider gateway.

Selects the best provider/model route based on configured strategies
(round-robin, lowest-cost, lowest-latency) and health statuses.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from agent.usage_pricing import estimate_usage_cost
from provider_gateway.circuit_breaker import CircuitBreaker

if TYPE_CHECKING:
    from provider_gateway.policy import ProviderRouteCandidate

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Intelligent router selecting optimal provider routes based on various strategies."""

    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self.circuit_breaker = circuit_breaker

    def select_route(
        self,
        candidates: list[ProviderRouteCandidate],
        strategy: str = "round-robin",
        current_provider: str | None = None,
        current_model: str | None = None,
    ) -> ProviderRouteCandidate | None:
        """Select the next best route candidate based on the active strategy and health.

        Returns None if no candidates are available.
        """
        if not candidates:
            return None

        # Filter candidates by circuit breaker availability
        healthy_candidates = [
            c for c in candidates
            if self.circuit_breaker.is_available(c.provider)
        ]

        # Fail-safe: if all providers are circuit-open, fall back to all candidates
        if not healthy_candidates:
            logger.warning(
                "All provider gateway candidates are circuit-open! Falling back to all configured candidates."
            )
            healthy_candidates = list(candidates)

        strategy = str(strategy).strip().lower()

        if strategy == "lowest-cost":
            return self._select_lowest_cost(healthy_candidates)
        elif strategy == "lowest-latency":
            return self._select_lowest_latency(healthy_candidates)
        else:
            # Default to round-robin
            return self._select_round_robin(
                healthy_candidates,
                current_provider=current_provider,
                current_model=current_model,
            )

    def _select_round_robin(
        self,
        candidates: list[ProviderRouteCandidate],
        current_provider: str | None = None,
        current_model: str | None = None,
    ) -> ProviderRouteCandidate | None:
        """Select the next candidate after the current active provider/model."""
        if not candidates:
            return None

        if current_provider is None or current_model is None:
            return candidates[0]

        # Search for current route index
        current_index = -1
        for idx, c in enumerate(candidates):
            if (
                c.provider.strip().lower() == current_provider.strip().lower()
                and c.model.strip() == current_model.strip()
            ):
                current_index = idx
                break

        if current_index == -1:
            # Current route not found in candidates, return the first one
            return candidates[0]

        # Select the next index, wrap around if needed
        next_index = (current_index + 1) % len(candidates)
        return candidates[next_index]

    def _select_lowest_cost(
        self,
        candidates: list[ProviderRouteCandidate],
    ) -> ProviderRouteCandidate | None:
        """Select the candidate with the lowest estimated cost for a standard request."""
        if not candidates:
            return None

        # Calculate cost for each candidate using standard mock usage (1K prompt + 1K completion tokens)
        mock_usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=1000,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            total_tokens=2000,
        )

        candidate_costs: list[tuple[ProviderRouteCandidate, float]] = []
        for c in candidates:
            cost = 0.0
            try:
                # Resolve provider base URL and API key from candidate hints if present
                base_url = getattr(c, "base_url", None)
                api_key = getattr(c, "api_key", "")
                res = estimate_usage_cost(
                    c.model,
                    mock_usage,
                    provider=c.provider,
                    base_url=base_url,
                    api_key=api_key,
                )
                if res.amount_usd is not None:
                    cost = float(res.amount_usd)
            except Exception as exc:
                logger.debug(
                    "Cost estimation failed in router for %s/%s: %s",
                    c.provider,
                    c.model,
                    exc,
                )
                # Fail-safe to a neutral cost to keep model eligible
                cost = 999.0 if c.provider != "ollama" else 0.0

            candidate_costs.append((c, cost))

        # Sort by cost ascending
        candidate_costs.sort(key=lambda item: item[1])
        logger.debug(
            "Gateway lowest-cost strategy scores: %s",
            [(f"{c.provider}/{c.model}", cost) for c, cost in candidate_costs],
        )
        return candidate_costs[0][0]

    def _select_lowest_latency(
        self,
        candidates: list[ProviderRouteCandidate],
    ) -> ProviderRouteCandidate | None:
        """Select the candidate with the lowest P50 latency based on circuit breaker stats."""
        if not candidates:
            return None

        candidate_latencies: list[tuple[ProviderRouteCandidate, float]] = []
        for c in candidates:
            health = self.circuit_breaker.get_health(c.provider)
            latency = 0.0
            if health is not None:
                latency = health.latency_p50

            candidate_latencies.append((c, latency))

        # Sort by latency. Models with no latency data (latency == 0.0) should be
        # placed after models that actually have successful low latency, but before
        # high latency models. We can map 0.0 latency to a neutral mid-value (e.g. 500ms) for sorting,
        # or treat 0.0 as high priority to discover them. Let's treat 0.0 as 9999.0 (unknown)
        # so proven low-latency models are preferred first, but untested ones are still eligible.
        def sort_key(item: tuple[ProviderRouteCandidate, float]) -> float:
            lat = item[1]
            return 99999.0 if lat == 0.0 else lat

        candidate_latencies.sort(key=sort_key)
        logger.debug(
            "Gateway lowest-latency strategy scores (P50): %s",
            [(f"{c.provider}/{c.model}", lat) for c, lat in candidate_latencies],
        )
        return candidate_latencies[0][0]
