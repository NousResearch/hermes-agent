"""Provider Health Scoring — dynamic failover with latency/error metrics.

Tracks per-provider performance (latency, error rate, consecutive failures)
and provides health-aware routing for the fallback chain. Providers that
consistently fail get demoted; recovered providers get promoted.

Inspired by agno's fallback_models with fallback_config pattern.

Usage:
    health = ProviderHealthTracker()
    health.record_success("openrouter", latency_ms=450)
    health.record_failure("anthropic", error="rate_limit")

    # Get sorted providers by health
    ranked = health.rank_providers(["openrouter", "anthropic", "ollama"])
    # -> ["openrouter", "ollama", "anthropic"]  (anthropic demoted due to failure)

    # Check if a provider should be skipped
    if health.should_skip("anthropic"):
        try_next_provider()

Config in cli-config.yaml:
    fallback:
      health_tracking: true
      demotion_threshold: 3  # consecutive failures before demotion
      recovery_window: 300   # seconds before retrying a demoted provider
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_DEMOTION_THRESHOLD = 3
DEFAULT_RECOVERY_WINDOW = 300  # 5 minutes
DEFAULT_LATENCY_WEIGHT = 0.3
DEFAULT_ERROR_WEIGHT = 0.7


@dataclass
class ProviderMetrics:
    """Performance metrics for a single provider."""
    provider: str = ""
    total_calls: int = 0
    total_errors: int = 0
    consecutive_failures: int = 0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    last_error: str = ""
    avg_latency_ms: float = 0.0
    _latency_sum: float = 0.0
    _latency_count: int = 0
    demoted: bool = False
    demoted_at: float = 0.0

    @property
    def error_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_errors / self.total_calls

    @property
    def health_score(self) -> float:
        """Composite health score 0.0-1.0 (higher = healthier)."""
        if self.total_calls == 0:
            return 0.5  # Unknown — neutral

        # Error component (0-1, lower error = higher score)
        error_score = 1.0 - min(self.error_rate, 1.0)

        # Latency component (0-1, lower latency = higher score)
        # Normalize: 0ms=1.0, 5000ms=0.0
        latency_score = max(0.0, 1.0 - (self.avg_latency_ms / 5000.0))

        # Consecutive failure penalty
        failure_penalty = min(self.consecutive_failures * 0.15, 0.6)

        score = (
            error_score * DEFAULT_ERROR_WEIGHT
            + latency_score * DEFAULT_LATENCY_WEIGHT
            - failure_penalty
        )
        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "consecutive_failures": self.consecutive_failures,
            "error_rate": round(self.error_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "health_score": round(self.health_score, 3),
            "demoted": self.demoted,
        }


class ProviderHealthTracker:
    """Tracks and scores provider health for the fallback chain."""

    def __init__(
        self,
        demotion_threshold: int = DEFAULT_DEMOTION_THRESHOLD,
        recovery_window: float = DEFAULT_RECOVERY_WINDOW,
    ):
        self.demotion_threshold = demotion_threshold
        self.recovery_window = recovery_window
        self._metrics: Dict[str, ProviderMetrics] = {}

    def _get_metrics(self, provider: str) -> ProviderMetrics:
        if provider not in self._metrics:
            self._metrics[provider] = ProviderMetrics(provider=provider)
        return self._metrics[provider]

    def record_success(self, provider: str, latency_ms: float = 0.0) -> None:
        """Record a successful API call."""
        m = self._get_metrics(provider)
        m.total_calls += 1
        m.consecutive_failures = 0
        m.last_success_at = time.time()

        if latency_ms > 0:
            m._latency_sum += latency_ms
            m._latency_count += 1
            m.avg_latency_ms = m._latency_sum / m._latency_count

        # Auto-promote if was demoted
        if m.demoted:
            m.demoted = False
            m.demoted_at = 0.0
            logger.info("Provider %s promoted (recovered after demotion)", provider)

    def record_failure(self, provider: str, error: str = "") -> None:
        """Record a failed API call."""
        m = self._get_metrics(provider)
        m.total_calls += 1
        m.total_errors += 1
        m.consecutive_failures += 1
        m.last_failure_at = time.time()
        m.last_error = error

        # Auto-demote on threshold
        if m.consecutive_failures >= self.demotion_threshold and not m.demoted:
            m.demoted = True
            m.demoted_at = time.time()
            logger.warning(
                "Provider %s demoted (%d consecutive failures: %s)",
                provider, m.consecutive_failures, error,
            )

    def should_skip(self, provider: str) -> bool:
        """Check if a provider should be skipped (demoted and not recovered)."""
        m = self._metrics.get(provider)
        if not m or not m.demoted:
            return False

        # Check recovery window
        elapsed = time.time() - m.demoted_at
        if elapsed >= self.recovery_window:
            # Recovery window passed — allow retry
            logger.info("Provider %s eligible for recovery (%.0fs since demotion)", provider, elapsed)
            return False

        return True

    def rank_providers(self, providers: List[str]) -> List[str]:
        """Sort providers by health score (healthiest first). Skip demoted ones."""
        available = [p for p in providers if not self.should_skip(p)]
        skipped = [p for p in providers if self.should_skip(p)]

        # Sort available by health score (descending)
        available.sort(key=lambda p: self._get_metrics(p).health_score, reverse=True)

        # Append skipped at the end (they'll only be tried as last resort)
        return available + skipped

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all provider metrics."""
        return {
            provider: m.to_dict()
            for provider, m in sorted(self._metrics.items(), key=lambda x: x[1].health_score, reverse=True)
        }

    def get_health(self, provider: str) -> float:
        """Get health score for a specific provider (0.0-1.0)."""
        m = self._metrics.get(provider)
        return m.health_score if m else 0.5
