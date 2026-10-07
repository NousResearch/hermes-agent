"""Rate-limit header capture for ``AIAgent``.

Parses provider response headers into ``_rate_limit_state``.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import logging
from typing import Any

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


def _response_headers(http_response: Any):
    """Headers of a response, or None when there is nothing to parse."""
    return getattr(http_response, "headers", None) if http_response is not None else None


class RateLimitMixin:
    """Rate-limit header capture (see module docstring)."""

    def _capture_rate_limits(self, http_response: Any) -> None:
        """Parse x-ratelimit-* headers from an HTTP response and cache the state (never raises)."""
        headers = _response_headers(http_response)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def _capture_anthropic_response_headers(self, http_response: Any) -> None:
        """Capture rate-limit + credits state from Anthropic Messages response headers (the SDK's
        aggregated ``Message`` drops them). Fail-open."""
        self._capture_rate_limits(http_response)

    def _check_openrouter_cache_status(self, http_response: Any) -> None:
        """Log X-OpenRouter-Cache-Status; HITs count in ``_or_cache_hits``. Never raises."""
        headers = _response_headers(http_response)
        if not headers:
            return
        try:
            status = headers.get("x-openrouter-cache-status")
            if not status:
                return
            if status.upper() == "HIT":
                self._or_cache_hits += 1
                logger.info("OpenRouter response cache HIT (total: %d)", self._or_cache_hits)
            else:
                logger.debug("OpenRouter response cache %s", status.upper())
        except Exception:
            pass  # Never let header parsing break the agent loop
