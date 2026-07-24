"""Cascade web search plugin — circuit breaker + multi-backend fallback.

Wraps registered web search providers (tavily, firecrawl, ddgs, etc.) with:
- Circuit breaker: skip backends that recently failed
- Three strategies: serial (sequential), hedge (parallel), hybrid (adaptive)
- Passive stats collection from real usage

Configure via config.yaml:
    web:
      search_backend: cascade
      cascade:
        strategy: serial          # serial | hedge | hybrid
        per_backend_timeout: 4    # seconds per backend attempt
        total_timeout: 10         # hard cap for entire cascade
        hybrid_trigger: 3         # hybrid: start fallbacks after N seconds
        circuit_open_duration: 300  # seconds to keep circuit open
        fallback_chain:
          - tavily
          - firecrawl
          - ddgs
"""

from __future__ import annotations

from .provider import CascadeSearchProvider


def register(ctx) -> None:
    """Register the cascade provider with the plugin context."""
    ctx.register_web_search_provider(CascadeSearchProvider())
