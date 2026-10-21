"""Vertex AI capacity-429 backoff policy for the turn recovery path.

Vertex answers a CAPACITY shortage on the shared PayGo pool with HTTP 429
``RESOURCE_EXHAUSTED`` and no ``Retry-After`` header. Google's guidance for these 429s is
exponential backoff with jitter and smoothed traffic: an immediate retry re-enters the same
contended window, and the generic ladder (2s base) was measured re-429ing on attempts 2 and 3.

Kept in a sibling so ``agent/turn_recovery.py`` stays under its file-size ratchet, and so the
policy table has one home: every provider that is NOT Vertex keeps the generic wait.
"""

from typing import Any

# Policy label surfaced on the retry status line, so the user sees WHY the turn is parked.
VERTEX_CAPACITY_POLICY = "vertex_capacity"

# Why the wait is longer than the generic one — the user must know the provider is out of
# capacity rather than the request being malformed.
VERTEX_CAPACITY_NOTE = " (Vertex capacity exhausted — backoff)"


def vertex_capacity_wait(
    *, base_url: Any, error: Any, attempt: int, default_wait: float,
) -> tuple[float, str | None]:
    """``(wait_seconds, policy_label)``: the Vertex ladder 15→30→60→120s for a Vertex capacity
    429, else ``(default_wait, None)`` so no other provider's behaviour changes."""
    from agent.retry_utils import is_vertex_capacity_exhausted, vertex_capacity_backoff

    if not is_vertex_capacity_exhausted(base_url=str(base_url or ""), error=error):
        return default_wait, None
    return vertex_capacity_backoff(attempt), VERTEX_CAPACITY_POLICY
