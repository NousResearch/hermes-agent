"""Best-effort accessors for the single-writer stream fence (#65991).

The fence itself lives on ``AIAgent`` (``_claim_stream_writer`` /
``_stream_writer_is_current`` in ``run_agent.py``), but the streaming code paths
that use it live in *other* modules — ``chat_completion_helpers`` (chat /
anthropic / bedrock) and ``codex_runtime`` (codex responses). Calling the fence
directly as ``agent._claim_stream_writer()`` from those modules makes them
hard-depend on the method being present on whatever object is passed in as
``agent``.

That coupling is a latent crash: a partially-updated checkout (the streaming
helper module newer than ``run_agent``), a hot-reloaded gateway, a duck-typed
agent, or a test double without the method turns an *additive* safety net into a
fatal ``AttributeError`` that aborts the whole turn. A cron job died exactly
this way with ``'AIAgent' object has no attribute '_claim_stream_writer'``.

The fence is only ever allowed to drop a *provably* superseded stream — never
the sole legitimate writer. So when the guard is unavailable (or raises), the
correct degradation is "no fence": keep streaming. These helpers make the
claim/check best-effort to guarantee that.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def claim_stream_writer(agent: Any) -> int:
    """Claim the delta sink for the calling stream attempt, best-effort.

    Returns the agent's monotonic writer token when the fence is available, or
    ``0`` when the agent doesn't expose it (or the claim raised). A ``0`` token
    pairs with :func:`stream_writer_is_current` always returning ``True``, so a
    guard-less agent is simply never fenced instead of crashing the turn.
    """
    claim = getattr(agent, "_claim_stream_writer", None)
    if callable(claim):
        try:
            return int(claim())
        except Exception:
            logger.debug(
                "stream single-writer: claim failed; proceeding unfenced",
                exc_info=True,
            )
    return 0


def stream_writer_is_current(agent: Any, token: int) -> bool:
    """True when ``token`` is still the active writer, best-effort.

    A falsy token (from a claim that no-oped) or an agent without the fence
    means we cannot prove supersession, so the stream is treated as current and
    never fenced. This preserves the single-writer invariant's one-way promise:
    only a demonstrably stale writer is ever stopped.
    """
    if not token:
        return True
    is_current = getattr(agent, "_stream_writer_is_current", None)
    if callable(is_current):
        try:
            return bool(is_current(token))
        except Exception:
            logger.debug(
                "stream single-writer: is_current check failed; treating as current",
                exc_info=True,
            )
    return True
