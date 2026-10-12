"""Context-window screening for runtime fallback chains (moved from ``agent.auxiliary_client.py``).

A topical sibling of the auxiliary client (see root AGENTS.md: moved code keeps its cap):
the screening helpers are self-contained policy — they only read model metadata — so they
live here while ``auxiliary_client.py`` keeps the ladder that calls them.
"""

import logging
from typing import Any, Optional

from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length

logger = logging.getLogger(__name__)

# ── Context-window screening for runtime fallback chains (issue #52392) ── When the runtime auxiliary
# fallback chain selects a candidate that is reachable but has a context window smaller than the compression
# task requires, the call errors out instead of continuing to the next, viable candidate. The startup
# feasibility check in ``agent.conversation_compression.check_compression_model_feasibility`` already
# filters too-small auxiliary models at startup, but the runtime fallback chain
# (``_try_configured_fallback_chain`` and ``_try_main_fallback_chain``) does not apply the same filter, so
# compression can stop at the first alive door even if the room behind it is too small. The helpers below
# screen each candidate by its effective context window before it is returned. ``None`` results from
# ``get_model_context_length`` are passed through (we cannot prove a model is too small, so we do not block
# it). This preserves the existing fallback surface for unrecognised/custom models while closing the gap on
# the well-known ones.
def _task_minimum_context_length(task: Optional[str]) -> Optional[int]:
    """Minimum context length for an auxiliary task; None = no floor (only ``compression`` has one)."""
    return MINIMUM_CONTEXT_LENGTH if task == "compression" else None


def _candidate_context_window(provider: str, model: str, base_url: str = "", api_key: str = "") -> Optional[int]:
    """Best-effort context window for a fallback candidate; ``None`` = unknown (never raises; callers pass it through)."""
    if not model:
        return None
    try:
        ctx = get_model_context_length(model, base_url=base_url, api_key=api_key, provider=provider)
    except Exception as exc:
        logger.debug("Auxiliary fallback: could not resolve context window for %s/%s: %s", provider, model, exc)
        return None
    return ctx if isinstance(ctx, int) and ctx > 0 else None


def _context_too_small(
    entry: dict[str, Any], provider: str, model: str, min_ctx: Optional[int], *,
    task: Optional[str], label: str, name_model: bool = False,
) -> Optional[str]:
    """Screen one fallback candidate by context window; returns the ``tried`` note when it is too small."""
    if min_ctx is None:
        return None
    from hermes_cli.fallback_config import resolve_entry_api_key
    fb_ctx = _candidate_context_window(
        provider, model, base_url=str(entry.get("base_url") or ""),
        api_key=resolve_entry_api_key(entry) or "")
    if fb_ctx is None or fb_ctx >= min_ctx:
        return None
    if name_model:
        logger.info("Auxiliary %s: skipping %s (%s context=%d < min=%d), continuing chain",
                    task, label, model, fb_ctx, min_ctx)
    else:
        logger.info("Auxiliary %s: skipping %s (context=%d < min=%d), continuing chain",
                    task or "call", label, fb_ctx, min_ctx)
    return f"{label} (context too small: {fb_ctx}<{min_ctx})"
