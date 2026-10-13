"""Preserve useful tool-result pruning when no summarizer window remains."""

import logging
from typing import Any

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)


def finish_pruned_or_defer(
    compressor: Any, pruned: list[dict], original: list[dict], pruned_count: int,
    spare_pending_images: bool, telemetry: dict, failure_class: str, reason: str,
) -> list[dict]:
    """Return a normal compression candidate if pruning itself reclaimed context.

    Choosing the protected tail after pruning can leave only a prior handoff
    in the middle. Returning the original in that case undoes the reclamation
    and repeats forever. Finalize the pruned copy through the same ownership,
    commit and provider-usage checks as a summarized candidate instead.
    """
    if pruned_count and estimate_messages_tokens_rough(pruned) < estimate_messages_tokens_rough(original):
        from agent.conversation_compression import _raise_if_stale_attempt

        _raise_if_stale_attempt(compressor)
        logger.info("No new summary needed; retaining pruning of %d tool results", pruned_count)
        return compressor._finalize_compressed(pruned, original, len(original), spare_pending_images, pruned_count)
    compressor._structural_no_op_result(telemetry, failure_class, reason)
    return original
