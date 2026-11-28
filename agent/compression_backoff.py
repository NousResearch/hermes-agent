"""Structural no-op backoff follows context growth as well as elapsed time."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def record_structural_no_op(compressor: Any, reason: str) -> None:
    """No eligible material is a temporary defer, never an ineffectiveness strike."""
    telemetry = compressor._active_compression_telemetry or {}
    compressor._structural_no_op_tokens = (
        telemetry.get("current_estimated_tokens") or max(0, compressor.last_prompt_tokens)
    )
    compressor._structural_no_op_backoff_until = time.monotonic() + compressor._STRUCTURAL_NO_OP_BACKOFF_SECONDS
    if not compressor.quiet_mode:
        logger.warning(
            "Compression skipped (%s): retrying after context growth or in %.0fs",
            reason, compressor._STRUCTURAL_NO_OP_BACKOFF_SECONDS,
        )


def rearm_after_growth(compressor: Any, tokens: int) -> None:
    """New material can become compressible long before a five-minute timer expires.

    Allow one probe per 5% window growth (1K–4K tokens), not per gate evaluation.
    A further no-op records a new baseline; summary failures and ineffective
    compactions retain their independent guards.
    """
    baseline = compressor._structural_no_op_tokens
    if baseline <= 0 or compressor._structural_no_op_backoff_until <= time.monotonic():
        return
    growth = max(1024, min(4096, int(compressor.context_length * 0.05)))
    if tokens - baseline >= growth:
        compressor._structural_no_op_backoff_until = 0.0
        logger.info("Context grew from %s to %s tokens; retrying structural compaction skip", baseline, tokens)
