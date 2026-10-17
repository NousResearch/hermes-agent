"""Bounded recovery of provider-confirmed ineffective compactions.

Use real usage for growth: preflight estimates and provider counts can differ by
more than the growth allowance without any new material. Keep a high-water mark
across unsuccessful probes so shrinking and regrowing the same prompt cannot
re-arm indefinitely. A provider rejection also earns one emergency probe between
healthy usage readings; it does not disable structural or summary-failure guards.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def recover_after_growth(compressor: Any, tokens: int) -> None:
    baseline = compressor._ineffective_growth_tokens
    growth = max(1024, min(4096, int(compressor.context_length * 0.05)))
    if compressor._tripped() and baseline > 0 and tokens - baseline >= growth:
        compressor._ineffective_growth_tokens = tokens
        _grant_probe(compressor, "provider-confirmed prompt growth")
        compressor._ineffective_growth_probe_pending = True


def recover_after_overflow(compressor: Any, *, trigger: str, bypass_cooldown: bool, force: bool) -> bool:
    """The outer overflow path gets one probe, including after a process rebuild.

    Only the built-in compressor owns these guards. Keep plugin engines unchanged,
    and do not spend the probe while an independent structural delay blocks it.
    An exhausted caller reserves exactly one slot before enforcing its budget.
    """
    from agent.context_compressor import ContextCompressor

    if trigger != "overflow" or not bypass_cooldown or force or not isinstance(compressor, ContextCompressor):
        return False
    compressor._refresh_durable_guards()
    if (compressor._tripped() and not compressor._ineffective_overflow_probe_used
            and compressor._structural_no_op_backoff_until <= time.monotonic()):
        compressor._ineffective_overflow_probe_used = True
        _grant_probe(compressor, "provider context overflow")
        return True
    return False


def take_growth_probe_budget(compressor: Any, attempts: int, cap: int) -> tuple[int, bool]:
    """Consume the provider-growth signal once; restore ONE slot, not a fresh budget.

    The returned flag also clears the loop's stale preflight-progress latch. Missing
    attributes (plugin engines) and truthy test doubles are not recovery signals.
    """
    if getattr(compressor, "_ineffective_growth_probe_pending", None) is not True:
        return attempts, False
    compressor._ineffective_growth_probe_pending = False
    logger.info("Compression growth probe admitted by caller (attempts=%s cap=%s)", attempts, cap)
    return min(attempts, max(0, cap - 1)), True


def reserve_overflow_probe_budget(compressor: Any, attempts: int, cap: int) -> int:
    """The overflow caller must admit the one-shot probe before enforcing its cap."""
    if attempts >= cap and recover_after_overflow(compressor, trigger="overflow", bypass_cooldown=True, force=False):
        return max(0, cap - 1)
    return attempts


def _grant_probe(compressor: Any, reason: str) -> None:
    compressor._record_ineffective_compression_verdict(1)
    compressor._set_anti_thrash_recovery_deadline(0.0)
    logger.info("Allowing one ineffective-compression recovery probe after %s", reason)


def apply_real_prompt_verdict(compressor: Any) -> None:
    """Judge a compaction once, against provider usage; ordinary growth may earn a probe."""
    compressor._ineffective_growth_probe_pending = False
    tokens = compressor.last_prompt_tokens
    if tokens <= 0:
        return
    compressor.last_real_prompt_tokens = tokens
    compressor._provider_omits_usage = False
    if tokens < compressor.threshold_tokens:
        compressor._record_ineffective_compression_verdict(0)
    elif compressor._verify_compaction_cleared_threshold:
        compressor._record_ineffective_compression_verdict(compressor._ineffective_compression_count + 1)
        compressor._ineffective_growth_tokens = max(compressor._ineffective_growth_tokens, tokens)
        if not compressor.quiet_mode:
            logger.warning(
                "Compaction did not clear the threshold: %d real tokens still >= %d. "
                "ineffective_compression_count=%d",
                tokens, compressor.threshold_tokens, compressor._ineffective_compression_count,
            )
    else:
        recover_after_growth(compressor, tokens)
