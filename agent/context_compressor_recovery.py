"""When a completed compaction counts as recovery (anti-thrash verdict and per-turn budget rearm)."""

# A compaction is recovery only when the provider's real prompt lands this far under the trigger.
# Landing a hair under it (re-summarising only the previous summary) re-fires on the next tool
# result; scored as recovery, every such pass cleared the strikes and compaction looped.
_RECOVERY_HEADROOM_RATIO = 0.10


def compaction_recovery_ceiling(threshold_tokens: int) -> int:
    """Real prompt count a completed compaction must land under to count as recovery."""
    return threshold_tokens - int(threshold_tokens * _RECOVERY_HEADROOM_RATIO)
