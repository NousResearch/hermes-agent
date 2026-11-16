"""Prepared compaction (``compression.prepare_ahead``): adoption and every way a prepared pass is wasted.

The daily snapshot carries ``compression_prepare_ahead`` (``on``/``off``), read with the agent's own
truthiness, so "active installs with the feature on" needs no join. ``hermes.prepared_compaction.count``
counts each pass and splice outcome from ``agent/prepared_compaction.py`` (and a spliced compaction that
then failed, from ``agent/conversation_compression_telemetry.py``) over closed sets only: the event, the
hook that started the pass, a reason code (never the free-text miss reason), the compression failure
class, and how long a compaction waited on a running pass, bucketed.

The contract imports this module at load time, so it never imports the contract at module level.
"""

from __future__ import annotations

from typing import Any

PREPARED_COMPACTION_MARK = PREPARED_COMPACTION_METRIC = "hermes.prepared_compaction.count"
PREPARED_COMPACTION_EVENTS = frozenset({
    "auto_disabled", "compaction_failed", "discarded", "internal_error", "pass_cancelled", "pass_completed",
    "pass_failed", "pass_started", "skipped_busy", "used",
})
PREPARED_COMPACTION_ORIGINS = frozenset({"none", "tool_batch", "turn_end"})
PREPARED_COMPACTION_REASONS = frozenset({
    "agent_released", "aux_fallback_main", "compaction_aborted", "fresh_required", "head_moved", "host_cancelled",
    "none", "other_session", "pass_failed_while_waiting", "provider_context", "rewound",
    "running_not_fit", "session_boundary", "shape_changed", "superseded", "text_changed",
    "transcript_shorter", "wait_timeout",
})
PREPARED_COMPACTION_WAIT_BUCKETS = frozenset({"none", "lt_1s", "1s_to_5s", "5s_to_30s", "30s_to_120s", "gte_120s"})
_WAIT_THRESHOLDS = ((1, "lt_1s"), (5, "1s_to_5s"), (30, "5s_to_30s"), (120, "30s_to_120s"))
# failure_class (the compression vocabulary) is added by the contract, which owns that set.
PREPARED_COMPACTION_DIMENSIONS = {
    "event": PREPARED_COMPACTION_EVENTS, "origin": PREPARED_COMPACTION_ORIGINS,
    "reason": PREPARED_COMPACTION_REASONS, "wait_bucket": PREPARED_COMPACTION_WAIT_BUCKETS,
}
PREPARED_COMPACTION_SNAPSHOT_DIMENSIONS = {"compression_prepare_ahead": frozenset({"off", "on"})}


def prepare_ahead_state(config: Any) -> str:
    """``on`` when the profile config turns ``compression.prepare_ahead`` on, read as the agent reads it."""
    from utils import is_truthy_value

    compression = config.get("compression") if isinstance(config, dict) else None
    value = compression.get("prepare_ahead") if isinstance(compression, dict) else None
    return "on" if is_truthy_value(value, default=False) else "off"


def wait_bucket(seconds: Any) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        return "none"
    return next((label for limit, label in _WAIT_THRESHOLDS if seconds < limit), "gte_120s")


def _closed(value: Any, allowed: frozenset[str], fallback: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in allowed else fallback


def prepared_compaction_fields(
    *, event: Any, origin: Any, reason: Any, failure_class: Any, wait_seconds: Any,
) -> dict[str, str] | None:
    """Closed dimensions for one event; ``None`` for an event outside the set (never recorded)."""
    from .shared_metrics_fields import compression_failure_class

    name = _closed(event, PREPARED_COMPACTION_EVENTS, "")
    if not name:
        return None
    return {
        "event": name,
        "failure_class": "none" if failure_class is None else compression_failure_class("failed", failure_class),
        "origin": _closed(origin, PREPARED_COMPACTION_ORIGINS, "none"),
        "reason": _closed(reason, PREPARED_COMPACTION_REASONS, "none"),
        "wait_bucket": wait_bucket(wait_seconds),
    }


def record_prepared_compaction(
    event: str, *, origin: str | None = None, reason: str | None = None, failure_class: str | None = None,
    wait_seconds: float | None = None,
) -> None:
    """Count one prepared-compaction event (no-op unless shared metrics are on; never raises)."""
    from .shared_metrics_events import _emit

    _emit(
        PREPARED_COMPACTION_MARK, prepared_compaction_fields, event=event, origin=origin, reason=reason,
        failure_class=failure_class, wait_seconds=wait_seconds,
    )
