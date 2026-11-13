"""``hermes.desktop.long_frames.sum``: how often the Desktop window stutters, per UTC day.

The renderer counts every long animation frame (``PerformanceObserver`` on ``long-animation-frame``,
``longtask`` where that is missing) while its window is visible, folds them into the closed duration
buckets below, and adds the day's visible time; the finished day rides ``shared_metrics.desktop_long_frames``.
One row per non-empty bucket, whose value is the day's frame count (a sum metric, never capped), and
every row carries the day's visible-time bucket so a rate (long frames per visible minute) is
computable. A visible day without a single long frame reports one ``none`` row (value 1) so clean
days still count in the denominator.

Bucket edges: the browser only reports frames of 50 ms or more (three dropped frames at 60 Hz), so
``50ms_to_100ms`` is the visible-but-mild band below the 100 ms ``slow_frame`` friction threshold
(RAIL's response budget); the rest match ``slow_frame``'s buckets so both read alike.

The contract imports this module at import time: stdlib only.
"""

from __future__ import annotations

from typing import Any

DESKTOP_LONG_FRAMES_MARK = DESKTOP_LONG_FRAMES_METRIC = "hermes.desktop.long_frames.sum"
DESKTOP_LONG_FRAME_BUCKETS = frozenset({"50ms_to_100ms", "100ms_to_250ms", "250ms_to_1s", "1s_to_5s", "gte_5s"})
DESKTOP_LONG_FRAME_FIELDS = frozenset({"frame_bucket", "visible_minutes_bucket"})
# Values a row's ``frame_bucket`` may carry: the duration buckets, or ``none`` for a clean visible day.
DESKTOP_LONG_FRAME_ROW_BUCKETS = DESKTOP_LONG_FRAME_BUCKETS | {"none"}


def long_frame_rows(frames: Any, visible_bucket: str) -> list[dict[str, Any]]:
    """``[{dimensions, amount}]`` for one finished day: a row per bucket the client counted (unknown
    buckets and non-positive counts dropped, duplicates summed), or one ``none`` row for a visible
    day without long frames. Nothing for a day the window was never visible."""
    totals: dict[str, int] = {}
    for entry in frames if isinstance(frames, list) else []:
        bucket = entry.get("bucket") if isinstance(entry, dict) else None
        count = entry.get("count") if isinstance(entry, dict) else None
        if bucket in DESKTOP_LONG_FRAME_BUCKETS and isinstance(count, int) and not isinstance(count, bool) and count > 0:
            totals[bucket] = totals.get(bucket, 0) + count
    if not totals and visible_bucket != "0":
        totals = {"none": 1}
    return [
        {"amount": count, "dimensions": {"frame_bucket": bucket, "visible_minutes_bucket": visible_bucket}}
        for bucket, count in sorted(totals.items())
    ]
