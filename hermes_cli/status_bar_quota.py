"""Provider plan allowance for the CLI status bar: how much is LEFT.

The status bar shows what remains of the active provider's plan windows (a rolling
5-hour limit and a weekly one on MiniMax; a session and a weekly window on Codex),
never what was consumed: the question this answers is "how much can I still spend".

The rolling window also carries its reset countdown — "82% left" alone does not say
whether the allowance comes back in ten minutes or in five hours, and that difference
decides whether to wait or to stop.

Every window is read from the shared account-usage cache, so a repaint never makes a
network request. The cache is filled by a background refresh this module asks for, and
a provider whose allowance is unknown contributes no segment at all.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Optional

# The status bar's own glyph vocabulary: ⏳ is the hourglass the bar already uses for
# elapsed time, so an allowance gets its own mark.
GLYPH = "⊛"

# The status-bar field that governs this segment (see ``display.status_bar.fields``).
FIELD = "quota"

# How often the background refresh may fire. The cache's own throttle (REFRESH_AFTER_S)
# already bounds the network work; this bounds how often the repaint path may even ask.
_REFRESH_AFTER_S = 120.0
_lock = threading.Lock()
_last_refresh: dict[str, float] = {}


def _remaining_style(remaining: float) -> str:
    """Colour by what is LEFT: a full allowance is good, an exhausted one critical.

    The thresholds mirror the bar's context gauge (which colours by what is USED), read
    the other way round: under a quarter left is the number worth noticing.
    """
    if remaining <= 5:
        return "class:status-bar-critical"
    if remaining <= 20:
        return "class:status-bar-bad"
    if remaining <= 50:
        return "class:status-bar-warn"
    return "class:status-bar-good"


def _reset_countdown(reset_at: Optional[datetime]) -> str:
    """``1h 56m`` until the window resets; "" when unknown, past, or under a minute.

    The repo's own compact vocabulary (``format_duration_compact``), so the bar and
    ``/usage`` speak the same language about the same instant.
    """
    if not isinstance(reset_at, datetime):
        return ""
    try:
        seconds = (reset_at - datetime.now(reset_at.tzinfo)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return ""
    if seconds < 60:
        return ""
    from cli import format_duration_compact

    return format_duration_compact(seconds)


def _reset_clock(reset_at: Optional[datetime]) -> str:
    """``11.00`` — the wall-clock time the window resets, in the reader's own zone.

    The countdown says how long; this says when. Together they answer "do I wait, or
    do I stop" — a countdown alone leaves the reader doing the arithmetic.

    A reset that has already passed shows nothing: the window is due to reopen on the
    next use, and a clock time in the past would read as a promise the plan does not
    keep. A reset under a minute away shows nothing either — the clock would tick
    every second, and the countdown has already stopped counting.
    """
    if not isinstance(reset_at, datetime):
        return ""
    try:
        seconds = (reset_at - datetime.now(reset_at.tzinfo)).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return ""
    if seconds < 60:
        return ""
    try:
        return reset_at.astimezone().strftime("%H.%M")
    except (TypeError, ValueError, OverflowError):
        return ""


def _window_segments(snapshot: Any) -> tuple[list[list[tuple[str, str]]], bool]:
    """One segment per window of a snapshot, in the provider's order.

    Each window reads its label plus the percent still available. The FIRST window —
    the rolling one — also carries its reset countdown: that window comes back soon
    enough to plan around, while a weekly reset is days out and does not belong in
    a one-line bar. A provider that reports no usable window yields nothing (the
    segment disappears).

    Returns the segments and whether one of them carries a wall-clock reset time —
    the bar trades its context meter for that clock, so it has to be able to tell.
    """
    from agent.account_usage import AccountUsageSnapshot

    if not isinstance(snapshot, AccountUsageSnapshot):
        return [], False
    segments: list[list[tuple[str, str]]] = []
    has_clock = False
    for index, window in enumerate(snapshot.windows):
        if window.used_percent is None:
            continue
        remaining = max(0.0, min(100.0, 100.0 - float(window.used_percent)))
        label = (window.short_label or window.label or "").strip()
        if not label:
            continue
        text = f"{GLYPH} {label} {remaining:.0f}%"
        if index == 0:
            countdown = _reset_countdown(window.reset_at)
            clock = _reset_clock(window.reset_at)
            if countdown and clock:
                text = f"{text} · {countdown} ({clock})"
                has_clock = True
            elif countdown:
                text = f"{text} · {countdown}"
            elif clock:
                text = f"{text} ({clock})"
                has_clock = True
        segments.append([(_remaining_style(remaining), text)])
    return segments, has_clock


def remaining_segments(provider: Optional[str]) -> list[list[tuple[str, str]]]:
    """Status-bar segments for the provider's cached plan allowance; [] when unknown.

    Reads the cache only. A snapshot older than the cache's staleness bound is treated
    as unknown rather than shown as a live number.
    """
    return remaining_segments_with_clock(provider)[0]


def remaining_segments_with_clock(
    provider: Optional[str],
) -> tuple[list[list[tuple[str, str]]], bool]:
    """The segments, and whether one carries a wall-clock reset time.

    The bar gives its context meter up to make room for that clock, so the renderer
    needs to know it is there.
    """
    slug = str(provider or "").strip().lower()
    if not slug:
        return [], False
    try:
        from agent.account_usage_cache import cached_account_usage, snapshot_is_stale
    except Exception:  # health: allow BLE001 -- a missing cache module must not break the bar
        return [], False
    try:
        snapshot = cached_account_usage(slug)
    except Exception:  # health: allow BLE001 -- a cache read must never break the bar
        return [], False
    if snapshot is None or snapshot_is_stale(snapshot):
        return [], False
    return _window_segments(snapshot)


def refresh_remaining(provider: Optional[str]) -> None:
    """Ask for a background refresh of the provider's plan allowance, off the repaint path.

    Fire-and-forget: the fetch runs on a daemon thread and its result lands in the
    account-usage cache, which the next repaint reads. A provider with no allowance
    API is never asked (the hook would only fail).
    """
    slug = str(provider or "").strip().lower()
    if not slug:
        return
    now = time.monotonic()
    with _lock:
        if now - _last_refresh.get(slug, float("-inf")) < _REFRESH_AFTER_S:
            return
        _last_refresh[slug] = now
    try:
        from agent.account_usage_cache import has_account_usage, refresh_account_usage_async

        if not has_account_usage(slug):
            return
        refresh_account_usage_async([slug])
    except Exception:  # health: allow BLE001 -- a failed refresh must not break the bar
        return


def plan_segments(
    provider: Optional[str], field_set: Optional[frozenset],
) -> tuple[list[list[tuple[str, str]]], bool]:
    """The plan-allowance segments for the status bar, honouring the field filter.

    A hidden field means the provider is never asked either — the refresh only happens
    for a segment that will actually be drawn. Never raises: a status bar that cannot
    read an allowance shows none, rather than failing the repaint.

    Returns the segments and whether one carries a wall-clock reset time.
    """
    try:
        if field_set is not None and FIELD not in field_set:
            return [], False
        refresh_remaining(provider)
        return remaining_segments_with_clock(provider)
    except Exception:  # health: allow BLE001 -- an unreadable allowance must not break the bar
        return [], False
