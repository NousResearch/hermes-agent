"""Rate limit tracking for inference API responses.

Two header families feed one ``RateLimitState``:

- ``x-ratelimit-{limit,remaining,reset}-{requests,tokens}[-1h]`` (Nous Portal format, also
  used by OpenRouter / OpenAI-compatible APIs). Reset values are a seconds OFFSET.
- ``anthropic-ratelimit-unified-[<window>-]{limit,remaining,reset,status}``, which Anthropic
  sends instead on Claude Pro/Max subscription (OAuth) responses. Reset values are an
  ABSOLUTE instant (ISO-8601 or unix epoch), and the subscription windows are anchored to
  the session's first request rather than to a fixed clock grid — so the instant is stored
  as sent and never floored into a bucket.

Parsing and display only: nothing here decides when a request is sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# (state attribute, header tag) for the four windows.
_BUCKET_TAGS = (
    ("requests_min", "requests"),
    ("requests_hour", "requests-1h"),
    ("tokens_min", "tokens"),
    ("tokens_hour", "tokens-1h"),
)

_UNIFIED_PREFIX = "anthropic-ratelimit-unified-"
_UNIFIED_FIELDS = frozenset({"limit", "remaining", "reset", "status"})
# The window-less form (``anthropic-ratelimit-unified-status``) is the account-wide verdict.
_UNIFIED_OVERALL = "overall"


@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests per minute, or a subscription 5-hour window)."""

    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0  # time.time() when this was captured
    # Absolute epoch seconds of the reset, for providers that report an instant rather than an
    # offset (Anthropic's unified windows). 0.0 means "not reported"; ``reset_seconds`` applies.
    reset_at: float = 0.0
    # Provider's own verdict for this window, when it sends one (Anthropic: allowed /
    # allowed_warning / rejected). Opaque here — recorded, never interpreted.
    status: str = ""

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        return (self.used / self.limit) * 100.0 if self.limit > 0 else 0.0

    @property
    def remaining_seconds_now(self) -> float:
        """Estimated seconds remaining until reset.

        An absolute ``reset_at`` is authoritative and needs no aging; an offset-style
        ``reset_seconds`` is aged against ``captured_at``.
        """
        if self.reset_at > 0:
            return max(0.0, self.reset_at - time.time())
        return max(0.0, self.reset_seconds - (time.time() - self.captured_at))


@dataclass
class RateLimitState:
    """Full rate-limit state parsed from response headers."""

    requests_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at: float = 0.0  # when the headers were captured
    provider: str = ""
    # Anthropic unified windows keyed by header tag ("overall", "5h", "7d", "7d-opus", ...).
    # A dict, not named fields: Anthropic adds windows without notice and each is self-describing.
    unified: dict[str, RateLimitBucket] = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")


def _safe_float(value: Any, default: Any = 0.0) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Any = 0) -> Any:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def lower_headers(headers: Optional[Mapping[str, str]]) -> dict[str, str]:
    """Lowercase header names (HTTP header names are case-insensitive)."""
    return {k.lower(): v for k, v in headers.items()} if headers else {}


def has_rate_limit_headers(lowered: Mapping[str, str]) -> bool:
    return any(k.startswith("x-ratelimit-") for k in lowered)


def has_unified_rate_limit_headers(lowered: Mapping[str, str]) -> bool:
    """True for Anthropic's unified (subscription) rate-limit family.

    Kept separate from :func:`has_rate_limit_headers` because callers that then read
    ``x-ratelimit-*`` keys by name (``nous_rate_guard``) must not be told those exist.
    """
    return any(k.startswith(_UNIFIED_PREFIX) for k in lowered)


def _reset_instant(value: Any) -> float:
    """Absolute epoch seconds from a unified ``*-reset`` value, or 0.0 when unparseable.

    Anthropic sends ISO-8601 here (``agent_runtime_helpers`` already reads its per-bucket
    ``anthropic-ratelimit-*-reset`` that way) but documents epoch seconds for some windows;
    the shared parser accepts both and is the one place that knowledge lives.
    """
    from agent.credential_pool import _parse_absolute_timestamp

    return _parse_absolute_timestamp(value) or 0.0


def parse_unified_rate_limit_buckets(lowered: Mapping[str, str]) -> dict[str, RateLimitBucket]:
    """``anthropic-ratelimit-unified-[<window>-]<field>`` headers → one bucket per window.

    Windows are discovered from the headers themselves ("5h", "7d", "7d-opus", and the
    window-less account-wide form) rather than enumerated, so a window Anthropic adds later
    is captured without a code change. Non-window members of the family
    (``-fallback-percentage``) have no recognised field suffix and are skipped.
    """
    now = time.time()
    by_window: dict[str, dict[str, str]] = {}
    for name, value in lowered.items():
        if not name.startswith(_UNIFIED_PREFIX):
            continue
        window, _, field_name = name[len(_UNIFIED_PREFIX):].rpartition("-")
        if field_name in _UNIFIED_FIELDS:
            by_window.setdefault(window or _UNIFIED_OVERALL, {})[field_name] = value
    return {
        window: RateLimitBucket(
            limit=_safe_int(values.get("limit")),
            remaining=_safe_int(values.get("remaining")),
            reset_at=_reset_instant(values.get("reset")),
            status=(values.get("status") or "").strip(),
            captured_at=now,
        )
        for window, values in by_window.items()
    }


def parse_rate_limit_headers(headers: Mapping[str, str], provider: str = "") -> Optional[RateLimitState]:
    """Parse ``x-ratelimit-*`` and ``anthropic-ratelimit-unified-*`` headers into a
    RateLimitState (None when neither family is present)."""
    lowered = lower_headers(headers)
    if not has_rate_limit_headers(lowered) and not has_unified_rate_limit_headers(lowered):
        return None

    now = time.time()
    buckets = {
        attr: RateLimitBucket(
            limit=_safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining=_safe_int(lowered.get(f"x-ratelimit-remaining-{tag}")),
            reset_seconds=_safe_float(lowered.get(f"x-ratelimit-reset-{tag}")),
            captured_at=now,
        )
        for attr, tag in _BUCKET_TAGS
    }
    return RateLimitState(
        captured_at=now, provider=provider,
        unified=parse_unified_rate_limit_buckets(lowered), **buckets,
    )


# ── Formatting ──────────────────────────────────────────────────────────


def _fmt_count(n: int) -> str:
    """Human-friendly number: 7999856 -> '8.0M', 33599 -> '33.6K', 799 -> '799'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_seconds(seconds: float) -> str:
    """Seconds -> human-friendly duration: '58s', '2m 14s', '58m 57s', '1h 2m'."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, m = divmod(s, 3600)
    m //= 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar: [████████░░░░░░░░░░░░] 40%."""
    filled = max(0, min(width, int(pct / 100.0 * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _bucket_line(label: str, bucket: RateLimitBucket, label_width: int = 14) -> str:
    """Format one bucket as a single line."""
    if bucket.limit <= 0:
        return f"  {label:<{label_width}}  (no data)"
    pct = bucket.usage_pct
    used, limit, remaining = map(_fmt_count, (bucket.used, bucket.limit, bucket.remaining))
    reset = _fmt_seconds(bucket.remaining_seconds_now)
    return f"  {label:<{label_width}} {_bar(pct)} {pct:5.1f}%  {used}/{limit} used  ({remaining} left, resets in {reset})"


# Display names for the unified windows we have seen; an unrecognised tag renders as sent so a
# newly introduced window is still visible rather than silently dropped.
_UNIFIED_WINDOW_LABELS = {_UNIFIED_OVERALL: "Account", "5h": "Session/5h", "7d": "Weekly/7d"}


def _unified_label(window: str) -> str:
    return _UNIFIED_WINDOW_LABELS.get(window, window)


def _sorted_unified(state: RateLimitState) -> list[tuple[str, RateLimitBucket]]:
    """Unified windows, account-wide first then alphabetical, for a stable display order."""
    return sorted(state.unified.items(), key=lambda kv: (kv[0] != _UNIFIED_OVERALL, kv[0]))


def _unified_line(window: str, bucket: RateLimitBucket, label_width: int = 14) -> str:
    """Format one unified window; windows that report only a status still get a line."""
    label = _unified_label(window)
    if bucket.limit > 0:
        line = _bucket_line(label, bucket, label_width)
    else:
        reset = f"resets in {_fmt_seconds(bucket.remaining_seconds_now)}" if bucket.reset_at > 0 else "no limit reported"
        line = f"  {label:<{label_width}}  ({reset})"
    return f"{line}  [{bucket.status}]" if bucket.status else line


def format_rate_limit_display(state: RateLimitState) -> str:
    """Format rate limit state for terminal/chat display."""
    if not state.has_data:
        return "No rate limit data yet — make an API request first."

    age = state.age_seconds
    freshness = "just now" if age < 5 else f"{int(age)}s ago" if age < 60 else f"{_fmt_seconds(age)} ago"

    provider_label = state.provider.title() if state.provider else "Provider"
    labeled = [("Requests/min", state.requests_min), ("Requests/hr", state.requests_hour),
               ("Tokens/min", state.tokens_min), ("Tokens/hr", state.tokens_hour)]
    lines = [f"{provider_label} Rate Limits (captured {freshness}):", ""]
    # A unified-only response (Anthropic subscription) carries none of these, so skip the
    # block entirely rather than printing four "(no data)" rows.
    if any(bucket.limit > 0 for _label, bucket in labeled):
        lines += [_bucket_line(label, bucket) for label, bucket in labeled[:2]]
        lines += [""] + [_bucket_line(label, bucket) for label, bucket in labeled[2:]]

    unified = _sorted_unified(state)
    if unified:
        lines += ["", "  Subscription windows:"]
        lines += [_unified_line(window, bucket) for window, bucket in unified]

    warnings = [
        f"  ⚠ {label.lower()} at {bucket.usage_pct:.0f}% — resets in {_fmt_seconds(bucket.remaining_seconds_now)}"
        for label, bucket in labeled
        if bucket.limit > 0 and bucket.usage_pct >= 80
    ]
    if warnings:
        lines += [""] + warnings
    return "\n".join(lines)


def format_rate_limit_compact(state: RateLimitState) -> str:
    """One-line compact summary for status bars / gateway messages."""
    if not state.has_data:
        return "No rate limit data."

    # (tag, bucket, count formatter, show reset) — RPM stays raw digits, hourly windows show the reset.
    windows = (
        ("RPM", state.requests_min, str, False),
        ("RPH", state.requests_hour, _fmt_count, True),
        ("TPM", state.tokens_min, _fmt_count, False),
        ("TPH", state.tokens_hour, _fmt_count, True),
    )
    parts = [
        f"{tag}: {fmt(b.remaining)}/{fmt(b.limit)}" + (f" (resets {_fmt_seconds(b.remaining_seconds_now)})" if reset else "")
        for tag, b, fmt, reset in windows if b.limit > 0
    ]
    # Anthropic subscription windows: the reset is an absolute instant, so it always shows.
    # A window that reports only a status (no counters) still gets an entry — it is the whole
    # signal that response carried.
    parts += [
        f"{_unified_label(window)}: "
        + (f"{_fmt_count(b.remaining)}/{_fmt_count(b.limit)}" if b.limit > 0 else b.status)
        + (f" (resets {_fmt_seconds(b.remaining_seconds_now)})" if b.reset_at > 0 else "")
        for window, b in _sorted_unified(state) if b.limit > 0 or b.status
    ]
    return " | ".join(parts)
