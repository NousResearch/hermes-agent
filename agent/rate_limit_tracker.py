"""Rate limit tracking for inference API responses.

Captures ``x-ratelimit-{limit,remaining,reset}-{requests,tokens}[-1h]``
headers (Nous Portal format, also used by OpenRouter / OpenAI-compatible APIs)
and formats them for the /usage slash command. Reset values are seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class CodexWindow:
    used_percent: float = 0.0
    window_minutes: int = 0
    reset_after_seconds: float = 0.0


@dataclass
class CodexRateLimitState:
    primary: CodexWindow = field(default_factory=CodexWindow)
    secondary: CodexWindow = field(default_factory=CodexWindow)
    active_limit: str = ""
    plan_type: str = ""
    credits_balance: str = ""
    credits_has_credits: Optional[bool] = None
    credits_unlimited: Optional[bool] = None
    captured_at: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.captured_at if self.has_data else float("inf")

# (state attribute, header tag) for the four windows.
_BUCKET_TAGS = (
    ("requests_min", "requests"),
    ("requests_hour", "requests-1h"),
    ("tokens_min", "tokens"),
    ("tokens_hour", "tokens-1h"),
)


@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests per minute)."""

    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0  # time.time() when this was captured

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        return (self.used / self.limit) * 100.0 if self.limit > 0 else 0.0

    @property
    def remaining_seconds_now(self) -> float:
        """Estimated seconds remaining until reset, adjusted for elapsed time."""
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


def parse_codex_headers(headers: Mapping[str, str]) -> Optional[CodexRateLimitState]:
    """Parse the OpenAI Codex ``x-codex-*`` response-header family."""
    lowered = lower_headers(headers)
    if not any(k.startswith("x-codex-") for k in lowered):
        return None

    def boolean(name: str) -> Optional[bool]:
        value = lowered.get(name)
        return str(value).strip().lower() in {"1", "true", "yes"} if value is not None else None

    def window(prefix: str) -> CodexWindow:
        return CodexWindow(
            used_percent=_safe_float(lowered.get(f"x-codex-{prefix}-used-percent")),
            window_minutes=_safe_int(lowered.get(f"x-codex-{prefix}-window-minutes")),
            reset_after_seconds=_safe_float(lowered.get(f"x-codex-{prefix}-reset-after-seconds")),
        )

    return CodexRateLimitState(
        primary=window("primary"), secondary=window("secondary"),
        active_limit=lowered.get("x-codex-active-limit", ""),
        plan_type=lowered.get("x-codex-plan-type", ""),
        credits_balance=lowered.get("x-codex-credits-balance", ""),
        credits_has_credits=boolean("x-codex-credits-has-credits"),
        credits_unlimited=boolean("x-codex-credits-unlimited"), captured_at=time.time(),
    )


def parse_rate_limit_headers(headers: Mapping[str, str], provider: str = "") -> Optional[RateLimitState]:
    """Parse x-ratelimit-* headers into a RateLimitState (None if none present)."""
    lowered = lower_headers(headers)
    if not has_rate_limit_headers(lowered):
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
    return RateLimitState(captured_at=now, provider=provider, **buckets)


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
    lines += [_bucket_line(label, bucket) for label, bucket in labeled[:2]]
    lines += [""] + [_bucket_line(label, bucket) for label, bucket in labeled[2:]]

    warnings = [
        f"  ⚠ {label.lower()} at {bucket.usage_pct:.0f}% — resets in {_fmt_seconds(bucket.remaining_seconds_now)}"
        for label, bucket in labeled
        if bucket.limit > 0 and bucket.usage_pct >= 80
    ]
    if warnings:
        lines += [""] + warnings
    return "\n".join(lines)


def format_codex_rate_limit_display(state: CodexRateLimitState) -> str:
    """Format Codex plan windows for /usage."""
    def window(label: str, item: CodexWindow) -> str:
        reset = _fmt_seconds(item.reset_after_seconds)
        return f"  {label:<12} {item.used_percent:5.1f}% used (resets in {reset})"

    lines = ["OpenAI-Codex Plan Limits:"]
    if state.plan_type:
        lines.append(f"Plan: {state.plan_type}" + (f" (limit: {state.active_limit})" if state.active_limit else ""))
    if state.credits_unlimited is True:
        lines.append("Credits: unlimited")
    elif state.credits_has_credits is not None:
        lines.append(f"Credits: {'available' if state.credits_has_credits else 'unavailable'}")
    lines.extend([window("5h window", state.primary), window("7d window", state.secondary)])
    return "\\n".join(lines)


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
    return " | ".join(
        f"{tag}: {fmt(b.remaining)}/{fmt(b.limit)}" + (f" (resets {_fmt_seconds(b.remaining_seconds_now)})" if reset else "")
        for tag, b, fmt, reset in windows if b.limit > 0
    )
