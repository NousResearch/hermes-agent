"""Rate limit tracking for inference API responses.

Captures x-ratelimit-* headers from provider responses and provides
formatted display for the /usage slash command.  Currently supports
the Nous Portal header format (also used by OpenRouter and OpenAI-compatible
APIs that follow the same convention).

Header schema (12 headers total):
    x-ratelimit-limit-requests          RPM cap
    x-ratelimit-limit-requests-1h       RPH cap
    x-ratelimit-limit-tokens            TPM cap
    x-ratelimit-limit-tokens-1h         TPH cap
    x-ratelimit-remaining-requests      requests left in minute window
    x-ratelimit-remaining-requests-1h   requests left in hour window
    x-ratelimit-remaining-tokens        tokens left in minute window
    x-ratelimit-remaining-tokens-1h     tokens left in hour window
    x-ratelimit-reset-requests          seconds until minute request window resets
    x-ratelimit-reset-requests-1h       seconds until hour request window resets
    x-ratelimit-reset-tokens            seconds until minute token window resets
    x-ratelimit-reset-tokens-1h         seconds until hour token window resets
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional


_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")
_RPM_THROTTLE_PROVIDERS = frozenset({"anthropic", "openai", "nous"})
_RPM_THROTTLE_THRESHOLD = 2
_RPM_WAIT_SLICE_SECONDS = 0.25


@dataclass
class RateLimitBucket:
    """One rate-limit window (e.g. requests per minute)."""

    limit: int = 0
    remaining: int = 0
    reset_seconds: float = 0.0
    captured_at: float = 0.0  # time.time() when this was captured
    has_remaining: bool = False

    @property
    def used(self) -> int:
        return max(0, self.limit - self.remaining)

    @property
    def usage_pct(self) -> float:
        if self.limit <= 0:
            return 0.0
        return (self.used / self.limit) * 100.0

    @property
    def remaining_seconds_now(self) -> float:
        """Estimated seconds remaining until reset, adjusted for elapsed time."""
        elapsed = time.time() - self.captured_at
        return max(0.0, self.reset_seconds - elapsed)


@dataclass
class RateLimitState:
    """Full rate-limit state parsed from response headers."""

    requests_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    requests_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_min: RateLimitBucket = field(default_factory=RateLimitBucket)
    tokens_hour: RateLimitBucket = field(default_factory=RateLimitBucket)
    captured_at: float = 0.0  # when the headers were captured
    provider: str = ""
    base_url: str = ""

    @property
    def has_data(self) -> bool:
        return self.captured_at > 0

    @property
    def age_seconds(self) -> float:
        if not self.has_data:
            return float("inf")
        return time.time() - self.captured_at


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalized(value: object) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _parse_reset_seconds(value: Any, *, now: float) -> float:
    """Parse numeric, compact-duration, or RFC 3339 reset values."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return max(0.0, float(text))
    except ValueError:
        pass

    compact = text.replace(" ", "")
    parts = list(_DURATION_PART_RE.finditer(compact))
    if parts and "".join(part.group(0) for part in parts) == compact:
        unit_seconds = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
        return sum(
            float(part.group("value")) * unit_seconds[part.group("unit")]
            for part in parts
        )

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        return 0.0
    return max(0.0, parsed.astimezone(timezone.utc).timestamp() - now)


def parse_rate_limit_headers(
    headers: Mapping[str, str],
    provider: str = "",
    base_url: str = "",
) -> Optional[RateLimitState]:
    """Parse OpenAI-compatible or Anthropic rate-limit headers into state.

    Returns None if no rate limit headers are present.
    """
    # Normalize to lowercase so lookups work regardless of how the server
    # capitalises headers (HTTP header names are case-insensitive per RFC 7230).
    lowered = {k.lower(): v for k, v in headers.items()}

    has_any = any(
        k.startswith(("x-ratelimit-", "anthropic-ratelimit-")) for k in lowered
    )
    if not has_any:
        return None

    now = time.time()

    def _bucket(resource: str, suffix: str = "") -> RateLimitBucket:
        # e.g. resource="requests", suffix="" -> per-minute
        #      resource="tokens", suffix="-1h" -> per-hour
        tag = f"{resource}{suffix}"
        remaining_key = f"x-ratelimit-remaining-{tag}"
        return RateLimitBucket(
            limit=_safe_int(lowered.get(f"x-ratelimit-limit-{tag}")),
            remaining=_safe_int(lowered.get(remaining_key)),
            reset_seconds=_parse_reset_seconds(
                lowered.get(f"x-ratelimit-reset-{tag}"), now=now
            ),
            captured_at=now,
            has_remaining=remaining_key in lowered,
        )

    def _anthropic_bucket(resource: str) -> RateLimitBucket:
        prefix = f"anthropic-ratelimit-{resource}"
        remaining_key = f"{prefix}-remaining"
        return RateLimitBucket(
            limit=_safe_int(lowered.get(f"{prefix}-limit")),
            remaining=_safe_int(lowered.get(remaining_key)),
            reset_seconds=_parse_reset_seconds(lowered.get(f"{prefix}-reset"), now=now),
            captured_at=now,
            has_remaining=remaining_key in lowered,
        )

    is_anthropic = any(k.startswith("anthropic-ratelimit-") for k in lowered)

    return RateLimitState(
        requests_min=(
            _anthropic_bucket("requests") if is_anthropic else _bucket("requests")
        ),
        requests_hour=_bucket("requests", "-1h"),
        tokens_min=(
            _anthropic_bucket("tokens") if is_anthropic else _bucket("tokens")
        ),
        tokens_hour=_bucket("tokens", "-1h"),
        captured_at=now,
        provider=provider,
        base_url=_normalized(base_url),
    )


def wait_for_low_rpm(
    state: Optional[RateLimitState],
    *,
    provider: str,
    base_url: str,
    threshold: Optional[int] = None,
    is_interrupted: Optional[Callable[[], bool]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> float:
    """Wait for a matching, nearly exhausted request bucket to reset.

    Captured state is local to one agent and is only trusted for the same
    provider route. Unknown providers may opt in through an explicit threshold.
    """
    active_provider = _normalized(provider)
    if (
        state is None
        or not state.has_data
        or not active_provider
        or _normalized(state.provider) != active_provider
        or _normalized(state.base_url) != _normalized(base_url)
    ):
        return 0.0
    if threshold is None and active_provider not in _RPM_THROTTLE_PROVIDERS:
        return 0.0
    if isinstance(threshold, bool):
        return 0.0
    try:
        effective_threshold = (
            _RPM_THROTTLE_THRESHOLD if threshold is None else int(threshold)
        )
    except (TypeError, ValueError):
        return 0.0
    bucket = state.requests_min
    if (
        effective_threshold < 0
        or not bucket.has_remaining
        or bucket.limit <= 0
        or bucket.remaining > effective_threshold
    ):
        return 0.0
    wait_seconds = bucket.remaining_seconds_now
    if wait_seconds <= 0:
        return 0.0

    deadline = monotonic_fn() + wait_seconds
    while not (callable(is_interrupted) and is_interrupted()):
        remaining = deadline - monotonic_fn()
        if remaining <= 0:
            return wait_seconds
        sleep_fn(min(_RPM_WAIT_SLICE_SECONDS, remaining))
    return 0.0


# ── Formatting ──────────────────────────────────────────────────────────


def _fmt_count(n: int) -> str:
    """Human-friendly number: 7999856 -> '8.0M', 33599 -> '33.6K', 799 -> '799'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
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
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


def _bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar: [████████░░░░░░░░░░░░] 40%."""
    filled = int(pct / 100.0 * width)
    filled = max(0, min(width, filled))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _bucket_line(label: str, bucket: RateLimitBucket, label_width: int = 14) -> str:
    """Format one bucket as a single line."""
    if bucket.limit <= 0:
        return f"  {label:<{label_width}}  (no data)"

    pct = bucket.usage_pct
    used = _fmt_count(bucket.used)
    limit = _fmt_count(bucket.limit)
    remaining = _fmt_count(bucket.remaining)
    reset = _fmt_seconds(bucket.remaining_seconds_now)

    bar = _bar(pct)
    return f"  {label:<{label_width}} {bar} {pct:5.1f}%  {used}/{limit} used  ({remaining} left, resets in {reset})"


def format_rate_limit_display(state: RateLimitState) -> str:
    """Format rate limit state for terminal/chat display."""
    if not state.has_data:
        return "No rate limit data yet — make an API request first."

    age = state.age_seconds
    if age < 5:
        freshness = "just now"
    elif age < 60:
        freshness = f"{int(age)}s ago"
    else:
        freshness = f"{_fmt_seconds(age)} ago"

    provider_label = state.provider.title() if state.provider else "Provider"

    lines = [
        f"{provider_label} Rate Limits (captured {freshness}):",
        "",
        _bucket_line("Requests/min", state.requests_min),
        _bucket_line("Requests/hr", state.requests_hour),
        "",
        _bucket_line("Tokens/min", state.tokens_min),
        _bucket_line("Tokens/hr", state.tokens_hour),
    ]

    # Add warnings if any bucket is getting hot
    warnings = []
    for label, bucket in [
        ("requests/min", state.requests_min),
        ("requests/hr", state.requests_hour),
        ("tokens/min", state.tokens_min),
        ("tokens/hr", state.tokens_hour),
    ]:
        if bucket.limit > 0 and bucket.usage_pct >= 80:
            reset = _fmt_seconds(bucket.remaining_seconds_now)
            warnings.append(f"  ⚠ {label} at {bucket.usage_pct:.0f}% — resets in {reset}")

    if warnings:
        lines.append("")
        lines.extend(warnings)

    return "\n".join(lines)


def format_rate_limit_compact(state: RateLimitState) -> str:
    """One-line compact summary for status bars / gateway messages."""
    if not state.has_data:
        return "No rate limit data."

    rm = state.requests_min
    tm = state.tokens_min
    rh = state.requests_hour
    th = state.tokens_hour

    parts = []
    if rm.limit > 0:
        parts.append(f"RPM: {rm.remaining}/{rm.limit}")
    if rh.limit > 0:
        parts.append(f"RPH: {_fmt_count(rh.remaining)}/{_fmt_count(rh.limit)} (resets {_fmt_seconds(rh.remaining_seconds_now)})")
    if tm.limit > 0:
        parts.append(f"TPM: {_fmt_count(tm.remaining)}/{_fmt_count(tm.limit)}")
    if th.limit > 0:
        parts.append(f"TPH: {_fmt_count(th.remaining)}/{_fmt_count(th.limit)} (resets {_fmt_seconds(th.remaining_seconds_now)})")

    return " | ".join(parts)
