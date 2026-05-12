"""Context-window HUD formatter for Telegram replies.

Pure formatting — no agent state, no platform calls.  The gateway wires this
into outbound Telegram messages via the stream consumer.

Output shape (two lines, ASCII bar, no emoji):

    23k / 250k  9%
    [██░░░░░░░░░░░░░░░░░░]

Returns ``None`` when the HUD should be hidden (low usage, missing context
length, etc.).  Callers treat ``None`` as "do not prepend anything".
"""

from __future__ import annotations

from typing import Optional

__all__ = ["compact_tokens", "format_hud"]


def compact_tokens(n: int) -> str:
    """Render a token count compactly: 1k, 23k, 250k, 1.2M.

    Negative inputs clamp to 0.  Values below 1000 render as plain integers.
    """
    if n < 0:
        n = 0
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n // 1_000}k"
    return f"{n / 1_000_000:.1f}M"


def _label_for_percent(
    pct: int,
    warn_percent: float,
    danger_percent: float,
    critical_percent: float,
) -> str:
    """Pick a one-word label for the current usage band, or '' for normal."""
    if pct >= critical_percent:
        return "critical"
    if pct >= danger_percent:
        return "danger"
    if pct >= warn_percent:
        return "warn"
    return ""


def format_hud(
    used_tokens: int,
    context_length: int,
    *,
    bar_width: int = 20,
    hide_below_percent: float = 5,
    warn_percent: float = 60,
    danger_percent: float = 75,
    critical_percent: float = 90,
    show_warning_label: bool = False,
) -> Optional[str]:
    """Format a two-line context HUD, or return ``None`` when hidden.

    Args:
        used_tokens: estimated prompt tokens consumed so far this turn.
        context_length: model's context window.  Non-positive values disable
            the HUD entirely (returns ``None``).
        bar_width: number of cells in the progress bar.
        hide_below_percent: when the computed percentage is strictly below
            this threshold, return ``None`` to keep low-usage replies clean.
            Set to ``0`` to always render.
        warn_percent / danger_percent / critical_percent: bucket thresholds
            for the optional label suffix.
        show_warning_label: when True, append ' warn' / ' danger' / ' critical'
            after the percentage so high-usage states stand out.  Off by
            default to keep the HUD compact.
    """
    if context_length <= 0:
        return None
    if bar_width <= 0:
        bar_width = 1
    if used_tokens < 0:
        used_tokens = 0

    pct_raw = used_tokens / context_length * 100.0
    pct = int(round(pct_raw))
    if pct < 0:
        pct = 0
    if pct > 100:
        pct = 100

    if hide_below_percent > 0 and pct < hide_below_percent:
        return None

    filled = int(round(pct / 100.0 * bar_width))
    if filled < 0:
        filled = 0
    if filled > bar_width:
        filled = bar_width
    bar = "█" * filled + "░" * (bar_width - filled)

    used_label = compact_tokens(used_tokens)
    limit_label = compact_tokens(context_length)
    first = f"{used_label} / {limit_label}  {pct}%"

    if show_warning_label:
        label = _label_for_percent(pct, warn_percent, danger_percent, critical_percent)
        if label:
            first = f"{first} {label}"

    return f"{first}\n[{bar}]"
