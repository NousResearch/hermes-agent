"""`display.quota` — the one place the quota read-out's modes are defined.

The TUI status bar, the ``/quota`` slash command on both surfaces, and the
config docs all resolve a user-typed value through here, so a new mode never
has to be spelled out (and spelled differently) in three places.

``ui-tui/src/app/useConfigSync.ts`` mirrors this table for its own read of
config.yaml (the gateway only ever writes canonical values back); a new mode
or alias belongs in both.
"""

from __future__ import annotations

from typing import Optional

# Canonical modes, in the order /quota lists them.
QUOTA_MODES: tuple[str, ...] = ("session", "both", "weekly", "tightest", "off")

DEFAULT_QUOTA_MODE = "session"

_ALIASES = {
    "5h": "session",
    "all": "both",
    "both": "both",
    "hidden": "off",
    "no": "off",
    "none": "off",
    "off": "off",
    "on": "session",
    "session": "session",
    "short": "session",
    "tightest": "tightest",
    "week": "weekly",
    "weekly": "weekly",
    "yes": "session",
}

_DESCRIPTIONS = {
    "session": "the short rolling window: percent left + reset countdown",
    "both": "the session window, with the weekly cap appended last",
    "weekly": "the weekly cap only",
    "tightest": "whichever window is closest to spent",
    "off": "hidden everywhere; the provider is not polled for quota",
}

# Shown when no live snapshot is available, so the shapes are still legible.
_SAMPLE = {"session": (100, "2h 13m"), "weekly": (81, "5d 0h")}


def normalize_quota_mode(value: object) -> Optional[str]:
    """Resolve a config value / slash argument to a canonical mode.

    Returns ``None`` for anything unrecognized so callers can show usage
    instead of silently picking a mode the user did not ask for. ``False``
    (the YAML boolean) reads as ``off``, matching the other display toggles.
    """
    if value is False:
        return "off"
    if value is True:
        return DEFAULT_QUOTA_MODE
    if not isinstance(value, str):
        return None

    return _ALIASES.get(value.strip().lower())


def describe_quota_mode(mode: str) -> str:
    """One-line description of a canonical mode, for /quota output."""
    return _DESCRIPTIONS.get(mode, mode)


def quota_usage() -> str:
    """Usage line shared by both surfaces."""
    return "Usage: /quota [" + "|".join(QUOTA_MODES) + "|status]"


def format_reset_in(reset_at) -> str:
    """Coarse countdown to a reset instant: ``12m``, ``2h 45m``, ``5d 3h``.

    Mirrors ``formatResetIn`` in ``ui-tui/src/app/useAccountUsagePoll.ts`` so
    the /quota examples read exactly like the live read-out.
    """
    if reset_at is None:
        return ""
    try:
        from datetime import datetime, timezone

        target = reset_at
        if isinstance(target, str):
            target = datetime.fromisoformat(target)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        # Rounded, not floored: ``formatResetIn`` rounds, and a countdown that
        # reads a minute short of the live segment is worse than no example.
        minutes = round(max(0.0, (target - datetime.now(timezone.utc)).total_seconds()) / 60)
    except Exception:
        return ""

    if minutes < 1:
        return "now"
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"

    return f"{hours}h {mins}m" if hours else f"{mins}m"


def _segment(pct: int, reset: str) -> str:
    return f"◔ {pct}%" + (f" {reset}" if reset else "")


def render_quota_menu(current: str, windows: Optional[dict] = None) -> str:
    """The /quota picker: every mode with a rendered example, current one marked.

    ``windows`` maps ``"session"`` / ``"weekly"`` to ``(percent_left, reset_label)``
    from the live snapshot; whatever is missing falls back to sample numbers so
    the shape is still clear on a provider with no quota API.
    """
    live = {k: v for k, v in (windows or {}).items() if v}
    sess_pct, sess_reset = live.get("session") or _SAMPLE["session"]
    week_pct, week_reset = live.get("weekly") or _SAMPLE["weekly"]
    session_seg = _segment(int(round(sess_pct)), sess_reset)
    weekly_seg = _segment(int(round(week_pct)), week_reset)
    # The live segment carries ONE glyph: the trailing window is appended bare.
    weekly_tail = weekly_seg.removeprefix("◔ ")

    examples = {
        "session": session_seg,
        "both": f"{session_seg} · {weekly_tail}",
        "weekly": weekly_seg,
        "tightest": weekly_seg if week_pct <= sess_pct else session_seg,
        "off": "(nothing)",
    }

    cmd_w = max(len(m) for m in QUOTA_MODES) + len("/quota ")
    ex_w = max(len(e) for e in examples.values())
    lines = [f"  Quota read-out — currently: {current}", ""]
    for mode in QUOTA_MODES:
        marker = "▸" if mode == current else " "
        cmd = f"/quota {mode}".ljust(cmd_w)
        lines.append(f"  {marker} {cmd}  {examples[mode].ljust(ex_w)}  {_DESCRIPTIONS[mode]}")
    lines += [
        "",
        "  Examples use "
        + ("your current limits." if live else "sample numbers — no live limits to show yet."),
        "  The branding panel always lists every window; this picks the status-bar segment.",
    ]

    return "\n".join(lines)
