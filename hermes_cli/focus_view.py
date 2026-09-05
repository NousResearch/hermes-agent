"""Focus view — display-only reduced output: "just my prompt and the answer, and tell me what you hid".
ON snaps ``tool_progress_mode`` to ``"off"`` and remembers the configured mode so the *existing* suppression
path does the hiding; OFF restores it verbatim. Focus adds a per-turn hidden-line count with a recovery
hint and a persistent ``focus`` status-bar segment."""

from __future__ import annotations

from typing import Optional

FOCUS_CONFIG_KEY = "display.focus_view"  # plain boolean under ``display``, like /battery /timestamps /footer
FOCUS_TOOL_PROGRESS_MODE = "off"  # the SAME value ``/verbose off`` uses so both features share one suppression path
# Modes in which the CLI commits a per-tool scrollback line. Mirrors the gate in
# ``HermesCLI._on_tool_progress`` so the hidden-line counter and the renderer never drift apart.
TOOL_PROGRESS_VISIBLE_MODES = frozenset({"new", "all", "verbose"})
TOOL_PROGRESS_MODES = ("off", "new", "all", "verbose")  # ``log`` is a gateway-only extra step
FOCUS_STATUSBAR_LABEL = "◉ focus"  # short on purpose — the bar is width-constrained

# /focus argument words -> (action, target); bare /focus toggles like /footer, /battery, /timestamps.
_FOCUS_WORDS = {
    **dict.fromkeys(("status", "show", "?"), ("status", None)),
    **dict.fromkeys(("on", "enable", "enabled", "true", "yes", "1"), ("set", True)),
    **dict.fromkeys(("off", "disable", "disabled", "false", "no", "0"), ("set", False)),
}


def normalize_tool_progress_mode(mode: object, default: str = "all") -> str:
    """Coerce a raw config/attr value into a known tool-progress mode."""
    if mode is False:
        return "off"
    if mode is True:
        return "all"
    text = str(mode or "").strip().lower()
    # ``log`` is a real gateway mode; any other unknown value becomes the default.
    return text if text in TOOL_PROGRESS_MODES or text == "log" else default


def resolve_focus_arg(arg: str, current: bool) -> tuple[str, Optional[bool]]:
    """Map a ``/focus`` argument onto ``(action, target)``: action is ``"set"``, ``"status"`` or ``"usage"``;
    target is the requested enabled-state for ``"set"`` and ``None`` otherwise."""
    text = str(arg or "").strip().lower()
    if text in ("", "toggle"):
        return "set", not bool(current)
    return _FOCUS_WORDS.get(text, ("usage", None))


def would_display_tool_line(mode: object, function_name: str, last_tool_name: Optional[str] = None) -> bool:
    """Would the CLI have committed a scrollback line for this tool call? Counts honestly: with ``/verbose off``
    focus view hides nothing extra and must not claim otherwise. ``new`` mode skips consecutive repeats of the
    same tool, so the counter does too."""
    if not function_name:
        return False
    normalized = normalize_tool_progress_mode(mode)
    return normalized in TOOL_PROGRESS_VISIBLE_MODES and not (normalized == "new" and function_name == last_tool_name)


def format_hidden_line(count: int) -> Optional[str]:
    """Dim post-turn recovery line, or ``None`` when nothing was hidden."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        return None
    return f"⋯ {n} {'tool line' if n == 1 else 'tool lines'} hidden · /focus off to show" if n > 0 else None


def focus_statusbar_segment(enabled: bool) -> str:
    """Status-bar segment text for focus view (empty when off)."""
    return FOCUS_STATUSBAR_LABEL if enabled else ""


def format_focus_status(enabled: bool, configured_mode: object) -> str:
    """Human-readable ``/focus status`` body (no ANSI — callers colour it)."""
    mode = normalize_tool_progress_mode(configured_mode).upper()
    if enabled:
        return f"Focus view: ON — only your prompt and the final response.\n  /focus off restores tool progress: {mode}"
    return f"Focus view: OFF — tool progress: {mode}"


def format_focus_toggle_message(enabled: bool, configured_mode: object) -> str:
    """Confirmation line printed when focus view is switched (no ANSI)."""
    if enabled:
        return "Focus view enabled — just your prompt and the final response"
    return f"Focus view disabled — tool progress: {normalize_tool_progress_mode(configured_mode).upper()}"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

FOCUS_USAGE = "Usage: /focus [on|off|status]"

def effective_tool_progress_mode(focus_enabled: bool, configured_mode: object) -> str:
    """Return the tool-progress mode that should actually be in force.

    Focus view wins while it is on (it *is* "tool progress off" plus reporting).
    When focus is off the user's configured mode is returned untouched — this is
    what makes ``/focus off`` restore ``/verbose verbose`` rather than clobbering
    it to ``all``.
    """
    normalized = normalize_tool_progress_mode(configured_mode)
    if focus_enabled:
        return FOCUS_TOOL_PROGRESS_MODE
    return normalized
# ---- END PLUGIN-COMPAT ----
