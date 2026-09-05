"""Focus view — a display-only reduced-output mode.

``/focus`` answers one question the existing ``/verbose`` cycle cannot:
*"just show me my prompt and the answer — and tell me what you hid."*

``/verbose off`` already silences per-tool progress lines (the
``tool_progress_mode == "off"`` gate in ``agent/tool_executor.py`` and the
scrollback gate in ``HermesCLI._on_tool_progress``).  Focus view **composes
with** that machinery instead of duplicating it:

* turning focus ON snaps ``tool_progress_mode`` to ``"off"`` and remembers the
  mode the user had configured, so the *existing* suppression path does the
  actual hiding;
* turning focus OFF restores that remembered mode verbatim;
* on top of that, focus view adds the two things ``/verbose off`` lacks —
  a per-turn count of what was hidden plus a recovery hint, and a persistent
  ``focus`` segment in the status bar so the reduced mode is never invisible.

Everything in this module is **display-only**.  Nothing here reads or mutates
conversation history, the system prompt, tool schemas, or any request payload.
Flipping focus view must never change a single byte of what is sent to the
model — that invariant is covered by
``tests/cli/test_focus_view.py::test_model_facing_messages_identical_with_focus_on_vs_off``.
"""

from __future__ import annotations

from typing import Optional

# Config key used by the sibling display toggles (/battery, /timestamps,
# /footer) — a plain boolean under ``display``.
FOCUS_CONFIG_KEY = "display.focus_view"

#: Tool-progress mode focus view snaps to. Deliberately the SAME value
#: ``/verbose off`` uses so both features share one suppression path.
FOCUS_TOOL_PROGRESS_MODE = "off"

#: Modes in which the CLI commits a per-tool scrollback line. Mirrors the gate
#: in ``HermesCLI._on_tool_progress``; kept here so the hidden-line counter and
#: the renderer can never drift apart.
TOOL_PROGRESS_VISIBLE_MODES = frozenset({"new", "all", "verbose"})

#: Valid tool-progress modes (``log`` is a gateway-only extra step).
TOOL_PROGRESS_MODES = ("off", "new", "all", "verbose")

#: Status-bar label. Short on purpose — the bar is width-constrained.
FOCUS_STATUSBAR_LABEL = "◉ focus"

_ON_WORDS = frozenset({"on", "enable", "enabled", "true", "yes", "1"})
_OFF_WORDS = frozenset({"off", "disable", "disabled", "false", "no", "0"})
_STATUS_WORDS = frozenset({"status", "show", "?"})
_TOGGLE_WORDS = frozenset({"", "toggle"})

FOCUS_USAGE = "Usage: /focus [on|off|status]"


def normalize_tool_progress_mode(mode: object, default: str = "all") -> str:
    """Coerce a raw config/attr value into a known tool-progress mode.

    YAML 1.1 parses a bare ``off`` as ``False``, and older configs stored
    ``True``/``False`` booleans, so this mirrors ``cli.py``'s normalisation.
    """
    if mode is False:
        return "off"
    if mode is True:
        return "all"
    text = str(mode or "").strip().lower()
    if text in TOOL_PROGRESS_MODES:
        return text
    # ``log`` is a real gateway mode; treat any other unknown value as default.
    if text == "log":
        return "log"
    return default


def resolve_focus_arg(arg: str, current: bool) -> tuple[str, Optional[bool]]:
    """Map a ``/focus`` argument onto an action, following the sibling toggles.

    Returns ``(action, target)`` where ``action`` is one of ``"set"``,
    ``"status"`` or ``"usage"``.  ``target`` is the requested enabled-state for
    ``"set"`` and ``None`` otherwise.  Bare ``/focus`` toggles, matching
    ``/footer`` / ``/battery`` / ``/timestamps``.
    """
    text = str(arg or "").strip().lower()
    if text in _STATUS_WORDS:
        return "status", None
    if text in _ON_WORDS:
        return "set", True
    if text in _OFF_WORDS:
        return "set", False
    if text in _TOGGLE_WORDS:
        return "set", not bool(current)
    return "usage", None


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


def would_display_tool_line(
    mode: object,
    function_name: str,
    last_tool_name: Optional[str] = None,
) -> bool:
    """Would the CLI have committed a scrollback line for this tool call?

    Used to count *honestly*: if the user already had ``/verbose off``, focus
    view is hiding nothing extra and must not claim otherwise.  ``new`` mode
    skips consecutive repeats of the same tool, so the counter skips them too.
    """
    if not function_name:
        return False
    normalized = normalize_tool_progress_mode(mode)
    if normalized not in TOOL_PROGRESS_VISIBLE_MODES:
        return False
    if normalized == "new" and function_name == last_tool_name:
        return False
    return True


def format_hidden_line(count: int) -> Optional[str]:
    """Dim post-turn recovery line, or ``None`` when nothing was hidden."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    noun = "tool line" if n == 1 else "tool lines"
    return f"⋯ {n} {noun} hidden · /focus off to show"


def focus_statusbar_segment(enabled: bool) -> str:
    """Status-bar segment text for focus view (empty when off)."""
    return FOCUS_STATUSBAR_LABEL if enabled else ""


def format_focus_status(enabled: bool, configured_mode: object) -> str:
    """Human-readable ``/focus status`` body (no ANSI — callers colour it)."""
    state = "ON" if enabled else "OFF"
    if enabled:
        restore = normalize_tool_progress_mode(configured_mode)
        return (
            f"Focus view: {state} — only your prompt and the final response.\n"
            f"  /focus off restores tool progress: {restore.upper()}"
        )
    mode = normalize_tool_progress_mode(configured_mode)
    return f"Focus view: {state} — tool progress: {mode.upper()}"


def format_focus_toggle_message(enabled: bool, configured_mode: object) -> str:
    """Confirmation line printed when focus view is switched (no ANSI)."""
    if enabled:
        return "Focus view enabled — just your prompt and the final response"
    mode = normalize_tool_progress_mode(configured_mode)
    return f"Focus view disabled — tool progress: {mode.upper()}"
