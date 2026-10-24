"""Rotating tip line under the classic CLI composer.

When the live-work footer has nothing to list (no subagents, shells, goal, loop or queued prompts)
and no prompt is open, the footer paints one dim ``✦ Tip: …`` line from the startup tip catalog
(``hermes_cli.tips``) and swaps it every ``ROTATE_SECONDS``. Tips that fit the terminal width
win over ones that would be clipped, the last few are never repeated, and the tips for CLI
features users miss most (``FEATURED_TIP_KEYS``) come up more often than the rest.
``display.cli_footer_tips`` turns it off.
"""
from __future__ import annotations

import random
import time
from collections import deque

from prompt_toolkit.utils import get_cwidth

from agent.i18n import t

ROTATE_SECONDS = 30.0
# Never show one of the last N tips again (when the pool leaves a choice).
_RECENT = 8
# Chance that a rotation draws from the featured tips instead of the whole catalog: with a
# 30 s rotation each featured tip comes up every few minutes instead of once in hours.
_FEATURED_SHARE = 1 / 3
# Footer controls, /mouse and /statusbar legend: easy to miss, useful once known.
FEATURED_TIP_KEYS = ("tips.t378", "tips.t379", "tips.t380", "tips.t381", "tips.t382", "tips.t383")
_INDENT = "  "


def _line_text(key: str) -> str:
    return _INDENT + t("cli.tip_line", tip=t(key))


class FooterTips:
    """The current tip and when it was picked; ``rotate`` swaps it for another."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self.key: str | None = None
        self.shown_at = 0.0
        self.columns = 80  # width of the last paint; a rotation picks for it
        self._recent: deque[str] = deque(maxlen=_RECENT)

    def _pick(self, columns: int) -> str | None:
        from hermes_cli.tips import tip_keys
        keys = tip_keys()
        if not keys:
            return None
        # Fit first (a clipped tip only when none fits), then skip the recent ones where that
        # leaves a choice, and never the tip already on screen if any other one fits.
        fitting = [k for k in keys if get_cwidth(_line_text(k)) <= columns] or list(keys)
        fresh = ([k for k in fitting if k not in self._recent]
                 or [k for k in fitting if k != self.key] or fitting)
        featured = [k for k in fresh if k in FEATURED_TIP_KEYS]
        pool = featured if featured and self._rng.random() < _FEATURED_SHARE else fresh
        return self._rng.choice(pool)

    def rotate(self, now: float | None = None) -> bool:
        """Pick the next tip; True when the text changed (the caller repaints only then)."""
        key = self._pick(self.columns)
        self.shown_at = time.monotonic() if now is None else now
        if key is None or key == self.key:
            return False
        self.key = key
        self._recent.append(key)
        return True

    def due(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return self.key is None or now - self.shown_at >= ROTATE_SECONDS

    def line(self, columns: int, now: float | None = None) -> list[tuple[str, str]]:
        """The tip as one footer line of ``(style, text)`` fragments, clipped to ``columns``."""
        from hermes_cli.cli_dock_footer import clip
        columns = max(0, columns - 1)  # the footer never paints the last column (no auto-wrap)
        resized = columns != self.columns
        self.columns = columns
        # First paint, or the terminal shrank under a tip that no longer fits: pick one that does
        # (a paint at an unchanged width never swaps the tip).
        if self.key is None or (resized and get_cwidth(_line_text(self.key)) > columns):
            self.rotate(now)
        if self.key is None:
            return []
        indent = _INDENT[:columns]
        tip = clip(t("cli.tip_line", tip=t(self.key)), columns - len(indent))
        return [("class:live-dock.dim", indent), ("class:live-dock.dim", tip)] if tip else []


def showing(cli, rows: int) -> bool:
    """The tip line is up: tips on, the footer has nothing else to show, no prompt or full-screen
    monitor is open, and the terminal is tall enough to spare the row (a terminal short enough
    that the footer's ``_row_budget`` shows a single task row has none to spare)."""
    from hermes_cli.cli_dock_footer import _row_budget
    from hermes_cli.cli_subagent_monitor import modal_prompt_active
    monitor = getattr(cli, "_subagent_monitor", None)
    return bool(monitor is not None and monitor.tips is not None and not monitor.has_rows
                and monitor.app is None and not monitor.opening and _row_budget(rows) > 1
                and not modal_prompt_active(cli))
