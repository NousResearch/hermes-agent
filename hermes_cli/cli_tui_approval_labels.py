"""Localized labels for approval-choice ids in the interactive TUI.

Split from :mod:`hermes_cli.cli_tui_mixin` (a facade over its line budget): the choice ids are
protocol tokens and are never translated, only their labels are, resolved at render time so the
active ``display.language`` applies.
"""
from __future__ import annotations

from agent.i18n import t

# Approval-choice identifiers (protocol tokens, never translated) → catalog keys for their labels.
_APPROVAL_CHOICE_KEYS = {
    "once": "cli.tui.approval_once",
    "session": "cli.tui.approval_session",
    "always": "cli.tui.approval_always",
    "yolo": "cli.tui.approval_yolo",
    "deny": "cli.tui.approval_deny",
    "view": "cli.tui.approval_view"}


def approval_choice_label(choice: str) -> str:
    """Localized label for an approval choice id; unknown ids render verbatim."""
    key = _APPROVAL_CHOICE_KEYS.get(choice)
    return t(key) if key else choice
