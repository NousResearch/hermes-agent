"""Discord exec-approval card helpers: the reason's share of the content cap, and how a cut is
labelled.

The card text itself is rendered by the shared ``BasePlatformAdapter._format_exec_approval``
core; these members supply the Discord-specific part — how much of the reason fits, what a cut
says, and the command-preview budget that keeps the two sharing Discord's 2000-char message cap.
"""

from __future__ import annotations

from typing import Callable, Optional

from agent.i18n import t


class DiscordExecApprovalMixin:
    """Discord-specific exec-approval budget + truncation behaviour."""

    # The reason shares the 2000-char content cap with the command preview; unbounded it would
    # starve the preview to zero. A static 300 starved the REASON instead: a reason that leads
    # with context (task metadata, a digest-style header) can put its decisive content past 300,
    # so the operator saw none of it. 1200 fits an agent-authored reason whole while still
    # leaving the command preview a share via ``_exec_approval_cmd_budget`` below.
    _EA_REASON_BUDGET = 1200

    def _ea_trunc_suffix(self) -> str:
        """A cut says what happened: a bare "..." reads as a render glitch, so the catalogued
        marker (the same one ``_self_contained_prompt_content`` appends) names it, per language."""
        return t("platform.discord.prompt.truncated_marker")

    def _ea_fit(self, text: str, budget: int, suffix: Optional[str] = None,
                escape: Optional[Callable[[str], str]] = None) -> str:
        """``suffix=None`` — the shared core's two fits never pass one — becomes the catalogued
        marker; an explicit suffix still wins. ``_exec_approval_cmd_budget`` reads the same
        accessor, so the marker in the text and the marker in the budget math cannot drift."""
        return super()._ea_fit(
            text, budget, self._ea_trunc_suffix() if suffix is None else suffix, escape)

    def _exec_approval_cmd_budget(self, description: str, smart_denied: bool) -> int:
        # Mentions ride in front of the content and count against the 2000-char message cap too.
        fixed = (len(self._EA_HEADER) + len(self._EA_CODE_OPEN) + len(self._EA_CODE_CLOSE)
                 + len(self._EA_REASON_LABEL) + len(description) + len(self._ea_trunc_suffix())
                 + len(self._ea_deadline_line())
                 + (len(self._EA_SMART_DENY_LINE) if smart_denied else 0)
                 + len(self._approval_mention_content() or "") + 1)
        return max(0, self.MAX_MESSAGE_LENGTH - fixed)
