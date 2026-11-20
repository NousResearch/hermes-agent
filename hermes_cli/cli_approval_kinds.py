"""Copy for the classic CLI approval panel, per approval ``kind``.

An approval carries the kind of card to draw (``tools/approval.py``). A ``command`` card shows the
command to run with the description under it; a ``schedule`` card (a new cron job) shows the
description as its body, because its ``command`` is a synthetic ``<cronjob_manage> ...`` string that
means nothing to the user. Unknown kinds render as ``command``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from agent.i18n import t

# Approval-choice identifiers (protocol tokens, never translated) → catalog keys for their labels.
_COMMAND_CHOICE_KEYS = MappingProxyType({
    "once": "cli.tui.approval_once",
    "session": "cli.tui.approval_session",
    "always": "cli.tui.approval_always",
    "deny": "cli.tui.approval_deny",
    "view": "cli.tui.approval_view"})

_COMMAND_OUTCOME_KEYS = MappingProxyType({
    "once": "cli.approval.outcome_once",
    "session": "cli.approval.outcome_session",
    "always": "cli.approval.outcome_always",
    "deny": "cli.approval.outcome_deny"})


@dataclass(frozen=True)
class ApprovalCard:
    """Catalog keys for one kind of approval panel."""

    icon: str
    title_key: str
    label_key: str
    timeout_key: str
    choice_keys: Mapping[str, str]
    outcome_keys: Mapping[str, str]
    body_is_description: bool
    truncated_key: str

    def subject(self, command: str, description: str) -> tuple[str, str]:
        """``(body, footnote)``: the text the panel must always show, and the text it may trim."""
        return (description, "") if self.body_is_description else (command, description)

    def prints_body_first(self, body: str) -> bool:
        """True when the whole body goes to scrollback before the panel: a description body (a job and
        its gate script) longer than a few lines may not fit the panel, and the user must be able to
        read all of it before allowing it."""
        return self.body_is_description and (body.count("\n") >= 4 or len(body) > 400)

    def choices(self, choices: list[str]) -> list[str]:
        """The offered choices this card shows: a description body has no "show full command"."""
        return [c for c in choices if c != "view"] if self.body_is_description else choices

    def choice_label(self, choice: str) -> str:
        """Localized label for an approval choice id; unknown ids render verbatim."""
        key = self.choice_keys.get(choice)
        return t(key) if key else choice

    def outcome_label(self, result) -> str:
        key = self.outcome_keys.get(result)
        return t(key) if key else str(result)


_CARDS = {
    "command": ApprovalCard(
        icon="⚠",
        title_key="cli.tui.approval_title",
        label_key="cli.approval.label",
        timeout_key="cli.approval.timeout_denying",
        choice_keys=_COMMAND_CHOICE_KEYS,
        outcome_keys=_COMMAND_OUTCOME_KEYS,
        body_is_description=False,
        truncated_key="cli.tui.approval_command_truncated"),
    "schedule": ApprovalCard(
        icon="⏰",
        title_key="cli.tui.approval_title_schedule",
        label_key="cli.approval.schedule_label",
        timeout_key="cli.approval.schedule_timeout",
        choice_keys=MappingProxyType({
            **_COMMAND_CHOICE_KEYS,
            "once": "cli.tui.approval_schedule_once",
            "session": "cli.tui.approval_schedule_session",
            "always": "cli.tui.approval_schedule_always"}),
        outcome_keys=MappingProxyType({**_COMMAND_OUTCOME_KEYS, "deny": "cli.approval.schedule_outcome_deny"}),
        body_is_description=True,
        truncated_key="cli.tui.approval_schedule_truncated"),
}


def approval_card(kind: str | None) -> ApprovalCard:
    """The card for *kind*; absent or unknown kinds get the command card."""
    return _CARDS.get(kind or "command", _CARDS["command"])


def approval_choice_label(choice: str, kind: str | None = None) -> str:
    """Localized label for an approval choice id on a *kind* card; unknown ids render verbatim."""
    return approval_card(kind).choice_label(choice)
