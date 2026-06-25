"""Shared brief primitives for Torben's hidden operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .action_ledger import ActionRecord


@dataclass
class ScopeBrief:
    """One hidden operator's staged output for the Signal-facing coordinator."""

    scope: str
    title: str
    text: str
    actions: list[ActionRecord] = field(default_factory=list)
    priority: str = "normal"
    status: str = "ready"
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "title": self.title,
            "text": self.text,
            "priority": self.priority,
            "status": self.status,
            "evidence_ids": self.evidence_ids,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class TorbenBrief:
    """Signal-ready operating brief across EA, GTM, and Finance."""

    text: str
    sections: list[ScopeBrief]

    @property
    def actions(self) -> list[ActionRecord]:
        actions: list[ActionRecord] = []
        for section in self.sections:
            actions.extend(section.actions)
        return actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sections": [section.to_dict() for section in self.sections],
            "actions": [action.to_dict() for action in self.actions],
        }
