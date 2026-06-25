"""Torben operator facade for Signal-facing workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .action_ledger import ActionLedger, ReplyResolution
from .briefs import TorbenBrief
from .coordinator import TorbenCoordinator
from .ea import EABrief, EASlice


class TorbenOperator:
    """Small facade that binds ledger, router, and first EA slice."""

    def __init__(self, *, ledger_path: str | Path):
        self.ledger = ActionLedger(ledger_path)
        self.ea = EASlice(self.ledger)
        self.coordinator = TorbenCoordinator(self.ledger)

    def generate_ea_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> EABrief:
        return self.ea.generate_daily_brief(evidence, now=now)

    def resolve_reply(
        self,
        reply_text: str,
        *,
        now: datetime | None = None,
    ) -> ReplyResolution:
        return self.ledger.resolve_reply(reply_text, now=now)

    def generate_operating_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> TorbenBrief:
        return self.coordinator.generate_operating_brief(evidence, now=now)
