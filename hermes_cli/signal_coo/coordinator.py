"""Torben coordinator across hidden EA, GTM, and Finance workers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .action_ledger import ActionLedger
from .briefs import ScopeBrief, TorbenBrief
from .ea import EASlice
from .finance import FinanceSlice
from .gtm import GTMSlice


REPLY_FOOTER = "Reply with the handle or tell me what to change."


def _scope_evidence(evidence: dict[str, Any], scope: str, aliases: tuple[str, ...] = ()) -> dict[str, Any]:
    for key in (scope, *aliases):
        value = evidence.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _strip_reply_footer(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith(REPLY_FOOTER):
        stripped = stripped[: -len(REPLY_FOOTER)].rstrip()
    return stripped


class TorbenCoordinator:
    """Single user-facing operator that coordinates hidden sub-operators."""

    def __init__(self, ledger: ActionLedger):
        self.ledger = ledger
        self.ea = EASlice(ledger)
        self.gtm = GTMSlice(ledger)
        self.finance = FinanceSlice(ledger)

    def generate_operating_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> TorbenBrief:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        sections = [
            self._ea_section(_scope_evidence(evidence, "ea"), now=now),
            self.gtm.generate_brief(_scope_evidence(evidence, "gtm", ("marketing", "magnus")), now=now),
            self.finance.generate_brief(_scope_evidence(evidence, "finance", ("financial", "ratatosk")), now=now),
        ]
        return TorbenBrief(
            text=self._render(sections, now=now),
            sections=sections,
        )

    def _ea_section(self, evidence: dict[str, Any], *, now: datetime) -> ScopeBrief:
        brief = self.ea.generate_daily_brief(evidence, now=now)
        status = "ready" if brief.actions else "quiet"
        priority = "high" if brief.actions else "low"
        return ScopeBrief(
            scope="ea",
            title="EA",
            text=_strip_reply_footer(brief.text),
            actions=brief.actions,
            priority=priority,
            status=status,
            evidence_ids=[
                evidence_id
                for action in brief.actions
                for evidence_id in action.evidence_ids
            ],
        )

    def _render(self, sections: list[ScopeBrief], *, now: datetime) -> str:
        actions = [action for section in sections for action in section.actions]
        active_sections = [section for section in sections if section.actions or section.status != "quiet"]
        if not active_sections:
            active_sections = sections

        lines = [f"Torben / Operating Brief / {now:%Y-%m-%d}", ""]
        if actions:
            scope_names = ", ".join(sorted({action.scope.upper() for action in actions}))
            lines.append(f"I staged {len(actions)} action(s) across {scope_names}.")
            lines.append("Nothing has been sent, posted, traded, or changed externally.")
        else:
            lines.append("No action is staged from the supplied evidence.")
        for section in active_sections:
            lines.extend(["", _strip_reply_footer(section.text)])
        lines.extend(["", REPLY_FOOTER])
        return "\n".join(line.rstrip() for line in lines).strip() + "\n"
