"""Finance worker for Torben's Signal COO operator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .action_ledger import ActionLedger
from .briefs import ScopeBrief


def _compact(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


class FinanceSlice:
    """Stage hard-limited trading and personal-finance proposals."""

    def __init__(self, ledger: ActionLedger):
        self.ledger = ledger

    def generate_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> ScopeBrief:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        trade_signals = _list(evidence.get("trade_signals") or evidence.get("market_signals"))
        personal_finance = _list(evidence.get("personal_finance_signals") or evidence.get("monarch_signals"))

        if trade_signals:
            return self._trade_brief(_dict(trade_signals[0]), now=now)
        if personal_finance:
            return self._personal_finance_brief(_dict(personal_finance[0]), now=now)
        return ScopeBrief(
            scope="finance",
            title="Finance",
            status="quiet",
            priority="low",
            text="Finance: no trade or Monarch signal supplied. No capital is at risk.",
        )

    def _trade_brief(self, signal: dict[str, Any], *, now: datetime) -> ScopeBrief:
        catalyst = _compact(signal.get("catalyst") or signal.get("event"), "market catalyst")
        thesis = _compact(signal.get("thesis"), "the trade needs a clearer thesis before execution")
        expression = _compact(signal.get("expression") or signal.get("instrument"), "supported option or marginable equity")
        max_loss = _compact(signal.get("max_loss") or signal.get("premium") or signal.get("risk_cap"), "unset")
        exit_rule = _compact(signal.get("exit_rule"), "exit rule required before execution")
        expected_payoff = _compact(signal.get("expected_payoff"), "payoff model required before execution")
        evidence_ids = [str(item) for item in signal.get("evidence_ids") or []]

        action = self.ledger.add_action(
            scope="FIN",
            summary=f"Review live-trading setup: {expression}",
            evidence_ids=evidence_ids,
            allowed_next_actions=["revise", "approve_trade_review", "discard"],
            status="approval_required",
            risk_class="critical",
            now=now,
            executor_state={
                "mutation_type": "broker_order",
                "provider": "robinhood-agentic-mcp",
                "mutation_status": "review_only",
                "auth_required": True,
                "execution_blocked_until": [
                    "broker_auth",
                    "account_eligibility_check",
                    "risk_policy_limits",
                    "explicit_signal_approval",
                ],
                "requires_options_margin_review": True,
            },
        )

        lines = [
            "Finance: I staged a trade review, not an order.",
            "",
            f"Catalyst: {catalyst}.",
            f"Thesis: {thesis}.",
            f"Expression: {expression}.",
            f"Max loss: {max_loss}.",
            f"Expected payoff: {expected_payoff}.",
            f"Exit rule: {exit_rule}.",
            "",
            f"[{action.handle}] Approve trade review or ask for a smaller risk version.",
        ]
        return ScopeBrief(
            scope="finance",
            title="Finance",
            text="\n".join(lines),
            priority="high",
            actions=[action],
            evidence_ids=evidence_ids,
        )

    def _personal_finance_brief(self, signal: dict[str, Any], *, now: datetime) -> ScopeBrief:
        summary = _compact(signal.get("summary") or signal.get("opportunity"), "personal finance opportunity")
        recommendation = _compact(signal.get("recommendation"), "review the expense and decide whether to cut it")
        evidence_ids = [str(item) for item in signal.get("evidence_ids") or []]
        action = self.ledger.add_action(
            scope="FIN",
            summary=f"Review Monarch finance action: {summary}",
            evidence_ids=evidence_ids,
            allowed_next_actions=["revise", "approve_note", "discard"],
            status="staged",
            risk_class="medium",
            now=now,
            executor_state={
                "mutation_type": "monarch_review",
                "provider": "monarch-money-mcp",
                "mutation_status": "draft_only",
                "external_change_blocked_until": "explicit_signal_approval",
            },
        )
        lines = [
            "Finance: I found a Monarch action to review.",
            "",
            f"Signal: {summary}.",
            f"Recommendation: {recommendation}.",
            "",
            f"[{action.handle}] Review the finance action. Nothing is changed in Monarch.",
        ]
        return ScopeBrief(
            scope="finance",
            title="Finance",
            text="\n".join(lines),
            priority="normal",
            actions=[action],
            evidence_ids=evidence_ids,
        )
