"""Signal policy — thresholds, templates, and text rendering for hermes_t runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SignalActionName = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class SignalAction:
    """A single signal action — name, template text, and optional threshold bound."""

    name: SignalActionName
    template: str
    threshold_ceiling: int | None = None  # score <= this triggers the action

    def render(self, seq: int) -> str:
        return self.template.replace("{seq}", str(seq))


@dataclass(frozen=True)
class SignalPolicy:
    """Parameters and actions for a runtime cycle signal evaluation."""

    max_buys: int = 4
    buy_unit_pct: float = 0.5  # unused in skeleton phase
    sell_unit_pct: float = 1.0  # unused in skeleton phase
    actions: tuple[SignalAction, ...] = field(default_factory=lambda: _DEFAULT_ACTIONS)

    def action_for_score(self, score: int) -> SignalActionName:
        """Map a 0-100 score to an action name.

        Buy triggers when ``score <= buy.threshold_ceiling``;
        sell triggers when ``score >= sell.threshold_ceiling``.
        """
        for action in self.actions:
            if action.name == "sell" and action.threshold_ceiling is not None and score >= action.threshold_ceiling:
                return "sell"
            if action.name == "buy" and action.threshold_ceiling is not None and score <= action.threshold_ceiling:
                return "buy"
        return "hold"


_DEFAULT_ACTIONS: tuple[SignalAction, ...] = (
    SignalAction(name="buy", template="第{seq}次买入", threshold_ceiling=20),
    SignalAction(name="sell", template="卖出", threshold_ceiling=80),
    SignalAction(name="hold", template="观望"),
)

DEFAULT_SIGNAL_POLICY = SignalPolicy()


def render_signal_text(action: str, seq: int, policy: SignalPolicy) -> str:
    """Render a human-readable signal text for an action name + sequence."""
    for a in policy.actions:
        if a.name == action:
            return a.render(seq)
    return f"Unknown action: {action}"
