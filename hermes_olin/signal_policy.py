from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SignalPolicy:
    """Signal policy config: score thresholds and text templates.

    Attributes:
        buy_score_threshold: Minimum score to trigger a buy signal (>= this value).
        sell_score_threshold: Maximum score to trigger a sell signal (<= this value).
        active_signal_hold_text: Text returned when an active pending signal blocks new suggestions.
        no_action_text: Default text returned when no execution suggestion is generated.
    """

    buy_score_threshold: int = 70
    sell_score_threshold: int = 30
    active_signal_hold_text: str = "当前已有待处理信号，暂停生成下一笔执行建议"
    no_action_text: str = "暂无新增执行建议"


DEFAULT_SIGNAL_POLICY: SignalPolicy = SignalPolicy()


def render_signal_text(
    *,
    action: str,
    sequence: int,
    trade_unit: int,
    policy: Optional[SignalPolicy] = None,
) -> str:
    """Render the human-readable execution text for a signal action.

    Args:
        action: "buy", "sell", or "hold".
        sequence: Execution sequence number (1-indexed).
        trade_unit: Number of shares per execution unit.
        policy: Signal policy used for default text/template selection.

    Returns:
        Formatted string like "第2次卖出 500 股".
    """
    effective_policy = policy or DEFAULT_SIGNAL_POLICY
    if action == "buy":
        return f"第{sequence}次买入 {trade_unit} 股"
    if action == "sell":
        return f"第{sequence}次卖出 {trade_unit} 股"
    return effective_policy.no_action_text
