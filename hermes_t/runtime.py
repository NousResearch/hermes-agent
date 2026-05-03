"""Generic runtime cycle for hermes_t — reads state, evaluates signal, returns result."""

from __future__ import annotations

from hermes_t.signal_policy import DEFAULT_SIGNAL_POLICY, SignalPolicy, render_signal_text
from hermes_t.store import TradingStateStore


def run_runtime_cycle(
    *,
    store: TradingStateStore,
    tech_data: dict,
    profile_id: str,
    symbol: str,
    trade_unit: int,
    max_trades: int,
    signal_policy: SignalPolicy | None = None,
) -> dict:
    """Execute one runtime cycle and return {pending, suggestion, summary}.

    The tech_data dict is the primary input — it may contain ``signal``,
    ``score``, or both.

    Args:
        store: Profile-scoped state store (already constructed).
        tech_data: Dict with at least ``signal`` and/or ``score``.
        profile_id: Profile identifier for state tracking.
        symbol: Trading symbol (e.g. ``"688319"``).
        trade_unit: Shares per trade.
        max_trades: Maximum number of buy trades per session.
        signal_policy: Optional runtime signal policy override.

    Returns:
        Dict with three keys:
        ``pending`` — the pending signal dict (``{}`` when none).
        ``suggestion`` — the suggested action with metadata.
        ``summary`` — cumulative action counts.
    """
    policy = signal_policy or DEFAULT_SIGNAL_POLICY
    state = store.load_execution_state()

    buy_count = state.get("buy_count", 0)
    sell_count = state.get("sell_count", 0)
    hold_count = state.get("hold_count", 0)
    previous_buys = buy_count
    previous_sells = sell_count
    previous_holds = hold_count

    # Determine action
    signal = tech_data.get("signal")
    score = tech_data.get("score", 50)

    if signal and signal in ("buy", "sell", "hold"):
        action = signal
    else:
        action = policy.action_for_score(score)

    # Evaluate the action
    pending: dict = {}
    suggestion: dict = {"action": action}

    if action == "buy":
        seq = buy_count + 1
        if seq > max_trades:
            # Blocked by max_trades
            store.clear_pending_signal()
            suggestion = {"action": "hold", "reason": f"max_trades ({max_trades}) reached"}
            hold_count += 1
        else:
            pending = {
                "status": "pending",
                "action": "buy",
                "seq": seq,
                "unit": trade_unit,
                "symbol": symbol,
                "text": render_signal_text("buy", seq, policy),
            }
            suggestion["seq"] = seq
            suggestion["unit"] = trade_unit
            suggestion["text"] = pending["text"]
            buy_count = seq
    elif action == "sell":
        seq = sell_count + 1
        pending = {
            "status": "pending",
            "action": "sell",
            "seq": seq,
            "unit": trade_unit,
            "symbol": symbol,
            "text": render_signal_text("sell", seq, policy),
        }
        suggestion["seq"] = seq
        suggestion["unit"] = trade_unit
        suggestion["text"] = pending["text"]
        sell_count = seq
    else:
        # hold — clear any existing pending signal
        store.clear_pending_signal()
        hold_count += 1

    # Persist state
    new_state = {
        "profile_id": profile_id,
        "symbol": symbol,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "max_trades": max_trades,
        "trade_unit": trade_unit,
    }
    store.save_execution_state(new_state)

    # Write pending signal to store (hold branch already cleared it)
    if pending:
        store.save_pending_signal(pending)

    return {
        "pending": pending,
        "suggestion": suggestion,
        "summary": {
            "total_actions": buy_count + sell_count + hold_count,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "previous_buys": previous_buys,
            "previous_sells": previous_sells,
            "previous_holds": previous_holds,
        },
    }
