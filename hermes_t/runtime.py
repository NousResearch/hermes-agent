"""Generic runtime cycle for hermes_t — reads state, evaluates signal, returns result."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from hermes_t.signal_policy import DEFAULT_SIGNAL_POLICY, SignalPolicy, render_signal_text
from hermes_t.store import TradingStateStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dispatch_pending_signal(
    *,
    store: TradingStateStore,
    profile_id: str,
    dispatch_target: str = "feishu",
) -> dict:
    """Dispatch the current pending signal and persist delivery state."""
    pending = store.load_pending_signal()
    if pending.get("status") != "pending":
        return {
            "status": "noop",
            "profile_id": profile_id,
            "reason": "no pending signal to dispatch",
            "signal": pending,
        }

    from tools.send_message_tool import send_message_tool

    message = str(pending.get("text", "")).strip()
    response_raw = send_message_tool({"action": "send", "target": dispatch_target, "message": message})
    try:
        response = json.loads(response_raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        response = {"success": False, "error": f"invalid send_message response: {response_raw!r}"}

    dispatched_at = _utc_now_iso()
    send_ok = bool(response.get("success")) and not response.get("error")
    status = "sent" if send_ok else "failed"
    error = str(response.get("error", "")).strip() or None

    signal_record = {
        **pending,
        "status": status,
        "profile_id": profile_id,
        "dispatch_target": dispatch_target,
        "dispatched_at": dispatched_at,
    }
    if error:
        signal_record["error"] = error
    else:
        signal_record.pop("error", None)

    store.save_pending_signal(signal_record)
    store.save_active_signal(signal_record)
    store.save_push_state(
        {
            "profile_id": profile_id,
            "last_status": status,
            "last_target": dispatch_target,
            "last_signal_text": message,
            "last_dispatched_at": dispatched_at,
        }
    )

    ledger_row = {
        "profile_id": profile_id,
        "status": status,
        "target": dispatch_target,
        "timestamp": dispatched_at,
        "signal": signal_record,
        "response": response,
    }
    store.append_signal_send_history(ledger_row)
    store.append_dispatch_ledger(ledger_row)

    result = {
        "status": status,
        "profile_id": profile_id,
        "signal": signal_record,
        "response": response,
    }
    if error:
        result["error"] = error
    return result


def retry_failed_signal(*, store: TradingStateStore, profile_id: str) -> dict:
    """Reopen a failed pending signal so it can be dispatched again."""
    failed_signal = store.load_pending_signal()
    if failed_signal.get("status") != "failed":
        return {
            "status": "noop",
            "profile_id": profile_id,
            "reason": "no failed signal to retry",
            "signal": failed_signal,
        }

    retried_signal = {
        **failed_signal,
        "status": "pending",
        "profile_id": profile_id,
    }
    retried_signal.pop("error", None)
    retried_signal.pop("dispatched_at", None)

    store.save_pending_signal(retried_signal)
    store.save_active_signal(retried_signal)

    push_state = store.load_push_state()
    dispatch_target = str(
        retried_signal.get("dispatch_target")
        or push_state.get("last_target")
        or "feishu"
    ).strip() or "feishu"
    last_signal_text = str(
        retried_signal.get("text")
        or push_state.get("last_signal_text")
        or ""
    )
    store.save_push_state(
        {
            **push_state,
            "profile_id": profile_id,
            "last_status": "pending",
            "last_target": dispatch_target,
            "last_signal_text": last_signal_text,
        }
    )

    return {
        "status": "pending",
        "profile_id": profile_id,
        "signal": retried_signal,
    }


def build_review_summary(
    *,
    store: TradingStateStore,
    profile_id: str,
    history_limit: int = 5,
) -> dict:
    """Build a compact review summary from current state and recent ledgers."""
    limit = max(int(history_limit), 0)
    dispatch_ledger = store.read_dispatch_ledger()
    signal_send_history = store.read_signal_send_history()
    return {
        "profile_id": profile_id,
        "pending": store.load_pending_signal(),
        "active": store.load_active_signal(),
        "push_state": store.load_push_state(),
        "dispatch_ledger_tail": dispatch_ledger[-limit:] if limit else [],
        "signal_send_history_tail": signal_send_history[-limit:] if limit else [],
        "counts": {
            "dispatch_ledger_count": len(dispatch_ledger),
            "signal_send_history_count": len(signal_send_history),
        },
    }


def _coerce_total_shares(value: object) -> int:
    """Best-effort coercion for persisted total_shares values."""
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0


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
    existing_pending = store.load_pending_signal()

    if action == "buy" and buy_count + 1 > max_trades:
        # Blocked by max_trades; clear only stale pending buy so dispatcher does not retry it.
        if existing_pending.get("status") == "pending" and existing_pending.get("action") == "buy":
            store.clear_pending_signal()
        elif existing_pending.get("status") == "pending":
            pending = existing_pending
        suggestion = {"action": "hold", "reason": f"max_trades ({max_trades}) reached"}
        hold_count += 1
    elif action != "hold" and existing_pending.get("status") == "pending":
        pending = existing_pending
        suggestion = {"action": "hold", "reason": "pending signal unresolved"}
        hold_count += 1
    elif action == "buy":
        seq = buy_count + 1
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
        position = store.load_position() or {}
        position_symbol = str(position.get("symbol", "")).strip()
        symbol_matches = position_symbol == str(symbol).strip()
        total_shares = _coerce_total_shares(position.get("total_shares", 0))
        if not symbol_matches or total_shares <= 0:
            suggestion = {"action": "hold", "reason": "no sellable position"}
            pending = existing_pending if existing_pending.get("status") == "pending" else {}
            hold_count += 1
        else:
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
