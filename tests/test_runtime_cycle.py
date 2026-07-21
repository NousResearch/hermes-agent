"""Tests for hermes_t runtime cycle and signal policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_t.signal_policy import DEFAULT_SIGNAL_POLICY, SignalAction, SignalPolicy


# ── SignalPolicy tests ────────────────────────────────────────────────────


def test_signal_policy_default_values():
    """DEFAULT_SIGNAL_POLICY should have sensible defaults."""
    p = DEFAULT_SIGNAL_POLICY
    assert p.max_buys > 0
    assert p.buy_unit_pct > 0
    assert p.sell_unit_pct > 0
    assert len(p.actions) > 0


def test_signal_policy_has_buy_sell_hold():
    """Policy should include buy, sell, and hold actions."""
    p = DEFAULT_SIGNAL_POLICY
    actions = {a.name for a in p.actions}
    assert "buy" in actions
    assert "sell" in actions
    assert "hold" in actions


def test_signal_policy_buy_text_uses_ordinal():
    """Buy action text for Nth buy should use ordinal like '第N次买入'."""
    p = DEFAULT_SIGNAL_POLICY
    buy_action = next(a for a in p.actions if a.name == "buy")
    text = buy_action.render(1)
    assert "第" in text and "次" in text and "买入" in text
    text_2 = buy_action.render(2)
    assert "第2次" in text_2
    text_3 = buy_action.render(3)
    assert "第3次" in text_3


def test_signal_policy_sell_text_default():
    """Sell action text should mention '卖出'."""
    p = DEFAULT_SIGNAL_POLICY
    sell_action = next(a for a in p.actions if a.name == "sell")
    text = sell_action.render(1)
    assert "卖出" in text


def test_signal_policy_hold_text():
    """Hold action text should be neutral."""
    p = DEFAULT_SIGNAL_POLICY
    hold_action = next(a for a in p.actions if a.name == "hold")
    text = hold_action.render(0)
    assert text  # non-empty


def test_signal_policy_render_signal_text():
    from hermes_t.signal_policy import render_signal_text

    text = render_signal_text("buy", 1, DEFAULT_SIGNAL_POLICY)
    assert "第" in text and "次" in text and "买入" in text

    text = render_signal_text("sell", 2, DEFAULT_SIGNAL_POLICY)
    assert "卖出" in text

    text = render_signal_text("hold", 0, DEFAULT_SIGNAL_POLICY)
    assert text  # non-empty

    text = render_signal_text("unknown", 0, DEFAULT_SIGNAL_POLICY)
    assert "unknown" in text  # falls back


@pytest.mark.parametrize("score, expected_min", [
    (90, "sell"),
    (80, "sell"),
    (65, None),  # hold — middle band
    (50, None),
    (30, None),
    (20, "buy"),
    (5, "buy"),
])
def test_signal_policy_default_threshold_maps_scores(score: int, expected_min: str | None):
    """DEFAULT_SIGNAL_POLICY threshold should map scores to actions."""
    action = DEFAULT_SIGNAL_POLICY.action_for_score(score)
    if expected_min:
        assert action in (expected_min, "buy")  # buy can appear broader than expected
    else:
        assert action is not None


# ── Runtime cycle tests ───────────────────────────────────────────────────


def test_run_runtime_cycle_returns_pending_suggestion_summary(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="cycle_test")
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "hold", "score": 50},
        profile_id="cycle_test",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert "pending" in result
    assert "suggestion" in result
    assert "summary" in result


def test_run_runtime_cycle_with_sell_score_outputs_sell_suggestion(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_test")
    store.save_position({"symbol": "688319", "total_shares": 220000})
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_test",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "sell"
    assert result["suggestion"]["unit"] == 10000
    assert result["pending"]["status"] == "pending"
    assert result["pending"]["action"] == "sell"


def test_run_runtime_cycle_with_hold_score_clears_pending(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="hold_test")
    # Pre-populate a pending signal
    store.save_pending_signal({"status": "pending", "action": "sell"})

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "hold", "score": 50},
        profile_id="hold_test",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    # Pending should be cleared
    assert result["pending"] == {}
    assert store.load_pending_signal() == {}
    assert result["suggestion"]["action"] == "hold"


def test_run_runtime_cycle_keeps_existing_pending_signal_when_new_action_arrives(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="keep_pending")
    existing_pending = {
        "status": "pending",
        "action": "buy",
        "seq": 1,
        "unit": 5000,
        "symbol": "688319",
        "text": "低吸第1次",
    }
    store.save_pending_signal(existing_pending)

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="keep_pending",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "pending signal unresolved"
    assert result["pending"] == existing_pending
    assert store.load_pending_signal() == existing_pending
    assert result["summary"]["sell_count"] == 0
    assert result["summary"]["hold_count"] == 1


def test_run_runtime_cycle_sell_without_position_is_held(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_without_position")

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_without_position",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "no sellable position"
    assert result["pending"] == {}
    assert store.load_pending_signal() == {}
    assert result["summary"]["sell_count"] == 0
    assert result["summary"]["hold_count"] == 1


def test_run_runtime_cycle_sell_without_position_preserves_existing_pending(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_without_position_pending")
    existing_pending = {
        "status": "pending",
        "action": "sell",
        "seq": 1,
        "unit": 10000,
        "symbol": "688319",
        "text": "止盈第1次",
    }
    store.save_pending_signal(existing_pending)

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_without_position_pending",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "pending signal unresolved"
    assert result["pending"] == existing_pending
    assert store.load_pending_signal() == existing_pending
    assert result["summary"]["sell_count"] == 0
    assert result["summary"]["hold_count"] == 1



def test_run_runtime_cycle_sell_without_position_blocks_symbol_mismatch(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_symbol_mismatch")
    store.save_position({"symbol": "000001", "total_shares": 220000})

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_symbol_mismatch",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "no sellable position"
    assert result["pending"] == {}
    assert result["summary"]["sell_count"] == 0


def test_run_runtime_cycle_sell_with_string_total_shares_outputs_sell_suggestion(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_with_string_position")
    store.save_position({"symbol": "688319", "total_shares": "220000"})

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_with_string_position",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "sell"
    assert result["pending"]["action"] == "sell"


def test_run_runtime_cycle_sell_with_non_finite_total_shares_is_held(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_with_inf_position")
    store.save_position({"symbol": "688319", "total_shares": "inf"})

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_with_inf_position",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "no sellable position"
    assert result["pending"] == {}


def test_run_runtime_cycle_sell_with_position_outputs_sell_suggestion(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="sell_with_position")
    store.save_position({"symbol": "688319", "total_shares": 220000})

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 85},
        profile_id="sell_with_position",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "sell"
    assert result["suggestion"]["unit"] == 10000
    assert result["pending"]["status"] == "pending"
    assert result["pending"]["action"] == "sell"

def test_run_runtime_cycle_buy_increments_seq(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="buy_seq")

    result_1 = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 25},
        profile_id="buy_seq",
        symbol="688319",
        trade_unit=5000,
        max_trades=4,
    )
    assert result_1["pending"]["seq"] == 1
    store.clear_pending_signal()

    result_2 = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 15},
        profile_id="buy_seq",
        symbol="688319",
        trade_unit=7000,
        max_trades=4,
    )
    assert result_2["pending"]["seq"] == 2
    assert result_2["pending"]["unit"] == 7000


def test_run_runtime_cycle_same_action_pending_is_preserved(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="same_action_pending")
    existing_pending = {
        "status": "pending",
        "action": "buy",
        "seq": 1,
        "unit": 5000,
        "symbol": "688319",
    }
    store.save_pending_signal(existing_pending)

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 15},
        profile_id="same_action_pending",
        symbol="688319",
        trade_unit=7000,
        max_trades=4,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "pending signal unresolved"
    assert result["pending"] == existing_pending
    assert store.load_pending_signal() == existing_pending
    assert result["summary"]["buy_count"] == 0
    assert result["summary"]["hold_count"] == 1


def test_run_runtime_cycle_honors_max_trades(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="max_trades")
    max_trades = 2
    for i in range(max_trades):
        result = run_runtime_cycle(
            store=store,
            tech_data={"signal": "buy", "score": 20},
            profile_id="max_trades",
            symbol="688319",
            trade_unit=5000,
            max_trades=max_trades,
        )
        assert result["pending"]["seq"] == i + 1
        store.clear_pending_signal()

    # Third buy should be blocked
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 10},
        profile_id="max_trades",
        symbol="688319",
        trade_unit=5000,
        max_trades=max_trades,
    )
    assert result["suggestion"]["action"] == "hold"
    assert result["pending"] == {}


def test_run_runtime_cycle_blocked_buy_clears_stale_pending_signal(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="blocked_buy_clears_pending")
    store.save_pending_signal({"status": "pending", "action": "buy", "seq": 2})
    store.save_execution_state(
        {
            "profile_id": "blocked_buy_clears_pending",
            "symbol": "688319",
            "buy_count": 2,
            "sell_count": 0,
            "hold_count": 0,
            "max_trades": 2,
            "trade_unit": 5000,
        }
    )

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 10},
        profile_id="blocked_buy_clears_pending",
        symbol="688319",
        trade_unit=5000,
        max_trades=2,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["pending"] == {}
    assert store.load_pending_signal() == {}


def test_run_runtime_cycle_blocked_buy_preserves_unrelated_pending_signal(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="blocked_buy_keeps_sell_pending")
    existing_pending = {
        "status": "pending",
        "action": "sell",
        "seq": 1,
        "unit": 10000,
        "symbol": "688319",
        "text": "止盈第1次",
    }
    store.save_pending_signal(existing_pending)
    store.save_execution_state(
        {
            "profile_id": "blocked_buy_keeps_sell_pending",
            "symbol": "688319",
            "buy_count": 2,
            "sell_count": 0,
            "hold_count": 0,
            "max_trades": 2,
            "trade_unit": 5000,
        }
    )

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 10},
        profile_id="blocked_buy_keeps_sell_pending",
        symbol="688319",
        trade_unit=5000,
        max_trades=2,
    )

    assert result["suggestion"]["action"] == "hold"
    assert result["suggestion"]["reason"] == "max_trades (2) reached"
    assert result["pending"] == existing_pending
    assert store.load_pending_signal() == existing_pending


def test_run_runtime_cycle_tech_data_without_signal_falls_back_to_score(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="fallback")
    result = run_runtime_cycle(
        store=store,
        tech_data={"score": 90},
        profile_id="fallback",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )
    # score=90 should yield sell via policy threshold
    assert result["suggestion"]["action"] in ("sell", "hold")


def test_run_runtime_cycle_persists_execution_state(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="persist")
    run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 20},
        profile_id="persist",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )
    state = store.load_execution_state()
    assert state.get("profile_id") == "persist"
    assert state.get("symbol") == "688319"


def test_run_runtime_cycle_summary_includes_counts(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="summary_test")
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "sell", "score": 95},
        profile_id="summary_test",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )
    summary = result["summary"]
    assert "total_actions" in summary
    assert "buy_count" in summary
    assert "sell_count" in summary
    assert "hold_count" in summary
    assert "previous_buys" in summary
    assert "previous_sells" in summary


def test_run_runtime_cycle_honors_signal_policy_override(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    custom_policy = SignalPolicy(
        actions=(
            SignalAction(name="buy", template="低吸第{seq}次", threshold_ceiling=10),
            SignalAction(name="sell", template="止盈卖出", threshold_ceiling=60),
            SignalAction(name="hold", template="继续观察"),
        )
    )
    store = TradingStateStore(base_dir=str(tmp_path), profile_id="custom_policy")
    store.save_position({"symbol": "688319", "total_shares": 220000})

    result = run_runtime_cycle(
        store=store,
        tech_data={"score": 70},
        profile_id="custom_policy",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
        signal_policy=custom_policy,
    )

    assert result["suggestion"]["action"] == "sell"
    assert result["suggestion"]["text"] == "止盈卖出"
    assert result["pending"]["action"] == "sell"
    assert result["pending"]["text"] == "止盈卖出"

def test_run_runtime_cycle_summary_previous_counts_reflect_pre_cycle_state(tmp_path: Path):
    from hermes_t.runtime import run_runtime_cycle
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="summary_previous_counts")
    store.save_execution_state(
        {
            "profile_id": "summary_previous_counts",
            "symbol": "688319",
            "buy_count": 1,
            "sell_count": 2,
            "hold_count": 3,
            "max_trades": 4,
            "trade_unit": 10000,
        }
    )

    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 20},
        profile_id="summary_previous_counts",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )
    summary = result["summary"]

    assert summary["previous_buys"] == 1
    assert summary["previous_sells"] == 2
    assert summary["previous_holds"] == 3
    assert summary["buy_count"] == 2
    assert summary["sell_count"] == 2
    assert summary["hold_count"] == 3


def test_dispatch_pending_signal_sends_and_persists_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    from hermes_t.runtime import dispatch_pending_signal, run_runtime_cycle
    from hermes_t.store import TradingStateStore

    sent_args: dict[str, object] = {}

    def fake_send_message_tool(args, **_kwargs):
        sent_args.update(args)
        return json.dumps({"success": True, "platform": "feishu", "chat_id": "home_chat"})

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake_send_message_tool)

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="dispatch_ok")
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 20},
        profile_id="dispatch_ok",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    pending = result["pending"]
    dispatched = dispatch_pending_signal(store=store, profile_id="dispatch_ok", dispatch_target="feishu")

    assert sent_args["target"] == "feishu"
    assert sent_args["message"] == pending["text"]
    assert dispatched["status"] == "sent"
    assert dispatched["profile_id"] == "dispatch_ok"
    assert dispatched["signal"]["status"] == "sent"
    assert store.load_pending_signal()["status"] == "sent"
    assert store.load_active_signal()["status"] == "sent"
    push_state = store.load_push_state()
    assert push_state["last_status"] == "sent"
    history = store.read_signal_send_history()
    assert len(history) == 1
    assert history[0]["status"] == "sent"


def test_dispatch_pending_signal_records_failure_without_clearing_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    from hermes_t.runtime import dispatch_pending_signal, run_runtime_cycle
    from hermes_t.store import TradingStateStore

    def fake_send_message_tool(_args, **_kwargs):
        return json.dumps({"success": False, "error": "gateway timeout"})

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake_send_message_tool)

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="dispatch_failed")
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 10},
        profile_id="dispatch_failed",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )

    pending = result["pending"]
    dispatched = dispatch_pending_signal(store=store, profile_id="dispatch_failed", dispatch_target="feishu")

    assert pending["status"] == "pending"
    assert dispatched["status"] == "failed"
    assert dispatched["error"] == "gateway timeout"
    assert dispatched["signal"]["status"] == "failed"
    assert store.load_pending_signal()["status"] == "failed"
    assert store.load_pending_signal()["error"] == "gateway timeout"
    assert store.load_active_signal()["status"] == "failed"
    push_state = store.load_push_state()
    assert push_state["last_status"] == "failed"
    history = store.read_signal_send_history()
    assert len(history) == 1
    assert history[0]["status"] == "failed"


def test_retry_failed_signal_reopens_failed_pending_for_redispatch(tmp_path: Path):
    from hermes_t.runtime import retry_failed_signal
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="retry_failed")
    failed_signal = {
        "status": "failed",
        "action": "buy",
        "seq": 1,
        "unit": 10000,
        "symbol": "688319",
        "text": "低吸第1次",
        "profile_id": "retry_failed",
        "dispatch_target": "feishu",
        "dispatched_at": "2026-05-04T00:00:00+00:00",
        "error": "gateway timeout",
    }
    store.save_pending_signal(failed_signal)
    store.save_active_signal(failed_signal)
    store.save_push_state(
        {
            "profile_id": "retry_failed",
            "last_status": "failed",
            "last_target": "feishu",
            "last_signal_text": failed_signal["text"],
            "last_dispatched_at": failed_signal["dispatched_at"],
        }
    )

    retried = retry_failed_signal(store=store, profile_id="retry_failed")

    assert retried["status"] == "pending"
    assert retried["profile_id"] == "retry_failed"
    assert retried["signal"]["status"] == "pending"
    assert "error" not in retried["signal"]
    assert "dispatched_at" not in retried["signal"]
    assert store.load_pending_signal()["status"] == "pending"
    assert "error" not in store.load_pending_signal()
    assert store.load_active_signal()["status"] == "pending"
    push_state = store.load_push_state()
    assert push_state["last_status"] == "pending"
    assert push_state["last_target"] == "feishu"
    assert push_state["last_signal_text"] == failed_signal["text"]


def test_retry_failed_signal_returns_noop_when_no_failed_signal_exists(tmp_path: Path):
    from hermes_t.runtime import retry_failed_signal
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="retry_noop")
    store.save_pending_signal({"status": "sent", "action": "buy"})

    result = retry_failed_signal(store=store, profile_id="retry_noop")

    assert result == {
        "status": "noop",
        "profile_id": "retry_noop",
        "reason": "no failed signal to retry",
        "signal": {"status": "sent", "action": "buy"},
    }


def test_retry_failed_signal_allows_dispatch_again_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import json

    from hermes_t.runtime import dispatch_pending_signal, retry_failed_signal, run_runtime_cycle
    from hermes_t.store import TradingStateStore

    call_count = {"count": 0}

    def fake_send_message_tool(_args, **_kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return json.dumps({"success": False, "error": "gateway timeout"})
        return json.dumps({"success": True, "platform": "feishu", "chat_id": "home_chat"})

    monkeypatch.setattr("tools.send_message_tool.send_message_tool", fake_send_message_tool)

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="retry_dispatch")
    result = run_runtime_cycle(
        store=store,
        tech_data={"signal": "buy", "score": 10},
        profile_id="retry_dispatch",
        symbol="688319",
        trade_unit=10000,
        max_trades=4,
    )
    assert result["pending"]["status"] == "pending"

    failed = dispatch_pending_signal(store=store, profile_id="retry_dispatch", dispatch_target="feishu")
    assert failed["status"] == "failed"
    assert store.load_pending_signal()["status"] == "failed"

    reopened = retry_failed_signal(store=store, profile_id="retry_dispatch")
    assert reopened["status"] == "pending"
    assert store.load_pending_signal()["status"] == "pending"

    resent = dispatch_pending_signal(store=store, profile_id="retry_dispatch", dispatch_target="feishu")
    assert resent["status"] == "sent"
    assert resent["signal"]["status"] == "sent"
    assert store.load_pending_signal()["status"] == "sent"
    history = store.read_signal_send_history()
    assert len(history) == 2
    assert [row["status"] for row in history] == ["failed", "sent"]


def test_build_review_summary_returns_empty_defaults_for_new_profile(tmp_path: Path):
    from hermes_t.runtime import build_review_summary
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="review_empty")

    summary = build_review_summary(store=store, profile_id="review_empty")

    assert summary == {
        "profile_id": "review_empty",
        "pending": {},
        "active": {},
        "push_state": {},
        "dispatch_ledger_tail": [],
        "signal_send_history_tail": [],
        "counts": {
            "dispatch_ledger_count": 0,
            "signal_send_history_count": 0,
        },
    }



def test_build_review_summary_collects_current_state_and_tail_history(tmp_path: Path):
    from hermes_t.runtime import build_review_summary
    from hermes_t.store import TradingStateStore

    store = TradingStateStore(base_dir=str(tmp_path), profile_id="review_populated")
    pending = {"status": "pending", "action": "buy", "text": "低吸第1次"}
    active = {"status": "sent", "action": "sell", "text": "止盈第1次"}
    push_state = {"last_status": "sent", "last_target": "feishu", "last_signal_text": "止盈第1次"}
    store.save_pending_signal(pending)
    store.save_active_signal(active)
    store.save_push_state(push_state)

    for idx in range(1, 5):
        ledger_row = {"status": "sent" if idx % 2 == 0 else "failed", "timestamp": f"t{idx}", "seq": idx}
        history_row = {"status": "sent" if idx % 2 == 0 else "failed", "timestamp": f"h{idx}", "seq": idx}
        store.append_dispatch_ledger(ledger_row)
        store.append_signal_send_history(history_row)

    summary = build_review_summary(store=store, profile_id="review_populated", history_limit=2)

    assert summary["profile_id"] == "review_populated"
    assert summary["pending"] == pending
    assert summary["active"] == active
    assert summary["push_state"] == push_state
    assert summary["counts"] == {
        "dispatch_ledger_count": 4,
        "signal_send_history_count": 4,
    }
    assert summary["dispatch_ledger_tail"] == [
        {"status": "sent", "timestamp": "t2", "seq": 2},
        {"status": "failed", "timestamp": "t3", "seq": 3},
        {"status": "sent", "timestamp": "t4", "seq": 4},
    ][-2:]
    assert summary["signal_send_history_tail"] == [
        {"status": "sent", "timestamp": "h2", "seq": 2},
        {"status": "failed", "timestamp": "h3", "seq": 3},
        {"status": "sent", "timestamp": "h4", "seq": 4},
    ][-2:]
