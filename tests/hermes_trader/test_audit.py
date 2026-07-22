"""Tests for P5 audit, rate limits, alerts, rollout, and size modifier."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from hermes_trader.audit.alerts import AlertStore, evaluate_alerts
from hermes_trader.audit.logger import McpAuditLog, audit_mcp_call
from hermes_trader.audit.rate_limit import WriteToolRateLimiter, check_write_rate_limit
from hermes_trader.audit.redact import hash_params, redact_for_log, redact_mapping
from hermes_trader.config import TraderConfig
from hermes_trader.risk.gate import RejectReason, RiskGate, TradeIntent
from hermes_trader.risk.mandate import sign_mandate
from hermes_trader.risk.rollout import (
    chain_allowed_by_rollout,
    resolve_rollout_policy,
    trade_within_rollout_cap,
)
from hermes_trader.risk.size_modifier import apply_size_multiplier, compute_size_multiplier

TEST_KEY = b"test-mandate-secret-key"
WALLET = "0xabcdef1234567890abcdef1234567890abcdef12"
ADDR = "0x1234567890123456789012345678901234567890"


@pytest.fixture(autouse=True)
def _mandate_secret(monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_MANDATE_SECRET", TEST_KEY.decode())


@pytest.fixture
def trader_home(tmp_path, monkeypatch):
    hh = tmp_path / "hermes-home"
    trader = hh / "trader"
    trader.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hh))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hh)
    return trader


def test_redact_for_log_masks_address():
    text = f"swap from {ADDR} to 0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    redacted = redact_for_log(text)
    assert ADDR not in redacted
    assert "0x…redacted" in redacted


def test_redact_mapping_masks_private_key():
    data = {"user_private_key": "0x" + "ab" * 32, "amount": 100}
    redacted = redact_mapping(data)
    assert redacted["user_private_key"] == "[REDACTED]"
    assert redacted["amount"] == 100


def test_hash_params_stable():
    args = {"token": ADDR, "amount": 50}
    assert hash_params(args) == hash_params(args)
    assert len(hash_params(args)) == 16


def test_write_rate_limiter_blocks_after_max(trader_home):
    state = trader_home / "write_rate_limit.json"
    limiter = WriteToolRateLimiter(max_per_hour=2, state_path=state)
    now = 1_700_000_000.0
    assert limiter.allow(now=now)
    limiter.record(now=now)
    limiter.record(now=now + 1)
    assert not limiter.allow(now=now + 2)
    assert limiter.remaining(now=now + 2) == 0


def test_check_write_rate_limit_message(trader_home):
    state = trader_home / "write_rate_limit.json"
    limiter = WriteToolRateLimiter(max_per_hour=1, state_path=state)
    limiter.record(now=time.time())
    ok, msg = check_write_rate_limit(max_per_hour=1, state_path=state)
    assert not ok
    assert "rate limit" in msg.lower()


def test_mcp_audit_log_append(trader_home):
    path = trader_home / "mcp_audit.jsonl"
    log = McpAuditLog(path=path)
    cid = audit_mcp_call(
        server_name="defi-trading",
        tool_name="execute_swap",
        args={"token_address": ADDR},
        result_status="ok",
        latency_ms=12.5,
        audit_log=log,
    )
    assert len(cid) == 12
    rows = log.list_recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["correlation_id"] == cid
    assert rows[0]["params_hash"]
    assert ADDR not in json.dumps(rows[0])


def test_evaluate_alerts_kill_switch(trader_home, monkeypatch):
    monkeypatch.setenv("HERMES_TRADER_KILL_SWITCH", "1")
    alerts = evaluate_alerts(TraderConfig())
    assert any(a.kind == "kill_switch" for a in alerts)


def test_evaluate_alerts_gate_spike(trader_home):
    cycles = trader_home / "cycles.jsonl"
    now = datetime.now(timezone.utc)
    for _ in range(12):
        row = {
            "timestamp": now.isoformat(),
            "approved": False,
            "reason_code": "LOW_CONFIDENCE",
        }
        cycles.write_text(
            (cycles.read_text(encoding="utf-8") if cycles.is_file() else "")
            + json.dumps(row)
            + "\n",
            encoding="utf-8",
        )
    cfg = TraderConfig(gate_reject_spike_threshold=10)
    alerts = evaluate_alerts(cfg, cycles_log_path=cycles, now=now)
    assert any(a.kind == "gate_block_spike" for a in alerts)


def test_alert_store_persists(trader_home):
    path = trader_home / "alerts.jsonl"
    store = AlertStore(path=path)
    from hermes_trader.audit.alerts import Alert

    store.emit([Alert(kind="test", message="hello")])
    assert "hello" in path.read_text(encoding="utf-8")


def test_rollout_policy_steady_has_no_cap():
    cfg = TraderConfig(rollout_stage="steady")
    policy = resolve_rollout_policy(cfg)
    assert policy.max_trade_usd(cfg) is None


def test_rollout_canary_cap():
    cfg = TraderConfig(rollout_stage="canary", mode="live")
    ok, _ = trade_within_rollout_cap(40.0, cfg)
    assert ok
    ok, msg = trade_within_rollout_cap(75.0, cfg)
    assert not ok
    assert "50" in msg


def test_rollout_limited_chain_filter():
    cfg = TraderConfig(
        rollout_stage="limited",
        allowed_chains=["base", "ethereum", "arbitrum"],
    )
    assert chain_allowed_by_rollout("base", cfg)
    assert chain_allowed_by_rollout("ethereum", cfg)
    assert not chain_allowed_by_rollout("arbitrum", cfg)


def test_size_modifier_never_increases():
    intent = TradeIntent.from_mapping(
        {
            "action": "buy",
            "chain": "base",
            "token_address": "0xtoken",
            "size_usd": 100.0,
            "confidence": 0.62,
            "pool_liquidity_usd": 80_000.0,
        }
    )
    mult = compute_size_multiplier(intent)
    assert 0.5 <= mult <= 1.0
    assert apply_size_multiplier(100.0, mult) <= 100.0


def test_gate_reject_rollout_cap(trader_home, monkeypatch):
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    cfg = TraderConfig(mode="live", rollout_stage="canary")
    mandate = sign_mandate(WALLET, signing_key=TEST_KEY)
    decision = RiskGate(config=cfg).evaluate(
        TradeIntent.from_mapping(
            {
                "action": "buy",
                "chain": "base",
                "token_address": "0xtoken",
                "size_usd": 200.0,
                "confidence": 0.9,
            }
        ),
        mandate=mandate,
        portfolio_value_usd_override=10_000,
    )
    assert decision.reason_code == RejectReason.ROLLOUT_CAP


def test_gate_reject_rollout_chain(trader_home, monkeypatch):
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    cfg = TraderConfig(mode="live", rollout_stage="canary")
    mandate = sign_mandate(WALLET, signing_key=TEST_KEY)
    decision = RiskGate(config=cfg).evaluate(
        TradeIntent.from_mapping(
            {
                "action": "buy",
                "chain": "ethereum",
                "token_address": "0xtoken",
                "size_usd": 10.0,
                "confidence": 0.9,
            }
        ),
        mandate=mandate,
        portfolio_value_usd_override=10_000,
    )
    assert decision.reason_code == RejectReason.ROLLOUT_CHAIN


def test_gate_size_modifier_reduces_order(trader_home, monkeypatch):
    monkeypatch.setenv("USER_ADDRESS", WALLET)
    cfg = TraderConfig(mode="live", enable_size_modifier=True, min_pool_liquidity_usd=50_000.0)
    mandate = sign_mandate(WALLET, signing_key=TEST_KEY)
    decision = RiskGate(config=cfg).evaluate(
        TradeIntent.from_mapping(
            {
                "action": "buy",
                "chain": "base",
                "token_address": "0xtoken",
                "size_usd": 100.0,
                "confidence": 0.62,
                "pool_liquidity_usd": 80_000.0,
            }
        ),
        mandate=mandate,
        portfolio_value_usd_override=10_000,
    )
    assert decision.approved
    assert decision.order is not None
    assert decision.order.size_usd < 100.0