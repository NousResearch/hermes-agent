from __future__ import annotations

import sys
import re
import types
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from plugins.blackbox import card, cost
from plugins.blackbox.record import TurnRecord


class CostResult:
    def __init__(self, amount_usd, status):
        self.amount_usd = amount_usd
        self.status = status


def test_cost_partial_when_one_call_unknown(monkeypatch):
    calls_seen = []

    def fake_estimate(model, usage, *, provider=None, base_url=None):
        calls_seen.append(usage)
        if len(calls_seen) == 1:
            return CostResult(Decimal("1.25"), "estimated")
        return CostResult(None, "unknown")

    monkeypatch.setattr(cost, "estimate_usage_cost", fake_estimate)

    amount, status = cost.compute_turn_cost(
        "model",
        "provider",
        "https://example.invalid",
        [
            {"input_tokens": 100, "output_tokens": 20},
            {"input_tokens": 200, "output_tokens": 40},
        ],
    )

    assert amount == 1.25
    assert status == "partial"
    assert calls_seen[0].input_tokens == 100


def test_cost_all_included(monkeypatch):
    monkeypatch.setattr(
        cost,
        "estimate_usage_cost",
        lambda *args, **kwargs: CostResult(Decimal("0"), "included"),
    )

    amount, status = cost.compute_turn_cost(
        "model",
        "provider",
        "",
        [{"input_tokens": 1}, {"output_tokens": 1}],
    )

    assert amount == 0.0
    assert status == "included"


def test_cost_status_vocabulary_pinned():
    """RC7: pin the real usage_pricing status values so a provider/library
    change that introduces a new status is caught here (compute_turn_cost
    ranks unknown statuses as worst-of → would silently mark turns unknown).
    The real estimate_usage_cost returns one of: actual, estimated, included,
    unknown. cost.compute_turn_cost maps 'actual'→'estimated' for ranking.
    """
    import agent.usage_pricing as up
    import inspect
    src = inspect.getsource(up)
    produced = set(re.findall(r'status\s*=\s*"([a-z]+)"', src))
    # Every status the library can produce must be handled by cost._STATUS_RANK
    # (or be the 'actual' alias we remap). Unhandled → ranks as 3 (unknown).
    handled = set(cost._STATUS_RANK) | {"actual"}
    unhandled = produced - handled
    assert not unhandled, f"usage_pricing emits unhandled status(es): {unhandled}"


def test_cost_actual_maps_to_estimated(monkeypatch):
    """'actual' status must be reconciled as 'estimated', not ranked unknown."""
    monkeypatch.setattr(
        cost, "estimate_usage_cost",
        lambda *a, **k: CostResult(Decimal("0.50"), "actual"),
    )
    amount, status = cost.compute_turn_cost("m", "p", "", [{"input_tokens": 1}])
    assert amount == 0.5
    assert status == "estimated"


def _record(**overrides):
    ts = datetime(2026, 4, 15, 15, 50, 21, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()
    data = {
        "turn_id": "turn_abc",
        "ts_start": ts - 23,
        "ts_end": ts,
        "profile": "Aegis",
        "provider": "openrouter",
        "model": "openai-codex/gpt-5.4",
        "platform": "Discord",
        "chat_id": "1488602178305130546",
        "chat_name": "ops",
        "api_calls": 1,
        "tools": ["exec", "exec", "exec", "read"],
        "input_tokens": 500_000,
        "output_tokens": 559,
        "cache_read_tokens": 499_000,
        "cache_write_tokens": 0,
        "context_used": 500_000,
        "context_length": 1_050_000,
        "cost_usd": 1.26,
        "cost_status": "estimated",
    }
    data.update(overrides)
    return TurnRecord(**data)


def test_card_renders_expected_bullets():
    text = card.render_card(_record(), 1.0)

    assert text.splitlines() == [
        "💸 Spending Alert",
        "• Turn Cost: $1.26",
        "• Threshold: $1.00",
        "• API Calls: 1",
        "• Tool Calls: 4 (exec×3, read)",
        "• Tokens: 500k in + 559 out",
        "• Context: 500k/1050k 🟢 (48% of model max)",
        "• Cached: 499k/500k 🟢 100%",
        "• Agent: Aegis",
        "• Model: openai-codex/gpt-5.4",
        "• Session: Discord <#1488602178305130546>",
        "• Latency: 23s",
        "• Datetime: 2026/04/15 15:50:21 PT",
        "• Investigate: /cost turn_abc",
    ]


def test_card_div_zero_and_platform_session_lines():
    assert "• Cached: n/a" in card.render_card(_record(input_tokens=0, cache_read_tokens=0), 1.0)
    assert "• Context: 500k" in card.render_card(_record(context_length=0), 1.0)
    assert "• Session: Telegram #general" in card.render_card(
        _record(platform="Telegram", chat_id="123", chat_name="general"),
        1.0,
    )
    assert "• Session: Slack <#C123|alerts>" in card.render_card(
        _record(platform="Slack", chat_id="C123", chat_name="alerts"),
        1.0,
    )


def test_card_health_colour_boundaries():
    assert card.context_health(69.9) == "🟢"
    assert card.context_health(70) == "🟡"
    assert card.context_health(90) == "🟡"
    assert card.context_health(90.1) == "🔴"
    assert card.cache_health(80.1) == "🟢"
    assert card.cache_health(50) == "🟡"
    assert card.cache_health(49.9) == "🔴"
    assert card.cost_health(0.49, 1.0) == "🟢"
    assert card.cost_health(0.50, 1.0) == "🟡"
    assert card.cost_health(1.00, 1.0) == "🔴"


@pytest.fixture
def blackbox(monkeypatch):
    import plugins.blackbox as bb

    bb._sessions.clear()
    monkeypatch.setattr(bb, "_profile_name", lambda: "Aegis")
    monkeypatch.setattr(bb, "_turn_id", lambda: "turn_test")
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *args, **kwargs: (0.0, "included"))

    store = types.SimpleNamespace(
        records=[],
        marked=[],
        insert_turn=lambda record: store.records.append(record),
        mark_alerted=lambda turn_id: store.marked.append(turn_id) or True,
    )
    monkeypatch.setitem(sys.modules, "plugins.blackbox.store", store)
    # The hook does a lazy `from plugins.blackbox import store`, which resolves
    # the attribute on the package. When the real store submodule was already
    # imported (e.g. by test_store.py earlier in the suite), it is bound as a
    # package attribute and `setitem` on sys.modules alone won't shadow it —
    # also set the package attribute so the mock wins regardless of import order.
    monkeypatch.setattr(bb, "store", store, raising=False)

    sent = []
    monkeypatch.setattr(bb.routing, "send_card", lambda *args: sent.append(args))

    return bb, store, sent


def _usage():
    return {
        "api_calls": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "context_used": 10,
        "context_length": 100,
        "calls": [{"input_tokens": 10, "output_tokens": 2}],
        "chat_id": "C1",
        "chat_name": "alerts",
    }


def test_hook_below_threshold_records_no_alert(blackbox, monkeypatch):
    bb, store, sent = blackbox
    monkeypatch.setattr(bb, "_config", lambda: {"enabled": True, "cost_alert_threshold_usd": 1.0})
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *args, **kwargs: (0.99, "estimated"))

    bb._on_session_start(session_id="s1")
    bb._on_post_tool_call(tool_name="exec", session_id="s1")
    bb._on_session_end(session_id="s1", model="m", platform="discord", provider="p", turn_usage=_usage())

    assert len(store.records) == 1
    assert store.records[0].tools == ["exec"]
    assert store.marked == []
    assert sent == []


def test_hook_at_threshold_marks_and_sends_once(blackbox, monkeypatch):
    bb, store, sent = blackbox
    monkeypatch.setattr(bb, "_config", lambda: {"enabled": True, "cost_alert_threshold_usd": 1.0})
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *args, **kwargs: (1.0, "estimated"))

    bb._on_session_end(session_id="s1", model="m", platform="discord", provider="p", turn_usage=_usage())

    assert store.marked == ["turn_test"]
    assert len(sent) == 1


def test_hook_interrupted_records_no_alert(blackbox, monkeypatch):
    bb, store, sent = blackbox
    monkeypatch.setattr(bb, "_config", lambda: {"enabled": True, "cost_alert_threshold_usd": 1.0})
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *args, **kwargs: (2.0, "estimated"))

    bb._on_session_end(
        session_id="s1",
        interrupted=True,
        model="m",
        platform="discord",
        provider="p",
        turn_usage=_usage(),
    )

    assert len(store.records) == 1
    assert store.records[0].interrupted is True
    assert store.marked == []
    assert sent == []


def test_hook_always_card_sends_regardless_of_cost(blackbox, monkeypatch):
    bb, store, sent = blackbox
    monkeypatch.setattr(
        bb,
        "_config",
        lambda: {"enabled": True, "cost_alert_threshold_usd": 1.0, "always_card": True},
    )
    monkeypatch.setattr(bb, "compute_turn_cost", lambda *args, **kwargs: (0.01, "estimated"))

    bb._on_session_end(session_id="s1", model="m", platform="discord", provider="p", turn_usage=_usage())

    assert store.marked == ["turn_test"]
    assert len(sent) == 1


def test_hook_config_off_or_missing_noop(blackbox, monkeypatch):
    bb, store, sent = blackbox
    monkeypatch.setattr(bb, "_config", lambda: None)

    bb._on_session_end(session_id="s1", model="m", platform="discord", provider="p", turn_usage=_usage())

    assert store.records == []
    assert store.marked == []
    assert sent == []
