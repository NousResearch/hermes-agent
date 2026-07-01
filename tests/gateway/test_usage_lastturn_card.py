#!/usr/bin/env python3
"""Phase-2 unit tests for the /usage last-turn card helper (PRD usage-format-codex).

Drives GatewaySlashCommandsMixin._render_last_turn_card directly with a stub
`source`, covering: channel-scoped render against a REAL blackbox store, the
channel-MATCH guard (D-7), and the reworded thin fallback (INV-4).

Hermetic: the suite's autouse ``_hermetic_environment`` fixture (tests/conftest.py)
already redirects HERMES_HOME to a per-test tempdir, so we seed the turn into THAT
sandbox store via insert_turn — never touching the live ~/.hermes store.

Run (from the worktree, with the active venv):
  PYTHONPATH=$PWD python -m pytest \
    tests/gateway/test_usage_lastturn_card.py -o addopts= -q
"""
import time

import pytest

import gateway.slash_commands as SC


class _Plat:
    def __init__(self, value):
        self.value = value


class _Source:
    def __init__(self, platform, chat_id):
        self.platform = _Plat(platform) if platform else None
        self.chat_id = chat_id


def _find_owner_class():
    for name in dir(SC):
        obj = getattr(SC, name)
        if isinstance(obj, type) and "_render_last_turn_card" in obj.__dict__:
            return obj
    return None


OWNER = _find_owner_class()


class _Bare:
    pass


def _call(source, thin_snap, fallback_label=None):
    assert OWNER is not None, "could not find the class defining _render_last_turn_card"
    inst = _Bare()
    if fallback_label is None:
        return OWNER._render_last_turn_card(inst, source, thin_snap)
    return OWNER._render_last_turn_card(inst, source, thin_snap, fallback_label=fallback_label)


CHAN = "1502228850338435153"


@pytest.fixture
def seeded_turn():
    """Insert one real turn for CHAN into the per-test sandbox blackbox store."""
    from plugins.blackbox import store as bb_store
    from plugins.blackbox.record import TurnRecord
    now = time.time()
    bb_store.insert_turn(TurnRecord(
        turn_id="turn_test_apollo_0001",
        ts_start=now - 30, ts_end=now,
        profile="default", provider="claude-app", model="claude-opus-4-8",
        platform="discord", chat_id=CHAN, chat_name="Daemonarchy / #apollo",
        api_calls=3, tools=["read_file", "patch"],
        input_tokens=42, output_tokens=900,
        cache_read_tokens=500000, cache_write_tokens=1200, reasoning_tokens=300,
        context_used=540000, context_length=1000000,
        last_cache_read_tokens=500000, last_cache_write_tokens=1200, last_uncached_tokens=42,
        cost_usd=1.23, cost_status="estimated",
    ))
    return CHAN


def test_real_channel_renders_full_card(seeded_turn):
    src = _Source("discord", CHAN)
    text = "\n".join(_call(src, None))
    assert "Last turn" in text
    assert "Tokens in" in text
    assert "uncached" in text
    assert "Tokens out" in text
    assert "billed" in text
    # Model line carries the provider prefix (the seeded turn is claude-app/claude-opus-4-8).
    assert "• Model: claude-app/claude-opus-4-8" in text


def test_channel_miss_falls_back_to_thin(seeded_turn):
    # Different channel than the seeded one → channel-match guard misses → thin.
    src = _Source("discord", "999999999999999999")
    thin = {"input_tokens": 5, "output_tokens": 1000, "cache_read_tokens": 500000,
            "cache_write_tokens": 400, "reasoning_tokens": 0}
    text = "\n".join(_call(src, thin))
    assert "persisted; agent not resident" in text
    assert "uncached" in text
    assert "Total (billed in+out)" in text
    assert "500,405" in text  # 5 + 500000 + 400


def test_fallback_label_override():
    src = _Source("discord", "999999999999999999")
    thin = {"input_tokens": 10, "output_tokens": 100, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "reasoning_tokens": 50}
    text = "\n".join(_call(src, thin, fallback_label="session totals; first turn not yet recorded"))
    assert "session totals; first turn not yet recorded" in text
    assert "persisted; agent not resident" not in text
    assert "150 billed" in text                  # 100 + 50 reasoning
    assert "Total (billed in+out): 160" in text   # 10 + 150


def test_no_blackbox_no_thin_returns_empty():
    src = _Source("discord", "999999999999999999")
    assert _call(src, None) == []


def _demo_rec():
    import time
    now = time.time()
    return {
        "found": True, "platform": "discord", "chat_id": "1", "chat_name": "x",
        "model": "claude-opus-4-8", "provider": "claude-app", "profile": "default",
        "ts_start": now - 10, "ts_end": now, "api_calls": 3, "tools": "[]",
        "input_tokens": 100, "output_tokens": 50, "cache_read": 900, "cache_write": 0,
        "reasoning": 0, "context_used": 1000, "context_length": 1_000_000,
        "cost_usd": 0.01, "cost_status": "estimated",
    }


def test_card_row_order_cached_before_context_window():
    """Cached row sits ABOVE the Context-window row; Compressions follows."""
    from plugins.blackbox.last_turn import render_last_turn_record
    lines = render_last_turn_record(_demo_rec(), compressions=3)
    assert "• Compressions: 3" in lines
    cached_i = next(i for i, l in enumerate(lines) if l.startswith("• Cached:"))
    ctx_i = next(i for i, l in enumerate(lines) if l.startswith("• Context window"))
    comp_i = next(i for i, l in enumerate(lines) if l.startswith("• Compressions:"))
    assert cached_i < ctx_i, "Cached must render above Context window"
    assert ctx_i < comp_i, "Compressions follows the Context-window row"


def test_compressions_row_omitted_when_none_or_zero():
    from plugins.blackbox.last_turn import render_last_turn_record
    assert not any("Compressions" in l for l in render_last_turn_record(_demo_rec(), compressions=None))
    assert not any("Compressions" in l for l in render_last_turn_record(_demo_rec(), compressions=0))

