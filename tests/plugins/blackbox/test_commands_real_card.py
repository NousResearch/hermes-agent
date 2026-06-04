"""Real /cost command path — NO card mock. Proves the card.render(dict) seam
(the diff-review BLOCK #1: commands called card.render but card only had
render_card(TurnRecord, threshold)). Exercises the actual store + real card.
"""
import importlib
import sys

import pytest


@pytest.fixture
def real_env(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    importlib.reload(hermes_constants)
    import plugins.blackbox.store as store
    importlib.reload(store)
    import plugins.blackbox.commands as commands
    importlib.reload(commands)
    # point the command module's store reference at the reloaded one
    monkeypatch.setattr(commands, "store", store, raising=False)
    return store, commands


def _seed(store, *, turn_id="turn_x", platform="discord", chat_id="C1", cost=1.26):
    import time as _t
    from plugins.blackbox.record import TurnRecord
    _now = _t.time()
    rec = TurnRecord(
        turn_id=turn_id, profile="Aegis", provider="openai-codex",
        model="openai-codex/gpt-5.4", platform=platform, chat_id=chat_id,
        chat_name="general", api_calls=1, tools=["exec", "exec", "exec", "read"],
        input_tokens=500_000, output_tokens=559, cache_read_tokens=499_000,
        cache_write_tokens=1000, reasoning_tokens=10, context_used=500_000,
        context_length=1_050_000, cost_usd=cost, cost_status="estimated",
        ts_start=_now - 23.0, ts_end=_now,
        user_text="hi", final_text="done",
    )
    store.insert_turn(rec)
    return rec


def _set_channel(commands, monkeypatch, platform, chat_id):
    monkeypatch.setattr(commands, "_current_channel", lambda: (platform, chat_id))


def test_cost_latest_renders_real_card(real_env, monkeypatch):
    store, commands = real_env
    _seed(store)
    _set_channel(commands, monkeypatch, "discord", "C1")
    out = commands.handle_cost("")
    # The real card.render(dict) must have produced the card — NOT an error.
    assert "💸 Spending Alert" in out
    assert "/cost error" not in out
    assert "Turn Cost: $1.26" in out
    assert "Tool Calls: 4 (exec×3, read)" in out
    assert "Session: Discord <#C1>" in out


def test_cost_turn_digin_real_card_and_acl(real_env, monkeypatch):
    store, commands = real_env
    _seed(store, turn_id="turn_dig", chat_id="C1")
    # Right channel → dig-in renders card + texts
    _set_channel(commands, monkeypatch, "discord", "C1")
    out = commands.handle_cost("turn_dig")
    assert "💸 Spending Alert" in out
    assert "/cost error" not in out
    # ACL: wrong channel → not found, no leak
    _set_channel(commands, monkeypatch, "discord", "OTHER")
    blocked = commands.handle_cost("turn_dig")
    assert "💸 Spending Alert" not in blocked
    assert "not found" in blocked.lower() or "no turn" in blocked.lower()


def test_cost_top_real(real_env, monkeypatch):
    store, commands = real_env
    _seed(store, turn_id="turn_a", cost=0.50)
    _seed(store, turn_id="turn_b", cost=2.00)
    _set_channel(commands, monkeypatch, "discord", "C1")
    out = commands.handle_cost("top 2")
    assert "/cost error" not in out
    assert "turn_b" in out  # priciest listed
