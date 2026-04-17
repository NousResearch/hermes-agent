"""Spec TDD for agent.task_state continuity helpers.

Ensures tiny inputs like "dale", "hazlo", "sigue", "", "ok" are treated
as continuations so the router keeps the current tier.
"""
from __future__ import annotations

from pathlib import Path

from agent import task_state as ts


def test_is_continuation_single_word():
    for w in ["sigue", "dale", "ok", "hazlo", "continua", "continúa", "ya"]:
        assert ts.is_continuation(w), w


def test_is_continuation_phrase():
    for p in ["haz lo tuyo", "sigue con eso", "mismo tema", "go on", "keep going"]:
        assert ts.is_continuation(p), p


def test_is_continuation_false_on_new_task():
    assert not ts.is_continuation("refactor la función de login")
    assert not ts.is_continuation("investiga el clima en Madrid")


def test_is_silence():
    assert ts.is_silence("")
    assert ts.is_silence("   ")
    assert not ts.is_silence("hola")


def test_record_turn_continuation_preserves_streak_and_task():
    s = ts.start_task(ts.default_state(), tier=4, model="qwen3-coder-next", category="code")
    s = ts.record_turn(s, "dale", was_easy=False)
    assert s["active_task"] is True
    assert s["last_tier"] == 4
    assert s["turns_in_task"] == 2
    # continuation keeps easy_streak unchanged
    assert s["easy_streak"] == 0


def test_record_turn_easy_increments_streak():
    s = ts.default_state()
    s = ts.record_turn(s, "what's 2+2", was_easy=True)
    s = ts.record_turn(s, "define HTTP", was_easy=True)
    assert s["easy_streak"] == 2


def test_record_turn_non_easy_resets_streak():
    s = ts.default_state()
    s["easy_streak"] = 3
    s = ts.record_turn(s, "implement full auth flow with tests", was_easy=False)
    assert s["easy_streak"] == 0


def test_roundtrip_save_load(tmp_path: Path):
    p = tmp_path / "state.json"
    s = ts.start_task(ts.default_state(), tier=5, model="qwen3.5:397b", category="analysis")
    ts.save(p, s)
    back = ts.load(p)
    assert back["last_tier"] == 5
    assert back["last_model"] == "qwen3.5:397b"
    assert back["active_task"] is True
