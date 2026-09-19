"""Temporal-context plugin: per-user-turn time header via pre_llm_call (#99942).

Pins the behaviour that matters: the interval is measured per session, the
header is injected only on the first LLM call of a user turn (not internal
tool-loop iterations or machine-driven turns), and the duration rendering /
clock-skew handling are correct.
"""
import importlib

import pytest

tc = importlib.import_module("plugins.temporal_context")


@pytest.fixture(autouse=True)
def _clear_state():
    tc._last_user_turn_at.clear()
    yield
    tc._last_user_turn_at.clear()


def test_humanize_scales():
    assert tc._humanize(0) == "0s"
    assert tc._humanize(45) == "45s"
    assert tc._humanize(60) == "1m"
    assert tc._humanize(3600) == "1h"
    assert tc._humanize(3720) == "1h2m"
    assert tc._humanize(90000) == "1d1h"
    assert tc._humanize(172800) == "2d"


def test_first_turn_reports_first_and_carries_clock():
    header = tc._temporal_header("s1", 1_756_000_000.0)
    assert "first turn in this session" in header
    assert "[Current time:" in header
    assert "temporal_context" in header


def test_second_turn_reports_interval_since_previous():
    now = 1_756_000_000.0
    tc._temporal_header("s1", now)                 # first turn
    header = tc._temporal_header("s1", now + 3 * 3600 + 12 * 60)  # +3h12m
    assert "3h12m since your previous turn" in header
    assert "first turn" not in header


def test_interval_is_per_session():
    now = 1_756_000_000.0
    tc._temporal_header("a", now)
    tc._temporal_header("b", now + 10)
    # Session "a" advancing does not leak into "b" and vice versa.
    ha = tc._temporal_header("a", now + 45)   # 45s since a's first turn
    hb = tc._temporal_header("b", now + 25)   # 15s since b's first turn (now+10)
    assert "45s since your previous turn" in ha
    assert "15s since your previous turn" in hb


def test_clock_skew_backwards_falls_back_to_first_turn():
    now = 1_756_000_000.0
    tc._temporal_header("s1", now)
    header = tc._temporal_header("s1", now - 500)  # clock moved backwards
    assert "first turn in this session" in header  # no negative interval, no crash


def test_hook_injects_only_on_first_call_of_a_user_turn():
    # Internal tool-loop iterations (api_call_count > 0) inject nothing.
    assert tc.on_pre_llm_call(session_id="s1", turn_type="user", api_call_count=3) is None
    # Non-user (cron / goal continuation) turns inject nothing.
    assert tc.on_pre_llm_call(session_id="s1", turn_type="cron", api_call_count=0) is None
    # First call of a user turn returns the header payload.
    out = tc.on_pre_llm_call(session_id="s1", turn_type="user", api_call_count=0)
    assert isinstance(out, dict) and "[Current time:" in out["context"]


def test_hook_accepts_unknown_future_payload_keys():
    # The hook payload evolves additively; the callback must tolerate extra keys.
    out = tc.on_pre_llm_call(
        session_id="s1", turn_type="user", api_call_count=0,
        model="gpt-x", provider="openai", some_future_field=123,
    )
    assert isinstance(out, dict) and "context" in out
