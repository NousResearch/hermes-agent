"""Unit tests for the last-resort fallback floor (apply_fallback_floor)."""
from agent.agent_init import apply_fallback_floor

FLOOR = {"provider": "opencode-go", "model": "minimax-m3"}


def test_appends_floor_to_empty_chain():
    # The bug we are fixing: an agent initialised with no fallback chain still
    # gets a reliable escape when its primary is rate-limited.
    chain = apply_fallback_floor([], FLOOR, primary_provider="ollama-cloud", primary_model="deepseek-v4-flash")
    assert chain == [{**FLOOR, "is_floor": True}]


def test_appends_floor_after_existing_entries():
    chain = apply_fallback_floor(
        [{"provider": "nous", "model": "x"}], FLOOR,
        primary_provider="ollama-cloud", primary_model="deepseek-v4-flash",
    )
    assert chain[-1] == {**FLOOR, "is_floor": True} and len(chain) == 2


def test_skips_when_floor_already_in_chain():
    chain = apply_fallback_floor(
        [{"provider": "opencode-go", "model": "minimax-m3"}], FLOOR,
        primary_provider="ollama-cloud", primary_model="deepseek-v4-flash",
    )
    assert len(chain) == 1


def test_skips_when_floor_is_primary():
    # No point failing over to the backend that just failed.
    chain = apply_fallback_floor([], FLOOR, primary_provider="opencode-go", primary_model="minimax-m3")
    assert chain == []


def test_dedup_is_case_insensitive():
    chain = apply_fallback_floor(
        [{"provider": "OpenCode-GO", "model": "MiniMax-M3"}], FLOOR,
        primary_provider="ollama-cloud", primary_model="x",
    )
    assert len(chain) == 1


def test_non_mutating():
    src = []
    out = apply_fallback_floor(src, FLOOR, primary_provider="p", primary_model="m")
    assert src == [] and out == [{**FLOOR, "is_floor": True}]


def test_noop_on_malformed_or_disabled_floor():
    assert apply_fallback_floor([], {}, primary_provider="p", primary_model="m") == []
    assert apply_fallback_floor([], None, primary_provider="p", primary_model="m") == []
    assert apply_fallback_floor([], {"provider": "x"}, primary_provider="p", primary_model="m") == []
