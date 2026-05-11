"""Unit tests for agent.adaptive_context.AdaptiveContextTracker.

The tracker is intentionally minimal — it observes model ids and
reports the new id only on change. These tests pin the boundary
behaviours so the run_agent integration can rely on them.
"""
from __future__ import annotations

import pytest

from agent.adaptive_context import AdaptiveContextTracker


def test_first_observation_returns_none_and_records_baseline():
    t = AdaptiveContextTracker()
    assert t.observe("meta-llama/llama-3.3-70b-instruct:free") is None
    assert t.last_seen == "meta-llama/llama-3.3-70b-instruct:free"
    assert t.change_count == 0


def test_same_model_returns_none():
    t = AdaptiveContextTracker()
    t.observe("model-a")
    assert t.observe("model-a") is None
    assert t.observe("model-a") is None
    assert t.change_count == 0


def test_change_returns_new_model_and_increments():
    t = AdaptiveContextTracker()
    t.observe("model-a")
    new = t.observe("model-b")
    assert new == "model-b"
    assert t.last_seen == "model-b"
    assert t.change_count == 1


def test_multiple_changes_increment_counter():
    t = AdaptiveContextTracker()
    t.observe("a")
    assert t.observe("b") == "b"
    assert t.observe("c") == "c"
    assert t.observe("c") is None
    assert t.observe("d") == "d"
    assert t.change_count == 3


def test_router_to_concrete_backend_transition_fires_once():
    # Realistic scenario: first call returns a concrete backend, second
    # call returns a *different* concrete backend (the router picked
    # someone else). The tracker should fire on the second.
    t = AdaptiveContextTracker()
    assert t.observe("meta-llama/llama-3.3-70b-instruct:free") is None
    fired = t.observe("qwen/qwen-2.5-72b-instruct:free")
    assert fired == "qwen/qwen-2.5-72b-instruct:free"
    assert t.change_count == 1


@pytest.mark.parametrize("bad", [None, "", 0, False, 42, [], {}, object()])
def test_invalid_inputs_return_none_without_changing_state(bad):
    t = AdaptiveContextTracker()
    t.observe("model-a")  # seed
    snapshot_last = t.last_seen
    snapshot_count = t.change_count
    assert t.observe(bad) is None
    assert t.last_seen == snapshot_last
    assert t.change_count == snapshot_count


def test_invalid_input_as_first_observation_does_not_seed():
    t = AdaptiveContextTracker()
    assert t.observe(None) is None
    assert t.observe("") is None
    assert t.last_seen is None
    # A subsequent valid observation should still be treated as the baseline
    assert t.observe("first-real") is None
    assert t.last_seen == "first-real"
    assert t.change_count == 0


def test_summary_fresh_tracker():
    t = AdaptiveContextTracker()
    s = t.summary()
    assert s == {
        "last_seen": None,
        "change_count": 0,
        "seconds_since_last_change": None,
    }


def test_summary_after_baseline_only():
    t = AdaptiveContextTracker()
    t.observe("model-a")
    s = t.summary()
    assert s["last_seen"] == "model-a"
    assert s["change_count"] == 0
    # No change yet, so no elapsed-since-change figure
    assert s["seconds_since_last_change"] is None


def test_summary_after_change_includes_elapsed():
    t = AdaptiveContextTracker()
    t.observe("model-a")
    t.observe("model-b")
    s = t.summary()
    assert s["last_seen"] == "model-b"
    assert s["change_count"] == 1
    assert isinstance(s["seconds_since_last_change"], float)
    assert s["seconds_since_last_change"] >= 0.0
    # Sanity: a brand-new change should be sub-second
    assert s["seconds_since_last_change"] < 5.0
