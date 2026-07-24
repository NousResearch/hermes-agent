"""Tests for live auto-decompose settings resolution (issue #49638).

The gateway dispatcher used to capture ``kanban.auto_decompose`` once at boot,
so a user who flipped it to ``false`` to STOP runaway auto-decompose (which had
created and launched tasks they didn't intend) found the flag had no effect
without a full gateway restart. ``_resolve_auto_decompose_settings`` is now
called every tick, reading the current config.
"""

from __future__ import annotations

import inspect

import pytest

from gateway import kanban_watchers
from gateway.kanban_watchers import (
    _AUTO_DECOMPOSE_FAILURES,
    MAX_AUTO_DECOMPOSE_FAILURES,
    _auto_decompose_is_poisoned,
    _auto_decompose_record_result,
    _resolve_auto_decompose_settings,
)


def test_enabled_by_default_when_key_absent():
    enabled, per_tick = _resolve_auto_decompose_settings(lambda: {"kanban": {}})
    assert enabled is True
    assert per_tick == 3


def test_disabled_when_flag_false():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": False}}
    )
    assert enabled is False


def test_per_tick_respected_and_clamped():
    enabled, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose": True, "auto_decompose_per_tick": 7}}
    )
    assert (enabled, per_tick) == (True, 7)

    # 0 is treated as "unset" by the `or 3` fallback → default 3 (a 0 per-tick
    # cap would disable progress, so falling back to the default is the safe read).
    _, per_tick_zero = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": 0}}
    )
    assert per_tick_zero == 3

    # A genuine negative value clamps up to 1.
    _, per_tick_neg = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": -5}}
    )
    assert per_tick_neg == 1


def test_malformed_per_tick_falls_back_to_default():
    _, per_tick = _resolve_auto_decompose_settings(
        lambda: {"kanban": {"auto_decompose_per_tick": "lots"}}
    )
    assert per_tick == 3


def test_config_read_error_fails_safe_disabled():
    """A transient config read failure must DISABLE auto-decompose, never
    silently fall back to the default-on behaviour the user turned off."""

    def _boom():
        raise RuntimeError("config read failed")

    enabled, per_tick = _resolve_auto_decompose_settings(_boom)
    assert enabled is False
    assert per_tick == 3


def test_non_dict_config_fails_safe():
    enabled, _ = _resolve_auto_decompose_settings(lambda: None)
    assert enabled is True  # no kanban key → default-on (not an error path)
    enabled2, _ = _resolve_auto_decompose_settings(lambda: ["not", "a", "dict"])
    assert enabled2 is True


def test_live_toggle_takes_effect_between_calls():
    """Simulate a user flipping the flag while the dispatcher runs: a later
    resolution reflects the new value without any restart."""
    state = {"kanban": {"auto_decompose": True}}
    assert _resolve_auto_decompose_settings(lambda: state)[0] is True
    # User edits config.yaml mid-run.
    state["kanban"]["auto_decompose"] = False
    assert _resolve_auto_decompose_settings(lambda: state)[0] is False


# ── Poison-card ceiling ────────────────────────────────────────────────────
# A decompose failure that leaves the card in triage ("LLM error", "LLM
# returned malformed JSON") is invisible to the block-loop circuit breaker,
# which only sees terminal outcomes of tasks that actually ran. Without a
# ceiling the dispatcher re-sends such a card to the aux model every tick,
# forever, at debug log level.


@pytest.fixture(autouse=True)
def _clear_failure_ledger():
    _AUTO_DECOMPOSE_FAILURES.clear()
    yield
    _AUTO_DECOMPOSE_FAILURES.clear()


def test_card_is_retried_up_to_the_cap_then_skipped():
    assert _auto_decompose_is_poisoned("t_poison") is False
    for expected in range(1, MAX_AUTO_DECOMPOSE_FAILURES):
        assert _auto_decompose_record_result("t_poison", False) == expected
        # Still eligible — transient aux failures deserve another chance.
        assert _auto_decompose_is_poisoned("t_poison") is False
    assert (
        _auto_decompose_record_result("t_poison", False)
        == MAX_AUTO_DECOMPOSE_FAILURES
    )
    assert _auto_decompose_is_poisoned("t_poison") is True


def test_success_clears_failure_history():
    """A card that fails transiently then decomposes must not stay penalised."""
    _auto_decompose_record_result("t_flaky", False)
    _auto_decompose_record_result("t_flaky", False)
    assert _auto_decompose_record_result("t_flaky", True) == 0
    assert _auto_decompose_is_poisoned("t_flaky") is False
    # And the next failure starts counting from scratch.
    assert _auto_decompose_record_result("t_flaky", False) == 1


def test_unconfigured_aux_client_never_poisons_a_card():
    """"auxiliary client unavailable" is returned before any model call: it
    spends nothing and is identical for every card. Counting it would leave a
    user who configures an aux model later with an inert triage column."""
    for _ in range(MAX_AUTO_DECOMPOSE_FAILURES * 2):
        assert (
            _auto_decompose_record_result(
                "t_noaux", False, "auxiliary client unavailable",
            )
            == 0
        )
    assert _auto_decompose_is_poisoned("t_noaux") is False
    # A genuine failure on the same card still counts from zero.
    assert (
        _auto_decompose_record_result(
            "t_noaux", False, "LLM returned malformed JSON",
        )
        == 1
    )


def test_failure_counts_are_per_card():
    for _ in range(MAX_AUTO_DECOMPOSE_FAILURES):
        _auto_decompose_record_result("t_bad", False)
    assert _auto_decompose_is_poisoned("t_bad") is True
    assert _auto_decompose_is_poisoned("t_good") is False


def test_poison_skip_precedes_the_per_tick_budget_spend():
    """Ordering invariant: the poison check must run BEFORE ``attempted += 1``.

    ``triage_ids`` is ordered, so if poisoned cards spent the per-tick budget
    they would sit at the head of the column and starve every decomposable
    card behind them — the cost bug would become a liveness bug.
    """
    src = inspect.getsource(kanban_watchers)
    body = src.split("def _auto_decompose_tick(")[1]
    skip = body.index("_auto_decompose_is_poisoned(tid)")
    spend = body.index("attempted += 1")
    assert skip < spend, (
        "poisoned cards must be skipped before the per-tick budget is spent"
    )
