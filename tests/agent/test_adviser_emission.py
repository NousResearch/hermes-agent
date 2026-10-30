"""Emission guard: noise, rank-aware dedupe, per-update budget, displacement."""

from __future__ import annotations

from agent.adviser_emission import EmissionGuard, normalize_note


def test_normalization_folds_case_and_punctuation():
    assert normalize_note("  *Stop*  ") == "stop"
    assert normalize_note("No issue; continue.") == "no issue continue"


def test_noise_and_empty_never_charge_the_budget():
    guard = EmissionGuard(budget_per_update=1)

    assert guard.admit("Stop.", rank=1, pending=False).reason == "noise"
    assert guard.admit("   ", rank=1, pending=False).reason == "empty"
    assert guard.admit("The retry never re-reads the file", rank=1, pending=False).accepted


def test_duplicate_and_rank_escalation():
    guard = EmissionGuard()
    assert guard.admit("same text", rank=1, pending=False).accepted
    assert guard.admit("Same text!", rank=1, pending=False).reason == "duplicate"
    assert guard.admit("SAME TEXT", rank=2, pending=False).accepted


def test_blocker_is_exempt_from_a_spent_budget():
    guard = EmissionGuard(budget_per_update=1)
    guard.begin_update()
    assert guard.admit("first nit", rank=1, pending=False).accepted
    assert guard.admit("second nit", rank=1, pending=False).reason == "rate-limit"
    assert guard.admit("stop the write; it truncates", rank=3, pending=False).accepted


def test_higher_rank_displaces_a_pending_note():
    guard = EmissionGuard(budget_per_update=1)
    guard.begin_update()
    first = guard.admit("small nit", rank=1, pending=True)
    assert first.accepted
    second = guard.admit("real concern", rank=2, pending=True)
    assert second.accepted and second.displaced_key == normalize_note("small nit")


def test_routed_notes_cannot_be_displaced():
    guard = EmissionGuard(budget_per_update=1)
    guard.begin_update()
    guard.admit("already delivered", rank=1, pending=False)
    assert guard.admit("higher concern", rank=2, pending=True).reason == "rate-limit"


def test_begin_update_resets_the_budget_but_not_dedupe():
    guard = EmissionGuard(budget_per_update=1)
    guard.begin_update()
    guard.admit("note a", rank=1, pending=False)
    guard.begin_update()
    assert guard.admit("note b", rank=1, pending=False).accepted
    assert guard.admit("note a", rank=1, pending=False).reason == "duplicate"
