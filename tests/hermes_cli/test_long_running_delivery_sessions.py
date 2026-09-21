"""The long-running-delivery-session watchdog (#93091 follow-up, card item 2).

The 2026-09-20 orphan (a delivery child whose requester was killed) held a target profile's
``state.db`` for 1h46m and, in the end, surfaced only as prose in somebody's next delivery: the
``target_session_live`` refusal. This watchdog makes it its own signal — WHICH profile, WHICH pid,
HOW LONG — and it is a QUERY: it flags and never kills, because a session's own owner is the only
authority on that session's life (the requester watch is what ends a stranded delivery child).

The population matters as much as the threshold: an interactive CLI session legitimately lives for
hours, so age alone proves nothing. Only a provable delivery child is ever named.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from hermes_cli import active_sessions as act


def _lease(pid: int, started_at: float, *, session_id: str = "s-1", surface: str = "cli") -> dict:
    return {
        "lease_id": f"lease-{pid}", "session_id": session_id, "surface": surface, "pid": pid,
        "process_start_time": None, "started_at": started_at, "updated_at": started_at,
    }


@pytest.fixture
def homes(monkeypatch, tmp_path):
    """A root home with one live named profile, both registry-homed onto *tmp_path*."""
    root = tmp_path / "hermes"
    (root / "profiles" / "firstmate").mkdir(parents=True)
    monkeypatch.setattr(act, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr(act, "named_profile_is_live", lambda _p: True)
    return root


def _snapshot(monkeypatch, entries, *, profile: str = "firstmate"):
    """A registry snapshot whose entries belong to ONE home.

    Real registries are per-home (a lease is written to the home its process ran under), so the
    root home and the profile home never serve the same lease — mirroring that here keeps the
    sweep's per-home fan-out honest instead of double-counting a stub.
    """
    def _entries(registry_home=None):
        home = str(registry_home or "")
        return list(entries) if home.endswith(profile) else []

    monkeypatch.setattr(act, "active_session_registry_snapshot", _entries)


# ── threshold: derived from the delivery turn's own ceiling ─────────────────

def test_threshold_exceeds_the_whole_delivery_turn_budget(monkeypatch):
    """A turn that runs its entire budget (every attempt) is slow, never "stuck"."""
    from tools.bot_relay import TURN_ATTEMPT_TIMEOUT_SECONDS, TURN_MAX_ATTEMPTS

    monkeypatch.delenv(act.LONG_RUNNING_SESSION_ENV, raising=False)
    assert act.long_running_session_seconds() > TURN_ATTEMPT_TIMEOUT_SECONDS * TURN_MAX_ATTEMPTS


def test_an_operator_can_tune_the_threshold(monkeypatch):
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "90")
    assert act.long_running_session_seconds() == 90.0


def test_an_invalid_threshold_falls_back_instead_of_disabling_the_watchdog(monkeypatch):
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "not-a-number")
    assert act.long_running_session_seconds() > 0


def test_a_zero_threshold_disables_the_watchdog(monkeypatch, homes):
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "0")
    _snapshot(monkeypatch, [_lease(os.getpid(), time.time() - 10_000)])
    assert act.long_running_delivery_sessions() == []


# ── the population: only a provable delivery child is ever named ────────────

def test_only_a_delivery_child_is_flagged(monkeypatch, homes):
    """An interactive session that has lived for hours is not this watchdog's business."""
    old = time.time() - 10_000
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(os.getpid(), old)])
    # This test process IS a live pid, but it is not a one-shot delivery child.
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: False)
    assert act.long_running_delivery_sessions() == []


def test_a_provably_long_delivery_child_is_flagged_with_its_evidence(monkeypatch, homes):
    old = time.time() - 7_200
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(4242, old, session_id="firstmate-bot", surface="cli")])
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: pid == 4242)

    flagged = act.long_running_delivery_sessions()

    assert len(flagged) == 1
    entry = flagged[0]
    assert entry["pid"] == 4242
    assert entry["age_seconds"] == pytest.approx(7_200, abs=60)
    assert entry["session_id"] == "firstmate-bot"
    assert entry["profile_home"].endswith("firstmate"), "the profile that was held is named"


def test_a_recent_delivery_child_is_not_flagged(monkeypatch, homes):
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(4242, time.time() - 5)])
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: True)
    assert act.long_running_delivery_sessions() == []


def test_the_longest_held_session_comes_first(monkeypatch, homes):
    now = time.time()
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(1, now - 120), _lease(2, now - 9_000), _lease(3, now - 600)])
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: True)

    assert [e["pid"] for e in act.long_running_delivery_sessions()] == [2, 3, 1]


# ── never raises, never kills ───────────────────────────────────────────────

def test_an_unreadable_registry_is_skipped_not_raised(monkeypatch, homes):
    """Housekeeping must never wedge on this: one bad profile must not stop the sweep."""
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")

    def _boom(registry_home=None):
        raise act.ActiveSessionRegistryError("registry unreadable")

    monkeypatch.setattr(act, "active_session_registry_snapshot", _boom)
    assert act.long_running_delivery_sessions() == []


def test_an_entry_without_a_start_time_is_never_called_stuck(monkeypatch, homes):
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(1, 0.0)])  # started_at=0 -> age falls back to 0
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: True)
    assert act.long_running_delivery_sessions() == []


def test_the_sweep_calls_no_process_termination(monkeypatch, homes):
    """Flag, never kill: the policy is query-only by design."""
    monkeypatch.setenv(act.LONG_RUNNING_SESSION_ENV, "60")
    _snapshot(monkeypatch, [_lease(4242, time.time() - 9_000)])
    monkeypatch.setattr(act, "_is_delivery_child", lambda pid: True)

    def _no_kill(*_a, **_k):
        raise AssertionError("the watchdog must never terminate a session")

    monkeypatch.setattr(os, "kill", _no_kill)
    assert len(act.long_running_delivery_sessions()) == 1


# ── the housekeeping chore that surfaces it ─────────────────────────────────

def test_the_gateway_chore_logs_a_warning_naming_the_holder(monkeypatch, caplog):
    from gateway import run as gw_run

    monkeypatch.setattr(
        act, "long_running_delivery_sessions",
        lambda: [{
            "pid": 4019250, "session_id": "firstmate-bot", "surface": "cli",
            "age_seconds": 6393.0, "profile_home": "/home/x/.hermes/profiles/firstmate",
        }],
    )
    with caplog.at_level("WARNING"):
        gw_run._housekeeping_long_running_sessions()

    logged = caplog.text
    assert "firstmate" in logged and "4019250" in logged
    assert "107" in logged, "the age is reported in minutes, so the operator sees 1h46m as such"
