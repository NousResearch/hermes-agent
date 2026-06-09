"""Tests for agent/codex_spend_guard.py — proactive global Codex spend guard.

Cross-process, file-backed sliding-window circuit breaker.  Tests cover the
pure window evaluation, config-clamping limit resolution, the file-backed
guard happy path, cross-process sharing (two instances / one ledger),
token-ceiling enforcement, pruning, fail-open behaviour, and snapshots.
"""

import json
import os

import pytest

from agent.codex_spend_guard import (
    DAY_SECONDS,
    HOUR_SECONDS,
    MAX_CALLS_PER_DAY,
    MAX_CALLS_PER_HOUR,
    MAX_TOKENS_PER_DAY,
    CodexSpendCapError,
    CodexSpendGuard,
    Decision,
    Limits,
    evaluate,
    resolve_limits,
)


# ---------------------------------------------------------------------------
# 1. Pure window evaluation
# ---------------------------------------------------------------------------


def _hard_limits() -> Limits:
    return resolve_limits(None)


def test_evaluate_under_all_ceilings_allowed():
    now = 1_000_000.0
    limits = _hard_limits()
    decision = evaluate([now - 10, now - 20], [(now - 5, 100)], now, limits)
    assert decision == Decision(True, None)


def test_evaluate_at_calls_per_hour_denied():
    now = 1_000_000.0
    limits = Limits(
        max_calls_per_hour=3,
        max_calls_per_day=MAX_CALLS_PER_DAY,
        max_tokens_per_day=MAX_TOKENS_PER_DAY,
    )
    call_times = [now - 1, now - 2, now - 3]  # exactly 3 in window
    decision = evaluate(call_times, [], now, limits)
    assert decision.allowed is False
    assert decision.reason == "calls_per_hour"


def test_evaluate_at_calls_per_day_denied():
    now = 1_000_000.0
    limits = Limits(
        max_calls_per_hour=MAX_CALLS_PER_HOUR,
        max_calls_per_day=2,
        max_tokens_per_day=MAX_TOKENS_PER_DAY,
    )
    # Two calls within the day window but outside the hour window so the
    # hour check does not trip first.
    call_times = [now - HOUR_SECONDS - 10, now - HOUR_SECONDS - 20]
    decision = evaluate(call_times, [], now, limits)
    assert decision.allowed is False
    assert decision.reason == "calls_per_day"


def test_evaluate_at_tokens_per_day_denied():
    now = 1_000_000.0
    limits = Limits(
        max_calls_per_hour=MAX_CALLS_PER_HOUR,
        max_calls_per_day=MAX_CALLS_PER_DAY,
        max_tokens_per_day=1_000,
    )
    token_events = [(now - 10, 600), (now - 20, 400)]  # sum == 1000
    decision = evaluate([], token_events, now, limits)
    assert decision.allowed is False
    assert decision.reason == "tokens_per_day"


def test_evaluate_excludes_old_entries():
    now = 1_000_000.0
    limits = Limits(
        max_calls_per_hour=1,
        max_calls_per_day=1,
        max_tokens_per_day=100,
    )
    # All entries older than the day window → excluded from every count → allowed.
    call_times = [now - DAY_SECONDS - 1, now - DAY_SECONDS - 100]
    token_events = [(now - DAY_SECONDS - 1, 10_000)]
    decision = evaluate(call_times, token_events, now, limits)
    assert decision == Decision(True, None)


def test_evaluate_check_order_hour_before_day_before_tokens():
    now = 1_000_000.0
    limits = Limits(max_calls_per_hour=1, max_calls_per_day=1, max_tokens_per_day=1)
    # Everything breaches; hour must win.
    decision = evaluate([now], [(now, 999)], now, limits)
    assert decision.reason == "calls_per_hour"


# ---------------------------------------------------------------------------
# 2. resolve_limits — clamp config to hard ceilings
# ---------------------------------------------------------------------------


def test_resolve_limits_none_is_hard_ceilings():
    limits = resolve_limits(None)
    assert limits.max_calls_per_hour == MAX_CALLS_PER_HOUR
    assert limits.max_calls_per_day == MAX_CALLS_PER_DAY
    assert limits.max_tokens_per_day == MAX_TOKENS_PER_DAY


def test_resolve_limits_empty_dict_is_hard_ceilings():
    limits = resolve_limits({})
    assert limits.max_calls_per_hour == MAX_CALLS_PER_HOUR


def test_resolve_limits_below_ceiling_honored():
    limits = resolve_limits(
        {
            "codex_spend_cap": {
                "max_calls_per_hour": 10,
                "max_calls_per_day": 50,
                "max_tokens_per_day": 1_000,
            }
        }
    )
    assert limits.max_calls_per_hour == 10
    assert limits.max_calls_per_day == 50
    assert limits.max_tokens_per_day == 1_000


def test_resolve_limits_above_ceiling_clamped():
    limits = resolve_limits(
        {
            "codex_spend_cap": {
                "max_calls_per_hour": 10_000,
                "max_calls_per_day": 10_000,
                "max_tokens_per_day": 10**12,
            }
        }
    )
    assert limits.max_calls_per_hour == MAX_CALLS_PER_HOUR
    assert limits.max_calls_per_day == MAX_CALLS_PER_DAY
    assert limits.max_tokens_per_day == MAX_TOKENS_PER_DAY


@pytest.mark.parametrize("bad", [0, -5, "x", 1.5, None, True])
def test_resolve_limits_bad_values_fall_back_to_ceiling(bad):
    limits = resolve_limits({"codex_spend_cap": {"max_calls_per_hour": bad}})
    assert limits.max_calls_per_hour == MAX_CALLS_PER_HOUR


def test_resolve_limits_partial_config_others_default():
    limits = resolve_limits({"codex_spend_cap": {"max_calls_per_hour": 5}})
    assert limits.max_calls_per_hour == 5
    assert limits.max_calls_per_day == MAX_CALLS_PER_DAY
    assert limits.max_tokens_per_day == MAX_TOKENS_PER_DAY


# ---------------------------------------------------------------------------
# 3. reserve happy path + per-hour ceiling
# ---------------------------------------------------------------------------


def _ledger(tmp_path) -> str:
    return str(tmp_path / "codex_spend.json")


def test_reserve_happy_path_then_denies_over_hour_ceiling(tmp_path):
    limits = Limits(max_calls_per_hour=3, max_calls_per_day=100, max_tokens_per_day=10**9)
    guard = CodexSpendGuard(ledger_path=_ledger(tmp_path), limits=limits)
    now = 5_000_000.0

    for i in range(3):
        res = guard.reserve(now=now + i)
        assert res.allowed is True
        assert res.failed_open is False

    denied = guard.reserve(now=now + 3)
    assert denied.allowed is False
    assert denied.reason == "calls_per_hour"
    assert denied.failed_open is False


def test_reserve_persists_calls_to_ledger(tmp_path):
    path = _ledger(tmp_path)
    guard = CodexSpendGuard(ledger_path=path, limits=resolve_limits(None))
    now = 5_000_000.0
    guard.reserve(now=now)
    guard.reserve(now=now + 1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["call_times"]) == 2


def test_reserve_denied_does_not_append_call(tmp_path):
    path = _ledger(tmp_path)
    limits = Limits(max_calls_per_hour=1, max_calls_per_day=100, max_tokens_per_day=10**9)
    guard = CodexSpendGuard(ledger_path=path, limits=limits)
    now = 5_000_000.0
    guard.reserve(now=now)
    guard.reserve(now=now + 1)  # denied

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["call_times"]) == 1


# ---------------------------------------------------------------------------
# 4. Cross-process sharing (two instances, one ledger)
# ---------------------------------------------------------------------------


def test_cross_process_sharing(tmp_path):
    path = _ledger(tmp_path)
    limits = Limits(max_calls_per_hour=4, max_calls_per_day=100, max_tokens_per_day=10**9)
    now = 7_000_000.0

    guard_a = CodexSpendGuard(ledger_path=path, limits=limits)
    for i in range(4):
        assert guard_a.reserve(now=now + i).allowed is True

    # Separate instance, same file — simulates a separate watcher OS process.
    guard_b = CodexSpendGuard(ledger_path=path, limits=limits)
    denied = guard_b.reserve(now=now + 4)
    assert denied.allowed is False
    assert denied.reason == "calls_per_hour"


# ---------------------------------------------------------------------------
# 5. Token ceiling
# ---------------------------------------------------------------------------


def test_record_tokens_accumulates_and_blocks_reserve(tmp_path):
    path = _ledger(tmp_path)
    limits = Limits(max_calls_per_hour=100, max_calls_per_day=100, max_tokens_per_day=1_000)
    guard = CodexSpendGuard(ledger_path=path, limits=limits)
    now = 8_000_000.0

    guard.record_tokens(400, now=now)
    guard.record_tokens(600, now=now + 1)  # total 1000 == ceiling

    snap = guard.snapshot(now=now + 2)
    assert snap["tokens_last_day"] == 1_000

    denied = guard.reserve(now=now + 2)
    assert denied.allowed is False
    assert denied.reason == "tokens_per_day"


def test_record_tokens_ignores_non_positive_and_non_int(tmp_path):
    path = _ledger(tmp_path)
    guard = CodexSpendGuard(ledger_path=path, limits=resolve_limits(None))
    now = 8_000_000.0
    guard.record_tokens(0, now=now)
    guard.record_tokens(-50, now=now)
    guard.record_tokens("nope", now=now)  # type: ignore[arg-type]
    guard.record_tokens(None, now=now)  # type: ignore[arg-type]

    assert guard.snapshot(now=now)["tokens_last_day"] == 0


# ---------------------------------------------------------------------------
# 6. Pruning
# ---------------------------------------------------------------------------


def test_pruning_drops_old_entries_on_reserve(tmp_path):
    path = _ledger(tmp_path)
    limits = Limits(max_calls_per_hour=2, max_calls_per_day=2, max_tokens_per_day=10**9)
    now = 9_000_000.0

    # Seed ledger with stale entries (older than the day window).
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "call_times": [now - DAY_SECONDS - 100, now - DAY_SECONDS - 200],
                "token_events": [[now - DAY_SECONDS - 100, 999_999]],
            },
            f,
        )

    guard = CodexSpendGuard(ledger_path=path, limits=limits)
    res = guard.reserve(now=now)
    assert res.allowed is True  # stale calls excluded → under ceiling

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Stale entries pruned; only the fresh reserve remains.
    assert data["call_times"] == [now]
    assert data["token_events"] == []


# ---------------------------------------------------------------------------
# 7. Fail-open
# ---------------------------------------------------------------------------


def test_reserve_fails_open_on_corrupt_ledger(tmp_path):
    path = _ledger(tmp_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("}{ not json at all \x00\xff garbage")

    guard = CodexSpendGuard(ledger_path=path, limits=resolve_limits(None))
    # Corrupt file is treated as empty (fail-open at the data layer), so the
    # reserve still succeeds; it must never raise.
    res = guard.reserve(now=1.0)
    assert res.allowed is True


def test_reserve_fails_open_when_parent_is_a_file(tmp_path):
    # A path whose parent is itself a file → mkdir/open will raise → fail open.
    parent_file = tmp_path / "iam_a_file"
    parent_file.write_text("x", encoding="utf-8")
    bad_path = str(parent_file / "ledger.json")

    guard = CodexSpendGuard(ledger_path=bad_path, limits=resolve_limits(None))
    res = guard.reserve(now=1.0)
    assert res.allowed is True
    assert res.failed_open is True


def test_record_tokens_does_not_raise_on_bad_path(tmp_path):
    parent_file = tmp_path / "afile"
    parent_file.write_text("x", encoding="utf-8")
    bad_path = str(parent_file / "ledger.json")
    guard = CodexSpendGuard(ledger_path=bad_path, limits=resolve_limits(None))
    # Must not raise.
    guard.record_tokens(100, now=1.0)


def test_snapshot_fails_open_to_zeros_on_bad_path(tmp_path):
    parent_file = tmp_path / "afile2"
    parent_file.write_text("x", encoding="utf-8")
    bad_path = str(parent_file / "ledger.json")
    guard = CodexSpendGuard(ledger_path=bad_path, limits=resolve_limits(None))
    snap = guard.snapshot(now=1.0)
    assert snap["calls_last_hour"] == 0
    assert snap["calls_last_day"] == 0
    assert snap["tokens_last_day"] == 0


# ---------------------------------------------------------------------------
# 8. Snapshot window counts
# ---------------------------------------------------------------------------


def test_snapshot_window_counts(tmp_path):
    path = _ledger(tmp_path)
    guard = CodexSpendGuard(ledger_path=path, limits=resolve_limits(None))
    now = 10_000_000.0

    # 2 calls in the last hour, plus 1 older-than-hour but within-day.
    guard.reserve(now=now - 10)
    guard.reserve(now=now - 20)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["call_times"].append(now - HOUR_SECONDS - 50)  # within day, outside hour
    data["token_events"] = [[now - 5, 1_234], [now - DAY_SECONDS - 5, 9_999]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    snap = guard.snapshot(now=now)
    assert snap["calls_last_hour"] == 2
    assert snap["calls_last_day"] == 3
    assert snap["tokens_last_day"] == 1_234
    assert snap["limits"]["max_calls_per_hour"] == MAX_CALLS_PER_HOUR
    assert snap["limits"]["max_calls_per_day"] == MAX_CALLS_PER_DAY
    assert snap["limits"]["max_tokens_per_day"] == MAX_TOKENS_PER_DAY


# ---------------------------------------------------------------------------
# CodexSpendCapError
# ---------------------------------------------------------------------------


def test_codex_spend_cap_error_carries_reason():
    err = CodexSpendCapError("calls_per_hour")
    assert err.reason == "calls_per_hour"
    assert "calls_per_hour" in str(err)


def test_default_ledger_path_expanduser(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    guard = CodexSpendGuard(limits=resolve_limits(None))
    assert str(guard.ledger_path).startswith(str(tmp_path))
    assert guard.ledger_path.name == "codex_spend.json"
