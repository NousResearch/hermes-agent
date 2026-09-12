import time
from types import SimpleNamespace

import pytest

from agent.credential_pool import (
    AUTH_TYPE_API_KEY,
    SOURCE_MANUAL,
    STATUS_EXHAUSTED,
    STRATEGY_EXPIRY_AWARE,
    CodexAccountUsageWindows,
    CredentialPool,
    PooledCredential,
    select_expiry_aware_entry,
)


NOW = 1_700_000_000.0


def test_short_window_is_admission_gate_not_weekly_tie_break():
    entries = [_entry("a"), _entry("b")]
    usage = {
        "a": _usage("acct-a", remaining=0.9, reset_at=NOW + 100, short_remaining=0.1),
        "b": _usage("acct-b", remaining=0.8, reset_at=NOW + 100, short_remaining=0.9),
    }
    assert select_expiry_aware_entry(entries, lambda e: usage[e.id], now=NOW).id == "a"


def test_exact_pin_becomes_eligible_after_its_recorded_reset():
    entry = _pool_entry("reset", 0)
    entry.last_status = STATUS_EXHAUSTED
    entry.last_error_reset_at = time.time() - 1
    pool = CredentialPool("openai-codex", [entry])
    assert pool.select_exact(entry.id) is not None


def _entry(entry_id: str):
    return SimpleNamespace(id=entry_id, extra={"account_id": f"acct-{entry_id}"})


def _usage(
    account_id: str,
    *,
    remaining: float,
    reset_at: float,
    observed_at: float = NOW,
    short_remaining: float = 1.0,
):
    return CodexAccountUsageWindows(
        account_id=account_id,
        remaining_fraction=remaining,
        reset_at=reset_at,
        observed_at=observed_at,
        short_window_remaining_fraction=short_remaining,
    )


def _pool_entry(entry_id: str, priority: int) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=entry_id,
        label=entry_id,
        auth_type=AUTH_TYPE_API_KEY,
        priority=priority,
        source=SOURCE_MANUAL,
        access_token=f"token-{entry_id}",
        extra={"account_id": f"acct-{entry_id}"},
    )


def test_expiry_aware_prefers_earliest_reset_for_fresh_usable_account_usage():
    entries = [_entry("late"), _entry("early")]
    usage_by_entry = {
        "late": _usage("acct-late", remaining=0.9, reset_at=NOW + 7_200),
        "early": _usage("acct-early", remaining=0.1, reset_at=NOW + 3_600),
    }

    selected = select_expiry_aware_entry(
        entries,
        lambda entry: usage_by_entry[entry.id],
        now=NOW,
        stale_after_seconds=60,
    )

    assert selected is not None
    assert selected.id == "early"


@pytest.mark.parametrize(
    "usage_by_entry",
    [
        {"first": _usage("acct-first", remaining=0.9, reset_at=NOW + 7_200)},
        {
            "first": _usage("acct-shared", remaining=0.9, reset_at=NOW + 7_200),
            "second": _usage("acct-shared", remaining=0.8, reset_at=NOW + 3_600),
        },
        {
            "first": _usage("acct-first", remaining=0.9, reset_at=NOW + 7_200, observed_at=NOW - 61),
            "second": _usage("acct-second", remaining=0.8, reset_at=NOW + 3_600),
        },
        {
            "first": _usage("acct-first", remaining=0.0, reset_at=NOW + 7_200),
            "second": _usage("acct-second", remaining=0.8, reset_at=NOW + 3_600),
        },
    ],
    ids=["missing", "conflicting_account_usage", "stale", "no_usable_quota"],
)
def test_expiry_aware_refuses_incomplete_conflicting_or_stale_usage(usage_by_entry):
    entries = [_entry("first"), _entry("second")]

    selected = select_expiry_aware_entry(
        entries,
        lambda entry: usage_by_entry.get(entry.id),
        now=NOW,
        stale_after_seconds=60,
    )

    assert selected is None


def test_expiry_aware_refuses_usage_without_an_explicit_matching_entry_account():
    entry = SimpleNamespace(id="first", extra={})

    assert select_expiry_aware_entry(
        [entry],
        lambda _: _usage("acct-first", remaining=0.9, reset_at=NOW + 3_600),
        now=NOW,
        stale_after_seconds=60,
    ) is None


def test_expiry_aware_pool_uses_prepared_snapshots_without_calling_a_lookup_under_lock():
    pool = CredentialPool("openai-codex", [_pool_entry("first", 0), _pool_entry("second", 1)])
    pool._strategy = STRATEGY_EXPIRY_AWARE
    observed_at = time.time()
    usage_by_entry = {
        "first": _usage("acct-first", remaining=0.9, reset_at=observed_at + 7_200, observed_at=observed_at),
        "second": _usage("acct-second", remaining=0.8, reset_at=observed_at + 3_600, observed_at=observed_at),
    }
    pool.set_expiry_aware_usage_snapshots(usage_by_entry)

    selected = pool.select()
    assert selected is not None
    assert selected.id == "second"

    pool.set_expiry_aware_usage_snapshots({})
    selected = pool.select()
    assert selected is not None
    assert selected.id == "first"


def test_expiry_aware_tie_uses_weekly_allowance_then_stable_entry_id():
    entries = [_entry("z-low"), _entry("a-high"), _entry("a-low")]
    usage_by_entry = {
        "z-low": _usage("acct-z-low", remaining=0.1, reset_at=NOW + 3_600),
        "a-high": _usage("acct-a-high", remaining=0.9, reset_at=NOW + 3_600),
        "a-low": _usage("acct-a-low", remaining=0.1, reset_at=NOW + 3_600),
    }

    selected = select_expiry_aware_entry(
        entries,
        lambda entry: usage_by_entry[entry.id],
        now=NOW,
        stale_after_seconds=60,
    )

    assert selected is not None
    assert selected.id == "a-high"

    equal_allowance = [_entry("z-low"), _entry("a-low")]
    selected = select_expiry_aware_entry(
        equal_allowance,
        lambda entry: usage_by_entry[entry.id],
        now=NOW,
        stale_after_seconds=60,
    )
    assert selected is not None
    assert selected.id == "a-low"


@pytest.mark.parametrize(
    ("now", "stale_after_seconds"),
    [
        (float("nan"), 60),
        (float("inf"), 60),
        (NOW, float("nan")),
        (NOW, float("inf")),
    ],
    ids=["nan_now", "infinite_now", "nan_ttl", "infinite_ttl"],
)
def test_expiry_aware_refuses_non_finite_time_inputs(now, stale_after_seconds):
    selected = select_expiry_aware_entry(
        [_entry("first")],
        lambda entry: _usage("acct-first", remaining=0.9, reset_at=NOW + 3_600),
        now=now,
        stale_after_seconds=stale_after_seconds,
    )

    assert selected is None


def test_expiry_aware_pool_fallback_skips_trusted_exhausted_usage():
    pool = CredentialPool("openai-codex", [_pool_entry("first", 0), _pool_entry("second", 1)])
    pool._strategy = STRATEGY_EXPIRY_AWARE
    observed_at = time.time()
    pool.set_expiry_aware_usage_snapshots({
        "first": _usage("acct-first", remaining=0.0, reset_at=observed_at + 7_200, observed_at=observed_at),
        "second": None,
    })

    selected = pool.select()

    assert selected is not None
    assert selected.id == "second"


def test_expiry_aware_pool_fallback_skips_fresh_short_window_exhaustion():
    pool = CredentialPool("openai-codex", [_pool_entry("first", 0), _pool_entry("second", 1)])
    pool._strategy = STRATEGY_EXPIRY_AWARE
    observed_at = time.time()
    pool.set_expiry_aware_usage_snapshots({
        "first": _usage(
            "acct-first", remaining=0.8, short_remaining=0.0,
            reset_at=observed_at + 7_200, observed_at=observed_at,
        ),
        "second": None,
    })

    selected = pool.select()

    assert selected is not None
    assert selected.id == "second"


def test_expiry_aware_pool_blocks_when_all_fresh_usage_is_exhausted():
    pool = CredentialPool("openai-codex", [_pool_entry("first", 0), _pool_entry("second", 1)])
    pool._strategy = STRATEGY_EXPIRY_AWARE
    observed_at = time.time()
    pool.set_expiry_aware_usage_snapshots({
        "first": _usage("acct-first", remaining=0.0, reset_at=observed_at + 7_200, observed_at=observed_at),
        "second": _usage("acct-second", remaining=0.0, reset_at=observed_at + 3_600, observed_at=observed_at),
    })

    assert pool.select() is None


def test_expiry_aware_pool_stale_exhaustion_does_not_block_fill_first_fallback():
    pool = CredentialPool("openai-codex", [_pool_entry("first", 0), _pool_entry("second", 1)])
    pool._strategy = STRATEGY_EXPIRY_AWARE
    observed_at = time.time()
    pool.set_expiry_aware_usage_snapshots({
        "first": _usage(
            "acct-first",
            remaining=0.0,
            reset_at=observed_at + 7_200,
            observed_at=observed_at - 61,
        ),
        "second": None,
    })

    selected = pool.select()

    assert selected is not None
    assert selected.id == "first"


def test_select_exact_requires_a_successful_same_account_refresh(monkeypatch):
    entry = _pool_entry("first", 0)
    pool = CredentialPool("openai-codex", [entry])
    monkeypatch.setattr(pool, "_entry_needs_refresh", lambda _entry: True)

    monkeypatch.setattr(pool, "_refresh_entry", lambda _entry, *, force: None)
    assert pool.select_exact("first") is None

    other_account = PooledCredential(
        **{**entry.__dict__, "extra": {"account_id": "acct-other"}},
    )
    monkeypatch.setattr(pool, "_refresh_entry", lambda _entry, *, force: other_account)
    assert pool.select_exact("first") is None

    monkeypatch.setattr(pool, "_refresh_entry", lambda _entry, *, force: entry)
    selected = pool.select_exact("first")
    assert selected is not None
    assert selected.id == "first"


def test_select_exact_does_not_return_a_currently_exhausted_entry():
    entry = _pool_entry("first", 0)
    entry.last_status = STATUS_EXHAUSTED
    pool = CredentialPool("openai-codex", [entry])

    assert pool.select_exact("first") is None
