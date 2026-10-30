from datetime import datetime

import agent.credential_pool as cp


def _entry(label, priority, *, status=cp.STATUS_OK):
    return cp.PooledCredential(
        provider="anthropic",
        id=f"id-{label}",
        label=label,
        auth_type=cp.AUTH_TYPE_API_KEY,
        priority=priority,
        source=cp.SOURCE_MANUAL,
        access_token=f"token-{label}",
        last_status=status,
        last_error_reset_at=4102444800 if status == cp.STATUS_EXHAUSTED else None,
    )


def _patch_scheduled_entries(monkeypatch, fixed_now):
    """Wrap _scheduled_entries with a fixed `now` so window tests pass at any wall-clock time."""
    orig = cp._scheduled_entries
    monkeypatch.setattr(
        cp,
        "_scheduled_entries",
        lambda provider, entries, now=None: orig(provider, entries, now=now or fixed_now),
    )


# Fixed daytime (KST 15:00 — inside 09:00–22:00 window).
_KST_DAYTIME = datetime.fromisoformat("2026-10-29T15:00:00+09:00")
# Fixed nighttime (KST 23:00 — inside 22:00–09:00 window).
_KST_NIGHTTIME = datetime.fromisoformat("2026-10-29T23:00:00+09:00")


def test_schedule_orders_fill_first_without_mutating_priority(monkeypatch):
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_strategies": {"anthropic": "fill_first"},
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "09:00", "end": "22:00", "prefer": ["second", "first"]},
                ],
            },
        },
    })
    # Use a fixed daytime so this test is not wall-clock-sensitive.
    _patch_scheduled_entries(monkeypatch, _KST_DAYTIME)

    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])
    persist_calls = []
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: persist_calls.append(kwargs))

    selected = pool.select()

    assert selected.label == "second"
    assert [entry.label for entry in pool.entries()] == ["first", "second"]
    assert [entry.label for entry in pool.effective_entries()] == ["second", "first"]
    # Schedule-aware fill_first must NOT write back to disk.
    assert len(persist_calls) == 0, (
        f"fill_first + schedule must not call _persist; called {len(persist_calls)} times"
    )


def test_schedule_falls_through_when_preferred_entry_exhausted_then_recovers(monkeypatch):
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "22:00", "end": "09:00", "prefer": ["first", "second"]},
                ],
            },
        },
    })
    # Window covers 22:00–09:00; fix clock at 23:00 KST so it is always active.
    _patch_scheduled_entries(monkeypatch, _KST_NIGHTTIME)

    first = _entry("first", 0, status=cp.STATUS_EXHAUSTED)
    second = _entry("second", 1)
    pool = cp.CredentialPool("anthropic", [first, second])
    persist_calls = []
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: persist_calls.append(kwargs))

    selected = pool.select()
    assert selected is not None
    assert selected.label == "second"
    pool._replace_entry(first, _entry("first", 0))
    selected = pool.select()
    assert selected is not None
    assert selected.label == "first"
    assert len(persist_calls) == 0, (
        f"fill_first + schedule must not call _persist; called {len(persist_calls)} times"
    )


def test_schedule_is_ignored_for_non_fill_first(monkeypatch):
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_strategies": {"anthropic": "round_robin"},
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "00:00", "end": "23:59", "prefer": ["second", "first"]},
                ],
            },
        },
    })
    # Arbitrary fixed clock — window covers the whole day so any time works.
    _patch_scheduled_entries(monkeypatch, _KST_DAYTIME)

    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    selected = pool.select()
    assert selected is not None
    assert selected.label == "first"


def test_invalid_schedule_does_not_break_pool(monkeypatch):
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Not/AZone",
                "windows": [{"start": "bad", "end": "09:00", "prefer": ["first"]}],
            },
        },
    })
    # Invalid timezone causes early return before `now` is used; fix it anyway
    # so the test is deterministic if the error path ever changes.
    _patch_scheduled_entries(monkeypatch, _KST_DAYTIME)

    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])

    selected = pool.select()
    assert selected is not None
    assert selected.label == "first"


def test_midnight_window_helper_uses_existing_time_window_semantics(monkeypatch):
    entries = [_entry("first", 0), _entry("second", 1)]
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [{"start": "22:00", "end": "09:00", "prefer": ["second"]}],
            },
        },
    })
    ordered, index = cp._scheduled_entries(
        "anthropic",
        entries,
        now=datetime.fromisoformat("2026-10-29T00:30:00+09:00"),
    )
    assert index == 1
    assert [entry.label for entry in ordered] == ["second", "first"]


# ---------------------------------------------------------------------------
# Jira 3: peek() / effective_entries() display-contract alignment
# ---------------------------------------------------------------------------

def test_peek_and_effective_entries_use_same_schedule_order(monkeypatch):
    """peek() must agree with effective_entries()[0] for fill_first + active schedule."""
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_strategies": {"anthropic": "fill_first"},
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "09:00", "end": "22:00", "prefer": ["second", "first"]},
                ],
            },
        },
    })
    _patch_scheduled_entries(monkeypatch, _KST_DAYTIME)

    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    peeked = pool.peek()
    effective = pool.effective_entries()

    assert peeked is not None, "peek() must not return None when entries are available"
    assert effective, "effective_entries() must not be empty"
    assert peeked.label == effective[0].label, (
        f"peek()={peeked.label!r} must match effective_entries()[0]={effective[0].label!r}"
    )
    assert peeked.label == "second"


def test_peek_after_select_returns_current_not_schedule_head(monkeypatch):
    """Once a current entry is set, peek() returns it regardless of the schedule."""
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {
        "credential_pool_strategies": {"anthropic": "fill_first"},
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "09:00", "end": "22:00", "prefer": ["second", "first"]},
                ],
            },
        },
    })
    _patch_scheduled_entries(monkeypatch, _KST_DAYTIME)

    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    selected = pool.select()
    assert selected.label == "second"
    assert pool.peek().label == "second"


# ---------------------------------------------------------------------------
# Jira 2: real-schema replay via load_pool (dummy tokens, prod-matching structure)
# ---------------------------------------------------------------------------

def test_schema_replay_load_pool_honours_schedule(monkeypatch):
    """load_pool('anthropic') honours schedule with prod-schema auth.json entries (dummy tokens).

    Replicates the structure of the kamiyu profile auth.json (without real
    tokens/fingerprints) and asserts that:
      - 주간창 (09:00–22:00 KST) → cg.chae-claude
      - 야간창 (22:00–09:00 KST) → claude_code
      - 스케줄 없음                → file priority order (priority 0 = cg.chae-claude)
    """
    # Prod-schema replica: label/source/auth_type/priority match production;
    # access_token replaced with a safe dummy value.
    prod_schema_entries = [
        {
            "id": "dummy-cg-id",
            "label": "cg.chae-claude",
            "auth_type": "oauth",
            "priority": 0,
            "source": "hermes_pkce_shared",
            "last_status": None,
            "access_token": "dummy-cg-token",
            "expires_at_ms": 9_999_999_999_999,
            "request_count": 0,
        },
        {
            "id": "dummy-cc-id",
            "label": "claude_code",
            "auth_type": "oauth",
            "priority": 1,
            "source": "claude_code",
            "last_status": "ok",
            "access_token": "dummy-cc-token",
            "expires_at_ms": 9_999_999_999_999,
            "request_count": 0,
        },
    ]

    # Replace the I/O layer with dummy data; preserve all pool logic under test.
    monkeypatch.setattr(cp, "read_credential_pool", lambda provider: list(prod_schema_entries))
    monkeypatch.setattr(cp, "write_credential_pool", lambda *a, **kw: None)
    monkeypatch.setattr(cp, "_seed_from_singletons", lambda *a, **kw: (False, set()))
    monkeypatch.setattr(cp, "_seed_from_env", lambda *a, **kw: (False, set()))
    monkeypatch.setattr(cp, "_prune_stale_seeded_entries", lambda *a, **kw: False)
    monkeypatch.setattr(cp, "_normalize_pool_priorities", lambda *a, **kw: False)
    monkeypatch.setattr(cp, "_migrate_legacy_hermes_pkce_entries", lambda *a, **kw: False)

    schedule_cfg = {
        "credential_pool_strategies": {"anthropic": "fill_first"},
        "credential_pool_schedule": {
            "anthropic": {
                "tz": "Asia/Seoul",
                "windows": [
                    {"start": "09:00", "end": "22:00", "prefer": ["cg.chae-claude"]},
                    {"start": "22:00", "end": "09:00", "prefer": ["claude_code"]},
                ],
            },
        },
    }
    monkeypatch.setattr(cp, "_load_config_safe", lambda: schedule_cfg)

    orig_se = cp._scheduled_entries

    persist_calls: list = []

    # --- 주간창 ---
    monkeypatch.setattr(
        cp, "_scheduled_entries",
        lambda p, e, now=None: orig_se(p, e, now=now or _KST_DAYTIME),
    )
    pool_day = cp.load_pool("anthropic")
    monkeypatch.setattr(pool_day, "_persist", lambda **kw: persist_calls.append(kw))
    day_selected = pool_day.select()
    assert day_selected is not None, "주간창: select() returned None"
    day_label = day_selected.label
    assert day_label == "cg.chae-claude", (
        f"주간창: expected cg.chae-claude, got {day_label!r}"
    )

    # --- 야간창 ---
    monkeypatch.setattr(
        cp, "_scheduled_entries",
        lambda p, e, now=None: orig_se(p, e, now=now or _KST_NIGHTTIME),
    )
    pool_night = cp.load_pool("anthropic")
    monkeypatch.setattr(pool_night, "_persist", lambda **kw: persist_calls.append(kw))
    night_selected = pool_night.select()
    assert night_selected is not None, "야간창: select() returned None"
    night_label = night_selected.label
    assert night_label == "claude_code", (
        f"야간창: expected claude_code, got {night_label!r}"
    )

    # --- 스케줄 없음: file priority 순서 ---
    monkeypatch.setattr(cp, "_load_config_safe", lambda: {})
    # Restore original _scheduled_entries (no-op schedule)
    monkeypatch.setattr(cp, "_scheduled_entries", orig_se)
    pool_nosched = cp.load_pool("anthropic")
    monkeypatch.setattr(pool_nosched, "_persist", lambda **kw: persist_calls.append(kw))
    nosched_selected = pool_nosched.select()
    assert nosched_selected is not None
    assert nosched_selected.label == "cg.chae-claude", (
        f"스케줄 없음: expected cg.chae-claude (priority 0), got {nosched_selected.label!r}"
    )

    # fill_first + schedule must NEVER call _persist.
    assert len(persist_calls) == 0, (
        f"fill_first + schedule must not call _persist; called {len(persist_calls)} times"
    )
