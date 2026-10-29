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
    pool = cp.CredentialPool("anthropic", [_entry("first", 0), _entry("second", 1)])
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    selected = pool.select()

    assert selected.label == "second"
    assert [entry.label for entry in pool.entries()] == ["first", "second"]
    assert [entry.label for entry in pool.effective_entries()] == ["second", "first"]


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
    first = _entry("first", 0, status=cp.STATUS_EXHAUSTED)
    second = _entry("second", 1)
    pool = cp.CredentialPool("anthropic", [first, second])
    monkeypatch.setattr(pool, "_persist", lambda **kwargs: None)

    selected = pool.select()
    assert selected is not None
    assert selected.label == "second"
    pool._replace_entry(first, _entry("first", 0))
    selected = pool.select()
    assert selected is not None
    assert selected.label == "first"


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
