"""A newly seeded singleton alias retains its token's exclusions and ownership."""

import json

import pytest

from tests.gateway import test_codex_effective_model_routing as fixtures
from tests.gateway.test_codex_effective_model_routing import BASE, OTHER, TARGET

home = fixtures.home


def _store(home, token="blocked", **status):
    row = {"id": "manual", "source": "manual:device_code", "label": "kept", "priority": 7,
           "auth_type": "oauth", "access_token": "blocked", "refresh_token": "manual-refresh",
           "base_url": BASE, "expires_at_ms": 4_000_000_000_000, **status}
    data = {"credential_pool": {"openai-codex": [row]}, "providers": {"openai-codex": {
        "tokens": {"access_token": token, "refresh_token": "singleton-refresh"}}}}
    (home / "auth.json").write_text(json.dumps(data))
    return row


@pytest.mark.parametrize("status", [
    {"last_status": "dead", "last_status_at": 1},
    {"model_cooldowns": {TARGET: 4_000_000_000}},
])
def test_alias_exclusion_survives_reload_and_manual_row_retirement(home, status):
    from agent.credential_pool import load_pool
    _store(home, **status)
    for _ in range(2):
        pool = load_pool("openai-codex")
        assert pool.select(model=TARGET) is None
        alias = next(entry for entry in pool.entries() if entry.source == "device_code")
        assert alias.refresh_token == "singleton-refresh"
        for key, value in status.items():
            assert getattr(alias, key) == value


@pytest.mark.parametrize("token", ["independent", "reauthenticated"])
def test_distinct_singleton_preserves_manual_status_and_uses_own_refresh_pair(home, token):
    from agent.credential_pool import load_pool
    original = _store(home, token, last_status="dead", model_cooldowns={TARGET: 4_000_000_000})
    pool = load_pool("openai-codex")
    selected = pool.select(model=TARGET)
    assert selected.runtime_api_key == token
    assert selected.refresh_token == "singleton-refresh"
    persisted = next(row for row in json.loads((home / "auth.json").read_text())["credential_pool"]["openai-codex"]
                     if row["id"] == "manual")
    for key, value in original.items():
        assert persisted[key] == value


def test_same_token_unrelated_bench_still_works_without_losing_account_metadata(home):
    from agent.credential_pool import load_pool
    original = _store(home, model_cooldowns={OTHER: 4_000_000_000})
    pool = load_pool("openai-codex")
    assert pool.select(model=TARGET).runtime_api_key == "blocked"
    assert pool.select(model=OTHER) is None
    manual = next(row for row in pool.entries() if row.id == "manual")
    assert (manual.label, manual.priority, manual.refresh_token) == (original["label"], 7, "manual-refresh")
