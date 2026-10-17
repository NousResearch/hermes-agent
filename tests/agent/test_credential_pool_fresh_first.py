"""fresh_first pool strategy.

Providers with rolling budget windows (e.g. Codex weekly) only start the window
clock when an account serves traffic: an unused entry reports
reset_at ≈ now+window forever and never regenerates quota. fresh_first selects
entries whose budget window is exactly 0% used first (activating every entry's
clock ASAP), then falls back to earliest_reset ordering within groups.
"""

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _pool_payload():
    return {
        "version": 1,
        "credential_pool": {
            "openai-codex": [
                {
                    "id": "ticking",
                    "label": "ticking-window",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual",
                    "access_token": "tok-ticking",
                },
                {
                    "id": "fresh-a",
                    "label": "fresh-window-a",
                    "auth_type": "oauth",
                    "priority": 1,
                    "source": "manual",
                    "access_token": "tok-fresh-a",
                },
                {
                    "id": "fresh-b",
                    "label": "fresh-window-b",
                    "auth_type": "oauth",
                    "priority": 2,
                    "source": "manual",
                    "access_token": "tok-fresh-b",
                },
            ]
        },
    }


def _cache(now):
    return {
        "version": 1,
        "fetched_at": now - 60,
        "providers": {
            "openai-codex": {
                "ticking": {
                    "label": "ticking-window",
                    "ok": True,
                    "windows": [
                        {"label": "Session", "used_percent": 42.0, "reset_at": now + 5 * 86400}
                    ],
                },
                "fresh-a": {
                    "label": "fresh-window-a",
                    "ok": True,
                    "windows": [
                        {"label": "Session", "used_percent": 0.0, "reset_at": now + 7 * 86400}
                    ],
                },
                "fresh-b": {
                    "label": "fresh-window-b",
                    "ok": True,
                    "windows": [
                        {"label": "Session", "used_percent": 0.0, "reset_at": now + 6 * 86400}
                    ],
                },
            }
        },
    }


@pytest.fixture()
def _pool_env(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy", lambda _provider: "fresh_first"
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_env",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._load_quota_cache", lambda: _cache(now)
    )
    _write_auth_store(tmp_path, _pool_payload())
    return now


def test_fresh_first_selects_unused_window_first(_pool_env):
    """A 0%-used entry wins over an earlier-resetting ticking entry."""
    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None
    assert entry.id in {"fresh-a", "fresh-b"}
    # Between the two fresh entries the earlier reset wins (fresh-b: +6d).
    assert entry.id == "fresh-b"


def test_fresh_first_falls_back_to_earliest_reset_when_all_ticking(_pool_env, monkeypatch):
    """No fresh windows left -> plain earliest_reset ordering."""
    from agent.credential_pool import load_pool

    now = _pool_env
    cache = _cache(now)
    cache["providers"]["openai-codex"]["fresh-a"]["windows"][0]["used_percent"] = 30.0
    cache["providers"]["openai-codex"]["fresh-b"]["windows"][0]["used_percent"] = 10.0
    monkeypatch.setattr("agent.credential_pool._load_quota_cache", lambda: cache)

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None
    assert entry.id == "ticking"  # earliest reset (+5d) among ticking entries


def test_fresh_first_no_cache_keeps_priority_order(tmp_path, monkeypatch):
    """No probe cache -> fail open to priority/file order."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy", lambda _provider: "fresh_first"
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_singletons",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr(
        "agent.credential_pool._seed_from_env",
        lambda provider, entries: (False, set()),
    )
    monkeypatch.setattr("agent.credential_pool._load_quota_cache", lambda: None)
    _write_auth_store(tmp_path, _pool_payload())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    entry = pool.select()
    assert entry is not None
    assert entry.id == "ticking"  # priority 0 / file order


def test_codex_construction_binding_honors_fresh_first(_pool_env, monkeypatch):
    """The raw file-order scan in _pool_codex_credential must be strategy-ordered."""
    import hermes_cli.auth_codex as auth_codex

    # The helper imports get_pool_strategy/_load_quota_cache from
    # agent.credential_pool at call time; the fixture already patched both there.
    entries = [
        {"id": "ticking", "access_token": "tok-ticking"},
        {"id": "fresh-a", "access_token": "tok-fresh-a"},
        {"id": "fresh-b", "access_token": "tok-fresh-b"},
    ]
    ordered = auth_codex._order_codex_pool_entries_for_strategy(entries)
    assert ordered[0]["id"] == "fresh-b"
    assert ordered[-1]["id"] == "ticking"


def test_codex_construction_binding_no_strategy_keeps_order(_pool_env, monkeypatch):
    import hermes_cli.auth_codex as auth_codex

    monkeypatch.setattr(
        "agent.credential_pool.get_pool_strategy", lambda _provider: "fill_first"
    )
    entries = [
        {"id": "ticking", "access_token": "tok-ticking"},
        {"id": "fresh-a", "access_token": "tok-fresh-a"},
    ]
    ordered = auth_codex._order_codex_pool_entries_for_strategy(entries)
    assert [e["id"] for e in ordered] == ["ticking", "fresh-a"]
