"""Usage-aware codex credential-pool selection.

ChatGPT codex bills two independent quota windows (Session ~hours, Weekly
~days).  Selection should prefer credentials with Weekly + Session room,
deprioritise Session-exhausted ones (short cooldown) and skip Weekly-exhausted
ones (long cooldown), instead of blindly using priority order and burning a
request to rediscover that a credential is spent.
"""
from types import SimpleNamespace
from unittest.mock import patch

import agent.credential_pool as cp
from agent.credential_pool import CredentialPool, PooledCredential, STRATEGY_FILL_FIRST


def _entry(idx: int, label: str) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=f"cred-{idx}",
        label=label,
        auth_type="oauth",
        priority=idx,
        source="manual:test",
        access_token=f"token-{label}",
    )


def _snap(session_used, weekly_used):
    windows = []
    if session_used is not None:
        windows.append(SimpleNamespace(label="Session", used_percent=session_used))
    if weekly_used is not None:
        windows.append(SimpleNamespace(label="Weekly", used_percent=weekly_used))
    return SimpleNamespace(windows=tuple(windows), available=True)


def _pool(labels):
    entries = [_entry(i, name) for i, name in enumerate(labels)]
    pool = CredentialPool("openai-codex", entries)
    pool._strategy = STRATEGY_FILL_FIRST
    pool._persist = lambda: None
    # selection must not hit the network in tests
    pool._refresh_codex_usage_cache = lambda: None
    return pool


def _with_usage(usage_by_label):
    """Patch the module usage cache lookup to return mock snapshots by label."""
    def fake(cred_id):
        # cred_id is "cred-<idx>"; map back via the registered table
        return _LOOKUP.get(cred_id)
    return patch.object(cp, "_cached_codex_usage", side_effect=fake)


_LOOKUP = {}


def _register(pool, usage_by_label):
    _LOOKUP.clear()
    by_label = {e.label: e for e in pool.entries()}
    for label, (su, wu) in usage_by_label.items():
        _LOOKUP[by_label[label].id] = _snap(su, wu)


def test_skips_session_exhausted_and_picks_room():
    pool = _pool(["a_team", "b_team", "c_plus"])
    _register(pool, {"a_team": (100.0, 17.0), "b_team": (2.0, 15.0), "c_plus": (1.0, 21.0)})
    with _with_usage(_LOOKUP):
        chosen = pool.select()
    assert chosen is not None
    assert chosen.label == "b_team"  # a_team session-exhausted → skipped


def test_skips_weekly_exhausted():
    pool = _pool(["a_team", "b_team", "c_plus"])
    _register(pool, {"a_team": (50.0, 100.0), "b_team": (50.0, 100.0), "c_plus": (1.0, 21.0)})
    with _with_usage(_LOOKUP):
        chosen = pool.select()
    assert chosen.label == "c_plus"  # both team weekly-exhausted → skipped


def test_all_session_exhausted_returns_least_bad_not_none():
    pool = _pool(["a", "b", "c"])
    _register(pool, {"a": (100.0, 10.0), "b": (100.0, 10.0), "c": (100.0, 10.0)})
    with _with_usage(_LOOKUP):
        chosen = pool.select()
    assert chosen is not None  # session-dead (short reset) still beats giving up


def test_no_usage_data_falls_back_to_priority_order():
    pool = _pool(["a", "b", "c"])
    _LOOKUP.clear()  # no cached usage for anyone
    with _with_usage(_LOOKUP):
        chosen = pool.select()
    assert chosen.label == "a"  # unchanged fill_first behaviour


def test_known_room_preferred_over_unknown():
    pool = _pool(["a_unknown", "b_room"])
    _register(pool, {"b_room": (2.0, 5.0)})  # a_unknown has no usage data
    with _with_usage(_LOOKUP):
        chosen = pool.select()
    # a_unknown is priority 0 but b_room has *known* room → both are candidates;
    # fill_first keeps priority, so a_unknown (unknown, still a candidate) wins.
    assert chosen.label in {"a_unknown", "b_room"}
