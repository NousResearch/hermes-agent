"""Usage-admission ceilings must hold on every credential-selection path, not only ``select()``.

``choose()``/``commit()`` (aux ``PoolChoice``, route binding), ``peek()`` and ``reclaim()`` read the same
``_available_entries`` filter as ``select()``, but the filter only sees what the admission cache holds. On a
cold cache (first use in a process, or after the 300 s TTL) they used to skip the probe and hand out a
credential that was over its configured ceiling.
"""
from datetime import datetime, timezone, UTC
from types import SimpleNamespace

import pytest

from agent import account_usage
from agent import credential_pool
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.credential_pool import AUTH_TYPE_API_KEY, CredentialPool, PooledCredential

LIMITED = "limited1"
FREE = "free0001"


def _entry(credential_id, token, priority):
    return PooledCredential(
        provider="openai-codex", id=credential_id, label=credential_id, auth_type=AUTH_TYPE_API_KEY,
        priority=priority, source="manual", access_token=token,
        base_url="https://chatgpt.com/backend-api/codex",
    )


def _snapshot(used):
    return AccountUsageSnapshot(
        provider="openai-codex", source="usage_api", fetched_at=datetime.now(UTC),
        windows=(AccountUsageWindow(label="Weekly", used_percent=used, limit_window_seconds=604_800),),
    )


@pytest.fixture(autouse=True)
def _cold_cache():
    def clear():
        with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
            credential_pool._USAGE_ADMISSION_CACHE.clear()
            credential_pool._USAGE_ADMISSION_INFLIGHT.clear()
    clear()
    yield
    clear()


@pytest.fixture
def probes(monkeypatch):
    """LIMITED is capped at 80% and reports 95%; every probe is recorded by api_key."""
    class Calls(list):
        used: dict

    calls = Calls()
    used = {"limited-token": 95}
    monkeypatch.setattr(credential_pool, "get_pool_usage_limits", lambda provider: {LIMITED: 80.0})

    def fetch(provider, **kwargs):
        calls.append(kwargs["api_key"])
        return _snapshot(used.get(kwargs["api_key"], 1))

    monkeypatch.setattr(account_usage, "fetch_account_usage", fetch)
    calls.used = used
    return calls


def _only_limited():
    return CredentialPool("openai-codex", [_entry(LIMITED, "limited-token", 0)])


def _with_sibling():
    return CredentialPool("openai-codex", [_entry(LIMITED, "limited-token", 0), _entry(FREE, "free-token", 1)])


# -- the regression: the cap must not be bypassed on a cold cache ---------------------------------------------

def test_select_denies_on_cold_cache(probes):
    """Baseline from the original change: select() probes and refuses."""
    assert _only_limited().select() is None
    assert probes == ["limited-token"]


def test_choose_denies_on_cold_cache(probes):
    assert _only_limited().choose() is None
    assert probes == ["limited-token"]


def test_commit_denies_on_cold_cache(probes):
    assert _only_limited().commit(LIMITED) is None
    assert probes == ["limited-token"]


def test_peek_denies_on_cold_cache(probes):
    assert _only_limited().peek() is None
    assert probes == ["limited-token"]


def test_reclaim_denies_on_cold_cache(probes):
    assert _only_limited().reclaim(LIMITED) is None
    assert probes == ["limited-token"]


@pytest.mark.parametrize("path", ["select", "choose", "peek"])
def test_over_ceiling_account_is_skipped_for_its_sibling_on_every_path(probes, path):
    picked = getattr(_with_sibling(), path)()
    assert picked is not None and picked.id == FREE


# -- per-hook behavior ----------------------------------------------------------------------------------------

def test_choose_still_writes_no_pool_state(probes):
    pool = _with_sibling()
    before = [(e.id, e.request_count, e.priority) for e in pool.entries()]
    assert pool.choose().id == FREE
    assert [(e.id, e.request_count, e.priority) for e in pool.entries()] == before


def test_commit_refuses_entry_that_crossed_its_ceiling_after_choose(probes):
    probes.used["limited-token"] = 10  # under the cap when chosen
    pool = _only_limited()
    assert pool.choose().id == LIMITED
    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        credential_pool._USAGE_ADMISSION_CACHE.clear()  # TTL lapsed before the key was used
    probes.used["limited-token"] = 95
    assert pool.commit(LIMITED) is None
    assert pool.entries()[0].request_count == 0


def test_commit_counts_an_admitted_entry(probes):
    pool = _with_sibling()
    assert pool.commit(FREE).request_count == 1


def test_peek_current_entry_shortcut_does_not_return_an_over_ceiling_entry(probes):
    probes.used["limited-token"] = 10
    pool = _with_sibling()
    assert pool.select().id == LIMITED  # becomes the pool's current entry
    with credential_pool._USAGE_ADMISSION_CACHE_LOCK:
        credential_pool._USAGE_ADMISSION_CACHE.clear()
    probes.used["limited-token"] = 95
    assert pool.peek().id == FREE


def test_reclaim_admits_an_entry_under_its_ceiling(probes):
    probes.used["limited-token"] = 10
    assert _only_limited().reclaim(LIMITED).id == LIMITED


def test_unlimited_account_is_never_probed_or_filtered_on_the_new_paths(monkeypatch):
    """No ceiling configured for the pool's entries: legacy behavior, zero probes."""
    calls = []
    monkeypatch.setattr(credential_pool, "get_pool_usage_limits", lambda provider: {})
    monkeypatch.setattr(account_usage, "fetch_account_usage", lambda *a, **k: calls.append(k) or _snapshot(100))
    pool = _with_sibling()
    assert pool.choose().id == LIMITED
    assert pool.commit(LIMITED).id == LIMITED
    assert pool.peek().id == LIMITED
    assert pool.reclaim(FREE).id == FREE
    assert pool.select().id == LIMITED
    assert calls == []


# -- call sites that consume the pool paths -------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["select", "choose", "peek"])
def test_choose_pool_key_applies_the_ceiling_in_every_mode(probes, mode):
    from hermes_cli.runtime_provider import choose_pool_key

    assert choose_pool_key(_only_limited(), "openai-codex", mode=mode) is None
    picked = choose_pool_key(_with_sibling(), "openai-codex", mode=mode)
    assert picked is not None and picked.entry.id == FREE


def test_pool_choice_commit_refuses_an_entry_over_its_ceiling(probes, caplog):
    from hermes_cli.runtime_provider import PoolChoice, PoolKey

    probes.used["limited-token"] = 95
    pool = _only_limited()
    choice = PoolChoice("gpt-5.6-codex")
    choice.picked = PoolKey(pool, pool.entries()[0], "limited-token")
    choice.commit()
    assert pool.entries()[0].request_count == 0


def test_revert_credential_rotation_does_not_move_back_onto_an_over_ceiling_entry(probes):
    from agent.agent_runtime_helpers import _revert_credential_rotation

    pool = _with_sibling()
    swapped = []
    agent = SimpleNamespace(
        _credential_pool=pool, _credential_pool_revert_id=LIMITED, _credential_pool_entry_id=FREE,
        model="gpt-5.6-codex", _swap_credential=lambda entry: swapped.append(entry.id),
    )
    _revert_credential_rotation(agent)
    assert swapped == []
    assert agent._credential_pool_revert_id == LIMITED  # checked again next turn
