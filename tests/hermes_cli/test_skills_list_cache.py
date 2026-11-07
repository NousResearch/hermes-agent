"""Concurrent ``/api/skills`` misses share one scan, and writes invalidate the cache.

A dashboard page load asks for the skill list twice within the same second. Each scan walks
every skill directory, so the pair — serialized by the dashboard's thread pool — measured
2.5-4.7s on a busy gateway against 0.25s cold and 6ms warm.
"""

from __future__ import annotations

import threading
import time

import hermes_cli.web_routers.skills as _skills

WAIT = 10


def _rows(*names: str) -> list[dict]:
    return [{"name": n} for n in names]


def test_concurrent_misses_share_one_scan(monkeypatch):
    calls: list = []
    entered = threading.Event()
    release = threading.Event()

    def _build(profile):
        calls.append(profile)
        entered.set()
        assert release.wait(WAIT), "first scan never released"
        return _rows("alpha")

    monkeypatch.setattr(_skills, "_skills_list_build", _build)
    _skills._invalidate_skills_list_cache()

    results: dict[int, list] = {}

    def _worker(key):
        results[key] = _skills._skills_list_cached(None)

    first = threading.Thread(target=_worker, args=(1,))
    first.start()
    assert entered.wait(WAIT), "first scan never started"
    second = threading.Thread(target=_worker, args=(2,))
    second.start()
    time.sleep(0.2)  # let the second caller reach the lock while the scan is parked
    release.set()
    first.join(WAIT)
    second.join(WAIT)

    assert calls == [None], f"expected one scan for the burst, got {len(calls)}"
    assert results[1] == results[2] == _rows("alpha")


def test_warm_call_skips_the_scan(monkeypatch):
    calls: list = []

    def _build(profile):
        calls.append(profile)
        return _rows("alpha")

    monkeypatch.setattr(_skills, "_skills_list_build", _build)
    _skills._invalidate_skills_list_cache()

    assert _skills._skills_list_cached(None) == _rows("alpha")
    assert _skills._skills_list_cached(None) == _rows("alpha")
    assert calls == [None], f"warm call re-scanned: {calls}"


def test_invalidation_forces_a_fresh_scan(monkeypatch):
    calls: list = []
    version = {"n": 0}

    def _build(profile):
        calls.append(profile)
        version["n"] += 1
        return _rows(f"alpha{version['n']}")

    monkeypatch.setattr(_skills, "_skills_list_build", _build)
    _skills._invalidate_skills_list_cache()

    assert _skills._skills_list_cached(None) == _rows("alpha1")
    _skills._invalidate_skills_list_cache()
    assert _skills._skills_list_cached(None) == _rows("alpha2")
    assert len(calls) == 2


def test_profiles_do_not_share_rows(monkeypatch):
    calls: list = []

    def _build(profile):
        calls.append(profile)
        return _rows(str(profile))

    monkeypatch.setattr(_skills, "_skills_list_build", _build)
    _skills._invalidate_skills_list_cache()

    assert _skills._skills_list_cached("default") == _rows("default")
    assert _skills._skills_list_cached("work") == _rows("work")
    assert calls == ["default", "work"]


def test_callers_cannot_poison_the_cache(monkeypatch):
    monkeypatch.setattr(_skills, "_skills_list_build", lambda profile: _rows("alpha"))
    _skills._invalidate_skills_list_cache()

    poisoned = _skills._skills_list_cached(None)
    poisoned[0]["name"] = "mutated"
    poisoned.append({"name": "extra"})

    assert _skills._skills_list_cached(None) == _rows("alpha")
