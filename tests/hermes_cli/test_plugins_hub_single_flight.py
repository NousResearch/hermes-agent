"""Concurrent plugins-hub misses share one rebuild and stay off the default thread pool (#147669)."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import web_server
import hermes_cli.web_routers.dashboard_ui as _rt_dashboard_ui
import hermes_cli.web_server_dashboard as _dashboard
import hermes_cli.config as _cfg_mod

WAIT = 10


class _Rebuilds:
    """Stands in for the rebuild body: counts calls and parks each one until released."""

    def __init__(self):
        self.calls = 0
        self.entered = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]
        self.fail_first = False

    def __call__(self):
        index = self.calls
        self.calls += 1
        self.entered[index].set()
        assert self.release[index].wait(WAIT)
        if self.fail_first and index == 0:
            raise RuntimeError("rebuild failed")
        return {"rebuild": index}


class _CountedFlight(Future):
    """Counts the callers that committed to waiting on a rebuild."""

    joined = 0

    def result(self, timeout=None):
        type(self).joined += 1
        return super().result(timeout)


def _wait_for_joined(count):
    deadline = time.monotonic() + WAIT
    while _CountedFlight.joined < count:
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.fixture
def rebuilds(monkeypatch):
    fake = _Rebuilds()
    monkeypatch.setattr(_CountedFlight, "joined", 0)
    monkeypatch.setattr(_dashboard, "Future", _CountedFlight)
    monkeypatch.setattr(_cfg_mod, "get_hermes_home", lambda: Path("/tmp/hermes-home"))
    monkeypatch.setattr(_dashboard, "_rebuild_plugins_hub", fake)
    _dashboard._invalidate_plugins_hub_cache()
    yield fake
    for event in fake.release:
        event.set()
    _dashboard._invalidate_plugins_hub_cache()


def _call_in_threads(count, force_refresh=False):
    results: list = []

    def _call():
        try:
            results.append(_dashboard._merged_plugins_hub(force_refresh=force_refresh))
        except Exception as exc:
            results.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(count)]
    for thread in threads:
        thread.start()
    return threads, results


def _join(threads):
    for thread in threads:
        thread.join(WAIT)
        assert not thread.is_alive()


def test_concurrent_misses_share_one_rebuild(rebuilds):
    leader, results = _call_in_threads(1)
    assert rebuilds.entered[0].wait(WAIT)
    followers, follower_results = _call_in_threads(3)
    _wait_for_joined(3)

    rebuilds.release[0].set()
    _join(leader + followers)

    assert rebuilds.calls == 1
    assert results + follower_results == [{"rebuild": 0}] * 4


def test_invalidation_during_a_rebuild_starts_a_new_one_and_keeps_the_old_out_of_the_cache(rebuilds):
    stale, stale_results = _call_in_threads(1)
    assert rebuilds.entered[0].wait(WAIT)

    _dashboard._invalidate_plugins_hub_cache()
    fresh, fresh_results = _call_in_threads(1)
    assert rebuilds.entered[1].wait(WAIT)
    rebuilds.release[1].set()
    _join(fresh)
    rebuilds.release[0].set()
    _join(stale)

    assert stale_results == [{"rebuild": 0}]
    assert fresh_results == [{"rebuild": 1}]
    assert _dashboard._merged_plugins_hub() == {"rebuild": 1}


def test_force_refresh_does_not_join_a_running_rebuild(rebuilds):
    older, older_results = _call_in_threads(1)
    assert rebuilds.entered[0].wait(WAIT)

    forced, forced_results = _call_in_threads(1, force_refresh=True)
    assert rebuilds.entered[1].wait(WAIT)
    rebuilds.release[1].set()
    _join(forced)
    rebuilds.release[0].set()
    _join(older)

    assert forced_results == [{"rebuild": 1}]
    assert older_results == [{"rebuild": 0}]
    assert _dashboard._merged_plugins_hub() == {"rebuild": 1}


def test_failed_rebuild_reaches_joined_callers_and_the_next_call_retries(rebuilds):
    rebuilds.fail_first = True
    leader, results = _call_in_threads(1)
    assert rebuilds.entered[0].wait(WAIT)
    followers, follower_results = _call_in_threads(2)
    _wait_for_joined(2)

    rebuilds.release[0].set()
    _join(leader + followers)

    assert all(isinstance(result, RuntimeError) for result in results + follower_results)
    assert rebuilds.calls == 1

    rebuilds.release[1].set()
    assert _dashboard._merged_plugins_hub() == {"rebuild": 1}


def test_hub_requests_leave_the_default_pool_free(rebuilds, monkeypatch):
    monkeypatch.setattr(web_server, "_require_token", lambda _request: None)

    async def scenario():
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(pool)
        hubs = [asyncio.create_task(_rt_dashboard_ui.get_plugins_hub(object())) for _ in range(8)]
        await asyncio.to_thread(rebuilds.entered[0].wait, WAIT)
        # With hub work on the default pool, this one-worker pool stays taken until the release below.
        await asyncio.wait_for(asyncio.to_thread(lambda: None), WAIT / 2)
        rebuilds.release[0].set()
        return await asyncio.gather(*hubs)

    assert asyncio.run(scenario()) == [{"rebuild": 0}] * 8
    assert rebuilds.calls == 1
