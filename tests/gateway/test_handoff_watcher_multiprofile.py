"""Handoff watcher must poll EVERY served profile's store, not just the root.

Regression guard for the multi-profile ``/handoff`` bug: ``/handoff`` writes
``handoff_state='pending'`` into the store of the profile the CLI ran under
(``hermes -p medicina``), while the gateway's watcher resolves ``_session_db``
from whatever HERMES_HOME is active on its task. Unscoped that is always the
ROOT store, so the pending row was never seen and the CLI timed out with the
gateway plainly alive and connected.

These tests pin the two halves of the fix:
  1. the scope list includes the root (``None``) plus every multiplexed home,
     and degrades to ``[None]`` for a single-profile gateway;
  2. the watcher actually enters ``_profile_runtime_scope`` for each non-root
     home, which is what re-points ``_session_db`` at that profile's store.
"""

import asyncio
import threading
import types
from pathlib import Path

import pytest

from gateway import run


class _FakeConfig:
    def __init__(self, multiplex, allowlist=None):
        self.multiplex_profiles = multiplex
        self.multiplex_profile_allowlist = allowlist


def test_scopes_single_profile_gateway_is_root_only():
    """No multiplexing → exactly the legacy unscoped poll."""
    runner = types.SimpleNamespace(config=_FakeConfig(multiplex=False))
    assert run._handoff_watch_scopes(runner) == [(None, None)]


def test_scopes_include_root_first_then_every_secondary_home(monkeypatch):
    """Multiplexed → root first, then each SECONDARY profile as (name, home).

    The default profile must NOT be yielded again: its home resolves to the
    same ``state.db`` as the unscoped root poll, so repeating it would double
    every tick's query count for zero benefit.
    """
    homes = [
        ("default", Path("/h")),
        ("bala", Path("/h/profiles/bala")),
        ("medicina", Path("/h/profiles/medicina")),
    ]
    monkeypatch.setattr(run, "_multiplex_profile_homes", lambda _cfg: homes)

    runner = types.SimpleNamespace(config=_FakeConfig(multiplex=True))
    scopes = run._handoff_watch_scopes(runner)

    assert scopes[0] == (None, None), "root store must still be polled first"
    assert scopes[1:] == [
        ("bala", Path("/h/profiles/bala")),
        ("medicina", Path("/h/profiles/medicina")),
    ]
    assert not any(name == "default" for name, _h in scopes[1:]), (
        "default profile must not be polled twice per tick"
    )


def test_scopes_degrade_to_root_when_resolution_raises(monkeypatch):
    """A broken profile resolver must not disable the watcher entirely."""
    def _boom(_cfg):
        raise RuntimeError("profiles dir unreadable")

    monkeypatch.setattr(run, "_multiplex_profile_homes", _boom)
    runner = types.SimpleNamespace(config=_FakeConfig(multiplex=True))
    assert run._handoff_watch_scopes(runner) == [(None, None)]


def test_scopes_tolerate_runner_without_config():
    """The watcher's own unit tests bind onto a config-less stand-in."""
    assert run._handoff_watch_scopes(types.SimpleNamespace()) == [(None, None)]


class _RecordingDB:
    """Minimal AsyncSessionDB-shaped stub; records nothing pending."""

    def __init__(self, tag=None):
        self.polls = 0
        self.tag = tag

    async def list_pending_handoffs(self):
        self.polls += 1
        return []


@pytest.mark.asyncio
async def test_watcher_enters_profile_scope_for_each_home(monkeypatch):
    """Each non-root home is polled INSIDE ``_profile_runtime_scope``.

    Entering that scope is the whole point of the fix — it is what redirects
    ``_session_db`` to the profile's own ``state.db``. Asserting on the scope
    entries (not just the poll count) keeps the test mutation-survivable:
    dropping the ``with`` still polls N times but records no scopes.
    """
    scopes = [
        (None, None),
        ("bala", Path("/h/profiles/bala")),
        ("medicina", Path("/h/profiles/medicina")),
    ]
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: scopes)

    entered = []

    class _SpyScope:
        def __init__(self, home):
            self.home = home

        async def __aenter__(self):
            entered.append(self.home)
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(run, "_async_profile_runtime_scope", _SpyScope)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    db = _RecordingDB()
    states = iter([True, False])

    class _Running:
        def __bool__(_self):
            try:
                return next(states)
            except StopIteration:
                return False

    fake = types.SimpleNamespace()
    fake._session_db = db
    fake._running = _Running()

    async def _process_handoff(row, profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    coro = run.GatewayRunner._handoff_watcher(fake, interval=0.0)
    await asyncio.wait_for(coro, timeout=5)

    secondary_homes = [h for _n, h in scopes[1:]]
    # Each secondary home is entered TWICE per watcher run: once by the
    # startup stale-handoff reclaim, once by the poll tick. Both must be
    # scoped — a reclaim outside the scope would clear the ROOT store's rows
    # while reporting the profile's.
    assert entered == secondary_homes * 2, (
        "each secondary home must be scoped for BOTH the startup reclaim "
        f"and the poll tick; got {entered}"
    )
    assert db.polls == 3, "root + both profiles polled once each per tick"


@pytest.mark.asyncio
async def test_slow_profile_secret_load_does_not_block_event_loop(monkeypatch, tmp_path):
    """A slow profile ``.env`` read must not stall unrelated loop work."""
    profile_home = tmp_path / "profiles" / "slow"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(
        run,
        "_handoff_watch_scopes",
        lambda _runner: [(None, None), ("slow", profile_home)],
    )

    from agent import secret_scope

    load_started = threading.Event()
    ticker_progressed = threading.Event()
    ticker_progressed_while_loading = []

    def _slow_build(_home):
        load_started.set()
        ticker_progressed_while_loading.append(
            ticker_progressed.wait(timeout=2)
        )
        return {}

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", _slow_build)

    class _DB:
        async def list_pending_handoffs(self):
            return []

    fake = types.SimpleNamespace(
        _session_db=_DB(),
        _running=False,
    )

    async def _process_handoff(_row, _profile_name=None):
        return None

    fake._process_handoff = _process_handoff

    real_sleep = asyncio.sleep

    async def _skip_initial_delay(seconds):
        await real_sleep(0 if seconds == 5 else seconds)

    monkeypatch.setattr(run.asyncio, "sleep", _skip_initial_delay)
    async def _ticker():
        assert await asyncio.to_thread(load_started.wait, 5)
        ticker_progressed.set()

    watcher = asyncio.create_task(
        run.GatewayRunner._handoff_watcher(fake, interval=0.0)
    )
    ticker = asyncio.create_task(_ticker())
    await asyncio.wait_for(asyncio.gather(watcher, ticker), timeout=5)

    assert ticker_progressed_while_loading == [True], (
        "profile secret loading blocked the asyncio event loop until the "
        "filesystem operation completed"
    )


@pytest.mark.asyncio
async def test_each_scope_resolves_its_own_store_and_profile(monkeypatch):
    """The whole point: a DIFFERENT ``state.db`` per scope, and the profile
    name reaches ``_process_handoff`` so delivery uses that profile's adapter.

    The earlier test proves the ``with`` runs; it cannot prove the store was
    re-resolved, because it pins one fake db for every scope. Here
    ``_session_db`` is a property whose value depends on the active scope, and
    each store yields a pending row tagged with its profile — so a regression
    that polls the root three times, or that drops ``profile_name``, fails.
    """
    scopes = [
        (None, None),
        ("bala", Path("/h/profiles/bala")),
        ("medicina", Path("/h/profiles/medicina")),
    ]
    monkeypatch.setattr(run, "_handoff_watch_scopes", lambda _r: scopes)

    active = {"home": None}

    class _SpyScope:
        def __init__(self, home):
            self.home = home

        async def __aenter__(self):
            active["home"] = self.home
            return self

        async def __aexit__(self, *exc):
            active["home"] = None
            return False

    monkeypatch.setattr(run, "_async_profile_runtime_scope", _SpyScope)

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(run.asyncio, "sleep", _no_sleep)

    class _ScopedDB:
        """Yields a row whose id identifies the store it came from."""

        def __init__(self, tag):
            self.tag = tag

        async def list_pending_handoffs(self):
            return [{"id": f"row-from-{self.tag}"}]

        async def claim_handoff(self, _sid):
            return True

        async def complete_handoff(self, _sid):
            return None

        async def fail_handoff(self, _sid, _err):
            return None

    stores = {
        None: _ScopedDB("root"),
        Path("/h/profiles/bala"): _ScopedDB("bala"),
        Path("/h/profiles/medicina"): _ScopedDB("medicina"),
    }

    processed = []
    states = iter([True, False])

    class _Fake:
        @property
        def _running(self):
            try:
                return next(states)
            except StopIteration:
                return False

        @property
        def _session_db(self):
            # Mirrors the real property: resolves from the ACTIVE scope.
            return stores[active["home"]]

        async def _process_handoff(self, row, profile_name=None):
            processed.append((row["id"], profile_name))

    coro = run.GatewayRunner._handoff_watcher(_Fake(), interval=0.0)
    await asyncio.wait_for(coro, timeout=5)

    assert processed == [
        ("row-from-root", None),
        ("row-from-bala", "bala"),
        ("row-from-medicina", "medicina"),
    ], "each scope must resolve its own store AND pass its profile name through"
