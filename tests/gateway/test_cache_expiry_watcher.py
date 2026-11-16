"""Gateway owner of cache-expiry compaction (``run_watchers``): each due idle chat is compacted in
its own background job under the chat's profile scope, holding the session authority's claim gate
so no turn starts on the history being replaced, and a detached Desktop conversation (cached agent,
no routing entry of its own) is resolved through its live conversation."""

import asyncio
import contextlib
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionEntry, SessionSource
from gateway.config import Platform
from gateway.session_authority import LiveSession
from tests.waiting import async_wait_until

HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


def _source(chat_id="c1", detached_route=None):
    return SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, user_id="u1", chat_type="dm",
                         detached_route=detached_route)


def _entry(key, session_id, source):
    now = datetime.now()
    return SessionEntry(session_key=key, session_id=session_id, created_at=now, updated_at=now,
                        origin=source, platform=source.platform, chat_type=source.chat_type)


class _Store:
    """The SessionStore surface the watcher and ``route_entry`` read."""

    def __init__(self, entries):
        self.entries = entries

    def lookup_by_session_key(self, key):
        return self.entries.get(key)

    def load_transcript(self, session_id):
        return list(HISTORY)

    def _named_profile_for_key(self, key):
        return None  # standalone gateway: every key is the launch profile's

    def _db_for_key(self, key):
        return None

    def _lazy(self, name, factory):
        return self.__dict__.setdefault(name, factory())


class _Authority:
    def __init__(self, runner):
        self.runner, self.sessions, self.profile_id = runner, {}, "default"

    def logical_owner(self, session_id):
        return session_id


@pytest.fixture
def make_runner(monkeypatch):
    import agent.prompt_cache_lifecycle as decision
    from gateway.run import GatewayRunner

    monkeypatch.setattr(decision, "claim_compaction", lambda agent, **k: True)
    runners = []

    def make(entries, keys, *, authority=False):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.session_store = _Store(entries)
        runner.session_authority = _Authority(runner) if authority else None
        runner._agent_cache = {key: (SimpleNamespace(session_id=entries[key].session_id if key in entries
                                                     else key.rsplit(":", 1)[-1]), "sig") for key in keys}
        runner._agent_cache_lock = threading.Lock()
        runner.compacted = []
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        for task in list(getattr(runner, "_background_tasks", ())):
            task.cancel()


async def _compact_and_settle(runner):
    """One scan, then every compaction job it started (they are background tasks)."""
    await runner._compact_idle_sessions_before_cache_expiry()
    await asyncio.gather(*getattr(runner, "_background_tasks", ()), return_exceptions=True)


@pytest.mark.asyncio
async def test_standalone_gateway_compacts_inside_the_launch_profile_scope(make_runner):
    """``multiplex_profiles`` off still binds the launch scope: once a hosted room flips the
    credential guard, an unscoped summary would fail closed on every credential read."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"])
    scope = []

    @contextlib.contextmanager
    def launch_scope():
        scope.append("entered")
        yield
        scope.append("exited")

    runner._standalone_launch_scope = launch_scope

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append((entry.session_id, trigger, list(scope)))
    runner._run_manual_compression = compress

    await _compact_and_settle(runner)

    assert runner.compacted == [("s1", "cache_expiry", ["entered"])]
    assert scope == ["entered", "exited"]


@pytest.mark.asyncio
async def test_a_turn_claim_waits_for_the_compaction_holding_the_claim_gate(make_runner):
    """A message arriving mid-summary must not start a reply on the old history while the summary
    replaces it: the authority's FIFO claim takes ``claim_gate``, which the compaction holds."""
    runner = make_runner({"k1": _entry("k1", "s1", _source())}, ["k1"], authority=True)
    live = runner.session_authority.sessions["s1"] = LiveSession(_source(), "k1")
    release, order = asyncio.Event(), []

    async def compress(source, entry, history, request, *, trigger=None):
        order.append("summary started")
        await release.wait()
        order.append("summary committed")
    runner._run_manual_compression = compress

    scan = asyncio.create_task(runner._compact_idle_sessions_before_cache_expiry())
    await async_wait_until(lambda: order == ["summary started"], timeout=5)

    async def claim():
        async with live.claim_gate:
            order.append("turn claimed")
    claimer = asyncio.create_task(claim())
    await asyncio.sleep(0)
    assert order == ["summary started"]  # the claim is parked behind the compaction
    release.set()
    await asyncio.wait_for(claimer, timeout=5)
    assert order == ["summary started", "summary committed", "turn claimed"]
    await asyncio.wait_for(scan, timeout=5)


@pytest.mark.asyncio
async def test_a_slow_summary_neither_delays_another_chat_nor_holds_up_housekeeping(make_runner):
    """A summary may run for minutes: the next due chat starts at once (it would otherwise go cold
    before its turn) and ``_session_housekeeping`` returns to its other sweeps."""
    entries = {"slow": _entry("slow", "s-slow", _source("c1")), "fast": _entry("fast", "s-fast", _source("c2"))}
    runner = make_runner(entries, ["slow", "fast"])
    runner._sweep_idle_cached_agents = lambda: 0
    runner._sweep_agent_cache_under_pressure = lambda: None
    runner._last_session_store_prune_ts = float("inf")
    never = asyncio.Event()

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append(entry.session_id)
        if entry.session_id == "s-slow":
            await never.wait()
    runner._run_manual_compression = compress

    await asyncio.wait_for(runner._session_housekeeping(), timeout=5)
    await async_wait_until(lambda: sorted(runner.compacted) == ["s-fast", "s-slow"], timeout=5)
    # The slow chat's job is still in flight: the next pass never compacts it a second time.
    await asyncio.wait_for(runner._compact_idle_sessions_before_cache_expiry(), timeout=5)
    await async_wait_until(lambda: runner.compacted.count("s-fast") == 2, timeout=5)
    assert runner.compacted.count("s-slow") == 1
    never.set()


@pytest.mark.asyncio
async def test_a_detached_desktop_conversation_without_a_routing_entry_is_compacted(make_runner):
    """A detached conversation has a cached agent but no store entry: its live conversation names
    the entry, so the window the scan already claimed is not wasted."""
    key = "agent:main:detached:s-det"
    runner = make_runner({}, [key], authority=True)
    runner.session_authority.sessions["s-det"] = LiveSession(_source(detached_route=key), key)

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append((entry.session_key, entry.session_id))
    runner._run_manual_compression = compress

    await _compact_and_settle(runner)

    assert runner.compacted == [(key, "s-det")]


@pytest.mark.asyncio
async def test_a_job_queued_behind_a_summary_skips_a_key_that_new_rerouted(make_runner):
    """The claim and the warm window belong to the chat the job was queued for: when ``/new``
    re-routes the key while the job waits for a slot, the new chat is not compacted."""
    runner = make_runner({"k1": _entry("k1", "s-old", _source())}, ["k1"])

    async def compress(source, entry, history, request, *, trigger=None):
        runner.compacted.append(entry.session_id)
    runner._run_manual_compression = compress
    runner.session_store.entries["k1"] = _entry("k1", "s-new", _source())  # /new before the slot

    await _compact_and_settle(runner)

    assert runner.compacted == []
