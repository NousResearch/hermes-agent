"""A warm budget ends at whichever comes first: its 3 keep-alives, or the owner releasing the agent.

On a 1h route the gateway evicts an idle agent after an hour, long before 3 keep-alives at ~54
minutes each run out, and eviction cancels the warmer: compaction at the cache edge would never fire.
So an eviction (idle TTL, LRU cap) of a chat whose step 2 is due at release compacts it first, through
the same claim-gated cache-expiry job, then releases the agent; nothing is warmed after the release.
"""

import asyncio
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest

import agent.prompt_cache_lifecycle as pcl
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource
from tests.waiting import async_wait_until

HISTORY = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


class _Store:
    def __init__(self, entries):
        self.entries = entries

    def lookup_by_session_key(self, key):
        return self.entries.get(key)

    def load_transcript(self, session_id):
        return list(HISTORY)

    def _named_profile_for_key(self, key):
        return None

    def _db_for_key(self, key):
        return None

    def _lazy(self, name, factory):
        return self.__dict__.setdefault(name, factory())


class _Warmer:
    """The live warmer the release decision reads: a budget still running (keep-alives left)."""

    def __init__(self):
        self.cancelled = []

    def cancel(self, reason):
        self.cancelled.append(reason)


def _runner(monkeypatch, *, due):
    from gateway.run import GatewayRunner

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    now = datetime.now()
    entry = SessionEntry(session_key="k1", session_id="s1", created_at=now, updated_at=now, origin=source,
                         platform=source.platform, chat_type=source.chat_type)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = _Store({"k1": entry})
    runner.session_authority = None
    runner._running = True
    runner._gateway_loop = asyncio.get_running_loop()
    agent = SimpleNamespace(_prompt_cache_warmer=_Warmer(), _last_activity_ts=0.0)
    runner._agent_cache = {"k1": (agent, "sig")}
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache_bounds_cache = SimpleNamespace(idle_ttl_secs=60.0, max_size=None)
    events = []

    def claim(a, **_):
        events.append("claimed")
        return due

    monkeypatch.setattr(pcl, "claim_compaction_at_release", claim)
    monkeypatch.setattr(pcl, "window_still_warm", lambda a, **k: True)

    async def compress(src, ent, history, request, *, trigger=None):
        events.append(("compacted", ent.session_id, trigger))
    runner._run_manual_compression = compress
    runner._commit_then_release_soft = lambda a, key, boundary=False: events.append(("released", key))
    return runner, agent, events


@pytest.mark.asyncio
async def test_an_idle_eviction_compacts_the_due_chat_before_it_releases_the_agent(monkeypatch):
    runner, _agent, events = _runner(monkeypatch, due=True)

    assert runner._sweep_idle_cached_agents() == 1

    await async_wait_until(lambda: ("released", "k1") in events, timeout=5)
    assert events == ["claimed", ("compacted", "s1", "cache_expiry"), ("released", "k1")]
    assert "k1" not in runner._agent_cache


@pytest.mark.asyncio
async def test_an_eviction_that_is_not_due_releases_without_compacting(monkeypatch):
    runner, _agent, events = _runner(monkeypatch, due=False)

    assert runner._sweep_idle_cached_agents() == 1

    await async_wait_until(lambda: ("released", "k1") in events, timeout=5)
    assert events == ["claimed", ("released", "k1")]
