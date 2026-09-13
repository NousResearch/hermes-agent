"""Multiplex invariant: every memory provider's background thread runs under the spawner's profile.

Profile isolation is a ContextVar-scoped HERMES_HOME override; a plain ``threading.Thread`` starts
with an EMPTY context, so a provider's prefetch/sync/writer thread would silently resolve the DEFAULT
profile's home (and fail closed on scoped secrets). Each case drives the provider's real spawn path
with a fake backend and asserts the thread saw the parent's home.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


def _probe_home(seen: dict, key: str = "home"):
    def _record(*_args, **_kwargs):
        seen[key] = get_hermes_home()
    return _record


def _mem0(seen, tmp_path):
    from plugins.memory.mem0 import Mem0MemoryProvider

    p = Mem0MemoryProvider()
    p._backend = MagicMock()
    p._config = {"mode": "platform"}
    p._add = _probe_home(seen)
    p.sync_turn("a long enough user message", "assistant reply")
    return [p._sync_thread]


def _retaindb(seen, tmp_path):
    import plugins.memory.retaindb as retaindb

    p = retaindb.RetainDBMemoryProvider()
    p._client = MagicMock()
    p._context_overlay = lambda query: {"context": seen.setdefault("home", get_hermes_home()) and "ctx"}
    p._client.ask_user.return_value = {"answer": ""}
    p._client.get_agent_model.return_value = {}
    p.queue_prefetch("what do you know")
    return list(p._prefetch_threads)


def _byterover(seen, tmp_path):
    import plugins.memory.byterover as byterover

    p = byterover.ByteRoverMemoryProvider()
    p._curate = _probe_home(seen)
    return [p._curate_in_background("content", name="brv-test", what="test")]


def _supermemory(seen, tmp_path):
    import plugins.memory.supermemory as supermemory

    p = supermemory.SupermemoryMemoryProvider()
    p._active = p._write_enabled = True
    p._client = MagicMock()
    p._client.add_memory = _probe_home(seen)
    p.on_memory_write("add", "user", "a fact")
    return [p._write_thread]


def _openviking(seen, tmp_path):
    import plugins.memory.openviking as openviking

    p = openviking.OpenVikingMemoryProvider()
    workers: set = set()
    p._spawn_tracked("ov-test", _probe_home(seen), threading.Lock(), lambda: workers)
    return list(workers)


def _hindsight(seen, tmp_path):
    import plugins.memory.hindsight as hindsight

    p = hindsight.HindsightMemoryProvider()
    p._recall_sync, p._memory_mode, p._auto_recall, p._prefetch_waits_for_retain = False, "hybrid", True, False
    p._do_recall = lambda query: (seen.setdefault("home", get_hermes_home()) and "text", 1)
    p.queue_prefetch("remember this")
    return [p._prefetch_thread]


def _honcho(seen, tmp_path):
    from plugins.memory.honcho import HonchoMemoryProvider

    return [HonchoMemoryProvider._spawn_write(_probe_home(seen), "honcho-test", "failed %s")]


_PROVIDERS = {
    "mem0": _mem0, "retaindb": _retaindb, "byterover": _byterover, "supermemory": _supermemory,
    "openviking": _openviking, "hindsight": _hindsight, "honcho": _honcho,
}


@pytest.mark.parametrize("name", sorted(_PROVIDERS))
def test_provider_background_thread_sees_spawner_profile_home(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "default"))
    profile_home = tmp_path / "profiles" / "b"
    profile_home.mkdir(parents=True)
    seen: dict = {}
    token = set_hermes_home_override(profile_home)
    try:
        threads = _PROVIDERS[name](seen, tmp_path)
    finally:
        reset_hermes_home_override(token)
    for t in threads:
        if t is not None:
            t.join(timeout=10)
    assert seen.get("home") == profile_home, f"{name}: background thread resolved {seen.get('home')}"
