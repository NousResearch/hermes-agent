"""PLG-11 (#140058): the in-repo memory providers meet the multiplexing contract.

On a multiplexed gateway every profile's agents share one process. Each agent's manager owns its own
provider instance, and the contract (``agent/memory_provider.py``) is that the instance binds its
store from the ``hermes_home`` ``initialize`` receives, keeps that binding on the instance, and
never rebinds a module/class-level one per call. These tests drive the real providers through real
managers with two profiles interleaving on threads, with only the network/CLI backend faked.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agent.memory_manager import MemoryManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    alpha, beta = root / "profiles" / "alpha", root / "profiles" / "beta"
    for home in (alpha, beta):
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))  # the launch profile: neither alpha nor beta
    return alpha, beta


class _Scope:
    def __init__(self, home):
        self.home = home

    def __enter__(self):
        self.token = set_hermes_home_override(str(self.home))

    def __exit__(self, *exc):
        reset_hermes_home_override(self.token)


# -- ByteRover ---------------------------------------------------------------------------------------

def test_byterover_binds_the_home_initialize_receives(homes):
    from plugins.memory.byterover import ByteRoverMemoryProvider
    alpha, _ = homes
    provider = ByteRoverMemoryProvider({})
    provider.initialize("s", hermes_home=str(alpha))  # no context override: the ambient home is the root
    assert Path(provider._cwd) == alpha / "byterover"


def test_byterover_two_profiles_interleaving_stay_in_their_own_trees(homes, monkeypatch):
    from plugins.memory import byterover
    calls: list = []
    lock = threading.Lock()

    def fake_run_brv(args, timeout=0, cwd=None):
        with lock:
            calls.append((args[0], args[-1], cwd))
        return {"success": True, "output": "x" * 40}

    monkeypatch.setattr(byterover, "_run_brv", fake_run_brv)
    monkeypatch.setattr(byterover, "_resolve_brv_path", lambda: "/bin/true")
    managers = {}
    for home in homes:
        manager = MemoryManager()
        with _Scope(home):
            manager.add_provider(byterover.ByteRoverMemoryProvider({}))
            manager.initialize_all(session_id=f"s-{home.name}", platform="api_server", hermes_home=str(home))
        managers[home] = manager
    _run_interleaved(managers, "brv_curate", {"content": None})
    wait_until(lambda: len(calls) >= 2 * 3 * 5, what="every brv call from both profiles")
    for home in homes:
        mine = [c for c in calls if c[1].endswith(f"{home.name}-content") or f"{home.name}-" in c[1]]
        assert mine and all(Path(cwd) == home / "byterover" for _, _, cwd in mine), mine


# -- RetainDB ----------------------------------------------------------------------------------------

class _FakeClient:
    calls: list = []
    lock = threading.Lock()

    def __init__(self, api_key, base_url, project):
        self.project = project

    def _record(self, what, content):
        with self.lock:
            type(self).calls.append((what, self.project, content))

    def add_memory(self, user_id, session_id, content, **kwargs):
        self._record("add_memory", content)
        return {"ok": True}

    def ingest_session(self, user_id, session_id, messages, timeout=15.0):
        self._record("ingest_session", messages[0]["content"])
        return {"ok": True}


@pytest.fixture
def fake_retaindb(monkeypatch):
    from plugins.memory import retaindb
    _FakeClient.calls = []
    monkeypatch.setattr(retaindb, "_Client", _FakeClient)
    monkeypatch.setattr(retaindb, "get_secret", lambda name, default=None: "rdb-key" if name == "RETAINDB_API_KEY" else default)
    monkeypatch.setattr(retaindb, "_load_retaindb_config", dict)
    return retaindb


def test_retaindb_binds_the_home_initialize_receives(homes, fake_retaindb):
    alpha, _ = homes
    provider = fake_retaindb.RetainDBMemoryProvider()
    provider.initialize("s", hermes_home=str(alpha))  # no context override: the ambient home is the root
    try:
        assert provider._queue._db_path == alpha / "retaindb_queue.db"
        assert provider._client.project == "hermes-alpha"
    finally:
        provider.shutdown()


def test_retaindb_two_profiles_interleaving_stay_in_their_own_projects(homes, fake_retaindb):
    managers = {}
    for home in homes:
        manager = MemoryManager()
        with _Scope(home):
            manager.add_provider(fake_retaindb.RetainDBMemoryProvider())
            manager.initialize_all(session_id=f"s-{home.name}", platform="api_server", hermes_home=str(home))
        managers[home] = manager
    try:
        _run_interleaved(managers, "retaindb_remember", {"content": None})
        wait_until(lambda: len(_FakeClient.calls) >= 2 * 3 * 5, what="every RetainDB write from both profiles")
    finally:
        for manager in managers.values():
            manager.shutdown_all()
    for home in homes:
        mine = [c for c in _FakeClient.calls if f"{home.name}-" in c[2]]
        assert {c[0] for c in mine} == {"add_memory", "ingest_session"}, mine
        assert all(project == f"hermes-{home.name}" for _, project, _ in mine), mine
        assert (home / "retaindb_queue.db").exists()


# -- shared driver -----------------------------------------------------------------------------------

def _run_interleaved(managers, tool, args_template, rounds=5):
    """Each profile's turn, on its own thread and scope: a tool call, a turn sync and a built-in
    memory-write mirror per round, all carrying that profile's name in their content."""
    barrier, errors = threading.Barrier(len(managers)), []

    def turns(home, manager):
        try:
            with _Scope(home):
                barrier.wait(timeout=5)
                for i in range(rounds):
                    args = {k: (v if v is not None else f"{home.name}-tool-{i}") for k, v in args_template.items()}
                    result = json.loads(manager.handle_tool_call(tool, args))
                    assert "error" not in result, result
                    manager.sync_all(f"{home.name}-turn-{i} what did we decide", "the plan", session_id="s")
                    manager.on_memory_write("add", "memory", f"{home.name}-write-{i}")
                assert manager.flush_pending(timeout=10)
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=turns, args=item) for item in managers.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors, errors
