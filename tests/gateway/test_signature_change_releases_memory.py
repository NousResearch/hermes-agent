"""A cache signature miss must retire the displaced agent's memory workers.

No LLM, credentials, disk session or external provider is used. The test provider
models a bound queue-writer; MemoryManager and its mem-sync executor, plus the
gateway lookup/build/release orchestration, are production code.
"""
import queue
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from gateway.session_execution import GatewaySessionAgentMixin


@pytest.fixture(autouse=True)
def no_source_updater(monkeypatch):
    # Cache and lifecycle tests must not probe or trigger background updater
    # recovery, venv synchronization, or environment activation.
    from hermes_cli import _early_recovery, venv_sync
    from pm import environments
    monkeypatch.setattr(_early_recovery, "restore_interrupted_pull", lambda *a: False)
    monkeypatch.setattr(_early_recovery, "recover_if_needed", lambda *a: None)
    monkeypatch.setattr(venv_sync, "prepare_launch", lambda *a: None)
    monkeypatch.setattr(environments, "activate_dependencies", lambda *a: None)


class QueueWriter(MemoryProvider):
    @property
    def name(self):
        return "builtin"  # no external provider discovery or configuration

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def __init__(self):
        self.jobs = queue.Queue()
        self.writer = threading.Thread(target=self._write, daemon=True, name="test-queue-writer")
        self.writer.start()
        self.stopped = False

    def _write(self):
        while self.jobs.get() is not None:
            pass

    def shutdown(self):
        if not self.stopped:
            self.stopped = True
            self.jobs.put(None)
            self.writer.join(timeout=2)


class ResourceAgent:
    def __init__(self):
        self._memory_manager = MemoryManager()
        self.provider = QueueWriter()
        self._memory_manager.add_provider(self.provider)
        # Real production lazy executor, with a completed task so it is idle.
        done = threading.Event()
        self._memory_manager._submit_background(done.set, kind="write")
        assert done.wait(2)
        self.executor = self._memory_manager._sync_executor
        self.workers = list(self.executor._threads)
        assert len(self.workers) == 1
        assert self.workers[0].is_alive()
        self.releases = 0

    def release_clients(self):
        self.releases += 1
        self._memory_manager.release_soft()

    def cleanup(self):
        self._memory_manager.release_soft()
        for worker in self.workers:
            worker.join(timeout=2)
        self.provider.shutdown()


def execution(old, signature="old"):
    cache = OrderedDict(route=(old, signature, 0, "s"))
    lock = threading.Lock()
    runner = SimpleNamespace(
        _agent_cache=cache, _agent_cache_lock=lock,
        _agent_config_signature=lambda *a, **kw: "new",
        _extract_cache_busting_config=lambda cfg: {},
        _refresh_fallback_model=lambda: None,
        _apply_fallback_chain_to_agent=lambda *a: None,
        _init_cached_agent_for_turn=lambda *a: None,
        _enforce_agent_cache_cap=lambda: None,
    )
    releases = []

    def spawn(target, args, name, **kw):
        # Synchronous deterministic transport, but production must call it OFF lock.
        assert lock.acquire(blocking=False), "release called while cache lock held"
        lock.release()
        releases.append(args[0])
        target(*args)

    runner._release_evicted_agent_soft = lambda agent: agent.release_clients()
    runner._spawn_release_thread = spawn
    turn = GatewaySessionAgentMixin()
    turn._runner = runner
    turn._ctx = SimpleNamespace(session_key="route", session_id="s", source=SimpleNamespace(),
                                user_config={}, enabled_toolsets=[], _interrupt_depth=0,
                                interim_assistant_messages_enabled=True)
    turn._skip_context_files = lambda key: False
    turn._cached_sid_is_dead = lambda *a: ("s", False)
    turn._current_message_count = lambda: 0
    turn._build_fresh_agent = Mock(return_value=SimpleNamespace())
    return turn, runner, releases


@pytest.mark.parametrize("changes", [1, 5])
def test_signature_rebuild_releases_displaced_workers(monkeypatch, changes):
    monkeypatch.setattr("gateway.session_execution.api_runtime_locked", lambda: None)
    agents = [ResourceAgent() for _ in range(changes)]
    try:
        for old in agents:
            turn, runner, releases = execution(old)
            fresh, reused = turn._resolve_turn_agent(
                {"model": "new-model", "runtime": {}}, "telegram", "", 20, None, {})
            assert not reused
            assert runner._agent_cache["route"][0] is fresh
            assert releases == [old], "signature miss overwrote the old agent without soft release"
            assert old.releases == 1
            for worker in old.workers:
                worker.join(timeout=2)
            assert not old.provider.writer.is_alive()
            assert not any(worker.is_alive() for worker in old.workers)
    finally:
        for agent in agents:
            agent.cleanup()


def test_matching_signature_reuses_without_tearing_down_workers():
    old = ResourceAgent()
    try:
        turn, runner, releases = execution(old, signature="new")
        found = turn._lookup_cached_agent("new", runner._agent_cache_lock, runner._agent_cache,
                                        20, "s", False, 0)
        assert found.agent is old and found.reused and found.evicted is None
        assert releases == [] and old.releases == 0
        assert old.provider.writer.is_alive()
        assert all(worker.is_alive() for worker in old.workers)
    finally:
        old.cleanup()


def test_pending_sentinel_is_not_released_on_signature_change():
    from gateway.run import _AGENT_PENDING_SENTINEL
    turn, runner, releases = execution(_AGENT_PENDING_SENTINEL)
    found = turn._lookup_cached_agent("new", runner._agent_cache_lock, runner._agent_cache,
                                    20, "s", False, 0)
    assert not found.reused and found.evicted is None
    assert releases == []


def test_fresh_build_failure_still_releases_old_workers(monkeypatch):
    monkeypatch.setattr("gateway.session_execution.api_runtime_locked", lambda: None)
    old = ResourceAgent()
    try:
        turn, runner, releases = execution(old)
        turn._build_fresh_agent.side_effect = RuntimeError("build failed")
        with pytest.raises(RuntimeError, match="build failed"):
            turn._resolve_turn_agent({"model": "new", "runtime": {}}, "telegram", "", 20, None, {})
        assert releases == [old]
        assert "route" not in runner._agent_cache
        assert not old.provider.writer.is_alive()
    finally:
        old.cleanup()


def test_signature_miss_uses_real_agent_release_path(monkeypatch):
    from types import MethodType
    from gateway.run_agent_cache import GatewayAgentCacheMixin
    from tests.gateway.test_agent_cache_memory_lifecycle import TestAgentCacheMemoryLifecycle

    monkeypatch.setattr("gateway.session_execution.api_runtime_locked", lambda: None)
    events = []
    old, provider = TestAgentCacheMemoryLifecycle._agent_with_writer_provider("signature-change", events)
    manager = old._memory_manager
    done = threading.Event()
    manager._submit_background(done.set, kind="write")
    assert done.wait(2)
    workers = list(manager._sync_executor._threads)
    try:
        turn, runner, releases = execution(old)
        runner._release_evicted_agent_soft = MethodType(GatewayAgentCacheMixin._release_evicted_agent_soft, runner)
        fresh, reused = turn._resolve_turn_agent(
            {"model": "new", "runtime": {}}, "telegram", "", 20, None, {})
        assert not reused and runner._agent_cache["route"][0] is fresh
        assert releases == [old]
        assert not provider.writer.is_alive()
        for worker in workers:
            worker.join(timeout=2)
        assert not any(worker.is_alive() for worker in workers)
        assert events == ["shutdown"]
    finally:
        old.close()
        for worker in workers:
            worker.join(timeout=2)
