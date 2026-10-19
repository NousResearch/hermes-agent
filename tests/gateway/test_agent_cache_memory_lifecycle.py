"""Memory-provider lifecycle of cached gateway agents: soft eviction (``release_clients``) versus
end-of-session teardown (``shutdown_memory_provider`` / ``close``)."""

import threading

import pytest

from tests.gateway.test_agent_cache import _make_runner


class TestAgentCacheMemoryLifecycle:
    @staticmethod
    def _agent_with_writer_provider(session_id, events):
        """Real AIAgent + real MemoryManager whose provider owns a writer thread released only by shutdown()."""
        import queue
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider
        from run_agent import AIAgent

        class _WriterProvider(MemoryProvider):
            def __init__(self):
                self._q = queue.Queue()
                self.writer = threading.Thread(target=self._run, name="test-mem-writer", daemon=True)
                self.writer.start()

            def _run(self):
                while self._q.get() is not None:
                    pass

            @property
            def name(self):
                return "writerstub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def get_tool_schemas(self):
                return []

            def on_session_end(self, messages):
                events.append("on_session_end")

            def shutdown(self):
                events.append("shutdown")
                self._q.put(None)
                self.writer.join(timeout=5)

        agent = AIAgent(
            model="anthropic/claude-sonnet-4", api_key="test",
            base_url="https://openrouter.ai/api/v1", provider="openrouter",
            max_iterations=5, quiet_mode=True,
            skip_context_files=True, skip_memory=True, session_id=session_id,
        )
        provider = _WriterProvider()
        manager = MemoryManager()
        manager.add_provider(provider)
        agent._memory_manager = manager
        return agent, provider

    def test_release_clients_joins_real_provider_writer_thread(self):
        """Soft eviction must stop a real provider's writer thread (the leak), without extraction."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-writer-evict-test", events)
        assert provider.writer.is_alive()
        agent.release_clients()
        agent.release_clients()  # idempotent: the teardown claim is once-only
        assert not provider.writer.is_alive()
        agent.close()  # a later hard close neither re-extracts nor shuts down twice
        assert events == ["shutdown"]

    def test_release_clients_lets_queued_writes_finish_before_shutdown(self, monkeypatch):
        """Soft release must not cancel the retired agent's queued memory writes: on main the worker
        finished them in the background, and shutdown_all()'s bounded drain alone would abandon them."""
        import agent.memory_manager as mm
        from tests.waiting import wait_until

        monkeypatch.setattr(mm, "_SYNC_DRAIN_TIMEOUT_S", 0.0)
        events = []
        agent, provider = self._agent_with_writer_provider("mem-release-flush-test", events)
        manager, gate, flushing = agent._memory_manager, threading.Event(), threading.Event()
        provider.sync_turn = lambda user, assistant, **kw: (gate.wait(10), events.append(f"sync:{user}"))
        real_flush = manager.flush_pending
        manager.flush_pending = lambda timeout=None: (flushing.set(), real_flush(timeout))[1]
        for turn in ("a", "b", "c"):
            manager.sync_all(turn, "reply")
        releaser = threading.Thread(target=agent.release_clients)
        releaser.start()
        try:
            wait_until(lambda: flushing.is_set() or not releaser.is_alive(), what="the soft release to start")
        finally:
            gate.set()
            releaser.join(timeout=10)
        assert events == ["sync:a", "sync:b", "sync:c", "shutdown"]

    def test_close_racing_an_in_progress_soft_release_does_not_reextract(self):
        """close() arriving while release_clients() is still inside the provider shutdown must not
        start extraction or a second shutdown against the providers being torn down."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-release-race-test", events)
        in_shutdown, let_finish = threading.Event(), threading.Event()
        real_shutdown = provider.shutdown

        def slow_shutdown():
            in_shutdown.set()
            assert let_finish.wait(5)
            real_shutdown()

        provider.shutdown = slow_shutdown
        releaser = threading.Thread(target=agent.release_clients)
        releaser.start()
        try:
            assert in_shutdown.wait(5)
            agent.close()
        finally:
            let_finish.set()
            releaser.join(timeout=5)
        assert events == ["shutdown"]

    def test_gateway_soft_evict_commits_memory_before_provider_shutdown(self):
        """The sweep eviction path must run end-of-session extraction BEFORE release_clients()
        shuts the providers down (a shut-down provider can no longer commit)."""
        events = []
        agent, provider = self._agent_with_writer_provider("mem-commit-order-test", events)
        runner = _make_runner()
        try:
            runner._commit_then_release_soft(agent, "agent:main:telegram:dm:1")
            assert events == ["on_session_end", "shutdown"]
            assert not provider.writer.is_alive()
        finally:
            agent.close()

    @pytest.mark.asyncio
    async def test_eviction_after_timed_out_reset_cleanup_leaves_shutdown_to_it(self, monkeypatch):
        """/new gives up waiting on its cleanup worker after the timeout and then evicts the same agent.
        The worker is still inside on_session_end(), so the eviction must not shut the providers
        down under that extraction, and the providers must be shut down exactly once, after it."""
        import gateway.slash_commands_session as session_cmds
        from tests.waiting import wait_until

        events = []
        agent, provider = self._agent_with_writer_provider("mem-reset-evict-race-test", events)
        in_session_end, let_finish = threading.Event(), threading.Event()

        def slow_session_end(messages):
            events.append("on_session_end:start")
            in_session_end.set()
            assert let_finish.wait(10)
            events.append("on_session_end:done")

        provider.on_session_end = slow_session_end
        agent._session_messages = [{"role": "user", "content": "remember this"}]
        key = "agent:main:telegram:dm:1"
        runner = _make_runner()
        runner._agent_cache[key] = (agent, "sig")
        # Only a deadline for giving up on the worker: on_session_end blocks until let_finish, so
        # /new always times out, however slow the runner.
        monkeypatch.setattr(session_cmds, "_RESET_CLEANUP_TIMEOUT_S", 0.05)
        try:
            await runner._cleanup_old_agent_for_reset(key)
            assert in_session_end.wait(10)
            runner._evict_cached_agent(key)
            wait_until(lambda: agent._session_messages == [], what="the soft eviction to finish")
            assert events == ["on_session_end:start"], "eviction shut providers down mid-extraction"
            assert provider.writer.is_alive()
        finally:
            let_finish.set()
            runner._housekeeping_executor.shutdown(wait=True)
        assert events == ["on_session_end:start", "on_session_end:done", "shutdown"]
        assert not provider.writer.is_alive()

    @pytest.mark.asyncio
    async def test_eviction_while_reset_cleanup_is_still_flushing_leaves_shutdown_to_it(self, monkeypatch):
        """/new's cleanup job can still be in its pre-extraction flush (up to 10 s) when /new stops
        waiting and evicts the agent; that eviction must not shut the providers down first."""
        import gateway.slash_commands_session as session_cmds
        from tests.waiting import wait_until

        events = []
        agent, provider = self._agent_with_writer_provider("mem-reset-flush-test", events)
        agent._session_messages = [{"role": "user", "content": "remember this"}]
        in_flush, let_finish = threading.Event(), threading.Event()

        def slow_flush(timeout=None):
            in_flush.set()
            assert let_finish.wait(10)
            return True

        agent._memory_manager.flush_pending = slow_flush
        key = "agent:main:telegram:dm:1"
        runner = _make_runner()
        runner._agent_cache[key] = (agent, "sig")
        monkeypatch.setattr(session_cmds, "_RESET_CLEANUP_TIMEOUT_S", 0.05)
        try:
            await runner._cleanup_old_agent_for_reset(key)
            assert in_flush.wait(10)
            runner._evict_cached_agent(key)
            wait_until(lambda: agent._session_messages == [], what="the soft eviction to finish")
            assert events == [], "eviction shut providers down before the /new cleanup extracted"
        finally:
            let_finish.set()
            runner._housekeeping_executor.shutdown(wait=True)
        assert events == ["on_session_end", "shutdown"]
        assert not provider.writer.is_alive()
