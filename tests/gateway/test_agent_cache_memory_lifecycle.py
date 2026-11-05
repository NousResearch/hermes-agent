"""Memory-provider lifecycle of cached gateway agents: soft eviction (``release_clients``) versus
end-of-session teardown (``shutdown_memory_provider`` / ``close``)."""

import threading

import pytest

from tests.gateway.test_agent_cache import _make_runner


class _OrderingEngine:
    """Records the context-engine lifecycle order an LCM-style engine depends on.

    ``on_session_end`` may need a live ingest pipeline and anything ingested after ``shutdown()`` is
    lost, so the pair must happen in that order exactly once. Appends into the same ``events`` list
    the provider stub writes, so one list pins how the hard and soft teardown paths interleave.
    """

    name = "lcm"
    compression_count = 0

    def __init__(self, events, label):
        self._events, self._label = events, label

    def on_session_end(self, session_id, messages):
        self._events.append(f"{self._label}:on_session_end")

    def shutdown(self):
        self._events.append(f"{self._label}:shutdown")


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

    @pytest.mark.asyncio
    async def test_session_end_never_follows_engine_shutdown(self, monkeypatch):
        """#147087 review finding 3: the session-end claim must pin on_session_end BEFORE the
        engine's own shutdown() — and never the other way round. Drives the real /new timeout
        path (worker still inside provider on_session_end when /new gives up and the agent is
        soft-evicted). The claim taken at the /new site is what makes the eviction defer the
        engine shutdown, so DELETE it and the eviction shuts the engine down first, the guard then
        skips the notify, and the ordering assertion below fails with events == ['engine:shutdown']
        — the exact defect the reviewer named. MOVING it later (into the cleanup job, i.e. ahead of
        _cleanup_agent_resources) is behaviourally identical whenever the job starts promptly —
        that is why the suite was green under it — so the placement itself is pinned separately
        below, and that pin goes red instead."""
        import gateway.slash_commands_session as session_cmds
        from tests.waiting import wait_until

        # Placement pin: the claim is taken before the hard cleanup is QUEUED, and the job's own
        # resource path never takes it. Read off the function objects production actually runs, so
        # it cannot drift from what the ordering assertion above depends on.
        import inspect

        from gateway.run import GatewayRunner

        reset_src = inspect.getsource(GatewayRunner._cleanup_old_agent_for_reset)
        assert "claim_engine_session_end(" in reset_src, (
            "/new no longer claims engine teardown before queueing its hard cleanup")
        assert reset_src.index("claim_engine_session_end(") < reset_src.index("def _hard_cleanup"), (
            "the session-end claim was moved into the queued cleanup job: a saturated housekeeping "
            "pool then leaves the claim unheld until the job starts, and the queued-but-never-started "
            "case leaks the engine to GC")
        assert "claim_engine_session_end" not in inspect.getsource(
            GatewayRunner._cleanup_agent_resources), (
            "_cleanup_agent_resources took the claim: it runs inside the cancellable job, so a "
            "timeout before the job starts leaves the engine's teardown owned by nobody")

        events = []
        agent, provider = self._agent_with_writer_provider("mem-engine-order-test", events)
        agent.context_compressor = _OrderingEngine(events, "engine")
        in_session_end, let_finish = threading.Event(), threading.Event()

        def slow_session_end(messages):
            in_session_end.set()
            assert let_finish.wait(10)

        provider.on_session_end = slow_session_end
        agent._session_messages = [{"role": "user", "content": "remember this"}]
        key = "agent:main:telegram:dm:1"
        runner = _make_runner()
        runner._agent_cache[key] = (agent, "sig")
        monkeypatch.setattr(session_cmds, "_RESET_CLEANUP_TIMEOUT_S", 0.05)
        try:
            await runner._cleanup_old_agent_for_reset(key)
            assert in_session_end.wait(10)
            runner._evict_cached_agent(key)
            agent.release_clients()  # synchronous soft release (sweep/_retire_agent shape): with
            # the claim at the /new site it must defer; delete/move the claim and THIS is where
            # the eviction half-kills the engine ahead of the still-draining session-end worker.
            assert [e for e in events if e.startswith("engine:")] == [], (
                "soft eviction shut the engine down before the session-end path ran on_session_end")
            let_finish.set()
            wait_until(lambda: getattr(agent, "_context_engine_shutdown", False),
                       what="the session-end worker to notify and shut the engine down")
            assert [e for e in events if e.startswith("engine:")] == [
                "engine:on_session_end", "engine:shutdown"]
        finally:
            let_finish.set()
            runner._housekeeping_executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_new_timeout_before_job_start_returns_claim(self, monkeypatch):
        """#147087 review: a saturated housekeeping pool can time out /new's bounded wait BEFORE
        the queued cleanup job ever starts — the session-end claim must be returned so a later
        soft release can finish the teardown instead of deferring forever to a job that will
        never run. Without the return, the eviction below defers (claim 'session_end'), the
        engine never shuts down here, and the fd waits for garbage collection."""
        import asyncio

        import gateway.slash_commands_session as session_cmds

        events = []
        agent, _provider = self._agent_with_writer_provider("mem-claim-return-test", events)
        agent.context_compressor = _OrderingEngine(events, "engine")
        key = "agent:main:telegram:dm:1"
        runner = _make_runner()
        runner._agent_cache[key] = (agent, "sig")
        monkeypatch.setattr(session_cmds, "_RESET_CLEANUP_TIMEOUT_S", 0.05)

        def _queued_forever(fn):  # the wrapped callable is never invoked: item queued, then cancelled
            return asyncio.get_running_loop().create_future()

        monkeypatch.setattr(runner, "_run_housekeeping_in_executor", _queued_forever)
        try:
            await runner._cleanup_old_agent_for_reset(key)
            assert getattr(agent, "_context_engine_teardown_claim", None) is None, (
                "claim stayed held although the cleanup job never started")
            agent.release_clients()  # the soft-release entry evictions/_retire_agent share
            # round-2 item 1: taking over the teardown also means taking over the delivery.
            assert [e for e in events if e.startswith("engine:")] == [
                "engine:on_session_end", "engine:shutdown"], (
                "soft release did not take over the teardown after the returned claim")
            # And the hard teardown that follows (gateway shutdown tail shape: close() on the same
            # agent object) must NOT re-enter the engine: with the soft release owning the teardown,
            # on_session_end after shutdown() is exactly the finding-3 half-kill.
            agent.close()
            assert [e for e in events if e.startswith("engine:")] == [
                "engine:on_session_end", "engine:shutdown"], (
                "hard cleanup called into the engine AFTER the soft release had shut it down")
        finally:
            ex = getattr(runner, "_housekeeping_executor", None)  # bypassed above: may never exist
            if ex is not None:
                ex.shutdown(wait=True)

    def test_direct_compress_paths_hold_engine_request(self, monkeypatch):
        # Review round 2 (#147087): the Codex /compress handler and the gateway hygiene
        # compaction drive the resident's engine through _compress_context, outside the turn
        # facade. While such a compression runs the agent counts as an in-flight engine user,
        # and a soft eviction landing mid-compression is deferred until the compression ends.
        # The choke point is on _compress_context itself; the shape leg pins that both gateway
        # sites still route through it.
        import inspect
        from agent.conversation_compression import CompressionCommitFence

        events = []
        agent, _provider = self._agent_with_writer_provider("mem-compress-req-test", events)
        agent.context_compressor = _OrderingEngine(events, "engine")
        agent._session_messages = []
        seen = {}

        def _fake_compress(compressor_agent, msgs, sysmsg, **kwargs):
            seen["inflight"] = int(getattr(compressor_agent, "_engine_inflight", 0))
            compressor_agent.release_clients()  # eviction shape landing mid-compression
            seen["events_mid"] = list(events)
            return ([], "")

        monkeypatch.setattr("agent.conversation_compression.compress_context", _fake_compress)
        try:
            agent._compress_context([], "", force=True, commit_fence=CompressionCommitFence())
            assert seen["inflight"] >= 1, "compression ran with no in-flight engine count"
            assert "engine:shutdown" not in seen["events_mid"], (
                "soft eviction shut the engine down while the compression was still running")
            assert [e for e in events if e.startswith("engine:")] == [
                "engine:on_session_end", "engine:shutdown"], (
                "the compression's own end must play out the deferred delivery+shutdown once")
        finally:
            agent.close()

        import agent.compression_facade as facade
        import gateway.run as grun
        from gateway.slash_commands_session import GatewaySessionCommandsMixin
        assert hasattr(facade.CompressionFacadeMixin._compress_context, "__wrapped__"), (
            "the choke point was unwrapped: every direct executor caller counts again")
        src_codex = inspect.getsource(GatewaySessionCommandsMixin._compress_codex_app_server_session)
        src_hygiene = inspect.getsource(grun.run_codex_hygiene_compaction)
        assert "_compress_context" in src_codex and "_compress_context" in src_hygiene, (
            "a gateway engine driver bypasses the wrapped choke point")

