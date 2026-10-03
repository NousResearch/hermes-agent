"""The idle watcher drains durable jobs without producing an outbound turn."""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity
from gateway.post_reply_idle_worker import GatewayPostReplyIdleWorkerMixin


POLICY = {"compression": {"post_reply_idle": {"channels": [
    {"platform": "signal", "chat_id": "chat", "after_seconds": 300}
]}}}


def _source():
    source = SessionSource(platform=Platform.SIGNAL, chat_id="chat", chat_type="group")
    source._identity = RoutingIdentity("default", "default", Path("."), Path("."), multiplexed=False)
    return source


@pytest.fixture(autouse=True)
def configured_idle_policy(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: POLICY)


class DB:
    def __init__(self, messages):
        self.messages = messages
        self.claimed = False
        self.releases = []

    def claim_due_post_reply_idle(self, holder, **kwargs):
        if self.claimed:
            return None
        self.claimed = True
        return ("sid", 3, 9)

    def get_session(self, sid):
        return {"id": sid, "session_key": "signal:chat", "ended_at": None,
                "system_prompt": "pinned prompt"}

    def get_messages_as_conversation(self, sid, **kwargs):
        return self.messages

    def release_post_reply_idle(self, sid, generation, holder, **kwargs):
        self.releases.append((sid, generation, holder, kwargs))
        return True


class Runner(GatewayPostReplyIdleWorkerMixin):
    def __init__(self, db):
        self._session_db = db
        self.session_store = SimpleNamespace(lookup_by_session_key=lambda key: SimpleNamespace(session_id="sid"))
        self._running = True
        self._running_agents = {}
        self.adapters = {}

    def _is_session_running(self, key):
        return False

    async def _session_has_compression_in_flight(self, key):
        return False


@pytest.mark.asyncio
async def test_disabled_default_does_not_open_session_db(monkeypatch):
    from gateway import run
    calls = []
    class NoDB(Runner):
        @property
        def _session_db(self):
            calls.append(True)
            raise AssertionError("disabled feature must not open SQLite")
        @_session_db.setter
        def _session_db(self, value):
            pass
    runner = NoDB(DB([]))
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens, builds", [(149999, 0), (150000, 1), (150001, 1)])
async def test_minimum_tokens_gates_agent_build_without_cache_eviction(monkeypatch, tokens, builds):
    from gateway import run
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 300}
               for i in range(10)]
    db = DB(history)
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    built, evicted = [], []
    async def build(*args):
        built.append(True)
        return SimpleNamespace(_cached_system_prompt="pinned prompt", context_compressor=SimpleNamespace(
            _compress_window=lambda messages: (len(messages), len(messages)))), db
    runner._hmwa_hygiene_build_agent = build
    runner._evict_cached_agent = evicted.append
    runner._cleanup_agent_resources_off_loop = lambda *args, **kw: asyncio.sleep(0)
    config = {"compression": {"post_reply_idle": {"channels": [
        {"platform": "signal", "chat_id": "chat", "after_seconds": 300, "min_tokens": 150000}
    ]}}}
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr("agent.model_metadata.estimate_messages_tokens_rough", lambda messages: tokens)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert len(built) == builds
    assert evicted == []


@pytest.mark.asyncio
async def test_tiny_due_job_is_claimed_and_delayed_without_model(monkeypatch):
    from gateway import run
    db = DB([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    runner = Runner(db)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    @asynccontextmanager
    async def scope(home):
        yield
    monkeypatch.setattr(run, "_async_profile_runtime_scope", scope)
    await runner._process_due_post_reply_idle()
    assert db.claimed
    assert len(db.releases) == 1
    assert db.releases[0][-1]["retry_at"] > 0


@pytest.mark.asyncio
async def test_scans_each_profile_and_uses_own_database(monkeypatch):
    from gateway import run
    default = DB([{"role": "user", "content": "hi"}])
    secondary = DB([{"role": "user", "content": "hi"}])
    selected = [default]
    class ScopedRunner(Runner):
        @property
        def _session_db(self):
            return selected[0]

        @_session_db.setter
        def _session_db(self, value):
            pass
    runner = ScopedRunner(default)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None), ("other", "/other")]))
    @asynccontextmanager
    async def scope(home):
        selected[0] = secondary
        try:
            yield
        finally:
            selected[0] = default
    monkeypatch.setattr(run, "_async_profile_runtime_scope", scope)
    await runner._process_due_post_reply_idle()
    assert default.claimed and secondary.claimed
    assert len(default.releases) == len(secondary.releases) == 1


@pytest.mark.asyncio
async def test_named_launch_scans_default_and_named_profile_once(monkeypatch, tmp_path):
    from gateway import run
    default = tmp_path / "default"
    named = tmp_path / "named"
    first = DB([{"role": "user", "content": "hi"}])
    second = DB([{"role": "user", "content": "hi"}])
    selected = [second]
    seen = []

    class ScopedRunner(Runner):
        @property
        def _session_db(self):
            seen.append(selected[0])
            return selected[0]
        @_session_db.setter
        def _session_db(self, value):
            pass

    runner = ScopedRunner(second)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._run_in_executor_with_context = lambda fn, *args: asyncio.to_thread(fn, *args)
    entered = []
    monkeypatch.setattr(run, "_multiplex_profile_homes", lambda config: [("default", default), ("named", named)])
    monkeypatch.setattr("hermes_constants.get_routing_process_hermes_home", lambda: named)
    @asynccontextmanager
    async def scope(home):
        entered.append(home)
        selected[0] = first if home == default else second
        try:
            yield
        finally:
            selected[0] = second
    monkeypatch.setattr(run, "_async_profile_runtime_scope", scope)
    await runner._process_due_post_reply_idle()
    assert entered == [default, named]
    assert first.claimed and second.claimed
    assert len(first.releases) == len(second.releases) == 1


@pytest.mark.asyncio
async def test_worker_renews_claim_while_summary_is_slow(monkeypatch):
    from gateway import run
    from gateway import post_reply_idle_worker as idle_module
    db = DB([{"role": "user", "content": "hi"}])
    renewals = []
    db.renew_post_reply_idle = lambda *args, **kwargs: renewals.append(args) or True
    runner = Runner(db)
    async def slow(*args):
        await asyncio.sleep(0.2)
        return 300
    runner._post_reply_idle_run_claim = slow
    monkeypatch.setattr(idle_module, "_LEASE_SECONDS", 0.12)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert renewals


@pytest.mark.asyncio
async def test_stale_route_does_not_build_agent(monkeypatch):
    from gateway import run
    db = DB([{"role": "user", "content": "a" * 2000}] * 10)
    runner = Runner(db)
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda key: SimpleNamespace(session_id="new"))
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert len(db.releases) == 1


@pytest.mark.asyncio
async def test_due_job_runs_detached_with_cas_claim_and_no_send(monkeypatch):
    from gateway import run
    db = DB([{"role": "user" if i % 2 == 0 else "assistant", "content": "long message " * 100}
             for i in range(10)])
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {"provider": "test"})
    bound = []
    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: bound.append(args),
                                             abort_on_summary_failure=False)
        def _compress_context(self, history, system, **kw):
            assert ("evicted", "signal:chat") in bound
            assert self.context_compressor.abort_on_summary_failure is True
            assert self._post_reply_idle_claim[0:2] == (3, 9)
            assert self.compression_in_place and self._end_session_on_close is False
            assert kw["task_id"] == "sid" and kw["commit_fence"]
            self._last_compression_attempt_in_place = True
            return history[:2], "summary"
    agent = Agent()
    async def build(model, runtime, entry):
        return agent, db
    runner._hmwa_hygiene_build_agent = build
    runner._track_deferred_agent_worker = lambda *args: None
    runner._evict_cached_agent = lambda key: bound.append(("evicted", key))
    async def cleanup(*args, **kwargs):
        bound.append(("cleaned", kwargs["session_key"]))
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert bound[0] == (db, "sid")
    assert ("evicted", "signal:chat") in bound
    assert ("cleaned", "signal:chat") in bound
    assert db.releases[0][-1]["retry_at"] > 0


@pytest.mark.asyncio
async def test_slow_due_chat_does_not_block_next_chat(monkeypatch):
    runner = Runner(DB([]))
    started = asyncio.Event()
    release = asyncio.Event()
    scans = []
    async def scan():
        scans.append(True)
        if len(scans) == 1:
            started.set()
            await release.wait()
        else:
            runner._running = False
            release.set()
    runner._process_due_post_reply_idle = scan
    await asyncio.wait_for(runner._post_reply_idle_watcher(interval=0.02), timeout=1)
    assert len(scans) == 2


@pytest.mark.asyncio
async def test_watcher_drains_on_first_tick_and_stops(monkeypatch):
    runner = Runner(DB([]))
    ticks = []
    async def process():
        ticks.append(True)
        runner._running = False
    runner._process_due_post_reply_idle = process
    await runner._post_reply_idle_watcher(interval=20)
    assert ticks == [True]


def test_runner_registers_supervised_idle_watcher():
    from gateway.run import GatewayRunner
    from gateway.run_startup import GatewayStartupMixin
    assert hasattr(GatewayRunner, "_process_due_post_reply_idle")
    assert "_post_reply_idle_watcher" in (GatewayStartupMixin._PRE_RECONNECT_WATCHERS + GatewayStartupMixin._POST_RECONNECT_WATCHERS)


@pytest.mark.asyncio
async def test_restart_due_claim_publishes_only_with_live_generation(tmp_path, monkeypatch):
    import time
    from hermes_state import SessionDB
    from gateway import run
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("sid", source="gateway", session_key="signal:chat", system_prompt="pinned prompt")
    for i in range(10):
        db.append_message("sid", role="user" if i % 2 == 0 else "assistant", content="message " * 150)
    gen = db.invalidate_post_reply_idle("sid")
    assert db.arm_post_reply_idle("sid", "signal:chat", time.time() - 1, expected_generation=gen)
    db.close()
    recovered = SessionDB(path)
    runner = Runner(recovered)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: None)
        def _compress_context(self, history, system, **kw):
            recovered.archive_and_compact("sid", [{"role": "assistant", "content": "summary"}],
                                          idle_claim=self._post_reply_idle_claim)
            self._last_compression_attempt_in_place = True
    async def build(model, runtime, entry):
        return Agent(), recovered
    runner._hmwa_hygiene_build_agent = build
    runner._track_deferred_agent_worker = lambda *args: None
    runner._evict_cached_agent = lambda key: None
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert [m["content"] for m in recovered.get_messages("sid")] == ["summary"]
    assert recovered.claim_due_post_reply_idle("again") is None
    recovered.close()


@pytest.mark.asyncio
async def test_standalone_restored_source_without_transport_pin_compacts(monkeypatch):
    from gateway import run
    db = DB([{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 300}
             for i in range(10)])
    runner = Runner(db)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._primary_profile_name = "default"
    runner._transport_owner = lambda source: None
    runner._restored_source = lambda entry: SessionSource(platform=Platform.SIGNAL, chat_id="chat", chat_type="group")
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    compressed = []
    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: None,
                                             abort_on_summary_failure=False)
        def _compress_context(self, *args, **kwargs):
            compressed.append(True)
            self._last_compression_attempt_in_place = True
    async def build(*args):
        return Agent(), db
    runner._hmwa_hygiene_build_agent = build
    runner._track_deferred_agent_worker = lambda *args: None
    runner._evict_cached_agent = lambda key: None
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert compressed
    assert db.claimed


@pytest.mark.asyncio
async def test_disabled_policy_leaves_existing_deadline_unclaimed(monkeypatch):
    from gateway import run
    db = DB([{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 300}
             for i in range(10)])
    runner = Runner(db)
    runner._restored_source = lambda entry: SimpleNamespace(platform="signal", chat_id="chat")
    async def forbidden(*args):
        raise AssertionError("a disabled channel cannot spend on a persisted job")
    runner._hmwa_hygiene_build_agent = forbidden
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert not db.claimed and db.releases == []


@pytest.mark.asyncio
async def test_no_compressible_window_keeps_cached_agent(monkeypatch):
    from gateway import run
    db = DB([{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 300}
             for i in range(10)])
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(_compress_window=lambda messages: (len(messages), len(messages)),
                                             abort_on_summary_failure=False)
        def _compress_context(self, *args, **kwargs):
            raise AssertionError("nothing compressible")
    runner._hmwa_hygiene_build_agent = lambda *args: asyncio.sleep(0, result=(Agent(), db))
    evicted = []
    runner._evict_cached_agent = evicted.append
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert evicted == []


@pytest.mark.asyncio
async def test_missing_restored_prompt_never_publishes_summary(monkeypatch):
    from gateway import run
    db = DB([{"role": "user" if i % 2 == 0 else "assistant", "content": "long " * 300}
             for i in range(10)])
    db.get_session = lambda sid: {"id": sid, "session_key": "signal:chat", "ended_at": None,
                                  "system_prompt": "original pinned prompt"}
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    class Agent:
        _cached_system_prompt = ""  # transient restore failure in hygiene builder
        context_compressor = SimpleNamespace(abort_on_summary_failure=False)
        def _compress_context(self, *args, **kwargs):
            raise AssertionError("missing prompt must not be published")
    async def build(model, runtime, entry):
        return Agent(), db
    runner._hmwa_hygiene_build_agent = build
    runner._evict_cached_agent = lambda key: None
    cleaned = []
    async def cleanup(agent, **kwargs):
        assert agent._end_session_on_close is False
        cleaned.append(True)
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert cleaned
    assert db.releases[0][-1]["retry_at"] < time.time() + 600


@pytest.mark.asyncio
async def test_timed_out_commit_does_not_block_event_loop(tmp_path, monkeypatch):
    import threading
    import time
    from gateway import run, post_reply_idle_worker as idle_module
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="gateway", session_key="signal:chat", system_prompt="pinned prompt")
    for i in range(10):
        db.append_message("sid", role="user" if i % 2 == 0 else "assistant", content="long " * 300)
    gen = db.invalidate_post_reply_idle("sid")
    db.arm_post_reply_idle("sid", "signal:chat", time.time() - 1, expected_generation=gen)
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    started, release, expired = threading.Event(), threading.Event(), threading.Event()

    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: None,
                                             abort_on_summary_failure=False)
        def _compress_context(self, history, system, **kwargs):
            fence = kwargs["commit_fence"]
            assert fence.begin_commit()
            started.set()
            try:
                if not release.wait(3):
                    expired.set()
            finally:
                fence.finish_commit()

    async def build(*args):
        return Agent(), db
    runner._hmwa_hygiene_build_agent = build
    tracked = []
    runner._track_deferred_agent_worker = lambda future, agent: tracked.append(future)
    runner._evict_cached_agent = lambda key: None
    cleaned = []
    async def cleanup(*args, **kwargs):
        cleaned.append(True)
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(idle_module, "_SUMMARY_TIMEOUT", 0.05)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    task = asyncio.create_task(runner._process_due_post_reply_idle())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.wait_for(task, timeout=4)
        assert not expired.is_set()  # the worker still owns the commit lock
        assert len(tracked) == 2 and not tracked[1].done()
    finally:
        release.set()
        if len(tracked) == 2:
            await asyncio.wait_for(tracked[1], timeout=2)
        assert cleaned
        db.close()


@pytest.mark.asyncio
async def test_buffered_adapter_text_blocks_commit_before_runner_admission(tmp_path, monkeypatch):
    import threading
    from hermes_state import SessionDB
    from gateway import run

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="gateway", session_key="signal:chat", system_prompt="pinned prompt")
    for i in range(10):
        db.append_message("sid", role="user" if i % 2 == 0 else "assistant", content="message " * 150)
    gen = db.invalidate_post_reply_idle("sid")
    db.arm_post_reply_idle("sid", "signal:chat", time.time() - 1, expected_generation=gen)
    original = [m["content"] for m in db.get_messages("sid")]
    runner = Runner(db)
    buffered = {}
    runner.adapters = {Platform.SIGNAL: SimpleNamespace(_pending_text_batches=buffered,
                                                        _active_sessions={}, _pending_messages={})}
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    started, resume = threading.Event(), threading.Event()
    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: None,
                                             abort_on_summary_failure=False)
        def _compress_context(self, history, system, **kwargs):
            started.set()
            assert resume.wait(5)
            db.archive_and_compact("sid", [{"role": "assistant", "content": "stale"}],
                                   idle_claim=self._post_reply_idle_claim)
            self._last_compression_attempt_in_place = True
    async def build(*args):
        return Agent(), db
    runner._hmwa_hygiene_build_agent = build
    runner._track_deferred_agent_worker = lambda *args: None
    runner._evict_cached_agent = lambda key: None
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    task = asyncio.create_task(runner._process_due_post_reply_idle())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        buffered["signal:chat"] = object()  # Weixin has not dispatched the batch yet.
    finally:
        resume.set()
    await task
    assert [m["content"] for m in db.get_messages("sid")] == original
    db.close()


@pytest.mark.asyncio
async def test_new_inbound_during_summary_blocks_stale_commit(tmp_path, monkeypatch):
    import threading
    import time
    from hermes_state import SessionDB
    from gateway import run

    db = SessionDB(tmp_path / "state.db")
    db.create_session("sid", source="gateway", session_key="signal:chat", system_prompt="pinned prompt")
    for i in range(10):
        db.append_message("sid", role="user" if i % 2 == 0 else "assistant", content="message " * 150)
    gen = db.invalidate_post_reply_idle("sid")
    db.arm_post_reply_idle("sid", "signal:chat", time.time() - 1, expected_generation=gen)
    original = [m["content"] for m in db.get_messages("sid")]
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("test-model", {})
    started, resume = threading.Event(), threading.Event()

    class Agent:
        _cached_system_prompt = "pinned prompt"
        context_compressor = SimpleNamespace(bind_session_state=lambda *args: None,
                                             abort_on_summary_failure=False)
        def _compress_context(self, history, system, **kwargs):
            started.set()
            assert resume.wait(5)
            db.archive_and_compact("sid", [{"role": "assistant", "content": "stale"}],
                                   idle_claim=self._post_reply_idle_claim)
            self._last_compression_attempt_in_place = True

    async def build(model, runtime, entry):
        return Agent(), db

    runner._hmwa_hygiene_build_agent = build
    runner._track_deferred_agent_worker = lambda *args: None
    runner._evict_cached_agent = lambda key: None
    async def cleanup(*args, **kwargs):
        pass
    runner._cleanup_agent_resources_off_loop = cleanup
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    task = asyncio.create_task(runner._process_due_post_reply_idle())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        db.invalidate_post_reply_idle("sid")
    finally:
        resume.set()
    await task
    assert [m["content"] for m in db.get_messages("sid")] == original
    db.close()


@pytest.mark.asyncio
async def test_codex_runtime_is_not_rewritten_as_local_mirror(monkeypatch):
    from gateway import run
    db = DB([{"role": "user", "content": "long" * 500}] * 10)
    runner = Runner(db)
    runner._restored_source = lambda entry: _source()
    runner._resolve_session_agent_runtime = lambda **kw: ("codex", {"api_mode": "codex_app_server"})
    async def forbidden(*args):
        raise AssertionError("must not build detached agent for server-side transcript")
    runner._hmwa_hygiene_build_agent = forbidden
    monkeypatch.setattr(run, "_load_gateway_config", lambda: POLICY)
    monkeypatch.setattr(run, "_resolve_handoff_watch_scopes", lambda runner: asyncio.sleep(0, result=[(None, None)]))
    await runner._process_due_post_reply_idle()
    assert len(db.releases) == 1
    assert db.releases[0][-1]["retry_at"] > time.time() + 3000
