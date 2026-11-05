"""The gateway binds a plugin context engine as main did: ``session_db`` is the deprecated ``hermes_state.SessionDB``
on the session's profile store and no other keyword, on a cold ``/compress`` and on session hygiene's rebind; the
``SessionDB`` closes when the compress ends or the hygiene agent closes."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_state
from agent.context_engine import ContextEngine
from gateway.run import GatewayRunner
from persistence_assembly import open_persistence, profile_scope
from run_agent import AIAgent
from tests.fixtures.storage_seed import seed_messages, seed_session


class _PluginEngine(ContextEngine):
    """A plugin engine whose bind takes main's two arguments and reads the session through ``SessionDB.get_session``."""

    def __init__(self):
        super().__init__()
        self.bound = []
        self.ended = []

    @property
    def name(self):
        return "stub-plugin"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        return [messages[0], {"role": "assistant", "content": "ENGINE_SUMMARY"}]

    def bind_session_state(self, session_db, session_id):
        self.bound.append((session_db, session_db.get_session(session_id)))

    def on_session_end(self, session_id, messages):
        self.ended.append(session_id)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(hermes_state, "_warned", True)
    handle = open_persistence(profile_scope(), eager=True)
    seed_session(handle, "s1", "cli", model="m1", title="alpha plan")
    seed_messages(handle, "s1", *({"role": role, "content": f"{role} turn {i} " + "x" * 200}
                                  for i in range(3) for role in ("user", "assistant")))
    yield handle
    handle.close()


@pytest.fixture
def closes(monkeypatch):
    calls = []
    real = hermes_state.SessionDB.close
    monkeypatch.setattr(hermes_state.SessionDB, "close", lambda self: (calls.append(self), real(self))[1])
    return calls


def test_a_cold_compress_binds_a_plugin_engine_to_a_session_db_closed_when_it_ends(monkeypatch, store, closes):
    from gateway import session_mutation_compress as compress
    engine = _PluginEngine()
    monkeypatch.setattr("agent.agent_init._select_context_engine", lambda _config: engine)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *_a, **_k: 1_000_000)
    monkeypatch.setattr("gateway.session_policy.restore_policy",
                        lambda _policy: SimpleNamespace(safe_mode=False, config=lambda _authority: {}))
    monkeypatch.setattr("gateway.session_policy.policy_scope", lambda _policy, authority=None: nullcontext())
    runner = SimpleNamespace(_resolve_session_agent_runtime=lambda **_kw: ("test/model", {}))
    authority = SimpleNamespace(runner=runner, persistence=store, profile_id=str(store.scope.profile_dir))
    live = SimpleNamespace(source=SimpleNamespace(platform="cli"), route="agent:main:cli")
    snapshot = {"receipt": {"policy": {}}, "target": "s1"}

    async def run():
        prepared = await compress.prepare_compress(authority, live, {"focus": "the plan"}, {"snapshot": snapshot})
        await compress.finish_compress_engine(authority, prepared, None)  # a refused commit ends the one-off engine
        return prepared

    prepared = asyncio.run(run())
    [(session_db, row)] = engine.bound
    assert isinstance(session_db, hermes_state.SessionDB)
    assert row == store.sessions.get("s1").as_session_row()
    assert (row["id"], row["source"], row["model"], row["title"]) == ("s1", "cli", "m1", "alpha plan")
    assert prepared["messages"][-1]["content"] == "ENGINE_SUMMARY"
    assert engine.ended == ["s1"]
    assert closes == [session_db]


def test_session_hygiene_rebinds_a_plugin_engine_to_the_agents_session_db(store, closes):
    engine = _PluginEngine()
    with patch("agent.agent_init._select_context_engine", return_value=engine):
        agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test/model", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, skip_background_review=True,
                        persistence=store, session_id="s1")
    agent._compress_context = lambda messages, *_a, **_k: (messages, None)
    [(session_db, _row)] = engine.bound

    async def build_agent(*_args):
        return agent, store

    async def wait_for_summary(attempt, *_args):
        return await attempt.future

    async def nothing(*_args, **_kwargs):
        return None

    async def cleanup(hyg_agent, **_kwargs):
        GatewayRunner._cleanup_agent_resources(gateway, hyg_agent)

    gateway = SimpleNamespace(
        _hmwa_hygiene_build_agent=build_agent, _hmwa_hygiene_wait_for_summary=wait_for_summary,
        _hmwa_hygiene_apply_result=nothing, _track_deferred_agent_worker=lambda *_a: None,
        _evict_cached_agent=lambda _key: None, _cleanup_agent_resources_off_loop=cleanup,
    )
    history = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    attempt = GatewayRunner._HygieneAttempt(agent=None, meta=None)
    asyncio.run(GatewayRunner._hmwa_hygiene_detached_attempt(
        gateway, attempt, SimpleNamespace(total_ceiling_seconds=60.0), SimpleNamespace(approx_tokens=100_000),
        history, history, "test/model", {}, None, SimpleNamespace(session_id="s1"), "agent:main:cli", "qk", 0,
    ))
    assert len(engine.bound) == 2
    assert engine.bound[-1][0] is session_db
    assert engine.bound[-1][1] == store.sessions.get("s1").as_session_row()
    assert closes == [session_db]
