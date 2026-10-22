"""A gateway session pin on a dead model must heal itself instead of re-arming every turn.

forge #142709: a session /model override pointing at a model the provider no longer serves makes
every turn fail with ``model_not_found``; the fallback rescues delivery, but the pin survives the
post-turn evict and the next turn rehydrates the same broken agent — forever. The heal is: when a
SUCCESSFUL fallback was triggered by ``model_not_found`` and the abandoned model IS this session's
pin, drop the pin (persisted first, then the in-memory view) so the next turn resolves the
configured default model again, and tell the user.

Every test drives the real chain end to end: a real :class:`~agent.error_classifier.classify_api_error`
verdict, a real ``try_activate_fallback`` activation stamping ``agent._fallback_abandoned_model``,
and the real ``GatewayRunner._run_agent_evict_on_fallback`` post-turn hook over a real
``SessionStore`` (state.db ``gateway_routing``). Negative cases pin the invariants: a
non-``model_not_found`` fallback never touches the pin, a failed run never touches the pin, and a
failing store write must not turn the rescued turn into a broken one.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from gateway.turn_context import TurnContext

DEFAULT_MODEL = "gpt-5.6-sol"
GHOST_MODEL = "ghost-model-x"
PIN = {"model": GHOST_MODEL, "provider": "openai", "base_url": "https://api.openai.example/v1"}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="tester", chat_type="dm")


@pytest.fixture
def store(tmp_path, monkeypatch) -> SessionStore:
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _make_agent(*, model=GHOST_MODEL, fallback_model=DEFAULT_MODEL, fallback_provider="openai"):
    """A real AIAgent sitting on *model*, with one fallback entry on a distinct backend."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            fallback_model=[{"provider": fallback_provider, "model": fallback_model}])
    agent.model = model
    agent.provider = "openai"
    return agent


def _activate_fallback(agent, reason, *, fb_model=DEFAULT_MODEL) -> None:
    """Drive the real activation so the agent carries a genuinely stamped signal."""
    client = SimpleNamespace(base_url="https://api.openai.com/v1", api_key="fb-key")
    with patch("agent.auxiliary_client.resolve_provider_client",
               return_value=(client, fb_model)):
        assert agent._try_activate_fallback(reason=reason) is True


def _make_runner(store) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    return runner


def _make_turn_ctx(session_key, agent, *, failed=False) -> TurnContext:
    result = {"failed": True, "error": "fallback exhausted"} if failed else {"reply": "delivered"}
    return TurnContext(
        source=_make_source(), session_key=session_key, agent_holder=[agent], result_holder=[result])


def _pin_memory(runner, session_key, override):
    runner._session_state(session_key).conversation.model_override = dict(override)


def _poison_pin(store, runner, session_key) -> None:
    store.set_model_override(session_key, dict(PIN))
    _pin_memory(runner, session_key, PIN)


def _capture_adapter():
    sent = []

    class _Adapter:
        async def emit_warning(self, chat_id, text, metadata=None, logical_platform=None):
            sent.append({"chat_id": chat_id, "text": text, "metadata": metadata})

    return _Adapter(), sent


def test_poisoned_pin_self_heals_and_next_turn_uses_default_model(store):
    """The defect class, end to end: pin on a dead model → model_not_found → fallback delivers →
    the pin is gone from both layers, and a restart-fresh turn resolves the config default."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    _poison_pin(store, runner, session_key)

    agent = _make_agent(model=GHOST_MODEL)
    err = SimpleNamespace(status_code=404, message=f"The model '{GHOST_MODEL}' does not exist",
                          response=SimpleNamespace(headers={}), body=None)
    classified = classify_api_error(err, provider=agent.provider)
    assert classified.reason == FailoverReason.model_not_found
    _activate_fallback(agent, classified.reason)  # fallback took over and DELIVERED the turn

    healed = runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent))

    # The heal happened and the post-turn evict policy is unchanged.
    assert healed is True
    assert agent._evicted_on_fallback == session_key
    # Both pin layers cleared.
    assert runner._session_model_override(session_key) is None
    assert store.get_model_override(session_key) is None
    # Persisted: a brand-new store over the same state.db sees no override (survives restart).
    fresh_store = SessionStore(sessions_dir=store.sessions_dir, config=GatewayConfig())
    assert fresh_store.get_model_override(session_key) is None

    # Next turn, same session, fresh runner: resolution lands on the config default, not the ghost.
    fresh_runner = _make_runner(fresh_store)
    default_runtime = {"api_key": "default-key", "provider": "openai"}
    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value=dict(default_runtime)), \
         patch("gateway.run._resolve_runtime_agent_kwargs_for_provider",
               return_value=dict(default_runtime)):
        model, _runtime = fresh_runner._resolve_session_agent_runtime(
            session_key=session_key, user_config={"model": {"default": DEFAULT_MODEL}})
    assert model == DEFAULT_MODEL


def test_notice_names_the_dead_model_and_uses_default(store):
    """The user-facing notice: names the dead model, says the pin was dropped for the default."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    _poison_pin(store, runner, session_key)
    agent = _make_agent(model=GHOST_MODEL)
    _activate_fallback(agent, FailoverReason.model_not_found)
    turn_ctx = _make_turn_ctx(session_key, agent)
    adapter, sent = _capture_adapter()
    runner._delivery_adapter_for = lambda source: adapter

    assert runner._run_agent_evict_on_fallback(turn_ctx) is True
    asyncio.run(runner._emit_model_pin_dropped_notice(turn_ctx))

    assert len(sent) == 1
    assert GHOST_MODEL in sent[0]["text"]
    assert "isn't available" in sent[0]["text"]
    assert "default model" in sent[0]["text"]
    # Interim send on stream-is-the-message adapters (gateway/AGENTS.md streaming contract #3).
    assert sent[0]["metadata"].get("_interim_send") is True
    assert sent[0]["chat_id"] == "c1"


def test_notice_send_failure_is_swallowed(store):
    """A failing delivery adapter must never escalate the notice into a turn failure."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    agent = _make_agent(model=GHOST_MODEL)
    _activate_fallback(agent, FailoverReason.model_not_found)
    turn_ctx = _make_turn_ctx(session_key, agent)

    class _BrokenAdapter:
        async def emit_warning(self, *args, **kwargs):
            raise RuntimeError("telegram went away")

    runner._delivery_adapter_for = lambda source: _BrokenAdapter()
    asyncio.run(runner._emit_model_pin_dropped_notice(turn_ctx))  # must not raise


def test_hand_written_pin_in_store_heals_without_memory_view(store):
    """Poison from any source heals — here the pin exists ONLY in the store (hand-edited state.db
    shape), the in-memory view never saw it. The heal reads and clears both layers anyway."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    store.set_model_override(session_key, dict(PIN))  # no memory override on purpose
    agent = _make_agent(model=GHOST_MODEL)
    _activate_fallback(agent, FailoverReason.model_not_found)

    assert runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent)) is True
    assert store.get_model_override(session_key) is None


def test_default_model_not_found_touches_nothing(store):
    """No pin, the config default itself 404s, fallback delivers: nothing may be cleared — the
    default model is not a pin, and no notice may fire (there was no heal)."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    assert store.get_model_override(session_key) is None

    agent = _make_agent(model=DEFAULT_MODEL, fallback_model="deepseek-v4-flash",
                        fallback_provider="deepseek")
    _activate_fallback(agent, FailoverReason.model_not_found, fb_model="deepseek-v4-flash")

    healed = runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent))
    assert not healed
    assert store.get_model_override(session_key) is None
    assert runner._session_model_override(session_key) is None


def test_fallback_also_failing_keeps_the_pin(store):
    """Fallback exhausted (both routes dead): the run FAILED, so the hook must not touch the pin —
    fail-open, never make a bad turn worse."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    _poison_pin(store, runner, session_key)
    agent = _make_agent(model=GHOST_MODEL)

    healed = runner._run_agent_evict_on_fallback(
        _make_turn_ctx(session_key, agent, failed=True))

    assert not healed
    assert store.get_model_override(session_key) == PIN
    assert runner._session_model_override(session_key) == PIN
    assert not hasattr(agent, "_evicted_on_fallback")


def test_rate_limit_fallback_keeps_poisoned_pin(store):
    """A fallback for ANY other reason abandoned a healthy model — the pin must survive it."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    _poison_pin(store, runner, session_key)
    agent = _make_agent(model=GHOST_MODEL)
    err = SimpleNamespace(status_code=429, message="Rate limit exceeded",
                          response=SimpleNamespace(headers={}), body=None)
    classified = classify_api_error(err, provider=agent.provider)
    assert classified.reason == FailoverReason.rate_limit
    _activate_fallback(agent, classified.reason)

    healed = runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent))

    assert not healed
    assert store.get_model_override(session_key) == PIN
    assert runner._session_model_override(session_key) == PIN
    # The ordinary evict-on-fallback policy still ran.
    assert agent._evicted_on_fallback == session_key


def test_healthy_pin_turn_is_untouched(store):
    """Valid pin, no fallback, successful turn: no heal, no evict — post-turn is a no-op."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    good_pin = {"model": DEFAULT_MODEL, "provider": "openai",
                "base_url": "https://api.openai.example/v1"}
    store.set_model_override(session_key, dict(good_pin))
    _pin_memory(runner, session_key, good_pin)
    agent = _make_agent(model=DEFAULT_MODEL)

    healed = runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent))

    assert not healed
    assert store.get_model_override(session_key) == good_pin
    assert runner._session_model_override(session_key) == good_pin
    assert not hasattr(agent, "_evicted_on_fallback")


def test_store_write_failure_is_fail_open(store, monkeypatch):
    """The heal itself failing (store write raises) keeps the status quo: pin intact, the evict
    still runs, and the rescued turn is not turned into a failed one."""
    session_key = store.get_or_create_session(_make_source()).session_key
    runner = _make_runner(store)
    _poison_pin(store, runner, session_key)
    agent = _make_agent(model=GHOST_MODEL)
    _activate_fallback(agent, FailoverReason.model_not_found)

    def _boom(key, override):
        raise RuntimeError("state.db locked")

    monkeypatch.setattr(store, "set_model_override", _boom)
    healed = runner._run_agent_evict_on_fallback(_make_turn_ctx(session_key, agent))

    # Store-first ordering: the failed persist means the in-memory view stays untouched too.
    assert not healed
    assert store.get_model_override(session_key) == PIN
    assert runner._session_model_override(session_key) == PIN
    assert agent._evicted_on_fallback == session_key
