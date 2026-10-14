"""Completed-turn model handoffs preserve worker identity and fail closed."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock
from copy import deepcopy

import pytest

from gateway.run_goals import GatewayGoalsMixin
from hermes_cli import goals
from hermes_cli.plugins import PluginManager


class Store:
    def __init__(self):
        self.override = None
        self.sid = "worker"
    def peek_session_id(self, key):
        return self.sid
    async def set_model_override(self, key, override):
        self.override = deepcopy(override)


class Runner(GatewayGoalsMixin):
    def __init__(self):
        self.session_store = Store()
        self.async_session_store = self.session_store
        self.agent = SimpleNamespace(model="original", provider="test")
        self.override = None
        self.lock = asyncio.Lock()
        self.source_allowed = True
        self.switched = []
        self._perform_model_switch = AsyncMock(return_value=(SimpleNamespace(
            new_model="alternate-1", target_provider="test", api_key="dummy",
            base_url="https://test.invalid/v1", api_mode="chat_completions",
            request_overrides={}, runtime_capabilities={},
        ), None))
    def _session_key_for_source(self, source):
        return "agent:main:telegram:dm:owner"
    def _profile_scope_for_source(self, source):
        return nullcontext()
    def _is_user_authorized_for_source(self, source):
        return self.source_allowed
    def _session_model_override(self, key):
        return self.override
    def _rehydrate_session_model_override(self, key):
        if self.session_store.override:
            self.override = deepcopy(self.session_store.override)
            self.agent.model = self.override["model"]
    def _cached_agent_for(self, key):
        return self.agent
    def _snapshot_session_model_override(self, key):
        return {"had_override": self.override is not None, "override": deepcopy(self.override)}
    def _model_switch_lock(self):
        return self.lock
    def _switch_cached_agent_model(self, result, ctx, picker):
        self.switched.append((ctx.session_key, result.new_model))
        self.agent.model = result.new_model
    async def _record_model_switch(self, result, ctx, **kwargs):
        assert ctx.persist_global is False
        self.override = {"provider": result.target_provider, "model": result.new_model, "base_url": result.base_url}
    async def _run_in_executor_with_context(self, fn, *args):
        return fn(*args)


@pytest.fixture
def pending(monkeypatch):
    # This runner supplies its route directly; no live gateway/bootstrap config is needed.
    monkeypatch.setattr("gateway.slash_commands_model._ModelSwitchContext.read_config", lambda self: None)
    cfg = {"goal_recovery": {"auto_effects": ["read_only"], "model_handoff": {
        "enabled": True, "session_ids": ["worker"], "routes": [{"provider":"test", "model":"alternate-1"}],
        "max_attempts": 3,
    }}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: deepcopy(cfg))
    pm = PluginManager()
    pm._hooks["goal_blocked"] = [lambda **kwargs: {
        "selected":"inspect", "automatic":True, "classifications":{"cause":{"label":"service"}},
        "options":[{"id":"inspect","route":"gather_evidence","plan":"Inspect checkpoint",
                    "verification":"Verify actual failure","effects":["read_only"],"risk":"low",
                    "authorization":"authorized","reversibility":"reversible","progress":"useful","evidence":"sufficient"}],
    }]
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: pm)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **kw: ("blocked","service unavailable",False,None,False))
    mgr = goals.GoalManager("worker", default_max_turns=20)
    mgr.set("Original task")
    mgr.worker_model_route = {"provider":"test","model":"original"}
    result = mgr.evaluate_after_turn("Failure")
    return result["model_handoff"], cfg


async def apply(runner, pending):
    method = getattr(runner, "_apply_pending_goal_model_handoff", None)
    assert callable(method), "gateway must consume a completed-turn model handoff"
    return await method(SimpleNamespace(session_id="worker"), SimpleNamespace(), pending)


@pytest.mark.asyncio
async def test_switch_commits_only_original_session_after_generation_probe(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    assert callable(getattr(runner, "_apply_pending_goal_model_handoff", None)), "missing handoff consumer"
    probe = AsyncMock(return_value=True)
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", probe)
    result = await apply(runner, request)
    assert result and "Recommended" in result["prompt"]
    assert runner.switched == [("agent:main:telegram:dm:owner", "alternate-1")]
    assert runner.session_store.override["model"] == "alternate-1"
    assert "api_key" not in runner.session_store.override
    assert goals.load_goal("worker").turns_used == 1
    assert goals.load_goal("worker").model_recovery["pending"]["status"] == "committed"
    assert probe.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["pause","routing","owner","model","probe","persistence"])
async def test_user_changes_or_failed_validation_never_start_another_worker(pending, monkeypatch, change):
    request, _ = pending
    runner = Runner()
    assert callable(getattr(runner, "_apply_pending_goal_model_handoff", None)), "missing handoff consumer"
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", AsyncMock(return_value=change != "probe"))
    actions = {
        "pause": lambda: goals.GoalManager("worker").pause("user stop"),
        "routing": lambda: setattr(runner.session_store, "sid", "replacement"),
        "owner": lambda: setattr(runner, "source_allowed", False),
        "model": lambda: setattr(runner.agent, "model", "user-choice"),
        "probe": lambda: None,
        "persistence": lambda: setattr(runner.session_store, "set_model_override",
                                       AsyncMock(side_effect=OSError("disk unavailable"))),
    }
    actions[change]()
    result = await apply(runner, request)
    assert result is None
    assert runner.switched == []
    assert goals.load_goal("worker").max_turns == 20


@pytest.mark.asyncio
async def test_restart_adopts_already_persisted_target_without_second_reservation(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    runner.session_store.override = {"provider":"test","model":"alternate-1"}
    from hermes_cli.goal_model_recovery import qualify
    qualify(goals.GoalManager("worker"), request["id"])
    assert callable(getattr(runner, "_apply_pending_goal_model_handoff", None)), "missing handoff consumer"
    probe = AsyncMock()
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", probe)
    result = await apply(runner, request)
    assert result and result["target"]["model"] == "alternate-1"
    assert runner.switched == []
    assert probe.await_count == 0
    assert len(goals.load_goal("worker").model_recovery["attempts"]) == 1


@pytest.mark.asyncio
async def test_completed_turn_queues_only_a_committed_handoff(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", AsyncMock(return_value=True))
    runner._post_turn_manager = AsyncMock(return_value=goals.GoalManager("worker"))
    runner._defer_goal_status_notice_after_delivery = AsyncMock()
    adapter = SimpleNamespace()
    runner._delivery_adapter_for = lambda source: adapter
    queued = []
    runner._enqueue_fifo = lambda key, event, target: queued.append((key, event, target))
    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="worker"), source=SimpleNamespace(),
        final_response="Blocked on the same small failure")
    assert len(queued) == 1, "completed-turn path must consume the handoff before continuation"
    assert queued[0][1]._goal_model_handoff_id == request["id"]
    assert goals.load_goal("worker").model_recovery["pending"]["status"] == "committed"


@pytest.mark.asyncio
async def test_pre_turn_claims_handoff_once_and_preserves_worker(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    assert callable(getattr(runner, "_apply_pending_goal_model_handoff", None))
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", AsyncMock(return_value=True))
    await apply(runner, request)
    from gateway.goal_model_handoff import before_turn
    runner._warm_goals_session_db = AsyncMock()
    event = SimpleNamespace(text="", _goal_model_handoff_id=request["id"])
    assert await before_turn(runner, event, SimpleNamespace(), "agent:main:telegram:dm:owner")
    assert event.text.startswith("[Continuing toward your standing goal]")
    assert goals.load_goal("worker").model_recovery.get("pending") is None
    assert not await before_turn(runner, event, SimpleNamespace(), "agent:main:telegram:dm:owner")


@pytest.mark.asyncio
async def test_persisted_but_unqualified_target_is_not_adopted(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    runner.session_store.override = {"provider":"test","model":"alternate-1"}
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", AsyncMock(return_value=False))
    assert await apply(runner, request) is None
    assert goals.load_goal("worker").status == "paused"


@pytest.mark.asyncio
async def test_restart_restores_pending_marker_only_for_authorized_owner(pending, monkeypatch):
    _request, _ = pending
    runner = Runner()
    from gateway.goal_model_handoff import restore_markers
    import threading
    entry = SimpleNamespace(session_id="worker", session_key="agent:main:telegram:dm:owner",
                            suspended=False, origin=SimpleNamespace())
    runner.session_store._lock = threading.RLock()
    runner.session_store._ensure_loaded_locked = lambda: None
    runner.session_store._entries = {"key":entry}
    runner._is_session_running = lambda key: False
    runner._restored_source = lambda entry: entry.origin
    runner._warm_goals_session_db = AsyncMock()
    runner._resume_owner_authorized = lambda key, source: runner.source_allowed
    runner.async_session_store.mark_resume_pending = AsyncMock()
    await restore_markers(runner)
    runner.async_session_store.mark_resume_pending.assert_awaited_once_with(entry.session_key, "goal_model_handoff")
    runner.source_allowed = False
    await restore_markers(runner)
    assert runner.async_session_store.mark_resume_pending.await_count == 1


@pytest.mark.asyncio
async def test_real_user_message_preempts_handoff_without_text_loss(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    monkeypatch.setattr("gateway.goal_model_handoff.probe_candidate", AsyncMock(return_value=True))
    await apply(runner, request)
    from gateway.goal_model_handoff import before_turn
    runner._warm_goals_session_db = AsyncMock()
    event = SimpleNamespace(text="Please prioritize the current bug instead.")
    assert await before_turn(runner, event, SimpleNamespace(), "agent:main:telegram:dm:owner")
    assert event.text == "Please prioritize the current bug instead."
    assert goals.load_goal("worker").model_recovery.get("pending") is None


@pytest.mark.asyncio
async def test_revoked_owner_finishes_reservation_instead_of_silent_stall(pending, monkeypatch):
    request, _ = pending
    runner = Runner()
    runner.source_allowed = False
    assert await apply(runner, request) is None
    state = goals.load_goal("worker")
    assert state.model_recovery.get("pending") is None
    assert state.status == "paused"


@pytest.mark.asyncio
async def test_disable_does_not_allow_stale_tagged_event(pending, monkeypatch):
    request, cfg = pending
    cfg["goal_recovery"]["model_handoff"]["enabled"] = False
    from gateway.goal_model_handoff import before_turn
    runner = Runner()
    event = SimpleNamespace(text=request["prompt"], _goal_model_handoff_id=request["id"])
    assert not await before_turn(runner, event, SimpleNamespace(), "agent:main:telegram:dm:owner")


@pytest.mark.asyncio
async def test_malformed_scope_does_not_break_normal_inbound(pending):
    _request, cfg = pending
    cfg["goal_recovery"]["model_handoff"]["session_ids"] = None
    from gateway.goal_model_handoff import before_turn
    runner = Runner()
    event = SimpleNamespace(text="Real user question")
    assert await before_turn(runner, event, SimpleNamespace(), "agent:main:telegram:dm:owner")
    assert event.text == "Real user question"
