from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.turn_timing import (
    finish_turn_timing,
    progress_turn_timing,
    start_turn_timing,
)


class TimingProvider:
    name = "timing"

    def __init__(self, estimate=None, fail=None):
        self.events = []
        self.estimate = estimate
        self.fail = fail

    def feature_capabilities(self):
        return {"contract_version": 1, "adaptive_eta": True, "remaining_time": True}

    def on_turn_timing_start(self, turn, **kwargs):
        if self.fail == "start":
            raise RuntimeError("start failed")
        self.events.append(("start", turn, kwargs))

    def on_turn_progress(self, turn, **kwargs):
        if self.fail == "progress":
            raise RuntimeError("progress failed")
        self.events.append(("progress", turn, kwargs))

    def on_turn_finish(self, turn, **kwargs):
        if self.fail == "finish":
            raise RuntimeError("finish failed")
        self.events.append(("finish", turn, kwargs))

    def estimate_turn(self, **kwargs):
        if self.fail == "estimate":
            raise RuntimeError("estimate failed")
        return self.estimate

    def abort_open_turns(self):
        self.events.append(("abort",))


def _manager(*providers):
    manager = MemoryManager.__new__(MemoryManager)
    manager._providers = list(providers)
    return manager


def _agent(manager):
    emitted = []
    return SimpleNamespace(
        _memory_manager=manager,
        session_id="private-session",
        _user_turn_count=0,
        platform="telegram",
        _emit_status=emitted.append,
        _turn_timing=None,
        _interrupt_requested=False,
        emitted=emitted,
    )


def test_manager_exposes_capabilities_and_isolates_optional_hook_failures():
    broken = TimingProvider(fail="start")
    healthy = TimingProvider(estimate={"sample_count": 5, "recommended_ms": 300_000})
    manager = _manager(broken, healthy)

    assert manager.feature_capabilities()["remaining_time"] is True
    manager.on_turn_timing_start(1, subject="development")
    assert healthy.events[0][0] == "start"
    assert manager.estimate_turn(subject="development")["recommended_ms"] == 300_000


def test_start_and_progress_emit_human_readable_total_and_remaining(monkeypatch):
    provider = TimingProvider(
        estimate={
            "sample_count": 8,
            "recommended_ms": 600_000,
            "p50_ms": 300_000,
            "p80_ms": 480_000,
            "confidence": "medium",
            "critical_path": "development",
        }
    )
    agent = _agent(_manager(provider))
    clock = iter([100.0, 220.0])
    monkeypatch.setattr("agent.turn_timing.time.monotonic", lambda: next(clock))

    state = start_turn_timing(agent, "Please debug and deploy this code")
    assert state is not None
    assert agent.emitted == ["⏱ Estimated total time: about 10 min"]
    progress_turn_timing(agent, "active", iteration=2, announce_remaining=True)
    assert agent.emitted[-1] == "⏱ Estimated time remaining: about 8 min"
    assert "private-session" not in repr(agent.emitted)


def test_insufficient_samples_silently_records_without_fake_eta():
    agent = _agent(_manager(TimingProvider(estimate=None)))
    state = start_turn_timing(agent, "research this")
    assert state is not None
    assert agent.emitted == []
    assert state["subject"] == "research"


def test_finish_classifies_every_exit_and_clears_state():
    provider = TimingProvider()
    agent = _agent(_manager(provider))
    start_turn_timing(agent, "deploy this")
    finish_turn_timing(agent, {"completed": False, "interrupted": True})
    assert provider.events[-1][2]["outcome"] == "interrupted"
    assert agent._turn_timing is None

    start_turn_timing(agent, "deploy this")
    finish_turn_timing(agent, None, RuntimeError("boom"))
    assert provider.events[-1][2]["outcome"] == "failed"
    assert agent._turn_timing is None


def test_feature_failures_never_block_the_agent_turn():
    agent = _agent(_manager(TimingProvider(fail="estimate")))
    assert start_turn_timing(agent, "do operations") is not None
    progress_turn_timing(agent, "wait", iteration=1, announce_remaining=True)
    finish_turn_timing(agent, {"completed": True})


def test_real_conversation_entrypoint_always_closes_timing(monkeypatch):
    import agent.conversation_loop as loop

    provider = TimingProvider()
    agent = _agent(_manager(provider))
    monkeypatch.setattr(
        loop,
        "_run_conversation_impl",
        lambda *_args, **_kwargs: {"completed": True, "final_response": "ok"},
    )

    result = loop.run_conversation(agent, "deploy this")
    assert result["completed"] is True
    assert [event[0] for event in provider.events] == ["start", "finish"]


def test_real_conversation_entrypoint_closes_escaped_exception(monkeypatch):
    import agent.conversation_loop as loop

    provider = TimingProvider()
    agent = _agent(_manager(provider))

    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_run_conversation_impl", explode)
    with pytest.raises(RuntimeError, match="boom"):
        loop.run_conversation(agent, "research this")
    assert provider.events[-1][0] == "finish"
    assert provider.events[-1][2]["outcome"] == "failed"
