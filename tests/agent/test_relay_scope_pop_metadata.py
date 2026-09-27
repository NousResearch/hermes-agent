"""Regression for #78993: scope.pop metadata kwarg on older NeMo Relay."""

from __future__ import annotations

import inspect
import logging
import tempfile
from types import SimpleNamespace

import pytest

from agent import relay_runtime


def test_pop_relay_scope_omits_unsupported_metadata_kwarg():
    calls: list[tuple[object, dict]] = []

    def pop_without_metadata(handle, *, output=None):
        calls.append((handle, {"output": output}))

    relay = SimpleNamespace(scope=SimpleNamespace(pop=pop_without_metadata))
    handle = ("scope", "hermes.turn", 1)

    relay_runtime.pop_relay_scope(
        relay,
        handle,
        output={"outcome": "success"},
        metadata={"hermes.relay.schema_version": "hermes.relay.runtime.v1"},
    )

    assert calls == [(handle, {"output": {"outcome": "success"}})]


def test_pop_relay_scope_forwards_metadata_when_supported():
    calls: list[tuple[object, dict]] = []

    def pop_with_metadata(handle, *, output=None, metadata=None, timestamp=None):
        calls.append(
            (
                handle,
                {
                    "output": output,
                    "metadata": metadata,
                    "timestamp": timestamp,
                },
            )
        )

    relay = SimpleNamespace(scope=SimpleNamespace(pop=pop_with_metadata))
    handle = ("scope", "hermes.turn", 2)
    metadata = {"hermes.relay.runtime_instance": "abc"}

    relay_runtime.pop_relay_scope(
        relay,
        handle,
        output={"outcome": "error"},
        metadata=metadata,
    )

    assert calls == [
        (
            handle,
            {
                "output": {"outcome": "error"},
                "metadata": metadata,
                "timestamp": None,
            },
        )
    ]


def test_end_turn_finalization_survives_pop_without_metadata(monkeypatch, caplog):
    """Mirror #78993: nemo-relay 0.3.x rejects metadata= on scope.pop."""
    pytest.importorskip("nemo_relay")

    monkeypatch.setenv("HERMES_HOME", tempfile.mkdtemp())
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-78993",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.relay_scope_pop")

    original_pop = lease.host.relay.scope.pop
    assert "metadata" in inspect.signature(original_pop).parameters

    def pop_without_metadata(handle, *, output=None, timestamp=None):
        return original_pop(handle, output=output, timestamp=timestamp)

    monkeypatch.setattr(lease.host.relay.scope, "pop", pop_without_metadata)

    logical = lease.host.run_in_session(
        lease.session,
        lease.host.relay.scope.push,
        "logical-llm",
        lease.host.relay.ScopeType.Custom,
        handle=turn.handle,
        input={},
        metadata={"hermes.test": True},
    )
    turn.logical_llm_calls["api-1"] = logical

    with caplog.at_level(logging.WARNING, logger="agent.relay_runtime"):
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")

    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "unexpected keyword argument 'metadata'" not in joined
    assert "turn finalization failed" not in joined
    assert "logical LLM finalization failed" not in joined
    assert turn.logical_llm_calls == {}
    assert turn.closed is True

    lease.host.release_managed_execution("test.relay_scope_pop")
    relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
    relay_runtime._reset_for_tests()
