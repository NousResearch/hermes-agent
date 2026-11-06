"""Relay scope metadata carries the session identity on every scope.

Relay's ``gen_ai`` projection derives ``gen_ai.conversation.id`` (every scope), ``session.id``
(trace roots), ``user.id`` and ``gen_ai.agent.name`` from the ``session_id``, ``user_id`` and
``gen_ai.agent.name`` metadata keys, and never inherits metadata from a parent scope. Backends
that follow the GenAI semantic conventions group sessions and discover agents on those keys, so
the session, turn, LLM and tool scopes each have to carry them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent import relay_llm, relay_runtime, relay_tools

EXPECTED = {"session_id": "session-1", "user_id": "profile", "gen_ai.agent.name": "hermes:profile"}


class _RecordingRelay:
    """Minimal Relay double recording ``scope.push`` keyword arguments."""

    def __init__(self) -> None:
        self.pushes: list[tuple[str, dict[str, Any]]] = []
        self.ScopeType = SimpleNamespace(Agent="agent", Function="function")
        self.plugin = SimpleNamespace(initialize=self._initialize)
        self.scope = SimpleNamespace(push=self._push, pop=self._pop)
        self.subscribers = SimpleNamespace(flush_async=self._flush_async)

    def get_scope_stack(self) -> None:
        return None

    async def _initialize(self, config: dict[str, Any], additional_plugins_toml: Any = None) -> Any:
        class _Activation:
            is_active = True
            report = SimpleNamespace(active=True)

            async def close(self) -> None:
                return None

        return _Activation()

    def _push(self, name: str, scope_type: Any, **kwargs: Any) -> Any:
        self.pushes.append((name, kwargs))
        return ("scope", name, len(self.pushes))

    def _pop(self, handle: Any, **kwargs: Any) -> None:
        return None

    async def _flush_async(self) -> None:
        return None


def test_session_identity_metadata_shape():
    assert relay_runtime.session_identity_metadata("session-1", "profile") == EXPECTED
    assert relay_runtime.session_identity_metadata("session-1", "") == {"session_id": "session-1"}
    assert relay_runtime.session_identity_metadata("", None) == {}


def test_session_scope_carries_identity(monkeypatch):
    monkeypatch.delenv(relay_runtime.RELAY_PLUGINS_CONFIG_ENV, raising=False)
    relay = _RecordingRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")
    try:
        host.ensure_session({"session_id": "session-1"}, metadata={"hermes.execution_surface": "cli"})
    finally:
        host.shutdown()

    name, kwargs = relay.pushes[0]
    assert name == relay_runtime.SESSION_SCOPE
    assert {key: kwargs["metadata"][key] for key in EXPECTED} == EXPECTED
    assert kwargs["metadata"]["hermes.execution_surface"] == "cli"


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.relay_session_identity")
    try:
        yield lease
    finally:
        lease.host.release_managed_execution("test.relay_session_identity")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _expected_for(lease: Any) -> dict[str, str]:
    return relay_runtime.session_identity_metadata("session-1", lease.profile_key)


def test_llm_scope_carries_identity(relay_turn, monkeypatch):
    relay = relay_turn.host.relay
    observed: list[dict[str, Any]] = []
    original_execute = relay.llm.execute

    async def capture(name, request, *args, **kwargs):
        observed.append(dict(kwargs["metadata"]))
        return await original_execute(name, request, *args, **kwargs)

    monkeypatch.setattr(relay.llm, "execute", capture)
    monkeypatch.setattr(relay_llm, "_codec", lambda *_args, **_kwargs: None)

    relay_llm.execute(
        {"model": "test-model", "messages": []},
        lambda _request: {"content": "done"},
        session_id="session-1",
        name="custom",
        model_name="test-model",
        metadata={"api_mode": "chat_completions", "user_id": "caller-wins"},
    )

    expected = _expected_for(relay_turn)
    assert observed[0]["session_id"] == expected["session_id"]
    assert observed[0]["gen_ai.agent.name"] == expected["gen_ai.agent.name"]
    assert observed[0]["user_id"] == "caller-wins"
    assert observed[0]["hermes.provider"] == "custom"


def test_tool_scope_carries_identity(relay_turn, monkeypatch):
    relay = relay_turn.host.relay
    observed: list[dict[str, Any]] = []
    original_execute = relay.tools.execute

    async def capture(tool_name, args, invoke, **kwargs):
        observed.append(dict(kwargs["metadata"]))
        return await original_execute(tool_name, args, invoke, **kwargs)

    monkeypatch.setattr(relay.tools, "execute", capture)

    relay_tools.execute(
        "terminal",
        {"command": "true"},
        lambda _args: {"output": "ok"},
        session_id="session-1",
        metadata={"tool_call_id": "tool-1"},
    )

    expected = _expected_for(relay_turn)
    assert {key: observed[0][key] for key in expected} == expected
    assert observed[0]["tool_call_id"] == "tool-1"
