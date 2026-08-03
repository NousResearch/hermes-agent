"""Tests for native NeMo Relay plugin configuration ownership."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent import relay_runtime


class _FakeRelay:
    def __init__(self, *, initialize_error: Exception | None = None) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.initialize_error = initialize_error
        self.ScopeType = SimpleNamespace(Agent="agent")
        self.plugin = SimpleNamespace(
            initialize=self._initialize_plugins,
            clear=self._clear_plugins,
        )
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
        )
        self.subscribers = SimpleNamespace(flush=self._flush)

    def get_scope_stack(self) -> None:
        return None

    async def _initialize_plugins(self, config: dict[str, Any]) -> dict[str, Any]:
        self.events.append(("plugin.initialize", config))
        if self.initialize_error is not None:
            raise self.initialize_error
        return {"diagnostics": []}

    def _clear_plugins(self) -> None:
        self.events.append(("plugin.clear",))

    def _scope_push(self, name: str, scope_type: Any, **kwargs: Any) -> Any:
        handle = ("scope", name, len(self.events))
        self.events.append(("scope.push", name, scope_type, kwargs))
        return handle

    def _scope_pop(self, handle: Any, **kwargs: Any) -> None:
        self.events.append(("scope.pop", handle, kwargs))

    def _flush(self) -> None:
        self.events.append(("subscribers.flush",))


@pytest.fixture(autouse=True)
def _reset_runtime():
    relay_runtime._reset_for_tests()
    yield
    relay_runtime._reset_for_tests()


def test_relay_discovers_plugins_before_first_session_scope():
    relay = _FakeRelay()
    host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert host.managed_execution_enabled()
        host.ensure_session({"session_id": "session"})
        assert relay.events[0] == ("plugin.initialize", {})
        assert relay.events[1][0:2] == ("scope.push", relay_runtime.SESSION_SCOPE)
    finally:
        host.shutdown()


def test_initialization_failure_is_fail_open(caplog):
    relay = _FakeRelay(initialize_error=RuntimeError("rejected config"))

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert not host.managed_execution_enabled()
        assert "Hermes Relay plugin initialization failed" in caplog.text
    finally:
        host.shutdown()


def test_missing_initialize_api_is_fail_open(caplog):
    relay = _FakeRelay()
    del relay.plugin.initialize

    with caplog.at_level("WARNING"):
        host = relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    try:
        assert not host.managed_execution_enabled()
        assert "does not expose plugin.initialize" in caplog.text
    finally:
        host.shutdown()


def test_two_profile_hosts_initialize_once_and_clear_after_final_shutdown():
    relay = _FakeRelay()
    host_a = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-a")
    host_b = relay_runtime.RelayRuntime(relay=relay, profile_key="profile-b")

    assert relay.events == [("plugin.initialize", {})]
    assert host_a.managed_execution_enabled()
    assert host_b.managed_execution_enabled()
    host_b.ensure_session({"session_id": "profile-b-session"})

    host_a.shutdown()
    assert ("plugin.clear",) not in relay.events

    host_b.shutdown()
    assert relay.events[-2:] == [
        ("subscribers.flush",),
        ("plugin.clear",),
    ]
    assert relay.events.count(("plugin.initialize", {})) == 1
    assert relay.events.count(("plugin.clear",)) == 1
    pop_index = next(
        index for index, event in enumerate(relay.events) if event[0] == "scope.pop"
    )
    assert pop_index < relay.events.index(("plugin.clear",))


def test_plugin_initialization_inside_running_event_loop():
    relay = _FakeRelay()

    async def construct_host() -> relay_runtime.RelayRuntime:
        return relay_runtime.RelayRuntime(relay=relay, profile_key="profile")

    host = asyncio.run(construct_host())
    try:
        assert relay.events == [("plugin.initialize", {})]
        assert host.managed_execution_enabled()
    finally:
        host.shutdown()


def test_real_binding_discovers_project_config_and_exports_native_activity(
    tmp_path,
    monkeypatch,
):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    from agent import relay_llm, relay_tools

    project_root = tmp_path / "project"
    working_directory = project_root / "workspace"
    config_directory = project_root / ".nemo-relay"
    atof_dir = tmp_path / "atof"
    atif_dir = tmp_path / "atif"
    working_directory.mkdir(parents=True)
    config_directory.mkdir()
    (config_directory / "plugins.toml").write_text(
        f"""
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 2

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{atof_dir}"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = "{atif_dir}"
filename_template = "trajectory-{{session_id}}.json"
agent_name = "Hermes Native Test"
agent_version = "test"
""".strip(),
        encoding="utf-8",
    )
    xdg_config_home = tmp_path / "xdg"
    xdg_config_home.mkdir()
    monkeypatch.chdir(working_directory)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: relay)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay.plugin.clear()

    profile_key = relay_runtime.current_profile_key()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=profile_key,
        session_id="native-export",
        platform="cli",
        model="test-model",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    try:
        assert lease.host.managed_execution_enabled()
        relay_llm.execute(
            {"model": "test-model", "messages": []},
            lambda _request: {
                "id": "response-1",
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
            session_id="native-export",
            name="test-provider",
            model_name="test-model",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "request-1",
            },
        )
        relay_tools.execute(
            "terminal",
            {"command": "true"},
            lambda _args: {"output": "ok"},
            session_id="native-export",
            metadata={"tool_call_id": "tool-1"},
        )
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime.SESSION_COORDINATOR.finalize_conversation(
            profile_key=profile_key,
            session_id="native-export",
        )
        relay_runtime._reset_for_tests()

    assert (atof_dir / "events.jsonl").is_file()
    trajectories = list(atif_dir.glob("trajectory-*.json"))
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text(encoding="utf-8"))
    observed_categories = {
        event["category"]
        for event in trajectory["extra"]["observed_events"]
        if event["kind"] == "scope"
    }
    assert {"agent", "llm", "tool"} <= observed_categories
