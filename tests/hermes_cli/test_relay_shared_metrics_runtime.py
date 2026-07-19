"""Tests for the direct Hermes-to-Relay shared-metrics runtime."""

from __future__ import annotations

import contextvars
import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import plugins
from hermes_cli.observability import relay_runtime, relay_shared_metrics
from hermes_cli.plugins import PluginManager


class _Request:
    def __init__(self, headers: dict[str, Any], content: dict[str, Any]) -> None:
        self.headers = headers
        self.content = content


class _Relay:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self._callbacks: dict[str, Any] = {}
        self._starts: dict[Any, dict[str, Any]] = {}
        self._scope_starts: dict[Any, dict[str, Any]] = {}
        self._scope = contextvars.ContextVar("relay_scope", default=None)
        self._scope_serial = 0
        self.ScopeType = SimpleNamespace(Agent="agent", Function="function")
        self.LLMRequest = _Request
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
            event=self._scope_event,
        )
        self.llm = SimpleNamespace(call=self._llm_call, call_end=self._llm_call_end)
        self.subscribers = SimpleNamespace(
            register=self._register,
            deregister=self._deregister,
            flush=self._flush,
        )
        self.get_scope_stack = self._get_scope_stack

    def _scope_push(self, name: str, scope_type: Any, **kwargs: Any) -> Any:
        self._scope_serial += 1
        handle = ("scope", name, self._scope_serial)
        self._scope.set(handle)
        self.events.append(("scope.push", name, scope_type, kwargs))
        if scope_type == self.ScopeType.Function:
            self._scope_starts[handle] = kwargs
            event = SimpleNamespace(
                kind="scope",
                category="function",
                name=name,
                scope_category="start",
                category_profile=None,
                metadata=kwargs.get("metadata"),
                data=kwargs.get("input"),
            )
            for callback in list(self._callbacks.values()):
                callback(event)
        return handle

    def _scope_pop(self, handle: Any, **kwargs: Any) -> None:
        self.events.append(("scope.pop", handle, kwargs))
        start = self._scope_starts.pop(handle, None)
        if start is not None:
            event = SimpleNamespace(
                kind="scope",
                category="function",
                name=handle[1],
                scope_category="end",
                category_profile=None,
                metadata={
                    **(start.get("metadata") or {}),
                    **(kwargs.get("metadata") or {}),
                },
                data=kwargs.get("output"),
            )
            for callback in list(self._callbacks.values()):
                callback(event)

    def _scope_event(self, name: str, **kwargs: Any) -> None:
        self.events.append(("scope.event", name, kwargs))

    def _get_scope_stack(self) -> Any:
        current = self._scope.get()
        self.events.append(("scope.sync", current))
        return current

    def _llm_call(
        self,
        name: str,
        request: _Request,
        **kwargs: Any,
    ) -> Any:
        handle = ("llm", name, len(self._starts))
        self._starts[handle] = kwargs
        self.events.append(("llm.call", name, request.content, kwargs))
        return handle

    def _llm_call_end(
        self,
        handle: Any,
        response: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        start = self._starts.pop(handle)
        self.events.append(("llm.call_end", handle, response, kwargs))
        event = SimpleNamespace(
            kind="scope",
            category="llm",
            name=handle[1],
            scope_category="end",
            category_profile={"model_name": start["model_name"]},
            metadata={
                **start["metadata"],
                **kwargs["metadata"],
                "otel.status_code": "OK",
            },
            data=response,
        )
        for callback in list(self._callbacks.values()):
            callback(event)

    def _register(self, name: str, callback: Any) -> None:
        self._callbacks[name] = callback
        self.events.append(("subscribers.register", name))

    def _deregister(self, name: str) -> None:
        self._callbacks.pop(name, None)
        self.events.append(("subscribers.deregister", name))

    def _flush(self) -> None:
        self.events.append(("subscribers.flush",))


@pytest.fixture
def direct_runtime(tmp_path, monkeypatch):
    fake = _Relay()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(plugins, "_plugin_manager", PluginManager())
    yield fake
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def test_direct_runtime_records_without_enabling_a_plugin(direct_runtime, tmp_path):
    base = {
        "session_id": "sensitive-session",
        "task_id": "task-1",
        "api_request_id": "request-1",
        "platform": "cli",
        "provider": "custom",
        "model": "gpt-sensitive-model-id",
        "base_url": "http://127.0.0.1:11434/v1",
    }

    assert plugins.has_hook("pre_api_request")
    plugins.invoke_hook("on_session_start", **base)
    plugins.invoke_hook("pre_llm_call", **base)
    plugins.invoke_hook(
        "pre_api_request",
        **base,
        request={"body": {"messages": ["sensitive-prompt"]}},
    )
    plugins.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="sensitive-tool-call",
        tool_name="terminal",
        args={"command": "sensitive-command"},
        result={"output": "sensitive-tool-result"},
        status="ok",
    )
    plugins.invoke_hook(
        "api_request_error",
        **base,
        retryable=True,
        error={"message": "sensitive-error"},
    )
    plugins.invoke_hook(
        "pre_api_request",
        **{
            **base,
            "provider": "anthropic",
            "model": "claude-sonnet",
            "base_url": "https://api.anthropic.com",
        },
        request={"body": {"messages": ["sensitive-prompt"]}},
    )
    plugins.invoke_hook(
        "post_api_request",
        **{
            **base,
            "provider": "anthropic",
            "model": "claude-sonnet",
            "base_url": "https://api.anthropic.com",
        },
        response={"content": "sensitive-response"},
    )
    plugins.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    plugins.invoke_hook("on_session_finalize", session_id=base["session_id"])

    starts = [event for event in direct_runtime.events if event[0] == "llm.call"]
    ends = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    scope_starts = [
        event for event in direct_runtime.events if event[0] == "scope.push"
    ]
    assert len(scope_starts) == 2
    assert scope_starts[0][2] == direct_runtime.ScopeType.Agent
    assert scope_starts[1][1] == "hermes.task_run"
    assert scope_starts[1][2] == direct_runtime.ScopeType.Function
    assert scope_starts[1][3]["handle"][1] == relay_runtime.SESSION_SCOPE
    assert scope_starts[1][3]["input"] == {
        "entrypoint": "interactive",
        "execution_surface": "cli",
    }
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0][2] == {}
    assert starts[0][3]["model_name"] == "gpt"
    assert ends[0][2] == {
        "call_role": "primary",
        "locality": "remote",
        "model_family": "claude",
        "outcome": "success",
        "provider_family": "direct",
    }
    serialized_events = json.dumps(direct_runtime.events)
    assert "sensitive-prompt" not in serialized_events
    assert "sensitive-response" not in serialized_events
    assert "sensitive-error" not in serialized_events
    assert "sensitive-command" not in serialized_events
    assert "sensitive-tool-result" not in serialized_events
    assert "sensitive-tool-call" not in serialized_events
    assert "gpt-sensitive-model-id" not in serialized_events
    assert plugins.get_plugin_manager().list_plugins() == []

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    packages = list((root / "outbox").glob("*.json"))
    assert len(packages) == 1
    package = json.loads(packages[0].read_text(encoding="utf-8"))
    metrics = {metric["name"]: metric for metric in package["metrics"]}
    assert set(metrics) == {
        "hermes.model_call.count",
        "hermes.task_run.finished",
        "hermes.task_run.started",
    }
    assert metrics["hermes.model_call.count"]["dimensions"]["model_family"] == "claude"
    assert metrics["hermes.model_call.count"]["value"] == 1
    assert metrics["hermes.task_run.started"] == {
        "name": "hermes.task_run.started",
        "type": "counter",
        "dimensions": {
            "entrypoint": "interactive",
            "execution_surface": "cli",
        },
        "value": 1,
    }
    terminal = metrics["hermes.task_run.finished"]["dimensions"]
    assert terminal["duration_bucket"] in {
        "lt_1s",
        "1s_to_5s",
        "5s_to_30s",
        "30s_to_2m",
        "2m_to_10m",
        "gte_10m",
    }
    assert {
        key: value for key, value in terminal.items() if key != "duration_bucket"
    } == {
        "end_reason": "completed",
        "entrypoint": "interactive",
        "execution_surface": "cli",
        "model_call_count_bucket": "1",
        "outcome": "success",
        "retry_count_bucket": "1",
        "termination": "none",
        "tool_call_count_bucket": "1",
    }


def test_direct_runtime_is_disabled_by_default(tmp_path, monkeypatch):
    fake = _Relay()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(plugins, "_plugin_manager", PluginManager())

    assert not plugins.has_hook("pre_api_request")
    plugins.invoke_hook("on_session_start", session_id="s1", platform="cli")
    plugins.invoke_hook("on_session_finalize", session_id="s1")

    assert fake.events == []
    assert not (tmp_path / "hermes-home" / "telemetry").exists()
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def test_core_runtime_is_fail_open_without_a_published_binding(monkeypatch, caplog):
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()

    def missing_relay(name: str):
        assert name == "nemo_relay"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(relay_runtime.importlib, "import_module", missing_relay)

    assert relay_runtime.get_runtime() is None
    assert not relay_runtime.emit_mark("hermes.probe", session_id="s1")
    assert "Hermes Relay runtime initialization failed" in caplog.text
    relay_runtime._reset_for_tests()


def test_core_mark_uses_the_shared_session_handle_without_a_plugin(direct_runtime):
    plugins.invoke_hook("on_session_start", session_id="s1", platform="cli")

    handle = relay_runtime.get_session_handle("s1")
    assert handle is not None
    assert relay_runtime.emit_mark(
        "hermes.skill.created",
        session_id="s1",
        data={"provenance": "agent_created"},
        metadata={"data_schema": "hermes.skill.lifecycle.v1"},
    )

    [mark] = [event for event in direct_runtime.events if event[0] == "scope.event"]
    assert mark[1] == "hermes.skill.created"
    assert mark[2]["handle"] == handle
    assert plugins.get_plugin_manager().list_plugins() == []


def test_core_mark_lazily_starts_relay_without_metrics_or_a_plugin(
    tmp_path,
    monkeypatch,
):
    fake = _Relay()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(plugins, "_plugin_manager", PluginManager())

    assert relay_runtime.emit_mark(
        "hermes.skill.created",
        session_id="s1",
        data={"provenance": "agent_created"},
    )
    plugins.invoke_hook("on_session_finalize", session_id="s1")

    assert [event[0] for event in fake.events] == [
        "scope.push",
        "scope.sync",
        "scope.event",
        "scope.sync",
        "scope.pop",
        "subscribers.flush",
    ]
    assert not any(event[0] == "subscribers.register" for event in fake.events)
    assert not (tmp_path / "hermes-home" / "telemetry").exists()
    relay_runtime._reset_for_tests()


def test_core_runtime_creates_one_session_under_concurrent_access(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    ready = threading.Barrier(8)
    sessions: list[Any] = []

    def ensure() -> None:
        ready.wait(timeout=5)
        sessions.append(runtime.ensure_session({"session_id": "shared"}))

    threads = [threading.Thread(target=ensure) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len({id(session) for session in sessions}) == 1
    assert (
        len([event for event in direct_runtime.events if event[0] == "scope.push"]) == 1
    )


def test_core_runtime_isolates_same_session_id_by_profile(direct_runtime, tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"

    token = set_hermes_home_override(profile_a)
    try:
        runtime_a = relay_runtime.get_runtime()
        session_a = runtime_a.ensure_session({"session_id": "shared"})
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        runtime_b = relay_runtime.get_runtime()
        session_b = runtime_b.ensure_session({"session_id": "shared"})
    finally:
        reset_hermes_home_override(token)

    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a is not runtime_b
    assert runtime_a.profile_key == str(profile_a.resolve())
    assert runtime_b.profile_key == str(profile_b.resolve())
    assert session_a is not session_b
    assert session_a.handle != session_b.handle


def test_shared_metrics_policy_and_store_are_profile_scoped(tmp_path, monkeypatch):
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    fake = _Relay()
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {
            "telemetry": {
                "shared_metrics": {"enabled": get_hermes_home() == profile_a}
            }
        },
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()

    token = set_hermes_home_override(profile_a)
    try:
        assert relay_shared_metrics.enabled()
        relay_shared_metrics.start_task_run(
            session_id="shared",
            task_id="task-a",
            platform="cli",
        )
        relay_shared_metrics.finish_task_run(
            session_id="shared",
            task_id="task-a",
            platform="cli",
            result={"completed": True},
        )
        relay_shared_metrics._get_runtime().close_session({"session_id": "shared"})
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(profile_b)
    try:
        assert not relay_shared_metrics.enabled()
        relay_shared_metrics.start_task_run(
            session_id="shared",
            task_id="task-b",
            platform="cli",
        )
    finally:
        reset_hermes_home_override(token)

    assert list((profile_a / "telemetry" / "shared_metrics" / "outbox").glob("*.json"))
    assert not (profile_b / "telemetry").exists()
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def test_shared_metrics_subscribers_isolate_two_enabled_profiles(tmp_path, monkeypatch):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    fake = _Relay()
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()

    for profile, task_id in ((profile_a, "task-a"), (profile_b, "task-b")):
        token = set_hermes_home_override(profile)
        try:
            relay_shared_metrics.start_task_run(
                session_id="shared",
                task_id=task_id,
                platform="cli",
            )
            relay_shared_metrics.finish_task_run(
                session_id="shared",
                task_id=task_id,
                platform="cli",
                result={"completed": True},
            )
            relay_shared_metrics._get_runtime().close_session(
                {"session_id": "shared"}
            )
        finally:
            reset_hermes_home_override(token)

    for profile in (profile_a, profile_b):
        packages = list(
            (profile / "telemetry" / "shared_metrics" / "outbox").glob("*.json")
        )
        assert len(packages) == 1
        package = json.loads(packages[0].read_text(encoding="utf-8"))
        metrics = {metric["name"]: metric for metric in package["metrics"]}
        assert metrics["hermes.task_run.started"]["value"] == 1
        assert metrics["hermes.task_run.finished"]["value"] == 1

    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def test_shared_metrics_isolates_same_task_id_across_sessions(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None

    task_a = runtime.start_task({
        "session_id": "session-a",
        "task_id": "shared-task",
        "platform": "cli",
    })
    task_b = runtime.start_task({
        "session_id": "session-b",
        "task_id": "shared-task",
        "platform": "gateway",
    })

    assert task_a is not None
    assert task_b is not None
    assert task_a is not task_b
    assert task_a.handle != task_b.handle

    runtime.finish_task({
        "session_id": "session-a",
        "task_id": "shared-task",
        "platform": "cli",
        "completed": True,
    })
    runtime.finish_task({
        "session_id": "session-b",
        "task_id": "shared-task",
        "platform": "gateway",
        "completed": True,
    })

    task_starts = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    task_ends = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert len(task_starts) == 2
    assert len(task_ends) == 2


def test_disabling_shared_metrics_stops_collection_and_shutdown_export(
    tmp_path, monkeypatch
):
    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    fake = _Relay()
    profile = tmp_path / "profile"
    policy = {"enabled": True}
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", lambda: fake)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"telemetry": {"shared_metrics": dict(policy)}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()

    relay_shared_metrics.start_task_run(
        session_id="session",
        task_id="task",
        platform="cli",
    )
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    policy["enabled"] = False

    assert not relay_shared_metrics.enabled()
    counters_before_stale_event = runtime.subscriber.store.counter_snapshot()
    runtime.subscriber(SimpleNamespace(
        kind="scope",
        category="function",
        category_profile=None,
        name="hermes.task_run",
        scope_category="start",
        metadata={
            "hermes.metrics.schema_version": "hermes.metrics.event.v1",
            relay_runtime.RUNTIME_INSTANCE_KEY: runtime.host.runtime_id,
        },
        data={"entrypoint": "interactive", "execution_surface": "cli"},
    ))
    assert runtime.subscriber.store.counter_snapshot() == counters_before_stale_event
    assert runtime.start_task({
        "session_id": "session",
        "task_id": "stale-runtime-task",
        "platform": "cli",
    }) is None
    relay_shared_metrics.finish_task_run(
        session_id="session",
        task_id="task",
        platform="cli",
        result={"completed": True},
    )
    relay_shared_metrics._reset_for_tests()

    root = profile / "telemetry" / "shared_metrics"
    store = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    assert [row["metric_name"] for row in store.counter_snapshot()] == [
        "hermes.task_run.started"
    ]
    assert list((root / "outbox").glob("*.json")) == []
    relay_runtime._reset_for_tests()


def test_shared_metrics_retries_transient_initialization_failure(
    direct_runtime, monkeypatch
):
    real_store = relay_shared_metrics.SharedMetricsStore
    attempts = 0

    def flaky_store():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("transient store failure")
        return real_store()

    monkeypatch.setattr(relay_shared_metrics, "SharedMetricsStore", flaky_store)

    relay_shared_metrics.start_task_run(
        session_id="session",
        task_id="first",
        platform="cli",
    )
    relay_shared_metrics.start_task_run(
        session_id="session",
        task_id="second",
        platform="cli",
    )

    assert attempts == 2
    task_starts = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    assert len(task_starts) == 1


def test_async_session_runner_awaits_inside_saved_relay_context(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    session = runtime.ensure_session({"session_id": "async-session"})
    assert session is not None

    async def probe() -> Any:
        await asyncio.sleep(0)
        return direct_runtime._scope.get()

    result = asyncio.run(runtime.run_in_session_async(session, probe))

    assert result == session.handle


def test_shared_metrics_creates_one_task_under_concurrent_access(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    ready = threading.Barrier(8)
    tasks: list[Any] = []

    def start() -> None:
        ready.wait(timeout=5)
        tasks.append(
            runtime.start_task({"session_id": "s1", "task_id": "t1", "platform": "cli"})
        )

    threads = [threading.Thread(target=start) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len({id(task) for task in tasks}) == 1
    task_starts = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    assert len(task_starts) == 1


def test_core_runtime_parents_subagent_session_without_exposing_ids(
    direct_runtime,
):
    plugins.invoke_hook("on_session_start", session_id="parent", platform="cli")
    parent_handle = relay_runtime.get_session_handle("parent")

    plugins.invoke_hook(
        "subagent_start",
        parent_session_id="parent",
        child_session_id="sensitive-child",
        child_subagent_id="sensitive-subagent",
    )
    plugins.invoke_hook(
        "on_session_start",
        session_id="sensitive-child",
        platform="cli",
    )

    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    child = runtime.get_session("sensitive-child")
    assert child is not None
    assert child.parent_session_id == "parent"
    pushes = [event for event in direct_runtime.events if event[0] == "scope.push"]
    assert len(pushes) == 2
    child_kwargs = pushes[1][3]
    assert child_kwargs["handle"] == parent_handle
    assert child_kwargs["metadata"] == {
        relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
        relay_runtime.RUNTIME_INSTANCE_KEY: runtime.runtime_id,
        "nemo_relay_scope_role": "subagent",
    }
    assert "sensitive-child" not in json.dumps(pushes)
    assert "sensitive-subagent" not in json.dumps(pushes)


def test_core_runtime_closes_child_session_on_subagent_stop(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    runtime.register_subagent({
        "parent_session_id": "parent",
        "child_session_id": "child",
    })
    child = runtime.ensure_session({"session_id": "child"})
    assert child is not None

    runtime.unregister_subagent({"child_session_id": "child"})

    assert runtime.get_session("child") is None
    child_closes = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1] == child.handle
    ]
    assert len(child_closes) == 1


def test_core_runtime_ignores_self_parenting_subagent_event(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None

    runtime.register_subagent({"parent_session_id": "same", "child_session_id": "same"})
    session = runtime.ensure_session({"session_id": "same"})

    assert session is not None
    assert session.parent_session_id == ""


def test_terminal_model_error_is_counted_as_failed(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "r1",
        "provider": "anthropic",
        "model": "claude-sonnet",
    }

    plugins.invoke_hook("pre_api_request", **base)
    plugins.invoke_hook("api_request_error", **base, retryable=False)
    plugins.invoke_hook("on_session_finalize", session_id="s1")

    [end] = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    assert end[2]["outcome"] == "failed"


def test_task_terminal_counts_logical_calls_retries_and_unique_tools(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "r1",
        "platform": "cli",
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
    }

    plugins.invoke_hook("pre_llm_call", **base)
    plugins.invoke_hook("pre_api_request", **base)
    plugins.invoke_hook("api_request_error", **base, retryable=True)
    plugins.invoke_hook("pre_api_request", **base)
    plugins.invoke_hook("api_request_error", **base, retryable=True)
    plugins.invoke_hook("pre_api_request", **base)
    plugins.invoke_hook("api_request_error", **base, retryable=False)
    for tool_call_id in ("tool-1", "tool-1", "tool-2"):
        plugins.invoke_hook(
            "post_tool_call",
            **base,
            tool_call_id=tool_call_id,
            tool_name="terminal",
            result={"output": "private"},
            status="ok",
        )
    plugins.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="all_retries_exhausted_no_response",
    )
    plugins.invoke_hook("on_session_finalize", session_id="s1")

    model_starts = [event for event in direct_runtime.events if event[0] == "llm.call"]
    model_ends = [
        event for event in direct_runtime.events if event[0] == "llm.call_end"
    ]
    assert len(model_starts) == 1
    assert len(model_ends) == 1
    assert model_ends[0][2]["outcome"] == "failed"
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"] == {
        "duration_bucket": task_end[2]["output"]["duration_bucket"],
        "end_reason": "failed",
        "entrypoint": "interactive",
        "execution_surface": "cli",
        "model_call_count_bucket": "1",
        "outcome": "failed",
        "retry_count_bucket": "2",
        "termination": "none",
        "tool_call_count_bucket": "2",
    }


def test_outer_agent_boundary_closes_early_returns_and_exceptions(
    direct_runtime,
    monkeypatch,
):
    from run_agent import AIAgent

    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        _parent_session_id=None,
        _session_db=None,
        _conversation_root_id=lambda: "s1",
    )

    def early_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "final_response": "private failure detail",
            "completed": False,
            "failed": True,
            "interrupted": False,
        }

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        early_failure,
    )
    result = AIAgent.run_conversation(agent, "private prompt", task_id="early")
    assert result["failed"] is True

    def raise_failure(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("private exception detail")

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        raise_failure,
    )
    with pytest.raises(RuntimeError, match="private exception detail"):
        AIAgent.run_conversation(agent, "private prompt", task_id="exception")

    def raise_interrupt(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        raise_interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        AIAgent.run_conversation(agent, "private prompt", task_id="cancelled")

    def raise_timeout(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("private timeout detail")

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        raise_timeout,
    )
    with pytest.raises(TimeoutError, match="private timeout detail"):
        AIAgent.run_conversation(agent, "private prompt", task_id="timed-out")

    plugins.invoke_hook("on_session_finalize", session_id="s1")

    task_ends = [
        event[2]["output"]
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert len(task_ends) == 4
    assert task_ends[0]["outcome"] == "failed"
    assert task_ends[0]["end_reason"] == "failed"
    assert task_ends[0]["termination"] == "none"
    assert task_ends[1]["outcome"] == "failed"
    assert task_ends[1]["end_reason"] == "system_aborted"
    assert task_ends[1]["termination"] == "system_aborted"
    assert task_ends[2]["outcome"] == "cancelled"
    assert task_ends[2]["end_reason"] == "user_cancelled"
    assert task_ends[2]["termination"] == "user_cancelled"
    assert task_ends[3]["outcome"] == "timed_out"
    assert task_ends[3]["end_reason"] == "timed_out"
    assert task_ends[3]["termination"] == "timed_out"
    serialized = json.dumps(direct_runtime.events)
    assert "private prompt" not in serialized
    assert "private failure detail" not in serialized
    assert "private exception detail" not in serialized
    assert "private timeout detail" not in serialized


def test_outer_agent_boundary_preserves_a_returned_timeout_reason(
    direct_runtime,
    monkeypatch,
):
    from run_agent import AIAgent

    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        _parent_session_id=None,
        _session_db=None,
        _conversation_root_id=lambda: "s1",
    )

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        lambda *_args, **_kwargs: {
            "final_response": "private timeout response",
            "completed": False,
            "failed": True,
            "failure_reason": "timeout",
        },
    )
    AIAgent.run_conversation(agent, "private prompt", task_id="timed-out")

    [task_end] = [
        event[2]["output"]
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end["outcome"] == "timed_out"
    assert task_end["end_reason"] == "timed_out"
    assert task_end["termination"] == "timed_out"
    serialized = json.dumps(direct_runtime.events)
    assert "private prompt" not in serialized
    assert "private timeout response" not in serialized


def test_session_finalize_closes_a_pending_task_as_system_aborted(direct_runtime):
    plugins.invoke_hook(
        "pre_llm_call",
        session_id="s1",
        task_id="t1",
        platform="cli",
    )

    plugins.invoke_hook("on_session_finalize", session_id="s1")

    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"] == {
        "duration_bucket": task_end[2]["output"]["duration_bucket"],
        "end_reason": "system_aborted",
        "entrypoint": "interactive",
        "execution_surface": "cli",
        "model_call_count_bucket": "0",
        "outcome": "failed",
        "retry_count_bucket": "0",
        "termination": "system_aborted",
        "tool_call_count_bucket": "0",
    }


def test_sequential_tasks_in_one_session_aggregate_once_each(direct_runtime, tmp_path):
    for task_id in ("t1", "t2"):
        plugins.invoke_hook(
            "pre_llm_call",
            session_id="s1",
            task_id=task_id,
            platform="cli",
        )
        plugins.invoke_hook(
            "on_session_end",
            session_id="s1",
            task_id=task_id,
            platform="cli",
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(stop)",
        )
    plugins.invoke_hook("on_session_finalize", session_id="s1")

    outbox = tmp_path / "hermes-home" / "telemetry" / "shared_metrics" / "outbox"
    [package_path] = list(outbox.glob("*.json"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    metrics = {metric["name"]: metric for metric in package["metrics"]}
    assert metrics["hermes.task_run.started"]["value"] == 2
    assert metrics["hermes.task_run.finished"]["value"] == 2


def test_task_ownership_survives_session_id_rotation(direct_runtime):
    plugins.invoke_hook(
        "pre_llm_call",
        session_id="before-compression",
        task_id="t1",
        platform="cli",
    )
    plugins.invoke_hook(
        "pre_api_request",
        session_id="after-compression",
        task_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    plugins.invoke_hook(
        "post_api_request",
        session_id="after-compression",
        task_id="t1",
        api_request_id="r1",
        platform="cli",
        provider="nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    plugins.invoke_hook(
        "on_session_end",
        session_id="after-compression",
        task_id="t1",
        platform="cli",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    plugins.invoke_hook("on_session_finalize", session_id="before-compression")

    task_starts = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    task_ends = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    model_ends = [
        event for event in direct_runtime.events if event[0] == "llm.call_end"
    ]
    assert len(task_starts) == 1
    assert len(task_ends) == 1
    assert len(model_ends) == 1
    assert model_ends[0][2]["outcome"] == "success"
    assert task_ends[0][2]["output"]["model_call_count_bucket"] == "1"
    assert task_ends[0][2]["output"]["outcome"] == "success"


def test_gateway_and_delegated_entrypoints_flow_through_relay(direct_runtime):
    tasks = [
        {
            "session_id": "gateway-session",
            "task_id": "gateway-task",
            "platform": "whatsapp_cloud",
        },
        {
            "session_id": "child-session",
            "task_id": "delegated-task",
            "platform": "cli",
            "parent_session_id": "private-parent-session",
        },
    ]
    for task in tasks:
        plugins.invoke_hook("pre_llm_call", **task)
        plugins.invoke_hook(
            "on_session_end",
            **task,
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(stop)",
        )

    starts = [
        event[3]["input"]
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    assert starts == [
        {"entrypoint": "gateway_message", "execution_surface": "gateway"},
        {"entrypoint": "delegated", "execution_surface": "cli"},
    ]
    assert "private-parent-session" not in json.dumps(direct_runtime.events)


def test_persistence_failure_does_not_escape_the_hook(
    direct_runtime,
    monkeypatch,
    caplog,
):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None

    def fail_record(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(runtime.subscriber.store, "record_counter", fail_record)
    plugins.invoke_hook(
        "pre_api_request",
        session_id="s1",
        task_id="t1",
        api_request_id="r1",
        provider="openai",
        model="gpt-5",
    )
    plugins.invoke_hook(
        "post_api_request",
        session_id="s1",
        task_id="t1",
        api_request_id="r1",
        provider="openai",
        model="gpt-5",
    )

    assert "Unable to persist the Hermes shared metric" in caplog.text


def test_close_does_not_reopen_a_session_after_scope_start_failure(
    direct_runtime,
    monkeypatch,
):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    original_push = direct_runtime.scope.push
    push_attempts = 0

    def fail_first_push(*args: Any, **kwargs: Any) -> Any:
        nonlocal push_attempts
        push_attempts += 1
        if push_attempts == 1:
            raise RuntimeError("simulated scope failure")
        return original_push(*args, **kwargs)

    direct_runtime.scope.push = fail_first_push
    with pytest.raises(RuntimeError, match="simulated scope failure"):
        runtime.ensure_session({"session_id": "s1"})

    close_started = threading.Event()
    allow_close = threading.Event()
    original_flush = direct_runtime.subscribers.flush

    def block_flush():
        session = runtime._sessions["s1"]
        assert session.closing is True
        close_started.set()
        assert allow_close.wait(timeout=5)
        original_flush()

    direct_runtime.subscribers.flush = block_flush
    close_thread = threading.Thread(
        target=runtime.close_session,
        args=({"session_id": "s1"},),
    )
    close_thread.start()
    assert close_started.wait(timeout=5)

    ensure_thread = threading.Thread(
        target=runtime.ensure_session,
        args=({"session_id": "s1"},),
    )
    ensure_thread.start()
    allow_close.set()
    close_thread.join(timeout=5)
    ensure_thread.join(timeout=5)

    assert not close_thread.is_alive()
    assert not ensure_thread.is_alive()
    assert push_attempts == 1
    assert "s1" not in runtime._sessions
