"""Tests for the direct Hermes-to-Relay shared-metrics runtime."""

from __future__ import annotations

import contextvars
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
        self._scope = contextvars.ContextVar("relay_scope", default=None)
        self._scope_serial = 0
        self.ScopeType = SimpleNamespace(Agent="agent")
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
        return handle

    def _scope_pop(self, handle: Any, **kwargs: Any) -> None:
        self.events.append(("scope.pop", handle, kwargs))

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
    plugins.invoke_hook(
        "pre_api_request",
        **base,
        request={"body": {"messages": ["sensitive-prompt"]}},
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
    plugins.invoke_hook("on_session_finalize", session_id=base["session_id"])

    starts = [event for event in direct_runtime.events if event[0] == "llm.call"]
    ends = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    session_starts = [
        event for event in direct_runtime.events if event[0] == "scope.push"
    ]
    assert len(session_starts) == 1
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
    assert "gpt-sensitive-model-id" not in serialized_events
    assert plugins.get_plugin_manager().list_plugins() == []

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    packages = list((root / "outbox").glob("*.json"))
    assert len(packages) == 1
    package = json.loads(packages[0].read_text(encoding="utf-8"))
    assert package["metrics"][0]["name"] == "hermes.model_call.count"
    assert package["metrics"][0]["dimensions"]["model_family"] == "claude"
    assert package["metrics"][0]["value"] == 1


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
        "nemo_relay_scope_role": "subagent",
    }
    assert "sensitive-child" not in json.dumps(pushes)
    assert "sensitive-subagent" not in json.dumps(pushes)


def test_core_runtime_ignores_self_parenting_subagent_event(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None

    runtime.register_subagent(
        {"parent_session_id": "same", "child_session_id": "same"}
    )
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


def test_persistence_failure_does_not_escape_the_hook(
    direct_runtime,
    monkeypatch,
    caplog,
):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None

    def fail_record(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("store unavailable")

    monkeypatch.setattr(runtime.subscriber.store, "record_model_call", fail_record)
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

    assert "Unable to persist the Hermes model-call metric" in caplog.text


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
