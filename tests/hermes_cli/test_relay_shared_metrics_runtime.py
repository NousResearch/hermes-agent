"""Tests for the direct Hermes-to-Relay shared-metrics runtime."""

from __future__ import annotations

import contextvars
import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_cli import lifecycle, plugins
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
        self._tool_starts: dict[Any, dict[str, Any]] = {}
        self._scope_starts: dict[Any, dict[str, Any]] = {}
        self._scope = contextvars.ContextVar("relay_scope", default=None)
        self._scope_stack = contextvars.ContextVar("relay_scope_stack", default=None)
        self._scope_serial = 0
        self.ScopeType = SimpleNamespace(
            Agent="agent", Function="function", Tool="tool"
        )
        self.LLMRequest = _Request
        self.scope = SimpleNamespace(
            push=self._scope_push,
            pop=self._scope_pop,
            event=self._scope_event,
        )
        self.llm = SimpleNamespace(call=self._llm_call, call_end=self._llm_call_end)
        self.tools = SimpleNamespace(call=self._tool_call, call_end=self._tool_call_end)
        self.subscribers = SimpleNamespace(
            register=self._register,
            deregister=self._deregister,
            flush=self._flush,
        )
        self.get_scope_stack = self._get_scope_stack

    def _scope_push(self, name: str, scope_type: Any, **kwargs: Any) -> Any:
        self._scope_serial += 1
        handle = ("scope", name, self._scope_serial)
        stack = self._scope_stack.get()
        if stack is None:
            stack = []
            self._scope_stack.set(stack)
        stack.append(handle)
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
        stack = self._scope_stack.get()
        if not stack or stack[-1] != handle:
            current = stack[-1] if stack else None
            self.events.append(("scope.pop.rejected", handle, current))
            raise RuntimeError("scope handle is not at the top of the stack")
        stack.pop()
        self._scope.set(stack[-1] if stack else None)
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
        event = SimpleNamespace(
            kind="mark",
            category=None,
            name=name,
            scope_category=None,
            category_profile=None,
            metadata=kwargs.get("metadata"),
            data=kwargs.get("data"),
        )
        for callback in list(self._callbacks.values()):
            callback(event)

    def _get_scope_stack(self) -> Any:
        stack = self._scope_stack.get()
        current = stack[-1] if stack else None
        self._scope.set(current)
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

    def _tool_call(
        self,
        name: str,
        args: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        handle = ("tool", name, len(self._tool_starts))
        self._tool_starts[handle] = kwargs
        self.events.append(("tool.call", name, args, kwargs))
        return handle

    def _tool_call_end(
        self,
        handle: Any,
        result: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        start = self._tool_starts.pop(handle)
        self.events.append(("tool.call_end", handle, result, kwargs))
        event = SimpleNamespace(
            kind="scope",
            category="tool",
            name=handle[1],
            scope_category="end",
            category_profile={},
            metadata={
                **start["metadata"],
                **kwargs["metadata"],
                "otel.status_code": "OK",
            },
            data=result,
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
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(plugins, "_plugin_manager", PluginManager())
    yield fake
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


@pytest.fixture
def real_binding_runtime(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": True}}},
    )
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()
    monkeypatch.setattr(plugins, "_plugin_manager", PluginManager())
    yield relay
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()


def test_direct_runtime_records_without_enabling_a_plugin(direct_runtime, tmp_path):
    base = {
        "session_id": "sensitive-session",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "api_request_id": "request-1",
        "platform": "cli",
        "provider": "custom",
        "model": "gpt-sensitive-model-id",
        "base_url": "http://127.0.0.1:11434/v1",
    }

    assert lifecycle.has_hook("pre_api_request")
    lifecycle.invoke_hook("on_session_start", **base)
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "pre_api_request",
        **base,
        request={"body": {"messages": ["sensitive-prompt"]}},
    )
    lifecycle.invoke_hook(
        "pre_tool_call",
        **base,
        tool_call_id="sensitive-tool-call",
        tool_name="terminal",
        toolset="terminal",
        args={"command": "sensitive-command"},
    )
    lifecycle.invoke_hook(
        "post_approval_response",
        turn_id=base["turn_id"],
        tool_call_id="sensitive-tool-call",
        choice="once",
        command="sensitive-command",
        description="sensitive-approval-description",
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="sensitive-tool-call",
        tool_name="terminal",
        toolset="terminal",
        args={"command": "sensitive-command"},
        result={"output": "sensitive-tool-result"},
        status="ok",
        duration_ms=275,
        retry_count=0,
    )
    lifecycle.invoke_hook(
        "api_request_error",
        **base,
        retryable=True,
        error={"message": "sensitive-error"},
    )
    lifecycle.invoke_hook(
        "pre_api_request",
        **{
            **base,
            "provider": "anthropic",
            "model": "claude-sonnet",
            "base_url": "https://api.anthropic.com",
        },
        request={"body": {"messages": ["sensitive-prompt"]}},
    )
    lifecycle.invoke_hook(
        "post_api_request",
        **{
            **base,
            "provider": "anthropic",
            "model": "claude-sonnet",
            "base_url": "https://api.anthropic.com",
        },
        response={"content": "sensitive-response"},
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id=base["session_id"])

    starts = [event for event in direct_runtime.events if event[0] == "llm.call"]
    ends = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    tool_starts = [event for event in direct_runtime.events if event[0] == "tool.call"]
    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
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
    assert len(tool_starts) == 1
    assert len(tool_ends) == 1
    assert tool_starts[0][1] == "hermes.tool_call"
    assert tool_starts[0][2] == {}
    assert tool_ends[0][2] == {
        "approval_outcome": "approved",
        "latency_bucket": "250ms_to_500ms",
        "outcome": "success",
        "retry_count_bucket": "0",
        "tool_category": "terminal",
    }
    assert starts[0][2] == {}
    assert starts[0][3]["model_name"] == "unknown"
    assert ends[0][2] == {
        "model": "claude-sonnet",
        "provider": "anthropic",
    }
    serialized_events = json.dumps(direct_runtime.events)
    assert "sensitive-prompt" not in serialized_events
    assert "sensitive-response" not in serialized_events
    assert "sensitive-error" not in serialized_events
    assert "sensitive-command" not in serialized_events
    assert "sensitive-tool-result" not in serialized_events
    assert "sensitive-tool-call" not in serialized_events
    assert "sensitive-approval-description" not in serialized_events
    assert "gpt-sensitive-model-id" not in serialized_events
    assert plugins.get_plugin_manager().list_plugins() == []

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    packages = list((root / "outbox").glob("*.json"))
    assert len(packages) == 1
    package = json.loads(packages[0].read_text(encoding="utf-8"))
    metrics = {metric["name"]: metric for metric in package["metrics"]}
    assert set(metrics) == {
        "hermes.model_route.count",
        "hermes.task_run.finished",
        "hermes.task_run.started",
        "hermes.tool_approval.count",
        "hermes.tool_call.count",
    }
    assert metrics["hermes.model_route.count"]["dimensions"] == {
        "model": "claude-sonnet",
        "provider": "anthropic",
    }
    assert metrics["hermes.model_route.count"]["value"] == 1
    assert metrics["hermes.tool_call.count"] == {
        "name": "hermes.tool_call.count",
        "type": "counter",
        "dimensions": {
            "approval_outcome": "approved",
            "latency_bucket": "250ms_to_500ms",
            "outcome": "success",
            "retry_count_bucket": "0",
            "tool_category": "terminal",
        },
        "value": 1,
    }
    assert metrics["hermes.tool_approval.count"] == {
        "name": "hermes.tool_approval.count",
        "type": "counter",
        "dimensions": {
            "attribution": "tool_call",
            "outcome": "approved",
        },
        "value": 1,
    }
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


def test_real_binding_drives_lifecycle_aggregation_export_and_snapshot(
    real_binding_runtime,
    tmp_path,
    monkeypatch,
):
    assert real_binding_runtime._native is not None
    prompt_canary = "real-relay-sensitive-prompt"
    response_canary = "real-relay-sensitive-response"
    model_canary = "gpt-real-relay-sensitive-model"
    tool_canary = "real-relay-sensitive-tool-result"

    def base(index: int) -> dict[str, Any]:
        return {
            "session_id": f"sensitive-session-{index}",
            "task_id": f"sensitive-task-{index}",
            "turn_id": f"sensitive-turn-{index}",
            "api_request_id": f"sensitive-request-{index}",
            "platform": "cli",
            "provider": "custom",
            "model": model_canary,
            "base_url": "http://127.0.0.1:11434/v1",
        }

    success = base(1)
    lifecycle.invoke_hook("on_session_start", **success)
    lifecycle.invoke_hook("pre_llm_call", **success, messages=[prompt_canary])
    lifecycle.invoke_hook("pre_api_request", **success, retry_count=0)
    lifecycle.invoke_hook(
        "api_request_error",
        **success,
        retry_count=0,
        retryable=True,
        error={"message": prompt_canary},
    )
    lifecycle.invoke_hook("pre_api_request", **success, retry_count=1)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **success,
        tool_call_id="sensitive-tool-call",
        tool_name="terminal",
        args={"command": prompt_canary},
    )
    lifecycle.invoke_hook(
        "post_approval_response",
        turn_id=success["turn_id"],
        tool_call_id="sensitive-tool-call",
        choice="session",
        command=prompt_canary,
        description="sensitive-approval-description",
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **success,
        tool_call_id="sensitive-tool-call",
        tool_name="terminal",
        args={"command": prompt_canary},
        result={"output": tool_canary},
        status="ok",
        duration_ms=125,
        retry_count=0,
    )
    lifecycle.invoke_hook(
        "post_api_request",
        **success,
        retry_count=1,
        response={"content": response_canary},
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **success,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id=success["session_id"])

    failed = base(2)
    lifecycle.invoke_hook("on_session_start", **failed)
    lifecycle.invoke_hook("pre_llm_call", **failed, messages=[prompt_canary])
    lifecycle.invoke_hook("pre_api_request", **failed, retry_count=0)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **failed,
        tool_call_id="sensitive-failed-tool-call",
        tool_name="read_file",
        args={"path": prompt_canary},
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **failed,
        tool_call_id="sensitive-failed-tool-call",
        tool_name="read_file",
        args={"path": prompt_canary},
        result={"error": tool_canary},
        status="error",
        duration_ms=750,
    )
    lifecycle.invoke_hook(
        "api_request_error",
        **failed,
        retry_count=0,
        retryable=False,
        error={"message": response_canary},
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **failed,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="system_aborted",
    )
    lifecycle.finalize_session(session_id=failed["session_id"])

    cancelled = base(3)
    lifecycle.invoke_hook("on_session_start", **cancelled)
    lifecycle.invoke_hook("pre_llm_call", **cancelled, messages=[prompt_canary])
    lifecycle.invoke_hook("pre_api_request", **cancelled, retry_count=0)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **cancelled,
        tool_call_id="sensitive-cancelled-tool-call",
        tool_name="browser_navigate",
        args={"url": prompt_canary},
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **cancelled,
        tool_call_id="sensitive-cancelled-tool-call",
        tool_name="browser_navigate",
        args={"url": prompt_canary},
        result={"error": tool_canary},
        status="cancelled",
        duration_ms=31_000,
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **cancelled,
        completed=False,
        failed=False,
        interrupted=True,
        turn_exit_reason="interrupted_by_user",
    )
    lifecycle.finalize_session(session_id=cancelled["session_id"])

    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    store = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics._utc_now",
        lambda: tomorrow,
    )
    assert len(store.create_and_export_package_if_due()) == 1
    snapshot = store.counter_snapshot()
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for counter in snapshot:
        by_metric.setdefault(counter["metric_name"], []).append(counter)

    assert len(by_metric["hermes.task_run.started"]) == 1
    assert by_metric["hermes.task_run.started"][0]["value"] == 3
    assert len(by_metric["hermes.model_route.count"]) == 1
    model_counter = by_metric["hermes.model_route.count"][0]
    assert model_counter["dimensions"] == {
        "model": model_canary,
        "provider": "custom",
    }
    assert model_counter["value"] == 3
    assert {
        counter["dimensions"]["outcome"]
        for counter in by_metric["hermes.tool_call.count"]
    } == {"success", "failed", "cancelled"}
    tool_by_outcome = {
        counter["dimensions"]["outcome"]: counter["dimensions"]
        for counter in by_metric["hermes.tool_call.count"]
    }
    assert tool_by_outcome["success"] == {
        "approval_outcome": "approved",
        "latency_bucket": "100ms_to_250ms",
        "outcome": "success",
        "retry_count_bucket": "0",
        "tool_category": "terminal",
    }
    assert tool_by_outcome["failed"] == {
        "approval_outcome": "not_required",
        "latency_bucket": "500ms_to_1s",
        "outcome": "failed",
        "retry_count_bucket": "unknown",
        "tool_category": "file",
    }
    assert tool_by_outcome["cancelled"] == {
        "approval_outcome": "not_required",
        "latency_bucket": "gte_30s",
        "outcome": "cancelled",
        "retry_count_bucket": "unknown",
        "tool_category": "browser",
    }
    assert len(by_metric["hermes.tool_approval.count"]) == 1
    approval_counter = by_metric["hermes.tool_approval.count"][0]
    assert approval_counter["dimensions"] == {
        "attribution": "tool_call",
        "outcome": "approved",
    }
    assert approval_counter["value"] == 1
    assert approval_counter["packaged_value"] == 1
    terminal_by_outcome = {
        counter["dimensions"]["outcome"]: counter
        for counter in by_metric["hermes.task_run.finished"]
    }
    assert set(terminal_by_outcome) == {"success", "failed", "cancelled"}
    assert terminal_by_outcome["success"]["dimensions"]["retry_count_bucket"] == "1"
    assert terminal_by_outcome["success"]["dimensions"]["tool_call_count_bucket"] == "1"
    assert terminal_by_outcome["failed"]["dimensions"]["end_reason"] == (
        "system_aborted"
    )
    assert terminal_by_outcome["cancelled"]["dimensions"]["termination"] == (
        "user_cancelled"
    )
    assert all(counter["packaged_value"] == counter["value"] for counter in snapshot)

    snapshot_values = {
        (
            counter["metric_name"],
            tuple(sorted(counter["dimensions"].items())),
        ): counter["value"]
        for counter in snapshot
    }
    package_values: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
    packages = sorted((root / "outbox").glob("*.json"))
    assert len(packages) == 2
    package_payloads = [
        json.loads(package.read_text(encoding="utf-8")) for package in packages
    ]
    for package in package_payloads:
        assert package["schema_version"] == "hermes.shared_metrics.v2"
        for metric in package["metrics"]:
            key = (metric["name"], tuple(sorted(metric["dimensions"].items())))
            package_values[key] = package_values.get(key, 0) + metric["value"]
    assert package_values == snapshot_values

    serialized_analytics = json.dumps({
        "snapshot": snapshot,
        "packages": package_payloads,
    })
    assert model_canary in serialized_analytics
    for canary in (
        prompt_canary,
        response_canary,
        tool_canary,
        "sensitive-session",
        "sensitive-task",
        "sensitive-request",
        "sensitive-tool-call",
        "sensitive-failed-tool-call",
        "sensitive-cancelled-tool-call",
        "sensitive-approval-description",
    ):
        assert canary not in serialized_analytics


def test_real_binding_correlates_plugin_approval_denial_to_tool_metric(
    real_binding_runtime,
    tmp_path,
    monkeypatch,
):
    from hermes_cli.observability.shared_metrics import SharedMetricsStore
    from tools import approval

    assert real_binding_runtime._native is not None
    base = {
        "session_id": "sensitive-session",
        "task_id": "sensitive-task",
        "turn_id": "sensitive-turn",
        "api_request_id": "sensitive-request",
        "platform": "cli",
    }

    def plugin_hook(hook_name: str, **kwargs: Any) -> list[dict[str, str]]:
        if hook_name == "pre_tool_call":
            return [{"action": "approve", "message": "sensitive-rule"}]
        return []

    monkeypatch.setattr(plugins, "invoke_hook", plugin_hook)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "is_approved", lambda *args: False)
    monkeypatch.setattr(approval, "get_current_session_key", lambda: "session-key")
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *args, **kwargs: "deny")

    lifecycle.invoke_hook("on_session_start", **base)
    lifecycle.invoke_hook("pre_llm_call", **base, messages=["sensitive-prompt"])
    block_message = plugins.resolve_pre_tool_block(
        "write_file",
        {"path": "sensitive-path"},
        task_id=base["task_id"],
        session_id=base["session_id"],
        turn_id=base["turn_id"],
        api_request_id=base["api_request_id"],
        tool_call_id="sensitive-tool-call",
    )
    assert block_message is not None
    assert "User denied" in block_message

    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="sensitive-tool-call",
        tool_name="write_file",
        args={"path": "sensitive-path"},
        result={"error": block_message},
        status="blocked",
        duration_ms=12,
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="approval_denied",
    )
    lifecycle.finalize_session(session_id=base["session_id"])

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    store = SharedMetricsStore(root / "metrics.sqlite3", root / "outbox")
    snapshot = store.counter_snapshot()
    tool_metrics = [
        counter
        for counter in snapshot
        if counter["metric_name"] == "hermes.tool_call.count"
    ]
    assert len(tool_metrics) == 1
    assert tool_metrics[0]["dimensions"] == {
        "approval_outcome": "denied",
        "latency_bucket": "lt_100ms",
        "outcome": "blocked",
        "retry_count_bucket": "unknown",
        "tool_category": "file",
    }
    approval_metrics = [
        counter
        for counter in snapshot
        if counter["metric_name"] == "hermes.tool_approval.count"
    ]
    assert len(approval_metrics) == 1
    assert approval_metrics[0]["dimensions"] == {
        "attribution": "tool_call",
        "outcome": "denied",
    }
    assert "sensitive" not in json.dumps(snapshot)


def test_real_binding_aggregates_tool_and_approval_timeouts(
    real_binding_runtime,
    tmp_path,
):
    from hermes_cli.observability.shared_metrics import SharedMetricsStore

    assert real_binding_runtime._native is not None
    base = {
        "session_id": "timeout-sensitive-session",
        "task_id": "timeout-sensitive-task",
        "turn_id": "timeout-sensitive-turn",
        "platform": "cli",
    }

    lifecycle.invoke_hook("on_session_start", **base)
    lifecycle.invoke_hook("pre_llm_call", **base, messages=["timeout-sensitive-prompt"])
    lifecycle.invoke_hook(
        "pre_tool_call",
        **base,
        tool_call_id="timeout-sensitive-tool-call",
        tool_name="terminal",
        args={"command": "timeout-sensitive-command"},
    )
    lifecycle.invoke_hook(
        "post_approval_response",
        **base,
        tool_call_id="timeout-sensitive-tool-call",
        choice="timeout",
        command="timeout-sensitive-command",
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="timeout-sensitive-tool-call",
        tool_name="terminal",
        result={"error": "timeout-sensitive-result"},
        status="timeout",
        duration_ms=30_000,
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="provider_timeout",
    )
    lifecycle.finalize_session(session_id=base["session_id"])

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    snapshot = SharedMetricsStore(
        root / "metrics.sqlite3",
        root / "outbox",
    ).counter_snapshot()
    [tool_metric] = [
        counter
        for counter in snapshot
        if counter["metric_name"] == "hermes.tool_call.count"
    ]
    assert tool_metric["dimensions"] == {
        "approval_outcome": "timed_out",
        "latency_bucket": "gte_30s",
        "outcome": "timed_out",
        "retry_count_bucket": "unknown",
        "tool_category": "terminal",
    }
    [approval_metric] = [
        counter
        for counter in snapshot
        if counter["metric_name"] == "hermes.tool_approval.count"
    ]
    assert approval_metric["dimensions"] == {
        "attribution": "tool_call",
        "outcome": "timed_out",
    }
    assert "timeout-sensitive" not in json.dumps(snapshot)




def test_execution_adapters_do_not_create_relay_host_without_a_consumer(
    monkeypatch,
):
    from agent import relay_llm, relay_tools

    relay_runtime._reset_for_tests()
    imports = []

    def load_relay():
        imports.append("nemo_relay")
        raise AssertionError("disabled execution adapter created Relay host")

    monkeypatch.setattr(relay_runtime, "_load_nemo_relay", load_relay)
    request = {"model": "test-model", "messages": []}
    response = object()
    tool_args = {"command": "true"}
    tool_result = object()

    assert (
        relay_llm.execute(
            request,
            lambda observed: response if observed is request else None,
            session_id="llm-session",
            name="test-provider",
            model_name="test-model",
        )
        is response
    )
    result, observed_args = relay_tools.execute(
        "terminal",
        tool_args,
        lambda observed: tool_result if observed is tool_args else None,
        session_id="tool-session",
    )

    assert result is tool_result
    assert observed_args is tool_args
    assert relay_runtime.get_host(create=False) is None
    assert imports == []






def test_core_runtime_is_fail_open_without_a_published_binding(monkeypatch, caplog):
    relay_shared_metrics._reset_for_tests()
    relay_runtime._reset_for_tests()

    def missing_relay(name: str):
        assert name == "nemo_relay"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(relay_runtime.importlib, "import_module", missing_relay)

    assert relay_runtime.get_runtime() is None
    host = relay_runtime.get_host()
    assert isinstance(host, relay_runtime.NoopRelayRuntime)
    assert host.profile_key == relay_runtime.current_profile_key()
    assert "nemo_relay" in host.reason
    assert host.apply_tool_request_intercepts(
        session_id="s1",
        tool_name="terminal",
        args={"command": "true"},
    ) == {"command": "true"}
    assert not relay_runtime.emit_mark("hermes.probe", session_id="s1")
    assert "Hermes Relay runtime initialization failed" in caplog.text
    relay_runtime._reset_for_tests()


def test_core_task_instrumentation_preserves_prompt_history_and_tool_schema(
    direct_runtime,
    monkeypatch,
):
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.session_id = "cache-stable-session"
    agent.platform = "cli"
    agent._parent_session_id = None
    agent._session_db = None
    agent._cached_system_prompt = "byte-stable-system-prompt\nwith exact spacing"
    agent.tools = [
        {
            "type": "function",
            "function": {
                "name": "probe",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
            },
        }
    ]
    history = [{"role": "user", "content": "sensitive-history-canary"}]
    prompt_before = agent._cached_system_prompt.encode("utf-8")
    history_before = json.dumps(history, ensure_ascii=False, sort_keys=True)
    tools_before = json.dumps(agent.tools, ensure_ascii=False, sort_keys=True)

    def fake_run_conversation(
        active_agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        **kwargs,
    ):
        del (
            user_message,
            system_message,
            task_id,
            stream_callback,
            persist_user_message,
            kwargs,
        )
        assert active_agent is agent
        assert conversation_history is history
        return {"final_response": "ok", "completed": True}

    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        fake_run_conversation,
    )

    for task_id in ("cache-task-1", "cache-task-2"):
        result = AIAgent.run_conversation(
            agent,
            "hello",
            conversation_history=history,
            task_id=task_id,
        )
        assert result["final_response"] == "ok"

    assert agent._cached_system_prompt.encode("utf-8") == prompt_before
    assert json.dumps(history, ensure_ascii=False, sort_keys=True) == history_before
    assert json.dumps(agent.tools, ensure_ascii=False, sort_keys=True) == tools_before


def test_skipped_turn_does_not_finish_another_sessions_matching_task(
    direct_runtime,
    monkeypatch,
):
    """A skipped turn must not use shared-metrics' task-id fallback on finish."""
    from run_agent import AIAgent

    owner_session = "instrumented-session"
    shared_task_id = "caller-supplied-task-id"
    relay_shared_metrics.start_task_run(
        session_id=owner_session,
        task_id=shared_task_id,
        platform="cli",
    )
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    assert (owner_session, shared_task_id) in runtime._task_sessions

    agent = object.__new__(AIAgent)
    agent.session_id = "skipped-session"
    agent.platform = "cli"
    agent._parent_session_id = None
    agent._session_db = None
    agent._cached_system_prompt = "stable"
    agent.tools = []

    skipped_turn = SimpleNamespace(relay_enabled=False)
    monkeypatch.setattr(
        relay_runtime.SESSION_COORDINATOR,
        "begin_turn",
        lambda *_args, **_kwargs: skipped_turn,
    )
    monkeypatch.setattr(
        relay_runtime.SESSION_COORDINATOR,
        "finish_logical_calls",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        relay_runtime.SESSION_COORDINATOR,
        "end_turn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "agent.conversation_loop.run_conversation",
        lambda *_args, **_kwargs: {"final_response": "ok", "completed": True},
    )

    result = AIAgent.run_conversation(
        agent,
        "hello",
        conversation_history=[],
        task_id=shared_task_id,
    )

    assert result["completed"] is True
    assert (owner_session, shared_task_id) in runtime._task_sessions
    assert not [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == relay_shared_metrics.TASK_SCOPE
    ]
    relay_shared_metrics.finish_task_run(
        session_id=owner_session,
        task_id=shared_task_id,
        platform="cli",
        result={"completed": True},
    )










@pytest.mark.parametrize(
    ("profile_enabled", "managed_enabled"),
    ((None, True), (False, True), (True, False)),
)
def test_managed_config_cannot_override_shared_metrics_consent(
    tmp_path,
    monkeypatch,
    profile_enabled,
    managed_enabled,
):
    from hermes_cli import config, managed_scope
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    profile = tmp_path / "profile"
    managed = tmp_path / "managed"
    profile.mkdir()
    managed.mkdir()
    profile_config = "{}\n"
    if profile_enabled is not None:
        profile_config = (
            "telemetry:\n"
            "  shared_metrics:\n"
            f"    enabled: {str(profile_enabled).lower()}\n"
        )
    (profile / "config.yaml").write_text(profile_config, encoding="utf-8")
    (managed / "config.yaml").write_text(
        "telemetry:\n"
        "  shared_metrics:\n"
        f"    enabled: {str(managed_enabled).lower()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    config._LOAD_CONFIG_CACHE.clear()
    config._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    token = set_hermes_home_override(profile)
    try:
        assert (
            config.load_config_readonly()["telemetry"]["shared_metrics"]["enabled"]
            is managed_enabled
        )
        assert relay_shared_metrics.enabled() is (profile_enabled is True)
    finally:
        reset_hermes_home_override(token)
        relay_shared_metrics._reset_for_tests()
        relay_runtime._reset_for_tests()
        managed_scope.invalidate_managed_cache()




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
        "hermes_cli.config.read_raw_config_readonly",
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
    runtime.subscriber(
        SimpleNamespace(
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
        )
    )
    assert runtime.subscriber.store.counter_snapshot() == counters_before_stale_event
    assert (
        runtime.start_task({
            "session_id": "session",
            "task_id": "stale-runtime-task",
            "platform": "cli",
        })
        is None
    )
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








def test_sync_session_runner_releases_lock_before_callback(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    session = runtime.ensure_session({"session_id": "sync-session"})
    assert session is not None
    acquired = threading.Event()
    contender = None

    def probe() -> Any:
        nonlocal contender

        def acquire_session_lock() -> None:
            with session.lock:
                acquired.set()

        contender = threading.Thread(target=acquire_session_lock)
        contender.start()
        assert acquired.wait(timeout=1)
        return direct_runtime._scope.get()

    result = runtime.run_in_session(session, probe)
    assert contender is not None
    contender.join(timeout=1)

    assert result == session.handle
    assert contender.is_alive() is False


def test_direct_runtime_fake_enforces_lifo_scope_contract(direct_runtime):
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    session = runtime.ensure_session({"session_id": "lifo-contract"})
    assert session is not None

    first = runtime.run_in_session(
        session,
        direct_runtime.scope.push,
        "first",
        direct_runtime.ScopeType.Function,
    )
    second = runtime.run_in_session(
        session,
        direct_runtime.scope.push,
        "second",
        direct_runtime.ScopeType.Function,
    )

    with pytest.raises(RuntimeError, match="not at the top"):
        runtime.run_in_session(session, direct_runtime.scope.pop, first)

    runtime.run_in_session(session, direct_runtime.scope.pop, second)
    runtime.run_in_session(session, direct_runtime.scope.pop, first)


def test_concurrent_turn_skips_relay_before_scope_stack_can_interleave(
    direct_runtime,
):
    coordinator = relay_runtime.SESSION_COORDINATOR
    profile_key = relay_runtime.current_profile_key()
    lease = coordinator.acquire_conversation(
        profile_key=profile_key,
        session_id="shared-session",
        platform="cli",
    )
    first = coordinator.begin_turn(lease, turn_id="first", task_id="first-task")
    second = coordinator.begin_turn(
        lease,
        turn_id="second",
        task_id="second-task",
    )

    assert first.relay_enabled is True
    assert first.handle is not None
    assert second.relay_enabled is False
    assert second.handle is None
    assert relay_runtime.resolve_execution_context("shared-session") == (
        None,
        None,
        None,
    )

    coordinator.end_turn(first, outcome="success")
    coordinator.end_turn(second, outcome="success")
    coordinator.release_conversation(lease)
    coordinator.finalize_conversation(
        profile_key=profile_key,
        session_id="shared-session",
    )

    turn_closes = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1] == first.handle
    ]
    assert len(turn_closes) == 1
    assert not [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop.rejected"
    ]


def test_concurrent_turn_skips_shared_metrics_scope_creation(direct_runtime):
    coordinator = relay_runtime.SESSION_COORDINATOR
    profile_key = relay_runtime.current_profile_key()
    lease = coordinator.acquire_conversation(
        profile_key=profile_key,
        session_id="shared-session",
        platform="cli",
    )
    first = coordinator.begin_turn(lease, turn_id="first", task_id="first-task")
    second = coordinator.begin_turn(lease, turn_id="second", task_id="second-task")

    relay_shared_metrics.observe_lifecycle(
        "pre_llm_call",
        session_id="shared-session",
        task_id="second-task",
        platform="cli",
    )
    relay_shared_metrics.observe_lifecycle(
        "pre_api_request",
        session_id="shared-session",
        task_id="second-task",
        api_request_id="second-request",
        platform="cli",
    )

    assert second.relay_enabled is False
    assert not [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == relay_shared_metrics.TASK_SCOPE
    ]

    coordinator.end_turn(first, outcome="success")
    coordinator.end_turn(second, outcome="success")
    coordinator.release_conversation(lease)


def test_skipped_turn_stays_gated_after_instrumented_turn_ends(direct_runtime):
    coordinator = relay_runtime.SESSION_COORDINATOR
    profile_key = relay_runtime.current_profile_key()
    lease = coordinator.acquire_conversation(
        profile_key=profile_key,
        session_id="shared-session",
        platform="cli",
    )
    first = coordinator.begin_turn(lease, turn_id="first", task_id="first-task")
    second = coordinator.begin_turn(lease, turn_id="second", task_id="second-task")
    inherited = contextvars.copy_context()

    coordinator.end_turn(first, outcome="success")

    assert relay_runtime.current_turn() is second
    assert inherited.run(relay_runtime.current_turn) is second
    assert not relay_runtime.relay_instrumentation_enabled()
    assert not inherited.run(relay_runtime.relay_instrumentation_enabled)
    assert relay_runtime.resolve_execution_context("shared-session") == (
        None,
        None,
        None,
    )

    relay_shared_metrics.observe_lifecycle(
        "pre_llm_call",
        session_id="shared-session",
        task_id="second-task",
        platform="cli",
    )
    inherited.run(
        relay_shared_metrics.observe_lifecycle,
        "pre_api_request",
        session_id="shared-session",
        task_id="second-task",
        api_request_id="second-request",
        platform="cli",
    )

    assert not [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push"
        and event[1]
        in {relay_shared_metrics.TASK_SCOPE, relay_shared_metrics.MODEL_CALL_SCOPE}
    ]

    coordinator.end_turn(second, outcome="success")
    assert relay_runtime.current_turn() is None
    assert inherited.run(relay_runtime.current_turn) is second
    assert not inherited.run(relay_runtime.relay_instrumentation_enabled)
    coordinator.release_conversation(lease)








@pytest.mark.parametrize(
    "terminal",
    ["return", "exception", "cancelled", "timeout"],
)
def test_subagent_agent_boundary_closes_its_own_scope(
    direct_runtime,
    monkeypatch,
    terminal,
):
    from run_agent import AIAgent

    coordinator = relay_runtime.SESSION_COORDINATOR
    profile_key = relay_runtime.current_profile_key()
    parent_lease = coordinator.acquire_conversation(
        profile_key=profile_key,
        session_id="parent",
        platform="cli",
    )
    parent_turn = coordinator.begin_turn(
        parent_lease,
        turn_id="parent-turn",
        task_id="parent-task",
    )
    child_agent = SimpleNamespace(
        session_id="child",
        platform="subagent",
        _parent_session_id="parent",
        _session_db=None,
        _conversation_root_id=lambda: "parent",
    )

    if terminal == "return":
        monkeypatch.setattr(
            "agent.conversation_loop.run_conversation",
            lambda *_args, **_kwargs: {
                "final_response": "done",
                "completed": True,
                "interrupted": False,
            },
        )
        AIAgent.run_conversation(child_agent, "private", task_id="child-task")
    elif terminal == "exception":

        def fail(*_args, **_kwargs):
            raise RuntimeError("child failed")

        monkeypatch.setattr("agent.conversation_loop.run_conversation", fail)
        with pytest.raises(RuntimeError, match="child failed"):
            AIAgent.run_conversation(child_agent, "private", task_id="child-task")
    elif terminal == "cancelled":

        def cancel(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("agent.conversation_loop.run_conversation", cancel)
        with pytest.raises(KeyboardInterrupt):
            AIAgent.run_conversation(child_agent, "private", task_id="child-task")
    else:

        def time_out(*_args, **_kwargs):
            raise TimeoutError("child timed out")

        monkeypatch.setattr("agent.conversation_loop.run_conversation", time_out)
        with pytest.raises(TimeoutError, match="child timed out"):
            AIAgent.run_conversation(child_agent, "private", task_id="child-task")

    runtime = relay_runtime.get_runtime(create=False)
    assert runtime is not None
    assert runtime.get_session("child") is None
    child_push = next(
        event
        for event in direct_runtime.events
        if event[0] == "scope.push"
        and event[1] == relay_runtime.SESSION_SCOPE
        and event[3]["metadata"].get("nemo_relay_scope_role") == "subagent"
    )
    assert child_push[3]["handle"] == parent_turn.handle
    child_closes = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == relay_runtime.SESSION_SCOPE
    ]
    assert len(child_closes) == 1
    assert relay_runtime.current_turn() is parent_turn

    coordinator.end_turn(parent_turn, outcome="success")
    coordinator.release_conversation(parent_lease)
    coordinator.finalize_conversation(
        profile_key=profile_key,
        session_id="parent",
    )
























def test_terminal_model_error_retains_the_failed_route(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "r1",
        "provider": "anthropic",
        "model": "claude-sonnet",
    }

    lifecycle.invoke_hook("pre_api_request", **base)
    lifecycle.invoke_hook(
        "api_request_error",
        **base,
        retryable=False,
        error={"message": "sensitive-error"},
    )
    assert not [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    runtime = relay_shared_metrics._get_runtime()
    session = runtime._session(base)
    assert session is not None
    [model_call] = session.model_calls.values()
    assert model_call.fields == {
        "model": "claude-sonnet",
        "provider": "anthropic",
    }
    lifecycle.finalize_session(session_id="s1")

    [end] = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    assert end[2] == {
        "model": "claude-sonnet",
        "provider": "anthropic",
    }


def test_nonretryable_provider_error_can_recover_within_one_logical_call(
    direct_runtime,
):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "r1",
        "provider": "anthropic",
        "model": "claude-sonnet",
    }

    lifecycle.invoke_hook("pre_api_request", **base, retry_count=0)
    lifecycle.invoke_hook(
        "api_request_error",
        **base,
        retry_count=0,
        retryable=False,
    )
    fallback = {
        **base,
        "provider": "openai-api",
        "model": "gpt-5",
    }
    lifecycle.invoke_hook("pre_api_request", **fallback, retry_count=0)
    lifecycle.invoke_hook(
        "post_api_request",
        **fallback,
        retry_count=0,
    )
    lifecycle.finalize_session(session_id="s1")

    [end] = [event for event in direct_runtime.events if event[0] == "llm.call_end"]
    [start] = [event for event in direct_runtime.events if event[0] == "llm.call"]
    assert start[3]["model_name"] == "unknown"
    assert end[2] == {
        "model": "gpt-5",
        "provider": "openai-api",
    }


def test_same_request_id_is_isolated_between_tasks(direct_runtime):
    common = {
        "session_id": "s1",
        "api_request_id": "shared-request",
        "platform": "cli",
        "provider": "anthropic",
        "model": "claude-sonnet",
    }
    for task_id in ("t1", "t2"):
        lifecycle.invoke_hook("pre_llm_call", **common, task_id=task_id)
        lifecycle.invoke_hook("pre_api_request", **common, task_id=task_id)

    lifecycle.invoke_hook("post_api_request", **common)
    assert not [event for event in direct_runtime.events if event[0] == "llm.call_end"]

    for task_id in ("t2", "t1"):
        lifecycle.invoke_hook("post_api_request", **common, task_id=task_id)
        lifecycle.invoke_hook(
            "on_session_end",
            **common,
            task_id=task_id,
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(stop)",
        )
    lifecycle.finalize_session(session_id="s1")

    model_ends = [
        event for event in direct_runtime.events if event[0] == "llm.call_end"
    ]
    assert len(model_ends) == 2
    task_ends = [
        event[2]["output"]
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert len(task_ends) == 2
    assert all(fields["model_call_count_bucket"] == "1" for fields in task_ends)
    assert all(fields["retry_count_bucket"] == "0" for fields in task_ends)


def test_reused_tool_call_id_is_counted_for_each_provider_request(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)

    for api_request_id in ("request-1", "request-2"):
        call = {
            **base,
            "api_request_id": api_request_id,
            "tool_call_id": "provider-reused-id",
            "tool_name": "terminal",
        }
        lifecycle.invoke_hook("pre_tool_call", **call, args={"command": "private"})
        lifecycle.invoke_hook(
            "post_tool_call",
            **call,
            result={"output": "private"},
            status="ok",
        )

    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="s1")

    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert len(tool_ends) == 2
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["tool_call_count_bucket"] == "2"


def test_partial_terminal_context_reuses_the_pending_tool_span(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "api_request_id": "request-1",
        "platform": "cli",
        "tool_call_id": "tool-1",
        "tool_name": "terminal",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook("pre_tool_call", **base)
    lifecycle.invoke_hook(
        "post_tool_call",
        **{key: value for key, value in base.items() if key != "api_request_id"},
        result={"output": "private"},
        status="ok",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="s1")

    [tool_end] = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert tool_end[2]["outcome"] == "success"
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["tool_call_count_bucket"] == "1"


def test_partial_terminal_variants_do_not_double_count_a_completed_call(
    direct_runtime,
):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "api_request_id": "request-1",
        "platform": "cli",
        "tool_call_id": "tool-1",
        "tool_name": "terminal",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook("pre_tool_call", **base)
    for omitted_field in ("api_request_id", "turn_id"):
        lifecycle.invoke_hook(
            "post_tool_call",
            **{key: value for key, value in base.items() if key != omitted_field},
            result={"output": "private"},
            status="ok",
        )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="s1")

    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert len(tool_ends) == 1
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["tool_call_count_bucket"] == "1"


def test_ambiguous_partial_terminal_does_not_create_a_phantom_tool_span(
    direct_runtime,
):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
        "tool_call_id": "provider-reused-id",
        "tool_name": "terminal",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    for api_request_id in ("request-1", "request-2"):
        lifecycle.invoke_hook(
            "pre_tool_call",
            **base,
            api_request_id=api_request_id,
        )

    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        result={"output": "ambiguous-private-result"},
        status="ok",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="system_aborted",
    )
    lifecycle.finalize_session(session_id="s1")

    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert len(tool_ends) == 2
    assert all(event[2]["outcome"] == "failed" for event in tool_ends)
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["tool_call_count_bucket"] == "2"


def test_reused_task_id_starts_a_new_run_for_each_turn(direct_runtime):
    for turn_id in ("turn-1", "turn-2"):
        base = {
            "session_id": "reused-session",
            "task_id": "reused-session",
            "turn_id": turn_id,
            "platform": "api",
        }
        lifecycle.invoke_hook("pre_llm_call", **base)
        lifecycle.invoke_hook(
            "post_tool_call",
            **base,
            api_request_id=f"request-{turn_id}",
            tool_call_id=f"tool-{turn_id}",
            tool_name="read_file",
            result={"output": "private"},
            status="ok",
        )
        lifecycle.invoke_hook(
            "on_session_end",
            **base,
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(stop)",
        )

    lifecycle.finalize_session(session_id="reused-session")

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
    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert len(task_starts) == 2
    assert len(task_ends) == 2
    assert len(tool_ends) == 2
    assert all(
        event[2]["output"]["tool_call_count_bucket"] == "1" for event in task_ends
    )


def test_late_tool_result_does_not_attach_to_reused_task_id(direct_runtime):
    first = {
        "session_id": "reused-session",
        "task_id": "reused-session",
        "turn_id": "turn-1",
        "platform": "api",
    }
    lifecycle.invoke_hook("pre_llm_call", **first)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **first,
        api_request_id="request-1",
        tool_call_id="tool-1",
        tool_name="terminal",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **first,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="timed_out",
    )

    second = {**first, "turn_id": "turn-2"}
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    assert runtime.start_task({
        "session_id": second["session_id"],
        "task_id": second["task_id"],
        "platform": second["platform"],
    })
    lifecycle.invoke_hook(
        "post_tool_call",
        **first,
        api_request_id="request-1",
        tool_call_id="tool-1",
        tool_name="terminal",
        result={"output": "late-private-result"},
        status="ok",
    )
    lifecycle.invoke_hook("pre_llm_call", **second)
    lifecycle.invoke_hook(
        "post_tool_call",
        **second,
        api_request_id="request-2",
        tool_call_id="tool-2",
        tool_name="read_file",
        result={"output": "current-private-result"},
        status="ok",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **second,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="reused-session")

    tool_ends = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert len(tool_ends) == 2
    assert [event[2]["outcome"] for event in tool_ends] == [
        "timed_out",
        "success",
    ]
    assert [event[2]["tool_category"] for event in tool_ends] == [
        "terminal",
        "file",
    ]
    task_ends = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert [
        event[2]["output"]["tool_call_count_bucket"] for event in task_ends
    ] == ["1", "1"]


def test_pending_tool_is_closed_and_counted_when_task_is_interrupted(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }

    lifecycle.invoke_hook("on_session_start", **base)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **base,
        tool_call_id="tool-1",
        tool_name="terminal",
        args={"command": "must-not-pass"},
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=False,
        interrupted=True,
        turn_exit_reason="interrupted_by_user",
    )
    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="tool-1",
        tool_name="terminal",
        result={"output": "late-result-must-not-pass"},
        status="ok",
    )
    lifecycle.finalize_session(session_id="s1")

    [tool_end] = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert tool_end[2] == {
        "approval_outcome": "not_required",
        "latency_bucket": tool_end[2]["latency_bucket"],
        "outcome": "cancelled",
        "retry_count_bucket": "unknown",
        "tool_category": "terminal",
    }
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["tool_call_count_bucket"] == "1"
    task_starts = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.push" and event[1] == "hermes.task_run"
    ]
    assert len(task_starts) == 1


def test_pending_tool_uses_the_outer_task_timeout_outcome(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "api",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "pre_tool_call",
        **base,
        tool_call_id="tool-1",
        tool_name="web_search",
    )

    relay_shared_metrics.finish_task_run(
        session_id="s1",
        task_id="t1",
        platform="api",
        error=TimeoutError("private timeout detail"),
    )
    lifecycle.finalize_session(session_id="s1")

    [tool_end] = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert tool_end[2]["outcome"] == "timed_out"
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["outcome"] == "timed_out"
    assert task_end[2]["output"]["tool_call_count_bucket"] == "1"
    assert "private timeout detail" not in repr(direct_runtime.events)


def test_late_model_start_does_not_create_an_orphan_after_task_completion(
    direct_runtime,
):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )

    lifecycle.invoke_hook(
        "pre_api_request",
        **base,
        api_request_id="late-request",
        provider="nvidia",
        model="nvidia/nemotron-3-super-120b-a12b",
    )
    lifecycle.finalize_session(session_id="s1")

    assert [event for event in direct_runtime.events if event[0] == "llm.call"] == []
    assert [
        event for event in direct_runtime.events if event[0] == "llm.call_end"
    ] == []


def test_approval_without_tool_context_is_counted_as_unattributed(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }

    lifecycle.invoke_hook("on_session_start", **base)
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "post_approval_response",
        turn_id="turn-1",
        choice="deny",
        command="must-not-pass",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="approval_denied",
    )
    lifecycle.finalize_session(session_id="s1")

    [approval] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.tool_approval"
    ]
    assert approval[2]["data"] == {
        "attribution": "unattributed",
        "outcome": "denied",
    }


def test_approval_with_unmatched_tool_id_is_counted_as_unattributed(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }

    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "post_approval_response",
        **base,
        tool_call_id="spoofed-tool-call",
        choice="deny",
        command="must-not-pass",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="approval_denied",
    )
    lifecycle.finalize_session(session_id="s1")

    [approval] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.tool_approval"
    ]
    assert approval[2]["data"] == {
        "attribution": "unattributed",
        "outcome": "denied",
    }


def test_tool_category_comes_from_runtime_registry_metadata(
    direct_runtime,
    monkeypatch,
):
    import model_tools

    monkeypatch.setattr(
        model_tools,
        "get_toolset_for_tool",
        lambda name: "terminal" if name == "runtime_only_tool" else None,
    )
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "turn_id": "turn-1",
        "platform": "cli",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "post_tool_call",
        **base,
        tool_call_id="tool-1",
        tool_name="runtime_only_tool",
        result={"output": "private"},
        status="ok",
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="s1")

    [tool_end] = [
        event for event in direct_runtime.events if event[0] == "tool.call_end"
    ]
    assert tool_end[2]["tool_category"] == "terminal"
def test_task_retry_count_survives_provider_fallback_ordinal_reset(direct_runtime):
    base = {
        "session_id": "s1",
        "task_id": "t1",
        "api_request_id": "r1",
        "platform": "cli",
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
    }

    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook("pre_api_request", **base, retry_count=0)
    lifecycle.invoke_hook(
        "api_request_error",
        **base,
        retry_count=0,
        retryable=True,
    )
    lifecycle.invoke_hook("pre_api_request", **base, retry_count=1)
    lifecycle.invoke_hook(
        "api_request_error",
        **base,
        retry_count=1,
        retryable=True,
    )
    lifecycle.invoke_hook(
        "pre_api_request",
        **{**base, "provider": "openai", "model": "gpt-5"},
        retry_count=0,
    )
    lifecycle.invoke_hook(
        "post_api_request",
        **{**base, "provider": "openai", "model": "gpt-5"},
        retry_count=0,
    )
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )
    lifecycle.finalize_session(session_id="s1")

    [model_end] = [
        event for event in direct_runtime.events if event[0] == "llm.call_end"
    ]
    assert model_end[2] == {
        "model": "gpt-5",
        "provider": "openai",
    }
    [task_end] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.pop" and event[1][1] == "hermes.task_run"
    ]
    assert task_end[2]["output"]["retry_count_bucket"] == "2"


def test_failed_flush_keeps_daily_export_open_for_later_task(
    direct_runtime, tmp_path, monkeypatch, caplog
):
    current_time = datetime(2026, 7, 28, 9, tzinfo=timezone.utc)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics._utc_now",
        lambda: current_time,
    )
    original_flush = direct_runtime.subscribers.flush
    flush_attempts = 0

    def fail_first_flush() -> None:
        nonlocal flush_attempts
        flush_attempts += 1
        if flush_attempts == 1:
            raise RuntimeError("simulated flush failure")
        original_flush()

    direct_runtime.subscribers.flush = fail_first_flush

    def finish_desktop_task(task_id: str) -> None:
        lifecycle.invoke_hook(
            "pre_llm_call",
            session_id="s1",
            task_id=task_id,
            platform="desktop",
        )
        lifecycle.invoke_hook(
            "on_session_end",
            session_id="s1",
            task_id=task_id,
            platform="desktop",
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="text_response(stop)",
        )

    finish_desktop_task("t1")

    root = tmp_path / "hermes-home" / "telemetry" / "shared_metrics"
    assert list((root / "outbox").glob("*.json")) == []
    with sqlite3.connect(root / "metrics.sqlite3") as connection:
        [package_count] = connection.execute(
            "SELECT COUNT(*) FROM package_outbox"
        ).fetchone()
    assert package_count == 0

    finish_desktop_task("t2")

    [package_path] = list((root / "outbox").glob("*.json"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    metrics = {metric["name"]: metric for metric in package["metrics"]}
    assert metrics["hermes.task_run.started"]["value"] == 2
    assert metrics["hermes.task_run.finished"]["value"] == 2
    assert flush_attempts == 2
    assert "Hermes shared-metrics task flush failed" in caplog.text


def test_skill_lifecycle_flows_through_relay_to_a_privacy_safe_package(
    direct_runtime,
    tmp_path,
):
    common = {
        "skill_name": "private-skill-name",
        "provenance": "agent_created",
    }
    lifecycle.invoke_hook("on_skill_lifecycle", **common, action="created")
    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **common,
        action="loaded",
        use_count=1,
        reused=False,
        reuse_after_patch=False,
    )
    lifecycle.invoke_hook("on_skill_lifecycle", **common, action="patched")
    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **common,
        action="loaded",
        use_count=2,
        reused=True,
        reuse_after_patch=True,
    )

    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    runtime.shutdown()

    marks = [event for event in direct_runtime.events if event[0] == "scope.event"]
    assert [event[1] for event in marks] == [
        "hermes.skill.lifecycle",
        "hermes.skill.load",
        "hermes.skill.lifecycle",
        "hermes.skill.load",
    ]
    assert "private-skill-name" not in json.dumps(marks)

    outbox = tmp_path / "hermes-home" / "telemetry" / "shared_metrics" / "outbox"
    [package_path] = list(outbox.glob("*.json"))
    package = json.loads(package_path.read_text(encoding="utf-8"))
    skill_metrics = [
        metric
        for metric in package["metrics"]
        if metric["name"].startswith("hermes.skill.")
    ]
    assert {metric["name"] for metric in skill_metrics} == {
        "hermes.skill.lifecycle.count",
        "hermes.skill.load.count",
    }
    assert "private-skill-name" not in json.dumps(package)


def test_skill_lifecycle_with_only_task_id_uses_unique_task_scope(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    task = runtime.start_task({
        "session_id": "session-1",
        "task_id": "task-1",
        "platform": "cli",
    })
    assert task is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="agent_created",
        task_id="task-1",
    )

    [mark] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ]
    assert mark[2]["handle"] == task.handle


def test_skill_task_only_correlation_does_not_guess_across_sessions(direct_runtime):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    for session_id in ("session-1", "session-2"):
        assert runtime.start_task({
            "session_id": session_id,
            "task_id": "shared-task",
            "platform": "cli",
        }) is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="agent_created",
        task_id="shared-task",
    )

    [mark] = [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ]
    assert "handle" not in mark[2]


def test_late_skill_lifecycle_is_not_reemitted_at_the_root(direct_runtime):
    base = {
        "session_id": "session-1",
        "task_id": "task-1",
        "turn_id": "turn-1",
        "platform": "cli",
    }
    lifecycle.invoke_hook("pre_llm_call", **base)
    lifecycle.invoke_hook(
        "on_session_end",
        **base,
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
    )

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        **base,
        action="loaded",
        skill_name="private-skill-name",
        provenance="local",
        use_count=2,
        reused=True,
        reuse_after_patch=False,
    )

    assert [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.load"
    ] == []


def test_skill_lifecycle_does_not_fallback_across_an_explicit_session(
    direct_runtime,
):
    runtime = relay_shared_metrics._get_runtime()
    assert runtime is not None
    assert runtime.start_task({
        "session_id": "session-1",
        "task_id": "task-1",
        "platform": "cli",
    }) is not None

    lifecycle.invoke_hook(
        "on_skill_lifecycle",
        action="created",
        skill_name="private-skill-name",
        provenance="local",
        session_id="wrong-session",
        task_id="task-1",
    )

    assert [
        event
        for event in direct_runtime.events
        if event[0] == "scope.event" and event[1] == "hermes.skill.lifecycle"
    ] == []
