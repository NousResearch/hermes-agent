"""Tests for the Hermes plugin middleware contract helpers."""

from __future__ import annotations

from hermes_cli.middleware import (
    apply_api_request_middleware,
    apply_tool_request_middleware,
    run_api_execution_middleware,
    run_tool_execution_middleware,
)


def test_tool_request_middleware_preserves_original_nested_payload(monkeypatch):
    def fake_invoke_middleware(kind, **kwargs):
        assert kind == "tool_request"
        assert kwargs["middleware_schema_version"] == "hermes.middleware.v1"
        next_args = kwargs["args"]
        next_args["nested"]["value"] = "mutated"
        return [{"args": next_args, "source": "test"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", fake_invoke_middleware)

    result = apply_tool_request_middleware(
        "demo_tool",
        {"nested": {"value": "original"}},
        task_id="task-1",
    )

    assert result.payload == {"nested": {"value": "mutated"}}
    assert result.original_payload == {"nested": {"value": "original"}}
    assert result.changed is True
    assert result.trace == [{"source": "test"}]


def test_api_request_middleware_preserves_original_nested_payload(monkeypatch):
    def fake_invoke_middleware(kind, **kwargs):
        assert kind == "api_request"
        next_request = kwargs["request"]
        next_request["body"]["metadata"] = {"trace": "added"}
        return [{"request": next_request, "source": "api-test"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_middleware", fake_invoke_middleware)

    result = apply_api_request_middleware({"body": {"messages": []}}, task_id="task-1")

    assert result.payload == {"body": {"messages": [], "metadata": {"trace": "added"}}}
    assert result.original_payload == {"body": {"messages": []}}
    assert result.changed is True
    assert result.trace == [{"source": "api-test"}]


def test_tool_execution_middleware_chains_in_order(monkeypatch):
    calls = []

    def first(**kwargs):
        calls.append(("first", kwargs["args"]))
        next_args = dict(kwargs["args"])
        next_args["first"] = True
        return kwargs["next_call"](next_args)

    def second(**kwargs):
        calls.append(("second", kwargs["args"]))
        next_args = dict(kwargs["args"])
        next_args["second"] = True
        return kwargs["next_call"](next_args)

    class Manager:
        _middleware = {"tool_execution": [first, second]}

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: Manager())

    result = run_tool_execution_middleware(
        "demo_tool",
        {"start": True},
        lambda args: {"final": args},
    )

    assert calls == [
        ("first", {"start": True}),
        ("second", {"start": True, "first": True}),
    ]
    assert result == {"final": {"start": True, "first": True, "second": True}}


def test_api_execution_middleware_fail_opens_to_next_callback(monkeypatch):
    def failing(**kwargs):
        raise RuntimeError("middleware failed")

    class Manager:
        _middleware = {"api_execution": [failing]}

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: Manager())

    result = run_api_execution_middleware(
        {"messages": []},
        lambda request: {"called": request},
    )

    assert result == {"called": {"messages": []}}
