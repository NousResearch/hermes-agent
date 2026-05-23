"""Cursor SDK provider integration tests."""

from __future__ import annotations

from types import SimpleNamespace


def test_cursor_sdk_profile_registered():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("cursor")
    assert profile is not None
    assert profile.name == "cursor-sdk"
    assert profile.api_mode == "cursor_sdk"
    assert profile.env_vars == ("CURSOR_API_KEY",)
    assert profile.supports_health_check is False
    assert "composer-latest" in profile.fallback_models


def test_cursor_sdk_transport_builds_kwargs():
    from agent.transports.cursor_sdk import CursorSdkTransport

    kwargs = CursorSdkTransport().build_kwargs(
        model="composer-latest",
        messages=[{"role": "user", "content": "ping"}],
        api_key="test-key",
        cwd="/tmp/project",
        runtime="local",
        timeout=12.5,
    )

    assert kwargs == {
        "__cursor_sdk__": True,
        "model": "composer-latest",
        "messages": [{"role": "user", "content": "ping"}],
        "api_key": "test-key",
        "cwd": "/tmp/project",
        "runtime": "local",
        "timeout": 12.5,
    }


def test_cursor_sdk_transport_normalizes_response():
    from agent.transports.cursor_sdk import CursorSdkTransport

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="hello", tool_calls=None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )

    normalized = CursorSdkTransport().normalize_response(response)
    assert normalized.content == "hello"
    assert normalized.tool_calls is None
    assert normalized.finish_reason == "stop"
    assert normalized.usage.total_tokens == 3


def test_cursor_sdk_adapter_uses_bridge_payload(monkeypatch, tmp_path):
    from agent import cursor_sdk_adapter

    captured = {}

    def fake_bridge(payload, **kwargs):
        captured["payload"] = payload
        captured["kwargs"] = kwargs
        if kwargs["on_delta"]:
            kwargs["on_delta"]("he")
            kwargs["on_delta"]("llo")
        return {"type": "final", "status": "finished", "result": "hello"}

    monkeypatch.setattr(cursor_sdk_adapter, "_run_bridge", fake_bridge)
    deltas = []

    response = cursor_sdk_adapter.create_cursor_sdk_response(
        api_key="test-key",
        model="composer-latest",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "ping"},
        ],
        cwd=str(tmp_path),
        runtime="local",
        timeout=9,
        stream=True,
        on_delta=deltas.append,
    )

    assert captured["payload"]["apiKey"] == "test-key"
    assert captured["payload"]["model"] == "composer-latest"
    assert captured["payload"]["cwd"] == str(tmp_path)
    assert captured["payload"]["runtime"] == "local"
    assert "System:" in captured["payload"]["prompt"]
    assert "User:" in captured["payload"]["prompt"]
    assert deltas == ["he", "llo"]
    assert response.choices[0].message.content == "hello"


def test_cursor_sdk_build_api_kwargs_path(monkeypatch, tmp_path):
    from agent.chat_completion_helpers import build_api_kwargs
    from agent.transports.cursor_sdk import CursorSdkTransport

    monkeypatch.setenv("CURSOR_SDK_CWD", str(tmp_path))
    agent = SimpleNamespace(
        api_mode="cursor_sdk",
        model="composer-latest",
        tools=[{"type": "function", "function": {"name": "ignored"}}],
        api_key="test-key",
        _resolved_api_call_timeout=lambda: 30,
        _get_transport=lambda: CursorSdkTransport(),
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "ping"}])
    assert kwargs["__cursor_sdk__"] is True
    assert kwargs["model"] == "composer-latest"
    assert kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert kwargs["api_key"] == "test-key"
    assert kwargs["cwd"] == str(tmp_path)


def test_runtime_provider_routes_cursor_sdk_to_cursor_api_mode(monkeypatch):
    import model_tools  # noqa: F401
    from hermes_cli import runtime_provider

    monkeypatch.setenv("CURSOR_API_KEY", "test-key")
    monkeypatch.setattr(
        runtime_provider,
        "_get_model_config",
        lambda: {"provider": "cursor-sdk", "default": "composer-latest"},
    )

    runtime = runtime_provider.resolve_runtime_provider(requested="cursor-sdk")
    assert runtime["provider"] == "cursor-sdk"
    assert runtime["api_mode"] == "cursor_sdk"
    assert runtime["base_url"] == "cursor-sdk://local"
    assert runtime["api_key"] == "test-key"
