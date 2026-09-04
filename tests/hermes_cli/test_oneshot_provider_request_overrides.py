"""Regression coverage for named custom-provider request overrides in -z mode."""

from __future__ import annotations

import sys
import types


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_oneshot_passes_named_provider_request_overrides_and_reasoning_to_agent(monkeypatch):
    """The one-shot constructor must receive provider and CLI request controls."""
    from hermes_cli.oneshot import _run_agent

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, _prompt, **_kwargs):
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self, *_args, **_kwargs):
            return None

        def close(self):
            return None

    class FakeSessionDB:
        def close(self):
            return None

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_state",
        _module("hermes_state", SessionDB=FakeSessionDB),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module(
            "hermes_cli.config",
            load_config=lambda: {"model": {"default": "gpt-5.6-sol"}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module(
            "hermes_cli.models",
            detect_provider_for_model=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "test-key",
                "base_url": "https://gateway.example/v1",
                "provider": "custom",
                "requested_provider": "custom:gateway",
                "api_mode": "chat_completions",
                "credential_pool": None,
                "request_overrides": {
                    "extra_body": {
                        "reasoning": {
                            "enabled": True,
                            "effort": "medium",
                        }
                    }
                },
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_args, **_kwargs: set(),
        ),
    )

    text, result = _run_agent("hello", reasoning="max")

    assert text == "ok"
    assert not result.get("failed")
    assert captured["request_overrides"] == {
        "extra_body": {
            "reasoning": {
                "enabled": True,
                "effort": "medium",
            }
        }
    }
    assert captured["reasoning_config"] == {
        "enabled": True,
        "effort": "max",
    }
