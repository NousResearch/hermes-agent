"""Tests for hermes -z memory provider cleanup (#60616)."""

import sys
import types
import pytest


def test_oneshot_shutdown_memory_provider_on_return(monkeypatch):
    """Shutdown should be called even when run_conversation returns normally."""
    from hermes_cli.oneshot import _run_agent

    cleanup_calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self._session_messages = [{"role": "user", "content": "test"}]

        def run_conversation(self, prompt, **_kwargs):
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self, messages=None):
            cleanup_calls.append(messages)

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    text, result = _run_agent("hello")
    assert text == "ok"
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0] == [{"role": "user", "content": "test"}]


def test_oneshot_shutdown_memory_provider_on_exception(monkeypatch):
    """Shutdown should be called even when run_conversation raises an exception."""
    from hermes_cli.oneshot import _run_agent

    cleanup_calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self._session_messages = [{"role": "user", "content": "test"}]

        def run_conversation(self, prompt, **_kwargs):
            raise ValueError("boom")

        def shutdown_memory_provider(self, messages=None):
            cleanup_calls.append(messages)

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    with pytest.raises(ValueError, match="boom"):
        _run_agent("hello")

    assert len(cleanup_calls) == 1
    assert cleanup_calls[0] == [{"role": "user", "content": "test"}]


def test_oneshot_shutdown_memory_provider_fallback_when_no_session_messages(monkeypatch):
    """Fallback to no-arg shutdown when _session_messages is missing or not a list."""
    from hermes_cli.oneshot import _run_agent

    cleanup_calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            # Don't set _session_messages at all
            pass

        def run_conversation(self, prompt, **_kwargs):
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self, messages=None):
            cleanup_calls.append(messages)

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: set()),
    )

    text, result = _run_agent("hello")
    assert text == "ok"
    assert len(cleanup_calls) == 1
    # When _session_messages is missing, fallback to no-arg call
    assert cleanup_calls[0] is None