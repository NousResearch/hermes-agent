"""Policy-aware startup fallback coverage for ``hermes -z``."""

from __future__ import annotations

import sys
import types

import pytest

from hermes_cli.auth import AuthError
import hermes_cli.oneshot as oneshot


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_oneshot_stubs(monkeypatch, *, config, resolve_runtime):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: config),
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
            resolve_runtime_provider=resolve_runtime,
            format_runtime_provider_error=lambda exc: f"formatted: {exc}",
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
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    return captured


def test_oneshot_auth_failure_uses_policy_eligible_fallback(monkeypatch):
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AuthError("primary token expired")
        return {
            "api_key": kwargs.get("explicit_api_key"),
            "base_url": kwargs.get("explicit_base_url"),
            "provider": "custom",
            "api_mode": "chat_completions",
            "credential_pool": None,
        }

    fallback = {
        "provider": "custom",
        "model": "local-draft",
        "base_url": "http://127.0.0.1:8000/v1",
        "key_env": "ONESHOT_BACKUP_KEY",
    }
    monkeypatch.setenv("ONESHOT_BACKUP_KEY", "local-secret")
    captured = _install_oneshot_stubs(
        monkeypatch,
        config={
            "model": {"default": "primary-model", "provider": "primary"},
            "fallback_policy": "any",
            "fallback_providers": [fallback],
        },
        resolve_runtime=resolve_runtime_provider,
    )

    text, result = oneshot._run_agent("keep going")

    assert text == "ok"
    assert result["failed"] is False
    assert calls[1] == {
        "requested": "custom",
        "target_model": "local-draft",
        "explicit_base_url": "http://127.0.0.1:8000/v1",
        "explicit_api_key": "local-secret",
    }
    assert captured["model"] == "local-draft"
    assert captured["fallback_model"] == [fallback]
    assert captured["fallback_chain_from_config"] is True
    assert captured["initial_fallback_entry"] == fallback
    assert "Fallback policy any" in captured["initial_fallback_decision"]
    assert "primary token expired" in captured["initial_fallback_decision"]
    assert "local-draft via custom" in captured["initial_fallback_decision"]
    assert captured["prompt"] == "keep going"


def test_oneshot_policy_off_fails_before_attempting_configured_fallback(monkeypatch):
    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs)
        raise AuthError("primary token expired")

    _install_oneshot_stubs(
        monkeypatch,
        config={
            "model": {"default": "primary-model", "provider": "primary"},
            "fallback_policy": "off",
            "fallback_providers": [
                {
                    "provider": "custom",
                    "model": "must-not-run",
                    "base_url": "http://127.0.0.1:8000/v1",
                }
            ],
        },
        resolve_runtime=resolve_runtime_provider,
    )

    with pytest.raises(RuntimeError) as exc_info:
        oneshot._run_agent("keep going")

    assert len(calls) == 1
    assert "formatted: primary token expired" in str(exc_info.value)
    assert "Fallback policy off: no backup provider was attempted." in str(
        exc_info.value
    )
