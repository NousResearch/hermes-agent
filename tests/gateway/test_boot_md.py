"""Tests for the built-in BOOT.md gateway hook."""

import sys
from types import SimpleNamespace

from gateway.builtin_hooks import boot_md


def test_resolve_boot_agent_kwargs_uses_config_runtime(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"default": "glm-5.1", "provider": "zai"},
            "fallback_providers": [
                {"provider": "openai-codex", "model": "gpt-5.5"},
                {"provider": "minimax", "model": "MiniMax-M2.7"},
            ],
        },
    )

    def fake_resolve_runtime_provider(*, requested=None, target_model=None, **_kwargs):
        calls.append({"requested": requested, "target_model": target_model})
        return {
            "api_key": "secret",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
            "provider": "zai",
            "api_mode": "chat_completions",
            "command": "agent",
            "args": ["run"],
            "credential_pool": object(),
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    kwargs = boot_md._resolve_boot_agent_kwargs()

    assert calls == [{"requested": "zai", "target_model": "glm-5.1"}]
    assert kwargs["model"] == "glm-5.1"
    assert kwargs["provider"] == "zai"
    assert kwargs["api_mode"] == "chat_completions"
    assert kwargs["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert kwargs["command"] == "agent"
    assert kwargs["args"] == ["run"]
    assert kwargs["fallback_model"] == [
        {"provider": "openai-codex", "model": "gpt-5.5"},
        {"provider": "minimax", "model": "MiniMax-M2.7"},
    ]


def test_resolve_boot_agent_kwargs_supports_legacy_fallback_model(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": "anthropic/claude-sonnet-4.5",
            "fallback_model": {"provider": "openrouter", "model": "openai/gpt-5.1"},
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {"provider": "openrouter", "api_mode": "chat_completions"},
    )

    kwargs = boot_md._resolve_boot_agent_kwargs()

    assert kwargs["model"] == "anthropic/claude-sonnet-4.5"
    assert kwargs["fallback_model"] == {
        "provider": "openrouter",
        "model": "openai/gpt-5.1",
    }


def test_resolve_boot_agent_kwargs_falls_back_to_defaults_on_error(monkeypatch):
    def fail_load_config():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr("hermes_cli.config.load_config", fail_load_config)

    assert boot_md._resolve_boot_agent_kwargs() == {}


def test_run_boot_agent_passes_resolved_runtime_kwargs(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            return {"final_response": "[SILENT]"}

    monkeypatch.setattr(
        boot_md,
        "_resolve_boot_agent_kwargs",
        lambda: {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "fallback_model": [{"provider": "minimax", "model": "MiniMax-M2.7"}],
        },
    )
    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        SimpleNamespace(AIAgent=FakeAgent),
    )

    boot_md._run_boot_agent("Check the system.")

    assert captured["kwargs"]["model"] == "gpt-5.5"
    assert captured["kwargs"]["provider"] == "openai-codex"
    assert captured["kwargs"]["fallback_model"] == [
        {"provider": "minimax", "model": "MiniMax-M2.7"}
    ]
    assert captured["kwargs"]["quiet_mode"] is True
    assert captured["kwargs"]["skip_context_files"] is True
    assert captured["kwargs"]["skip_memory"] is True
    assert "Check the system." in captured["prompt"]
