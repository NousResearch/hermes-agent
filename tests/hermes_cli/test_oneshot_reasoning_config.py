"""``hermes -p`` (oneshot) must honor configured reasoning settings (#85153).

``hermes_cli.oneshot._run_agent`` builds its ``AIAgent`` from config like an
interactive CLI turn, but never passed ``reasoning_config`` — the same defect
class as the ACP path: ``agent.reasoning_effort: none`` was silently ignored
and non-reasoning models received an unsupported reasoning parameter.

These tests exercise the real config-file → ``load_config`` →
``resolve_reasoning_config`` chain against the per-test ``HERMES_HOME``; only
the ``AIAgent`` constructor and provider resolution are stubbed.
"""

import os
from pathlib import Path

import pytest
import yaml

from hermes_cli.oneshot import _run_agent


class _CapturingAgent:
    """Stand-in for AIAgent that records kwargs and satisfies cleanup."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).instances.append(self)

    def run_conversation(self, prompt, **_kwargs):
        return {"final_response": "ok", "completed": True}

    def shutdown_memory_provider(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


@pytest.fixture
def oneshot_env(monkeypatch):
    """Stub the heavyweight edges of ``_run_agent``; return a config writer."""
    _CapturingAgent.instances = []
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kwargs: {
            "provider": requested or "openai-api",
            "api_mode": "chat_completions",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.detect_provider_for_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )

    def _write_config(cfg: dict) -> None:
        home = Path(os.environ["HERMES_HOME"])
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return _write_config


def _captured_agent() -> _CapturingAgent:
    assert len(_CapturingAgent.instances) == 1
    return _CapturingAgent.instances[0]


def test_oneshot_agent_receives_reasoning_effort_none(oneshot_env):
    """``reasoning_effort: none`` must reach the oneshot agent as disabled."""
    oneshot_env(
        {
            "model": {"default": "gpt-4o-mini", "provider": "openai-api"},
            "agent": {"reasoning_effort": "none"},
        }
    )

    response, result = _run_agent("say hi")

    assert response == "ok"
    assert _captured_agent().kwargs["reasoning_config"] == {"enabled": False}


def test_oneshot_agent_receives_global_effort_level(oneshot_env):
    """A global effort level propagates to oneshot like an interactive turn."""
    oneshot_env(
        {
            "model": {"default": "deepseek/deepseek-v4", "provider": "openrouter"},
            "agent": {"reasoning_effort": "high"},
        }
    )

    _run_agent("say hi")

    assert _captured_agent().kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }


def test_oneshot_per_model_override_keys_off_explicit_model(oneshot_env):
    """Overrides resolve against the model the run actually uses (--model)."""
    oneshot_env(
        {
            "model": {"default": "default-model", "provider": "openai-api"},
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {"cli-model": "none"},
            },
        }
    )

    _run_agent("say hi", model="cli-model")

    agent = _captured_agent()
    assert agent.kwargs["model"] == "cli-model"
    assert agent.kwargs["reasoning_config"] == {"enabled": False}


def test_oneshot_unset_reasoning_keeps_provider_default(oneshot_env):
    """No configured reasoning → None, so the provider default still applies."""
    oneshot_env({"model": {"default": "gpt-4o-mini", "provider": "openai-api"}})

    _run_agent("say hi")

    assert _captured_agent().kwargs.get("reasoning_config") is None
