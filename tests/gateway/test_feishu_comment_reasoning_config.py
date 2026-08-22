"""Feishu doc-comment agents must honor configured reasoning settings (#85153).

``feishu_comment._resolve_model_and_runtime`` resolves its model through the
gateway config chain (``_load_gateway_config`` / ``_resolve_gateway_model``) —
its docstring says "same as gateway message handling" — but the gateway
resolves reasoning through ``resolve_reasoning_config`` and this path did not.
``_resolve_runtime_agent_kwargs()`` returns credentials only, so the
``AIAgent`` built at ``_run_comment_agent`` fell through to the provider
default: ``agent.reasoning_effort: none`` was ignored, and a non-reasoning
model then rejects the request outright.

Same defect class as the ACP and oneshot construction sites fixed alongside.

These tests exercise the real config-file → ``_load_gateway_config`` →
``resolve_reasoning_config`` chain against the per-test ``HERMES_HOME``; only
credential resolution is stubbed.
"""

import os
from pathlib import Path

import pytest
import yaml

from plugins.platforms.feishu import feishu_comment


@pytest.fixture
def feishu_env(monkeypatch):
    """Stub credential resolution only; return a config writer."""
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-api",
            "api_mode": "chat_completions",
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
        },
    )

    def _write_config(cfg: dict) -> None:
        home = Path(os.environ["HERMES_HOME"])
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        # _load_gateway_config caches on the resolved config home.
        monkeypatch.setattr("gateway.run._gateway_config_home", lambda: home)

    return _write_config


def test_feishu_agent_receives_reasoning_effort_none(feishu_env):
    """``reasoning_effort: none`` must disable thinking, not fall through."""
    feishu_env(
        {
            "model": {"default": "gpt-4o-mini", "provider": "openai-api"},
            "agent": {"reasoning_effort": "none"},
        }
    )

    _model, runtime_kwargs = feishu_comment._resolve_model_and_runtime()

    assert runtime_kwargs["reasoning_config"] == {"enabled": False}


def test_feishu_agent_receives_global_effort_level(feishu_env):
    """A configured effort level reaches the agent."""
    feishu_env(
        {
            "model": {"default": "gpt-5", "provider": "openai-api"},
            "agent": {"reasoning_effort": "high"},
        }
    )

    _model, runtime_kwargs = feishu_comment._resolve_model_and_runtime()

    assert runtime_kwargs["reasoning_config"]["effort"] == "high"


def test_feishu_per_model_override_keys_off_resolved_model(feishu_env):
    """Overrides resolve against the model this run uses, not a bare default."""
    feishu_env(
        {
            "model": {"default": "quiet-model", "provider": "openai-api"},
            "agent": {
                "reasoning_effort": "high",
                "reasoning_overrides": {"quiet-model": "none"},
            },
        }
    )

    model, runtime_kwargs = feishu_comment._resolve_model_and_runtime()

    assert model == "quiet-model"
    assert runtime_kwargs["reasoning_config"] == {"enabled": False}


def test_feishu_unset_reasoning_keeps_provider_default(feishu_env):
    """Unset config stays None — no eager default is invented."""
    feishu_env({"model": {"default": "gpt-5", "provider": "openai-api"}})

    _model, runtime_kwargs = feishu_comment._resolve_model_and_runtime()

    assert runtime_kwargs["reasoning_config"] is None
