"""Regression tests for reasoning configuration in ``hermes -z``."""

from types import SimpleNamespace

import pytest

import hermes_cli.main as main_mod
import hermes_cli.oneshot as oneshot_mod
from hermes_cli.oneshot import run_oneshot


class _CapturingAgent:
    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        self.suppress_status_output = False
        self.stream_delta_callback = None
        self.tool_gen_callback = None

    def run_conversation(self, prompt, conversation_history=None):
        return {"final_response": "done"}


def _wire_agent_stubs(monkeypatch, cfg):
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot_mod, "get_fallback_chain", lambda _cfg: None)
    monkeypatch.setattr(oneshot_mod, "_close_agent", lambda _agent, _db: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "test-key",
            "base_url": "https://api.example.test/v1",
            "provider": "deepseek",
            "requested_provider": "deepseek",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    _CapturingAgent.captured = {}


def _run_agent(monkeypatch, cfg, reasoning=None):
    _wire_agent_stubs(monkeypatch, cfg)
    response, _result = oneshot_mod._run_agent(
        "hello",
        model="deepseek-v4-flash",
        provider="deepseek",
        use_config_toolsets=False,
        reasoning=reasoning,
    )
    assert response == "done"
    return _CapturingAgent.captured["reasoning_config"]


def test_oneshot_agent_inherits_configured_reasoning(monkeypatch):
    cfg = {
        "model": {"default": "deepseek-v4-flash", "provider": "deepseek"},
        "agent": {"reasoning_effort": "high"},
    }

    assert _run_agent(monkeypatch, cfg) == {"enabled": True, "effort": "high"}


def test_explicit_reasoning_overrides_config(monkeypatch):
    cfg = {
        "model": {"default": "deepseek-v4-flash", "provider": "deepseek"},
        "agent": {"reasoning_effort": "low"},
    }

    assert _run_agent(monkeypatch, cfg, reasoning="max") == {
        "enabled": True,
        "effort": "max",
    }


def test_explicit_none_disables_reasoning(monkeypatch):
    cfg = {
        "model": {"default": "deepseek-v4-flash", "provider": "deepseek"},
        "agent": {"reasoning_effort": "high"},
    }

    assert _run_agent(monkeypatch, cfg, reasoning="none") == {"enabled": False}


def test_unconfigured_oneshot_keeps_reasoning_unset(monkeypatch):
    cfg = {
        "model": {"default": "deepseek-v4-flash", "provider": "deepseek"},
        "agent": {},
    }

    assert _run_agent(monkeypatch, cfg) is None


def test_config_override_uses_remapped_model(monkeypatch):
    from hermes_cli import model_switch as model_switch_mod
    from hermes_cli import models as models_mod

    cfg = {
        "model": {"default": "unused/model"},
        "agent": {
            "reasoning_effort": "low",
            "reasoning_overrides": {"real/remapped-model": "high"},
        },
    }
    _wire_agent_stubs(monkeypatch, cfg)
    monkeypatch.setattr(model_switch_mod, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setattr(model_switch_mod, "DIRECT_ALIASES", {})
    monkeypatch.setattr(
        models_mod,
        "detect_provider_for_model",
        lambda _model, _provider: ("deepseek", "real/remapped-model"),
    )

    response, _result = oneshot_mod._run_agent(
        "hello",
        model="requested-alias",
        use_config_toolsets=False,
    )

    assert response == "done"
    assert _CapturingAgent.captured["model"] == "real/remapped-model"
    assert _CapturingAgent.captured["reasoning_config"] == {
        "enabled": True,
        "effort": "high",
    }


def test_run_oneshot_forwards_reasoning(monkeypatch):
    captured = {}

    def _fake_run_agent(prompt, **kwargs):
        captured.update(kwargs, prompt=prompt)
        return "ok", {"final_response": "ok"}

    monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent)

    assert (
        run_oneshot(
            "hello",
            model="deepseek-v4-flash",
            provider="deepseek",
            reasoning="max",
        )
        == 0
    )
    assert captured["reasoning"] == "max"


def test_main_forwards_reasoning_to_oneshot(monkeypatch):
    captured = {}
    args = SimpleNamespace(
        oneshot="hello",
        model="deepseek-v4-flash",
        provider="deepseek",
        reasoning="max",
        toolsets=None,
        skills=None,
        usage_file=None,
        resume=None,
    )

    monkeypatch.setattr(
        main_mod, "_confirm_startup_expensive_model_override", lambda _args: None
    )
    monkeypatch.setattr(
        main_mod, "_resolve_chat_session_args", lambda _args, use_tui: None
    )
    monkeypatch.setattr(
        main_mod,
        "_run_and_exit_oneshot",
        lambda prompt, **kwargs: captured.update(kwargs, prompt=prompt),
    )

    main_mod._run_oneshot_from_args(args)

    assert captured["reasoning"] == "max"
