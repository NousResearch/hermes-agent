"""Regression tests for session-scoped model/provider overrides in gateway agents.

These cover the bug where `/model ...` stored a session override, but fresh
agent constructions still resolved model/provider from global config/runtime.
That let helper agents (and cache-miss main agents) route GPT-5.4 to the wrong
provider, e.g. Nous instead of OpenAI Codex.
"""

import asyncio
import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = None
    runner.config = None
    runner._voice_mode = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._service_tier = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._pending_approvals = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    return runner


def _codex_override():
    return {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }


def _explode_runtime_resolution():
    raise AssertionError(
        "global runtime resolution should not run when a complete session override exists"
    )


def test_gateway_auth_fallback_uses_fallback_model_from_config(tmp_path, monkeypatch):
    """Regression: fallback provider must not inherit the primary model.

    If primary openai-codex auth fails and fallback_providers selects
    OpenRouter/minimax, the gateway must instantiate AIAgent with the fallback
    model, not the primary config model (e.g. gpt-5.5). Otherwise OpenRouter
    receives an unintended GPT request.
    """
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  default: gpt-5.5
  provider: openai-codex
fallback_providers:
  - provider: openrouter
    model: minimax/minimax-m2.7
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    def fake_resolve_runtime_provider(*, requested=None, explicit_base_url=None, explicit_api_key=None,
                                      target_model=None):
        if requested in {None, "", "openai-codex"}:
            from hermes_cli.auth import AuthError
            raise AuthError("No Codex credentials stored. Run `hermes auth` to authenticate.")
        assert requested == "openrouter"
        # The fallback rung is resolved against the model it will send, not the primary default.
        assert target_model == "minimax/minimax-m2.7"
        return {
            "api_key": "sk-openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    import hermes_cli.runtime_provider as runtime_provider

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", fake_resolve_runtime_provider)

    runner = _make_runner()
    model, runtime_kwargs = runner._resolve_session_agent_runtime(
        session_key="agent:main:telegram:group:-1003715515980:63",
        user_config={
            "model": {"default": "gpt-5.5", "provider": "openai-codex"},
            "fallback_providers": [{"provider": "openrouter", "model": "minimax/minimax-m2.7"}],
        },
    )

    assert model == "minimax/minimax-m2.7"
    assert runtime_kwargs["provider"] == "openrouter"
    assert runtime_kwargs["api_key"] == "sk-openrouter"


def test_gateway_expands_configured_direct_alias_before_creating_agent(monkeypatch):
    """A gateway turn must send a direct alias's wire model, not its short name."""
    runner = _make_runner()
    direct_runtime = {
        "provider": "openrouter", "api_key": "alias-key",
        "base_url": "https://openrouter.example/v1", "api_mode": "chat_completions",
    }
    monkeypatch.setattr(
        gateway_run, "_resolve_direct_alias_agent_runtime",
        lambda model: ("z-ai/glm-5.3-flash", dict(direct_runtime)) if model == "zdr-flash" else None,
    )
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs",
        lambda: {"provider": "fallback", "api_key": "fallback-key", "base_url": "https://fallback.example/v1"},
    )

    model, runtime = runner._resolve_session_agent_runtime(
        session_key="agent:sharik:signal:dm:test",
        user_config={"model": {"default": "zdr-flash"}},
    )

    assert model == "z-ai/glm-5.3-flash"
    assert runtime == direct_runtime


def test_gateway_direct_alias_uses_canonical_startup_route(monkeypatch):
    """Gateway aliases use the same resolver and credential boundary as other callers."""
    from hermes_cli import model_switch, runtime_provider

    route = model_switch.StartupModelRoute(
        model="z-ai/glm-5.3-flash",
        provider="openrouter",
        base_url="https://openrouter.example/v1",
        api_key="alias-key",
    )
    monkeypatch.setattr(model_switch, "_ensure_direct_aliases", lambda: None)
    monkeypatch.setitem(
        model_switch.DIRECT_ALIASES, "zdr-flash",
        model_switch.DirectAlias("z-ai/glm-5.3-flash", "openrouter", "https://openrouter.example/v1"),
    )
    monkeypatch.setattr(
        model_switch, "resolve_startup_model_route",
        lambda model: route if model == "zdr-flash" else None,
    )
    seen = {}

    def resolve_runtime_provider(**kwargs):
        seen.update(kwargs)
        return {
            "provider": "openrouter", "api_key": "alias-key",
            "base_url": "https://openrouter.example/v1", "api_mode": "chat_completions",
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve_runtime_provider)

    model, runtime = gateway_run._resolve_direct_alias_agent_runtime("zdr-flash")

    assert model == route.model
    assert seen == {
        "requested": route.provider,
        "explicit_base_url": route.base_url,
        "explicit_api_key": route.api_key,
        "target_model": route.model,
    }
    assert runtime["provider"] == route.provider


