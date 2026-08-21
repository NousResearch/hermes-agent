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

    def fake_resolve_runtime_provider(*, requested=None, explicit_base_url=None, explicit_api_key=None):
        if requested in {None, "", "openai-codex"}:
            from hermes_cli.auth import AuthError
            raise AuthError("No Codex credentials stored. Run `hermes auth` to authenticate.")
        assert requested == "openrouter"
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


def test_session_override_threads_named_requested_provider(monkeypatch):
    """A per-session /model override to a named provider must expose that
    provider's *named* identity via ``requested_provider`` in the resolved
    runtime, so a shared-endpoint sibling declared globally cannot lend its
    exact context metadata to this session (High review finding on #89714)."""
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:c1"
    runner._session_model_overrides[session_key] = {
        "model": "vllm/DeepSeek-V4-Flash-0731",
        "provider": "beta",
        "requested_provider": "beta",
        "api_key": "sk-live",
        "base_url": "http://da-aihost01:4000/v1",
        "api_mode": "chat_completions",
    }
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda cfg=None: "global-model")
    monkeypatch.setattr(gateway_run, "_credential_pool_for_provider", lambda _p: None)

    model, runtime_kwargs = runner._resolve_session_agent_runtime(
        session_key=session_key,
        user_config={"model": {"default": "global-model", "provider": "alpha"}},
    )

    assert model == "vllm/DeepSeek-V4-Flash-0731"
    assert runtime_kwargs.get("provider") == "beta"
    assert runtime_kwargs.get("requested_provider") == "beta"


def test_apply_session_model_override_threads_requested_provider():
    """The credential-less override merge path must also carry the named
    identity onto the runtime kwargs it patches."""
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:c2"
    runner._session_model_overrides[session_key] = {
        "model": "vllm/DeepSeek-V4-Flash-0731",
        "provider": "beta",
        "requested_provider": "beta",
        "base_url": "http://da-aihost01:4000/v1",
    }

    model, runtime_kwargs = runner._apply_session_model_override(
        session_key, "old-model", {"provider": "alpha", "requested_provider": "alpha"}
    )

    assert model == "vllm/DeepSeek-V4-Flash-0731"
    assert runtime_kwargs.get("provider") == "beta"
    assert runtime_kwargs.get("requested_provider") == "beta"


def test_apply_session_model_override_replaces_stale_requested_provider():
    """A legacy override (provider only, no requested_provider) must not leave a
    stale global requested_provider paired with the new provider — the
    override's provider is authoritative for identity."""
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:c3"
    runner._session_model_overrides[session_key] = {
        "model": "vllm/DeepSeek-V4-Flash-0731",
        "provider": "beta",
        "base_url": "http://da-aihost01:4000/v1",
    }

    model, runtime_kwargs = runner._apply_session_model_override(
        session_key, "old-model", {"provider": "alpha", "requested_provider": "alpha"}
    )

    assert runtime_kwargs.get("provider") == "beta"
    assert runtime_kwargs.get("requested_provider") == "beta"


