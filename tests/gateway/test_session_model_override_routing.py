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


def test_run_agent_prefers_session_override_over_global_runtime(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _explode_runtime_resolution)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )
    session_key = "agent:main:local:dm"
    runner._session_model_overrides[session_key] = _codex_override()
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key=session_key,
        )
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "gpt-5.4"
    assert _CapturingAgent.last_init["provider"] == "openai-codex"
    assert _CapturingAgent.last_init["api_mode"] == "codex_responses"
    assert _CapturingAgent.last_init["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert _CapturingAgent.last_init["api_key"] == "***"
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "high"}


@pytest.mark.asyncio
async def test_background_task_prefers_session_override_over_global_runtime(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", _explode_runtime_resolution)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    _CapturingAgent.last_init = None
    runner = _make_runner()

    adapter = AsyncMock()
    adapter.send = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "ok"))
    adapter.extract_images = MagicMock(return_value=([], "ok"))
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    session_key = runner._session_key_for_source(source)
    runner._session_model_overrides[session_key] = _codex_override()
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

    await runner._run_background_task("say hello", source, "bg_test")

    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["model"] == "gpt-5.4"
    assert _CapturingAgent.last_init["provider"] == "openai-codex"
    assert _CapturingAgent.last_init["api_mode"] == "codex_responses"
    assert _CapturingAgent.last_init["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert _CapturingAgent.last_init["api_key"] == "***"
    assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "high"}


# ---------------------------------------------------------------------------
# credential_pool propagation (#16678)
#
# Without the override carrying credential_pool, /model switching to a new
# provider in the gateway leaves runtime_kwargs["credential_pool"] pointing
# at the *original* provider's pool — so a 429 on the new provider rotates
# (or fails to rotate) against the wrong provider's credentials, and the
# session falls through to the configured fallback model instead of
# rotating to the next credential on the new provider.
# ---------------------------------------------------------------------------


def test_apply_session_model_override_propagates_credential_pool():
    """An override-supplied credential_pool overwrites the global one."""
    runner = _make_runner()
    session_key = "agent:main:local:dm"

    new_pool = MagicMock(name="new_provider_pool")
    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "credential_pool": new_pool,
    }

    old_pool = MagicMock(name="old_provider_pool")
    runtime_kwargs = {
        "api_key": "old-key",
        "base_url": "https://api.old.example",
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
        "credential_pool": old_pool,
    }

    model, kwargs = gateway_run.GatewayRunner._apply_session_model_override(
        runner, session_key, "claude-sonnet", runtime_kwargs
    )

    assert model == "gpt-5.4"
    assert kwargs["provider"] == "openai-codex"
    # Critical assertion — without the fix this would still be ``old_pool``.
    assert kwargs["credential_pool"] is new_pool


def test_apply_session_model_override_replaces_pool_with_none():
    """An explicit ``credential_pool: None`` clears the prior pool.

    A switched-to provider may legitimately have no pool (e.g. only an env
    var key configured).  Skipping ``None`` would leave the original
    provider's pool live and rotate against the wrong credentials on 429.
    """
    runner = _make_runner()
    session_key = "agent:main:local:dm"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "credential_pool": None,
    }

    runtime_kwargs = {
        "api_key": "old-key",
        "base_url": "https://api.old.example",
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
        "credential_pool": MagicMock(name="old_provider_pool"),
    }

    _model, kwargs = gateway_run.GatewayRunner._apply_session_model_override(
        runner, session_key, "claude-sonnet", runtime_kwargs
    )

    assert kwargs["credential_pool"] is None


def test_apply_session_model_override_omitting_pool_keeps_existing():
    """A pre-fix override (no ``credential_pool`` key) is left alone.

    Older callers (and persisted overrides, if any) may not carry the
    new key at all.  Those should keep the existing runtime pool — not
    blank it — so we don't regress sessions that never switched provider.
    """
    runner = _make_runner()
    session_key = "agent:main:local:dm"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4",
        "provider": "openai-codex",
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        # credential_pool intentionally omitted (legacy override shape)
    }

    existing_pool = MagicMock(name="existing_pool")
    runtime_kwargs = {
        "api_key": "old-key",
        "base_url": "https://api.old.example",
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
        "credential_pool": existing_pool,
    }

    _model, kwargs = gateway_run.GatewayRunner._apply_session_model_override(
        runner, session_key, "claude-sonnet", runtime_kwargs
    )

    assert kwargs["credential_pool"] is existing_pool
