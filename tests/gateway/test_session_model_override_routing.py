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


def test_no_override_scan_skipped_when_debug_disabled(monkeypatch):
    """Measured-work pin: the O(n) session scan is lazy.

    ``_resolve_session_agent_runtime``'s no-override branch previously
    evaluated ``[k for k, st in list(self._sessions_map().items()) ...]`` as
    an eager ``logger.debug`` argument on every call — even at WARNING level.
    With the fix, ``_sessions_map()`` must not be touched when debug logging
    is off, and must still be scanned (producing the override-key preview)
    when debug logging is on.
    """
    import logging

    from gateway.run import GatewayRunner

    scans = {"n": 0}

    def counting_sessions_map(self):
        scans["n"] += 1
        return {f"sess_{i}": MagicMock() for i in range(50)}

    # Resolve with no session key -> no-override branch. Stub the seams so
    # nothing else in the resolver touches real state.
    monkeypatch.setattr(GatewayRunner, "_sessions_map", counting_sessions_map)
    monkeypatch.setattr(GatewayRunner, "_session_key_for_source", lambda self, s: None)
    monkeypatch.setattr(GatewayRunner, "_peek_session_state", lambda self, k: None)
    monkeypatch.setattr(
        GatewayRunner, "_rehydrate_session_model_override", lambda self, k: None
    )
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    runner = object.__new__(GatewayRunner)
    runner.config = None

    gw_logger = logging.getLogger("gateway.run")

    # At WARNING level the scan must NOT run. setLevel() (not direct .level
    # assignment) is required: Logger.isEnabledFor consults an internal
    # level cache that only setLevel() invalidates.
    monkeypatch.setattr(gw_logger, "disabled", False)
    gw_logger.setLevel(logging.WARNING)
    GatewayRunner._resolve_session_agent_runtime(runner, source=None)
    assert scans["n"] == 0, "session scan ran at WARNING level"

    # At DEBUG level the scan runs and builds the override-key preview.
    gw_logger.setLevel(logging.DEBUG)
    GatewayRunner._resolve_session_agent_runtime(runner, source=None)
    assert scans["n"] == 1, "session scan did not run at DEBUG level"


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


