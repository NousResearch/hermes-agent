"""Regression tests for the UnboundLocalError in _resolve_session_agent_runtime.

The bug: ``model`` was read (in the per-platform override block and in the
session-override block) before its first assignment in the same method.  Because
``model`` is assigned later in the method, Python treats it as function-local for
the whole scope, so any read before the first assignment raised
``UnboundLocalError``.  The caller's broad ``except Exception`` then mislabeled
the failure as "Provider authentication failed" (e.g. ``/model qwen3.8:27b`` on
a platform with no ``model_platforms`` entry).

The fix initializes ``model = _resolve_gateway_model(user_config)`` at the top of
the method, before any override is read.
"""

from __future__ import annotations

import threading
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_runner():
    """Bare GatewayRunner with the attributes _resolve_session_agent_runtime touches."""
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


def _telegram_source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        user_id="u1",
        chat_type="dm",
    )


def _set_session_override(runner, session_key, override):
    """Persist a /model session override on the runner's session state."""
    runner._session_state(session_key).conversation.model_override = dict(override)


def test_session_override_without_platform_entry_does_not_raise(monkeypatch):
    """Scenario 1: session override + platform with NO model_platforms entry.

    This is the exact combination that triggered the bug: the per-platform
    override block is skipped (no entry for telegram), then the session-override
    block reads ``model`` before it was ever assigned -> UnboundLocalError.
    """
    runner = _make_runner()
    session_key = "agent:main:telegram:dm:123"
    override = {
        "model": "qwen3.8:27b",
        "provider": "custom:ollama-cte700air",
        "api_key": "sk-test",
        "base_url": "http://cte700air:11434/v1",
        "api_mode": "chat_completions",
    }
    _set_session_override(runner, session_key, override)

    # Platform telegram has NO entry in model_platforms (only email).
    user_config = {
        "model": {"default": "gemma4:e4b", "provider": "ollama"},
        "model_platforms": {"email": "qwen3.5:9b"},
    }

    # Must not raise UnboundLocalError.
    model, runtime = runner._resolve_session_agent_runtime(
        source=_telegram_source(),
        session_key=session_key,
        user_config=user_config,
    )

    assert model == "qwen3.8:27b"
    assert runtime.get("provider") == "custom:ollama-cte700air"
    assert runtime.get("api_key") == "sk-test"


def test_platform_override_without_default_model_does_not_raise(monkeypatch):
    """Scenario 2: platform WITH a model_platforms entry (second bug branch).

    Here the per-platform override block assigns ``model``, but the logger.info
    on that branch reads ``model`` before the assignment — the other way the
    UnboundLocalError could fire.
    """
    runner = _make_runner()
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "ollama", "api_key": "sk-ollama", "base_url": None},
    )
    user_config = {
        "model": {"default": "gemma4:e4b", "provider": "ollama"},
        "model_platforms": {"telegram": "qwen3.5:9b"},
    }

    model, runtime = runner._resolve_session_agent_runtime(
        source=_telegram_source(),
        session_key="agent:main:telegram:dm:123",
        user_config=user_config,
    )

    assert model == "qwen3.5:9b"
    assert runtime.get("provider") == "ollama"


def test_no_override_uses_default_model(monkeypatch):
    """Scenario 3: no override and no platform (regression for the default path)."""
    runner = _make_runner()
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "gemma4:e4b",
    )
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "ollama", "api_key": "sk-ollama", "base_url": None},
    )

    model, runtime = runner._resolve_session_agent_runtime(
        session_key="agent:main:telegram:dm:123",
        user_config={"model": {"default": "gemma4:e4b", "provider": "ollama"}},
    )

    assert model == "gemma4:e4b"
    assert runtime.get("provider") == "ollama"
