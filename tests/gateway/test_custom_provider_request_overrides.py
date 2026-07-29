"""Regression tests for gateway preservation of provider-derived request_overrides.

Named custom providers can return request_overrides (for example
``extra_body.text.verbosity`` for OpenAI Responses). The gateway must preserve
those overrides on the runtime path and merge fast-mode overrides on top rather
than replacing them with an empty dict.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []
        self.request_overrides = dict(kwargs.get("request_overrides") or {})

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


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


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="ou_test",
        chat_type="dm",
        user_id="user-1",
        user_name="tester",
    )


def test_resolve_runtime_agent_kwargs_preserves_request_overrides(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda: {
            "api_key": "***",
            "base_url": "https://example.test/v1",
            "provider": "custom",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
            "credential_pool": None,
            "request_overrides": {
                "extra_body": {"text": {"verbosity": "low"}},
            },
        },
    )

    result = gateway_run._resolve_runtime_agent_kwargs()

    assert result["request_overrides"] == {
        "extra_body": {"text": {"verbosity": "low"}},
    }


def test_turn_route_preserves_provider_request_overrides_without_fast_mode():
    runner = _make_runner()
    runner._service_tier = None
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://example.test/v1",
        "provider": "custom",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": {
            "extra_body": {"text": {"verbosity": "low"}},
        },
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "hi",
        "gpt-5.4",
        runtime_kwargs,
    )

    assert route["request_overrides"] == {
        "extra_body": {"text": {"verbosity": "low"}},
    }


def test_turn_route_merges_fast_mode_with_provider_request_overrides():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://example.test/v1",
        "provider": "custom",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": {
            "extra_body": {"text": {"verbosity": "low"}},
        },
    }

    with patch(
        "hermes_cli.models.resolve_fast_mode_overrides",
        return_value={"service_tier": "priority"},
    ):
        route = gateway_run.GatewayRunner._resolve_turn_agent_config(
            runner,
            "hi",
            "gpt-5.4",
            runtime_kwargs,
        )

    assert route["request_overrides"] == {
        "extra_body": {"text": {"verbosity": "low"}},
        "service_tier": "priority",
    }


@pytest.mark.asyncio
async def test_run_agent_preserves_provider_request_overrides_on_gateway_path(monkeypatch):
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "api_mode": "codex_responses",
            "base_url": "https://example.test/v1",
            "api_key": "***",
            "request_overrides": {
                "extra_body": {"text": {"verbosity": "low"}},
            },
        },
    )
    _install_fake_agent(monkeypatch)

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    runner = _make_runner()
    source = _make_source()
    session_key = "agent:main:feishu:dm:ou_test"

    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda _source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda _session_id: [],
    )

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=source,
        session_id="session-1",
        session_key=session_key,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["request_overrides"] == {
        "extra_body": {"text": {"verbosity": "low"}},
    }
