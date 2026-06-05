"""Tests for gateway /fast support and Priority Processing routing."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None
    instances = []

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        type(self).instances.append(self)
        self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
            "ephemeral_system_prompt": self.ephemeral_system_prompt,
        }
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def test_transient_auto_skills_detects_youtube_urls():
    event = _make_event("check this https://youtube.com/shorts/11K3tGgFnJs")

    assert gateway_run._transient_auto_skills_for_event(event) == ["youtube-content"]


def test_transient_auto_skills_ignores_regular_chat():
    event = _make_event("just chatting about something")

    assert gateway_run._transient_auto_skills_for_event(event) == []


def test_turn_route_injects_priority_processing_without_changing_runtime():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openrouter"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}


def test_turn_route_skips_priority_processing_for_unsupported_models():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.3-codex", runtime_kwargs)

    assert route["request_overrides"] == {}


def test_gateway_model_override_uses_scoped_primary_and_fallback(monkeypatch):
    runner = _make_runner()
    runner._service_tier = None
    runner._fallback_model = [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    ]
    runtime_kwargs = {
        "api_key": "primary-key",
        "base_url": "http://127.0.0.1:8020/v1",
        "provider": "omlx",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": "primary-pool",
    }

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "omlx"
        return {
            "api_key": "override-key",
            "base_url": "http://127.0.0.1:8020/v1",
            "provider": "omlx",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "hi",
        "qwen3-30b-a3b-instruct-2507-4bit",
        runtime_kwargs,
        user_config={
            "gateway": {
                "agent": {
                    "model_override": {
                        "enabled": True,
                        "provider": "omlx",
                        "model": "qwen3-4b-instruct-2507-4bit",
                        "fallback_providers": [
                            {
                                "provider": "openrouter",
                                "model": "deepseek/deepseek-v4-pro",
                            }
                        ],
                    },
                    "local_failover": {
                        "enabled": True,
                        "no_first_chunk_timeout_seconds": 20,
                        "stale_chunk_timeout_seconds": 30,
                        "max_primary_retries": 1,
                        "failover_on_stream_errors": [
                            "RemoteProtocolError",
                            "incomplete chunked read",
                        ],
                    },
                }
            }
        },
    )

    assert route["model"] == "qwen3-4b-instruct-2507-4bit"
    assert route["runtime"]["provider"] == "omlx"
    assert route["runtime"]["credential_pool"] is None
    assert route["fallback_model"] == [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]
    assert route["gateway_local_failover"] == {
        "enabled": True,
        "no_first_chunk_timeout_seconds": 20.0,
        "stale_chunk_timeout_seconds": 30.0,
        "max_primary_retries": 1,
        "failover_on_stream_errors": (
            "remoteprotocolerror",
            "incomplete chunked read",
        ),
    }


def test_gateway_model_override_defaults_off_keep_global_route():
    runner = _make_runner()
    runner._service_tier = None
    runner._fallback_model = [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]
    runtime_kwargs = {
        "api_key": "primary-key",
        "base_url": "http://127.0.0.1:8020/v1",
        "provider": "omlx",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": "primary-pool",
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "hi",
        "qwen3-30b-a3b-instruct-2507-4bit",
        runtime_kwargs,
        user_config={},
    )

    assert route["model"] == "qwen3-30b-a3b-instruct-2507-4bit"
    assert route["runtime"]["provider"] == "omlx"
    assert route["runtime"]["credential_pool"] == "primary-pool"
    assert route["fallback_model"] == [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]
    assert route["gateway_local_failover"] == {"enabled": False}


def test_gateway_semantic_escalation_routes_image_turn_to_pro(monkeypatch):
    runner = _make_runner()
    runner._service_tier = None
    runtime_kwargs = {
        "api_key": "primary-key",
        "base_url": "http://127.0.0.1:8000/v1",
        "provider": "omlx",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "openrouter"
        return {
            "api_key": "override-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )

    user_config = {
        "gateway": {
            "agent": {
                "model_override": {
                    "enabled": True,
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-flash",
                    "fallback_providers": [
                        {
                            "provider": "openrouter",
                            "model": "deepseek/deepseek-v4-pro",
                        }
                    ],
                },
                "semantic_escalation": {
                    "enabled": True,
                    "platforms": ["telegram"],
                    "triggers": ["image"],
                    "provider": "openrouter",
                    "model": "deepseek/deepseek-v4-pro",
                },
            }
        }
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "what is in this image?",
        "qwen3-30b-a3b-instruct-2507-4bit",
        runtime_kwargs,
        user_config=user_config,
        platform_key="telegram",
        route_hints={"has_image": True},
    )

    assert route["model"] == "deepseek/deepseek-v4-pro"
    assert route["runtime"]["provider"] == "openrouter"
    assert route["fallback_model"] == [
        {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}
    ]


def test_gateway_semantic_escalation_leaves_plain_text_on_flash(monkeypatch):
    runner = _make_runner()
    runner._service_tier = None
    runtime_kwargs = {
        "api_key": "primary-key",
        "base_url": "http://127.0.0.1:8000/v1",
        "provider": "omlx",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "openrouter"
        return {
            "api_key": "override-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "what time is it?",
        "qwen3-30b-a3b-instruct-2507-4bit",
        runtime_kwargs,
        user_config={
            "gateway": {
                "agent": {
                    "model_override": {
                        "enabled": True,
                        "provider": "openrouter",
                        "model": "deepseek/deepseek-v4-flash",
                    },
                    "semantic_escalation": {
                        "enabled": True,
                        "platforms": ["telegram"],
                        "triggers": ["image"],
                        "provider": "openrouter",
                        "model": "deepseek/deepseek-v4-pro",
                    },
                }
            }
        },
        platform_key="telegram",
        route_hints={},
    )

    assert route["model"] == "deepseek/deepseek-v4-flash"
    assert route["runtime"]["provider"] == "openrouter"


def test_session_info_reports_gateway_model_override(monkeypatch):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "qwen3-30b-a3b-instruct-2507-4bit")
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {
                "provider": "omlx",
                "default": "qwen3-30b-a3b-instruct-2507-4bit",
            },
            "gateway": {
                "agent": {
                    "model_override": {
                        "enabled": True,
                        "provider": "openrouter",
                        "model": "deepseek/deepseek-v4-flash",
                        "fallback_providers": [
                            {
                                "provider": "openrouter",
                                "model": "deepseek/deepseek-v4-pro",
                            }
                        ],
                    }
                }
            },
        },
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "api_key": "global-key",
            "base_url": "http://127.0.0.1:8000/v1",
            "provider": "omlx",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )

    def _fake_resolve_runtime_provider(**kwargs):
        assert kwargs["requested"] == "openrouter"
        return {
            "api_key": "override-key",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "command": None,
            "args": [],
            "credential_pool": None,
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _fake_resolve_runtime_provider,
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *args, **kwargs: 128000,
    )

    info = gateway_run.GatewayRunner._format_session_info(runner)

    assert "deepseek/deepseek-v4-flash" in info
    assert "Provider: openrouter" in info
    assert "qwen3-30b-a3b-instruct-2507-4bit" not in info
    assert "Endpoint: http://127.0.0.1:8000/v1" not in info


@pytest.mark.asyncio
async def test_handle_fast_command_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")

    response = await runner._handle_fast_command(_make_event("/fast fast"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"


@pytest.mark.asyncio
async def test_run_agent_passes_priority_processing_to_gateway_agent(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  service_tier: fast\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    # ``_load_service_tier`` was refactored to call ``_load_gateway_runtime_config``
    # (which wraps ``_load_gateway_config`` plus env-expansion).  Since the test
    # stubs ``_load_gateway_config`` to ``{}``, also stub the runtime wrapper
    # directly so the priority routing assertions still exercise the live tier.
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["service_tier"] == "priority"
    assert _CapturingAgent.last_init["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.asyncio
async def test_run_agent_injects_telegram_runtime_time_context(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("timezone: America/Chicago\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"timezone": "America/Chicago"},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="what time is it?",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:telegram:dm:12345",
    )

    assert result["final_response"] == "ok"
    prompt = _CapturingAgent.last_init["ephemeral_system_prompt"]
    assert "Context prompt" in prompt
    assert "<runtime_context>" in prompt
    assert "Timezone: America/Chicago" in prompt
    assert "do not call terminal just to determine the current time or date" in prompt


@pytest.mark.asyncio
async def test_cached_telegram_agent_refreshes_runtime_time_context(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("timezone: America/Chicago\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"timezone": "America/Chicago"},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    contexts = iter(["<runtime_context>first</runtime_context>", "<runtime_context>second</runtime_context>"])
    monkeypatch.setattr(gateway_run, "_telegram_runtime_context_prompt", lambda user_config: next(contexts))

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.instances = []
    session_key = "agent:main:telegram:dm:12345"
    for message in ("first", "second"):
        result = await runner._run_agent(
            message=message,
            context_prompt="",
            history=[],
            source=_make_source(),
            session_id="session-1",
            session_key=session_key,
        )
        assert result["final_response"] == "ok"

    assert len(_CapturingAgent.instances) == 1
    assert _CapturingAgent.last_run["ephemeral_system_prompt"] == (
        "<runtime_context>second</runtime_context>"
    )
