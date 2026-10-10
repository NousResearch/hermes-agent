"""A routed multiplex profile's turns use ITS prefill_messages_file and provider_routing.

``GatewayRunner`` read both once at boot, from the launch profile's config, and handed that copy
to every agent it built — so a secondary profile's ``provider_routing`` (including
``data_collection: deny``) and prefill file never applied to its own turns. The turn paths now
read them inside the routed profile's ``_profile_runtime_scope`` (#89161 fixed the ephemeral
system prompt the same way).
"""

from __future__ import annotations

import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.run import _profile_runtime_scope
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []
        self.request_overrides = dict(kwargs.get("request_overrides") or {})

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {"final_response": "ok", "messages": [], "api_calls": 1}


def _write_profile(home, label: str) -> None:
    home.mkdir(parents=True)
    (home / "prefill.json").write_text(
        json.dumps([{"role": "user", "content": f"{label}-PREFILL"}]), encoding="utf-8")
    (home / "config.yaml").write_text(
        "prefill_messages_file: prefill.json\n"
        "provider_routing:\n"
        f"  only: [{label}-only]\n"
        + ("  data_collection: deny\n" if label == "beta" else ""),
        encoding="utf-8",
    )


@pytest.fixture
def homes(tmp_path, monkeypatch):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "beta"
    _write_profile(default_home, "default")
    _write_profile(routed_home, "beta")
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    return default_home, routed_home


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.session_store = None
    runner.config = None
    runner._voice_mode = {}
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._show_reasoning = False
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
    # What boot used to snapshot from the launch profile; a turn must not hand this to another profile.
    runner._prefill_messages = runner._load_prefill_messages()
    runner._provider_routing = runner._load_provider_routing()
    return runner


def _source() -> SessionSource:
    return SessionSource(platform=Platform.FEISHU, chat_id="ou_test", chat_type="dm",
                         user_id="user-1", user_name="tester")


@pytest.mark.asyncio
async def test_agent_turn_uses_the_routed_profiles_prefill_and_provider_routing(homes, monkeypatch):
    _, routed_home = homes
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs",
                        lambda: {"provider": "openrouter", "api_mode": "chat_completions",
                                 "base_url": "https://openrouter.ai/api/v1", "api_key": "***"})
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    runner = _make_runner()
    assert runner._provider_routing == {"only": ["default-only"]}  # the launch profile's boot copy
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda _source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda _session_id: [],
    )

    async def turn(session_key: str) -> dict:
        _CapturingAgent.last_init = None
        result = await runner._run_agent(message="hi", context_prompt="", history=[], source=_source(),
                                         session_id="session-1", session_key=session_key)
        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        return _CapturingAgent.last_init

    with _profile_runtime_scope(routed_home):
        routed = await turn("agent:beta:feishu:dm:ou_test")
    assert routed["prefill_messages"] == [{"role": "user", "content": "beta-PREFILL"}]
    assert routed["providers_allowed"] == ["beta-only"]
    assert routed["provider_data_collection"] == "deny"

    # The launch profile's own turns are unchanged.
    launch = await turn("agent:main:feishu:dm:ou_test")
    assert launch["prefill_messages"] == [{"role": "user", "content": "default-PREFILL"}]
    assert launch["providers_allowed"] == ["default-only"]
    assert launch["provider_data_collection"] is None


@pytest.mark.asyncio
async def test_background_task_uses_the_routed_profiles_provider_routing(homes):
    _, routed_home = homes
    runner = _make_runner()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "done"))
    adapter.extract_images = MagicMock(return_value=([], "done"))
    runner.adapters[Platform.TELEGRAM] = adapter
    store = MagicMock()
    store.get_model_override.return_value = None
    runner.session_store = store
    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="u")

    with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
            patch("run_agent.AIAgent") as mock_agent:
        mock_agent.return_value.run_conversation.return_value = {"final_response": "done", "messages": []}
        with _profile_runtime_scope(routed_home):
            await runner._run_background_task("say hello", source, "bg_routed")

    kwargs = mock_agent.call_args.kwargs
    assert kwargs["providers_allowed"] == ["beta-only"]
    assert kwargs["provider_data_collection"] == "deny"
