"""Tests for gateway /fast support and Priority Processing routing."""

import asyncio
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import hermes_yaml as yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None
    last_run = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
        persist_user_timestamp=None,
    ):
        type(self).last_run = {
            "user_message": user_message,
            "conversation_history": conversation_history,
            "task_id": task_id,
            "persist_user_message": persist_user_message,
            "persist_user_timestamp": persist_user_timestamp,
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


def _make_discord_auto_thread_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="999",
        chat_type="thread",
        user_id="user-1",
        thread_id="999",
        parent_chat_id="100",
        auto_thread_created=True,
        auto_thread_initial_name="raw user prompt",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def test_turn_route_injects_priority_processing_without_changing_runtime():
    runner = _make_runner()
    runner._service_tier = "priority"
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://api.openai.com/v1",
        "provider": "openai",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)

    assert route["runtime"]["provider"] == "openai"
    assert route["runtime"]["api_mode"] == "chat_completions"
    assert route["request_overrides"] == {"service_tier": "priority"}

    # Proxied routes never receive first-party fast params; OpenRouter does.
    runtime_kwargs.update(base_url="https://openrouter.ai/api/v1", provider="openrouter")
    route = gateway_run.GatewayRunner._resolve_turn_agent_config(runner, "hi", "gpt-5.4", runtime_kwargs)
    assert route["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.asyncio
async def test_handle_fast_command_global_flag_persists_config(monkeypatch, tmp_path):
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    # /fast now resolves eligibility through the session runtime resolver; with
    # no session override that path calls the real provider resolver, so stub it.
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(_make_event("/fast fast --global"))

    assert "FAST" in response
    assert runner._service_tier == "priority"

    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert saved["agent"]["service_tier"] == "fast"
    # Global write supersedes the session override.
    assert not runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_session_fast_override_beats_config_default(monkeypatch, tmp_path):
    """A session /fast normal wins over agent.service_tier: fast in config."""
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"agent": {"service_tier": "fast"}},
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
    # Eligibility routes through the session runtime resolver; stub the real
    # provider resolution the no-override path would otherwise trigger.
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    event = _make_event("/fast normal")
    session_key = runner._session_key_for_source(event.source)

    response = await runner._handle_fast_command(event)

    assert "NORMAL" in response
    # Override stores explicit None (normal) and wins over config "fast".
    assert session_key in runner._session_service_tier_overrides
    assert runner._resolve_session_service_tier(session_key=session_key) is None
    # A different session still gets the config default.
    assert runner._resolve_session_service_tier(session_key="other-session") == "priority"


@pytest.mark.asyncio
async def test_handle_fast_flex_is_session_scoped(monkeypatch, tmp_path):
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "openai/gpt-5")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    event = _make_event("/fast flex")
    response = await runner._handle_fast_command(event)
    session_key = runner._session_key_for_source(event.source)

    assert "FLEX" in response
    assert runner._service_tier == "flex"
    assert runner._resolve_session_service_tier(session_key=session_key) == "flex"
    assert not (tmp_path / "config.yaml").exists()


@pytest.mark.asyncio
async def test_fast_status_ungated_for_session_model(monkeypatch, tmp_path):
    """Status uses the session-effective model and does not capability-gate."""
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", lambda: {"agent": {"service_tier": "flex"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "claude-sonnet-4-6")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "model": "claude-sonnet-4-6", "provider": "anthropic",
        "base_url": "https://api.anthropic.com", "api_key": "***",
    })

    event = _make_event("/fast status")
    runner._try_send_choice_picker = AsyncMock(return_value=False)
    response = await runner._handle_fast_command(event)
    assert "not_supported" not in (response or "")
    assert "flex" in response.lower()


@pytest.mark.asyncio
async def test_fast_switch_uses_session_openrouter_route(monkeypatch, tmp_path):
    runner = _make_runner()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {
        "model": {"provider": "anthropic", "default": "claude-sonnet-4-6"},
    })
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "claude-sonnet-4-6")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "model": "claude-sonnet-4-6", "provider": "anthropic",
        "base_url": "https://api.anthropic.com", "api_key": "***",
    })

    event = _make_event("/fast fast")
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    }
    response = await runner._handle_fast_command(event)
    assert "FAST" in response
    assert runner._service_tier == "priority"


class _ProvenanceBgAgent:
    """Ctor leaves bake unmarked (real ``AIAgent`` contract) and records the instance."""

    last = None

    def __init__(self, *args, **kwargs):
        self.request_overrides = dict(kwargs.get("request_overrides") or {})
        self.service_tier = kwargs.get("service_tier")
        self._framework_baked_tier_keys = frozenset()
        self._service_tier_session_pinned = False
        self._block_service_tier_escalation = False
        type(self).last = self

    def run_conversation(self, **kwargs):
        return {"final_response": "ok", "messages": []}

    def shutdown_memory_provider(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _openrouter_runtime(**extra):
    runtime = {
        "api_key": "k",
        "base_url": "https://openrouter.ai/api/v1",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "request_overrides": {"extra_body": {"keep": 1}},
    }
    runtime.update(extra)
    return runtime


def _isolate_gateway_background_home(monkeypatch, tmp_path):
    """Pin gateway home + iteration budget so /bg tests never touch launch-profile ``.env``.

    ``_run_background_task_inner`` calls ``_current_max_iterations`` *before* vision
    enrichment. That helper reloads dotenv via module-level ``gateway.run._hermes_home``
    (import-time path, not the pytest ``HERMES_HOME``). Unstubbed, it reads the real
    home and ``home_io_guard`` aborts the try — ``agent is None`` and vision never
    sets ``started``.
    """
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 32)


def _bind_gateway_background_adapter(runner, source):
    """Wire the production delivery seam (``_delivery_adapter_for``, not ``_adapter_for_source``)."""
    mock_adapter = AsyncMock()
    mock_adapter.send = AsyncMock()
    mock_adapter.extract_media = MagicMock(return_value=([], "ok"))
    mock_adapter.extract_images = MagicMock(return_value=([], "ok"))
    runner.adapters[source.platform] = mock_adapter
    runner._delivery_adapter_for = lambda src: mock_adapter
    runner._thread_metadata_for_source = lambda src, anchor=None: {}
    return mock_adapter


async def _drive_gateway_background_task(
    runner, source, monkeypatch, tmp_path, *, pinned, service_tier, runtime=None,
):
    """Run production ``_run_background_task`` with the ``run_sync`` body intact.

    Stubs adapter/runtime/pin/toolsets so the test can reach
    ``gateway/run_turn.py`` ``_apply_background_agent_tier_provenance`` after
    construction. The executor is inlined; dropping that post-ctor call goes red
    because ``_ProvenanceBgAgent`` starts unmarked.
    """
    _isolate_gateway_background_home(monkeypatch, tmp_path)
    _bind_gateway_background_adapter(runner, source)
    runner._resolve_session_agent_runtime = lambda source=None, user_config=None, session_key=None: (
        "openai/gpt-5",
        runtime or _openrouter_runtime(),
    )
    runner._resolve_session_service_tier = lambda source=None, session_key=None: service_tier
    runner._session_service_tier_is_pinned = lambda session_key=None: pinned
    runner._session_key_for_source = lambda src: "session-1"
    runner._resolve_session_reasoning_config = lambda source=None, session_key=None, model="": None
    runner._resolve_turn_toolsets = lambda user_config, src, platform_key: (["file"], None)
    runner._refresh_fallback_model = lambda: None

    async def _inline_executor(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = _inline_executor
    _ProvenanceBgAgent.last = None
    with patch("gateway.run._load_gateway_config", return_value={}), \
            patch("run_agent.AIAgent", _ProvenanceBgAgent):
        await runner._run_background_task("hi", source, "bg_wire")
    return _ProvenanceBgAgent.last


def test_gateway_bg_from_pinned_session_child_does_not_inherit_bake(monkeypatch):
    """Gateway /bg inherits the session pin; a delegated child does not inherit the bake."""
    from run_agent import AIAgent
    from tools import delegate_tool as dt
    import tools.delegate_tool_config as dtc

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
    )
    monkeypatch.setattr(dt, "_load_config", lambda: {})
    monkeypatch.setattr(dtc, "_load_config", lambda: {})

    _openrouter = "https://openrouter.ai/api/v1"
    _match = "openai/gpt-5"
    runner = _make_runner()
    runner._service_tier = "priority"
    runner._session_service_tier_is_pinned = lambda session_key: True
    runtime_kwargs = {
        "api_key": "k",
        "base_url": _openrouter,
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }
    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", _match, runtime_kwargs,
    )
    bg = AIAgent(
        api_key="k",
        base_url=_openrouter,
        provider="openrouter",
        api_mode="chat_completions",
        model=_match,
        platform="telegram",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
        request_overrides=route["request_overrides"],
        service_tier="priority",
        session_id="bg-gw-fast",
    )
    child = None
    try:
        runner._apply_background_agent_tier_provenance(bg, {
            "pinned": True,
            "framework_baked_tier_keys": route.get("framework_baked_tier_keys"),
        })
        assert bg._service_tier_session_pinned is True
        assert "service_tier" in (getattr(bg, "_framework_baked_tier_keys", None) or ())
        assert bg._build_api_kwargs([{"role": "user", "content": "hi"}])["service_tier"] == "priority"
        creds = dt._resolve_delegation_credentials({"model": "", "provider": ""}, bg)
        child = dt._build_child_agent(
            task_index=0,
            goal="goal",
            context=None,
            toolsets=["file"],
            model=None,
            max_iterations=4,
            task_count=1,
            parent_agent=bg,
            override_request_overrides=creds.get("request_overrides"),
        )
        assert child._service_tier_session_pinned is False
        assert "service_tier" not in (child.request_overrides or {})
        kwargs = child._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert "service_tier" not in kwargs
        assert "speed" not in kwargs
    finally:
        if child is not None:
            child.close()
        bg.close()


def test_gateway_bg_unpinned_does_not_spurious_mark():
    """Unpinned session: provenance helper must not mark copied raw tier keys."""
    from agent.fast_mode import TIER_WIRE_KEYS

    runner = _make_runner()
    runner._service_tier = None
    runner._session_service_tier_is_pinned = lambda session_key: False
    raw = {"service_tier": "flex", "extra_body": {"keep": 1}}
    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "hi", "openai/gpt-5", _openrouter_runtime(request_overrides=dict(raw)),
    )
    assert route.get("framework_baked_tier_keys") in (None, {})
    bg = SimpleNamespace(
        request_overrides=dict(route["request_overrides"]),
        _framework_baked_tier_keys=frozenset({"service_tier"}),
        _service_tier_session_pinned=True,
    )
    runner._apply_background_agent_tier_provenance(bg, {
        "pinned": False,
        "framework_baked_tier_keys": route.get("framework_baked_tier_keys"),
    })
    assert not bg._framework_baked_tier_keys
    assert bg._service_tier_session_pinned is False
    assert bg.request_overrides.get("service_tier") == "flex"
    assert bg.request_overrides.get("extra_body") == {"keep": 1}
    # * Helper clears the marker; it does not strip raw user tier keys.
    assert all(
        key in bg.request_overrides or key not in raw
        for key in TIER_WIRE_KEYS
    )


@pytest.mark.asyncio
async def test_gateway_bg_run_turn_applies_tier_provenance(monkeypatch, tmp_path):
    """``_run_background_task_inner`` must apply bake after construction.

    Seam: ``gateway/run_turn.py`` ``run_sync`` constructs ``AIAgent`` then calls
    ``_apply_background_agent_tier_provenance``. A capturing ctor starts unmarked
    (real ``AIAgent`` contract); dropping the post-ctor call leaves the marker empty.
    """
    runner = _make_runner()
    source = _make_source()
    agent = await _drive_gateway_background_task(
        runner, source, monkeypatch, tmp_path, pinned=True, service_tier="priority",
    )
    assert agent is not None
    assert agent._service_tier_session_pinned is True
    assert "service_tier" in (agent._framework_baked_tier_keys or ())
    assert agent.request_overrides.get("service_tier") == "priority"
    assert agent.request_overrides.get("extra_body") == {"keep": 1}
    assert agent._block_service_tier_escalation is True
    assert agent.service_tier == "priority"


@pytest.mark.asyncio
async def test_gateway_bg_vision_stall_keeps_snapshotted_pin(monkeypatch, tmp_path):
    """Vision await must not pick up another session's shared ``_service_tier``.

    Session A is pinned normal. While /bg stalls on vision, session B writes
    ``_service_tier='priority'`` and flips the pin probe. A's agent is built
    from the pre-await snapshot: value ``None`` + pinned ``True``.
    """
    runner = _make_runner()
    source = _make_source()
    started = asyncio.Event()
    release = asyncio.Event()

    async def _stall_vision(prompt, image_paths):
        started.set()
        await release.wait()
        return "ENRICHED"

    _isolate_gateway_background_home(monkeypatch, tmp_path)
    runner._enrich_message_with_vision = _stall_vision
    _bind_gateway_background_adapter(runner, source)
    runner._resolve_session_agent_runtime = lambda source=None, user_config=None, session_key=None: (
        "openai/gpt-5",
        _openrouter_runtime(),
    )
    runner._resolve_session_service_tier = lambda source=None, session_key=None: None
    runner._session_service_tier_is_pinned = lambda session_key=None: True
    runner._session_key_for_source = lambda src: "session-a"
    runner._resolve_session_reasoning_config = lambda source=None, session_key=None, model="": None
    runner._resolve_turn_toolsets = lambda user_config, src, platform_key: (["file"], None)
    runner._refresh_fallback_model = lambda: None

    async def _inline_executor(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = _inline_executor
    _ProvenanceBgAgent.last = None

    async def _run_a():
        with patch("gateway.run._load_gateway_config", return_value={}), \
                patch("run_agent.AIAgent", _ProvenanceBgAgent):
            await runner._run_background_task(
                "hi", source, "bg_a",
                media_urls=["https://example.com/a.png"],
                media_types=["image/png"],
            )

    task = asyncio.create_task(_run_a())
    await started.wait()
    runner._service_tier = "priority"
    runner._session_service_tier_is_pinned = lambda session_key=None: False
    runner._resolve_session_service_tier = lambda source=None, session_key=None: "priority"
    release.set()
    await task

    agent = _ProvenanceBgAgent.last
    assert agent is not None
    assert agent.service_tier is None
    assert agent._service_tier_session_pinned is True
    assert not (agent._framework_baked_tier_keys or ())
    assert "service_tier" not in (agent.request_overrides or {})


_ASTRA_ON_CODEX = {"model": "gpt-6-astra", "provider": "openai-codex",
                   "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "***"}
_GPT_ON_OPENROUTER = {"model": "openai/gpt-5.4", "provider": "openrouter",
                      "base_url": "https://openrouter.ai/api/v1", "api_key": "***"}


@pytest.mark.asyncio
@pytest.mark.parametrize("default_model, override, command, accepted, expected_tier", [
    # #118761: Astra picked with session /model over a default /fast can't serve.
    ("claude-sonnet-4-6", _ASTRA_ON_CODEX, "/fast ultrafast", True, "ultrafast"),
    # OpenRouter is a first-class service-tier route (PR #104586); /fast maps to priority.
    ("claude-opus-5-5", _GPT_ON_OPENROUTER, "/fast fast", True, "priority"),
])
async def test_fast_gate_follows_the_session_route(
    monkeypatch, tmp_path, default_model, override, command, accepted, expected_tier,
):
    """Real fast-mode tables: /fast accepts exactly the tiers the session's next turn would send."""
    runner = _make_runner()
    event = _make_event(command)
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = dict(override)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: default_model)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(event)

    tier = runner._resolve_session_service_tier(session_key=session_key)
    model, runtime = runner._resolve_session_agent_runtime(source=event.source)
    route = runner._resolve_turn_agent_config("hi", model, runtime)
    if accepted:
        assert "only available" not in response
        assert tier == expected_tier
        assert route["request_overrides"] == {"service_tier": expected_tier}
    else:
        assert "only available" in response
        assert session_key not in runner._session_service_tier_overrides


@pytest.mark.asyncio
async def test_fast_override_lands_under_the_recovered_telegram_topic_key(monkeypatch, tmp_path):
    """/fast keys its eligibility check AND its tier override by the topic-recovered source the next
    turn uses (#30479), not the raw lobby-shaped event source."""
    import dataclasses

    runner = _make_runner()
    source = _make_source()
    monkeypatch.setattr(runner, "_recover_telegram_topic_thread_id", lambda src: "77")
    raw_key = runner._session_key_for_source(source)
    turn_key = runner._session_key_for_source(dataclasses.replace(source, thread_id="77"))
    assert turn_key != raw_key
    runner._session_model_overrides[turn_key] = dict(_ASTRA_ON_CODEX)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "claude-sonnet-4-6")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})

    response = await runner._handle_fast_command(MessageEvent(text="/fast fast", source=source, message_id="m1"))

    assert "FAST" in response
    assert runner._resolve_session_service_tier(session_key=turn_key) == "priority"
    assert raw_key not in runner._session_service_tier_overrides
