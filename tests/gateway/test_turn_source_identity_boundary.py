"""Gateway-boundary regressions for transport-authored turn source identity."""

import sys
import threading
import types
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


_CAPTURED_SOURCES = []


class _CapturingAgent:
    def __init__(self, *args, **kwargs):
        self.tools = []
        self.model = kwargs.get("model", "test-model")
        self.provider = kwargs.get("provider", "test-provider")
        self.session_id = kwargs.get("session_id", "source-session")
        self.context_compressor = None
        self.is_interrupted = False
        self._current_source_identity = None

    def run_conversation(self, user_message, conversation_history=None, **kwargs):
        _CAPTURED_SOURCES.append(self._current_source_identity)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _source():
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C0BTEFMAAJX",
        chat_type="channel",
        user_id="U123",
    )


def _runner(monkeypatch, tmp_path):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

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
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._gateway_loop = None
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._clear_session_env = lambda tokens: None
    runner._is_session_run_current = lambda key, generation: True
    runner._reply_anchor_for_event = lambda event: event.message_id

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"agent": {"model": "test-model"}})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "test-model")
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-only",
        },
    )
    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})
    monkeypatch.setattr(
        "gateway.run_heartbeat_acceptance.heartbeat_owner_is_current",
        lambda *args: True,
    )

    source = _source()
    session = SimpleNamespace(session_id="source-session")
    runner._hmwa_resolve_session = AsyncMock(
        return_value=(source, session, "agent:main:slack:channel:C0BTEFMAAJX")
    )

    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._hmwa_shape_agent_response = AsyncMock(return_value=("done", False, []))
    runner._hmwa_prepend_reasoning = lambda result, response, source, silence: response
    runner._hmwa_runtime_footer_line = lambda *args: None
    runner._hmwa_post_turn_hooks = AsyncMock()
    runner._hmwa_classify_turn_failure = lambda *args: (False, False, False)
    runner._hmwa_compression_exhaustion_reset = AsyncMock(
        side_effect=lambda result, response, entry, *args: (response, entry)
    )
    runner._hmwa_persist_turn_transcript = AsyncMock()
    runner._hmwa_deliver_turn_response = AsyncMock(return_value="done")
    runner._hmwa_agent_error_reply = AsyncMock(side_effect=lambda error, *args: (_ for _ in ()).throw(error))
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("internal", "metadata", "expected_state"),
    [
        (
            False,
            {"completion_gate_source_state": {"request_id": "1789000000.123456", "status": "closed"}},
            {"request_id": "1789000000.123456", "status": "closed"},
        ),
        (True, {}, None),
        (False, {"completion_gate_source_state": {"status": "closed"}}, None),
    ],
)
async def test_handle_message_reaches_agent_with_exact_source_identity(
    monkeypatch, tmp_path, internal, metadata, expected_state
):
    runner = _runner(monkeypatch, tmp_path)
    source = _source()
    timestamp = datetime(2026, 9, 26, 16, 27, 32, tzinfo=timezone.utc)
    event = MessageEvent(
        text="normal slack turn",
        source=source,
        message_id="1789000000.123456",
        internal=internal,
        metadata=metadata,
        timestamp=timestamp,
    )
    prepared = runner._PreparedTurn(
        [], "", event.text, event.text, timestamp.timestamp(), None,
        "source-session", "owner-1",
    )
    runner._hmwa_prepare_turn = AsyncMock(return_value=(prepared, {}))
    _CAPTURED_SOURCES.clear()

    assert await runner._handle_message_with_agent(
        event, source, "agent:main:slack:channel:C0BTEFMAAJX", 1
    ) == "done"

    assert _CAPTURED_SOURCES == [{
        "platform": "slack",
        "channel_id": "C0BTEFMAAJX",
        "request_id": "1789000000.123456",
        "timestamp": timestamp.timestamp(),
        "internal": internal,
        "source_state": expected_state,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_proxy_stays_compatible_and_queued_followup_rebinds_exact_source_identity(
    monkeypatch, tmp_path, internal
):
    runner = _runner(monkeypatch, tmp_path)
    source = _source()
    timestamp = datetime(2026, 9, 26, 16, 30, 1, tzinfo=timezone.utc)
    source_state = {"request_id": "1789000001.654321", "status": "closed"}

    runner._get_proxy_url = lambda: "http://proxy.invalid"
    runner._run_agent_via_proxy = AsyncMock(return_value={"final_response": "proxied"})
    proxied = await runner._run_agent_inner(
        "proxied turn", "", [], source, "source-session",
        inbound_internal=True, inbound_source_state=source_state,
    )
    assert proxied == {"final_response": "proxied"}
    assert runner._run_agent_via_proxy.await_count == 1

    pending_event = MessageEvent(
        text="queued turn",
        source=source,
        message_id="1789000001.654321",
        internal=internal,
        metadata={"completion_gate_source_state": source_state},
        timestamp=timestamp,
    )
    adapter = SimpleNamespace(
        _active_sessions={"agent:main:slack:channel:C0BTEFMAAJX": __import__("asyncio").Event()},
        _streaming_tts_completed_turns=set(),
        send_typing=AsyncMock(),
    )
    ctx = TurnContext(
        source=source,
        session_id="source-session",
        session_key="agent:main:slack:channel:C0BTEFMAAJX",
        run_generation=1,
        context_prompt="",
        history=[],
        _interrupt_depth=0,
    )
    runner._get_proxy_url = lambda: None
    runner._is_goal_continuation_event = lambda event: False
    runner._session_key_for_source = lambda next_source: ctx.session_key
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="queued turn")
    runner._adapter_for_source = lambda source: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    _CAPTURED_SOURCES.clear()

    result = await runner._run_agent_queued_followup(
        ctx, adapter, "queued turn", pending_event,
        "discarded", {"interrupted": True, "messages": []}, None,
    )

    assert result["final_response"] == "done"
    assert _CAPTURED_SOURCES == [{
        "platform": "slack",
        "channel_id": "C0BTEFMAAJX",
        "request_id": "1789000001.654321",
        "timestamp": timestamp.timestamp(),
        "internal": internal,
        "source_state": source_state,
    }]

    _CAPTURED_SOURCES.clear()
    result = await runner._run_agent_queued_followup(
        ctx, adapter, "interrupt fallback", None,
        "discarded", {"interrupted": True, "messages": []}, None,
    )
    assert result["final_response"] == "done"
    assert _CAPTURED_SOURCES == [{
        "platform": "slack",
        "channel_id": "C0BTEFMAAJX",
        "request_id": "",
        "timestamp": None,
        "internal": False,
        "source_state": None,
    }]
