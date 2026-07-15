"""Gateway structured delivery-outcome behavior."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import should_suppress_delivery


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _event():
    return MessageEvent(
        text="side chatter",
        source=_source(),
        message_id="msg-42",
    )


def _runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:group:-1001:12345",
        session_id="sess-silent",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


class _SuppressingAgent:
    """Primary-model result used to exercise the real gateway shape."""

    def __init__(self, **kwargs):
        self.tools = []
        self.session_id = kwargs.get("session_id")
        self.model = kwargs.get("model", "gpt-5.6-sol")
        self.provider = kwargs.get("provider")
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.api_mode = kwargs.get("api_mode")
        self.context_compressor = SimpleNamespace(
            last_prompt_tokens=0,
            context_length=100_000,
        )
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self._current_goal_generation_id = ""

    def run_conversation(self, _message, **_kwargs):
        return {
            "final_response": "model-authored quiet outcome",
            "messages": [
                {"role": "user", "content": "side chatter"},
                {
                    "role": "assistant",
                    "content": "model-authored quiet outcome",
                },
            ],
            "api_calls": 1,
            "completed": True,
            "failed": False,
            "turn_id": "turn-real-run",
            "delivery_outcome": {
                "action": "suppress",
                "reason": "the primary model chose not to post this turn",
                "turn_id": "turn-real-run",
            },
        }


def test_response_text_never_controls_delivery():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert not should_suppress_delivery(
            {
                "failed": False,
                "turn_id": "turn-1",
                "delivery_outcome": None,
                "final_response": token,
            }
        )


@pytest.mark.asyncio
async def test_structured_outcome_suppresses_delivery_but_preserves_transcript(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "[SILENT]",
        "messages": [
            {"role": "user", "content": "side chatter"},
            {"role": "assistant", "content": "[SILENT]"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "turn_id": "turn-1",
        "delivery_outcome": {
            "action": "suppress",
            "reason": "the model decided this message needs no delivery",
            "turn_id": "turn-1",
        },
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert {"role": "assistant", "content": "[SILENT]"}.items() <= appended[-1].items()
    assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_real_run_shape_preserves_model_outcome_for_normal_delivery(
    monkeypatch, tmp_path
):
    """The non-streaming run/result boundary must not drop model control."""

    runner = _runner(monkeypatch, tmp_path)
    runner.session_store.get_model_override.return_value = None
    monkeypatch.setattr("run_agent.AIAgent", _SuppressingAgent)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_runtime_config",
        lambda: {},
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "gpt-5.6-sol",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "test-only",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda _config, _platform: {"core"},
    )

    source = _source()
    session_key = "agent:main:telegram:group:-1001:12345"
    shaped = await runner._run_agent(
        message="side chatter",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-silent",
        session_key=session_key,
    )

    assert shaped.get("failed") is False, shaped
    assert shaped["delivery_outcome"] == {
        "action": "suppress",
        "reason": "the primary model chose not to post this turn",
        "turn_id": "turn-real-run",
    }
    assert should_suppress_delivery(shaped) is True

    # Continue through the normal handler with the exact result produced by
    # the real run-shaping path.  The assistant transcript remains durable,
    # while the public response is suppressed by the model-authored receipt.
    runner._run_agent = AsyncMock(return_value=shaped)
    response = await runner._handle_message_with_agent(
        _event(), source, session_key, 1
    )

    assert response == ""
    appended = [
        call.args[1]
        for call in runner.session_store.append_to_transcript.call_args_list
    ]
    assert {
        "role": "assistant",
        "content": "model-authored quiet outcome",
    }.items() <= appended[-1].items()


@pytest.mark.asyncio
async def test_empty_success_still_gets_empty_response_warning(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert "no response was generated" in response


@pytest.mark.asyncio
async def test_prose_mentioning_silence_token_is_delivered(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    text = "Use [SILENT] when no answer is needed."
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == text


@pytest.mark.asyncio
async def test_failed_turn_delivers_even_with_structured_suppress(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    text = "The provider failed before the task completed."
    runner._run_agent = AsyncMock(return_value={
        "final_response": text,
        "messages": [],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": True,
        "turn_id": "turn-1",
        "delivery_outcome": {
            "action": "suppress",
            "reason": "stale choice before failure",
            "turn_id": "turn-1",
        },
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert text in response
