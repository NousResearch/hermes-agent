"""Tests that /new (and its /reset alias) clears session-scoped overrides."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key
from hermes_cli.model_switch import ModelSwitchResult


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None  # disables _evict_cached_agent lock path
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
async def test_new_command_clears_session_model_override():
    """/new must remove the session-scoped model override for that session."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    # Simulate a prior /model switch stored as a session override
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "***",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_new_command_no_override_is_noop():
    """/new with no prior model override must not raise."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_new_command_only_clears_own_session():
    """/new must only clear the override for the session that triggered it."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    other_key = "other_session_key"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_model_overrides[other_key] = {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_key": "***",
        "base_url": "",
        "api_mode": "anthropic",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides[other_key] = {"enabled": True, "effort": "low"}

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert other_key in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    assert other_key in runner._session_reasoning_overrides


def test_clear_codex_reasoning_replay_for_session_rewrites_stripped_history():
    """Provider switches should strip replay-only Codex reasoning blobs from the
    persisted transcript so the next turn cannot resend them to a new backend."""
    runner = _make_runner()

    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "Reply with exactly: OK"},
        {
            "role": "assistant",
            "content": "OK",
            "codex_reasoning_items": [
                {
                    "type": "reasoning",
                    "id": "rs_tmp_hwkdj18eemj",
                    "encrypted_content": "enc_legacy",
                    "summary": [],
                }
            ],
        },
    ]

    runner._clear_codex_reasoning_replay_for_session("sess-1")

    runner.session_store.rewrite_transcript.assert_called_once()
    rewritten_messages = runner.session_store.rewrite_transcript.call_args.args[1]
    assert rewritten_messages[1].get("codex_reasoning_items") is None


@pytest.mark.asyncio
async def test_model_command_clears_replay_history_when_backend_changes(monkeypatch):
    """The real /model command path should sanitize persisted reasoning replay
    state before storing the new session override."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._agent_cache = {}
    runner._evict_cached_agent = lambda _session_key: None

    runner._session_model_overrides[session_key] = {
        "model": "openai/gpt-5.4",
        "provider": "openrouter",
        "api_key": "***",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "codex_responses",
    }
    runner.session_store.load_transcript.return_value = [
        {"role": "user", "content": "Reply with exactly: OK"},
        {
            "role": "assistant",
            "content": "OK",
            "codex_reasoning_items": [
                {
                    "type": "reasoning",
                    "id": "rs_tmp_hwkdj18eemj",
                    "encrypted_content": "enc_legacy",
                    "summary": [],
                }
            ],
        },
    ]

    monkeypatch.setattr(
        "hermes_cli.model_switch.parse_model_flags",
        lambda raw_args: ("gpt-5.4", "openai-codex", False),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: ModelSwitchResult(
            success=True,
            target_provider="openai-codex",
            new_model="gpt-5.4",
            api_key="codex-token",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            provider_label="OpenAI Codex",
            model_info=None,
            warning_message=None,
            error_message=None,
        ),
    )

    response = await runner._handle_model_command(_make_event("/model gpt-5.4 --provider openai-codex"))

    assert "Model switched to `gpt-5.4`" in response
    runner.session_store.rewrite_transcript.assert_called_once()
    rewritten_messages = runner.session_store.rewrite_transcript.call_args.args[1]
    assert rewritten_messages[1].get("codex_reasoning_items") is None
