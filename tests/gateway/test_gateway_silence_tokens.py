"""Gateway intentional-silence token behavior."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.inbound_queue import GatewayInboxStore, INBOX_METADATA_KEY
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from gateway.response_filters import (
    is_intentional_silence_agent_result,
    is_intentional_silence_response,
)


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


def test_exact_silence_tokens_are_intentional_silence():
    for token in ("[SILENT]", " SILENT ", "NO_REPLY", "no reply"):
        assert is_intentional_silence_response(token)


def test_blank_and_prose_mentions_are_not_silence():
    assert not is_intentional_silence_response("")
    assert not is_intentional_silence_response("Use NO_REPLY when no answer is needed.")
    assert not is_intentional_silence_response("The reply was [SILENT], intentionally.")


def test_failed_agent_result_never_counts_as_intentional_silence():
    assert is_intentional_silence_agent_result({"failed": False}, "NO_REPLY")
    assert not is_intentional_silence_agent_result({"failed": True}, "NO_REPLY")


def test_legacy_delivery_payload_never_infers_durable_completion():
    for payload in (None, "visible error", {"final_response": "done"}):
        outcome = gateway_run._coerce_agent_turn_outcome(payload)
        assert outcome.delivery_response == payload
        assert outcome.completed is False
        assert outcome.terminally_handled is False


@pytest.mark.asyncio
async def test_silence_token_suppresses_delivery_but_preserves_transcript(monkeypatch, tmp_path):
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
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response.delivery_response == ""
    assert response.completed is True
    appended = [call.args[1] for call in runner.session_store.append_to_transcript.call_args_list]
    assert {"role": "assistant", "content": "[SILENT]"}.items() <= appended[-1].items()
    assert [msg["role"] for msg in appended if msg.get("role") in {"user", "assistant"}] == ["user", "assistant"]


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

    assert "no response was generated" in response.delivery_response
    assert response.completed is True


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

    assert response.delivery_response == text
    assert response.completed is True


@pytest.mark.asyncio
async def test_confirmed_streaming_delivery_preserves_completed_turn(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "already delivered",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "already delivered"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "already_sent": True,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response.delivery_response is None
    assert response.completed is True


@pytest.mark.asyncio
async def test_post_commit_delivery_error_does_not_reopen_agent_turn(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    runner._should_send_voice_reply = lambda *_a, **_kw: True
    runner._send_voice_reply = AsyncMock(side_effect=RuntimeError("delivery failed"))
    runner._run_agent = AsyncMock(return_value={
        "final_response": "durably recorded answer",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "durably recorded answer"},
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

    assert "unexpected error" in response.delivery_response
    assert response.completed is True
    assert response.terminally_handled is False


@pytest.mark.asyncio
async def test_visible_failed_turn_remains_retryable(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "Provider failed after all fallbacks.",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "Provider failed after all fallbacks."},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": True,
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response.delivery_response == "Provider failed after all fallbacks."
    assert response.completed is False
    assert response.terminally_handled is False


@pytest.mark.asyncio
async def test_partial_visible_turn_remains_retryable(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={
        "final_response": "partial answer before interruption",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "partial answer before interruption"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "partial": True,
        "error": "stream interrupted",
    })

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response.delivery_response == "partial answer before interruption"
    assert response.completed is False
    assert response.terminally_handled is False


@pytest.mark.asyncio
async def test_safely_rejected_input_is_terminally_handled(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None)

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response.delivery_response is None
    assert response.completed is False
    assert response.terminally_handled is True
    runner._prepare_profile_scoped_inbound_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_only_streamed_turn_runs_once_through_real_handler(
    monkeypatch, tmp_path
):
    """Incident regression: a resume-only event that streams its final answer
    must finish the durable row instead of generating another empty recovery.
    """
    runner = _runner(monkeypatch, tmp_path)
    runner._inbox_store = GatewayInboxStore(hermes_home=tmp_path)
    runner._inbox_wakeup = None
    runner.session_store._db.has_platform_message_id.return_value = True
    event = _event()
    session_key = "agent:main:telegram:group:-1001:12345"

    await runner._inbox_enqueue_event(event, session_key, origin="direct")
    claimed = await runner._inbox_claim_event(event)
    assert claimed is not None
    assert await runner._inbox_retry_event(
        event,
        "gateway stopped after trigger persistence",
        resume_only=True,
    )
    event.text = ""
    event.internal = True

    runner._run_agent = AsyncMock(return_value={
        "final_response": "recovered answer",
        "messages": [
            {"role": "user", "content": "internal recovery"},
            {"role": "assistant", "content": "recovered answer"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "already_sent": True,
    })

    delivery_response = await runner._handle_message(event)

    assert delivery_response is None
    row = runner._inbox_store.get(event.metadata[INBOX_METADATA_KEY]["queue_id"])
    assert row is not None
    assert row.state == "completed"
    assert row.attempts == 2
    assert row.last_error is None
    assert runner._run_agent.await_args.kwargs["durable_inbox_resume"] is True


@pytest.mark.asyncio
async def test_pre_commit_persistence_error_keeps_durable_turn_retryable(
    monkeypatch, tmp_path
):
    runner = _runner(monkeypatch, tmp_path)
    runner._inbox_store = GatewayInboxStore(hermes_home=tmp_path)
    runner._inbox_wakeup = None
    runner.session_store._db.has_platform_message_id.return_value = True
    runner.session_store.append_to_transcript = MagicMock(
        side_effect=RuntimeError("database is locked")
    )
    event = _event()
    session_key = "agent:main:telegram:group:-1001:12345"

    await runner._inbox_enqueue_event(event, session_key, origin="direct")
    claimed = await runner._inbox_claim_event(event)
    assert claimed is not None

    runner._run_agent = AsyncMock(return_value={
        "final_response": "answer not yet committed",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer not yet committed"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
    })

    delivery_response = await runner._handle_message(event)

    assert "unexpected error" in delivery_response
    row = runner._inbox_store.get(event.metadata[INBOX_METADATA_KEY]["queue_id"])
    assert row is not None
    assert row.state == "resume_ready"
    assert row.attempts == 1
    assert row.resume_only is True
