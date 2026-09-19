"""Regression tests for #115325: the api_server peer-DM lane (POST /api/sessions/{id}/chat)
must share the retry policy the local and relay DM lanes already have.

A failed turn classified as transient (429 / 5xx / context overflow) is re-run exactly once,
resuming the DM row the failed attempt already persisted instead of appending a duplicate.
Auth/quota/config/model/unknown failures are never retried, and a re-run that cannot safely
adopt the persisted tail row is skipped (fail closed).
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.context_compressor import _DB_PERSISTED_MARKER
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    return app


def _failed_result(text: str, *, failure_reason: str = "rate_limit") -> dict:
    """The run_conversation failure shape the route must inspect."""
    return {
        "completed": False,
        "failed": True,
        "error": text,
        "final_response": text,
        "failure_reason": failure_reason,
        "messages": [],
        "api_calls": 1,
    }


def _ok_result(session_id: str, text: str = "recovered reply") -> dict:
    return {"completed": True, "failed": False, "final_response": text, "session_id": session_id,
            "messages": [], "api_calls": 1}


async def _post_chat(adapter, session_id, message="hi"):
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post(f"/api/sessions/{session_id}/chat", json={"message": message})
        assert resp.status == 200
        return await resp.json()


@pytest.mark.asyncio
async def test_session_chat_retries_transient_failure_once(session_db, adapter):
    """A 429-classified failed turn is re-run once; the completion carries the retry's reply."""
    session_id = session_db.create_session("retry-session", "api_server")
    session_db.replace_messages(session_id, [{"role": "user", "content": "hi"}])
    mock_run = AsyncMock(side_effect=[
        (_failed_result("Error code: 429 rate limit exceeded"), {"total_tokens": 1}),
        (_ok_result(session_id), {"total_tokens": 2}),
    ])
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert mock_run.await_count == 2
    assert payload["message"]["content"] == "recovered reply"


@pytest.mark.asyncio
async def test_session_chat_retry_resumes_the_persisted_dm_row(session_db, adapter):
    """The re-run adopts the failed attempt's tail user row: scaffolding is stripped from the
    handed history and the adopted row is passed as ``pending_cli_user_message`` (so the agent
    stages it instead of appending a duplicate user row)."""
    session_id = session_db.create_session("resume-session", "api_server")
    session_db.replace_messages(session_id, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ])
    mock_run = AsyncMock(side_effect=[
        (_failed_result("Error code: 503 server error"), {"total_tokens": 1}),
        (_ok_result(session_id), {"total_tokens": 2}),
    ])
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert payload["message"]["content"] == "recovered reply"
    retry_call = mock_run.call_args_list[1]
    # Tail adopted out of the handed history (CLI-twin semantics): the retried turn receives
    # the stripped transcript and re-stages the adopted row via pending_cli_user_message.
    assert retry_call.kwargs["conversation_history"] == []
    adopted = retry_call.kwargs["pending_cli_user_message"]
    assert adopted.get("content") == "hi"
    assert adopted.get(_DB_PERSISTED_MARKER) is True


@pytest.mark.asyncio
async def test_session_chat_does_not_retry_non_retryable_failure(session_db, adapter):
    """Auth failures are never retried — the completion keeps carrying the failure text."""
    session_id = session_db.create_session("auth-session", "api_server")
    session_db.replace_messages(session_id, [{"role": "user", "content": "hi"}])
    mock_run = AsyncMock(return_value=(
        _failed_result("Error code: 401 invalid api key", failure_reason="auth"),
        {"total_tokens": 1},
    ))
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert mock_run.await_count == 1
    assert "401" in payload["message"]["content"]


@pytest.mark.asyncio
async def test_session_chat_does_not_retry_success(session_db, adapter):
    """A completed turn is untouched: exactly one run, no adoption, no second fetch."""
    session_id = session_db.create_session("success-session", "api_server")
    session_db.replace_messages(session_id, [{"role": "user", "content": "hi"}])
    mock_run = AsyncMock(return_value=(_ok_result(session_id, "hello"), {"total_tokens": 1}))
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert mock_run.await_count == 1
    assert payload["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_session_chat_fails_closed_when_tail_cannot_be_adopted(session_db, adapter):
    """A plain assistant row after the DM blocks adoption, so the re-run must NOT happen —
    a blind re-run would append a duplicate user row."""
    session_id = session_db.create_session("occupied-session", "api_server")
    session_db.replace_messages(session_id, [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "an earlier reply"},
    ])
    mock_run = AsyncMock(return_value=(
        _failed_result("Error code: 429 rate limit exceeded"), {"total_tokens": 1},
    ))
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert mock_run.await_count == 1
    assert "429" in payload["message"]["content"]


@pytest.mark.asyncio
async def test_session_chat_does_not_retry_non_dict_result(session_db, adapter):
    """Non-dict results (interrupt shapes) keep today's behavior: single run, no crash."""
    session_id = session_db.create_session("interrupt-session", "api_server")
    session_db.replace_messages(session_id, [{"role": "user", "content": "hi"}])
    mock_run = AsyncMock(return_value=({"final_response": "stopped"}, {"total_tokens": 0}))
    with patch.object(adapter, "_run_agent", mock_run):
        payload = await _post_chat(adapter, session_id)

    assert mock_run.await_count == 1
    assert payload["message"]["content"] == "stopped"
