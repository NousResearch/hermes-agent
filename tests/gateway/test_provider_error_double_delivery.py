"""#72131: provider errors delivered twice on adapters without send_or_update_status.

A provider failure is surfaced through two gateway channels — the mid-run
status callback (rewritten to a user-safe reply and, on adapters without
``send_or_update_status``, delivered as a PERSISTENT message by the plain-send
fallback) and the failed turn's final response, which the delivery path
sanitizes to the byte-identical text. These tests cover the dedup:

- the canonical replies are recognized by set membership, not by re-running
  the raw-envelope classifier (which misses e.g. the rate-limit reply);
- the status bridge records a delivered reply on the TurnContext only after a
  SUCCESSFUL fallback send, and never for ``send_or_update_status`` adapters;
- recording is per-turn — interleaved turns never see each other's state;
- the outer delivery path (`_handle_message_with_agent`), which re-derives
  the sanitized text from the raw result, suppresses the duplicate there —
  so inner-path reconstruction cannot resurrect it — while transcript
  persistence is untouched.
"""

import concurrent.futures
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, SendResult
from gateway.run import (
    _GATEWAY_PROVIDER_ERROR_AUTH_REPLY,
    _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY,
    _GATEWAY_PROVIDER_ERROR_REPLIES,
    TurnRunner,
    _gateway_provider_error_reply,
    _is_gateway_provider_error_reply,
    _looks_like_gateway_provider_error,
    _should_suppress_provider_error_final,
)
from gateway.session import SessionEntry, SessionSource
from gateway.turn_context import TurnContext


# ---------------------------------------------------------------------------
# Canonical-reply recognition
# ---------------------------------------------------------------------------


class TestProviderErrorReplyRecognition:
    def test_every_canonical_reply_is_recognized(self):
        for reply in _GATEWAY_PROVIDER_ERROR_REPLIES:
            assert _is_gateway_provider_error_reply(reply), reply

    def test_rate_limit_reply_is_recognized_without_the_envelope_classifier(self):
        """The regression that sank the previous fix attempt: the rewritten
        rate-limit reply says "rate-limiting requests", which the raw-envelope
        shape matcher does not accept — recognition must not go through it."""
        reply = _gateway_provider_error_reply("rate limited after 3 retries")
        assert reply == _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
        assert not _looks_like_gateway_provider_error(reply)
        assert _is_gateway_provider_error_reply(reply)

    def test_raw_envelopes_and_prose_are_not_replies(self):
        assert not _is_gateway_provider_error_reply("rate limited after 3 retries")
        assert not _is_gateway_provider_error_reply("API call failed: HTTP 500")
        assert not _is_gateway_provider_error_reply("All good, here is your answer.")
        assert not _is_gateway_provider_error_reply("")
        assert not _is_gateway_provider_error_reply(None)


class TestShouldSuppressProviderErrorFinal:
    def test_suppresses_identical_delivered_reply(self):
        assert _should_suppress_provider_error_final(
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY,
            [_GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY],
        )

    def test_no_suppression_without_delivered_statuses(self):
        assert not _should_suppress_provider_error_final(
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY, None,
        )
        assert not _should_suppress_provider_error_final(
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY, [],
        )

    def test_no_suppression_for_a_different_error_category(self):
        assert not _should_suppress_provider_error_final(
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY,
            [_GATEWAY_PROVIDER_ERROR_AUTH_REPLY],
        )

    def test_never_suppresses_non_canonical_text(self):
        """Only the closed reply set qualifies — ordinary content that merely
        equals a delivered status must never be suppressed."""
        assert not _should_suppress_provider_error_final(
            "working on it", ["working on it"],
        )


# ---------------------------------------------------------------------------
# Status-bridge recording (TurnRunner._status_callback_sync)
# ---------------------------------------------------------------------------


class _FallbackAdapter:
    """Adapter WITHOUT send_or_update_status — statuses go through plain send."""

    async def send(self, chat_id, content, metadata=None):  # pragma: no cover
        raise AssertionError("send is not reached: scheduling is stubbed")


class _EditableAdapter(_FallbackAdapter):
    """Adapter WITH send_or_update_status — bubble semantics are its own."""

    async def send_or_update_status(self, chat_id, status_key, content, metadata=None):
        raise AssertionError("not reached")  # pragma: no cover


def _source(chat_id="-1001"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_id="12345",
    )


def _turn_runner(adapter, chat_id="-1001"):
    ctx = TurnContext(source=_source(chat_id), _run_still_current=lambda: True)
    ctx._status_adapter = adapter
    ctx._status_chat_id = chat_id
    return TurnRunner(runner=MagicMock(), ctx=ctx), ctx


def _stub_scheduler(monkeypatch, send_result):
    """Replace safe_schedule_threadsafe with a synchronous completed future."""
    sent = []

    def _fake_schedule(coro, loop, logger=None, log_message=None):
        sent.append(coro)
        coro.close()
        fut = concurrent.futures.Future()
        fut.set_result(send_result)
        return fut

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", _fake_schedule)
    return sent


RAW_RATE_LIMIT_STATUS = "Rate limited after 3 retries"


class TestStatusBridgeRecording:
    def test_successful_fallback_send_records_the_rewritten_reply(self, monkeypatch):
        runner, ctx = _turn_runner(_FallbackAdapter())
        sent = _stub_scheduler(
            monkeypatch, SendResult(success=True, message_id="77"),
        )

        runner._status_callback_sync("status", RAW_RATE_LIMIT_STATUS)

        assert len(sent) == 1
        assert ctx._provider_error_statuses_delivered == [
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
        ]

    def test_failed_send_records_nothing(self, monkeypatch):
        runner, ctx = _turn_runner(_FallbackAdapter())
        _stub_scheduler(monkeypatch, SendResult(success=False, message_id=None, error="flood wait"))

        runner._status_callback_sync("status", RAW_RATE_LIMIT_STATUS)

        assert ctx._provider_error_statuses_delivered == []

    def test_send_or_update_status_adapter_records_nothing(self, monkeypatch):
        """The gateway cannot assume an editable/ephemeral bubble persists, so
        the final response must keep flowing on these adapters."""
        runner, ctx = _turn_runner(_EditableAdapter())
        _stub_scheduler(monkeypatch, SendResult(success=True, message_id="78"))

        runner._status_callback_sync("status", RAW_RATE_LIMIT_STATUS)

        assert ctx._provider_error_statuses_delivered == []

    def test_ordinary_status_text_records_nothing(self, monkeypatch):
        runner, ctx = _turn_runner(_FallbackAdapter())
        sent = _stub_scheduler(
            monkeypatch, SendResult(success=True, message_id="79"),
        )

        runner._status_callback_sync("status", "reading the config file")

        assert len(sent) == 1  # delivered normally
        assert ctx._provider_error_statuses_delivered == []

    def test_interleaved_turns_keep_separate_state(self, monkeypatch):
        """Per-turn scoping: state lives on the TurnContext, so a concurrent
        turn (own ctx) never inherits or consumes another turn's record."""
        runner_a, ctx_a = _turn_runner(_FallbackAdapter(), chat_id="-1001")
        runner_b, ctx_b = _turn_runner(_FallbackAdapter(), chat_id="-2002")
        _stub_scheduler(monkeypatch, SendResult(success=True, message_id="80"))

        runner_a._status_callback_sync("status", RAW_RATE_LIMIT_STATUS)

        assert ctx_a._provider_error_statuses_delivered == [
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
        ]
        assert ctx_b._provider_error_statuses_delivered == []


# ---------------------------------------------------------------------------
# Delivery-level suppression (_handle_message_with_agent)
# ---------------------------------------------------------------------------


def _event():
    return MessageEvent(
        text="hello",
        source=_source(),
        message_id="msg-72131",
    )


def _gateway_runner(monkeypatch, tmp_path):
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
        session_id="sess-72131",
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


RAW_FAILED_FINAL = "⚠️ API call failed: rate limited after 3 retries"


def _failed_agent_result(**extra):
    result = {
        "final_response": RAW_FAILED_FINAL,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": RAW_FAILED_FINAL},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": True,
    }
    result.update(extra)
    return result


@pytest.mark.asyncio
async def test_duplicate_provider_error_final_is_suppressed(monkeypatch, tmp_path):
    """The outer path re-derives the sanitized reply from the raw result, so
    the suppression must live there — and does: an identical delivered status
    turns the final delivery into ""."""
    runner = _gateway_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_failed_agent_result(
        provider_error_statuses_delivered=[
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
        ],
    ))

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == ""


@pytest.mark.asyncio
async def test_transcript_persistence_is_untouched_by_the_suppression(
    monkeypatch, tmp_path,
):
    """Like intentional silence, this is a delivery decision — the turn's
    transcript rows are written exactly as without the dedup."""
    runner = _gateway_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_failed_agent_result(
        provider_error_statuses_delivered=[
            _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
        ],
    ))

    await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )
    suppressed_rows = [
        c.args[1] for c in runner.session_store.append_to_transcript.call_args_list
    ]

    runner2 = _gateway_runner(monkeypatch, tmp_path)
    runner2._run_agent = AsyncMock(return_value=_failed_agent_result())
    await runner2._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )
    plain_rows = [
        c.args[1] for c in runner2.session_store.append_to_transcript.call_args_list
    ]

    assert [
        (r.get("role"), r.get("content")) for r in suppressed_rows
    ] == [(r.get("role"), r.get("content")) for r in plain_rows]


@pytest.mark.asyncio
async def test_without_delivered_status_the_error_reply_is_still_sent(
    monkeypatch, tmp_path,
):
    runner = _gateway_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_failed_agent_result())

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY


@pytest.mark.asyncio
async def test_mismatched_error_category_is_not_suppressed(monkeypatch, tmp_path):
    """A delivered auth-error status must not swallow a rate-limit final."""
    runner = _gateway_runner(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value=_failed_agent_result(
        provider_error_statuses_delivered=[_GATEWAY_PROVIDER_ERROR_AUTH_REPLY],
    ))

    response = await runner._handle_message_with_agent(
        _event(), _source(), "agent:main:telegram:group:-1001:12345", 1
    )

    assert response == _GATEWAY_PROVIDER_ERROR_RATE_LIMIT_REPLY
