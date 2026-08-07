from __future__ import annotations

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import TurnRunner, _is_gateway_failure_delivery, _sanitize_gateway_final_response
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext


class ErrorPolicyAdapter(BasePlatformAdapter):
    def __init__(self, *, alert_success: bool = True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.WHATSAPP)
        self.sent: list[dict[str, str]] = []
        self.alert_success = alert_success

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(
            success=self.alert_success,
            message_id="sent-1" if self.alert_success else None,
            error=None if self.alert_success else "bridge rejected alert",
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def event(*, governed: bool) -> MessageEvent:
    metadata = {}
    if governed:
        metadata["gateway_error_policy"] = {
            "suppress_reply": True,
            "alert_chat_id": "control-room@g.us",
            "alert_message": (
                "⚠️ Nova could not complete a management request from a job-card group. "
                "No technical error was posted there. Please retry in this control room."
            ),
        }
    return MessageEvent(
        text="Nova, push this card to the app",
        message_id="m1",
        source=SessionSource(
            platform=Platform.WHATSAPP,
            user_id="maff@s.whatsapp.net",
            chat_id="job-card@g.us",
            chat_type="group",
        ),
        metadata=metadata,
    )


@pytest.mark.parametrize(
    "provider_status",
    [
        "Provider authentication failed. Check the configured credentials.",
        "❌ Non-retryable error (HTTP 401): token rejected",
        "❌ Rate limited after 3 retries — too many requests",
        "❌ API failed after 3 retries — provider unavailable",
    ],
)
def test_governed_provider_status_is_suppressed_before_source_send(monkeypatch, provider_status):
    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="maff@s.whatsapp.net",
        chat_id="job-card@g.us",
        chat_type="group",
    )
    source._gateway_error_policy = {
        "suppress_reply": True,
        "alert_chat_id": "control-room@g.us",
        "alert_message": "Management request incomplete. Check the app before retrying.",
    }
    scheduled = []

    def fake_schedule(coro, *_args, **_kwargs):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", fake_schedule)
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: True,
        _status_adapter=object(),
        _status_chat_id=source.chat_id,
    )

    TurnRunner(None, ctx)._status_callback_sync("provider_error", provider_status)

    assert scheduled == []


def test_governed_normal_status_still_schedules(monkeypatch):
    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id="maff@s.whatsapp.net",
        chat_id="job-card@g.us",
        chat_type="group",
    )
    source._gateway_error_policy = {"suppress_reply": True}
    scheduled = []

    def fake_schedule(coro, *_args, **_kwargs):
        scheduled.append(coro)
        coro.close()
        return None

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", fake_schedule)
    ctx = TurnContext(
        source=source,
        _run_still_current=lambda: True,
        _status_adapter=object(),
        _status_chat_id=source.chat_id,
    )

    TurnRunner(None, ctx)._status_callback_sync("working", "Checking the job card")

    assert len(scheduled) == 1


async def failing_handler(_event):
    raise RuntimeError("internal import mismatch")


@pytest.mark.asyncio
async def test_governed_error_is_silent_at_source_and_alerts_control_room():
    adapter = ErrorPolicyAdapter()
    adapter.set_message_handler(failing_handler)
    inbound = event(governed=True)

    await adapter._process_message_background(inbound, build_session_key(inbound.source))

    assert adapter.sent == [{
        "chat_id": "control-room@g.us",
        "content": (
            "⚠️ Nova could not complete a management request from a job-card group. "
            "No technical error was posted there. Please retry in this control room."
        ),
    }]


@pytest.mark.asyncio
async def test_ordinary_error_still_notifies_source_chat():
    adapter = ErrorPolicyAdapter()
    adapter.set_message_handler(failing_handler)
    inbound = event(governed=False)

    await adapter._process_message_background(inbound, build_session_key(inbound.source))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["chat_id"] == "job-card@g.us"
    assert "internal import mismatch" in adapter.sent[0]["content"]


@pytest.mark.asyncio
async def test_governed_error_policy_can_follow_session_source_into_queued_delivery():
    adapter = ErrorPolicyAdapter()
    source = event(governed=False).source
    setattr(source, "_gateway_error_policy", {
        "suppress_reply": True,
        "alert_chat_id": "control-room@g.us",
        "alert_message": "Nova could not complete a queued management request.",
    })

    suppressed = await adapter._handle_governed_error(source)

    assert suppressed is True
    assert adapter.sent == [
        {
            "chat_id": "control-room@g.us",
            "content": "Nova could not complete a queued management request.",
        }
    ]


@pytest.mark.asyncio
async def test_failed_control_room_delivery_is_logged_without_replying_to_source(caplog):
    adapter = ErrorPolicyAdapter(alert_success=False)
    inbound = event(governed=True)

    suppressed = await adapter._handle_governed_error(inbound)

    assert suppressed is True
    assert [message["chat_id"] for message in adapter.sent] == ["control-room@g.us"]
    assert "Governed error alert delivery failed: bridge rejected alert" in caplog.text


@pytest.mark.parametrize(
    ("agent_result", "response_was_empty", "normalized_response", "expected"),
    [
        ({"failed": True}, False, "partial provider text", True),
        ({"api_calls": 0}, True, "⚠️ Your message wasn't processed. Please retry.", True),
        ({"api_calls": 1}, False, "⚠️ Provider authentication failed. Check the configured credentials.", True),
        ({"api_calls": 1}, False, "HTTP 401: invalid provider credentials", True),
        ({"api_calls": 1}, False, "⚠️ I couldn't complete this request because the AI service connection needs attention. Nothing was changed.", True),
        ({"api_calls": 1}, False, "Card submitted.", False),
        ({"partial": True}, True, "", False),
    ],
)
def test_gateway_failure_delivery_covers_all_generated_retry_text(
    agent_result,
    response_was_empty,
    normalized_response,
    expected,
):
    assert _is_gateway_failure_delivery(
        agent_result,
        response_was_empty=response_was_empty,
        normalized_response=normalized_response,
    ) is expected


def test_verbose_provider_envelope_is_still_classified_after_sanitization():
    raw = (
        "HTTP 401: invalid provider credentials\n"
        + "provider diagnostic context that must never reach WhatsApp\n" * 20
    )

    sanitized = _sanitize_gateway_final_response(Platform.WHATSAPP, raw)

    assert len(raw) > 400
    assert "provider diagnostic" not in sanitized
    assert _is_gateway_failure_delivery(
        {"api_calls": 1},
        response_was_empty=False,
        normalized_response=sanitized,
    ) is True


def test_long_normal_answer_that_mentions_http_mid_paragraph_is_not_rewritten():
    answer = "Here is the explanation. HTTP 404 means not found. " + ("More context. " * 60)

    assert _sanitize_gateway_final_response(Platform.WHATSAPP, answer) == answer


@pytest.mark.asyncio
async def test_governed_error_alert_is_sent_at_most_once_per_event():
    adapter = ErrorPolicyAdapter()
    inbound = event(governed=True)

    assert await adapter._handle_governed_error(inbound) is True
    assert await adapter._handle_governed_error(inbound) is True

    assert adapter.sent == [{
        "chat_id": "control-room@g.us",
        "content": (
            "⚠️ Nova could not complete a management request from a job-card group. "
            "No technical error was posted there. Please retry in this control room."
        ),
    }]
