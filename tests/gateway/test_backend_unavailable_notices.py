"""Regression coverage for backend-unavailable notices at the real gateway boundary."""

import asyncio
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BackendNoticeState,
    BackendUnavailableReply,
    _LLM_CONNECTION_ERROR_COOLDOWN_SECONDS,
    _LLM_ERROR_TRACKER_MAX_SESSIONS,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionEntry, SessionSource, build_session_key


_NOTICE = (
    "The AI backend is temporarily unavailable. "
    "Please try sending your message again in a moment."
)


class CaptureSlackAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token"), Platform.SLACK
        )
        self.sent = []
        self.processing_hooks = []
        self.delivery_results = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"slack-{len(self.sent)}")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.processing_hooks.append(("start", event.message_id))

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        self.processing_hooks.append(("complete", event.message_id, outcome))


class FailingDeliverySlackAdapter(CaptureSlackAdapter):
    async def _send_with_retry(self, *args, **kwargs) -> SendResult:
        result = self.delivery_results.pop(0)
        if result.success:
            self.sent.append(
                {
                    "chat_id": kwargs["chat_id"],
                    "content": kwargs["content"],
                    "reply_to": kwargs.get("reply_to"),
                    "metadata": kwargs.get("metadata"),
                }
            )
        return result


def _backend_failure(error: str) -> dict:
    return {
        "final_response": f"API call failed after 3 retries: {error}",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "history_offset": 0,
        "api_calls": 3,
        "failed": True,
        "failure_reason": "timeout",
        "completed": False,
        "interrupted": False,
        "error": error,
        "last_prompt_tokens": 0,
    }


def _successful_result(text: str = "normal response") -> dict:
    return {
        "final_response": text,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": text},
        ],
        "tools": [],
        "history_offset": 0,
        "api_calls": 1,
        "failed": False,
        "completed": True,
        "interrupted": False,
        "last_prompt_tokens": 0,
    }


def _make_runner(adapter: CaptureSlackAdapter, results: list[dict]) -> gateway_run.GatewayRunner:
    runner = cast(Any, object.__new__(gateway_run.GatewayRunner))
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, token="fake-token")}
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner._backend_notice_state = BackendNoticeState()
    runner._share_backend_notice_state(adapter, profile_name="default")
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:slack:channel:C123:171717",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.SLACK,
        chat_type="channel",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store.has_platform_message_id = MagicMock(return_value=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda source: True
    runner._set_session_env = lambda context: []
    runner._run_agent = AsyncMock(side_effect=results)
    return runner


def _make_event(message_id: str) -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="channel",
            thread_id="171717",
            user_id="U123",
        ),
        message_id=message_id,
    )


def _wire_runner(monkeypatch, tmp_path, adapter, results):
    runner = _make_runner(adapter, results)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *_args, **_kwargs: 100
    )
    monkeypatch.setenv("SLACK_HOME_CHANNEL", "C123")
    adapter.set_message_handler(runner._handle_message)
    adapter._keep_typing = lambda *_args, **_kwargs: asyncio.Event().wait()
    return runner


@pytest.mark.asyncio
async def test_real_gateway_path_sanitizes_backend_transport_failure(monkeypatch, tmp_path):
    raw_error = "APIConnectionError: connection refused at https://api.example.com/v1"
    adapter = CaptureSlackAdapter()
    _wire_runner(monkeypatch, tmp_path, adapter, [_backend_failure(raw_error)])

    event = _make_event("m-1")
    await adapter._process_message_background(event, build_session_key(event.source))

    assert [message["content"] for message in adapter.sent] == [_NOTICE]
    assert "api.example.com" not in adapter.sent[0]["content"]
    assert adapter.processing_hooks[-1] == (
        "complete",
        "m-1",
        ProcessingOutcome.FAILURE,
    )


@pytest.mark.asyncio
async def test_mixed_transport_exceptions_share_one_cooldown(
    monkeypatch, tmp_path, caplog
):
    adapter = CaptureSlackAdapter()
    _wire_runner(
        monkeypatch,
        tmp_path,
        adapter,
        [
            _backend_failure("ConnectError: connection refused"),
            _backend_failure("APITimeoutError: read timed out"),
        ],
    )

    for message_id in ("m-1", "m-2"):
        event = _make_event(message_id)
        await adapter._process_message_background(event, build_session_key(event.source))

    assert [message["content"] for message in adapter.sent] == [_NOTICE]
    assert "response_delivery_dropped" not in caplog.text
    assert [hook for hook in adapter.processing_hooks if hook[0] == "complete"] == [
        ("complete", "m-1", ProcessingOutcome.FAILURE),
        ("complete", "m-2", ProcessingOutcome.FAILURE),
    ]


@pytest.mark.asyncio
async def test_notice_posts_again_after_cooldown(monkeypatch, tmp_path):
    adapter = CaptureSlackAdapter()
    _wire_runner(
        monkeypatch,
        tmp_path,
        adapter,
        [
            _backend_failure("ConnectError: connection refused"),
            _backend_failure("ConnectError: connection refused"),
        ],
    )

    first = _make_event("m-1")
    await adapter._process_message_background(first, build_session_key(first.source))
    (session_key,) = adapter._llm_error_last_posted
    kind, timestamp = adapter._llm_error_last_posted[session_key]
    adapter._llm_error_last_posted[session_key] = (
        kind,
        timestamp - _LLM_CONNECTION_ERROR_COOLDOWN_SECONDS - 1.0,
    )

    second = _make_event("m-2")
    await adapter._process_message_background(second, build_session_key(second.source))

    assert [message["content"] for message in adapter.sent] == [_NOTICE, _NOTICE]


@pytest.mark.asyncio
async def test_failed_notice_delivery_does_not_arm_cooldown(monkeypatch, tmp_path):
    adapter = FailingDeliverySlackAdapter()
    adapter.delivery_results = [
        SendResult(success=False, error="Slack unavailable"),
        SendResult(success=True, message_id="slack-2"),
    ]
    _wire_runner(
        monkeypatch,
        tmp_path,
        adapter,
        [
            _backend_failure("ConnectError: connection refused"),
            _backend_failure("ConnectError: connection refused"),
        ],
    )

    for message_id in ("m-1", "m-2"):
        event = _make_event(message_id)
        await adapter._process_message_background(event, build_session_key(event.source))

    assert [message["content"] for message in adapter.sent] == [_NOTICE]


@pytest.mark.asyncio
async def test_cancelled_notice_delivery_releases_inflight_claim(monkeypatch, tmp_path):
    adapter = CaptureSlackAdapter()
    _wire_runner(
        monkeypatch,
        tmp_path,
        adapter,
        [_backend_failure("ConnectError: connection refused")],
    )
    event = _make_event("m-1")
    session_key = build_session_key(event.source)

    async def cancel_delivery(*_args, **_kwargs):
        raise asyncio.CancelledError

    adapter._send_with_retry = cancel_delivery

    with pytest.raises(asyncio.CancelledError):
        await adapter._process_message_background(event, session_key)

    assert adapter._backend_notice_state.inflight == set()
    assert adapter._llm_error_last_posted == {}


@pytest.mark.asyncio
async def test_reconnect_rechecks_cooldown_on_adapter_that_will_send(
    monkeypatch, tmp_path
):
    stale_adapter = CaptureSlackAdapter()
    runner = _wire_runner(
        monkeypatch,
        tmp_path,
        stale_adapter,
        [_backend_failure("ConnectError: connection refused")],
    )
    cast(Any, stale_adapter).gateway_runner = runner

    replacement = CaptureSlackAdapter()
    cast(Any, replacement).gateway_runner = runner
    replacement.set_backend_notice_state(runner._backend_notice_state)
    event = _make_event("m-1")
    session_key = build_session_key(event.source)
    replacement._record_llm_error_notice(
        session_key,
        "backend_unavailable",
        time.monotonic(),
    )

    original_unwrap = stale_adapter._unwrap_ephemeral

    def replace_before_delivery(response):
        runner.adapters[Platform.SLACK] = replacement
        return original_unwrap(response)

    stale_adapter._unwrap_ephemeral = replace_before_delivery

    await stale_adapter._process_message_background(event, session_key)

    assert replacement.sent == []
    assert stale_adapter.sent == []


@pytest.mark.asyncio
async def test_concurrent_adapter_generations_share_one_inflight_claim(
    monkeypatch, tmp_path
):
    first = CaptureSlackAdapter()
    runner = _wire_runner(
        monkeypatch,
        tmp_path,
        first,
        [_backend_failure("ConnectError: connection refused")],
    )
    cast(Any, first).gateway_runner = runner

    second = CaptureSlackAdapter()
    second.set_backend_notice_state(runner._backend_notice_state)
    cast(Any, second).gateway_runner = runner
    second.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    second._keep_typing = keep_typing

    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def hold_first_send(*args, **kwargs):
        send_started.set()
        await release_send.wait()
        first.sent.append({"content": kwargs["content"]})
        return SendResult(success=True, message_id="first")

    first._send_with_retry = hold_first_send
    first_event = _make_event("m-1")
    second_event = _make_event("m-2")
    session_key = build_session_key(first_event.source)

    first_task = asyncio.create_task(
        first._process_message_background(first_event, session_key)
    )
    await send_started.wait()
    runner.adapters[Platform.SLACK] = second
    second_task = asyncio.create_task(
        second._process_message_background(second_event, session_key)
    )
    await asyncio.sleep(0)
    assert second_task.done() is False
    release_send.set()
    await asyncio.gather(first_task, second_task)

    assert len(first.sent) == 1
    assert second.sent == []


@pytest.mark.asyncio
async def test_replacement_retries_notice_after_inflight_owner_delivery_fails(
    monkeypatch, tmp_path
):
    first = CaptureSlackAdapter()
    runner = _wire_runner(
        monkeypatch,
        tmp_path,
        first,
        [_backend_failure("ConnectError: connection refused")],
    )
    cast(Any, first).gateway_runner = runner

    second = CaptureSlackAdapter()
    second.set_backend_notice_state(runner._backend_notice_state)
    cast(Any, second).gateway_runner = runner
    second.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    second._keep_typing = keep_typing

    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def fail_first_send(*_args, **_kwargs):
        send_started.set()
        await release_send.wait()
        return SendResult(success=False, error="stale transport disconnected")

    first._send_with_retry = fail_first_send
    first_event = _make_event("m-1")
    second_event = _make_event("m-2")
    session_key = build_session_key(first_event.source)

    first_task = asyncio.create_task(
        first._process_message_background(first_event, session_key)
    )
    await send_started.wait()
    runner.adapters[Platform.SLACK] = second
    second_task = asyncio.create_task(
        second._process_message_background(second_event, session_key)
    )
    await asyncio.sleep(0)
    assert second_task.done() is False

    release_send.set()
    await asyncio.gather(first_task, second_task)

    assert first.sent == []
    assert [message["content"] for message in second.sent] == [_NOTICE]


@pytest.mark.asyncio
async def test_cancelled_owner_waits_for_ambiguous_send_success_before_suppressing(
    monkeypatch, tmp_path
):
    first = CaptureSlackAdapter()
    runner = _wire_runner(
        monkeypatch,
        tmp_path,
        first,
        [_backend_failure("ConnectError: connection refused")],
    )
    cast(Any, first).gateway_runner = runner

    second = CaptureSlackAdapter()
    second.set_backend_notice_state(runner._backend_notice_state)
    cast(Any, second).gateway_runner = runner
    second.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    second._keep_typing = keep_typing
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def ambiguous_send(*_args, **kwargs):
        send_started.set()
        await release_send.wait()
        first.sent.append({"content": kwargs["content"]})
        return SendResult(success=True, message_id="first")

    first._send_with_retry = ambiguous_send
    first_event = _make_event("m-1")
    second_event = _make_event("m-2")
    session_key = build_session_key(first_event.source)
    first_task = asyncio.create_task(
        first._process_message_background(first_event, session_key)
    )
    await send_started.wait()
    runner.adapters[Platform.SLACK] = second
    second_task = asyncio.create_task(
        second._process_message_background(second_event, session_key)
    )
    await asyncio.sleep(0)

    first_task.cancel()
    await asyncio.sleep(0)
    assert first_task.done() is False
    assert second_task.done() is False
    release_send.set()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    await second_task

    assert [message["content"] for message in first.sent] == [_NOTICE]
    assert second.sent == []


@pytest.mark.asyncio
async def test_cancelled_owner_releases_waiter_after_definite_send_failure(
    monkeypatch, tmp_path
):
    first = CaptureSlackAdapter()
    runner = _wire_runner(
        monkeypatch,
        tmp_path,
        first,
        [_backend_failure("ConnectError: connection refused")],
    )
    cast(Any, first).gateway_runner = runner

    second = CaptureSlackAdapter()
    second.set_backend_notice_state(runner._backend_notice_state)
    cast(Any, second).gateway_runner = runner
    second.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    second._keep_typing = keep_typing
    send_started = asyncio.Event()
    release_send = asyncio.Event()

    async def definite_failure(*_args, **_kwargs):
        send_started.set()
        await release_send.wait()
        return SendResult(success=False, error="stale transport disconnected")

    first._send_with_retry = definite_failure
    first_event = _make_event("m-1")
    second_event = _make_event("m-2")
    session_key = build_session_key(first_event.source)
    first_task = asyncio.create_task(
        first._process_message_background(first_event, session_key)
    )
    await send_started.wait()
    runner.adapters[Platform.SLACK] = second
    second_task = asyncio.create_task(
        second._process_message_background(second_event, session_key)
    )
    await asyncio.sleep(0)

    first_task.cancel()
    await asyncio.sleep(0)
    assert first_task.done() is False
    assert second_task.done() is False
    release_send.set()
    with pytest.raises(asyncio.CancelledError):
        await first_task
    await second_task

    assert first.sent == []
    assert [message["content"] for message in second.sent] == [_NOTICE]


@pytest.mark.asyncio
async def test_shared_runner_state_does_not_cross_suppress_profiles():
    state = BackendNoticeState()
    first = CaptureSlackAdapter()
    second = CaptureSlackAdapter()
    first.set_backend_notice_state(state, profile_name="alpha")
    second.set_backend_notice_state(state, profile_name="beta")
    first.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))
    second.set_message_handler(AsyncMock(return_value=BackendUnavailableReply(_NOTICE)))

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    first._keep_typing = keep_typing
    second._keep_typing = keep_typing
    first_event = _make_event("m-alpha")
    second_event = _make_event("m-beta")
    raw_session_key = build_session_key(first_event.source)

    await first._process_message_background(first_event, raw_session_key)
    await second._process_message_background(second_event, raw_session_key)

    assert [message["content"] for message in first.sent] == [_NOTICE]
    assert [message["content"] for message in second.sent] == [_NOTICE]
    assert set(state.posted) == {
        "profile:5:alpha:agent:main:slack:channel:C123:171717",
        "profile:4:beta:agent:main:slack:channel:C123:171717",
    }


@pytest.mark.asyncio
async def test_default_profile_does_not_collide_with_named_main_profile():
    state = BackendNoticeState()
    default_adapter = CaptureSlackAdapter()
    named_main_adapter = CaptureSlackAdapter()
    default_adapter.set_backend_notice_state(state, profile_name="default")
    named_main_adapter.set_backend_notice_state(state, profile_name="main")
    default_adapter.set_message_handler(
        AsyncMock(return_value=BackendUnavailableReply(_NOTICE))
    )
    named_main_adapter.set_message_handler(
        AsyncMock(return_value=BackendUnavailableReply(_NOTICE))
    )

    async def keep_typing(*_args, **_kwargs):
        await asyncio.Event().wait()

    default_adapter._keep_typing = keep_typing
    named_main_adapter._keep_typing = keep_typing
    default_event = _make_event("m-default")
    named_main_event = _make_event("m-main")
    raw_session_key = build_session_key(default_event.source)

    await default_adapter._process_message_background(default_event, raw_session_key)
    await named_main_adapter._process_message_background(
        named_main_event, raw_session_key
    )

    assert [message["content"] for message in default_adapter.sent] == [_NOTICE]
    assert [message["content"] for message in named_main_adapter.sent] == [_NOTICE]
    assert set(state.posted) == {
        "profile:7:default:agent:main:slack:channel:C123:171717",
        "profile:4:main:agent:main:slack:channel:C123:171717",
    }


def test_shared_notice_state_wiring_attaches_runner_for_direct_builtins():
    adapter = CaptureSlackAdapter()
    runner = cast(Any, object.__new__(gateway_run.GatewayRunner))
    runner._backend_notice_state = BackendNoticeState()

    runner._share_backend_notice_state(adapter, profile_name="default")

    assert cast(Any, adapter).gateway_runner is runner


@pytest.mark.asyncio
async def test_backend_notice_skips_auto_tts(monkeypatch, tmp_path):
    adapter = CaptureSlackAdapter()
    _wire_runner(
        monkeypatch,
        tmp_path,
        adapter,
        [_backend_failure("ConnectError: connection refused")],
    )
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.play_tts = AsyncMock(
        side_effect=AssertionError("backend notice must stay on text delivery path")
    )
    event = _make_event("m-1")
    event.message_type = MessageType.VOICE

    await adapter._process_message_background(
        event,
        build_session_key(event.source),
    )

    adapter.play_tts.assert_not_awaited()
    assert [message["content"] for message in adapter.sent] == [_NOTICE]


@pytest.mark.asyncio
async def test_platform_send_connection_error_is_not_backend_outage(monkeypatch, tmp_path):
    adapter = CaptureSlackAdapter()
    _wire_runner(monkeypatch, tmp_path, adapter, [_successful_result()])

    async def fail_platform_delivery(*args, **kwargs):
        raise ConnectionError(
            "Slack socket disconnected at "
            "https://hooks.slack.com/services/private?token=super-secret-token"
        )

    adapter._send_with_retry = fail_platform_delivery
    event = _make_event("m-1")
    await adapter._process_message_background(event, build_session_key(event.source))

    contents = [message["content"] for message in adapter.sent]
    assert _NOTICE not in contents
    assert contents == [
        "Sorry, I encountered an internal error. "
        "Please try again or use /reset to start a fresh session."
    ]
    assert "hooks.slack.com" not in contents[0]
    assert "super-secret-token" not in contents[0]


@pytest.mark.parametrize(
    ("failure_reason", "expected"),
    [
        ("timeout", True),
        ("overloaded", True),
        ("server_error", True),
        ("rate_limit", False),
        ("auth_permanent", False),
        ("billing", False),
        ("context_overflow", False),
    ],
)
def test_only_transient_backend_failures_get_outage_marker(failure_reason, expected):
    result = _backend_failure("provider failed")
    result["failure_reason"] = failure_reason
    assert gateway_run._is_backend_unavailable_agent_result(result) is expected


@pytest.mark.parametrize(
    "platform",
    [Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK, Platform.MSGRAPH_WEBHOOK],
)
def test_programmatic_surfaces_are_excluded_from_notice_marker(platform):
    assert gateway_run._gateway_surface_passes_raw_text(platform) is True


def test_notice_tracker_prunes_expired_sessions():
    adapter = CaptureSlackAdapter()
    now = time.monotonic()
    stale = now - _LLM_CONNECTION_ERROR_COOLDOWN_SECONDS - 1.0
    for index in range(50):
        adapter._llm_error_last_posted[f"stale-{index}"] = (
            "backend_unavailable",
            stale,
        )

    adapter._record_llm_error_notice("live", "backend_unavailable", now)

    assert set(adapter._llm_error_last_posted) == {"profile:7:default:live"}


def test_notice_tracker_is_bounded():
    adapter = CaptureSlackAdapter()
    now = time.monotonic()
    overflow = _LLM_ERROR_TRACKER_MAX_SESSIONS + 200

    for index in range(overflow):
        adapter._record_llm_error_notice(
            f"session-{index}",
            "backend_unavailable",
            now + index * 0.001,
        )

    assert len(adapter._llm_error_last_posted) <= _LLM_ERROR_TRACKER_MAX_SESSIONS
    assert (
        f"profile:7:default:session-{overflow - 1}"
        in adapter._llm_error_last_posted
    )
    assert "session-0" not in adapter._llm_error_last_posted
