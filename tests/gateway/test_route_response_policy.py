"""Silent profile routes run turns while producing no gateway reply text."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnContext, TurnRunner
from gateway.session import SessionSource
from gateway.session_identity import RoutingIdentity


def _silent_source(tmp_path):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="3232")
    source._identity = RoutingIdentity(
        transport_profile="default",
        runtime_profile="cfo",
        authorization_home=tmp_path,
        runtime_home=tmp_path / "profiles" / "cfo",
        response_policy="silent",
    )
    return source


class _Runner(GatewayTurnMixin):
    def __init__(self):
        self.async_session_store = SimpleNamespace(
            clear_resume_pending=self._noop,
            has_input_owner=self._false,
            append_to_transcript=self._noop,
        )

    async def _noop(self, *_a, **_k):
        return None

    async def _false(self, *_a, **_k):
        return False

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None

    async def _hmwa_stop_typing_for_turn(self, *_a, **_k):
        return None

    async def _hmwa_close_failed_turn(self, *_a, **_k):
        return None

    async def _refresh_agent_cache_message_count(self, *_a, **_k):
        return None


@pytest.mark.parametrize("failed", [False, True])
def test_silent_route_uses_intentional_silence_for_success_and_agent_failure(
    tmp_path, failed
):
    result = {
        "final_response": "ordinary final text"
        if not failed
        else "provider failure text",
        "messages": [],
        "failed": failed,
    }

    async def shape_response():
        return await _Runner()._hmwa_shape_agent_response(
            result,
            _silent_source(tmp_path),
            history=[],
            session_entry=SimpleNamespace(session_id="s"),
            session_key=None,
            _quick_key=None,
            run_generation=0,
            _run_start_session_id="s",
            _platform_name="telegram",
            _msg_start_time=0.0,
        )

    response, intentional_silence, _ = asyncio.run(shape_response())

    assert intentional_silence is True
    assert response == ""


def test_silent_route_never_opens_text_or_interim_streams(tmp_path):
    ctx = TurnContext(source=_silent_source(tmp_path), message="run the full turn")
    runner = SimpleNamespace(config=SimpleNamespace())

    consumer, delta, interim, wants_interim = TurnRunner(
        runner, ctx
    )._setup_stream_consumer("telegram")

    assert (consumer, delta, interim, wants_interim) == (None, None, None, False)


def test_silent_route_never_opens_proxy_stream(tmp_path):
    runner = _Runner()
    runner.config = SimpleNamespace(streaming=StreamingConfig(enabled=True))
    runner._delivery_adapter_for = MagicMock(
        return_value=SimpleNamespace(
            SUPPORTS_MESSAGE_EDITING=True,
            SUPPORTS_NATIVE_STREAMING=False,
        )
    )

    consumer = runner._proxy_stream_consumer(
        _silent_source(tmp_path), None, None, lambda: True
    )

    assert consumer is None
    runner._delivery_adapter_for.assert_not_called()


def test_silent_route_never_starts_typing_refresh(tmp_path):
    adapter = SimpleNamespace(
        config=SimpleNamespace(typing_indicator=True),
        _keep_typing=AsyncMock(),
        _accepts_kwarg=MagicMock(return_value=False),
    )
    event = MessageEvent(text="run", source=_silent_source(tmp_path))

    async def start_typing():
        return BasePlatformAdapter._start_typing_refresh(
            adapter, event, asyncio.Event(), None
        )

    assert asyncio.run(start_typing()) is None
    adapter._accepts_kwarg.assert_not_called()


def test_silent_route_suppresses_tool_and_subagent_progress(tmp_path):
    ctx = TurnContext(source=_silent_source(tmp_path), message="run the full turn")
    turn_runner = TurnRunner(SimpleNamespace(config=SimpleNamespace()), ctx)
    turn_runner._progress_subagent_notice = MagicMock()
    turn_runner._progress_live_status = MagicMock()

    turn_runner.progress_callback(
        "subagent.complete", preview="failed", status="failed", goal="delegate"
    )
    turn_runner.progress_callback("tool.started", tool_name="terminal", args={"command": "true"})

    turn_runner._progress_subagent_notice.assert_not_called()
    turn_runner._progress_live_status.assert_not_called()


def test_silent_route_suppresses_first_response_before_queued_followup(tmp_path):
    runner = _Runner()
    runner._deliver_queued_first_response = AsyncMock(return_value=True)
    runner._pop_post_delivery_callback = MagicMock(return_value=None)
    ctx = TurnContext(
        source=_silent_source(tmp_path), message="first", session_key="key"
    )

    asyncio.run(
        runner._run_agent_deliver_first_response(
            ctx,
            SimpleNamespace(),
            {"final_response": "must not leak"},
            {"final_response": "must not leak"},
            None,
        )
    )

    runner._deliver_queued_first_response.assert_not_awaited()


def test_silent_route_keeps_unhandled_agent_exception_text_free(tmp_path):
    source = _silent_source(tmp_path)
    prepared = SimpleNamespace(
        history=[],
        message_text="do work",
        persistence_session_id="s",
        persistence_owner="owner",
        persist_user_message=None,
        persist_user_timestamp=None,
        persist_user_display_kind=None,
    )

    response = asyncio.run(
        _Runner()._hmwa_agent_error_reply(
            RuntimeError("boom"),
            SimpleNamespace(message_id="m"),
            source,
            SimpleNamespace(session_id="s"),
            "key",
            prepared,
        )
    )

    assert response == ""


def test_silent_route_closes_gateway_persisted_turn_for_role_alternation(tmp_path):
    runner = _Runner()
    runner._session_db = None
    runner.async_session_store.append_to_transcript = AsyncMock()
    runner.async_session_store.update_session = AsyncMock()
    source = _silent_source(tmp_path)
    event = MessageEvent(text="run", source=source, message_id="m")
    history = [{"role": "assistant", "content": "before"}]
    prepared = SimpleNamespace(
        history=history,
        persist_user_message=None,
        message_text="run",
        persist_user_timestamp=None,
        persist_user_display_kind=None,
        persistence_owner="owner",
    )

    asyncio.run(
        runner._hmwa_persist_turn_transcript(
            event=event,
            source=source,
            session_entry=SimpleNamespace(session_id="s", session_key="key"),
            session_key="key",
            agent_result={"agent_persisted": False},
            agent_messages=history,
            prepared=prepared,
            response="",
            intentional_silence=True,
            agent_failed_early=False,
            hidden_reasoning_incomplete=False,
            is_context_overflow_failure=False,
        )
    )

    rows = [
        entry.args[1]
        for entry in runner.async_session_store.append_to_transcript.await_args_list
    ]
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert rows[-1]["content"] == "[SILENT]"
