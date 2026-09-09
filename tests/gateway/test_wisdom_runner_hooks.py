"""Production runner integration for Wisdom without live messaging transports."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import run as runner_module, run_wisdom
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = object()
    return runner


@pytest.fixture
def source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="7")


def event(source, text, *, internal=False):
    return MessageEvent(source=source, text=text, message_type=MessageType.TEXT, internal=internal)


@pytest.mark.asyncio
@pytest.mark.parametrize("text,arguments", [
    ("/wisdom browse", "browse"),
    ("/collective-wisdom-install example", "install example"),
])
async def test_runner_dispatches_wisdom_to_native_adapter(runner, source, text, arguments):
    adapter = SimpleNamespace(send_wisdom_command=AsyncMock())
    runner._adapter_for_source = Mock(return_value=adapter)
    handler = runner._gateway_plain_command_handlers()["wisdom"]
    assert await handler(event(source, text)) == ""
    adapter.send_wisdom_command.assert_awaited_once_with(arguments, source=source)


@pytest.mark.asyncio
async def test_start_continuation_checks_wisdom_access_before_adapter(runner, source):
    adapter = SimpleNamespace(send_wisdom_continuation=AsyncMock())
    runner._adapter_for_source = Mock(return_value=adapter)
    runner._check_slash_access = Mock(return_value="Not allowed")
    start = event(source, "/start wisdom_private-token")

    assert await runner._hm_cmd_start(start, source, "session-key") == (True, "Not allowed")
    runner._check_slash_access.assert_called_once_with(source, "wisdom")
    adapter.send_wisdom_continuation.assert_not_awaited()

    runner._check_slash_access.return_value = None
    assert await runner._hm_cmd_start(start, source, "session-key") == (True, "")
    adapter.send_wisdom_continuation.assert_awaited_once_with("private-token", source=source)


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_post_turn_dispatches_exact_session_with_activity_origin(runner, source, internal):
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
    )
    runner._post_turn_loop_completion = AsyncMock()
    runner._post_turn_goal_continuation = AsyncMock()
    runner._defer_wisdom_candidate_notice_after_delivery = AsyncMock()
    await runner._run_post_turn_hooks(
        agent_result={"final_response": "Done"}, source=source, is_internal=internal,
    )
    runner._defer_wisdom_candidate_notice_after_delivery.assert_awaited_once_with(
        source, "session-1", user_activity=not internal,
    )
    runner.async_session_store.get_or_create_session.assert_awaited_once_with(source, touch_activity=not internal)


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [False, True])
async def test_resolved_user_session_is_observed_without_starting_assessment(runner, source, monkeypatch, internal):
    entry = SimpleNamespace(session_key="session-key", session_id="session-1")
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, get_or_create_session=AsyncMock(return_value=entry),
    )
    runner._recover_telegram_topic_thread_id = Mock(return_value=None)
    runner._cache_session_source = Mock()
    runner._is_telegram_topic_lane = Mock(return_value=False)
    adapter = object()
    runner._adapter_for_source = Mock(return_value=adapter)
    monkeypatch.setattr("gateway.run_heartbeat_acceptance.resolve_heartbeat_owner", AsyncMock(return_value=True))
    observe = AsyncMock()
    monkeypatch.setattr("gateway.wisdom_mediation.schedule", observe)

    resolved = await runner._hmwa_resolve_session(event(source, "hello", internal=internal), source)
    assert resolved == (source, entry, "session-key")
    if internal:
        observe.assert_not_awaited()
    else:
        observe.assert_awaited_once_with(runner, adapter, source, "session-1", observe_only=True)


def test_housekeeping_registers_card_refresh_and_local_weekly_queue(monkeypatch):
    refresh = Mock()
    queue = Mock()
    monkeypatch.setattr(runner_module, "WisdomCardRefresh", Mock(return_value=refresh))
    monkeypatch.setattr(runner_module, "enqueue_weekly_review", queue)

    def run_wisdom_only(label, fn):
        if label.startswith("Wisdom "):
            fn()

    monkeypatch.setattr(runner_module, "_housekeeping_chore", run_wisdom_only)

    class SixtyTicks:
        ticks = 0

        def is_set(self):
            return self.ticks == 60

        def wait(self, timeout):
            self.ticks += 1

    runner_module._start_gateway_housekeeping(SixtyTicks(), interval=1)
    assert refresh.tick.call_count == 60
    queue.assert_called_once_with()


def test_publication_refresh_does_not_overlap_and_recovers_from_failed_pass(monkeypatch):
    submitted = []
    adapters, loop = object(), object()

    def submit(coro, requested_loop):
        assert requested_loop is loop
        coro.close()
        pending = Future()
        submitted.append(pending)
        return pending

    monkeypatch.setattr(run_wisdom.asyncio, "run_coroutine_threadsafe", submit)
    refresher = run_wisdom.WisdomCardRefresh(adapters, loop)
    refresher.tick()
    refresher.tick()
    assert len(submitted) == 1
    submitted[0].set_exception(RuntimeError("temporary send failure"))
    refresher.tick()
    assert len(submitted) == 2
    submitted[1].set_result(None)
    refresher.tick()
    assert len(submitted) == 3
