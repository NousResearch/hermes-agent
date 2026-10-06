"""Regression tests for the fresh-turn compression gate in _handle_message (#134239).

A post-turn compression starts ~0.4 s after "Turn ended", i.e. while
``_is_session_running`` is already False, so a follow-up message took the
fresh-turn path with the #56391 guard never firing: the turn it started read
the pre-rotation history and its own post-turn compression later committed a
snapshot taken before the previous commit — double-compressing the transcript
(352→48→64) while the session looked idle for ~5.5 minutes. The running-agent
path already demoted for this; the fresh-turn path must refuse too.
"""

import pytest
from unittest.mock import AsyncMock, patch

from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}

    async def send(self, *args, **kwargs):
        pass


def _make_runner(store) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._pending_messages = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = AsyncMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = AsyncMock()
    runner.session_store = store
    runner.delivery_router = AsyncMock()
    return runner


def _source(chat_id="555010") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=chat_id,
    )


def _store(tmp_path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_path, config=GatewayConfig())


def _event(src, text="follow-up after turn end", internal=False) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=src,
        internal=internal,
    )


async def _drive(runner, event, compression_in_flight=False):
    cold_path = AsyncMock(return_value="COLD_PATH_REPLY")
    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", cold_path),
        patch.object(GatewayRunner, "_run_post_turn_hooks", AsyncMock()),
        patch.object(GatewayRunner, "_clear_durable_active_turn", AsyncMock()),
        patch.object(GatewayRunner, "_persist_active_agents", lambda self: None),
        patch.object(
            GatewayRunner,
            "_session_has_compression_in_flight",
            AsyncMock(return_value=compression_in_flight),
        ),
    ):
        result = await runner._handle_message(event)
    return result, cold_path


@pytest.mark.asyncio
async def test_compression_in_flight_refuses_fresh_turn(tmp_path):
    """Fresh-turn message while a post-turn compression holds the lock: the turn
    must NOT start on the pre-rotation history; the user gets a resend notice."""
    store = _store(tmp_path)
    src = _source()
    store.get_or_create_session(src)

    runner = _make_runner(store)
    result, cold_path = await _drive(runner, _event(src), compression_in_flight=True)

    assert result == t("gateway.busy.compressing_retry")
    assert result and "compress" in result.lower()
    assert cold_path.await_count == 0, (
        "a fresh turn started while compression was in flight"
    )


@pytest.mark.asyncio
async def test_no_compression_starts_fresh_turn(tmp_path):
    """Control: without compression in flight the fresh-turn path is unchanged."""
    store = _store(tmp_path)
    src = _source(chat_id="555011")
    store.get_or_create_session(src)

    runner = _make_runner(store)
    result, cold_path = await _drive(runner, _event(src))

    assert result == "COLD_PATH_REPLY"
    assert cold_path.await_count == 1


@pytest.mark.asyncio
async def test_internal_message_not_gated(tmp_path):
    """The gate mirrors the external-drain gate and only refuses user traffic;
    internal events keep the pre-gate fresh-turn behavior."""
    store = _store(tmp_path)
    src = _source(chat_id="555012")
    store.get_or_create_session(src)

    runner = _make_runner(store)
    result, cold_path = await _drive(runner, _event(src, internal=True))

    assert result == "COLD_PATH_REPLY"
    assert cold_path.await_count == 1
