"""Invariant: /stop must not drop a parked internal wake (#114456).

Runs the real /stop busy path (``GatewayRunner._busy_stop_command``) with a
parked ``internal=True`` notice (e.g. an async-delegation completion), then the
real post-command drain tail (``_drain_pending_after_session_command``, the same
tail ``_dispatch_active_session_command`` runs after the runner handler), and
asserts the notice reaches ``_start_session_processing``.

Red on unpatched main: ``_interrupt_and_clear_session`` consumes-and-discards
the adapter pending slot, so the drain finds nothing and the idle session
stalls until the next user message.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key


class _ProbeAdapter(BasePlatformAdapter):
    """Real pending-slot + drain machinery, minus the typing side branch.

    ``interrupt_session_activity`` is incidental here (typing indicator); setting
    it non-callable keeps ``_interrupt_and_clear_session`` on the exact seam
    under test (pending-slot handling).
    """

    interrupt_session_activity = None

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):  # pragma: no cover - unused seam
        raise NotImplementedError

    async def send(self, *args, **kwargs):  # pragma: no cover - drain sends nothing
        raise NotImplementedError


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_adapter() -> _ProbeAdapter:
    adapter = object.__new__(_ProbeAdapter)
    adapter._pending_messages = {}
    adapter._active_sessions = {}
    adapter._session_tasks = {}
    return adapter


def _make_runner(adapter: _ProbeAdapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    # Stubbed exactly like test_agent_loop_stopped_hook.py: exercise the
    # pending-slot seam, not generation/lease bookkeeping.
    runner._invalidate_session_run_generation = lambda *a, **kw: 1
    runner._release_running_agent_state = lambda *a, **kw: None
    return runner


@pytest.mark.asyncio
async def test_stop_busy_path_restarts_parked_internal_notice():
    source = _make_source()
    session_key = build_session_key(source)
    adapter = _make_adapter()
    runner = _make_runner(adapter)

    parked = MessageEvent(
        text="[ASYNC DELEGATION BATCH COMPLETE] 1 task done",
        source=source,
        internal=True,
    )
    parked._gateway_accepted = True
    adapter._pending_messages[session_key] = parked

    # Mirror _dispatch_active_session_command: a command guard is installed while
    # the runner handles /stop; the drain below releases it.
    command_guard = asyncio.Event()
    adapter._active_sessions[session_key] = command_guard

    restarted = []
    adapter._start_session_processing = (
        lambda event, key: restarted.append((event, key)) or True
    )

    await runner._busy_stop_command(MessageEvent(text="/stop", source=source), session_key, source)
    await adapter._drain_pending_after_session_command(session_key, command_guard)

    assert [key for _, key in restarted] == [session_key], (
        "parked internal notice never reached _start_session_processing after /stop"
    )
    assert restarted[0][0] is parked
