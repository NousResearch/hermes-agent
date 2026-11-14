"""A restart waits for real work only. Two units used to hold it forever:

- a saved async-delegation completion for a platform this gateway has no adapter for: the watcher
  requeues it every pass (``_completion_delivery_ready`` is False before any delivery attempt is
  spent), so the queued-notice count never reached zero;
- a managed worker turn whose tool or model call never returns: the inactivity escape covered
  in-process chat turns only.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.restart_when_idle import queued_notice_units
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner


def test_a_completion_with_no_transport_here_does_not_hold_the_restart():
    runner, _adapter = make_restart_runner()  # serves Telegram only
    evt = {"type": "async_delegation", "delegation_id": "d1", "session_key": "agent:main:discord:dm:9"}
    runner._build_process_event_source = lambda _evt: SessionSource(
        platform=Platform.DISCORD, chat_id="9", chat_type="dm", user_id="u1")
    runner._resolve_injection_adapter = lambda _platform, _source: runner.adapters.get(Platform.DISCORD)
    runner._async_notices_in_delivery = [evt]
    assert len(queued_notice_units(runner)) == 1  # not yet tried: a turn may still come of it

    assert asyncio.run(runner._completion_delivery_ready(evt)) is False
    assert queued_notice_units(runner) == []

    # The platform connects: the same notice is deliverable here again, and holds the restart.
    runner.adapters[Platform.DISCORD] = MagicMock(supports_async_delivery=True)
    assert asyncio.run(runner._completion_delivery_ready(evt)) is True
    assert len(queued_notice_units(runner)) == 1


def test_a_managed_turn_silent_past_the_gateway_timeout_does_not_hold_the_restart(monkeypatch):
    from gateway.session_managed_worker import ManagedWorker
    monkeypatch.delenv("HERMES_AGENT_TIMEOUT", raising=False)  # 1800 s
    runner, _adapter = make_restart_runner()
    worker = ManagedWorker(MagicMock(pid=4242))
    worker._arrived = asyncio.Event()
    runner.session_authority = SimpleNamespace(_managed_workers={"s1": worker}, sessions={})
    assert runner._awaitable_work_count() == 1

    worker.last_frame_at = time.monotonic() - 4000  # no frame for longer than agent.gateway_timeout
    assert runner._awaitable_work_count() == 0

    worker._deliver({"type": "delta", "text": "x"})  # a frame arrives: it is live work again
    assert runner._awaitable_work_count() == 1
