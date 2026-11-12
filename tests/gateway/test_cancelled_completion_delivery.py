"""Regression for #146807: a /stop-cancelled delegation is recorded, never woken.

The gateway used to inject every async-delegation completion as a new turn. For a unit the
user had just cancelled that turn made the parent read the interrupt as a failure and
re-dispatch the work. The cancelled unit must land in the transcript so its partial result
stays available, while an ordinary completion still wakes the session.
"""

import asyncio
import contextlib
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.run import GatewayRunner


def _runner(db):
    runner = object.__new__(GatewayRunner)
    runner._session_db = db
    runner.session_authority = None
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._completion_event_scope = lambda evt: contextlib.nullcontext()
    runner._preflight_completion_delivery = AsyncMock(return_value=SimpleNamespace(
        proceed=True, claim_id="", delegation_id="", owner_session_id=None,
        proven_session_id=None, defer=False, early_result=None,
    ))
    runner._inject_watch_notification = AsyncMock(return_value=True)
    return runner


def _event(**overrides):
    evt = {
        "type": "async_delegation",
        "delegation_id": "deleg_146807",
        "session_key": "agent:main:feishu:dm:12345",
        "parent_session_id": "sess-146807",
        "status": "interrupted",
        "goals": ["review the archive"],
        "results": [{"task_index": 0, "status": "interrupted", "summary": "partial finding"}],
        "dispatched_at": 1.0,
        "completed_at": 2.0,
    }
    evt.update(overrides)
    return evt


def _db():
    db = MagicMock()
    db._db = db  # ``_gateway_session_db_inner`` unwraps the async facade
    db.resolve_resume_session_id = lambda session_id: session_id
    return db


def _patch_pins(monkeypatch):
    async def _keep_pin(_runner, evt):
        return evt

    monkeypatch.setattr("gateway.session_detached.follow_detached_pin", _keep_pin)
    monkeypatch.setattr("gateway.session_local_recovery.restore_completion_route",
                        lambda *args, **kwargs: None)


def test_stop_cancelled_completion_is_recorded_without_waking(monkeypatch):
    _patch_pins(monkeypatch)
    db = _db()
    runner = _runner(db)

    result = asyncio.run(runner._deliver_completion_notification(
        "CANCELLED BATCH TEXT", _event(interrupt_reason="stop_command")))

    assert result is True
    db.append_delegation_delivery.assert_called_once()
    session_id, text, metadata = db.append_delegation_delivery.call_args.args
    assert session_id == "sess-146807"
    assert text == "CANCELLED BATCH TEXT"
    assert metadata["delegation_id"] == "deleg_146807"
    runner._inject_watch_notification.assert_not_called()


def test_ordinary_completion_still_wakes_the_session(monkeypatch):
    _patch_pins(monkeypatch)
    db = _db()
    runner = _runner(db)

    result = asyncio.run(runner._deliver_completion_notification(
        "NORMAL BATCH TEXT", _event(status="completed")))

    assert result is True
    runner._inject_watch_notification.assert_awaited_once()
    db.append_delegation_delivery.assert_not_called()


def test_persist_failure_stays_retryable(monkeypatch):
    _patch_pins(monkeypatch)
    db = _db()
    db.append_delegation_delivery = MagicMock(side_effect=RuntimeError("turn lease held"))
    runner = _runner(db)

    result = asyncio.run(runner._deliver_completion_notification(
        "CANCELLED BATCH TEXT", _event(interrupt_reason="stop_command")))

    assert result is False
    runner._inject_watch_notification.assert_not_called()
