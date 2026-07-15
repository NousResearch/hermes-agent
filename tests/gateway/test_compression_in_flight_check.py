"""#5 regression: _session_has_compression_in_flight must offload both blocking sources to thread pool."""
import inspect
import threading
from unittest.mock import MagicMock

import pytest


def _make_runner(holder_value=None, record_thread=False, thread_sink=None):
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)

    store = MagicMock()
    store._lock = threading.Lock()
    store._loaded = True
    store._entries = {"k": MagicMock(session_id="sess-123")}
    store._ensure_loaded_locked = lambda: None
    runner.session_store = store

    raw_db = MagicMock()
    if record_thread and thread_sink is not None:
        def _holder(sid):
            thread_sink["thread"] = threading.get_ident()
            return holder_value
        raw_db.get_compression_lock_holder = _holder
    else:
        raw_db.get_compression_lock_holder = MagicMock(return_value=holder_value)

    session_db = MagicMock()
    session_db._db = raw_db
    runner._session_db = session_db
    return runner


def test_method_is_coroutine():
    from gateway.run import GatewayRunner
    assert inspect.iscoroutinefunction(
        GatewayRunner._session_has_compression_in_flight
    ), "#5: method must be async, blocking calls offloaded"


@pytest.mark.asyncio
async def test_returns_true_when_lock_held():
    runner = _make_runner(holder_value="agent-1")
    assert await runner._session_has_compression_in_flight("k") is True


@pytest.mark.asyncio
async def test_returns_false_when_no_lock():
    runner = _make_runner(holder_value=None)
    assert await runner._session_has_compression_in_flight("k") is False


@pytest.mark.asyncio
async def test_returns_false_when_no_session_store():
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = None
    runner._session_db = MagicMock()
    assert await runner._session_has_compression_in_flight("k") is False


@pytest.mark.asyncio
async def test_structural_lock_absence_still_fails_open():
    runner = _make_runner(holder_value=None)
    runner._session_db._db.get_compression_lock_holder = MagicMock(
        side_effect=AttributeError("old SessionDB has no lock helper")
    )

    assert await runner._session_has_compression_in_flight("k") is False


@pytest.mark.asyncio
async def test_db_lock_probe_error_fails_closed():
    runner = _make_runner(holder_value=None)
    runner._session_db._db.get_compression_lock_holder = MagicMock(
        side_effect=RuntimeError("sqlite temporarily unavailable")
    )

    assert await runner._session_has_compression_in_flight("k") is True


@pytest.mark.asyncio
async def test_store_lookup_error_fails_closed():
    runner = _make_runner(holder_value=None)
    runner.session_store._ensure_loaded_locked = MagicMock(
        side_effect=RuntimeError("routing index temporarily unavailable")
    )

    assert await runner._session_has_compression_in_flight("k") is True


@pytest.mark.asyncio
async def test_db_call_runs_off_event_loop():
    """Regression core: get_compression_lock_holder MUST execute in non-event-loop thread."""
    sink = {}
    runner = _make_runner(holder_value="agent-1", record_thread=True, thread_sink=sink)
    loop_thread = threading.get_ident()
    await runner._session_has_compression_in_flight("k")
    assert "thread" in sink, "underlying db.get_compression_lock_holder was not called"
    assert sink["thread"] != loop_thread, (
        "DB call still on event loop thread — #5 NOT fixed (to_thread not applied)"
    )
