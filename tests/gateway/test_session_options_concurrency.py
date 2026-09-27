"""Turn admission is serialised with session runtime-option commits (#92185, PR #92187).

F2: the busy check before a runtime-options write was check-then-act. A turn could be admitted
between "idle" and the write, so the write landed under a running turn. Turn admission, the
commit and the boot-resume claim now share one per-session admission lock.
"""

import asyncio
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore

HIGH = {"enabled": True, "effort": "high"}


def _no_db(**_kw):
    raise RuntimeError("SQLite disabled in test")


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1")


def _event(text="hello") -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=_source())


@pytest.fixture
def store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _no_db)
    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


def _runner(store):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {}
    runner.session_store = store
    runner._session_db = None
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._draining = False
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.delivery_router = MagicMock()
    return runner


class _ParkedWrites:
    """Park ``set_runtime_options`` on its worker thread until released."""

    def __init__(self, store):
        self.started = threading.Event()
        self.release = threading.Event()
        real = store.set_runtime_options

        def _parked(*args, **kwargs):
            self.started.set()
            assert self.release.wait(10), "parked write was never released"
            return real(*args, **kwargs)

        store.set_runtime_options = _parked

    async def wait_started(self):
        assert await asyncio.to_thread(self.started.wait, 5), "no durable write was submitted"


def _spy_on_admission(runner, session_key):
    """Count claimers that start waiting for the admission lock (the real lock is used)."""
    reached = []
    real_lookup = runner._session_admission_lock

    class _Spy:
        def __init__(self, lock):
            self._lock = lock

        def locked(self):
            return self._lock.locked()

        async def __aenter__(self):
            reached.append(True)
            await self._lock.acquire()

        async def __aexit__(self, *exc):
            self._lock.release()

    runner._session_admission_lock = lambda key: _Spy(real_lookup(key)) if key == session_key else real_lookup(key)
    return reached


@pytest.mark.asyncio
async def test_inbound_turn_parks_behind_in_flight_commit_then_runs_once(store):
    """Two messages arrive while a commit is writing. Both park at the claim; the first is
    admitted once the commit settled, the second sees that turn and takes the running-session
    path instead of claiming over it."""
    source = _source()
    session_key = store.get_or_create_session(source).session_key
    runner = _runner(store)
    parked = _ParkedWrites(store)
    seen = []
    first_turn_done = asyncio.Event()
    busy_path = AsyncMock(return_value="queued")
    runner._hm_handle_running_session_message = busy_path

    async def _turn(_self, _event, _source, quick_key, _generation):
        live = runner._session_state(quick_key).conversation.reasoning_override
        seen.append((live, store.lookup_by_session_key(quick_key).reasoning_override))
        await first_turn_done.wait()
        return "ok"

    with patch.object(GatewayRunner, "_handle_message_with_agent", _turn):
        commit = asyncio.create_task(
            runner._commit_session_runtime_options(source, {"reasoning_override": HIGH}))
        turns = []
        try:
            await parked.wait_started()
            reached = _spy_on_admission(runner, session_key)
            turns = [asyncio.create_task(runner._handle_message(_event(text))) for text in ("one", "two")]
            for _ in range(200):
                if len(reached) == 2 or seen or any(t.done() for t in turns):
                    break
                await asyncio.sleep(0.01)

            # Both turns are parked at the claim: not admitted, nothing ran.
            assert len(reached) == 2 and not any(t.done() for t in turns), f"admitted mid-write: {seen}"
            assert not runner._is_session_running(session_key)
        finally:
            parked.release.set()
        assert await commit is True
        assert await asyncio.wait_for(turns[1], 5) == "queued"
        first_turn_done.set()
        assert await asyncio.wait_for(turns[0], 5) == "ok"

    # One turn, admitted after the write and the live assignment both landed; the second message
    # went to the running-session path instead of clobbering the first claim.
    assert seen == [(HIGH, HIGH)]
    busy_path.assert_awaited_once()


@pytest.mark.asyncio
async def test_boot_resume_defers_under_commit_then_resumes(store):
    source = _source()
    entry = store.get_or_create_session(source)
    session_key = entry.session_key
    entry.resume_pending = True
    entry.resume_reason = "restart_interrupted"
    entry.last_resume_marked_at = datetime.now()
    runner = _runner(store)
    runner._delivery_adapter_for = lambda _source: MagicMock()
    runner._resume_owner_authorized = lambda _key, _source: True
    runner._persist_active_agents = lambda: None
    resumed = []

    async def _resume(_adapter, _event, key):
        resumed.append((key, runner._session_state(key).conversation.reasoning_override))

    runner._run_startup_resume_event = _resume
    parked = _ParkedWrites(store)
    commit = asyncio.create_task(runner._commit_session_runtime_options(source, {"reasoning_override": HIGH}))
    try:
        await parked.wait_started()
        # The synchronous boot-resume claim cannot await the lock: it defers instead of claiming.
        assert runner._schedule_resume_pending_sessions() == 0
        assert not runner._is_session_running(session_key)
    finally:
        parked.release.set()
    assert await commit is True
    for _ in range(5):  # the reschedule is queued behind the lock release, then spawns the task
        await asyncio.sleep(0)

    assert resumed == [(session_key, HIGH)]
    assert runner._is_session_running(session_key)
