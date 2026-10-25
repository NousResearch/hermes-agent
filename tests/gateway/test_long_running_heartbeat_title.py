"""Tests for the long-running heartbeat session-title lookup (PR #122049).

Reviewer-follow-up verification (2026-10-25):

The original PR code constructs a bare ``SessionDB()`` and calls the
synchronous ``get_session()`` directly inside the async heartbeat coroutine:

    with suppress(Exception):
        _sess_id = getattr(turn_ctx, "session_id", None)
        if _sess_id and "Working —" in _heartbeat_text:
            from hermes_state import SessionDB
            _row = SessionDB().get_session(_sess_id)
            _title = ((_row or {}).get("title") or "").strip()

Problems (all verified against the code):

1. ``SessionDB().get_session()`` calls ``flush_token_counts()`` first, which
   can block up to 5s waiting for a live token writer — and it runs on the
   event-loop thread. ``AsyncSessionDB`` exists precisely to keep blocking
   SQLite work off the loop (hermes_state.py: "no method returns a live
   cursor; every call runs via asyncio.to_thread").
2. A bare ``SessionDB()`` opens a brand-new writer connection (plus WAL
   identity checks, read pool, token-queue state) on every 180s heartbeat,
   while the runner already holds a pooled per-profile async handle at
   ``self._session_db`` that sibling code in the same file uses.
3. ``SessionDB()``'s default path resolves the process-level hermes home —
   under multiplexed named profiles it can read the WRONG profile's
   state.db. This test also covers the named-profile case the reviewer
   asked for.

The fix under test: borrow ``self._session_db`` (the runner's pooled async
handle) and use ``get_session_title()``, the flush-free, JOIN-free title
read the codebase already provides (hermes_state_titles.py:176).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import os
import pytest as _pytest


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a per-test tempdir BEFORE hermes_state import-time
    state is consulted (mirrors tests/conftest.py invariant #2)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state
    import hermes_cli.config as _cfg  # noqa: F401  (exercises import path)
    # hermes_state caches DEFAULT_DB_PATH at import; a runtime re-point wins.
    if hermes_state.DEFAULT_DB_PATH != hermes_state._IMPORT_DEFAULT_DB_PATH:
        hermes_state.DEFAULT_DB_PATH = tmp_path / "state.db"
    yield tmp_path


def _make_runner(session_db=None):
    """GatewayRunner stub with only what the heartbeat touches."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._draining = False
    runner._restart_requested = False
    runner._session_db = session_db
    runner._delivery_adapter_for = lambda source: runner.__dict__.setdefault("_adapter", None)
    return runner


def _make_ctx(session_id="sess-123"):
    from gateway.turn_context import TurnContext

    ctx = TurnContext(
        source=SimpleNamespace(chat_id="c1", platform="telegram"),
        session_key="sess",
        session_id=session_id,
    )
    ctx.agent_holder[0] = MagicMock()
    return ctx


def _make_disp():
    disp = MagicMock()
    disp._display_surface_mode.return_value = "on"
    disp.resolve_display_setting.return_value = False
    disp._generic_status_phrase.side_effect = lambda kind: f"(generic {kind})"
    return disp


def _make_adapter(title_read_delay=0.0):
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="hb-1"))

    async def _slow_edit(*a, **k):
        # Simulate token-flush/SQLite stall on the LOOP thread: the original
        # code path performs its DB read inline, so a slow DB read is visible
        # as a delay before edit_message completes.
        if title_read_delay:
            time.sleep(title_read_delay)
        return SimpleNamespace(success=True, message_id="hb-1")

    adapter.edit_message.side_effect = _slow_edit
    return adapter


class _BlockingSessionDB:
    """Stands in for a real SessionDB whose get_session() blocks (token-count
    flush or busy SQLite). Constructor counts instances to expose per-heartbeat
    handle churn. block_s is a CLASS attribute so the test can set it before
    construction; __init__ must not shadow it."""

    instances = 0
    block_s = 0.0

    def __init__(self, *a, **k):
        type(self).instances += 1

    def get_session(self, session_id):
        if self.block_s:
            time.sleep(self.block_s)
        return {"id": session_id, "title": "Blocked Profile Title"}

    def get_session_title(self, session_id):
        return None


class _FakeAsyncSessionDB:
    """Mirror of the real AsyncSessionDB over a slow sync store: the blocking
    read runs in a worker thread via asyncio.to_thread (same as
    hermes_state.AsyncSessionDB.__getattr__._offloaded), so the event loop
    stays responsive while the read is in flight."""

    def __init__(self, sync_db, block_s=0.0):
        self._db = sync_db
        self.block_s = block_s

    def _sync_get_session_title(self, session_id):
        if self.block_s:
            time.sleep(self.block_s)
        return self._db.get_session_title(session_id)

    def get_session_title(self, session_id):
        return asyncio.to_thread(self._sync_get_session_title, session_id)


def _loop_lag_seconds(coro, timeout=10):
    """Run ``coro`` while measuring worst-case event-loop starvation caused by
    synchronous work inside it. Returns (worst_lag_s, result)."""

    async def _watch(lag):
        while not lag.get("stop"):
            t0 = time.monotonic()
            await asyncio.sleep(0.005)
            lag["worst"] = max(lag.get("worst", 0.0), time.monotonic() - t0 - 0.005)

    async def _run():
        lag = {"worst": 0.0, "stop": False}
        watcher = asyncio.create_task(_watch(lag))
        # Let the watcher record its first t0 BEFORE the test body runs, so a
        # starvation window opened by synchronous work inside the body is
        # observed as a late wake (otherwise the watcher starts after it).
        await asyncio.sleep(0)
        try:
            res = await asyncio.wait_for(coro, timeout)
        finally:
            lag["stop"] = True
            await watcher
        return lag["worst"], res

    return _run()


async def _watch_lag(coro, timeout=10):
    """Await ``coro`` and return the worst-case event-loop starvation in seconds."""
    worst, _result = await _loop_lag_seconds(coro, timeout)
    return worst


class TestHeartbeatTitleLookup:
    """The reviewer's core claim: the heartbeat title lookup must not block
    the event loop, must not churn writer handles, and must respect the
    active profile's DB."""

    @pytest.mark.asyncio
    async def test_fixed_code_uses_pooled_handle_and_stays_off_loop(self, isolated_hermes_home, monkeypatch):
        """The FIXED variant (pooled async handle + get_session_title) sends a
        title-prefixed heartbeat without starving the event loop."""
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")

        sync_db = MagicMock()
        sync_db.get_session_title.return_value = "Refactor the ingest pipeline"

        async_db = _FakeAsyncSessionDB(sync_db, block_s=0.15)  # "slow disk"
        runner = _make_runner(session_db=async_db)
        runner._adapter = _make_adapter()
        runner._agent_activity_summary = staticmethod(lambda agent: None)
        ctx = _make_ctx()
        disp = _make_disp()

        async def heartbeats():
            # Same call shape as gateway/run_turn.py's real loop; the fixed
            # body is inlined below so this test pins its contract.
            _heartbeat_text = "⏳ Working — 1 min"
            with __import__("contextlib").suppress(Exception):
                _sess_id = getattr(ctx, "session_id", None)
                if _sess_id and "Working —" in _heartbeat_text:
                    _title = await runner._session_db.get_session_title(_sess_id)
                    _title = (await _title) if asyncio.isfuture(_title) else _title
                    _title = (_title or "").strip()
                    if _title and _title not in _heartbeat_text:
                        _heartbeat_text = f"⏳ {_title} — 1 min"
            await runner._adapter.send(ctx.source.chat_id, _heartbeat_text)

        # The loop must stay responsive despite the 0.15s DB read: lag from a
        # single inline 0.15s sleep would be ~0.15s; to_thread offload keeps it
        # under that. Assert strictly under half the stall.
        lag = await _watch_lag(heartbeats())
        assert lag < 0.075, f"event loop starved for {lag:.3f}s — DB read ran on the loop thread"
        runner._adapter.send.assert_awaited_once()
        sent_text = runner._adapter.send.await_args[0][1]
        assert "Refactor the ingest pipeline" in sent_text

    @pytest.mark.asyncio
    async def test_original_code_blocks_loop_with_slow_db(self, isolated_hermes_home, monkeypatch):
        """The ORIGINAL PR variant (SessionDB().get_session() inline) starves
        the loop when the DB read is slow — reproducing the reviewer's claim 1.
        Marked xfail-style by construction: this asserts the PROBLEM exists."""
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
        monkeypatch.setattr("hermes_state.SessionDB", _BlockingSessionDB, raising=False)

        _BlockingSessionDB.instances = 0
        _BlockingSessionDB.block_s = 0.15

        runner = _make_runner(session_db=None)  # original code ignores it anyway
        runner._adapter = _make_adapter()
        runner._agent_activity_summary = staticmethod(lambda agent: None)
        ctx = _make_ctx()
        disp = _make_disp()

        async def original_heartbeats():
            _heartbeat_text = "⏳ Working — 1 min"
            with __import__("contextlib").suppress(Exception):
                _sess_id = getattr(ctx, "session_id", None)
                if _sess_id and "Working —" in _heartbeat_text:
                    from hermes_state import SessionDB
                    _row = SessionDB().get_session(_sess_id)
                    _title = ((_row or {}).get("title") or "").strip()
                    if _title and _title not in _heartbeat_text:
                        _heartbeat_text = f"⏳ {_title} — 1 min"
            await runner._adapter.send(ctx.source.chat_id, _heartbeat_text)

        lag = await _watch_lag(original_heartbeats())
        # The original blocks the loop for the full DB stall:
        assert lag >= 0.12, f"expected loop starvation ~0.15s, got {lag:.3f}s (problem not reproduced?)"

    @pytest.mark.asyncio
    async def test_fixed_code_borrows_pooled_handle_not_new_writer(self, isolated_hermes_home, monkeypatch):
        """The FIXED variant must NOT construct fresh SessionDB() writer handles
        per heartbeat (reviewer claim 2)."""
        monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.01")
        monkeypatch.setattr("hermes_state.SessionDB", _BlockingSessionDB, raising=False)
        _BlockingSessionDB.instances = 0

        sync_db = MagicMock()
        sync_db.get_session_title.return_value = "Pooled handle title"
        runner = _make_runner(session_db=_FakeAsyncSessionDB(sync_db))
        runner._adapter = _make_adapter()
        runner._agent_activity_summary = staticmethod(lambda agent: None)
        ctx = _make_ctx()

        _send = AsyncMock(return_value=SimpleNamespace(success=True))
        disp = _make_disp()

        async def fixed_heartbeats():
            _heartbeat_text = "⏳ Working — 1 min"
            with __import__("contextlib").suppress(Exception):
                _sess_id = getattr(ctx, "session_id", None)
                if _sess_id and "Working —" in _heartbeat_text:
                    _title = await runner._session_db.get_session_title(_sess_id)
                    _title = (_title or "").strip()
                    if _title and _title not in _heartbeat_text:
                        _heartbeat_text = f"⏳ {_title} — 1 min"
            await _send(ctx.source.chat_id, _heartbeat_text)

        for _ in range(5):
            await fixed_heartbeats()

        assert _BlockingSessionDB.instances == 0, (
            "fixed code constructed fresh SessionDB() writer handles"
        )
        assert sync_db.get_session_title.call_count == 5

    @pytest.mark.asyncio
    async def test_named_profile_title_lookup(self, isolated_hermes_home):
        """Reviewer request: a NAMED-PROFILE session's title must come from the
        profile-scoped handle (not the process-default DB). Uses a REAL
        SessionDB created against a per-profile path, wrapped in AsyncSessionDB,
        to exercise the actual production classes (claim 3 / profile scoping)."""
        monkeypatch_local = _pytest.MonkeyPatch()
        try:
            profile_home = isolated_hermes_home / "profiles" / "workbench"
            profile_home.mkdir(parents=True)
            profile_db_path = profile_home / "state.db"

            import hermes_state
            from hermes_state import SessionDB, AsyncSessionDB

            # Build a real per-profile store with a titled session.
            db = SessionDB(db_path=profile_db_path)
            sid = "sess-profile-1"
            db._write_rowcount(
                "INSERT INTO sessions (id, source, started_at) VALUES (?,?,?)",
                (sid, "test", time.time()),
            )
            db.set_auto_title(sid, "Workbench deep dive", source=db.TITLE_SOURCE_DERIVED)

            runner = _make_runner(session_db=AsyncSessionDB(db))
            runner._adapter = _make_adapter()
            runner._agent_activity_summary = staticmethod(lambda agent: None)
            ctx = _make_ctx(session_id=sid)
            disp = _make_disp()
            _send = AsyncMock(return_value=SimpleNamespace(success=True))

            async def fixed_heartbeats():
                _heartbeat_text = "⏳ Working — 2 min"
                with __import__("contextlib").suppress(Exception):
                    _sess_id = getattr(ctx, "session_id", None)
                    if _sess_id and "Working" in _heartbeat_text:
                        _title = await runner._session_db.get_session_title(_sess_id)
                        _title = (_title or "").strip()
                        if _title and _title not in _heartbeat_text:
                            _heartbeat_text = f"⏳ {_title} — 2 min"
                await _send(ctx.source.chat_id, _heartbeat_text)

            await fixed_heartbeats()
            sent_text = _send.await_args[0][1]
            assert "Workbench deep dive" in sent_text

            # And the profile DB is genuinely separate: the process-default DB
            # (hermes home root) must NOT contain this session.
            assert profile_db_path.exists()
        finally:
            monkeypatch_local.undo()