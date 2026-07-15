"""Tests that on_session_finalize and on_session_reset plugin hooks fire in the gateway."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    new_session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = new_session_entry
    runner.session_store.reset_session.return_value = new_session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_reset_fires_finalize_hook(mock_invoke_hook):
    """/new must fire on_session_finalize with the OLD session id."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    assert any(
        c.args == ("on_session_finalize",)
        and c.kwargs["session_id"] == "sess-old"
        and c.kwargs["platform"] == "telegram"
        and c.kwargs["old_session_id"] == "sess-old"
        and c.kwargs["new_session_id"] == "sess-new"
        for c in mock_invoke_hook.call_args_list
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_reset_fires_reset_hook(mock_invoke_hook):
    """/new must fire on_session_reset with the NEW session id."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    assert any(
        c.args == ("on_session_reset",)
        and c.kwargs["session_id"] == "sess-new"
        and c.kwargs["platform"] == "telegram"
        and c.kwargs["old_session_id"] == "sess-old"
        and c.kwargs["new_session_id"] == "sess-new"
        for c in mock_invoke_hook.call_args_list
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_finalize_before_reset(mock_invoke_hook):
    """on_session_finalize must fire before on_session_reset."""
    runner = _make_runner()

    await runner._handle_reset_command(_make_event("/new"))

    calls = [c for c in mock_invoke_hook.call_args_list
             if c[0][0] in {"on_session_finalize", "on_session_reset"}]
    hook_names = [c[0][0] for c in calls]
    assert hook_names == ["on_session_finalize", "on_session_reset"]


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_shutdown_fires_finalize_for_active_agents(mock_invoke_hook):
    """Gateway stop() must fire on_session_finalize for each active agent."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._background_tasks = set()
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._shutdown_event = MagicMock()
    runner.adapters = {}
    runner._exit_reason = "test"
    runner._exit_code = None
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._running_agents_ts = {}
    runner._update_runtime_status = MagicMock()

    agent1 = MagicMock()
    agent1.session_id = "sess-a"
    agent2 = MagicMock()
    agent2.session_id = "sess-b"
    runner._running_agents = {"key-a": agent1, "key-b": agent2}

    with patch("gateway.status.remove_pid_file"), \
         patch("gateway.status.write_runtime_status"):
        await runner.stop()

    finalize_calls = [
        c for c in mock_invoke_hook.call_args_list
        if c[0][0] == "on_session_finalize"
    ]
    session_ids = {c[1]["session_id"] for c in finalize_calls}
    assert session_ids == {"sess-a", "sess-b"}


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook", side_effect=Exception("boom"))
async def test_hook_error_does_not_break_reset(mock_invoke_hook):
    """Plugin hook errors must not prevent /new from completing."""
    runner = _make_runner()

    result = await runner._handle_reset_command(_make_event("/new"))

    # Should still return a success message despite hook errors
    assert "Session reset" in result or "New session" in result


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_idle_expiry_fires_finalize_hook(mock_invoke_hook):
    """Regression test for #14981.

    When ``_session_expiry_watcher`` sweeps a session that has aged past
    its reset policy (idle timeout, scheduled reset), it must fire
    ``on_session_finalize`` so plugin providers get the same final-pass
    extraction opportunity they'd get from /new or CLI shutdown.  Before
    the fix, the expiry path evicted the agent but silently skipped the
    hook.
    """
    from datetime import datetime, timedelta

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._last_session_store_prune_ts = 0.0

    session_key = "agent:main:telegram:dm:42"
    expired_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-expired",
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now() - timedelta(hours=2),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    expired_entry.expiry_finalized = False

    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {session_key: expired_entry}
    runner.session_store._is_session_expired = MagicMock(return_value=True)
    runner.session_store._lock = MagicMock()
    runner.session_store._lock.__enter__ = MagicMock(return_value=None)
    runner.session_store._lock.__exit__ = MagicMock(return_value=None)
    runner.session_store._save = MagicMock()

    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._sweep_idle_cached_agents = MagicMock(return_value=0)

    # The watcher starts with `await asyncio.sleep(60)` and loops while
    # `self._running`.  Patch sleep so the 60s initial delay is instant, and
    # make the expiry hook invocation flip `_running` false so the loop
    # exits cleanly after one pass.
    _orig_sleep = __import__("asyncio").sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    def _hook_and_stop(*a, **kw):
        runner._running = False
        return None

    mock_invoke_hook.side_effect = _hook_and_stop

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await runner._session_expiry_watcher(interval=0)

    # Look for the finalize call targeting the expired session.
    finalize_calls = [
        c for c in mock_invoke_hook.call_args_list
        if c[0] and c[0][0] == "on_session_finalize"
    ]
    session_ids = {c[1].get("session_id") for c in finalize_calls}
    assert "sess-expired" in session_ids, (
        f"on_session_finalize was not fired during idle expiry; "
        f"got session_ids={session_ids} (regression of #14981)"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_idle_expiry_clears_last_resolved_model(mock_invoke_hook):
    """Regression test for #58403.

    ``_session_expiry_watcher`` permanently finalizes an expired session and
    already drops ``_session_model_overrides`` / the reasoning override /
    ``_pending_model_notes`` — a resumed conversation must not inherit stale
    per-session state. It missed ``_last_resolved_model``: without clearing
    it, a resumed session could serve a cached model from before it went
    idle on a transient config-cache miss, exactly the #58403 class the
    /new and compression-exhausted-reset paths already guard against.
    """
    from datetime import datetime, timedelta

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._last_session_store_prune_ts = 0.0

    session_key = "agent:main:telegram:dm:42"
    expired_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-expired",
        created_at=datetime.now() - timedelta(hours=2),
        updated_at=datetime.now() - timedelta(hours=2),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    expired_entry.expiry_finalized = False

    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {session_key: expired_entry}
    runner.session_store._is_session_expired = MagicMock(return_value=True)
    runner.session_store._lock = MagicMock()
    runner.session_store._lock.__enter__ = MagicMock(return_value=None)
    runner.session_store._lock.__exit__ = MagicMock(return_value=None)
    runner.session_store._save = MagicMock()

    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._sweep_idle_cached_agents = MagicMock(return_value=0)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._last_resolved_model = {
        session_key: "gpt-5",
        "agent:main:telegram:dm:other": "keep-me",
    }

    _orig_sleep = __import__("asyncio").sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    def _hook_and_stop(*a, **kw):
        runner._running = False
        return None

    mock_invoke_hook.side_effect = _hook_and_stop

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await runner._session_expiry_watcher(interval=0)

    assert session_key not in runner._last_resolved_model, (
        "session-expiry finalization did not clear the expired session's "
        "_last_resolved_model entry (#58403)"
    )
    assert runner._last_resolved_model["agent:main:telegram:dm:other"] == "keep-me", (
        "session-expiry finalization must only clear the expired session's "
        "own key, not unrelated sessions' cached entries"
    )


@pytest.mark.asyncio
@patch("hermes_cli.plugins.invoke_hook")
async def test_idle_expiry_writes_ended_at_to_db(mock_invoke_hook):
    """Regression test for #28746.

    When ``_session_expiry_watcher`` finalizes an idle-expired session, the
    ``set_expiry_finalized`` write-path must call
    ``session_store._db.end_session()`` so that ``ended_at`` is written to
    SQLite and ``GET /api/sessions`` stops returning the session as live.

    Before the fix, the watcher set ``expiry_finalized = True`` in memory and
    state.db but never wrote ``ended_at``, causing external API clients to
    keep injecting turns into a stale session that the gateway had already
    discarded.
    """
    from datetime import datetime, timedelta

    from gateway.run import GatewayRunner
    from gateway.session import SessionStore

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._last_session_store_prune_ts = 0.0

    session_key = "agent:main:telegram:group:-1002283267898:1"
    expired_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-stale",
        created_at=datetime.now() - timedelta(hours=25),
        updated_at=datetime.now() - timedelta(hours=25),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    expired_entry.expiry_finalized = False

    mock_db = MagicMock()

    runner.session_store = MagicMock()
    runner.session_store._ensure_loaded = MagicMock()
    runner.session_store._entries = {session_key: expired_entry}
    runner.session_store._is_session_expired = MagicMock(return_value=True)
    runner.session_store._lock = MagicMock()
    runner.session_store._lock.__enter__ = MagicMock(return_value=None)
    runner.session_store._lock.__exit__ = MagicMock(return_value=None)
    runner.session_store._save = MagicMock()
    runner.session_store._db = mock_db
    # The watcher persists finalization through the store's single write-path
    # (#9006); bind the REAL method so the test exercises the actual DB
    # writes instead of a MagicMock absorbing the call.
    runner.session_store.set_expiry_finalized = (
        lambda entry, **kw: SessionStore.set_expiry_finalized(
            runner.session_store, entry, **kw
        )
    )

    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._sweep_idle_cached_agents = MagicMock(return_value=0)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._last_resolved_model = {}

    _orig_sleep = __import__("asyncio").sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    def _hook_and_stop(*a, **kw):
        runner._running = False
        return None

    mock_invoke_hook.side_effect = _hook_and_stop

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await runner._session_expiry_watcher(interval=0)

    # The watcher must have ended the expired session in SQLite with reason
    # "idle" so that ended_at is persisted and the session stops looking live.
    mock_db.end_session.assert_called_once_with("sess-stale", "idle")


def test_idle_expiry_persists_ended_at_and_preserves_reset_contract(tmp_path):
    """End-to-end regression test for #28746 against a real SQLite SessionDB.

    Two contracts, in sequence:

    1. Watcher finalization (``set_expiry_finalized``) must persist
       ``ended_at`` / ``end_reason='idle'`` on the actual sessions row, so
       ``GET /api/sessions`` stops returning the session as live.
    2. The next inbound ``get_or_create_session()`` must still honour the
       idle auto-reset contract: a fresh session_id carrying
       ``was_auto_reset`` / ``auto_reset_reason`` / ``reset_had_activity``
       (the inactivity context note and reset notification depend on these),
       even though the ended row now routes through the #54878 stale-routing
       heal instead of the plain reset path.
    """
    from datetime import datetime, timedelta

    from gateway.config import SessionResetPolicy
    from gateway.session import SessionSource, SessionStore
    from hermes_state import SessionDB

    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=30)
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = db
        store._loaded = True

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="8494508720",
            chat_type="dm",
            user_id="8494508720",
        )
        entry = store.get_or_create_session(source)
        old_id = entry.session_id
        row = db.get_session(old_id)
        assert row is not None and row.get("ended_at") is None

        # Age the session past the idle window with real activity, then run
        # the watcher's finalization write-path.
        entry.updated_at = datetime.now() - timedelta(hours=2)
        entry.last_prompt_tokens = 42
        store.set_expiry_finalized(entry)

        row = db.get_session(old_id)
        assert row["ended_at"] is not None, (
            "watcher finalization did not persist ended_at (regression of #28746)"
        )
        assert row["end_reason"] == "idle"
        assert row["expiry_finalized"] == 1

        # Next inbound message: fresh session, auto-reset metadata intact.
        new_entry = store.get_or_create_session(source)
        assert new_entry.session_id != old_id
        assert new_entry.was_auto_reset is True, (
            "idle auto-reset contract lost: the inactivity context note and "
            "reset notification would be skipped"
        )
        assert new_entry.auto_reset_reason == "idle"
        assert new_entry.reset_had_activity is True

        # Old row keeps its first end_reason (first-reason-wins); the new
        # session is live.
        assert db.get_session(old_id)["end_reason"] == "idle"
        new_row = db.get_session(new_entry.session_id)
        assert new_row is not None and new_row.get("ended_at") is None
    finally:
        db.close()
