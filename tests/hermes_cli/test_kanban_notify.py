import asyncio
import pytest

from pathlib import Path
from hermes_cli import kanban_db as kb
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.asyncio
async def test_notifier_unsubs_after_completed_event(kanban_home):
    """
    Subscription should be remove after completed event
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="test task", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        kb.complete_task(conn, tid, result="completed by agent")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()

    async def _send_and_stop(chat_id, msg, metadata=None):
        runner._running = False

    fake_adapter.send = AsyncMock(side_effect=_send_and_stop)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    fake_adapter.send.assert_called_once()
    call_msg = fake_adapter.send.call_args[0][1]
    assert "completed" in call_msg

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert subs == [], "Subscription should be unsub after completed event"


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ["gave_up", "crashed", "timed_out"])
async def test_notifier_unsubs_after_abnormal_events(kind, kanban_home):
    """
    Event kind of gave_up, crashed, time_out would be cover, and remove subscription
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()

    try:
        tid = kb.create_task(conn, title=f"test {kind} task", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        kb._append_event(conn, tid, kind=kind)
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()

    async def _send_and_stop(chat_id, msg, metadata=None):
        runner._running = False

    fake_adapter.send = AsyncMock(side_effect=_send_and_stop)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    fake_adapter.send.assert_called_once()
    assert kind.replace('_', ' ') in fake_adapter.send.call_args[0][1]

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert subs == [], "Subscription should be unsub after abnormal crash"


@pytest.mark.asyncio
async def test_notifier_second_blocked_delivers(kanban_home):
    """
    After the first blocked, should receive second blocked notification.
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    delivered_msgs: list[str] = []

    async def _capture_send(chat_id, msg, metadata=None):
        delivered_msgs.append(msg)

    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock(side_effect=_capture_send)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 6:
            runner._running = False

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="test task", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")

        # Cycle 1: blocked
        kb.block_task(conn, tid, reason="first block")
    finally:
        conn.close()

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Cycle 2: unblock → block run again
    runner._running = True
    tick_count = 0

    conn = kb.connect()
    try:
        kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, reason="second block")
    finally:
        conn.close()

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    blocked_deliveries = [m for m in delivered_msgs if "blocked" in m]
    assert "second block" not in blocked_deliveries[0]
    assert "second block" in blocked_deliveries[1]
    assert len(blocked_deliveries) == 2, (
        f"Should receive 2 blocked notification, but only get {len(blocked_deliveries)} count\n"
        f"Message {delivered_msgs}"
    )


# ---------------------------------------------------------------------------
# Regression: gateway watchers must not double-init the kanban DB.
#
# Both the notifier watcher (`_kanban_notifier_watcher`) and the dispatcher
# tick (`_tick_once_for_board`) used to call `_kb.connect(board=slug)`
# immediately followed by `_kb.init_db(board=slug)`. Since `connect()`
# already runs the schema + idempotent migration on first open per process,
# the explicit `init_db()` was redundant — and worse, `init_db()`
# deliberately busts the per-process cache and re-runs the migration on a
# *second* connection, which races the first.  On legacy DBs this surfaced
# as `duplicate column name: <col>` (now tolerated by
# `_add_column_if_missing`) and intermittent `database is locked` errors
# (issue #21378).
#
# The fix removes the `init_db()` calls in both watchers; this regression
# test pins that behaviour so we don't reintroduce them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notifier_does_not_call_init_db(kanban_home):
    """Notifier watcher path must not invoke `_kb.init_db` (issue #21378)."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    init_db_calls: list[object] = []
    real_init_db = kb.init_db

    def _spy_init_db(*args, **kwargs):
        init_db_calls.append((args, kwargs))
        return real_init_db(*args, **kwargs)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("hermes_cli.kanban_db.init_db", side_effect=_spy_init_db):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    assert init_db_calls == [], (
        "_kanban_notifier_watcher must not call init_db on every tick — "
        "connect() handles first-run schema init. "
        "Reintroducing init_db revives issue #21378. "
        f"Got {len(init_db_calls)} call(s): {init_db_calls}"
    )


def test_dispatcher_tick_does_not_call_init_db(kanban_home, monkeypatch):
    """`_tick_once_for_board` must not invoke `_kb.init_db` (issue #21378).

    `connect()` already runs the schema + idempotent migration on first open
    per process. The explicit `init_db()` call was redundant and triggered a
    second migration on a second connection that raced the first.
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from unittest.mock import patch

    runner = object.__new__(GatewayRunner)

    init_db_calls: list[object] = []
    real_init_db = kb.init_db

    def _spy_init_db(*args, **kwargs):
        init_db_calls.append((args, kwargs))
        return real_init_db(*args, **kwargs)

    # The dispatcher watcher's tick lives as a local closure inside
    # `_kanban_dispatcher_watcher`. Read the source and assert the
    # specific patterns that would reintroduce the bug are absent.
    import inspect
    src = inspect.getsource(GatewayRunner._kanban_dispatcher_watcher)
    assert "_kb.init_db(board=slug)" not in src, (
        "_kanban_dispatcher_watcher must not call _kb.init_db(board=slug) — "
        "see issue #21378. Use connect() alone; it runs migrations on first "
        "open per process."
    )

    notifier_src = inspect.getsource(GatewayRunner._kanban_notifier_watcher)
    assert "_kb.init_db(board=slug)" not in notifier_src, (
        "_kanban_notifier_watcher must not call _kb.init_db(board=slug) — "
        "see issue #21378."
    )


# ---------------------------------------------------------------------------
# Global block-notifier (kanban.notify_on_block)
# ---------------------------------------------------------------------------
#
# Companion to the per-task notifier above. These tests exercise
# `_kanban_global_block_watcher`: the gateway watcher that fans out
# `blocked` / `gave_up` events to a configured channel (or all home
# channels) without requiring per-task subscriptions.

def _make_runner_with_adapter(platform):
    """Construct a GatewayRunner stub with a single mocked adapter."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner._running = True
    fake_adapter = MagicMock()
    delivered: list[tuple[str, str, dict]] = []

    async def _capture(chat_id, msg, metadata=None):
        delivered.append((chat_id, msg, metadata or {}))

    fake_adapter.send = AsyncMock(side_effect=_capture)
    runner.adapters = {platform: fake_adapter}
    return runner, fake_adapter, delivered


def _stop_after(runner, ticks=3):
    """Return a fast-sleep coroutine that stops the watcher after N ticks."""
    counter = {"n": 0}
    _orig = asyncio.sleep

    async def _fast(_):
        counter["n"] += 1
        if counter["n"] >= ticks:
            runner._running = False
        await _orig(0)
    return _fast


@pytest.mark.asyncio
async def test_global_block_watcher_disabled_by_default(kanban_home):
    """When kanban.notify_on_block is unset/false, the watcher exits quietly."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner, fake_adapter, _delivered = _make_runner_with_adapter(Platform.TELEGRAM)

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="ignored", assignee="w")
        kb.block_task(conn, tid, reason="x")
    finally:
        conn.close()

    # `load_config` returning an empty kanban section keeps the flag false.
    with patch(
        "hermes_cli.config.load_config",
        return_value={"kanban": {}},
    ):
        await asyncio.wait_for(
            runner._kanban_global_block_watcher(interval=1),
            timeout=5.0,
        )
    fake_adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_global_block_watcher_seeds_cursor_then_delivers_new_block(kanban_home):
    """First enable seeds cursor at MAX(id); only NEW block events fire."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform, HomeChannel

    runner, fake_adapter, delivered = _make_runner_with_adapter(Platform.TELEGRAM)

    # Provide a home channel for the runner's config.
    runner.config = MagicMock()
    runner.config.get_home_channel = MagicMock(
        return_value=HomeChannel(
            platform=Platform.TELEGRAM, chat_id="home-chat", name="home",
        )
    )

    # Pre-existing block (must NOT fire — it predates the cursor seed).
    conn = kb.connect()
    try:
        old_tid = kb.create_task(conn, title="old", assignee="w")
        kb.block_task(conn, old_tid, reason="historical")
    finally:
        conn.close()

    cfg = {"kanban": {"notify_on_block": True, "notify_on_block_channel": ""}}

    # Tick #1 should seed the cursor and find no NEW events. Then we
    # inject a fresh block and a follow-up tick delivers it. The
    # fast-sleep helper stops the watcher after a few ticks.
    new_tid_holder: dict[str, str] = {}

    real_sleep = asyncio.sleep
    tick = {"n": 0}

    async def _fast(_):
        tick["n"] += 1
        # tick 1 = initial 5s delay; tick 2 = first inter-iteration sleep
        # AFTER the cursor-seeding pass. Insert at tick 2 so the seed
        # pins the cursor at the historical event and the fresh block
        # is seen as new on the next iteration.
        if tick["n"] == 2:
            conn2 = kb.connect()
            try:
                tid = kb.create_task(conn2, title="fresh", assignee="w")
                kb.block_task(conn2, tid, reason="halt")
                new_tid_holder["tid"] = tid
            finally:
                conn2.close()
        if tick["n"] >= 5:
            runner._running = False
        await real_sleep(0)

    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("gateway.run.asyncio.sleep", side_effect=_fast):
        await asyncio.wait_for(
            runner._kanban_global_block_watcher(interval=1),
            timeout=10.0,
        )

    assert "tid" in new_tid_holder
    # Exactly one delivery, and it references the fresh block (not the
    # historical one that predated the cursor seed).
    assert fake_adapter.send.called
    msgs = [m for (_chat, m, _meta) in delivered]
    assert any(new_tid_holder["tid"] in m for m in msgs), (
        f"expected delivery to mention fresh task {new_tid_holder['tid']}, got {msgs}"
    )
    assert not any("historical" in m for m in msgs), (
        f"historical block must NOT be delivered, got {msgs}"
    )

    # Cursor was persisted past the new event id.
    conn = kb.connect()
    try:
        cursor = kb.get_block_notify_cursor(conn)
    finally:
        conn.close()
    assert cursor is not None and cursor > 0


@pytest.mark.asyncio
async def test_global_block_watcher_delivers_gave_up_with_failure_count(kanban_home):
    """Auto-block via failure_limit emits `gave_up` and the watcher delivers it."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform, HomeChannel

    runner, fake_adapter, delivered = _make_runner_with_adapter(Platform.TELEGRAM)
    runner.config = MagicMock()
    runner.config.get_home_channel = MagicMock(
        return_value=HomeChannel(
            platform=Platform.TELEGRAM, chat_id="ops", name="ops",
        )
    )

    cfg = {"kanban": {"notify_on_block": True, "notify_on_block_channel": ""}}

    # Seed the cursor first so the historical task creation events
    # don't get re-attributed as block events. (They aren't `blocked`
    # or `gave_up`, so they wouldn't fire anyway, but we want the
    # cursor seeded before we induce the auto-block to mirror real life.)
    real_sleep = asyncio.sleep
    tick = {"n": 0}
    induced = {"done": False}

    async def _fast(_):
        tick["n"] += 1
        # Insert at tick 2 (after cursor seeding) so the gave_up event
        # registers as new. See the seed-cursor test above. We emit
        # `gave_up` directly with a payload that matches what
        # `_record_task_failure` writes — simpler than driving a real
        # auto-block through the dispatcher (which is already covered
        # in test_kanban_core_functionality.py).
        if tick["n"] == 2 and not induced["done"]:
            conn2 = kb.connect()
            try:
                tid = kb.create_task(conn2, title="will-fail", assignee="w")
                kb._append_event(
                    conn2, tid, "gave_up",
                    {
                        "failures": 2,
                        "effective_limit": 2,
                        "limit_source": "dispatcher",
                        "error": "boom",
                        "trigger_outcome": "spawn_failed",
                    },
                )
                induced["done"] = True
                induced["tid"] = tid
            finally:
                conn2.close()
        if tick["n"] >= 5:
            runner._running = False
        await real_sleep(0)

    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("gateway.run.asyncio.sleep", side_effect=_fast):
        await asyncio.wait_for(
            runner._kanban_global_block_watcher(interval=1),
            timeout=10.0,
        )

    msgs = [m for (_chat, m, _meta) in delivered]
    assert any(induced["tid"] in m and "blocked" in m for m in msgs), (
        f"expected gave_up delivery for {induced.get('tid')!r}, got {msgs}"
    )


@pytest.mark.asyncio
async def test_global_block_watcher_explicit_channel_override(kanban_home):
    """Explicit notify_on_block_channel overrides the home-channel fan-out."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform, HomeChannel

    runner, fake_adapter, delivered = _make_runner_with_adapter(Platform.TELEGRAM)
    # Set a DIFFERENT home channel so we can verify the explicit override
    # routes elsewhere (specifically: chat_id="ops-chat", thread "42").
    runner.config = MagicMock()
    runner.config.get_home_channel = MagicMock(
        return_value=HomeChannel(
            platform=Platform.TELEGRAM, chat_id="home-default", name="home",
        )
    )

    cfg = {"kanban": {
        "notify_on_block": True,
        "notify_on_block_channel": "telegram:ops-chat:42",
    }}

    real_sleep = asyncio.sleep
    tick = {"n": 0}
    induced = {"tid": None}

    async def _fast(_):
        tick["n"] += 1
        if tick["n"] == 2:
            conn2 = kb.connect()
            try:
                tid = kb.create_task(conn2, title="halt-me", assignee="w")
                kb.block_task(conn2, tid, reason="manual halt")
                induced["tid"] = tid
            finally:
                conn2.close()
        if tick["n"] >= 5:
            runner._running = False
        await real_sleep(0)

    with patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("gateway.run.asyncio.sleep", side_effect=_fast):
        await asyncio.wait_for(
            runner._kanban_global_block_watcher(interval=1),
            timeout=10.0,
        )

    # Routed to the explicit chat, not the home default; thread metadata set.
    assert fake_adapter.send.called
    chat_ids = {chat for (chat, _msg, _meta) in delivered}
    assert "ops-chat" in chat_ids
    assert "home-default" not in chat_ids
    metas = [meta for (_chat, _msg, meta) in delivered]
    assert any(meta.get("thread_id") == "42" for meta in metas)


@pytest.mark.asyncio
async def test_global_block_watcher_rejects_malformed_channel(kanban_home):
    """Misconfigured channel disables the watcher loudly (no sends, no crash)."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner, fake_adapter, _delivered = _make_runner_with_adapter(Platform.TELEGRAM)

    cfg = {"kanban": {
        "notify_on_block": True,
        "notify_on_block_channel": "this-is-not-valid",
    }}
    with patch("hermes_cli.config.load_config", return_value=cfg):
        await asyncio.wait_for(
            runner._kanban_global_block_watcher(interval=1),
            timeout=5.0,
        )
    fake_adapter.send.assert_not_called()
