import asyncio
import sqlite3
import pytest

from pathlib import Path
from types import SimpleNamespace
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
    Event kinds gave_up / crashed / timed_out send a notification but DO
    NOT delete the subscription. The dispatcher may respawn the task and
    fire the same event kind again (e.g. a worker that crashes, gets
    reclaimed, and crashes a second time); the user must hear about the
    second event too. Subscriptions are removed only when the task hits
    a truly final status (done / archived) — see the comment on
    TERMINAL_KINDS in gateway/run.py and PR #21398.
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

    # The user is notified about the abnormal event...
    fake_adapter.send.assert_called_once()
    assert kind.replace('_', ' ') in fake_adapter.send.call_args[0][1]

    # ...but the subscription survives so a respawn-then-same-event cycle
    # reaches the user too. The cursor (last_event_id) advanced inside
    # the same write txn as the claim, so the same event won't re-fire.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert len(subs) == 1, (
        f"Subscription should survive {kind!r} so the next cycle of the "
        f"same event reaches the user; got {subs!r}"
    )
    assert int(subs[0]["last_event_id"]) >= 1, (
        "Cursor should have advanced past the delivered event "
        "(claim_unseen_events_for_sub advances atomically inside the "
        "same write txn as the read)."
    )


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


@pytest.mark.asyncio
async def test_notifier_skips_subscription_owned_by_other_profile(kanban_home):
    """Each gateway keeps its watcher on, but only the subscribing profile claims."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned task", assignee="backend-engineer")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat1",
            notifier_profile="default",
        )
        kb.complete_task(conn, tid, result="done")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "business-partner"

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

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    fake_adapter.send.assert_not_called()
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert int(subs[0]["last_event_id"]) == 0, "wrong profile must not claim the event"


@pytest.mark.asyncio
async def test_notifier_delivers_subscription_owned_by_current_profile(kanban_home):
    """The gateway for the profile that created/subscribed the task reports it."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned task", assignee="backend-engineer")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat1",
            notifier_profile="default",
        )
        kb.complete_task(conn, tid, result="done")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "default"

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
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert subs == []


@pytest.mark.asyncio
async def test_gateway_create_autosubscribes_on_explicit_board(kanban_home):
    """`/kanban --board <slug> create ...` must subscribe on that board.

    The gateway handler currently auto-subscribes after `/kanban create`,
    but the create detection must still work when the shared `--board`
    flag appears before the subcommand, and the subscription must land in
    that board's DB rather than the ambient/default board.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    kb.create_board("projx")

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        thread_id="th1",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice',
        source=source,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)

    assert "subscribed" in out.lower()

    conn = kb.connect(board="projx")
    try:
        subs = kb.list_notify_subs(conn)
        tasks = kb.list_tasks(conn)
    finally:
        conn.close()

    assert [t.title for t in tasks] == ["hello"]
    assert len(subs) == 1
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "th1"

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_notifier_uploads_artifacts_on_completion(kanban_home, tmp_path):
    """When a completed event carries ``artifacts`` in its payload, the
    notifier uploads each file to the subscribed chat as a native
    attachment. Images batch through send_multiple_images; documents
    route through send_document. See the artifacts wiring in
    gateway/run.py._deliver_kanban_artifacts.
    """
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    # Materialize real files so os.path.isfile passes inside the helper.
    chart_path = tmp_path / "q3-revenue.png"
    chart_path.write_bytes(b"PNG-fake-bytes")
    report_path = tmp_path / "report.pdf"
    report_path.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="render q3 chart", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
    finally:
        conn.close()

    # Use the production handler so we exercise the full path: tool args
    # → metadata.artifacts → event payload promotion.
    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    try:
        out = kt._handle_complete({
            "summary": "rendered the chart",
            "artifacts": [str(chart_path), str(report_path)],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)
    import json as _json
    assert _json.loads(out)["ok"] is True

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()
    fake_adapter.name = "telegram"

    sends: list = []
    images_uploaded: list = []
    documents_uploaded: list = []

    async def _send(chat_id, msg, metadata=None):
        sends.append((chat_id, msg))
        runner._running = False

    async def _send_images(chat_id, images, metadata=None, **_kw):
        images_uploaded.extend(p for p, _ in images)

    async def _send_document(chat_id, file_path, metadata=None, **_kw):
        documents_uploaded.append(file_path)

    fake_adapter.send = AsyncMock(side_effect=_send)
    fake_adapter.send_multiple_images = AsyncMock(side_effect=_send_images)
    fake_adapter.send_document = AsyncMock(side_effect=_send_document)
    # extract_local_files is used internally for legacy path fallback;
    # the real BasePlatformAdapter implementation lives there, so wire it.
    from gateway.platforms.base import BasePlatformAdapter
    fake_adapter.extract_local_files = BasePlatformAdapter.extract_local_files

    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # The text completion notification fired.
    assert len(sends) == 1
    # The PNG rode the image-batch path.
    assert any("q3-revenue.png" in p for p in images_uploaded), images_uploaded
    # The PDF rode the document path.
    assert any("report.pdf" in p for p in documents_uploaded), documents_uploaded


@pytest.mark.asyncio
async def test_notifier_artifact_delivery_skips_missing_files(kanban_home, tmp_path):
    """Missing artifact paths are silently skipped — they may have been
    referenced by name only. The notifier must not crash and must still
    deliver any artifacts that do exist."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
    finally:
        conn.close()

    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    try:
        kt._handle_complete({
            "summary": "one real, one ghost",
            "artifacts": [str(real_pdf), "/tmp/definitely-does-not-exist.pdf"],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()
    fake_adapter.name = "telegram"

    documents_uploaded: list = []

    async def _send(chat_id, msg, metadata=None):
        runner._running = False

    async def _send_document(chat_id, file_path, metadata=None, **_kw):
        documents_uploaded.append(file_path)

    fake_adapter.send = AsyncMock(side_effect=_send)
    fake_adapter.send_document = AsyncMock(side_effect=_send_document)
    fake_adapter.send_multiple_images = AsyncMock()
    from gateway.platforms.base import BasePlatformAdapter
    fake_adapter.extract_local_files = BasePlatformAdapter.extract_local_files

    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Only the real file was uploaded.
    assert len(documents_uploaded) == 1
    assert "real.pdf" in documents_uploaded[0]

# ---------------------------------------------------------------------------
# Opt-in active wake (`--trigger-agent`)
# ---------------------------------------------------------------------------


def test_trigger_agent_defaults_false_and_persists(kanban_home):
    """`add_notify_sub` defaults to passive; the opt-in flag round-trips."""
    conn = kb.connect()
    try:
        passive = kb.create_task(conn, title="passive", assignee="w")
        active = kb.create_task(conn, title="active", assignee="w")
        kb.add_notify_sub(conn, task_id=passive, platform="telegram", chat_id="c1")
        kb.add_notify_sub(
            conn, task_id=active, platform="telegram", chat_id="c2",
            trigger_agent=True,
        )
        passive_sub = kb.list_notify_subs(conn, passive)[0]
        active_sub = kb.list_notify_subs(conn, active)[0]
    finally:
        conn.close()

    # Default is passive (stored as 0, falsy).
    assert not passive_sub["trigger_agent"]
    # Opt-in persisted as a truthy integer.
    assert active_sub["trigger_agent"] == 1


def test_trigger_agent_column_migrates_on_legacy_db(tmp_path):
    """A legacy `kanban_notify_subs` without `trigger_agent` migrates safely.

    Old DB files predate the column; `init_db` must add it via the
    additive migration pass and existing rows must default to 0 (passive).
    """
    db_path = tmp_path / "legacy-kanban.db"
    legacy = sqlite3.connect(str(db_path))
    try:
        legacy.execute(
            """
            CREATE TABLE kanban_notify_subs (
                task_id       TEXT NOT NULL,
                platform      TEXT NOT NULL,
                chat_id       TEXT NOT NULL,
                thread_id     TEXT NOT NULL DEFAULT '',
                user_id       TEXT,
                notifier_profile TEXT,
                created_at    INTEGER NOT NULL,
                last_event_id INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (task_id, platform, chat_id, thread_id)
            )
            """
        )
        legacy.execute(
            "INSERT INTO kanban_notify_subs "
            "(task_id, platform, chat_id, created_at) VALUES (?, ?, ?, ?)",
            ("legacy-task", "telegram", "chat1", 0),
        )
        legacy.commit()
    finally:
        legacy.close()

    # init_db re-runs the additive migration pass against the legacy file.
    kb.init_db(db_path=db_path)

    conn = kb.connect(db_path=db_path)
    try:
        cols = {row["name"] for row in conn.execute(
            "PRAGMA table_info(kanban_notify_subs)"
        )}
        assert "trigger_agent" in cols, "migration must add trigger_agent"
        row = conn.execute(
            "SELECT trigger_agent FROM kanban_notify_subs WHERE task_id = ?",
            ("legacy-task",),
        ).fetchone()
        assert row["trigger_agent"] == 0, "legacy rows default to passive"
    finally:
        conn.close()


def test_cli_notify_subscribe_trigger_agent_flag(kanban_home):
    """`/kanban notify-subscribe --trigger-agent` opts the row into active wake."""
    from hermes_cli import kanban as kb_cli

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cli flag", assignee="w")
    finally:
        conn.close()

    # Without the flag → passive.
    out = kb_cli.run_slash(
        f"notify-subscribe {tid} --platform telegram --chat-id passive-chat"
    )
    assert "Subscribed" in out
    assert "active wake" not in out

    # With the flag → active wake.
    out = kb_cli.run_slash(
        f"notify-subscribe {tid} --platform telegram "
        f"--chat-id active-chat --trigger-agent"
    )
    assert "active wake" in out

    conn = kb.connect()
    try:
        subs = {s["chat_id"]: s for s in kb.list_notify_subs(conn, tid)}
    finally:
        conn.close()
    assert not subs["passive-chat"]["trigger_agent"]
    assert subs["active-chat"]["trigger_agent"] == 1

    # notify-list surfaces the flag on the active row only.
    listing = kb_cli.run_slash(f"notify-list {tid}")
    active_line = next(ln for ln in listing.splitlines() if "active-chat" in ln)
    passive_line = next(ln for ln in listing.splitlines() if "passive-chat" in ln)
    assert "trigger-agent" in active_line
    assert "trigger-agent" not in passive_line


# ---------------------------------------------------------------------------
# Origin/return_to auto-subscription
#
# A final/reporter task whose body carries an `Origin/return_to:` directive
# must auto-subscribe the named origin chat so the terminal ACK is routed
# back even when no explicit `notify-subscribe` was issued. The Warroom
# fixture relied on this and silently lost the ACK because `create_task`
# stored the body verbatim without parsing the directive.
# ---------------------------------------------------------------------------

# The exact directive shape from the failing Warroom fixture.
_FIXTURE_BODY = (
    "Final reporter task.\n\n"
    "Origin/return_to: #hermes-main Discord chat_id=1497895797579190357; "
    "source report was #research -> #hermes-main handoff on 2026-05-15."
)
_COLON_FIXTURE_BODY = (
    "Origin/return_to: discord:1497895797579190357 (#hermes-main)\n"
    "Standing lane anchor: #hermes-main owns Hermes/Agent OS/Kanban."
)
_FIXTURE_CHAT_ID = "1497895797579190357"


def test_parse_origin_return_to_extracts_platform_and_chat():
    parsed = kb.parse_origin_return_to(_FIXTURE_BODY)
    assert parsed == {
        "platform": "discord",
        "chat_id": _FIXTURE_CHAT_ID,
        "thread_id": None,
    }


def test_parse_origin_return_to_supports_platform_colon_target():
    parsed = kb.parse_origin_return_to(
        "Origin/return_to: discord:1497895797579190357 (#hermes-main)"
    )
    assert parsed == {
        "platform": "discord",
        "chat_id": _FIXTURE_CHAT_ID,
        "thread_id": None,
    }


def test_parse_origin_return_to_supports_thread_id():
    parsed = kb.parse_origin_return_to(
        "Origin/return_to: telegram chat_id=42 thread_id=7; prose"
    )
    assert parsed == {"platform": "telegram", "chat_id": "42", "thread_id": "7"}


def test_parse_origin_return_to_none_when_unresolvable():
    # No directive at all.
    assert kb.parse_origin_return_to("just a normal body") is None
    assert kb.parse_origin_return_to(None) is None
    # Directive present but no chat id → not resolvable.
    assert kb.parse_origin_return_to("Origin/return_to: #hermes-main Discord") is None
    # Directive with a chat id but no recognisable platform → not resolvable.
    assert kb.parse_origin_return_to("Origin/return_to: chat_id=99") is None


def test_create_task_auto_subscribes_from_origin_return_to(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="final report", assignee="reporter", body=_FIXTURE_BODY,
        )
        subs = kb.list_notify_subs(conn, tid)
        events = list(kb.list_events(conn, tid))
        event_kinds = [e.kind for e in events]
    finally:
        conn.close()

    assert len(subs) == 1, f"expected one auto-subscription, got {subs!r}"
    assert subs[0]["platform"] == "discord"
    assert subs[0]["chat_id"] == _FIXTURE_CHAT_ID
    # Auto-subscriptions stay passive — the directive only asks for return
    # routing, not an active wake.
    assert not subs[0]["trigger_agent"]
    # The auto-subscription is recorded durably so `kanban show` reflects it.
    assert "origin_subscribed" in event_kinds
    # The legacy prose is materialized into a structured OriginReturnContract:
    # the durable event carries a stable, opaque return_id handle.
    origin_ev = next(e for e in events if e.kind == "origin_subscribed")
    assert (origin_ev.payload or {}).get("return_id", "").startswith("ret_")


def test_create_task_auto_subscribes_from_platform_colon_return_to(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="final report", assignee="reporter", body=_COLON_FIXTURE_BODY,
        )
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert len(subs) == 1, f"expected one auto-subscription, got {subs!r}"
    assert subs[0]["platform"] == "discord"
    assert subs[0]["chat_id"] == _FIXTURE_CHAT_ID
    assert not subs[0]["trigger_agent"]


def test_create_task_without_directive_does_not_subscribe(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="plain task", assignee="w", body="no directive")
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert subs == []


def test_completion_without_subscription_records_ack_skipped(kanban_home):
    """A task can finish GO without an origin subscription; that ACK absence
    must be durable and typed instead of being inferred from missing events."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="plain task", assignee="w", body="no directive")
        assert kb.complete_task(conn, tid, result="GO") is True
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()

    kinds = [e.kind for e in events]
    assert "completed" in kinds
    ack_ev = next(e for e in events if e.kind == "ack_skipped")
    assert ack_ev.payload["work"]["verdict"] == "GO"
    assert ack_ev.payload["ack"]["status"] == "SKIPPED_WITH_REASON"
    assert ack_ev.payload["ack"]["reason"] == "no_subscription"
    assert ack_ev.payload["ack_required"] is False


def test_block_without_subscription_records_ack_skipped(kanban_home):
    """A task can finish BLOCK without an origin subscription; that missing
    ACK must also be durable and typed, separate from the work verdict."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="plain blocked task", assignee="w", body="no directive")
        assert kb.block_task(conn, tid, reason="BLOCK: needs input") is True
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()

    kinds = [e.kind for e in events]
    assert "blocked" in kinds
    ack_ev = next(e for e in events if e.kind == "ack_skipped")
    assert ack_ev.payload is not None
    assert ack_ev.payload["work"]["verdict"] == "BLOCK"
    assert ack_ev.payload["ack"]["status"] == "SKIPPED_WITH_REASON"
    assert ack_ev.payload["ack"]["reason"] == "no_subscription"
    assert ack_ev.payload["ack_required"] is False


# --------------------------------------------------------------------------
# Control-plane authority gate wired into real Kanban mutation entrypoints
# --------------------------------------------------------------------------

def test_create_task_blocked_by_stale_compaction_authority(kanban_home, monkeypatch):
    """A real create_task call fails closed before the row is written when
    the session env declares a stale compaction summary."""
    from hermes_cli.control_plane_contracts import StaleAuthorityError

    monkeypatch.setenv("HERMES_CONTROL_PLANE_LANE", "#hermes-main")
    monkeypatch.setenv("HERMES_CONTROL_PLANE_EPOCH", "100")
    monkeypatch.setenv("HERMES_RESUME_COMPACTION_EPOCH", "1")  # stale

    conn = kb.connect()
    try:
        with pytest.raises(StaleAuthorityError):
            kb.create_task(conn, title="should not be created", assignee="w")
        # No task row materialized — the gate fired before any write.
        assert kb.list_tasks(conn) == []
    finally:
        conn.close()


def test_complete_task_blocked_by_cross_lane_mutation(kanban_home, monkeypatch):
    """A real complete_task call into a foreign lane fails closed without an
    explicit approval / pre-approved route."""
    from hermes_cli.control_plane_contracts import StaleAuthorityError

    conn = kb.connect()
    try:
        # Created with no authority env — ordinary path, unaffected.
        tid = kb.create_task(conn, title="cross-lane target", assignee="w")
        # Now the session is on #hermes-main but the mutation targets the
        # #warroom lane: cross-lane without approval must be rejected.
        monkeypatch.setenv("HERMES_CONTROL_PLANE_LANE", "#hermes-main")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_EPOCH", "100")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_TARGET_LANE", "#warroom")
        with pytest.raises(StaleAuthorityError):
            kb.complete_task(conn, tid, result="GO")
        # Verdict unchanged — the task did not transition to done.
        assert kb.get_task(conn, tid).status != "done"
    finally:
        conn.close()


def test_cross_lane_mutation_allowed_with_explicit_approval(kanban_home, monkeypatch):
    """The same cross-lane mutation succeeds once an explicit approval is
    present — the gate blocks silent cross-lane writes, not approved ones."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="approved cross-lane", assignee="w")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_LANE", "#hermes-main")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_EPOCH", "100")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_TARGET_LANE", "#warroom")
        monkeypatch.setenv("HERMES_CONTROL_PLANE_APPROVAL", "1")
        assert kb.complete_task(conn, tid, result="GO") is True
        assert kb.get_task(conn, tid).status == "done"
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_final_completion_with_origin_return_to_delivers_passive_ack(kanban_home):
    """End-to-end: a final task with an Origin/return_to body and no manual
    subscribe still posts a passive ACK to the origin chat on GO."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="final report", assignee="reporter", body=_COLON_FIXTURE_BODY,
        )
        kb.complete_task(conn, tid, result="GO")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()

    async def _send_and_stop(chat_id, msg, metadata=None):
        runner._running = False

    fake_adapter.send = AsyncMock(side_effect=_send_and_stop)
    runner.adapters = {Platform.DISCORD: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1), timeout=10.0,
        )

    fake_adapter.send.assert_called_once()
    assert fake_adapter.send.call_args[0][0] == _FIXTURE_CHAT_ID
    assert "done" in fake_adapter.send.call_args[0][1]

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        comments = [c.body for c in kb.list_comments(conn, tid)]
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()
    assert subs == [], "subscription should be cleaned up after final delivery"
    # notify-list now shows nothing; the durable comment keeps `show` honest
    # about the fact that delivery actually happened.
    assert any("final ACK delivered" in c for c in comments), comments
    # A typed ack_delivered event carries the DeliveryEnvelope projection:
    # the work verdict and the ACK status are recorded as separate facets.
    ack_ev = next(e for e in events if e.kind == "ack_delivered")
    assert ack_ev.payload["work"]["verdict"] == "GO"
    assert ack_ev.payload["ack"]["status"] == "SENT"
    assert ack_ev.payload["ack"]["mode"] == "passive_sent"
    # No active wake was requested on this passive subscription.
    assert "active_wake" not in ack_ev.payload


def test_cli_create_subscribe_then_final_completion(kanban_home):
    """Regression for the documented manual flow: create -> notify-subscribe
    with `--platform discord --chat-id <id>` -> final completion delivers."""
    import re as _re
    from hermes_cli import kanban as kb_cli

    out = kb_cli.run_slash('create "manual final" --assignee reporter')
    m = _re.search(r"Created\s+(t_[0-9a-f]+)", out)
    assert m, f"unexpected create output: {out!r}"
    tid = m.group(1)

    out = kb_cli.run_slash(
        f"notify-subscribe {tid} --platform discord --chat-id {_FIXTURE_CHAT_ID}"
    )
    assert "Subscribed" in out

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1 and subs[0]["platform"] == "discord"
        kb.complete_task(conn, tid, result="GO")
        _, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="discord",
            chat_id=_FIXTURE_CHAT_ID, kinds=["completed"],
        )
    finally:
        conn.close()
    assert [e.kind for e in events] == ["completed"]


@pytest.mark.asyncio
async def test_notifier_send_failure_records_ack_failed_and_retries(kanban_home):
    """Adapter send failures are durable ACK_FAILED events; they do not erase
    the GO work verdict and they keep retry/drop behavior explicit."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="final report", assignee="reporter")
        kb.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="chatX")
        kb.complete_task(conn, tid, result="GO")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()

    async def _fail_and_stop(chat_id, msg, metadata=None):
        runner._running = False
        raise RuntimeError("boom token=abcdefghijklmnopqrstuvwxyz123456")

    fake_adapter.send = AsyncMock(side_effect=_fail_and_stop)
    runner.adapters = {Platform.DISCORD: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1), timeout=10.0,
        )

    fake_adapter.send.assert_called_once()
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()

    assert len(subs) == 1, "transient failure should keep subscription for retry"
    ack_ev = next(e for e in events if e.kind == "ack_failed")
    assert ack_ev.payload["work"]["verdict"] == "GO"
    assert ack_ev.payload["ack"]["status"] == "FAILED"
    assert "<redacted>" in ack_ev.payload["ack"]["error"]
    assert ack_ev.payload["retrying"] is True
    assert ack_ev.payload["attempts"] == 1


@pytest.mark.asyncio
async def test_notifier_helper_records_no_live_adapter_ack_skipped(kanban_home):
    """The reachable disconnected-adapter helper path records typed
    ACK_SKIPPED/no_live_adapter without triggering active wake behavior."""
    from gateway.run import GatewayRunner

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="final report", assignee="reporter")
        kb.add_notify_sub(conn, task_id=tid, platform="discord", chat_id="chatX")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._kanban_record_ack_outcome(
        {"task_id": tid, "platform": "discord", "chat_id": "chatX"},
        None,
        "SKIPPED_WITH_REASON",
        "no_live_adapter",
        None,
        "completed",
        {"retrying": True},
    )

    conn = kb.connect()
    try:
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()
    ack_ev = next(e for e in events if e.kind == "ack_skipped")
    assert ack_ev.payload["work"]["verdict"] == "GO"
    assert ack_ev.payload["ack"]["status"] == "SKIPPED_WITH_REASON"
    assert ack_ev.payload["ack"]["reason"] == "no_live_adapter"
    assert ack_ev.payload["retrying"] is True


@pytest.mark.asyncio
async def test_notifier_records_trigger_agent_unavailable_fallback(kanban_home):
    """When `--trigger-agent` is set but the active wake target is
    unavailable, the passive ACK is still delivered and a durable comment
    records why the wake did not happen."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="active wake task", assignee="reporter")
        kb.add_notify_sub(
            conn, task_id=tid, platform="discord", chat_id="chatX",
            trigger_agent=True,
        )
        kb.complete_task(conn, tid, result="GO")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()

    async def _send_and_stop(chat_id, msg, metadata=None):
        runner._running = False

    fake_adapter.send = AsyncMock(side_effect=_send_and_stop)
    runner.adapters = {Platform.DISCORD: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.run.trigger_agent_message",
               return_value={"trigger_error": "no live adapter"}):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1), timeout=10.0,
        )

    # Passive ACK still delivered despite the unavailable wake target.
    fake_adapter.send.assert_called_once()

    conn = kb.connect()
    try:
        comments = [c.body for c in kb.list_comments(conn, tid)]
        events = list(kb.list_events(conn, tid))
    finally:
        conn.close()
    delivery = [c for c in comments if "final ACK delivered" in c]
    assert delivery, f"expected a durable delivery comment, got {comments!r}"
    assert "unavailable" in delivery[0].lower(), delivery[0]
    # The active-wake failure is recorded as a SEPARATE facet — it must
    # NOT downgrade the work verdict or the passive ACK.
    ack_ev = next(e for e in events if e.kind == "ack_delivered")
    assert ack_ev.payload["work"]["verdict"] == "GO", "work verdict stays GO"
    assert ack_ev.payload["ack"]["status"] == "SENT"
    assert ack_ev.payload["ack"]["mode"] == "passive_sent"
    assert ack_ev.payload["active_wake"]["status"] == "unavailable"
    assert ack_ev.payload["active_wake"]["delivered"] is False
