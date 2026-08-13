import asyncio
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
    # Allow the kanban notifier path-validator to upload artifacts the
    # tests write under ``tmp_path``. Without this, every artifact-delivery
    # test silently drops files because ``tmp_path`` isn't inside the
    # default ``MEDIA_DELIVERY_SAFE_ROOTS`` cache dirs.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    return home


def _assert_inherited_notify_sub(subs: list[dict]) -> None:
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "topic1"
    assert subs[0]["user_id"] == "user1"
    assert subs[0]["notifier_profile"] == "default"


def test_notify_sub_delivery_mode_persists_and_last_write_wins(kanban_home):
    """delivery_mode persists; an explicit re-subscribe is last-write-wins, a
    ``None`` re-subscribe leaves the existing mode untouched, an unknown value
    is ignored, and none of this clobbers the notifier_profile owner."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="mode sub task", assignee="worker1")
        # Fresh sub without a mode -> defaults to "notify".
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="owner-a",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["delivery_mode"] == "notify"
        assert subs[0]["notifier_profile"] == "owner-a"

        # Explicit re-subscribe changes the mode (last-write-wins) and must NOT
        # overwrite the existing owner (owner self-heals only when unset).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="owner-b", delivery_mode="wake",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["delivery_mode"] == "wake"
        assert subs[0]["notifier_profile"] == "owner-a"

        # A None re-subscribe leaves the existing mode untouched.
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["delivery_mode"] == "wake"

        # An unknown mode is ignored (treated like None: no clobber).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="bogus",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["delivery_mode"] == "wake"
    finally:
        conn.close()


def test_child_task_inherits_parent_delivery_mode(kanban_home):
    """Graph children inherit the parent's ACK edge AND its delivery_mode."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="root", assignee=None)
        kb.add_notify_sub(
            conn, task_id=parent, platform="telegram", chat_id="chat1",
            thread_id="42", user_id="u1", user_id_alt="alt-u1", notifier_profile="default",
            delivery_mode="notify+wake",
        )
        child = kb.create_task(
            conn, title="review child", assignee="ccreviewer", parents=[parent],
        )
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "42"
    assert subs[0]["user_id"] == "u1"
    assert subs[0]["user_id_alt"] == "alt-u1"
    assert subs[0]["notifier_profile"] == "default"
    assert subs[0]["delivery_mode"] == "notify+wake"


def test_notify_sub_chat_type_persists_and_last_write_wins(kanban_home):
    """chat_type persists, defaults to 'dm', an explicit re-subscribe is
    last-write-wins, and a None re-subscribe leaves it untouched. The
    active-wake path replays this field so the woken turn keys to the
    operator's real channel."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="chat_type sub", assignee="worker1")
        # Fresh sub without chat_type -> defaults to "dm".
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "dm"

        # Explicit re-subscribe corrects the recorded chat_type (last-write-wins).
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            chat_type="group",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "group"

        # A None re-subscribe (here changing only the mode) must NOT clobber
        # the recorded chat_type.
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="wake",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["chat_type"] == "group"
        assert subs[0]["delivery_mode"] == "wake"
    finally:
        conn.close()


def test_notify_sub_user_id_alt_persists_and_backfills_legacy_rows(kanban_home):
    """user_id_alt is persisted with the notify subscription routing tuple and
    can backfill a pre-existing row created before the alt id was known."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="alt sub", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            user_id="open-id",
        )
        subs = kb.list_notify_subs(conn, tid)
        assert subs[0]["user_id"] == "open-id"
        assert subs[0]["user_id_alt"] is None

        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            user_id="open-id", user_id_alt="union-id",
        )
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()

    assert subs[0]["user_id"] == "open-id"
    assert subs[0]["user_id_alt"] == "union-id"


def test_child_task_inherits_parent_chat_type(kanban_home):
    """Graph children inherit the parent's chat_type alongside its ACK edge and
    delivery_mode, so a woken child notification keys to the same session as
    the parent's originating channel."""
    import hermes_cli.kanban_db as kb

    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="root", assignee=None)
        kb.add_notify_sub(
            conn, task_id=parent, platform="telegram", chat_id="chat1",
            user_id="u1", user_id_alt="alt-u1", chat_type="group",
            delivery_mode="notify+wake",
        )
        child = kb.create_task(
            conn, title="impl child", assignee="coder", parents=[parent],
        )
        subs = kb.list_notify_subs(conn, child)
    finally:
        conn.close()

    assert len(subs) == 1
    assert subs[0]["chat_type"] == "group"
    assert subs[0]["delivery_mode"] == "notify+wake"
    assert subs[0]["user_id"] == "u1"
    assert subs[0]["user_id_alt"] == "alt-u1"


@pytest.mark.asyncio
async def test_notifier_notify_plus_wake_sends_and_wakes(kanban_home):
    """notify+wake delivers the passive message AND wakes the agent; a plain
    notify sub only sends. The agent is woken only for the notify+wake sub."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        passive_tid = kb.create_task(conn, title="passive task", assignee="worker1")
        active_tid = kb.create_task(conn, title="active task", assignee="worker1")
        kb.add_notify_sub(conn, task_id=passive_tid, platform="telegram", chat_id="chat1")
        kb.add_notify_sub(
            conn, task_id=active_tid, platform="telegram", chat_id="chat1",
            delivery_mode="notify+wake",
        )
        kb.block_task(conn, passive_tid, reason="passive block")
        kb.block_task(conn, active_tid, reason="active block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    fake_adapter = MagicMock()
    sent_msgs: list[str] = []

    async def _send(chat_id, msg, metadata=None):
        sent_msgs.append(msg)

    fake_adapter.send = AsyncMock(side_effect=_send)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    wake_mock = AsyncMock()
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.wake.deliver_wake", new=wake_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Both subs still get a passive send (notify AND notify+wake send).
    assert len(sent_msgs) == 2
    assert any("passive block" in m for m in sent_msgs)
    assert any("active block" in m for m in sent_msgs)
    # Only the notify+wake sub woke the agent, exactly once.
    wake_mock.assert_awaited_once()
    assert active_tid in wake_mock.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_notifier_plain_notify_never_wakes_even_with_session_id(kanban_home):
    """Plain/default notify must remain passive even when the task carries a
    creator session_id. This guards against the older unconditional wake path
    that forged adapter.handle_message events after every terminal delivery."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="legacy passive task",
            assignee="worker1",
            session_id="origin-session-id",
        )
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
        kb.block_task(conn, tid, reason="plain notify block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()
    fake_adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    wake_mock = AsyncMock()
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.wake.deliver_wake", new=wake_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    fake_adapter.send.assert_awaited_once()
    wake_mock.assert_not_awaited()
    fake_adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_notifier_notify_wake_does_not_wake_on_status_event(kanban_home):
    """notify+wake wakes on terminal outcomes, not on dashboard status churn."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="status-only task", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="notify+wake",
        )
        kb._append_event(conn, tid, kind="status", payload={"status": "review"})
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
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

    wake_mock = AsyncMock()
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.wake.deliver_wake", new=wake_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    fake_adapter.send.assert_awaited_once()
    wake_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_notifier_wake_forwards_persisted_chat_type_and_user_id(kanban_home):
    """The active-wake call must carry the subscription's persisted chat_type and
    user_id so ``deliver_wake`` resolves the operator's real (e.g. group)
    session instead of a hardcoded one."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="group wake", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="grp1",
            user_id="op-42", chat_type="group", delivery_mode="wake",
            notifier_profile="owner-profile",
        )
        kb.block_task(conn, tid, reason="group block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    fake_adapter = MagicMock()
    fake_adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: fake_adapter}
    runner._profile_adapters = {"owner-profile": {Platform.TELEGRAM: fake_adapter}}
    runner._authorization_adapter = lambda platform, profile=None: fake_adapter

    _orig_sleep = asyncio.sleep
    tick_count = 0

    async def _fast_sleep(_):
        nonlocal tick_count
        await _orig_sleep(0)
        tick_count += 1
        if tick_count >= 3:
            runner._running = False

    wake_mock = AsyncMock()
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.wake.deliver_wake", new=wake_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    wake_mock.assert_awaited_once()
    source = wake_mock.await_args.kwargs["source"]
    assert source.chat_type == "group"
    assert source.user_id == "op-42"
    assert source.profile == "owner-profile"


@pytest.mark.asyncio
async def test_notifier_wake_only_skips_send_and_advances_cursor(kanban_home):
    """wake-only: NO passive send, the agent is woken exactly once, and the
    cursor advances so repeated ticks do not re-wake."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="wake only task", assignee="worker1")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            delivery_mode="wake",
        )
        kb.block_task(conn, tid, reason="wake only block")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
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

    wake_mock = AsyncMock()
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep), \
         patch("gateway.wake.deliver_wake", new=wake_mock):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # wake-only never uses the passive transport...
    fake_adapter.send.assert_not_awaited()
    # ...and wakes the agent exactly once across several ticks (proves the
    # cursor advanced; otherwise it would re-wake on every poll).
    wake_mock.assert_awaited_once()
    assert tid in wake_mock.await_args.kwargs["text"]

    # The subscription survives (blocked is non-terminal) but its cursor moved
    # past the blocked event.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
    finally:
        conn.close()
    assert len(subs) == 1
    assert int(subs[0]["last_event_id"]) > 0


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
    runner._owns_kanban_dispatcher_lock = lambda: True
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
async def test_notifier_wakes_origin_for_review_and_keeps_subscription(kanban_home):
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review handoff", assignee="builder")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat1",
        )
        task = kb.claim_task(conn, task_id, claimer="builder:1")
        assert task is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="Implementation and tests ready.",
            reviewer="reviewer",
            expected_run_id=task.current_run_id,
        )

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    # This legacy subscription has no notifier-profile stamp. Current gateway
    # ownership rules intentionally expose such rows only to the singleton
    # dispatcher owner, which this focused watcher harness represents.
    runner._kanban_dispatcher_lock_handle = object()
    delivered: list[str] = []

    async def _send(chat_id, message, metadata=None):
        delivered.append(message)
        runner._running = False

    adapter = MagicMock()
    adapter.name = "telegram"
    adapter.send = AsyncMock(side_effect=_send)
    runner.adapters = {Platform.TELEGRAM: adapter}

    real_sleep = asyncio.sleep

    async def _fast_sleep(_seconds):
        await real_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    assert any("ready for review" in message for message in delivered)
    assert any("Implementation and tests ready" in message for message in delivered)
    with kb.connect() as conn:
        assert kb.list_notify_subs(conn), "review is non-final; subscription must survive"


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
    runner._owns_kanban_dispatcher_lock = lambda: True
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type="dm",
        thread_id="20197",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice',
        source=source,
        message_id="462",
        reply_to_message_id=None,
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
    assert subs[0]["thread_id"] == "20197"
    assert subs[0]["delivery_metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.parametrize(
    "chat_type,thread_id,thread_sessions_per_user",
    [
        # Isolated group/channel: no thread, so the participant id is appended
        # to the key when group_sessions_per_user is on.
        ("channel", None, False),
        # Thread with per-user isolation on: the participant id is still
        # appended, so the alt id must survive the round-trip here too.
        ("group", "th-42", True),
    ],
)
@pytest.mark.asyncio
async def test_gateway_autosubscribe_roundtrips_user_id_alt_for_session_key(
    kanban_home, chat_type, thread_id, thread_sessions_per_user,
):
    """The gateway `/kanban create` auto-subscribe must persist ``user_id_alt``
    so a replayed wake rebuilds the *same* session key as the original event.

    ``build_session_key`` keys the participant on ``user_id_alt or user_id``
    (Signal UUID / Feishu union_id carry the canonical participant in the alt
    slot). If the subscription row drops ``user_id_alt``, the replayed source
    falls back to ``user_id`` and lands in a different session — the woken turn
    answers into a parallel conversation. Drive the real handler and compare the
    key built from the original source with the key rebuilt from the persisted
    row.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from gateway.session import SessionSource, build_session_key

    runner = object.__new__(GatewayRunner)
    runner._owns_kanban_dispatcher_lock = lambda: True
    # user_id != user_id_alt is the whole point: the alt id is the canonical
    # participant, so dropping it silently corrupts the session key.
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type=chat_type,
        thread_id=thread_id,
        user_id="open-id",
        user_id_alt="union-id",
    )
    event = SimpleNamespace(
        text='/kanban create "hello" --assignee alice',
        source=source,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)
    assert "subscribed" in out.lower()

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn)
    finally:
        conn.close()
    assert len(subs) == 1
    row = subs[0]
    assert row["user_id"] == "open-id"
    assert row["user_id_alt"] == "union-id"

    original = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type=chat_type,
        thread_id=thread_id,
        user_id="open-id",
        user_id_alt="union-id",
    )
    # Reconstruct exactly as the wake path does: from the persisted row.
    replayed = SessionSource(
        platform=Platform(row["platform"]),
        chat_id=row["chat_id"],
        chat_type=row["chat_type"],
        thread_id=row["thread_id"] or None,
        user_id=row["user_id"],
        user_id_alt=row["user_id_alt"],
    )

    original_key = build_session_key(
        original, thread_sessions_per_user=thread_sessions_per_user
    )
    replayed_key = build_session_key(
        replayed, thread_sessions_per_user=thread_sessions_per_user
    )
    assert original_key == replayed_key
    # Regression guard: the canonical alt id — not the raw user_id — is what
    # keys the participant. Proves the alt id actually reached the key.
    assert "union-id" in replayed_key
    assert "open-id" not in replayed_key


@pytest.mark.asyncio
async def test_notifier_artifact_delivery_skips_missing_files(kanban_home, tmp_path, monkeypatch):
    """Missing artifact paths are silently skipped — they may have been
    referenced by name only. The notifier must not crash and must still
    deliver any artifacts that do exist."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    # Allow ``tmp_path`` through the media-delivery safety filter. See the
    # companion test for the full explanation.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))

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
    runner._owns_kanban_dispatcher_lock = lambda: True
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()

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
