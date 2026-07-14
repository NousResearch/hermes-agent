import asyncio
import json
import pytest

from pathlib import Path
from types import SimpleNamespace
from hermes_cli import config as hermes_config
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


def _configure_subscription_policy(**kanban):
    """Persist policy settings through the normal profile-scoped config path."""
    config = hermes_config.load_config()
    config["kanban"].update(kanban)
    hermes_config.save_config(config)


def _subscription_identities(conn, task_id):
    return {
        (row["platform"], row["chat_id"], row["thread_id"])
        for row in kb.list_notify_subs(conn, task_id)
    }


def test_task_creation_subscription_policy_precedence(kanban_home):
    """The public task seam chooses one source in the documented order."""
    _configure_subscription_policy(
        auto_subscribe_on_create=True,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    ambient = kb.NotificationTarget("telegram", "ambient", "thread", "user", "owner")
    explicit = kb.NotificationTarget("discord", "explicit")

    with kb.connect() as conn:
        legacy_task = kb.create_task(conn, title="legacy")
        default_task = kb.create_task(
            conn,
            title="defaults",
            subscription_context=kb.SubscriptionContext(),
        )
        ambient_task = kb.create_task(
            conn,
            title="ambient",
            subscription_context=kb.SubscriptionContext(ambient_origin=ambient),
        )
        explicit_task = kb.create_task(
            conn,
            title="explicit",
            subscription_context=kb.SubscriptionContext(explicit_targets=(explicit,), ambient_origin=ambient),
        )
        quiet_task = kb.create_task(
            conn,
            title="quiet",
            subscription_context=kb.SubscriptionContext(
                no_subscribe=True, explicit_targets=(explicit,), ambient_origin=ambient
            ),
        )

        assert _subscription_identities(conn, legacy_task) == set()
        assert _subscription_identities(conn, default_task) == {("default", "fallback", "")}
        assert _subscription_identities(conn, ambient_task) == {("telegram", "ambient", "thread")}
        assert _subscription_identities(conn, explicit_task) == {("discord", "explicit", "")}
        assert _subscription_identities(conn, quiet_task) == set()


def test_task_creation_explicit_subscriptions_ignore_disabled_automatic_gate(kanban_home):
    _configure_subscription_policy(
        auto_subscribe_on_create=False,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
    )
    explicit = kb.NotificationTarget("discord", "explicit")
    ambient = kb.NotificationTarget("telegram", "ambient")

    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        kb.add_notify_sub(conn, task_id=parent, platform="parent", chat_id="inherited")
        automatic_task = kb.create_task(
            conn,
            title="automatic",
            subscription_context=kb.SubscriptionContext(
                ambient_origin=ambient,
            ),
            parents=(parent,),
        )
        explicit_task = kb.create_task(
            conn,
            title="explicit",
            subscription_context=kb.SubscriptionContext(explicit_targets=(explicit,), ambient_origin=ambient),
        )

        assert _subscription_identities(conn, automatic_task) == set()
        assert _subscription_identities(conn, explicit_task) == {("discord", "explicit", "")}


@pytest.mark.parametrize(
    ("depth", "expected"),
    [(0, {("default", "fallback", "")}), (1, {("parent", "one", "")}), ("unlimited", {("parent", "one", ""), ("ancestor", "two", "")})],
)
def test_task_creation_inherits_subscriptions_at_configured_depth(kanban_home, depth, expected):
    _configure_subscription_policy(
        auto_subscribe_on_create=False,
        notify_default_targets=[{"platform": "default", "chat_id": "fallback"}],
        notify_inherit_depth=depth,
    )
    with kb.connect() as conn:
        grandparent = kb.create_task(conn, title="grandparent")
        kb.add_notify_sub(conn, task_id=grandparent, platform="ancestor", chat_id="two")
        parent = kb.create_task(conn, title="parent", parents=(grandparent,))
        kb.add_notify_sub(conn, task_id=parent, platform="parent", chat_id="one")
        _configure_subscription_policy(auto_subscribe_on_create=True)
        child = kb.create_task(
            conn,
            title="child",
            parents=(parent,),
            subscription_context=kb.SubscriptionContext(),
        )

        assert kb.parent_ids(conn, child) == [parent]
        assert _subscription_identities(conn, child) == expected


def test_task_creation_deduplicates_normalized_targets_and_keeps_ownership(kanban_home):
    _configure_subscription_policy(auto_subscribe_on_create=True)
    target = kb.NotificationTarget(" Telegram ", "chat", "topic", "user-1", "notifier-1")
    duplicate = kb.NotificationTarget("telegram", "chat", "topic", "user-2", "notifier-2")

    with kb.connect() as conn:
        first_task_id = kb.create_task(
            conn,
            title="dedupe",
            idempotency_key="delivery-123",
            subscription_context=kb.SubscriptionContext(explicit_targets=(target, duplicate)),
        )
        retry_task_id = kb.create_task(
            conn,
            title="dedupe retry",
            idempotency_key="delivery-123",
            subscription_context=kb.SubscriptionContext(explicit_targets=(duplicate,)),
        )
        rows = kb.list_notify_subs(conn, first_task_id)
        task_count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?", ("delivery-123",)
        ).fetchone()[0]

    assert retry_task_id == first_task_id
    assert task_count == 1
    assert [(row["platform"], row["chat_id"], row["thread_id"], row["user_id"], row["notifier_profile"])
            for row in rows] == [("telegram", "chat", "topic", "user-1", "notifier-1")]


@pytest.mark.parametrize(
    ("invalid_kanban", "message"),
    [
        ("not-a-mapping", "kanban must be a mapping"),
        ({"auto_subscribe_on_create": 1}, "kanban.auto_subscribe_on_create must be a bool"),
        ({"notify_default_targets": {}}, "kanban.notify_default_targets must be a list"),
        ({"notify_default_targets": ["not-a-mapping"]}, "target mappings"),
        ({"notify_default_targets": [{"platform": "telegram", "chat_id": None}]}, "chat_id"),
        ({"notify_default_targets": [{"platform": True, "chat_id": "chat"}]}, "platform"),
        ({"notify_default_targets": [{"platform": "telegram", "chat_id": "chat", "thread_id": 1}]}, "thread_id"),
        ({"notify_default_targets": [{"platform": "telegram", "chat_id": "chat", "user_id": False}]}, "user_id"),
        ({"notify_default_targets": [{"platform": "telegram", "chat_id": "chat", "notifier_profile": 1}]}, "notifier_profile"),
        ({"notify_inherit_depth": True}, "kanban.notify_inherit_depth"),
        ({"notify_inherit_depth": -1}, "kanban.notify_inherit_depth"),
    ],
)
def test_invalid_subscription_policy_rolls_back_entire_task_creation(
    kanban_home, invalid_kanban, message,
):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="existing parent")

    config = hermes_config.load_config()
    config["kanban"] = invalid_kanban
    hermes_config.save_config(config, strip_defaults=False)

    with kb.connect() as conn:
        before_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        before_links = conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0]
        before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        before_subscriptions = conn.execute("SELECT COUNT(*) FROM kanban_notify_subs").fetchone()[0]

        with pytest.raises(ValueError, match=message):
            kb.create_task(
                conn,
                title="must not persist",
                parents=(parent,),
                subscription_context=kb.SubscriptionContext(),
            )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before_tasks
        assert conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0] == before_links
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before_events
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs").fetchone()[0] == before_subscriptions


def test_default_target_normalization_keeps_blank_optional_metadata_empty(kanban_home):
    _configure_subscription_policy(
        notify_default_targets=[{
            "platform": " Telegram ",
            "chat_id": " chat ",
            "thread_id": " ",
            "user_id": " ",
            "notifier_profile": " ",
        }],
    )

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="defaults", subscription_context=kb.SubscriptionContext(),
        )
        rows = kb.list_notify_subs(conn, task_id)

    assert [(row["platform"], row["chat_id"], row["thread_id"], row["user_id"], row["notifier_profile"])
            for row in rows] == [("telegram", "chat", "", None, None)]


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

        # Cycle 1: blocked for one reason
        kb.block_task(conn, tid, reason="first block", kind="needs_input")
    finally:
        conn.close()

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Cycle 2: unblock → block again for a DIFFERENT reason. A distinct
    # block cause must still notify. (A *same*-cause re-block instead trips
    # the unblock-loop breaker and routes to triage — covered by
    # test_kanban_block_kinds.py; here we exercise two genuinely different
    # blocks, which is the case the user wants notified twice.)
    runner._running = True
    tick_count = 0

    conn = kb.connect()
    try:
        kb.unblock_task(conn, tid)
        kb.block_task(conn, tid, reason="second block", kind="capability")
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

    The shared creation seam receives the gateway origin even when the
    shared `--board` flag appears before the subcommand.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    kb.create_board("projx")

    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "gateway-prof"
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        thread_id="th1",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice --json',
        source=source,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)

    assert json.loads(out)["title"] == "hello"

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
    assert subs[0]["user_id"] == "u1"
    assert subs[0]["notifier_profile"] == "gateway-prof"

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_gateway_create_json_output_stays_parseable_and_subscribes(kanban_home):
    """Gateway origin subscription does not alter structured command output."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "gateway-prof"
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        thread_id="th1",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban create "json hello" --assignee alice --json',
        source=source,
    )

    data = json.loads(await GatewayRunner._handle_kanban_command(runner, event))

    assert data["title"] == "json hello"
    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, data["id"])
    assert [(row["platform"], row["chat_id"], row["thread_id"], row["user_id"], row["notifier_profile"])
            for row in subs] == [("telegram", "chat1", "th1", "u1", "gateway-prof")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("automatic", "command", "expected"),
    [
        (False, 'create "disabled ambient" --json', set()),
        (False, 'create "disabled explicit" --notify discord:channel --json',
         {("discord", "channel", "", None, "gateway-prof")}),
        (True, 'create "explicit wins" --notify discord:channel --json',
         {("discord", "channel", "", None, "gateway-prof")}),
        (True, 'create "same-platform explicit" --notify telegram:target-chat --json',
         {("telegram", "target-chat", "", "u1", "gateway-prof")}),
        (True, 'create "no subscriptions" --notify discord:channel --no-subscribe --json', set()),
    ],
    ids=("disabled-suppresses-ambient", "disabled-keeps-explicit", "explicit-overrides-ambient", "same-platform-explicit-keeps-user", "no-subscribe-suppresses-all"),
)
async def test_gateway_create_subscription_policy(
    kanban_home, automatic, command, expected,
):
    """Gateway creation applies the shared subscription policy at its public seam."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    _configure_subscription_policy(auto_subscribe_on_create=automatic)
    runner = object.__new__(GatewayRunner)
    runner._kanban_notifier_profile = "gateway-prof"
    event = SimpleNamespace(
        text=f"/kanban {command}",
        source=SimpleNamespace(
            platform=Platform.TELEGRAM,
            chat_id="ambient-chat",
            thread_id="ambient-thread",
            user_id="u1",
        ),
    )

    task = json.loads(await GatewayRunner._handle_kanban_command(runner, event))
    with kb.connect() as conn:
        subscriptions = kb.list_notify_subs(conn, task["id"])

    assert {
        (row["platform"], row["chat_id"], row["thread_id"],
         row["user_id"], row["notifier_profile"])
        for row in subscriptions
    } == expected


@pytest.mark.asyncio
async def test_notifier_uploads_artifacts_on_completion(kanban_home, tmp_path, monkeypatch):
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

    # ``_deliver_kanban_artifacts`` routes candidates through
    # ``BasePlatformAdapter.filter_local_delivery_paths``, which only accepts
    # paths under ``MEDIA_DELIVERY_SAFE_ROOTS`` or roots explicitly allowlisted
    # via ``HERMES_MEDIA_ALLOW_DIRS``. Test fixtures live under ``tmp_path``,
    # so allowlist it for the duration of the test.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))

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
