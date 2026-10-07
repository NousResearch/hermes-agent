"""A relayed Discord interaction carries the same chat/user labels as a relayed message in that chat.

The text lane gets ``chat_name``, ``chat_topic`` and ``user_display_name`` from the connector; a
forwarded interaction (slash command, component) is the raw Discord body. Both land in one session,
and the pinned session-context prompt renders those labels, so a slash turn built without them
re-rendered the cached prefix and the next message rendered it back.
"""

import asyncio
import json
import sqlite3
import threading
import time
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
import gateway.session as gateway_session
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.relay.ws_transport import _event_from_wire
from gateway.session import SessionStore, build_session_context, build_session_key
from tests.gateway.relay.test_relay_interactive import _adapter


def _forward(**payload):
    body = {"type": 2, "id": "i1", "channel_id": "ch1", "guild_id": "g1", "data": {"name": "status"}}
    body.update(payload)

    class Forward:
        platform, method, path = "discord", "POST", "/interactions/bot1"

    Forward.body = json.dumps(body).encode()
    return Forward()


def _pinned_prompt(source):
    runner = object.__new__(gateway_run.GatewayRunner)
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    return runner._pinned_session_context_prompt(build_session_context(source, config), False, "k")


def _message(chat, user="u1", chat_name="Hermes Server / #ops", chat_topic="Incident triage"):
    message_id = f"{user}:{chat_name}"
    return _event_from_wire({"text": "hi", "message_type": "text", "message_id": message_id, "source": {
        "platform": "discord", **chat, "scope_id": "g1",
        "user_id": user, "user_name": "ben", "user_display_name": "Ben D",
        "chat_name": chat_name, "chat_topic": chat_topic, "message_id": message_id}})


async def _slash(adapter, **interaction):
    """Forward an interaction the way the transport does and return the event it admits."""
    adapter.handle_message = AsyncMock()
    await adapter._on_passthrough(_forward(**interaction))
    return adapter.handle_message.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "warm", "restart", "thread", "rename", "peer-rename", "cleared", "late-session", "read-retry",
    "read-unavailable", "write-unavailable", "cancelled", "cancelled-rename"])
async def test_slash_between_messages_keeps_one_pinned_prompt(tmp_path, monkeypatch, case):
    """Every case but ``warm`` and ``thread`` restarts the gateway after the last message, so the
    slash command is the first event the new process sees in that chat; its chat labels are the
    ones the session store recorded when the text lane carried them.
    ``thread``: both events happen inside a thread, which the text lane keys on chat_type "thread" +
    thread_id; the slash command must land in that session, not in a per-user "group" one.
    ``rename`` / ``peer-rename``: the channel was renamed before the restart. In ``peer-rename``
    another user's message carried the new labels and this user's session was reset since: a reset
    inherits the origin (old labels) under a fresh ``created_at``. ``cleared``: the last message carried
    no labels at all (name and topic removed upstream); the slash command must not revive the old ones.
    ``late-session``: another user's older message, still carrying the old labels, only gets its
    session after the rename was observed. What the chat is called follows the order the labels
    were observed in, not the order its sessions were created in.
    ``read-retry`` / ``read-unavailable``: the first read after the restart raises, or finds no
    database handle (open failed or in backoff). That interaction goes through without labels;
    the next one asks the store again instead of keeping "no labels".
    ``write-unavailable``: the first message's record found no database handle, so nothing was
    written; the next message with the same labels must write them.
    ``cancelled``: the reader is cancelled while the message's labels are being written, before
    the message was admitted. The connector replays the un-ACKed frame, and the replay is admitted.
    ``cancelled-rename``: the cancelled write is still running when the replay and a rename are
    recorded; it must not land after them and bring the old labels back."""
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    chat = ({"chat_id": "th1", "chat_type": "thread", "thread_id": "th1", "parent_chat_id": "ch1"}
            if case == "thread" else {"chat_id": "ch1", "chat_type": "group"})
    renamed = {"chat_name": "Hermes Server / #triage", "chat_topic": "Renamed"}
    calls, held, entered, settled = [], threading.Event(), threading.Event(), threading.Event()

    def watch(store):
        # Note which thread touches the label record; hold the first write in the ``cancelled`` cases.
        db = store._routing_db
        for name in ("set_meta", "get_meta"):
            real = getattr(db, name)

            def spy(key, *args, _real=real, _name=name, **kwargs):
                if key.startswith("gateway_chat_labels:"):
                    calls.append((_name, threading.get_ident()))
                    if _name == "set_meta" and case.startswith("cancelled") and not entered.is_set():
                        entered.set()
                        held.wait(5)
                        try:
                            return _real(key, *args, **kwargs)
                        finally:
                            settled.set()
                return _real(key, *args, **kwargs)

            setattr(db, name, spy)

    def unavailable_once(store, name):
        # What _routing_db_method returns while the database handle cannot be opened.
        real, faults = store._routing_db_method, [name]

        def method(wanted, **kwargs):
            if wanted in faults:
                faults.remove(wanted)
                return None
            return real(wanted, **kwargs)

        store._routing_db_method = method

    watch(store)
    if case == "write-unavailable":
        unavailable_once(store, "set_meta")

    async def relay(event):
        await adapter._on_inbound(event)
        return store.get_or_create_session(event.source)

    if case == "late-session":
        late = _message(chat, "u2")
        await adapter._on_inbound(late)
        message = _message(chat, **renamed)
        await relay(message)
        store.get_or_create_session(late.source)
    else:
        if case == "peer-rename":
            await relay(_message(chat, "u2"))
        message = _message(chat)
        if case.startswith("cancelled"):
            intake = asyncio.create_task(adapter._on_inbound(message))
            await asyncio.to_thread(entered.wait, 5)
            intake.cancel()
            if case == "cancelled":
                held.set()
            else:
                # Released only after the replay and the rename below have had time to record.
                asyncio.get_running_loop().call_later(0.2, held.set)
            with pytest.raises(asyncio.CancelledError):
                await intake
        entry = await relay(message)
        if case.startswith("cancelled"):
            adapter.handle_message.assert_awaited_once_with(message)
        if case == "cancelled-rename":
            message = _message(chat, **renamed)
            await relay(message)
            await asyncio.to_thread(settled.wait, 5)
        if case == "write-unavailable":
            await relay(_message(chat, "u3"))
        if case == "cleared":
            renamed = {"chat_name": None, "chat_topic": None}
        if case in ("rename", "peer-rename", "cleared"):
            await relay(_message(chat, "u1" if case != "peer-rename" else "u2", **renamed))
            message = _message(chat, **renamed)
        if case == "peer-rename":
            store.reset_session(entry.session_key)
    if case not in ("warm", "thread"):
        store = SessionStore(tmp_path, config)
        watch(store)
        adapter, _stub = _adapter(platform="discord")
        adapter.set_session_store(store)
    interaction = {"member": {"nick": "Ben D", "user": {"id": "u1", "username": "ben"}}}
    if case == "thread":
        interaction.update(channel_id="th1", channel={"id": "th1", "type": 11, "parent_id": "ch1"})
    if case == "read-retry":
        get_meta, faults = store._routing_db.get_meta, [OSError("database is locked")]

        def flaky(key):
            if faults:
                raise faults.pop()
            return get_meta(key)

        store._routing_db.get_meta = flaky
    if case == "read-unavailable":
        unavailable_once(store, "get_meta")
    if case in ("read-retry", "read-unavailable"):
        assert (await _slash(adapter, **interaction)).source.chat_name is None
    slash = await _slash(adapter, **interaction)

    assert build_session_key(slash.source) == build_session_key(message.source)
    # With Discord tools loaded the prompt also renders the thread parent and whether a triggering
    # message id exists, so both renderings must agree.
    for tools_loaded in (False, True):
        monkeypatch.setattr(gateway_session, "_discord_tools_loaded", lambda: tools_loaded)
        assert len({_pinned_prompt(event.source) for event in (message, slash, message)}) == 1
    # The label record is disk I/O behind the database's writer lock; the gateway loop never waits on it.
    assert ("set_meta" in {name for name, _ in calls}
            and threading.get_ident() not in {ident for _, ident in calls})


@pytest.mark.asyncio
@pytest.mark.parametrize("member, expected", [
    ({"nick": "Benny", "user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Benny"),
    ({"user": {"id": "u1", "username": "ben", "global_name": "Ben D"}}, "Ben D"),
    ({"user": {"id": "u1", "username": "ben"}}, "ben"),
])
async def test_interaction_names_the_user_as_the_text_lane_would_now(member, expected):
    """The text lane's user_display_name is the native author.display_name (guild nick, else global
    name, else username), and an interaction carries all three as of now. The name an earlier
    message carried ("Ben D" here) must not win: it predates a nickname change."""
    adapter, _stub = _adapter(platform="discord")
    adapter.handle_message = AsyncMock()
    await adapter._on_inbound(_message({"chat_id": "ch1", "chat_type": "group"}))
    assert (await _slash(adapter, member=member)).source.user_name == expected


def _hold_label_writes(db):
    """Hold the next label write until ``held`` is set; ``done`` is set once it has returned or raised."""
    entered, held, done = threading.Event(), threading.Event(), threading.Event()
    real = db.set_meta

    def held_write(key, *args, **kwargs):
        entered.set()
        held.wait(5)
        try:
            return real(key, *args, **kwargs)
        finally:
            done.set()

    db.set_meta = held_write
    return entered, held, done


async def _cancel_mid_write(adapter, entered):
    intake = asyncio.create_task(adapter._on_inbound(_message({"chat_id": "ch1", "chat_type": "group"})))
    await asyncio.to_thread(entered.wait, 5)
    intake.cancel()
    with pytest.raises(asyncio.CancelledError):
        await intake


def _labelled_store(tmp_path):
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="x")})
    store = SessionStore(tmp_path, config)
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    return store, adapter


@pytest.mark.asyncio
async def test_disconnect_waits_for_a_label_write_its_cancelled_reader_left_running(tmp_path):
    """Cancelling the reader does not stop the label write's worker, and shutdown closes the
    session database right after the adapters disconnect. disconnect() must not return while that
    write is still running, or it lands after the close and reopens state.db."""
    store, adapter = _labelled_store(tmp_path)
    entered, held, done = _hold_label_writes(store._routing_db)
    await _cancel_mid_write(adapter, entered)
    asyncio.get_running_loop().call_later(0.2, held.set)
    await adapter.disconnect()
    assert done.is_set()


@pytest.mark.asyncio
async def test_busy_database_delays_a_label_write_briefly_and_the_next_message_records_it(tmp_path):
    """The relay reader awaits the label write before the next frame, so a busy state.db must cost
    the short observation budget, not the routine 20 s one; the next message writes the labels."""
    store, adapter = _labelled_store(tmp_path)
    chat = {"chat_id": "ch1", "chat_type": "group"}
    assert store.chat_labels(Platform.DISCORD, "g1", "ch1") is None
    blocker = sqlite3.connect(store._routing_db.db_path, timeout=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        await adapter._on_inbound(_message(chat))
        elapsed = time.monotonic() - started
    finally:
        blocker.rollback()
        blocker.close()
    assert elapsed < 5
    adapter.handle_message.assert_awaited_once()
    await adapter._on_inbound(_message(chat, "u2"))
    assert store.chat_labels(Platform.DISCORD, "g1", "ch1") == ("Hermes Server / #ops", "Incident triage")


@pytest.mark.asyncio
async def test_a_label_write_running_past_disconnect_cannot_write_after_the_database_closes(
        tmp_path, monkeypatch):
    """disconnect()'s wait is bounded. A write still running past it meets the database shutdown
    closed next, and must fail rather than reopen state.db behind the close checkpoint."""
    import gateway.relay.ws_transport as ws_transport
    from hermes_state_registry import close_all

    store, adapter = _labelled_store(tmp_path)
    db = store._routing_db
    entered, held, done = _hold_label_writes(db)
    await _cancel_mid_write(adapter, entered)
    monkeypatch.setattr(ws_transport, "_env_disconnect_budget_s", lambda: 0.0)
    await adapter.disconnect()
    assert not done.is_set()
    close_all()
    assert db._conn is None
    held.set()
    await asyncio.to_thread(done.wait, 5)
    assert db._conn is None
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute("SELECT count(*) FROM state_meta WHERE key LIKE 'gateway_chat_labels:%'")
        assert rows.fetchone() == (0,)


@pytest.mark.asyncio
async def test_a_label_write_never_opens_the_database_the_relay_reader_waits_on(tmp_path):
    """Opening state.db waits out a sibling's write lock with the routine 20 s patience. With no open
    handle (an earlier open failed and its backoff ran out) the label write is skipped, not an open;
    the next message after the store reopens records the labels."""
    store, adapter = _labelled_store(tmp_path)
    chat = {"chat_id": "ch1", "chat_type": "group"}
    store._db_handle_cache.handles.clear()
    opens, real_open = [], store._open_session_db_for_active_scope

    def counted_open(*args, **kwargs):
        opens.append(1)
        return real_open(*args, **kwargs)

    store._open_session_db_for_active_scope = counted_open
    await adapter._on_inbound(_message(chat))
    adapter.handle_message.assert_awaited_once()
    assert opens == []
    store._open_session_db_for_active_scope = real_open
    assert store._routing_db is not None
    await adapter._on_inbound(_message(chat, "u2"))
    assert store.chat_labels(Platform.DISCORD, "g1", "ch1") == ("Hermes Server / #ops", "Incident triage")


def _restarted(store):
    """A fresh adapter on *store*: its in-memory labels are empty, as after a gateway restart."""
    adapter, _stub = _adapter(platform="discord")
    adapter.set_session_store(store)
    adapter.handle_message = AsyncMock()
    return adapter


def _hold_db_lock(db, seconds=10.0):
    """Hold SessionDB's in-process lock on another thread (as maintenance can) until released."""
    taken, release = threading.Event(), threading.Event()

    def hold():
        with db._lock:
            taken.set()
            release.wait(seconds)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    taken.wait(5)
    return release, holder


@pytest.mark.asyncio
async def test_an_interaction_never_opens_the_database_for_its_labels(tmp_path):
    """After a restart the first interaction reads the recorded labels while the relay reader waits.
    With no open handle that read is skipped, not an open (which waits up to 20 s on a sibling's
    lock), and nothing is cached: the next interaction after the store reopens gets the labels."""
    store, adapter = _labelled_store(tmp_path)
    await adapter._on_inbound(_message({"chat_id": "ch1", "chat_type": "group"}))
    restarted = _restarted(store)
    store._db_handle_cache.handles.clear()
    opens, real_open = [], store._open_session_db_for_active_scope

    def counted_open(*args, **kwargs):
        opens.append(1)
        return real_open(*args, **kwargs)

    store._open_session_db_for_active_scope = counted_open
    assert (await _slash(restarted)).source.chat_name is None
    assert opens == []
    store._open_session_db_for_active_scope = real_open
    assert store._routing_db is not None
    event = await _slash(restarted)
    assert (event.source.chat_name, event.source.chat_topic) == ("Hermes Server / #ops", "Incident triage")


@pytest.mark.asyncio
async def test_a_held_database_lock_costs_the_relay_reader_at_most_the_label_budget(tmp_path):
    """SessionDB's in-process lock has no timeout, and the 0.5 s write budget starts only once it
    is taken. A label read or write waiting on it must give up within the label budget and leave
    nothing cached or recorded, so the next message retries."""
    store, adapter = _labelled_store(tmp_path)
    chat = {"chat_id": "ch1", "chat_type": "group"}
    await adapter._on_inbound(_message(chat))
    restarted = _restarted(store)
    release, holder = _hold_db_lock(store._routing_db)
    try:
        started = time.monotonic()
        assert (await _slash(restarted)).source.chat_name is None
        await restarted._on_inbound(_message(chat, "u2", chat_name="Renamed"))
        elapsed = time.monotonic() - started
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 5)
    assert elapsed < 5
    restarted.handle_message.assert_awaited()
    await restarted._on_inbound(_message(chat, "u3", chat_name="Renamed"))
    assert store.chat_labels(Platform.DISCORD, "g1", "ch1") == ("Renamed", "Incident triage")


@pytest.mark.asyncio
async def test_label_reads_stuck_on_the_database_lock_occupy_one_private_thread(tmp_path):
    """A label read the budget gave up on keeps its thread until SessionDB's lock frees. Retries
    must reuse that read on the adapter's own thread, not start one per interaction on the default
    executor other platforms' I/O shares; once the lock frees, the next interaction gets the labels."""
    store, adapter = _labelled_store(tmp_path)
    await adapter._on_inbound(_message({"chat_id": "ch1", "chat_type": "group"}))
    restarted = _restarted(store)
    restarted._DISCORD_LABELS_IO_BUDGET_S = 0.05
    threads, real_read = [], store.chat_labels

    def counted_read(*args):
        threads.append(threading.current_thread().name)
        return real_read(*args)

    store.chat_labels = counted_read
    release, holder = _hold_db_lock(store._routing_db)
    try:
        for _ in range(5):
            assert (await _slash(restarted)).source.chat_name is None
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 5)
    assert len(threads) == 1
    assert threads[0].startswith("relay-discord-labels-read")
    for _ in range(100):
        if not restarted._discord_labels_reads:
            break
        await asyncio.sleep(0.02)
    event = await _slash(restarted)
    assert (event.source.chat_name, event.source.chat_topic) == ("Hermes Server / #ops", "Incident triage")
    assert len(threads) == 2
    await restarted.disconnect()
    assert restarted._discord_labels_reader is None and restarted._discord_labels_reads == {}
