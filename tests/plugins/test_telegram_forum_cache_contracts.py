"""Regressions from independent review R1/R2/R3 of the forum-name salvage."""

import asyncio
import json
import sqlite3
import threading
from unittest.mock import AsyncMock

import pytest

from gateway.channel_directory import build_channel_directory
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageType
from gateway.session import SessionSource, SessionStore
from hermes_constants import get_hermes_home
from hermes_state_errors import StateDbCorruptError, StateDbReplacedError
from hermes_state_registry import acquire, release
from tests.plugins.test_telegram_forum_names import adapter, dispatch, label, message, registered_app


def text_ingress(owner):
    """Only auth/trigger and external housekeeping are replaced, not preparation/enqueue."""
    owner._is_user_authorized_from_message = lambda msg: True
    owner._gate_or_observe = lambda *args: True
    owner._ensure_forum_commands = AsyncMock()
    enqueued = []
    owner._enqueue_text_event = enqueued.append
    return enqueued


@pytest.mark.asyncio
async def test_populated_history_requires_this_bot_observation_or_config():
    owner, other = adapter(), adapter(token="654321:public-test-fixture")
    store = SessionStore(get_hermes_home() / "sessions", GatewayConfig())
    try:
        await dispatch(registered_app(owner), message(created="Only A"))
        store.get_or_create_session(owner._build_message_event(message(), MessageType.TEXT).source)
        assert label(other) is None  # a label lookup alone is not an ingress observation
        assert await other.list_channels() == []
        assert (await build_channel_directory({Platform.TELEGRAM: other}))["platforms"]["telegram"] == []
        enqueued = text_ingress(other)
        await dispatch(registered_app(other), message())
        assert len(enqueued) == 1
        channels = await other.list_channels()
        assert channels[0]["name"] == "Workshop / topic 42"
        assert channels[0]["id"] == "-100123:42"
        assert label(other) is None
    finally:
        await owner.disconnect()
        await other.disconnect()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_unowned_legacy_forum_fails_closed_but_config_and_dm_remain():
    owner = adapter()
    path = get_hermes_home() / "sessions" / "sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "agent:secondary:telegram:group:-100123:topic:42": {"chat_type": "group", "origin": {
            "platform": "telegram", "profile": "secondary", "chat_id": "-100123", "thread_id": "42", "chat_name": "Legacy"}},
        "dm": {"chat_type": "dm", "origin": {"platform": "telegram", "chat_id": "321", "chat_name": "Reader"}},
        "other-transport-dm": {"transport_profile": "secondary", "chat_type": "dm", "origin": {
            "platform": "telegram", "chat_id": "999", "chat_name": "Other transport"}},
    }))
    try:
        assert [row["id"] for row in await owner.list_channels()] == ["321"]
        owner.config.extra["group_topics"] = {"-100123": [{"thread_id": 42, "name": "Configured"}]}
        rows = {row["id"]: row for row in await owner.list_channels()}
        assert rows["-100123:42"]["name"].endswith(" / Configured")
        enqueued = text_ingress(owner)
        owner.config.extra.clear()
        await dispatch(registered_app(owner), message())
        assert len(enqueued) == 1
        assert any(row["id"] == "-100123:42" for row in await owner.list_channels())
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_directory_operation_keeps_registry_lease_through_disconnect(monkeypatch, cancel):
    owner = adapter()
    db = owner._get_telegram_group_topic_cache_db()
    entered, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    original = db.list_gateway_sessions
    original_list = owner._list_group_topic_channels

    def blocked(**kwargs):
        entered.set()
        assert proceed.wait(10), "directory barrier timed out"
        return original(**kwargs)

    def run():
        try:
            return original_list()
        finally:
            finished.set()

    monkeypatch.setattr(db, "list_gateway_sessions", blocked)
    monkeypatch.setattr(owner, "_list_group_topic_channels", run)
    worker = asyncio.create_task(owner.list_channels())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        if cancel:
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
        await owner.disconnect()
        assert db._conn is not None, "pending worker lost its last registry reference"
        current = acquire(get_hermes_home() / "state.db")
        try:
            assert current is db
        finally:
            release(current)
        proceed.set()
        if not cancel:
            await worker
        assert await asyncio.to_thread(finished.wait, 10)
        assert db._conn is None, "finished disconnected worker leaked its reference"
        assert owner._telegram_group_topic_cache_db is None
    finally:
        proceed.set()
        await asyncio.gather(worker, return_exceptions=True)
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("failure", ["open", "read", "reply_write"])
async def test_optional_cache_failure_never_blocks_registered_text(monkeypatch, caplog, configured, failure):
    topics = {"-100123": [{"thread_id": 42, "name": "Pinned", "skill": "research"}]} if configured else None
    owner = adapter(topics=topics)
    enqueued = text_ingress(owner)
    error = sqlite3.OperationalError("synthetic-private-detail")

    def fail(*args, **kwargs):
        raise error

    if failure == "open":
        monkeypatch.setattr(owner, "_get_telegram_group_topic_cache_db", fail)
    else:
        db = owner._get_telegram_group_topic_cache_db()
        method = "get_telegram_group_topic_name" if failure == "read" else "set_telegram_group_topic_name"
        monkeypatch.setattr(db, method, fail)
    app = registered_app(owner)
    errors = []

    async def capture(update, context):
        errors.append(context.error)

    app.add_error_handler(capture)
    try:
        await dispatch(app, message(reply=message(created="Old reply")))
        assert not errors
        assert len(enqueued) == 1
        assert (enqueued[0].source.chat_topic, enqueued[0].auto_skill) == (("Pinned", "research") if configured else (None, None))
        assert "synthetic-private-detail" not in caplog.text
        if failure != "read" or not configured:
            assert "forum topic cache" in caplog.text.lower()
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
async def test_real_cache_table_failure_and_service_failure_stay_non_agent():
    owner = adapter(topics={"-100123": [{"thread_id": 42, "name": "Pinned", "skill": "research"}]})
    db = owner._get_telegram_group_topic_cache_db()
    db._execute_write(lambda conn: conn.execute("DROP TABLE telegram_group_topic_cache"))
    enqueued = text_ingress(owner)
    try:
        await dispatch(registered_app(owner), message(reply=message(created="Original")))
        assert len(enqueued) == 1
        assert (enqueued[0].source.chat_topic, enqueued[0].auto_skill) == ("Pinned", "research")
        await dispatch(registered_app(owner), message(created="Service"))
        assert len(enqueued) == 1
        owner.handle_message.assert_not_awaited()
        db.set_meta("unrelated", "works")
        assert db.get_meta("unrelated") == "works"
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [StateDbCorruptError("storage"), StateDbReplacedError("storage"), OSError("storage"), sqlite3.DatabaseError("storage")])
async def test_known_storage_errors_are_optional(monkeypatch, error):
    owner = adapter()
    def fail():
        raise error
    monkeypatch.setattr(owner, "_get_telegram_group_topic_cache_db", fail)
    try:
        assert label(owner) is None
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("live-system guard"), ValueError("programming"), sqlite3.ProgrammingError("programming"), sqlite3.IntegrityError("programming"), FileNotFoundError("profile guard")])
async def test_identity_guards_and_programming_errors_are_not_swallowed(monkeypatch, error):
    owner = adapter()
    def fail():
        raise error
    monkeypatch.setattr(owner, "_get_telegram_group_topic_cache_db", fail)
    try:
        with pytest.raises(type(error)):
            label(owner)
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("cold", [False, True])
async def test_acquisition_racing_disconnect_cannot_publish_late_reference(monkeypatch, cold):
    owner = adapter()
    if not cold:
        owner._get_telegram_group_topic_cache_db()
    entered, proceed = threading.Event(), threading.Event()
    held = []

    def blocked_acquire(path):
        db = acquire(path)
        held.append(db)
        entered.set()
        assert proceed.wait(10), "acquisition barrier timed out"
        return db

    monkeypatch.setattr("hermes_state_registry.acquire", blocked_acquire)
    worker = asyncio.create_task(owner.list_channels())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        await owner.disconnect()
        proceed.set()
        assert await worker == []
        assert owner._telegram_group_topic_cache_db is None
        assert held[0]._conn is None
        with pytest.raises(RuntimeError):
            owner._get_telegram_group_topic_cache_db()
        fresh = acquire(get_hermes_home() / "state.db")
        try:
            assert fresh is not held[0]
            assert held[0]._conn is None
        finally:
            release(fresh)
    finally:
        proceed.set()
        await asyncio.gather(worker, return_exceptions=True)
        await owner.disconnect()


@pytest.mark.asyncio
async def test_only_explicit_reconnect_resumes_cache(monkeypatch):
    from telegram import Update
    from tests.plugins.test_telegram_update_admission import connected

    async with connected(monkeypatch) as (owner, app, delivered):
        update = Update(10, message=message(created="Persisted"))
        update.set_bot(app.bot)
        await app.process_update(update)
        assert label(owner) == "Persisted"
        await owner.disconnect()
        assert label(owner) is None
        assert await owner.list_channels() == []
        assert owner._telegram_group_topic_cache_db is None
        assert await owner.connect(is_reconnect=True)
        assert label(owner) == "Persisted"
        assert not delivered


@pytest.mark.asyncio
async def test_real_missing_named_profile_guard_is_not_optional(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    root.mkdir()
    missing = root / "profiles" / "missing"
    token = set_hermes_home_override(missing)
    try:
        owner = adapter(profile="missing")
        try:
            with pytest.raises(FileNotFoundError, match="Named profile"):
                label(owner)
            assert not missing.exists()
        finally:
            await owner.disconnect()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_cache_table_failure_preserves_dm_nonforum_and_configured_directory():
    owner = adapter(topics={"-100123": [{"thread_id": 42, "name": "Pinned"}]})
    store = SessionStore(get_hermes_home() / "sessions", GatewayConfig())
    try:
        for source in (
            SessionSource(Platform.TELEGRAM, "321", chat_name="Reader", chat_type="dm"),
            SessionSource(Platform.TELEGRAM, "-100999", chat_name="Basic group", chat_type="group"),
            SessionSource(Platform.TELEGRAM, "-100123", chat_name="Unowned forum", chat_type="group", thread_id="99"),
        ):
            store.get_or_create_session(source)
        db = owner._get_telegram_group_topic_cache_db()
        db._execute_write(lambda conn: conn.execute("DROP TABLE telegram_group_topic_cache"))
        channels = {row["id"]: row for row in await owner.list_channels()}
        assert set(channels) == {"321", "-100999", "-100123:42"}
        assert channels["321"]["name"] == "Reader"
        assert channels["-100999"]["name"] == "Basic group"
        assert channels["-100123:42"]["name"].endswith(" / Pinned")
    finally:
        await owner.disconnect()
        store.close_all_db_handles()
