"""Persistent forum labels via registered ingress (salvage of #51839 / #30014)."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Chat, ForumTopicCreated, ForumTopicEdited, Message, Update, User
from telegram.ext import Application
from telegram.request import BaseRequest

from gateway.channel_directory import build_channel_directory, load_directory
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.session import SessionStore
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter


def adapter(*, token="123456:public-test-fixture", profile="default", topics=None):
    result = TelegramAdapter(PlatformConfig(enabled=True, token=token, extra={"group_topics": topics}))
    result.set_owner_profile(profile)
    result.handle_message = AsyncMock()
    return result


def message(*, created=None, edited=None, reply=None, thread=42, chat_type="supergroup", forum=True, chat_id=-100123):
    return Message(
        message_id=100, date=datetime.now(UTC),
        chat=Chat(chat_id, chat_type, title="Workshop", is_forum=forum),
        from_user=User(7, "Reader", False), message_thread_id=thread,
        is_topic_message=True, text=None if created or edited is not None else "hello",
        forum_topic_created=ForumTopicCreated(created, 7322096) if created else None,
        forum_topic_edited=ForumTopicEdited(name=edited) if edited is not None else None,
        reply_to_message=reply,
    )


class NoNetwork(BaseRequest):
    @property
    def read_timeout(self):
        return 1

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        assert url.rsplit("/", 1)[-1] == "getMe", "No Telegram ingress/egress permitted"
        return 200, json.dumps({"ok": True, "result": {
            "id": 123456, "is_bot": True, "first_name": "Offline", "username": "offline_bot",
        }}).encode()


def registered_app(owner):
    app = (Application.builder().token("123456:public-test-fixture")
           .request(NoNetwork()).get_updates_request(NoNetwork()).build())
    owner._register_handlers(app)
    return app


async def dispatch(app, msg):
    """Exercise PTB dispatch, with only the HTTP transport replaced."""
    update = Update(1, message=msg)
    update.set_bot(app.bot)
    matched = any(handler.check_update(update) for handler in app.handlers[0])
    await app.initialize()
    try:
        await app.process_update(update)
    finally:
        await app.shutdown()
    return matched


def label(owner, msg=None):
    return owner._build_message_event(msg or message(), MessageType.TEXT).source.chat_topic


@pytest.mark.asyncio
async def test_registered_service_updates_and_restart():
    owner = adapter()
    app = registered_app(owner)
    assert await dispatch(app, message(created="Planning")), "textless create must have a registered handler"
    assert label(owner) == "Planning"
    assert await dispatch(app, message(edited="Delivery"))
    assert label(owner) == "Delivery"
    assert await dispatch(app, message(edited=""))
    assert label(owner) == "Delivery"
    owner.handle_message.assert_not_awaited()
    await owner.disconnect()
    fresh = adapter()
    try:
        assert label(fresh) == "Delivery", "restart read must work before any learning write"
        assert fresh._dm_topics == {}
        with SessionDB(get_hermes_home() / "state.db") as db:
            tables = {r[0] for r in db._read_all("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "telegram_dm_topic_mode" not in tables
    finally:
        await fresh.disconnect()


@pytest.mark.asyncio
async def test_rebuilt_application_registers_service_handler(monkeypatch):
    owner = adapter()
    old = registered_app(owner)
    rebuilt = registered_app(owner)
    old.initialize = AsyncMock(side_effect=OSError("synthetic network failure"))
    rebuilt.initialize = AsyncMock(wraps=rebuilt.initialize)
    owner._app = old
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())
    monkeypatch.setattr("plugins.platforms.telegram.adapter._shutdown_abandoned_app", AsyncMock())
    owner._wire_plugin_handlers = lambda app: None
    # Start rebuilt empty: the retry path itself must register its handlers.
    rebuilt.handlers.clear()
    await owner._initialize_app_with_retries(SimpleNamespace(build=lambda: rebuilt))
    assert owner._app is rebuilt
    assert await dispatch(rebuilt, message(created="Restored"))
    assert await dispatch(rebuilt, message(edited="Fresh"))
    assert label(owner) == "Fresh"
    owner.handle_message.assert_not_awaited()
    owner._app = None
    await owner.disconnect()


@pytest.mark.asyncio
async def test_optional_reply_seed_never_rolls_back_rename():
    owner = adapter()
    app = registered_app(owner)
    old_reply = message(created="Original")
    owner._is_user_authorized_from_message = lambda msg: True
    owner._gate_or_observe = lambda *args: True
    owner._ensure_forum_commands = AsyncMock()
    owner._enqueue_text_event = lambda event: None
    try:
        await dispatch(app, message(reply=old_reply))
        assert label(owner, message(reply=old_reply)) == "Original"
        await dispatch(app, message(edited="Renamed"))
        await dispatch(app, message(reply=old_reply))
        assert label(owner, message(reply=old_reply)) == "Renamed"
        assert label(owner) == "Renamed"  # absent reply metadata is normal
        assert label(owner, message(thread=99)) is None
        assert label(owner, message(reply=old_reply, chat_type="private")) is None
        await dispatch(app, message(created="Not a forum", forum=False, thread=98))
        assert label(owner, message(thread=98)) is None
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("mapping", [False, True])
async def test_config_precedence_directory_and_send_list(mapping, capsys, monkeypatch):
    topics = [None, {"thread_id": 42, "name": "Pinned", "skill": "research"}, {"thread_id": 88, "name": "Unseen"}]
    config = {"-100123": topics, "bad": "invalid"} if mapping else [None, {"chat_id": -100123, "chat_name": "Workshop", "topics": topics}]
    owner = adapter(topics=config)
    app = registered_app(owner)
    try:
        await dispatch(app, message(created="Discovered"))
        await dispatch(app, message(edited="Renamed"))
        event = owner._build_message_event(message(), MessageType.TEXT)
        assert (event.source.chat_topic, event.auto_skill) == ("Pinned", "research")
        directory = await build_channel_directory({Platform.TELEGRAM: owner})
        by_id = {r["id"]: r for r in directory["platforms"]["telegram"]}
        assert by_id["-100123:42"]["name"].endswith(" / Pinned")
        assert by_id["-100123:88"]["name"].endswith(" / Unseen")
        assert load_directory() == directory
        from hermes_cli.send_cmd import cmd_send
        monkeypatch.setattr("hermes_cli.send_cmd._load_hermes_env", lambda: None)
        with pytest.raises(SystemExit) as result:
            cmd_send(SimpleNamespace(list_targets=True, message="telegram", json=True))
        assert result.value.code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["platforms"]["telegram"] == directory["platforms"]["telegram"]
        assert (await build_channel_directory({}))["platforms"] == {}
    finally:
        await owner.disconnect()


@pytest.mark.asyncio
async def test_shared_db_transport_bot_chat_thread_isolation(tmp_path):
    owners = [adapter(), adapter(profile="secondary"), adapter(token="654321:public-test-fixture")]
    from hermes_state_registry import acquire, release

    for owner in owners:
        # All adapter homes deliberately point at the SAME DB.
        owner._update_receipt_dir = tmp_path

    shared = acquire(tmp_path / "state.db")
    try:
        await dispatch(registered_app(owners[0]), message(created="Private namespace"))
        assert label(owners[1]) is None
        assert label(owners[2]) is None
        assert label(owners[0], message(thread=77)) is None
        assert label(owners[0], message(chat_id=-100999)) is None
        assert not (await owners[1].list_channels())
        assert not (await owners[2].list_channels())
        assert owners[0]._get_telegram_group_topic_cache_db() is shared
        await owners[0].disconnect()
        shared.set_meta("still-open", "yes")
        assert shared.get_meta("still-open") == "yes"
    finally:
        for owner in owners:
            await owner.disconnect()
        release(shared)


@pytest.mark.asyncio
async def test_reused_origin_and_directory_refresh_without_new_session():
    owner = adapter()
    store = SessionStore(sessions_dir=get_hermes_home() / "sessions", config=GatewayConfig())
    initial = owner._build_observed_message_event(message(), MessageType.TEXT).source
    entry = store.get_or_create_session(initial)
    sid, key = entry.session_id, entry.session_key
    db = store._db_for_key(key)
    db.set_session_title(sid, "Human chosen title")
    try:
        before = await build_channel_directory({Platform.TELEGRAM: owner})
        assert before["platforms"]["telegram"][0]["name"] == "Workshop / topic 42"
        await dispatch(registered_app(owner), message(edited="Release"))
        # A directory rebuild alone must refresh stale session labels.
        after = await build_channel_directory({Platform.TELEGRAM: owner})
        assert after["platforms"]["telegram"][0]["name"] == "Workshop / Release"
        refreshed = owner._build_message_event(message(), MessageType.TEXT).source
        refreshed.chat_name = "Workshop renamed"
        reused = store.get_or_create_session(refreshed)
        assert (reused.session_id, reused.session_key) == (sid, key)
        assert reused.origin.chat_topic == "Release"
        assert reused.origin.chat_name == "Workshop renamed"
        assert reused.display_name == "Workshop renamed"
        row = db.get_session(sid)
        assert json.loads(row["origin_json"])["chat_topic"] == "Release"
        assert row["display_name"] == "Workshop renamed"
        assert row["title"] == "Human chosen title"
        reused.display_name = "Human chosen display label"
        await dispatch(registered_app(owner), message(edited="Shipped"))
        updated = store.get_or_create_session(owner._build_message_event(message(), MessageType.TEXT).source)
        assert updated.origin.chat_topic == "Shipped"
        assert updated.display_name == "Human chosen display label"
        assert db.get_session(sid)["display_name"] == "Human chosen display label"
        assert db.get_session(sid)["title"] == "Human chosen title"
    finally:
        await owner.disconnect()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_cache_uses_transport_home_not_ambient_routed_home(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    transport_home, routed_home = tmp_path / "transport", tmp_path / "routed"
    scope = set_hermes_home_override(transport_home)
    try:
        owner = adapter(profile="receiver")
    finally:
        reset_hermes_home_override(scope)
    scope = set_hermes_home_override(routed_home)
    try:
        await dispatch(registered_app(owner), message(created="Transport scoped"))
        assert label(owner) == "Transport scoped"
        assert (transport_home / "state.db").exists()
        assert not (routed_home / "state.db").exists()
        await owner.disconnect()
        # A fresh adapter in the original transport home can read before any write.
        reset_hermes_home_override(scope)
        scope = set_hermes_home_override(transport_home)
        fresh = adapter(profile="receiver")
        assert label(fresh) == "Transport scoped"
        await fresh.disconnect()
    finally:
        reset_hermes_home_override(scope)


@pytest.mark.asyncio
async def test_legacy_session_directory_refresh_and_dm_unchanged():
    owner = adapter()
    path = get_hermes_home() / "sessions" / "sessions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "legacy-group": {"chat_type": "group", "origin": {
            "platform": "telegram", "chat_id": "-100123", "thread_id": "42", "chat_name": "Workshop",
        }},
        "legacy-dm": {"chat_type": "dm", "origin": {
            "platform": "telegram", "chat_id": "321", "thread_id": "8", "chat_name": "Reader", "chat_topic": "Private topic",
        }},
    }))
    try:
        await dispatch(registered_app(owner), message(created="Learned"))
        channels = {row["id"]: row for row in await owner.list_channels()}
        assert channels["-100123:42"]["name"] == "Workshop / Learned"
        assert channels["321:8"]["name"] == "Reader / Private topic"
    finally:
        await owner.disconnect()
