"""A delivered cron brief must reach the actual reply transcript, across profiles and legacy kinds."""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionStore


@pytest.fixture
def homes(tmp_path, monkeypatch):
    import hermes_state
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "hermes"
    ops = root / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: ops)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "ops")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kw: [("default", root), ("ops", ops)])
    return root, ops



class DeliveryAdapter(BasePlatformAdapter):
    supports_inchannel_continuable = True

    async def get_chat_info(self, chat_id):
        return {"type": "dm", "scope_id": "T1"}

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id="brief-ts")


DeliveryAdapter.__abstractmethods__ = frozenset()


@pytest.mark.asyncio
async def test_delivered_brief_stays_in_owning_runtime_store(homes, monkeypatch):
    from cron.scheduler_delivery import _deliver_result
    from gateway.profile_routing import parse_profile_routes
    from gateway.run import GatewayRunner
    from gateway.session_identity import identity_of

    root, ops = homes
    pconfig = PlatformConfig(enabled=True, extra={"cron_continuable_surface": "in_channel"})
    config = GatewayConfig(multiplex_profiles=True, platforms={Platform.SLACK: pconfig})
    config.profile_routes = parse_profile_routes([
        {"platform": "slack", "profile": "ops", "chat_id": "D1"}])
    runner = object.__new__(GatewayRunner)
    runner.config, runner._primary_profile_name = config, "default"
    adapter = DeliveryAdapter.__new__(DeliveryAdapter)
    adapter.platform, adapter.gateway_runner, adapter.config = Platform.SLACK, runner, pconfig
    adapter._session_store = SessionStore(root / "sessions", config)
    adapter.sent = []
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {"ops": {}}
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    job = {"id": "cross-runtime-brief", "deliver": "slack:D1", "attach_to_session": True}
    error = await asyncio.to_thread(_deliver_result, job, "cross-runtime brief", runner.adapters, asyncio.get_running_loop())
    assert error is None
    assert adapter.sent == [("D1", "cross-runtime brief")]
    reply = adapter.build_source(chat_id="D1", chat_type="dm", user_id="alice", scope_id="T1")
    adapter._canonicalize(reply)
    assert identity_of(reply).runtime_profile == "ops"
    entry = adapter._session_store.get_or_create_session(reply)
    with sqlite3.connect(ops / "state.db") as db:
        assert db.execute("SELECT id FROM sessions WHERE id=?", (entry.session_id,)).fetchone()
        transcript = db.execute("SELECT content FROM messages WHERE session_id=?", (entry.session_id,)).fetchall()
    assert any("cross-runtime brief" in (row[0] or "") for row in transcript), (entry.session_key, transcript)


@pytest.mark.asyncio
async def test_legacy_dingtalk_brief_and_actual_reply_share_session(homes, monkeypatch):
    from cron.scheduler_delivery import _deliver_result
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    root, _ = homes
    pconfig = PlatformConfig(enabled=True)
    config = GatewayConfig(platforms={Platform.DINGTALK: pconfig})
    adapter = DingTalkAdapter(pconfig)
    store = adapter._session_store = SessionStore(root / "sessions", config)
    events = []
    async def capture(event):
        events.append(event)
    adapter.handle_message = capture
    adapter._http_client = SimpleNamespace(post=AsyncMock(return_value=SimpleNamespace(status_code=200)))
    def incoming(mid):
        return SimpleNamespace(message_id=mid, conversation_id="cidABC123", conversation_type="2",
            sender_id="alice", sender_nick="Alice", text={"content": "continue"},
            is_in_at_list=True, session_webhook="https://oapi.dingtalk.com/fake-review-webhook",
            session_webhook_expired_time=9999999999999)
    await adapter._on_message(incoming("before-cron"))
    assert len(events) == 1
    before = store.get_or_create_session(events[-1].source)
    assert before.origin.chat_type == "group"
    origin = {"platform": "dingtalk", "chat_id": "cidABC123", "user_id": "alice"}
    job = {"id": "legacy-dingtalk", "deliver": "origin", "origin": origin, "attach_to_session": True}
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler.load_config", lambda: {"cron": {"wrap_response": False}})
    error = await asyncio.to_thread(_deliver_result, job, "DingTalk brief", {Platform.DINGTALK: adapter}, asyncio.get_running_loop())
    assert error is None
    adapter._http_client.post.assert_awaited_once()
    assert adapter._http_client.post.call_args.kwargs["json"]["markdown"]["text"] == "DingTalk brief"
    await adapter._on_message(incoming("after-cron"))
    assert len(events) == 2
    reply = store.get_or_create_session(events[-1].source)
    assert reply.session_id == before.session_id
    transcript = store.load_transcript(reply.session_id)
    assert any("DingTalk brief" in message.get("content", "") for message in transcript), (
        reply.session_key, [(e.session_key, e.session_id) for e in store.list_sessions()])


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["dingtalk", "wecom", "qqbot"])
async def test_unobserved_conversation_never_guesses_kind_from_id(platform):
    from gateway.platforms.qqbot.adapter import QQAdapter
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter
    from plugins.platforms.wecom.adapter import WeComAdapter

    cls = {"dingtalk": DingTalkAdapter, "wecom": WeComAdapter, "qqbot": QQAdapter}[platform]
    adapter = cls(PlatformConfig(enabled=True))
    for chat_id in ("cidABC123", "groupLookingId"):
        assert (await adapter.get_chat_info(chat_id))["type"] == "unknown"


def test_retry_spool_uses_the_target_profile_home(homes, monkeypatch):
    import sqlite3
    from cron.scheduler_sessions import _seed_cron_session
    from gateway.session import SessionSource
    from gateway.session_identity import RoutingIdentity

    root, ops = homes
    store = SessionStore(root / "sessions", GatewayConfig(multiplex_profiles=True))
    source = SessionSource(Platform.SLACK, "D1", scope_id="T1", profile="ops")
    source._identity = RoutingIdentity("default", "ops", root, ops)
    entry = store.get_or_create_session(source)
    db = store._db_for_key(entry.session_key)
    def fail_append(**kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(db, "append_message", fail_append)
    monkeypatch.setattr(store, "_MAX_PENDING_PER_SESSION", 1)
    adapter = SimpleNamespace(_session_store=store)
    for text in ("first brief", "second brief"):
        assert _seed_cron_session({"id": "digest"}, adapter, "slack", "D1", text, thread_id=None,
                                  chat_type="dm", user_id=None, chat_name=None, scope_id="T1", source=source)
    assert list((ops / "pending_messages").glob("*.json"))
    assert not (root / "pending_messages").exists()
    store.close_all_db_handles()
