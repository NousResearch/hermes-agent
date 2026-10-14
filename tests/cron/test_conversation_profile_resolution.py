"""Cron surfaces must re-resolve the current inbound profile before seeding."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.conversation import conversation_origin, resolve_delivery_source
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session_identity import identity_of


class Adapter(BasePlatformAdapter):
    supports_inchannel_continuable = True


Adapter.__abstractmethods__ = frozenset()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from pathlib import Path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / "home"
    served = [(name, home if name == "default" else home / "profiles" / name)
              for name in ("default", "ops", "team_b")]
    for _, path in served:
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: served)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda n: dict(served)[n])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda n: n in dict(served))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    config = PlatformConfig(enabled=True, extra={"cron_continuable_surface": "in_channel", "reply_in_thread": False})
    runner.config.platforms = {Platform.SLACK: config}
    runner._primary_profile_name = "default"
    adapter = Adapter.__new__(Adapter)
    adapter.platform, adapter.gateway_runner, adapter.config = Platform.SLACK, runner, config
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {"ops": {}, "team_b": {}}
    return SimpleNamespace(runner=runner, adapter=adapter, home=home)


def inbound(rig, thread_id=None):
    from plugins.platforms.slack.adapter import SlackAdapter
    thread_id = SlackAdapter._session_thread_ts(
        rig.adapter, {"thread_ts": thread_id} if thread_id else {}, "reply-ts", False, {})
    source = rig.adapter.build_source(chat_id="C1", chat_type="group", user_id="alice",
                                      scope_id="T1", thread_id=thread_id)
    rig.adapter._canonicalize(source)
    assert identity_of(source) is not None
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize("new_profile", ["team_b", "unserved"])
async def test_captured_profile_must_not_bypass_current_inbound_route(rig, new_profile):
    rig.runner.config.profile_routes = parse_profile_routes([
        {"platform": "slack", "profile": "ops", "chat_id": "C1"}])
    source = inbound(rig)
    assert identity_of(source).runtime_profile == "ops"
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}
    rig.runner.config.profile_routes = parse_profile_routes([
        {"platform": "slack", "profile": new_profile, "chat_id": "C1"}])
    with pytest.raises(PermissionError):
        await resolve_delivery_source(rig.adapter, "slack", "C1", None, origin)


@pytest.mark.asyncio
async def test_flattening_cannot_transfer_continuation_to_another_profile(rig):
    from cron.scheduler_delivery import _prepare_target_delivery

    rig.runner.config.profile_routes = parse_profile_routes([
        {"platform": "slack", "profile": "ops", "chat_id": "C1", "thread_id": "old-thread"}])
    source = inbound(rig, "old-thread")
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id="old-thread")}
    job = {"id": "digest", "deliver": "origin", "origin": origin, "attach_to_session": True}
    loop = asyncio.get_running_loop()
    errors = []
    delivery = await asyncio.to_thread(
        _prepare_target_delivery, job,
        {"platform": "slack", "chat_id": "C1", "thread_id": "old-thread"},
        adapters=rig.runner.adapters, loop=loop, config=rig.runner.config,
        notify_delivery=True, mirror_enabled=True, mirror_text="brief", delivery_errors=errors)
    assert delivery is None
    assert any("another runtime profile" in error for error in errors)


@pytest.mark.asyncio
@pytest.mark.parametrize("companion_profile", ["ops", "team_b"])
async def test_final_surface_and_companion_use_fresh_routes(rig, monkeypatch, companion_profile):
    from unittest.mock import AsyncMock
    from cron.scheduler_delivery import _deliver_result
    from gateway.platforms.base import SendResult
    from gateway.session import SessionStore

    rig.runner.config.thread_sessions_per_user = True
    rig.runner.config.profile_routes = parse_profile_routes([
        {"platform": "slack", "profile": "ops", "chat_id": "C1"},
        {"platform": "slack", "profile": companion_profile, "chat_id": "C1", "thread_id": "brief-ts"},
    ])
    rig.adapter._session_store = SessionStore(rig.runner.config.sessions_dir, rig.runner.config)
    rig.adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="brief-ts"))
    source = inbound(rig, "old-thread")
    job = {"id": "digest", "deliver": "origin", "attach_to_session": True,
           "origin": {**source.to_dict(), "conversation": conversation_origin(source, thread_id="old-thread")}}
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: rig.runner.config)
    error = await asyncio.to_thread(_deliver_result, job, "private brief", rig.runner.adapters,
                                   asyncio.get_running_loop())
    assert error is None
    store = rig.adapter._session_store
    flat = store.get_or_create_session(inbound(rig))
    assert any("private brief" in m["content"] for m in store.load_transcript(flat.session_id))
    companion = store.get_or_create_session(inbound(rig, "brief-ts"))
    brief = any("private brief" in m["content"] for m in store.load_transcript(companion.session_id))
    assert brief == (companion_profile == "ops")
