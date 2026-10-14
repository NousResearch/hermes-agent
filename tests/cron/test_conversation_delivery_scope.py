"""A persisted workspace address must survive the standalone delivery lane."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.platforms.slack import adapter as slack


@pytest.mark.parametrize("scope", ["TA", "missing"])
def test_standalone_cron_never_tries_a_token_from_another_workspace(scope, monkeypatch):
    posted = []

    async def post(session, token, method, payload, req_kw):
        if method == "auth.test":
            return {"ok": True, "team_id": "TA" if token == "xoxb-a" else "TB"}
        assert method == "chat.postMessage"
        posted.append(token)
        return {"ok": True, "ts": "1.2"}

    monkeypatch.setattr(slack, "_slack_json_post", post)
    monkeypatch.setattr(slack, "_load_slack_bot_tokens", lambda raw, quiet: ["xoxb-b", "xoxb-a"])
    monkeypatch.setattr("tools.send_message_tool._live_adapter", lambda platform: (None, None))
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
    monkeypatch.setattr(platform_registry.get("slack"), "standalone_sender_fn", slack._standalone_send)
    config = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, token="xoxb-b,xoxb-a")})
    job = {"id": "digest", "deliver": "origin", "origin": {
        "platform": "slack", "chat_id": "D1", "scope_id": scope, "chat_type": "dm",
        "conversation": {"version": 1, "kind": "dm", "address": {
            "platform": "slack", "chat_id": "D1", "scope_id": scope,
            "thread_id": None, "profile": "default", "account": None,
        }},
    }}
    with patch("gateway.config.load_gateway_config", return_value=config):
        error = _deliver_result(job, "private report")
    if scope == "TA":
        assert error is None
        assert posted == ["xoxb-a"]
    else:
        assert error
        assert not posted


@pytest.mark.asyncio
@pytest.mark.parametrize("caller,resolved", [("cron", True), ("cron", False), ("branch", True)])
async def test_new_slack_thread_keeps_captured_workspace_after_restart(caller, resolved):
    from cron.scheduler_delivery import _prepare_target_delivery
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.conversation import conversation_origin
    from gateway.session import SessionSource
    from plugins.platforms.slack.adapter import SlackAdapter

    first = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "wrong-root"}))
    intended = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ok": True, "ts": "right-root"}))
    pconfig = PlatformConfig(enabled=True)
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = pconfig
    adapter._app = SimpleNamespace(client=first)
    adapter._team_clients = {"TA": first, "TB": intended}
    adapter._channel_team = {}
    adapter._channel_teams = {}
    source = SessionSource(Platform.SLACK, "C1", chat_type="group", user_id="alice", scope_id="TB")
    if caller == "branch":
        from gateway.run import GatewayRunner
        runner = SimpleNamespace(_delivery_adapter_for=lambda source: adapter)
        destination = await GatewayRunner._branch_open_thread(runner, source, "digest")
        assert destination is not None and destination.scope_id == "TB"
        assert destination.thread_id == "right-root"
        assert first.chat_postMessage.await_count == 0
        intended.chat_postMessage.assert_awaited_once()
        return
    origin = {**source.to_dict(), "conversation": conversation_origin(source, thread_id=None)}
    if not resolved:
        origin.pop("conversation")
        adapter.get_chat_info = AsyncMock(return_value={"type": "unknown"})
    errors = []
    loop = asyncio.get_running_loop()
    delivery = await asyncio.to_thread(
        _prepare_target_delivery,
        {"id": "digest", "origin": origin, "deliver": "origin", "attach_to_session": True},
        {"platform": "slack", "chat_id": "C1"},
        adapters={Platform.SLACK: adapter}, loop=loop,
        config=GatewayConfig(platforms={Platform.SLACK: pconfig}),
        notify_delivery=True, mirror_enabled=True, mirror_text="brief", delivery_errors=errors,
    )
    assert delivery is not None, errors
    if not resolved:
        assert delivery.session_source is None
        assert first.chat_postMessage.await_count == intended.chat_postMessage.await_count == 0
        assert delivery.thread_id is None
        return
    assert delivery.session_source.scope_id == "TB"
    assert first.chat_postMessage.await_count == 0, "Thread root was posted through another workspace's bot"
    assert intended.chat_postMessage.await_count == 1
    assert delivery.thread_id == "right-root"
