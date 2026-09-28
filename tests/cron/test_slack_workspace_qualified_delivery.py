"""Workspace-qualified Slack cron delivery invariants.

A target such as ``slack:T123:C456`` must retain the workspace identity through
cron resolution and both delivery lanes.  Losing it can silently select the
primary workspace's Slack client when channel IDs overlap.
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cron import scheduler_delivery
from cron.scheduler_delivery import (
    _TargetDelivery,
    _live_route_metadata,
    _resolve_delivery_targets,
)
from gateway.config import Platform, PlatformConfig


def _ensure_slack_modules(monkeypatch):
    """Install lightweight Slack modules when the optional SDK is absent."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock
    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock
    for name, module in (
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def test_workspace_qualified_targets_parse_and_dedupe_per_workspace(monkeypatch):
    targets = _resolve_delivery_targets({
        "deliver": (
            "slack:TA0C9NPFB:C0B2YUEV7QE,"
            "slack:T0B2UNZKQG3:C0BLAJZR05A,"
            "slack:TA0C9NPFB:C0B2YUEV7QE"
        ),
        "origin": None,
    })

    assert targets == [
        {
            "platform": "slack",
            "chat_id": "C0B2YUEV7QE",
            "thread_id": None,
            "scope_id": "TA0C9NPFB",
            "_workspace_pinned": True,
            "_resolved_from": "explicit",
        },
        {
            "platform": "slack",
            "chat_id": "C0BLAJZR05A",
            "thread_id": None,
            "scope_id": "T0B2UNZKQG3",
            "_workspace_pinned": True,
            "_resolved_from": "explicit",
        },
    ]

    origin = {
        "platform": "slack",
        "chat_id": "C0B2YUEV7QE",
        "thread_id": None,
        "scope_id": "TA0C9NPFB",
        "_resolved_from": "origin",
    }
    real_resolve = scheduler_delivery._resolve_single_delivery_target

    def _resolve(job, value, *, from_broadcast=False):
        if value == "origin":
            return dict(origin)
        return real_resolve(job, value, from_broadcast=from_broadcast)

    monkeypatch.setattr(scheduler_delivery, "_resolve_single_delivery_target", _resolve)
    targets = scheduler_delivery._resolve_delivery_targets({
        "deliver": [
            "slack:C0B2YUEV7QE",
            "origin",
            "slack:TA0C9NPFB:C0B2YUEV7QE",
            "slack:T0B2UNZKQG3:C0B2YUEV7QE",
        ],
        "origin": None,
    })

    assert [(t.get("scope_id"), t["chat_id"]) for t in targets] == [
        ("TA0C9NPFB", "C0B2YUEV7QE"),
        ("T0B2UNZKQG3", "C0B2YUEV7QE"),
    ]


def test_workspace_identity_reaches_live_and_standalone_slack_clients(monkeypatch):
    _ensure_slack_modules(monkeypatch)
    from plugins.platforms.slack import adapter as slack_adapter
    from tools import send_message_tool

    target = _resolve_delivery_targets({
        "id": "workspace-route",
        "deliver": "slack:TA0C9NPFB:C0B2YUEV7QE",
        "origin": None,
    })[0]
    delivery = _TargetDelivery(
        job={"id": "workspace-route"},
        platform=Platform.SLACK,
        platform_name="slack",
        chat_id=target["chat_id"],
        thread_id=None,
        scope_id=target["scope_id"],
        workspace_pinned=target["_workspace_pinned"],
        transport=None,
        pconfig=PlatformConfig(enabled=True, token="unused", extra={}),
        runtime_adapter=None,
        target_adapters={},
        config=None,
        loop=None,
        notify_delivery=True,
        origin={},
        origin_target=False,
        origin_user_id=None,
        is_dm_target=False,
        mirror_text="report",
        mirror_this_target=False,
        in_channel_surface=False,
        inchannel_continuable=False,
        opened_thread_id=None,
    )
    _, route_metadata, media_metadata = _live_route_metadata(delivery)
    assert route_metadata["slack_team_id"] == "TA0C9NPFB"
    assert route_metadata["scope_id"] == "TA0C9NPFB"
    assert route_metadata["workspace_pinned"] is True
    assert media_metadata["slack_team_id"] == "TA0C9NPFB"
    assert media_metadata["workspace_pinned"] is True

    primary = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ts": "primary"}))
    selected = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ts": "selected"}))
    live = slack_adapter.SlackAdapter(PlatformConfig(enabled=True, token="unused", extra={}))
    live._app = SimpleNamespace(client=primary)
    live._team_clients = {"TA0C9NPFB": selected}
    result = asyncio.run(live.send(target["chat_id"], "report", metadata=route_metadata))
    assert result.success is True
    selected.chat_postMessage.assert_awaited_once()
    primary.chat_postMessage.assert_not_awaited()

    missing_metadata = dict(route_metadata, slack_team_id="TUNKNOWN", scope_id="TUNKNOWN")
    missing = asyncio.run(live.send(target["chat_id"], "report", metadata=missing_metadata))
    assert missing.success is False
    primary.chat_postMessage.assert_not_awaited()

    captured = {}

    async def standalone_sender(_pconfig, chat_id, message, **kwargs):
        captured.update(chat_id=chat_id, message=message, kwargs=kwargs)
        return {"success": True, "message_id": "standalone"}

    monkeypatch.setattr(send_message_tool, "_live_adapter", lambda _platform: (None, None))
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry.get",
        lambda name: SimpleNamespace(standalone_sender_fn=standalone_sender) if name == "slack" else None,
    )
    standalone = asyncio.run(send_message_tool._send_to_platform(
        Platform.SLACK,
        PlatformConfig(enabled=True, token="unused", extra={}),
        target["chat_id"],
        "report",
        slack_team_id=target["scope_id"],
    ))
    assert standalone["success"] is True
    assert captured["kwargs"]["team_id"] == "TA0C9NPFB"

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    workspace_lookup = AsyncMock(return_value="selected-token")
    posted_tokens = []

    async def _json_post(_session, token, method, payload, _request_kwargs):
        posted_tokens.append((token, method, payload))
        return {"ok": True, "ts": "standalone-selected"}

    monkeypatch.setattr(slack_adapter, "_resolve_workspace_token", workspace_lookup)
    monkeypatch.setattr(slack_adapter, "_slack_json_post", _json_post)
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: _Session())
    actual = asyncio.run(slack_adapter._standalone_send(
        PlatformConfig(enabled=True, token="primary-token,selected-token", extra={}),
        target["chat_id"],
        "report",
        team_id=target["scope_id"],
    ))
    assert actual["success"] is True
    workspace_lookup.assert_awaited_once_with(
        ["primary-token", "selected-token"], "TA0C9NPFB")
    assert posted_tokens == [
        ("selected-token", "chat.postMessage", {
            "channel": "C0B2YUEV7QE",
            "text": "report",
            "mrkdwn": True,
        })
    ]
