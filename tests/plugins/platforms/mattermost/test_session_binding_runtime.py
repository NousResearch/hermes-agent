"""Mattermost binding runtime wiring stays entirely inside the plugin."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.mattermost.adapter import register
from plugins.platforms.mattermost.session_binding_runtime import MattermostSessionBindingRuntime


@pytest.mark.asyncio
async def test_target_normalization_resolves_reply_to_canonical_root():
    runtime = MattermostSessionBindingRuntime()
    adapter = SimpleNamespace(
        is_connected=True,
        _api_get=AsyncMock(return_value={
            "id": "reply1", "root_id": "root1", "channel_id": "channel1"
        }),
    )
    runtime.wire_mattermost(None, adapter)

    assert await runtime.normalize_target("channel1", "reply1") == ("channel1", "root1")
    adapter._api_get.assert_awaited_once_with("posts/reply1")


@pytest.mark.asyncio
async def test_target_normalization_rejects_cross_channel_post():
    runtime = MattermostSessionBindingRuntime()
    runtime.wire_mattermost(None, SimpleNamespace(
        is_connected=True,
        _api_get=AsyncMock(return_value={
            "id": "root1", "root_id": "", "channel_id": "otherchannel"
        }),
    ))

    with pytest.raises(LookupError, match="does not belong"):
        await runtime.normalize_target("channel1", "root1")


def test_plugin_registers_both_platform_handlers_without_core_changes():
    ctx = MagicMock()
    register(ctx)

    registered = [call.args[0] for call in ctx.register_platform_handler.call_args_list]
    assert registered == ["mattermost", "api_server"]
    ctx.register_platform.assert_called_once()
