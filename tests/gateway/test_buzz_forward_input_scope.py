"""Forward-only intake stays inside BUZZ_CHANNELS, including DMs (item 6)."""
from __future__ import annotations

import pytest

from pathlib import Path

from tests.gateway.buzz_forward_support import (
    CHANNEL,
    DM_CHANNEL,
    OTHER_PUBKEY,
    _ScriptedCli,
    _make_adapter,
    forward_adapter,
    short_sock_path,
)

pytest_plugins = ["tests.gateway.buzz_forward_support"]


class TestForwardInputScope:
    @pytest.mark.asyncio
    async def test_dm_outside_channels_is_not_adopted(self):
        adapter = _make_adapter(
            {
                "forward_only": True,
                "forward_socket": "/tmp/unused.sock",
                "channels": [CHANNEL],
                "allowed_users": [OTHER_PUBKEY],
                "allowed_destinations": [CHANNEL],
            }
        )
        adapter._input_scope = {CHANNEL}
        cli = _ScriptedCli()
        cli.script("dms", "list", [{"dm_id": DM_CHANNEL}])
        cli.script("channels", "list", [
            {"channel_id": DM_CHANNEL, "name": "DM", "description": ""},
            {"channel_id": CHANNEL, "name": "ops", "description": "ops"},
        ])
        adapter._run_cli = cli
        await adapter._discover_dms(seed=False)
        assert DM_CHANNEL not in adapter._channel_state

    @pytest.mark.asyncio
    async def test_empty_channels_refuses_connect_in_forward_only(self, tmp_path):
        adapter = forward_adapter(short_sock_path(), extra={"channels": []})
        adapter.cli_path = str(tmp_path / "buzz")
        Path(adapter.cli_path).write_text("x", encoding="utf-8")
        Path(adapter._forward_socket).write_text("not-a-socket", encoding="utf-8")
        ok = await adapter.connect()
        assert ok is False
        assert adapter._fatal_error_code == "config_missing"
