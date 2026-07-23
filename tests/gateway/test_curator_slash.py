"""Gateway /curator slash command (#68880)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin


class _Runner(GatewaySlashCommandsMixin):
    pass


def _event(text: str) -> MessageEvent:
    source = SessionSource(platform="telegram", chat_id="1", user_id="u1")
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=None,
    )


@pytest.mark.asyncio
async def test_gateway_curator_status_returns_cli_output():
    runner = _Runner()

    def fake_cli_main(tokens):
        assert tokens == ["status"]
        print("curator: ENABLED")
        print("  runs:           0")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake_cli_main):
        out = await runner._handle_curator_command(_event("/curator status"))

    assert "curator: ENABLED" in out
    assert "runs:" in out


@pytest.mark.asyncio
async def test_gateway_curator_defaults_to_status():
    runner = _Runner()
    seen = {}

    def fake_cli_main(tokens):
        seen["tokens"] = list(tokens)
        print("curator: ENABLED")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake_cli_main):
        out = await runner._handle_curator_command(_event("/curator"))

    assert seen["tokens"] == ["status"]
    assert "curator: ENABLED" in out


@pytest.mark.asyncio
async def test_gateway_run_dispatches_curator_canonical():
    """Source-level: run.py routes canonical curator to the mixin handler."""
    src = open("gateway/run.py", encoding="utf-8").read()
    assert 'if canonical == "curator":' in src
    assert "return await self._handle_curator_command(event)" in src
