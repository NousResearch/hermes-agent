"""Gateway /curator slash command (#68880, #68884 review fix)."""

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


# ── #68884 review: run_slash entry point tests ────────────────────────


def test_run_slash_status():
    """run_slash returns captured stdout for status."""
    from hermes_cli.curator import run_slash

    with patch("hermes_cli.curator.cli_main", side_effect=lambda tokens: print("curator: ENABLED")):
        out = run_slash("status")
    assert "curator: ENABLED" in out


def test_run_slash_defaults_to_status():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        print("curator: ok")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("")
    assert seen["tokens"] == ["status"]
    assert "curator: ok" in out


def test_run_slash_strips_curator_prefix():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        print("ok")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        run_slash("/curator status")
    assert seen["tokens"] == ["status"]


def test_run_slash_interactive_subcommand_blocked_without_y():
    """rollback without -y must be rejected with a targeted message."""
    from hermes_cli.curator import run_slash

    out = run_slash("rollback")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_interactive_subcommand_allowed_with_y():
    """rollback with -y must not be blocked by the interactive gate."""
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        print("curator: rolled back")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("rollback -y")
    assert "rolled back" in out
    assert seen["tokens"][0] == "rollback"


def test_run_slash_uses_lock_for_serialization():
    """run_slash source must use the module-level lock."""
    import inspect
    from hermes_cli import curator

    src = inspect.getsource(curator.run_slash)
    assert "_curator_slash_lock" in src


def test_gateway_curator_uses_run_slash_not_cli_main():
    """The gateway handler must call run_slash, not cli_main directly."""
    src = open("gateway/slash_commands.py", encoding="utf-8").read()
    assert "run_slash" in src
    # The old inline redirect pattern must be gone
    assert "contextlib.redirect_stdout" not in src.split("_handle_curator_command")[1].split("def _handle_status_command")[0]