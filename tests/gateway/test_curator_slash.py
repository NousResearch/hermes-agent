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


def _emit(*lines):
    """Emit through the curator module's output sink (as the real CLI does)."""
    from hermes_cli import curator

    for line in lines:
        curator._emit(line)


# ── Handler-level: _handle_curator_command delegates to run_slash ──────────


@pytest.mark.asyncio
async def test_gateway_curator_status_returns_cli_output():
    """The handler returns the captured output of the curator CLI."""
    runner = _Runner()

    def fake_cli_main(tokens):
        assert tokens == ["status"]
        _emit("curator: ENABLED", "  runs:           0")
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
        _emit("curator: ENABLED")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake_cli_main):
        out = await runner._handle_curator_command(_event("/curator"))

    assert seen["tokens"] == ["status"]
    assert "curator: ENABLED" in out


@pytest.mark.asyncio
async def test_gateway_curator_calls_run_slash_not_cli_main_directly():
    """The handler must route through run_slash (the concurrency-safe entry)."""
    runner = _Runner()

    with patch("hermes_cli.curator.run_slash", return_value="curator: ok") as run_slash:
        out = await runner._handle_curator_command(_event("/curator status"))

    run_slash.assert_called_once()
    assert out == "curator: ok"


# ── run_slash entry point ──────────────────────────────────────────────────


def test_run_slash_status():
    from hermes_cli.curator import run_slash

    with patch("hermes_cli.curator.cli_main", side_effect=lambda tokens: _emit("curator: ENABLED")):
        out = run_slash("status")
    assert "curator: ENABLED" in out


def test_run_slash_defaults_to_status():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: ok")
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
        _emit("ok")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        run_slash("/curator status")
    assert seen["tokens"] == ["status"]


def test_run_slash_captures_output_without_global_stream_swap():
    """run_slash must not mutate process-global stdout/stderr.

    A behavioral proxy: after run_slash returns, sys.stdout is still the real
    stream (not a StringIO), proving the ContextVar buffer path was used
    instead of contextlib.redirect_stdout.
    """
    import sys

    from hermes_cli.curator import run_slash

    real_stdout = sys.stdout
    with patch("hermes_cli.curator.cli_main", side_effect=lambda tokens: _emit("curator: ok")):
        out = run_slash("status")
    assert "curator: ok" in out
    assert sys.stdout is real_stdout


# ── Interactive-subcommand gating (#68884 review) ─────────────────────────


def test_run_slash_rollback_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("rollback")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_prune_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("prune")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_adopt_all_unmanaged_blocked_without_y():
    from hermes_cli.curator import run_slash

    out = run_slash("adopt --all-unmanaged")
    assert "interactive" in out.lower()
    assert "-y" in out


def test_run_slash_adopt_named_skill_not_blocked():
    """Named-skill adopt is non-interactive and must not be gated."""
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: adopted")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("adopt my-skill")
    assert "adopted" in out
    assert seen["tokens"][0] == "adopt"


def test_run_slash_rollback_allowed_with_y():
    from hermes_cli.curator import run_slash

    seen = {}

    def fake(tokens):
        seen["tokens"] = list(tokens)
        _emit("curator: rolled back")
        return 0

    with patch("hermes_cli.curator.cli_main", side_effect=fake):
        out = run_slash("rollback -y")
    assert "rolled back" in out
    assert seen["tokens"][0] == "rollback"
