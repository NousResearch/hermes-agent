"""Answering "always" in /reload-mcp must not claim opt-out when save failed.

``save_config_value`` reports the outcome through its return value; the
caller's ``try``/``except`` never sees a swallowed write failure.
``_on_confirm`` previously ignored the return value and unconditionally
appended the follow-up message. This mirrors the destructive-slash fix
(#75827 sibling fix).
"""

from __future__ import annotations

import sys
import types

import pytest


class _Source:
    platform = "telegram"


class _Event:
    source = _Source()


def _runner():
    """Bare object carrying only what _handle_reload_mcp_command touches."""
    obj = types.SimpleNamespace()
    obj._read_user_config = lambda: {"approvals": {"mcp_reload_confirm": True}}
    obj._session_key_for_source = lambda source: "sess-1"
    captured = {}

    async def _request_slash_confirm(*, event, command, title, message, handler):
        captured["handler"] = handler
        return "prompted"

    obj._request_slash_confirm = _request_slash_confirm
    obj._captured = captured
    return obj


async def _resolve(runner, choice, *, execute_returns="🔄 MCP servers reloaded."):
    """Drive the gate, then invoke the captured handler with *choice*."""
    async def _execute(event=None):
        return execute_returns

    # We test _handle_reload_mcp_command's handler — convert execute to a
    # simple _execute_mcp_reload stub so we don't need a full GatewayRunner.
    runner._execute_mcp_reload = _execute

    from gateway.slash_commands import GatewaySlashCommandsMixin

    await GatewaySlashCommandsMixin._handle_reload_mcp_command(runner, _Event())
    return await runner._captured["handler"](choice)


@pytest.fixture
def fake_cli(monkeypatch):
    """Install a stub `cli` module whose save_config_value outcome is settable."""
    calls = []

    module = types.ModuleType("cli")

    def save_config_value(key_path, value):
        calls.append((key_path, value))
        return module._outcome

    module.save_config_value = save_config_value
    module._outcome = True
    monkeypatch.setitem(sys.modules, "cli", module)
    module.calls = calls
    return module


@pytest.mark.asyncio
async def test_successful_persist_includes_always_followup(fake_cli):
    fake_cli._outcome = True

    out = await _resolve(_runner(), "always")

    assert fake_cli.calls == [("approvals.mcp_reload_confirm", False)]
    # The reload itself succeeded and the opt-out persisted.
    assert "MCP servers reloaded" in out
    assert "will run without confirmation" in out.lower()


@pytest.mark.asyncio
async def test_failed_persist_still_reloads_but_omits_followup(fake_cli):
    fake_cli._outcome = False

    out = await _resolve(_runner(), "always")

    assert fake_cli.calls == [("approvals.mcp_reload_confirm", False)]
    # The reload the user approved still ran.
    assert "MCP servers reloaded" in out
    # ...but must not promise an opt-out that was never written.
    assert "will run without confirmation" not in out.lower()


@pytest.mark.asyncio
async def test_raising_persist_still_reloads_without_followup(fake_cli):
    def _boom(key_path, value):
        raise OSError(30, "Read-only file system")

    fake_cli.save_config_value = _boom

    out = await _resolve(_runner(), "always")

    # The reload still ran despite the write failure.
    assert "MCP servers reloaded" in out
    assert "will run without confirmation" not in out.lower()


@pytest.mark.asyncio
async def test_once_reloads_without_persisting(fake_cli):
    out = await _resolve(_runner(), "once")

    assert fake_cli.calls == []
    assert "MCP servers reloaded" in out


@pytest.mark.asyncio
async def test_cancel_does_not_reload(fake_cli):
    out = await _resolve(_runner(), "cancel")

    assert fake_cli.calls == []
    assert "cancelled" in out.lower()
    assert "MCP servers reloaded" not in out
