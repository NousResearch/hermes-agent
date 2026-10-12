"""Answering "Always Approve" must not claim an opt-out that was not saved.

``save_config_value`` swallows its own errors and reports the outcome through
its return value, so the caller's ``try``/``except`` never sees a failed write.
``_on_confirm`` ignored that return value and appended "Future /clear, /new,
/reset, and /undo will run without confirmation" unconditionally, so on any
install whose ``config.yaml`` is not writable (read-only bind mount, container
recreation) the user was told the preference stuck when it had not. The prompt
comes back on the next restart with no explanation.
"""

from __future__ import annotations

import sys
import types

import pytest

import gateway.run as gw


class _Source:
    platform = "telegram"


class _Event:
    source = _Source()


def _runner():
    """Bare object carrying only what _maybe_confirm_destructive_slash touches."""
    obj = types.SimpleNamespace()
    obj._read_user_config = lambda: {"approvals": {"destructive_slash_confirm": True}}
    obj._session_key_for_source = lambda source: "sess-1"
    obj._typed_command_prefix_for = lambda platform: "/"
    captured = {}

    async def _request_slash_confirm(*, event, command, title, message, handler):
        captured["handler"] = handler
        return "prompted"

    obj._request_slash_confirm = _request_slash_confirm
    obj._captured = captured
    return obj


async def _resolve(runner, choice, *, result="🧹 Conversation cleared."):
    """Drive the gate, then invoke the captured handler with *choice*."""
    async def _execute():
        return result

    await gw.GatewayRunner._maybe_confirm_destructive_slash(
        runner,
        event=_Event(),
        command="clear",
        title="Clear conversation",
        detail="This discards history.",
        execute=_execute,
    )
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
async def test_failed_persist_is_reported_as_failed(fake_cli):
    fake_cli._outcome = False

    out = await _resolve(_runner(), "always")

    assert fake_cli.calls == [("approvals.destructive_slash_confirm", False)]
    # The action the user approved still ran.
    assert out.startswith("🧹 Conversation cleared.")
    # ...but the message must not promise an opt-out that was never written.
    assert "will run without confirmation" not in out
    assert "Could not save that preference" in out


@pytest.mark.asyncio
async def test_successful_persist_still_reports_success(fake_cli):
    fake_cli._outcome = True

    out = await _resolve(_runner(), "always")

    assert fake_cli.calls == [("approvals.destructive_slash_confirm", False)]
    assert "will run without confirmation" in out
    assert "Could not save that preference" not in out


@pytest.mark.asyncio
async def test_raising_persist_is_also_reported_as_failed(fake_cli):
    def _boom(key_path, value):
        raise OSError(30, "Read-only file system")

    fake_cli.save_config_value = _boom

    out = await _resolve(_runner(), "always")

    assert out.startswith("🧹 Conversation cleared.")
    assert "will run without confirmation" not in out


@pytest.mark.asyncio
async def test_once_neither_persists_nor_annotates(fake_cli):
    out = await _resolve(_runner(), "once")

    assert fake_cli.calls == []
    assert out == "🧹 Conversation cleared."


@pytest.mark.asyncio
async def test_cancel_does_not_persist(fake_cli):
    out = await _resolve(_runner(), "cancel")

    assert fake_cli.calls == []
    assert "cancelled" in out.lower()
