"""Tests for write-command RBAC and audit trail in GatewayRunner._handle_message."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text: str = "/sethome") -> MessageEvent:
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chan-1",
        chat_type="group",
        user_id="user-1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = lambda source: True
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_runtime_overrides = {}
    runner.session_store = MagicMock()
    runner.session_store._generate_session_key.return_value = "sess:key"
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner._handle_set_home_command = AsyncMock(return_value="ok-home")
    return runner


def _read_audit_lines(root: Path) -> list[dict]:
    audit_path = root / "logs" / "gateway_command_audit.jsonl"
    if not audit_path.exists():
        return []
    lines = [line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


@pytest.mark.asyncio
async def test_write_command_denied_without_allowlist_match(tmp_path, monkeypatch):
    runner = _make_runner()

    monkeypatch.setenv("GATEWAY_WRITE_ALLOWLIST", "someone-else")
    monkeypatch.delenv("DISCORD_WRITE_ALLOWLIST", raising=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("gateway.run._hermes_home", tmp_path)
        result = await runner._handle_message(_make_event("/sethome"))

    assert "not authorized" in result.lower()
    runner._handle_set_home_command.assert_not_awaited()

    audits = _read_audit_lines(tmp_path)
    assert len(audits) == 1
    assert audits[0]["command"] == "sethome"
    assert audits[0]["authorized"] is False
    assert audits[0]["reason"] == "missing_write_permission"


@pytest.mark.asyncio
async def test_write_command_allowed_and_audited(tmp_path, monkeypatch):
    runner = _make_runner()

    monkeypatch.setenv("DISCORD_WRITE_ALLOWLIST", "user-1")
    monkeypatch.delenv("GATEWAY_WRITE_ALLOWLIST", raising=False)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("gateway.run._hermes_home", tmp_path)
        result = await runner._handle_message(_make_event("/sethome"))

    assert "**/sethome**" in result
    assert "ok-home" in result
    runner._handle_set_home_command.assert_awaited_once()

    audits = _read_audit_lines(tmp_path)
    assert len(audits) == 1
    assert audits[0]["command"] == "sethome"
    assert audits[0]["authorized"] is True
    assert audits[0]["reason"] == "allowlist"
