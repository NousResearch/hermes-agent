from __future__ import annotations

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    script_dir = home / "scripts"
    script_dir.mkdir(parents=True)
    script = script_dir / "ack_escalation_notify.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._message_handler = None
    return adapter, script


@pytest.mark.asyncio
async def test_ack_alert_callback_runs_ack_script(monkeypatch, tmp_path):
    adapter, script = _make_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr("plugins.platforms.telegram.adapter.subprocess.run", fake_run)
    query = SimpleNamespace(
        data="ha:abc12345",
        from_user=SimpleNamespace(id=777, first_name="Sasha"),
        message=SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type="supergroup"),
            message_thread_id=1276,
        ),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    assert calls
    cmd = calls[0][0]
    assert cmd[1] == str(script)
    assert cmd[2:] == ["ack", "abc12345", "--user", "Sasha", "--user-id", "777", "--json"]
    query.answer.assert_awaited_once_with(text="✅ Подтверждено. Звонки остановлены.")


@pytest.mark.asyncio
async def test_ack_alert_callback_requires_authorized_user(monkeypatch, tmp_path):
    adapter, _script = _make_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(adapter, "_is_callback_user_authorized", lambda *a, **kw: False)
    ran = False

    def fake_run(*args, **kwargs):
        nonlocal ran
        ran = True
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("plugins.platforms.telegram.adapter.subprocess.run", fake_run)
    query = SimpleNamespace(
        data="ha:abc12345",
        from_user=SimpleNamespace(id=999, first_name="Intruder"),
        message=SimpleNamespace(
            chat_id=-100123,
            chat=SimpleNamespace(type="supergroup"),
            message_thread_id=1276,
        ),
        answer=AsyncMock(),
    )
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, SimpleNamespace())

    assert ran is False
    query.answer.assert_awaited_once_with(text="⛔ You are not authorized to acknowledge this alert.")
