"""gmail-triage inline buttons: a timed-out triage script must be killed, not left to act on the email later."""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig


@pytest.mark.asyncio
async def test_timed_out_triage_script_is_killed(monkeypatch):
    from hermes_constants import get_hermes_home
    from plugins.platforms.telegram.adapter import TelegramAdapter

    scripts = get_hermes_home() / "scripts" / "gmail-triage"
    scripts.mkdir(parents=True)
    (scripts / "archive.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter._callback_authorized = AsyncMock(return_value=True)
    query = SimpleNamespace(answer=AsyncMock(), message=None, edit_message_text=AsyncMock())

    real_exec, real_wait_for = asyncio.create_subprocess_exec, asyncio.wait_for
    spawned = []

    async def _hung_script(*_argv, **kwargs):
        proc = await real_exec(sys.executable, "-c", "import time; time.sleep(120)", **kwargs)
        spawned.append(proc)
        return proc

    async def _expire(aw, timeout):
        # Behave like wait_for() hitting the 60 s cap, without waiting 60 s.
        if getattr(aw, "__qualname__", "") != "Process.communicate":
            return await real_wait_for(aw, timeout)
        aw.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _hung_script)
    monkeypatch.setattr(asyncio, "wait_for", _expire)

    await adapter._handle_gmail_triage_callback(query, "gt:archive:msg-1", {})

    query.answer.assert_awaited_once_with(text="❌ archive timed out")
    [proc] = spawned
    try:
        await real_wait_for(proc.wait(), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        pytest.fail("timed-out gmail-triage script was left running")
