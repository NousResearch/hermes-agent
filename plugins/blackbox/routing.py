"""Best-effort card delivery for blackbox alerts."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

# Retain references to fire-and-forget tasks so the event loop doesn't GC them
# mid-flight ("Task was destroyed but it is pending!").
_PENDING_TASKS: set = set()


def _send_with_gateway(card_text: str, platform: str, chat_id: str) -> bool:
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return False
        adapters = getattr(runner, "adapters", {}) or {}
        try:
            adapter = adapters.get(Platform(platform))
        except Exception:
            adapter = None
        adapter = adapter or adapters.get(platform) or adapters.get(str(platform).lower())
        if adapter is None:
            return False

        coro = adapter.send(str(chat_id), card_text)
        if not asyncio.iscoroutine(coro):
            return True

        # on_session_end runs synchronously inside run_conversation, which the
        # gateway drives from a worker thread (run_in_executor). So there is
        # usually NO running loop in this thread. Prefer scheduling onto the
        # gateway's own loop (so the adapter's live connection is used) via
        # run_coroutine_threadsafe; fall back to a fresh loop only as a last
        # resort. When we DO have a running loop (CLI path), retain the task.
        gw_loop = getattr(runner, "loop", None) or getattr(runner, "_loop", None)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            task = running.create_task(coro)
            _PENDING_TASKS.add(task)
            task.add_done_callback(_PENDING_TASKS.discard)
            return True
        if gw_loop is not None and gw_loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, gw_loop)
            return True
        # No reachable loop — last resort: run to completion in a temp loop.
        asyncio.run(coro)
        return True
    except Exception:
        return False


def _notify_script() -> Path:
    return Path.home() / ".hermes" / "skills" / "devops" / "scheduler" / "scripts" / "notify.py"


def _send_with_notify(card_text: str, profile: str) -> None:
    script = _notify_script()
    if not script.exists():
        return
    cmd = [
        sys.executable,
        str(script),
        "--send",
        card_text,
        "--channel",
        "telegram",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    try:
        subprocess.run(
            cmd,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return


def send_card(card_text: str, platform: str = "", chat_id: str = "", profile: str = "") -> None:
    """Deliver a card to the session channel, or Telegram home as fallback."""
    try:
        if platform and chat_id and _send_with_gateway(card_text, platform, chat_id):
            return
        _send_with_notify(card_text, profile)
    except Exception:
        return
