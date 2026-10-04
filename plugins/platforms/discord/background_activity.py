"""Discord presence and single-message dashboard for background AI work."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli.background_activity import list_all_active_work
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
USAGE_CHANNEL_ID = 1550862255925502072
_MARKER = "`Hermes background work`"
_STATE_FILE = "discord_background_activity.json"


def render_presence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Idle"
    first = str(items[0].get("title") or "Background worker")
    prefix = f"{len(items)} worker{'s' if len(items) != 1 else ''} · "
    return (prefix + first)[:96]


def render_dashboard(items: list[dict[str, Any]], *, now: float | None = None) -> str:
    stamp = int(time.time() if now is None else now)
    if not items:
        return f"{_MARKER}\n🟢 **Idle**\nNo background AI workers are active.\n<t:{stamp}:R>"
    count = len(items)
    lines = [
        _MARKER,
        f"🟠 **{count} active worker{'s' if count != 1 else ''}**",
    ]
    for item in items[:8]:
        identity = " · ".join(part for part in (
            str(item.get("profile") or "default"), str(item.get("model") or ""),
            str(item.get("worker") or "worker"),
        ) if part)
        elapsed = str(item.get("elapsed") or "0s")
        started = int(float(item.get("started_at") or stamp))
        lines.append(f"• **{item.get('title') or 'Background worker'}** — {elapsed}\n  {identity} · <t:{started}:T>")
    if count > 8:
        lines.append(f"• …and {count - 8} more")
    lines.append(f"Updated <t:{stamp}:R>")
    return "\n".join(lines)


def _identity(items: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(item.get("key") or "") for item in items))


@dataclass
class PublishGate:
    periodic_seconds: float = 15.0
    minimum_interval_seconds: float = 1.0
    _last_identity: tuple[str, ...] | None = None
    _last_publish: float = 0.0

    def should_publish(self, items: list[dict[str, Any]], *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if self._last_identity is not None and current - self._last_publish < self.minimum_interval_seconds:
            return False
        identity = _identity(items)
        changed = identity != self._last_identity
        due = bool(items) and current - self._last_publish >= self.periodic_seconds
        return changed or due

    def mark_published(self, items: list[dict[str, Any]], *, now: float | None = None) -> None:
        self._last_identity = _identity(items)
        self._last_publish = time.monotonic() if now is None else now


class DiscordActivityPublisher:
    """Poll cheap local leases, flushing start/stop transitions within 0.5 seconds."""

    def __init__(self, adapter: Any, *, poll_seconds: float = 0.5) -> None:
        self.adapter = adapter
        self.poll_seconds = poll_seconds
        self.gate = PublishGate()
        self.task: asyncio.Task | None = None
        self._message = None
        self._failures = 0
        self._retry_not_before = 0.0

    def start(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run(), name="discord-background-activity")

    async def stop(self) -> None:
        task, self.task = self.task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            try:
                items = await asyncio.to_thread(list_all_active_work, get_hermes_home())
                now = time.monotonic()
                if now >= self._retry_not_before and self.gate.should_publish(items, now=now):
                    await self._publish(items)
                    self.gate.mark_published(items, now=now)
                    self._failures = 0
                    self._retry_not_before = 0.0
            except asyncio.CancelledError:
                raise
            except Exception:
                self._failures += 1
                self._retry_not_before = time.monotonic() + min(15.0, 2.0 ** (self._failures - 1))
                logger.warning("Discord background-work indicator update failed", exc_info=True)
            await asyncio.sleep(self.poll_seconds)

    async def _publish(self, items: list[dict[str, Any]]) -> None:
        client = self.adapter._client
        import discord

        await client.change_presence(activity=discord.Game(name=render_presence(items)))
        message = await self._dashboard_message(client)
        if message is None:
            raise RuntimeError("Discord background-work dashboard is unavailable")
        await message.edit(content=render_dashboard(items))

    def _state_path(self) -> Path:
        return get_hermes_home() / "gateway" / _STATE_FILE

    def _stored_message_id(self) -> int | None:
        try:
            return int(json.loads(self._state_path().read_text(encoding="utf-8"))["message_id"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None

    def _store_message_id(self, message_id: int) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"message_id": message_id}), encoding="utf-8")
        tmp.replace(path)

    async def _dashboard_message(self, client: Any):
        if self._message is not None:
            return self._message
        channel = client.get_channel(USAGE_CHANNEL_ID)
        if channel is None:
            try:
                channel = await client.fetch_channel(USAGE_CHANNEL_ID)
            except Exception:
                logger.warning("Discord #usage channel %s is unavailable", USAGE_CHANNEL_ID)
                return None
        message_id = self._stored_message_id()
        if message_id:
            try:
                self._message = await channel.fetch_message(message_id)
                return self._message
            except Exception:
                pass
        try:
            async for candidate in channel.history(limit=50):
                if getattr(candidate, "author", None) == client.user and _MARKER in str(candidate.content):
                    self._message = candidate
                    self._store_message_id(candidate.id)
                    return candidate
        except Exception:
            logger.debug("Could not search Discord #usage history", exc_info=True)
        self._message = await channel.send(render_dashboard([]), silent=True)
        self._store_message_id(self._message.id)
        try:
            await self._message.pin(reason="Hermes live background-work dashboard")
        except Exception:
            logger.debug("Could not pin Discord background-work dashboard", exc_info=True)
        return self._message
