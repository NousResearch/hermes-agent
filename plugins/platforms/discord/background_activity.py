"""Discord presence and single-message dashboard for background AI work.

Presence is always published. The pinned dashboard message is strictly opt-in: it is
created only when an operator configures a channel, so a default install never posts
anything. Configure with ``discord.background_activity_channel_id`` in ``config.yaml``
(or the ``DISCORD_BACKGROUND_ACTIVITY_CHANNEL`` env bridge); the value is a channel id
(``123456789012345678``, ``<#123456789012345678>`` and a bare ID are all accepted).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_cli.background_activity import list_all_active_work
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
_CONFIG_KEY = "background_activity_channel_id"
_ENV_KEY = "DISCORD_BACKGROUND_ACTIVITY_CHANNEL"
# Distinguishes "resolve from config" from an explicit ``None`` (presence-only).
_UNSET = object()
_MARKER = "`Hermes background work`"
_STATE_FILE = "discord_background_activity.json"


def parse_channel_id(raw: Any) -> int | None:
    """Return a positive integer channel id, or None for an absent / malformed value.

    Accepts a bare snowflake, a numeric string, or a ``<#id>`` mention. Anything else
    (empty, ``"usage"``, a channel *name*, a negative number) is treated as unset so a
    typo disables the dashboard instead of breaking the adapter.
    """
    if raw is None or isinstance(raw, bool):
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("<#") and text.endswith(">"):
        text = text[2:-1]
    try:
        value = int(text)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def configured_channel_id(adapter: Any) -> int | None:
    """Resolve the dashboard channel id from this adapter's ``extra``, then scoped env."""
    extra = getattr(getattr(adapter, "config", None), "extra", None)
    if isinstance(extra, dict) and extra.get(_CONFIG_KEY) not in (None, ""):
        parsed = parse_channel_id(extra.get(_CONFIG_KEY))
        if parsed is not None:
            return parsed
    try:
        from gateway.platforms._shared import platform_gate_env
        return parse_channel_id(platform_gate_env(_ENV_KEY, ""))
    except Exception:
        return parse_channel_id(os.getenv(_ENV_KEY, ""))


def dashboard_channel_id_from_adapter(adapter: Any) -> int | None:
    """Public alias used by adapters/tests for the resolved dashboard channel id."""
    return configured_channel_id(adapter)


def render_presence(items: list[dict[str, Any]]) -> str:
    if not items:
        return "Idle"
    first = str(items[0].get("title") or "Background worker")
    prefix = f"{len(items)} worker{'s' if len(items) != 1 else ''} · "
    return (prefix + first)[:96]


def render_worker_row(item: dict[str, Any], *, now: float | None = None) -> str:
    """One compact, sanitized row per active worker (never prompts or paths)."""
    stamp = int(time.time() if now is None else now)
    fields = [
        f"worker={item.get('worker') or 'worker'}",
        f"profile={item.get('profile') or 'default'}",
    ]
    model = str(item.get("model") or "")
    if model:
        fields.append(f"model={model}")
    provider = str(item.get("provider") or "")
    if provider:
        fields.append(f"provider={provider}")
    fields.append(f"state={item.get('state') or 'running'}")
    elapsed = str(item.get("elapsed") or "0s")
    started = int(float(item.get("started_at") or stamp))
    title = item.get("title") or "Background worker"
    return f"• **{title}** — {elapsed}\n  {' · '.join(fields)} · started <t:{started}:T>"


def _sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (float(item.get("started_at") or 0), str(item.get("key") or "")),
    )


def render_dashboard(items: list[dict[str, Any]], *, now: float | None = None) -> str:
    stamp = int(time.time() if now is None else now)
    if not items:
        return f"{_MARKER}\n🟢 **Idle**\nNo background AI workers are active.\n<t:{stamp}:R>"
    ordered = _sorted_items(items)
    count = len(ordered)
    lines = [
        _MARKER,
        f"🟠 **{count} active worker{'s' if count != 1 else ''}**",
    ]
    for item in ordered[:8]:
        lines.append(render_worker_row(item, now=stamp))
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
    """Poll cheap local leases, flushing start/stop transitions within 0.5 seconds.

    Presence is unconditional. The dashboard message is published only when
    ``channel_id`` is configured; an unconfigured or inaccessible channel leaves
    presence working and never raises out of the adapter's event loop.
    """

    def __init__(
        self,
        adapter: Any,
        *,
        poll_seconds: float = 0.5,
        channel_id: Any = _UNSET,
    ) -> None:
        self.adapter = adapter
        self.poll_seconds = poll_seconds
        self.channel_id = (
            configured_channel_id(adapter) if channel_id is _UNSET else channel_id
        )
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
        if self.channel_id is None:
            return
        message = await self._dashboard_message(client)
        if message is None:
            raise RuntimeError("Discord background-work dashboard is unavailable")
        await message.edit(content=render_dashboard(items))

    def _state_path(self) -> Path:
        return get_hermes_home() / "gateway" / _STATE_FILE

    def _stored_message_id(self) -> int | None:
        """A stored message id is honored only when it belongs to the configured channel."""
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if self.channel_id is not None and data.get("channel_id") not in (None, self.channel_id):
            return None
        try:
            return int(data["message_id"])
        except (KeyError, TypeError, ValueError):
            return None

    def _store_message_id(self, message_id: int) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        payload = {"message_id": message_id, "channel_id": self.channel_id}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    async def _dashboard_message(self, client: Any):
        if self._message is not None:
            return self._message
        channel = client.get_channel(self.channel_id)
        if channel is None:
            try:
                channel = await client.fetch_channel(self.channel_id)
            except Exception:
                logger.warning(
                    "Discord background-work channel %s is unavailable; presence stays active",
                    self.channel_id,
                )
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
            logger.debug("Could not search Discord background-work channel history", exc_info=True)
        self._message = await channel.send(render_dashboard([]), silent=True)
        self._store_message_id(self._message.id)
        try:
            await self._message.pin(reason="Hermes live background-work dashboard")
        except Exception:
            logger.debug("Could not pin Discord background-work dashboard", exc_info=True)
        return self._message
