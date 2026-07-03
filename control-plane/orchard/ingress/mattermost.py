"""Mattermost ingress — the SINGLE bot connection for the whole company.

One bot account, one WebSocket. Every employee DMs this bot; the router maps
each sender to their isolated worker. Workers never touch Mattermost.

Mirrors the wire shape of Hermes' own Mattermost adapter (v4 REST + WS), kept
minimal. Untested against a live server in this scaffold — verify auth/event
field names against your Mattermost version before production.
"""
from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

from .base import Handler, Ingress
from ..models import InboundMessage

log = logging.getLogger("orchard.mattermost")

_RECONNECT_MAX = 60.0


class MattermostIngress(Ingress):
    def __init__(self, base_url: str, token: str):
        if not base_url or not token:
            raise ValueError("Mattermost URL and TOKEN are required")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._bot_user_id: str | None = None
        self._session: aiohttp.ClientSession | None = None

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    async def run(self, handler: Handler) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers())
        try:
            me = await self._get("/api/v4/users/me")
            self._bot_user_id = me["id"]
            log.info("connected as bot user %s", self._bot_user_id)
            delay = 2.0
            while True:
                try:
                    await self._listen(handler)
                    delay = 2.0
                except Exception:
                    log.exception("websocket loop failed; reconnecting in %.0fs", delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, _RECONNECT_MAX)
        finally:
            await self._session.close()

    async def _listen(self, handler: Handler) -> None:
        ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        async with self._session.ws_connect(f"{ws_url}/api/v4/websocket", heartbeat=30) as ws:
            await ws.send_json(
                {"seq": 1, "action": "authentication_challenge", "data": {"token": self.token}}
            )
            async for raw in ws:
                if raw.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(raw.data)
                if event.get("event") != "posted":
                    continue
                msg = self._parse_post(event)
                if msg:
                    await handler(msg)

    def _parse_post(self, event: dict) -> InboundMessage | None:
        data = event.get("data", {})
        try:
            post = json.loads(data["post"])
        except (KeyError, json.JSONDecodeError):
            return None
        sender = post.get("user_id")
        if not sender or sender == self._bot_user_id:
            return None  # ignore our own messages
        channel_type = data.get("channel_type")  # "D" = direct message
        text = post.get("message", "")
        # DMs always; channels only when the bot is mentioned.
        if channel_type != "D" and (self._bot_user_id or "") not in text:
            return None
        return InboundMessage(
            sender_id=sender,
            channel_id=post["channel_id"],
            text=text,
            thread_id=post.get("root_id") or None,
            sender_name=data.get("sender_name"),   # e.g. "@ivanov"
        )

    async def post(self, channel_id: str, text: str, thread_id: str | None = None) -> None:
        body = {"channel_id": channel_id, "message": text}
        if thread_id:
            body["root_id"] = thread_id
        await self._post("/api/v4/posts", body)

    async def typing(self, channel_id: str) -> None:
        # Best-effort; not fatal if it fails.
        try:
            await self._post(f"/api/v4/channels/{channel_id}/typing", {})
        except Exception:
            pass

    # --- REST helpers --------------------------------------------------------
    async def _get(self, path: str) -> dict:
        async with self._session.get(self.base_url + path) as r:
            r.raise_for_status()
            return await r.json()

    async def _post(self, path: str, body: dict) -> dict:
        async with self._session.post(self.base_url + path, json=body) as r:
            r.raise_for_status()
            return await r.json() if r.content_type == "application/json" else {}
