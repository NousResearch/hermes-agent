"""Discord stand-in: REST API v10 + Gateway websocket (https://discord.com/developers/docs).

The real ``plugins/platforms/discord`` adapter reaches it through discord.py itself: the child
gateway gets ``tests/fakes/platforms/discord_shim`` on ``PYTHONPATH`` whose ``sitecustomize``
repoints ``Route.BASE`` at ``{base}/api/v10`` and ``DiscordWebSocket.DEFAULT_GATEWAY`` at
``ws://{host}/gw``. Every request then crosses discord.py's real HTTP client, rate-limit handling,
gateway handshake (HELLO -> IDENTIFY -> READY -> GUILD_CREATE, heartbeat/ACK) and model parsing.

Only the routes the adapter actually calls are implemented, shaped per the API reference. Anything
else is recorded as ``unknown`` and answered ``404 {"message": "404: Not Found", "code": 0}`` (what
Discord returns for an unknown route) so a test can see it in ``describe()``.

Recorded call names: ``get_me``, ``get_app``, ``list_commands``, ``bulk_commands``,
``create_command``, ``edit_command``, ``delete_command``, ``create_message``, ``edit_message``,
``delete_message``, ``get_message``, ``history``, ``typing``, ``add_reaction``, ``remove_reaction``,
``get_channel``, ``edit_channel``, ``start_thread_from_message``, ``start_thread``,
``join_thread``, ``create_dm``, ``get_user``, ``get_member``, ``interaction_callback``, ``followup``,
``edit_followup``, ``attachment_download``, ``gw_connect``, ``gw_identify``, ``gw_resume``,
``gw_dispatch``, ``unknown``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import itertools
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import WSMsgType, web

from tests.fakes.platforms._standin import StandinServer, Visible

DISCORD_EPOCH_MS = 1420070400000
MAX_TEXT = 2000
BOT_ID = "1100000000000000001"
APP_ID = BOT_ID  # a bot's application id equals its user id for modern apps
BOT_USERNAME = "hermes-standin"
GUILD_ID = "1200000000000000001"
TOKEN = "MTEwMDAwMDAwMDAwMDAwMDAwMQ.standin.token"
ALL_PERMISSIONS = str((1 << 50) - 1)
HEARTBEAT_MS = 41250

_ROUTES: List[Tuple[str, str, str]] = [
    ("GET", r"/users/@me", "get_me"),
    ("GET", r"/oauth2/applications/@me", "get_app"),
    ("GET", r"/applications/@me", "get_app"),
    ("GET", r"/applications/(?P<app>\d+)/commands", "list_commands"),
    ("PUT", r"/applications/(?P<app>\d+)/commands", "bulk_commands"),
    ("POST", r"/applications/(?P<app>\d+)/commands", "create_command"),
    ("PATCH", r"/applications/(?P<app>\d+)/commands/(?P<cmd>\d+)", "edit_command"),
    ("DELETE", r"/applications/(?P<app>\d+)/commands/(?P<cmd>\d+)", "delete_command"),
    ("POST", r"/channels/(?P<channel_id>\d+)/messages", "create_message"),
    ("GET", r"/channels/(?P<channel_id>\d+)/messages", "history"),
    ("GET", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)", "get_message"),
    ("PATCH", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)", "edit_message"),
    ("DELETE", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)", "delete_message"),
    ("POST", r"/channels/(?P<channel_id>\d+)/typing", "typing"),
    ("PUT", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)/reactions/(?P<emoji>[^/]+)/@me",
     "add_reaction"),
    ("DELETE", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)/reactions/(?P<emoji>[^/]+)/@me",
     "remove_reaction"),
    ("POST", r"/channels/(?P<channel_id>\d+)/messages/(?P<message_id>\d+)/threads", "start_thread_from_message"),
    ("POST", r"/channels/(?P<channel_id>\d+)/threads", "start_thread"),
    ("PUT", r"/channels/(?P<channel_id>\d+)/thread-members/@me", "join_thread"),
    ("GET", r"/channels/(?P<channel_id>\d+)", "get_channel"),
    ("PATCH", r"/channels/(?P<channel_id>\d+)", "edit_channel"),
    ("POST", r"/users/@me/channels", "create_dm"),
    ("GET", r"/users/(?P<user_id>\d+)", "get_user"),
    ("GET", r"/guilds/(?P<guild_id>\d+)/members/(?P<user_id>\d+)", "get_member"),
    ("POST", r"/interactions/(?P<interaction_id>\d+)/(?P<token>[^/]+)/callback", "interaction_callback"),
    ("POST", r"/webhooks/(?P<app>\d+)/(?P<token>[^/]+)", "followup"),
    ("PATCH", r"/webhooks/(?P<app>\d+)/(?P<token>[^/]+)/messages/(?P<message_id>@original|\d+)", "edit_followup"),
    ("GET", r"/webhooks/(?P<app>\d+)/(?P<token>[^/]+)/messages/(?P<message_id>@original|\d+)", "get_followup"),
    ("DELETE", r"/webhooks/(?P<app>\d+)/(?P<token>[^/]+)/messages/(?P<message_id>@original|\d+)",
     "delete_followup"),
]
_COMPILED = [(m, re.compile(f"^{p}$"), n) for m, p, n in _ROUTES]


def _json(obj: Any, status: int = 200, headers: Optional[Dict[str, str]] = None) -> web.Response:
    """Discord answers ``Content-Type: application/json`` exactly; discord.py's ``json_or_text`` compares
    the header verbatim, so aiohttp's ``json_response`` (``; charset=utf-8``) would arrive as a str."""
    return web.Response(body=json.dumps(obj).encode(), status=status,
                        headers={"Content-Type": "application/json", **(headers or {})})


def _iso(ts: Optional[float] = None) -> str:
    return _dt.datetime.fromtimestamp(ts or time.time(), _dt.timezone.utc).isoformat()


class DiscordStandin(StandinServer):
    def __init__(self, token: str = TOKEN) -> None:
        super().__init__()
        self.token = token
        self._seq = itertools.count(1)
        self._sockets: List[Dict[str, Any]] = []
        self.users: Dict[str, Dict[str, Any]] = {BOT_ID: self._bot_user()}
        self.channels: Dict[str, Dict[str, Any]] = {}
        self.dm_of_user: Dict[str, str] = {}
        self.messages: Dict[str, Dict[str, Any]] = {}  # every message object by id (bot + inbound)
        self.commands: List[Dict[str, Any]] = []
        self.interactions: Dict[str, Dict[str, Any]] = {}  # token -> interaction payload
        self.files: Dict[str, bytes] = {}
        self._visible: Dict[Tuple[str, str], Visible] = {}
        self.guild = {"id": GUILD_ID, "name": "Standin Guild", "owner_id": "1", "channels": []}

    # ids / shapes ----------------------------------------------------------------------------
    def snowflake(self) -> str:
        ms = int(time.time() * 1000) - DISCORD_EPOCH_MS
        return str((ms << 22) | (next(self._seq) & 0x3FFFFF))

    @staticmethod
    def _bot_user() -> Dict[str, Any]:
        return {"id": BOT_ID, "username": BOT_USERNAME, "discriminator": "0", "global_name": "Hermes",
                "avatar": None, "bot": True, "flags": 0, "public_flags": 0, "verified": True,
                "mfa_enabled": False}

    def user(self, user_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        uid = str(user_id)
        if uid not in self.users:
            uname = name or f"user{uid[-4:]}"
            self.users[uid] = {"id": uid, "username": uname, "discriminator": "0", "global_name": uname.title(),
                               "avatar": None, "bot": False, "public_flags": 0}
        return self.users[uid]

    def member(self, user_id: str, *, with_user: bool = True) -> Dict[str, Any]:
        m = {"roles": [], "nick": None, "avatar": None, "joined_at": _iso(1700000000), "premium_since": None,
             "deaf": False, "mute": False, "flags": 0, "pending": False}
        if with_user:
            m["user"] = self.user(user_id)
        return m

    def add_text_channel(self, channel_id: str, name: str) -> Dict[str, Any]:
        ch = {"id": str(channel_id), "type": 0, "guild_id": GUILD_ID, "name": name, "position": len(self.channels),
              "permission_overwrites": [], "nsfw": False, "parent_id": None, "topic": None,
              "last_message_id": None, "rate_limit_per_user": 0, "flags": 0}
        self.channels[str(channel_id)] = ch
        return ch

    def dm_channel(self, user_id: str) -> Dict[str, Any]:
        uid = str(user_id)
        if uid not in self.dm_of_user:
            cid = self.snowflake()
            self.dm_of_user[uid] = cid
            self.channels[cid] = {"id": cid, "type": 1, "last_message_id": None, "flags": 0,
                                  "recipients": [self.user(uid)]}
        return self.channels[self.dm_of_user[uid]]

    def _guild_payload(self) -> Dict[str, Any]:
        everyone = {"id": GUILD_ID, "name": "@everyone", "color": 0, "hoist": False, "icon": None,
                    "unicode_emoji": None, "position": 0, "permissions": ALL_PERMISSIONS, "managed": False,
                    "mentionable": False, "flags": 0}
        members = [self.member(uid) for uid, u in self.users.items() if u.get("_in_guild") or uid == BOT_ID]
        return {**self.guild, "icon": None, "splash": None, "discovery_splash": None, "afk_channel_id": None,
                "afk_timeout": 300, "verification_level": 0, "default_message_notifications": 0,
                "explicit_content_filter": 0, "roles": [everyone], "emojis": [], "stickers": [], "features": [],
                "mfa_level": 0, "application_id": None, "system_channel_id": None, "system_channel_flags": 0,
                "rules_channel_id": None, "max_members": 500000, "vanity_url_code": None, "description": None,
                "banner": None, "premium_tier": 0, "premium_subscription_count": 0, "preferred_locale": "en-US",
                "public_updates_channel_id": None, "nsfw_level": 0, "premium_progress_bar_enabled": False,
                "joined_at": _iso(1700000000), "large": False, "unavailable": False,
                "member_count": len(members), "voice_states": [], "members": members,
                "channels": [dict(c) for c in self.channels.values() if c.get("guild_id") == GUILD_ID],
                "threads": [], "presences": [], "stage_instances": [], "guild_scheduled_events": [],
                "soundboard_sounds": []}

    # server ----------------------------------------------------------------------------------
    @property
    def api_base(self) -> str:
        return f"{self.base_url}/api/v10"

    @property
    def gateway_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/gw"

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_get("/gw", self._gateway)
        app.router.add_get("/gw/", self._gateway)
        app.router.add_get("/attachments/{att_id}/{filename}", self._attachment)
        app.router.add_route("*", "/api/v10/{path:.*}", self._api)
        return app

    async def on_shutdown(self) -> None:
        for sock in list(self._sockets):
            try:
                await sock["ws"].close(code=1001)
            except Exception:
                pass

    async def _attachment(self, request: web.Request) -> web.Response:
        att_id = request.match_info["att_id"]
        self.record("attachment_download", {"id": att_id, "filename": request.match_info["filename"]}, None)
        data = self.files.get(att_id)
        return web.Response(body=data) if data is not None else web.Response(status=404)

    async def _params(self, request: web.Request) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(request.query)
        ctype = request.content_type or ""
        if ctype == "application/json":
            raw = await request.read()
            if raw:
                body = json.loads(raw)
                params.update(body if isinstance(body, dict) else {"_body": body})
        elif ctype.startswith("multipart/"):
            files: List[Dict[str, Any]] = []
            reader = await request.multipart()
            async for part in reader:
                if part.filename:
                    data = await part.read()
                    files.append({"name": part.name, "filename": part.filename, "size": len(data)})
                elif part.name == "payload_json":
                    params.update(json.loads(await part.text()))
                else:
                    params[part.name] = await part.text()
            params["_files"] = files
        return params

    def _route(self, method: str, path: str) -> Tuple[str, Dict[str, str]]:
        for m, rx, name in _COMPILED:
            if m == method:
                hit = rx.match(path)
                if hit:
                    return name, hit.groupdict()
        return "unknown", {}

    async def _api(self, request: web.Request) -> web.Response:
        path = "/" + request.match_info["path"].rstrip("/")
        name, args = self._route(request.method, path)
        params = {**await self._params(request), **args}
        if not path.startswith(("/interactions/", "/webhooks/")) and \
                request.headers.get("Authorization") != f"Bot {self.token}":
            body = {"message": "401: Unauthorized", "code": 0}
            self.record(name, params, body, faulted=True)
            return _json(body, status=401)
        if name == "unknown":
            body = {"message": "404: Not Found", "code": 0}
            self.record(name, {**params, "_method": request.method, "_path": path}, body, faulted=True)
            return _json(body, status=404)
        fault = self.take_fault(name, params)
        if fault is not None:
            self.record(name, params, fault.body, faulted=True)
            headers = {"Retry-After": str(fault.body.get("retry_after", 1)), "X-RateLimit-Scope": "user"} \
                if fault.status == 429 and isinstance(fault.body, dict) else None
            return _json(fault.body, status=fault.status, headers=headers)
        try:
            status, result = getattr(self, f"_r_{name}")(params)
        except Exception as exc:  # a stand-in bug must be loud in describe(), not a silent 5xx retry loop
            body = {"message": f"stand-in handler error: {exc!r}", "code": 0}
            self.record(name, {**params, "_error": repr(exc)}, body, faulted=True)
            return _json(body, status=500)
        self.record(name, params, result, faulted=status >= 400)
        if status == 204:
            return web.Response(status=204)
        return _json(result, status=status)

    # REST handlers: each returns (status, json) -------------------------------------------------
    def _r_get_me(self, _p: Dict[str, Any]) -> Tuple[int, Any]:
        return 200, self._bot_user()

    def _r_get_app(self, _p: Dict[str, Any]) -> Tuple[int, Any]:
        owner = {"id": "1", "username": "owner", "discriminator": "0", "global_name": None, "avatar": None}
        return 200, {"id": APP_ID, "name": "Hermes Standin", "icon": None, "description": "", "summary": "",
                     "type": None, "bot_public": True, "bot_require_code_grant": False, "verify_key": "0" * 64,
                     "flags": (1 << 19) | (1 << 15), "owner": owner, "team": None, "bot": self._bot_user(),
                     "interactions_endpoint_url": None, "redirect_uris": [], "tags": [],
                     "approximate_guild_count": 1, "integration_types_config": {"0": {}, "1": {}}}

    def _command(self, body: Dict[str, Any]) -> Dict[str, Any]:
        cmd = {k: v for k, v in body.items() if not k.startswith("_") and k not in ("app",)}
        cmd.setdefault("id", self.snowflake())
        cmd.update(application_id=APP_ID, version=self.snowflake(), type=cmd.get("type", 1))
        cmd.setdefault("default_member_permissions", None)
        cmd.setdefault("dm_permission", True)
        cmd.setdefault("nsfw", False)
        cmd.setdefault("description", "")
        return cmd

    def _r_list_commands(self, _p: Dict[str, Any]) -> Tuple[int, Any]:
        return 200, list(self.commands)

    def _r_bulk_commands(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        self.commands = [self._command(c) for c in (p.get("_body") or [])]
        return 200, list(self.commands)

    def _r_create_command(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        cmd = self._command(p)
        self.commands = [c for c in self.commands if c.get("name") != cmd.get("name")] + [cmd]
        return 201, cmd

    def _r_edit_command(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        for c in self.commands:
            if c["id"] == p["cmd"]:
                c.update({k: v for k, v in p.items() if not k.startswith("_") and k not in ("app", "cmd")})
                return 200, c
        return 404, {"message": "Unknown application command", "code": 10063}

    def _r_delete_command(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        self.commands = [c for c in self.commands if c["id"] != p["cmd"]]
        return 204, None

    def _bot_message(self, channel_id: str, p: Dict[str, Any], *, interaction: Optional[Dict[str, Any]] = None,
                     ephemeral: bool = False) -> Dict[str, Any]:
        mid = self.snowflake()
        ch = self.channels.get(channel_id) or {}
        attachments = [{"id": self.snowflake(), "filename": f["filename"], "size": f["size"],
                        "url": f"{self.base_url}/attachments/out/{f['filename']}",
                        "proxy_url": f"{self.base_url}/attachments/out/{f['filename']}"}
                       for f in p.get("_files") or []]
        msg = {"id": mid, "channel_id": channel_id, "author": self._bot_user(), "content": p.get("content") or "",
               "timestamp": _iso(), "edited_timestamp": None, "tts": False, "mention_everyone": False,
               "mentions": [], "mention_roles": [], "attachments": attachments, "embeds": p.get("embeds") or [],
               "components": p.get("components") or [], "pinned": False, "type": 0 if not p.get(
                   "message_reference") else 19, "flags": (64 if ephemeral else 0)}
        if p.get("message_reference"):
            msg["message_reference"] = p["message_reference"]
        if p.get("nonce") is not None:
            msg["nonce"] = p["nonce"]
        if ch.get("guild_id"):
            msg["guild_id"] = ch["guild_id"]
        if interaction is not None:
            msg["interaction_metadata"] = {"id": interaction["id"], "type": interaction["type"],
                                           "user": interaction["_user"]}
            msg["webhook_id"] = APP_ID
        self.messages[mid] = msg
        with self._lock:
            self._visible[(channel_id, mid)] = Visible(mid, msg["content"], extra={
                "components": msg["components"], "embeds": msg["embeds"], "attachments": attachments,
                "reply_to": (p.get("message_reference") or {}).get("message_id"), "ephemeral": ephemeral})
        if ch:
            ch["last_message_id"] = mid
        return msg

    def _unknown_channel(self) -> Tuple[int, Any]:
        return 404, {"message": "Unknown Channel", "code": 10003}

    def _r_create_message(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        cid = p["channel_id"]
        if cid not in self.channels:
            return self._unknown_channel()
        if len(p.get("content") or "") > MAX_TEXT:
            return 400, {"message": "Invalid Form Body", "code": 50035, "errors": {"content": {"_errors": [
                {"code": "BASE_TYPE_MAX_LENGTH", "message": "Must be 2000 or fewer in length."}]}}}
        if not str(p.get("content") or "").strip() and not any(
                p.get(k) for k in ("embeds", "components", "_files", "sticker_ids", "poll", "attachments")):
            return 400, {"message": "Cannot send an empty message", "code": 50006}
        return 200, self._bot_message(cid, p)

    def _edit(self, cid: str, mid: str, p: Dict[str, Any]) -> Tuple[int, Any]:
        msg = self.messages.get(mid)
        if msg is None or msg["channel_id"] != cid:
            return 404, {"message": "Unknown Message", "code": 10008}
        if msg["author"]["id"] != BOT_ID:
            return 403, {"message": "Cannot edit a message authored by another user", "code": 50005}
        if len(p.get("content") or "") > MAX_TEXT:
            return 400, {"message": "Invalid Form Body", "code": 50035}
        for key in ("content", "embeds", "components"):
            if key in p:
                msg[key] = p[key] if p[key] is not None else ([] if key != "content" else "")
        msg["edited_timestamp"] = _iso()
        with self._lock:
            vis = self._visible.get((cid, mid))
            if vis is not None:
                vis.text = msg["content"]
                vis.edits += 1
                vis.extra["components"] = msg["components"]
                vis.extra["embeds"] = msg["embeds"]
        return 200, msg

    def _r_edit_message(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        return self._edit(p["channel_id"], p["message_id"], p)

    def _r_delete_message(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        msg = self.messages.get(p["message_id"])
        if msg is None:
            return 404, {"message": "Unknown Message", "code": 10008}
        with self._lock:
            vis = self._visible.get((p["channel_id"], p["message_id"]))
            if vis is not None:
                vis.deleted = True
        return 204, None

    def _r_get_message(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        msg = self.messages.get(p["message_id"])
        if msg is None or msg["channel_id"] != p["channel_id"]:
            return 404, {"message": "Unknown Message", "code": 10008}
        return 200, msg

    def _r_history(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        rows = [m for m in self.messages.values() if m["channel_id"] == p["channel_id"]]
        rows.sort(key=lambda m: int(m["id"]), reverse=True)
        if p.get("after"):
            rows = [m for m in rows if int(m["id"]) > int(p["after"])]
        if p.get("before"):
            rows = [m for m in rows if int(m["id"]) < int(p["before"])]
        return 200, rows[: int(p.get("limit") or 50)]

    def _r_typing(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        return (204, None) if p["channel_id"] in self.channels else self._unknown_channel()

    def _r_add_reaction(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        return 204, None

    _r_remove_reaction = _r_add_reaction

    def _r_get_channel(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        ch = self.channels.get(p["channel_id"])
        return (200, ch) if ch else self._unknown_channel()

    def _make_thread(self, parent_id: str, thread_id: str, p: Dict[str, Any]) -> Dict[str, Any]:
        parent = self.channels[parent_id]
        now = _iso()
        th = {"id": thread_id, "type": int(p.get("type") or 11), "guild_id": parent.get("guild_id"),
              "parent_id": parent_id, "name": p.get("name") or "thread", "owner_id": BOT_ID, "message_count": 0,
              "member_count": 1, "total_message_sent": 0, "rate_limit_per_user": int(p.get("rate_limit_per_user") or 0),
              "last_message_id": None, "flags": 0, "applied_tags": [],
              "thread_metadata": {"archived": False, "auto_archive_duration": int(p.get("auto_archive_duration") or 1440),
                                  "archive_timestamp": now, "locked": False, "create_timestamp": now},
              "member": {"id": thread_id, "user_id": BOT_ID, "join_timestamp": now, "flags": 0}}
        self.channels[thread_id] = th
        if self._loop is not None:  # Discord announces the new thread on the gateway too
            asyncio.ensure_future(self._broadcast("THREAD_CREATE", {**th, "newly_created": True}), loop=self._loop)
        return th

    def _r_start_thread_from_message(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        msg = self.messages.get(p["message_id"])
        if p["channel_id"] not in self.channels or msg is None:
            return 404, {"message": "Unknown Message", "code": 10008}
        if p["message_id"] in self.channels:
            return 400, {"message": "A thread has already been created for this message", "code": 160004}
        return 201, self._make_thread(p["channel_id"], p["message_id"], p)

    def _r_start_thread(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        if p["channel_id"] not in self.channels:
            return self._unknown_channel()
        th = self._make_thread(p["channel_id"], self.snowflake(), p)
        if isinstance(p.get("message"), dict):  # forum/media post: starter message id == thread id
            starter = self._bot_message(th["id"], {**p["message"], "_files": p.get("_files")})
            return 201, {**th, "message": starter}
        return 201, th

    def _r_join_thread(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        return (204, None) if p["channel_id"] in self.channels else self._unknown_channel()

    def threads_of(self, channel_id: Any) -> List[Dict[str, Any]]:
        return [c for c in self.channels.values() if c.get("parent_id") == str(channel_id)]

    def _r_edit_channel(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        ch = self.channels.get(p["channel_id"])
        if ch is None:
            return self._unknown_channel()
        for key in ("name", "topic", "rate_limit_per_user", "nsfw", "applied_tags"):
            if key in p:
                ch[key] = p[key]
        meta = ch.get("thread_metadata")
        if meta is not None:
            for key in ("archived", "locked", "auto_archive_duration"):
                if key in p:
                    meta[key] = p[key]
        if self._loop is not None:
            event = "THREAD_UPDATE" if meta is not None else "CHANNEL_UPDATE"
            asyncio.ensure_future(self._broadcast(event, dict(ch)), loop=self._loop)
        return 200, ch

    def _r_create_dm(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        return 200, self.dm_channel(str(p["recipient_id"]))

    def _r_get_user(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        u = self.users.get(p["user_id"])
        return (200, u) if u else (404, {"message": "Unknown User", "code": 10013})

    def _r_get_member(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        u = self.users.get(p["user_id"])
        return (200, self.member(p["user_id"])) if u else (404, {"message": "Unknown Member", "code": 10007})

    def _r_interaction_callback(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        inter = self.interactions.get(p["token"])
        if inter is None or inter["id"] != p["interaction_id"]:
            return 404, {"message": "Unknown interaction", "code": 10062}
        if inter.get("_answered"):
            return 400, {"message": "Interaction has already been acknowledged.", "code": 40060}
        inter["_answered"] = True
        rtype, data = int(p.get("type") or 0), (p.get("data") or {})
        ephemeral = bool(int(data.get("flags") or 0) & 64)
        cb: Dict[str, Any] = {"interaction": {"id": inter["id"], "type": inter["type"],
                                              "response_message_loading": rtype == 5,
                                              "response_message_ephemeral": ephemeral}}
        msg = None
        if rtype == 4:  # CHANNEL_MESSAGE_WITH_SOURCE
            msg = self._bot_message(inter["channel_id"], {**data, "_files": p.get("_files")}, interaction=inter,
                                    ephemeral=ephemeral)
            inter["_original"] = msg["id"]
        elif rtype == 7 and inter.get("message"):  # UPDATE_MESSAGE on the component's message
            _, msg = self._edit(inter["channel_id"], inter["message"]["id"], data)
        elif rtype == 5:  # deferred: a loading placeholder becomes @original
            msg = self._bot_message(inter["channel_id"], {"content": ""}, interaction=inter, ephemeral=ephemeral)
            inter["_original"] = msg["id"]
        if msg is not None:
            cb["interaction"]["response_message_id"] = msg["id"]
            cb["resource"] = {"type": rtype, "message": msg}
        elif rtype:
            cb["resource"] = {"type": rtype}
        return 200, cb

    def _r_followup(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        inter = self.interactions.get(p["token"])
        if inter is None:
            return 404, {"message": "Unknown Webhook", "code": 10015}
        ephemeral = bool(int(p.get("flags") or 0) & 64)
        return 200, self._bot_message(inter["channel_id"], p, interaction=inter, ephemeral=ephemeral)

    def _followup_target(self, p: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        inter = self.interactions.get(p["token"])
        if inter is None:
            return None
        mid = inter.get("_original") if p["message_id"] == "@original" else p["message_id"]
        if mid is None and inter.get("message"):
            mid = inter["message"]["id"]
        return (inter["channel_id"], mid) if mid else None

    def _r_edit_followup(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        target = self._followup_target(p)
        if target is None:
            return 404, {"message": "Unknown Message", "code": 10008}
        return self._edit(target[0], target[1], p)

    def _r_get_followup(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        target = self._followup_target(p)
        msg = self.messages.get(target[1]) if target else None
        return (200, msg) if msg else (404, {"message": "Unknown Message", "code": 10008})

    def _r_delete_followup(self, p: Dict[str, Any]) -> Tuple[int, Any]:
        target = self._followup_target(p)
        if target is None:
            return 404, {"message": "Unknown Message", "code": 10008}
        return self._r_delete_message({"channel_id": target[0], "message_id": target[1]})

    # gateway ---------------------------------------------------------------------------------
    async def _gateway(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=True, max_msg_size=0)
        await ws.prepare(request)
        sock: Dict[str, Any] = {"ws": ws, "seq": 0, "ready": False, "session_id": None,
                                "query": dict(request.query)}
        self.record("gw_connect", dict(request.query), None)
        await ws.send_str(json.dumps({"op": 10, "d": {"heartbeat_interval": HEARTBEAT_MS}, "s": None, "t": None}))
        try:
            async for frame in ws:
                if frame.type != WSMsgType.TEXT:
                    if frame.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                        break
                    continue
                await self._on_gateway_frame(sock, json.loads(frame.data))
        finally:
            if sock in self._sockets:
                self._sockets.remove(sock)
        return ws

    async def _on_gateway_frame(self, sock: Dict[str, Any], msg: Dict[str, Any]) -> None:
        op, d = msg.get("op"), msg.get("d")
        ws = sock["ws"]
        if op == 1:  # HEARTBEAT
            await ws.send_str(json.dumps({"op": 11, "d": None, "s": None, "t": None}))
        elif op == 2:  # IDENTIFY
            self.record("gw_identify", {"intents": d.get("intents"), "shard": d.get("shard"),
                                        "token_ok": d.get("token") == self.token}, None)
            if d.get("token") != self.token:
                await ws.close(code=4004, message=b"Authentication failed.")
                return
            sock["session_id"] = f"standin-{self.snowflake()}"
            sock["intents"] = d.get("intents")
            await self._send_dispatch(sock, "READY", {
                "v": 10, "user": self._bot_user(), "guilds": [{"id": GUILD_ID, "unavailable": True}],
                "session_id": sock["session_id"], "session_type": "normal",
                "resume_gateway_url": self.gateway_url, "shard": d.get("shard") or [0, 1],
                "application": {"id": APP_ID, "flags": (1 << 19) | (1 << 15)}, "private_channels": [],
                "relationships": [], "presences": [], "user_settings": {}, "guild_join_requests": [],
                "geo_ordered_rtc_regions": [], "auth": {}, "_trace": ["standin"]})
            await self._send_dispatch(sock, "GUILD_CREATE", self._guild_payload())
            sock["ready"] = True
            self._sockets.append(sock)
        elif op == 6:  # RESUME: no event buffer is kept, so ask for a fresh IDENTIFY (resumable=false)
            self.record("gw_resume", {"session_id": (d or {}).get("session_id"), "seq": (d or {}).get("seq")}, None)
            await ws.send_str(json.dumps({"op": 9, "d": False, "s": None, "t": None}))
        # op 3 presence / op 4 voice state / op 8 request members: accepted silently

    async def _send_dispatch(self, sock: Dict[str, Any], event: str, data: Dict[str, Any]) -> None:
        sock["seq"] += 1
        await sock["ws"].send_str(json.dumps({"op": 0, "t": event, "s": sock["seq"], "d": data}))

    async def _broadcast(self, event: str, data: Dict[str, Any]) -> int:
        sent = 0
        for sock in list(self._sockets):
            if sock["ready"] and not sock["ws"].closed:
                await self._send_dispatch(sock, event, data)
                sent += 1
        return sent

    def dispatch(self, event: str, data: Dict[str, Any]) -> int:
        """Send one DISPATCH (op 0) to every identified gateway session; returns sessions reached."""
        sent = self.run_in_loop(self._broadcast(event, data))
        self.record("gw_dispatch", {"t": event, "id": data.get("id"), "channel_id": data.get("channel_id"),
                                    "sessions": sent}, None)
        return sent

    def sessions(self) -> int:
        return sum(1 for s in self._sockets if s["ready"] and not s["ws"].closed)

    def drop_sessions(self, code: int = 4000) -> None:
        """Close every gateway socket (a Discord-side disconnect; the client reconnects/resumes)."""

        async def _drop() -> None:
            for sock in list(self._sockets):
                await sock["ws"].close(code=code)

        self.run_in_loop(_drop())

    # test-facing inbound ---------------------------------------------------------------------
    def inbound_message(self, channel_id: str, user_id: str, content: str, *, mentions: Optional[List[str]] = None,
                        attachments: Optional[List[Dict[str, Any]]] = None, dispatch: bool = True) -> Dict[str, Any]:
        ch = self.channels[str(channel_id)]
        author = self.user(user_id)
        mid = self.snowflake()
        msg: Dict[str, Any] = {"id": mid, "channel_id": ch["id"], "author": author, "content": content,
                               "timestamp": _iso(), "edited_timestamp": None, "tts": False,
                               "mention_everyone": False, "mention_roles": [], "attachments": attachments or [],
                               "embeds": [], "components": [], "pinned": False, "type": 0, "flags": 0,
                               "nonce": mid, "mentions": []}
        for uid in mentions or []:
            mu = dict(self.users.get(uid) or self.user(uid))
            if ch.get("guild_id"):
                mu["member"] = self.member(uid, with_user=False)
            msg["mentions"].append(mu)
        if ch.get("guild_id"):
            author["_in_guild"] = True
            msg["guild_id"] = ch["guild_id"]
            msg["member"] = self.member(user_id, with_user=False)
        payload = {k: v for k, v in msg.items()}
        payload["author"] = {k: v for k, v in author.items() if not k.startswith("_")}
        payload["mentions"] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in msg["mentions"]]
        self.messages[mid] = payload
        ch["last_message_id"] = mid
        if dispatch:
            self.dispatch("MESSAGE_CREATE", payload)
        return payload

    def attachment(self, filename: str, data: bytes, mime: str) -> Dict[str, Any]:
        att_id = self.snowflake()
        self.files[att_id] = data
        url = f"{self.base_url}/attachments/{att_id}/{filename}"
        return {"id": att_id, "filename": filename, "size": len(data), "url": url, "proxy_url": url,
                "content_type": mime}

    def click(self, user_id: str, channel_id: str, message_id: str, custom_id: str,
              component_type: int = 2, values: Optional[List[str]] = None) -> Dict[str, Any]:
        """A user presses a component (button/select) on bot message ``message_id``: INTERACTION_CREATE."""
        ch = self.channels[str(channel_id)]
        iid, token = self.snowflake(), f"itoken-{self.snowflake()}"
        data: Dict[str, Any] = {"custom_id": custom_id, "component_type": component_type}
        if values is not None:
            data["values"] = values
        inter: Dict[str, Any] = {
            "id": iid, "application_id": APP_ID, "type": 3, "data": data, "channel_id": ch["id"],
            "channel": {k: v for k, v in ch.items() if k != "recipients"} | {"type": ch["type"]},
            "token": token, "version": 1, "message": self.messages.get(str(message_id)),
            "app_permissions": ALL_PERMISSIONS, "locale": "en-US", "entitlements": [],
            "authorizing_integration_owners": {"0": GUILD_ID} if ch.get("guild_id") else {"1": str(user_id)},
            "context": 0 if ch.get("guild_id") else 1, "attachment_size_limit": 26214400,
        }
        user = {k: v for k, v in self.user(user_id).items() if not k.startswith("_")}
        if ch.get("guild_id"):
            inter["guild_id"] = ch["guild_id"]
            inter["member"] = {**self.member(user_id, with_user=False), "user": user,
                               "permissions": ALL_PERMISSIONS}
            inter["guild_locale"] = "en-US"
        else:
            inter["user"] = user
        self.interactions[token] = {**inter, "_user": user}
        self.dispatch("INTERACTION_CREATE", inter)
        return inter

    def redeliver(self, payload: Dict[str, Any]) -> None:
        """Deliver the very same MESSAGE_CREATE again (a gateway replay of an already-seen event)."""
        self.dispatch("MESSAGE_CREATE", payload)

    # ground truth ----------------------------------------------------------------------------
    def visible(self, channel_id: Any) -> List[Visible]:
        with self._lock:
            rows = [v for (c, _), v in self._visible.items() if c == str(channel_id) and not v.deleted]
        return sorted(rows, key=lambda v: int(v.message_id))

    def buttons(self, channel_id: Any) -> List[Dict[str, Any]]:
        """Every component (button/select) currently attached to a visible bot message in the channel."""
        out: List[Dict[str, Any]] = []
        for vis in self.visible(channel_id):
            for row in vis.extra.get("components") or []:
                for comp in row.get("components") or []:
                    out.append({**comp, "message_id": vis.message_id})
        return out
