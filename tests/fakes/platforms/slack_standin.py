"""Slack Web API + Socket Mode stand-in (https://api.slack.com/methods, /apis/socket-mode).

The real ``plugins/platforms/slack`` adapter reaches it through slack_bolt/slack_sdk: the
``slack_shim/sitecustomize.py`` module (on the child's ``PYTHONPATH``) defaults every slack_sdk Web
API client's ``base_url`` to ``{base}/api/``, so ``auth.test``, ``chat.*``, ``conversations.*`` and
``apps.connections.open`` all land here. ``apps.connections.open`` hands out ``ws://…/ws``; the SDK's
Socket Mode client connects, receives ``hello``, and from then on the test pushes envelopes
(``events_api`` / ``interactive`` / ``slash_commands``) that the client acks by ``envelope_id``.

Web API semantics kept faithful to Slack: every response is HTTP 200 JSON with ``ok``; failures are
``{"ok": false, "error": "<code>"}`` (the SDK raises ``SlackApiError`` on them); bodies arrive
form-encoded or JSON (structured fields like ``blocks`` may be JSON strings); ``chat.postMessage``
above 40,000 chars answers ``msg_too_long``. Unknown methods answer ``{"ok": false, "error":
"unknown_method"}`` like Slack (recorded as faulted), so an unmodelled call is never a silent success.

Native streams: a fault's ``match`` for ``chat.appendStream``/``chat.stopStream`` also sees
``_stream_text`` (the message text as it would read after this call), so a test can reject the call
that completes a given piece of the reply however the deltas were cut.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import WSMsgType, web

from tests.fakes.platforms._standin import StandinServer, Visible

TEAM_ID = "T0STANDIN"
APP_ID = "A0STANDIN"
BOT_USER_ID = "UBOT0001"
BOT_ID = "B0STANDIN"
BOT_NAME = "hermes"
MAX_TEXT = 40_000  # chat.postMessage hard limit (msg_too_long above it)
_STRUCTURED = ("blocks", "attachments", "metadata", "files", "chunks", "file_uploads")
_SCOPES = ",".join((
    "app_mentions:read", "assistant:write", "channels:history", "channels:read", "chat:write",
    "commands", "files:read", "files:write", "groups:history", "groups:read", "im:history", "im:read",
    "im:write", "mpim:history", "mpim:read", "mpim:write", "reactions:read", "reactions:write",
    "users:read"))


def _decode(key: str, value: Any) -> Any:
    if key in _STRUCTURED and isinstance(value, str) and value.strip()[:1] in "[{":
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


class SlackStandin(StandinServer):
    def __init__(self, bot_token: str = "xoxb-standin", app_token: str = "xapp-standin") -> None:
        super().__init__()
        self.bot_token = bot_token
        self.app_token = app_token
        self._ts_base = int(time.time())
        self._ts_seq = itertools.count(100)
        self._ids = itertools.count(1)
        self._sockets: List[web.WebSocketResponse] = []
        self.acks: Dict[str, Dict[str, Any]] = {}
        self.envelopes: List[Dict[str, Any]] = []
        # (channel, ts) -> Visible for BOT messages; every message (bot + inbound) per channel for
        # conversations.history/replies.
        self._visible: Dict[tuple, Visible] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self.files: Dict[str, Dict[str, Any]] = {}
        self.uploads: Dict[str, bytes] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        # Ephemeral replies (response_url POSTs + chat.postEphemeral): seen only by one user, so kept
        # out of ``visible()``. response id -> (channel, user) for response_url routing.
        self._responses: Dict[str, tuple] = {}
        self._ephemerals: List[Dict[str, Any]] = []

    @property
    def api_base(self) -> str:
        return f"{self.base_url}/api/"

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/ws"

    def next_ts(self) -> str:
        return f"{self._ts_base}.{next(self._ts_seq):06d}"

    # server ------------------------------------------------------------------------------------
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/api/{method}", self._api)
        app.router.add_get("/ws", self._ws)
        app.router.add_post("/upload/{file_id}", self._upload)
        app.router.add_get("/files-pri/{path:.*}", self._download)
        app.router.add_post("/response/{rid}", self._response_url)
        return app

    async def _params(self, request: web.Request) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(request.query)
        ctype = request.content_type or ""
        if ctype == "application/json":
            params.update(await request.json() or {})
        elif ctype.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                if part.filename:
                    params[part.name] = {"filename": part.filename, "size": len(await part.read())}
                else:
                    params[part.name] = await part.text()
        elif request.can_read_body:
            params.update({k: v for k, v in (await request.post()).items()})
        return {k: _decode(k, v) for k, v in params.items()}

    @staticmethod
    def _token(request: web.Request, params: Dict[str, Any]) -> str:
        auth = request.headers.get("Authorization", "")
        return auth[7:] if auth.startswith("Bearer ") else str(params.pop("token", ""))

    def _reply(self, body: Dict[str, Any]) -> web.Response:
        return web.json_response(body, headers={"x-oauth-scopes": _SCOPES, "x-accepted-oauth-scopes": ""})

    async def _api(self, request: web.Request) -> web.Response:
        method = request.match_info["method"]
        params = await self._params(request)
        token = self._token(request, params)
        want = self.app_token if method == "apps.connections.open" else self.bot_token
        if token != want:
            body = {"ok": False, "error": "invalid_auth"}
            self.record(method, params, body, faulted=True)
            return self._reply(body)
        fault = self.take_fault(method, self._fault_view(method, params))
        if fault is not None:
            self.record(method, params, fault.body, faulted=True)
            return web.json_response(fault.body, status=fault.status)
        handler = getattr(self, "_m_" + method.replace(".", "_"), None)
        if handler is None:
            body = {"ok": False, "error": "unknown_method"}
            self.record(method, params, body, faulted=True)
            return self._reply(body)
        try:
            result = handler(params)
        except _ApiError as exc:
            body = {"ok": False, "error": exc.code}
            self.record(method, params, body, faulted=True)
            return self._reply(body)
        body = {"ok": True, **(result or {})}
        self.record(method, params, body)
        return self._reply(body)

    def _fault_view(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method not in ("chat.appendStream", "chat.stopStream"):
            return params
        with self._lock:
            vis = self._visible.get((str(params.get("channel", "")), str(params.get("ts", ""))))
            before = vis.text if vis is not None else ""
        return {**params, "_stream_text": before + str(params.get("markdown_text") or "")}

    async def _upload(self, request: web.Request) -> web.Response:
        fid = request.match_info["file_id"]
        data = await request.read()
        self.uploads[fid] = data
        self.record("file_upload", {"file_id": fid, "size": len(data)}, None)
        return web.Response(text=f"OK - {len(data)}")

    async def _download(self, request: web.Request) -> web.Response:
        path = request.match_info["path"]
        self.record("file_download", {"path": path}, None)
        for f in self.files.values():
            if f.get("_path") == path:
                return web.Response(body=f["_data"], content_type=f.get("mimetype", "application/octet-stream"))
        return web.Response(status=404)

    async def _response_url(self, request: web.Request) -> web.Response:
        rid = request.match_info["rid"]
        params = await self._params(request)
        target = self._responses.get(rid)
        if target is None:
            self.record("response_url", {"rid": rid, **params}, None, faulted=True)
            return web.Response(status=404, text="no_such_response_url")
        self.record("response_url", {"rid": rid, "channel": target[0], **params}, None)
        with self._lock:
            self._ephemerals.append({"channel": target[0], "user": target[1], "text": params.get("text", ""),
                                     "replace_original": params.get("replace_original"), "via": "response_url"})
        return web.Response(text="ok")

    def _new_response_url(self, channel: str, user_id: str) -> str:
        rid = uuid.uuid4().hex[:12]
        self._responses[rid] = (channel, user_id)
        return f"{self.base_url}/response/{rid}"

    def ephemerals(self, channel: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._ephemerals if e["channel"] == str(channel)]

    # Socket Mode -------------------------------------------------------------------------------
    async def _ws(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(autoping=True)
        await ws.prepare(request)
        self._sockets.append(ws)
        self.record("socket_connect", {"sockets": len(self._sockets)}, None)
        await ws.send_str(json.dumps({"type": "hello", "num_connections": len(self._sockets),
                                      "debug_info": {"host": "standin", "approximate_connection_time": 3600},
                                      "connection_info": {"app_id": APP_ID}}))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    ack = json.loads(msg.data)
                except ValueError:
                    continue
                if isinstance(ack, dict) and ack.get("envelope_id"):
                    with self._lock:
                        self.acks[ack["envelope_id"]] = ack
                    self.record("socket_ack", ack, None)
        finally:
            if ws in self._sockets:
                self._sockets.remove(ws)
            self.record("socket_close", {"sockets": len(self._sockets)}, None)
        return ws

    async def on_shutdown(self) -> None:
        for ws in list(self._sockets):
            await ws.close()

    def socket_count(self) -> int:
        return len([ws for ws in self._sockets if not ws.closed])

    async def _asend(self, envelope: Dict[str, Any]) -> None:
        live = [ws for ws in self._sockets if not ws.closed]
        if not live:
            raise RuntimeError("no Socket Mode client connected to the Slack stand-in")
        # Slack delivers one envelope to ONE of the app's connections.
        await live[-1].send_str(json.dumps(envelope))

    def push(self, envelope_type: str, payload: Dict[str, Any], *, retry_attempt: int = 0,
             retry_reason: str = "", accepts_response_payload: bool = False) -> Dict[str, Any]:
        envelope = {"envelope_id": str(uuid.uuid4()), "type": envelope_type, "payload": payload,
                    "accepts_response_payload": accepts_response_payload,
                    "retry_attempt": retry_attempt, "retry_reason": retry_reason}
        with self._lock:
            self.envelopes.append(envelope)
        self.run_in_loop(self._asend(envelope))
        return envelope

    def acked(self, envelope: Dict[str, Any]) -> bool:
        with self._lock:
            return envelope["envelope_id"] in self.acks

    # Web API methods ---------------------------------------------------------------------------
    def _user(self, user_id: str) -> Dict[str, Any]:
        if user_id in self.users:
            return self.users[user_id]
        is_bot = user_id == BOT_USER_ID
        name = BOT_NAME if is_bot else f"user{user_id.lower()}"
        return {"id": user_id, "team_id": TEAM_ID, "name": name, "real_name": name.title(), "deleted": False,
                "is_bot": is_bot, "is_app_user": False, "tz": "UTC",
                "profile": {"display_name": name, "real_name": name.title(), "bot_id": BOT_ID if is_bot else None}}

    def _channel(self, channel: str) -> Dict[str, Any]:
        is_im = channel.startswith("D")
        info: Dict[str, Any] = {"id": channel, "is_im": is_im, "is_channel": channel.startswith("C"),
                                "is_group": channel.startswith("G"), "is_mpim": False, "is_private": not channel.startswith("C"),
                                "is_member": True, "is_archived": False, "context_team_id": TEAM_ID}
        if is_im:
            info["user"] = "U" + channel[1:]
        else:
            info["name"] = f"standin-{channel.lower()}"
        return info

    def _m_auth_test(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        return {"url": "https://standin.slack.com/", "team": "Standin", "user": BOT_NAME, "team_id": TEAM_ID,
                "user_id": BOT_USER_ID, "bot_id": BOT_ID, "is_enterprise_install": False}

    def _m_apps_connections_open(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        return {"url": self.ws_url}

    def _m_users_info(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {"user": self._user(str(p.get("user", "")))}

    def _m_bots_info(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {"bot": {"id": p.get("bot", BOT_ID), "name": BOT_NAME, "user_id": BOT_USER_ID, "app_id": APP_ID}}

    def _m_conversations_info(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {"channel": self._channel(str(p.get("channel", "")))}

    def _m_users_conversations(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            ids = sorted(c for c in self._history if not c.startswith("D"))
        return {"channels": [self._channel(c) for c in ids], "response_metadata": {"next_cursor": ""}}

    def _m_conversations_open(self, p: Dict[str, Any]) -> Dict[str, Any]:
        users = str(p.get("users", "")).split(",")[0]
        return {"channel": self._channel("D" + users[1:])}

    def _thread_messages(self, channel: str, ts: str) -> List[Dict[str, Any]]:
        with self._lock:
            msgs = list(self._history.get(channel, []))
        return [m for m in msgs if m["ts"] == ts or m.get("thread_ts") == ts]

    def _m_conversations_replies(self, p: Dict[str, Any]) -> Dict[str, Any]:
        msgs = self._thread_messages(str(p.get("channel", "")), str(p.get("ts", "")))
        limit = int(p.get("limit") or 1000)
        return {"messages": msgs[:limit], "has_more": False, "response_metadata": {"next_cursor": ""}}

    def _m_conversations_history(self, p: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            msgs = [m for m in self._history.get(str(p.get("channel", "")), []) if not m.get("thread_ts")
                    or m.get("thread_ts") == m["ts"]]
        limit = int(p.get("limit") or 100)
        return {"messages": list(reversed(msgs))[:limit], "has_more": False,
                "response_metadata": {"next_cursor": ""}}

    def _remember(self, channel: str, msg: Dict[str, Any]) -> None:
        with self._lock:
            self._history.setdefault(channel, []).append(msg)

    def _bot_post(self, channel: str, text: str, p: Dict[str, Any], kind: str) -> Dict[str, Any]:
        ts = self.next_ts()
        msg = {"type": "message", "user": BOT_USER_ID, "bot_id": BOT_ID, "app_id": APP_ID, "text": text,
               "ts": ts, "team": TEAM_ID}
        if p.get("blocks"):
            msg["blocks"] = p["blocks"]
        thread_ts = p.get("thread_ts")
        if thread_ts:
            msg["thread_ts"] = str(thread_ts)
        self._remember(channel, msg)
        with self._lock:
            self._visible[(channel, ts)] = Visible(ts, text, extra={
                "kind": kind, "thread_ts": msg.get("thread_ts"), "blocks": p.get("blocks")})
        return msg

    def _m_chat_postMessage(self, p: Dict[str, Any]) -> Dict[str, Any]:
        channel = str(p.get("channel", ""))
        text = str(p.get("text") or "")
        if not channel:
            raise _ApiError("channel_not_found")
        if len(text) > MAX_TEXT:
            raise _ApiError("msg_too_long")
        if not text and not p.get("blocks") and not p.get("attachments"):
            raise _ApiError("no_text")
        msg = self._bot_post(channel, text, p, "text")
        return {"channel": channel, "ts": msg["ts"], "message": msg}

    def _m_chat_postEphemeral(self, p: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._ephemerals.append({"channel": str(p.get("channel", "")), "user": p.get("user"),
                                     "text": p.get("text", ""), "via": "chat.postEphemeral"})
        return {"message_ts": self.next_ts()}

    def _vis(self, p: Dict[str, Any]) -> Visible:
        with self._lock:
            vis = self._visible.get((str(p.get("channel", "")), str(p.get("ts", ""))))
        if vis is None or vis.deleted:
            raise _ApiError("message_not_found")
        return vis

    def _m_chat_update(self, p: Dict[str, Any]) -> Dict[str, Any]:
        vis = self._vis(p)
        text = str(p.get("text") or "")
        if len(text) > MAX_TEXT:
            raise _ApiError("msg_too_long")
        with self._lock:
            vis.text = text
            vis.edits += 1
            if "blocks" in p:
                vis.extra["blocks"] = p.get("blocks")
        return {"channel": p["channel"], "ts": p["ts"], "text": text,
                "message": {"type": "message", "user": BOT_USER_ID, "bot_id": BOT_ID, "text": text}}

    def _m_chat_delete(self, p: Dict[str, Any]) -> Dict[str, Any]:
        vis = self._vis(p)
        with self._lock:
            vis.deleted = True
        return {"channel": p["channel"], "ts": p["ts"]}

    def _m_chat_startStream(self, p: Dict[str, Any]) -> Dict[str, Any]:
        channel = str(p.get("channel", ""))
        if not p.get("thread_ts"):
            raise _ApiError("invalid_arguments")
        msg = self._bot_post(channel, str(p.get("markdown_text") or ""), p, "stream")
        return {"channel": channel, "ts": msg["ts"]}

    def _append(self, p: Dict[str, Any], stop: bool) -> Dict[str, Any]:
        vis = self._vis(p)
        if vis.extra.get("stopped"):
            raise _ApiError("message_not_in_streaming_state")
        with self._lock:
            vis.text += str(p.get("markdown_text") or "")
            vis.edits += 1
            if stop:
                vis.extra["stopped"] = True
                if p.get("blocks"):
                    vis.extra["blocks"] = p["blocks"]
        return {"channel": p["channel"], "ts": p["ts"]}

    def _m_chat_appendStream(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self._append(p, stop=False)

    def _m_chat_stopStream(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self._append(p, stop=True)

    def _m_ok(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    # Side-effect-only methods the adapter calls (reactions, assistant thread status/title/prompts).
    _m_reactions_add = _m_reactions_remove = _m_ok
    _m_assistant_threads_setStatus = _m_assistant_threads_setTitle = _m_assistant_threads_setSuggestedPrompts = _m_ok

    def _m_files_getUploadURLExternal(self, p: Dict[str, Any]) -> Dict[str, Any]:
        fid = f"F{next(self._ids):08d}"
        self.files[fid] = {"id": fid, "name": p.get("filename"), "title": p.get("filename"),
                           "size": int(p.get("length") or 0)}
        return {"upload_url": f"{self.base_url}/upload/{fid}", "file_id": fid}

    def _m_files_completeUploadExternal(self, p: Dict[str, Any]) -> Dict[str, Any]:
        entries = p.get("files") or []
        channel = str(p.get("channel_id") or p.get("channels") or "")
        out = []
        for e in entries if isinstance(entries, list) else []:
            f = self.files.setdefault(e["id"], {"id": e["id"]})
            f["title"] = e.get("title") or f.get("title")
            out.append({k: v for k, v in f.items() if not k.startswith("_")})
            if channel:
                comment = str(p.get("initial_comment") or "")
                msg = self._bot_post(channel, comment, {"thread_ts": p.get("thread_ts")}, "file")
                with self._lock:
                    self._visible[(channel, msg["ts"])].extra.update(file_id=e["id"], title=f.get("title"))
        return {"files": out}

    def _m_files_info(self, p: Dict[str, Any]) -> Dict[str, Any]:
        f = self.files.get(str(p.get("file", "")))
        if f is None:
            raise _ApiError("file_not_found")
        return {"file": {k: v for k, v in f.items() if not k.startswith("_")}}

    # test-facing inbound -----------------------------------------------------------------------
    def dm_channel(self, user_id: str) -> str:
        return "D" + user_id[1:]

    def message_event(self, channel: str, user_id: str, text: str, *, channel_type: str,
                      thread_ts: Optional[str] = None, event_type: str = "message", ts: Optional[str] = None,
                      **extra: Any) -> Dict[str, Any]:
        ts = ts or self.next_ts()
        event: Dict[str, Any] = {"type": event_type, "user": user_id, "text": text, "ts": ts, "event_ts": ts,
                                 "channel": channel, "team": TEAM_ID, "client_msg_id": str(uuid.uuid4()),
                                 "blocks": [{"type": "rich_text", "block_id": "b1", "elements": [
                                     {"type": "rich_text_section", "elements": [{"type": "text", "text": text}]}]}]}
        if event_type == "message":
            event["channel_type"] = channel_type
        if thread_ts:
            event["thread_ts"] = thread_ts
        event.update(extra)
        return event

    def deliver(self, event: Dict[str, Any], *, event_id: Optional[str] = None, retry_attempt: int = 0,
                retry_reason: str = "") -> Dict[str, Any]:
        """Wrap ``event`` in an ``event_callback`` and push it over Socket Mode."""
        if retry_attempt == 0 and event.get("type") in ("message", "app_mention") and "subtype" not in event:
            self._remember(event["channel"], {k: v for k, v in event.items() if k != "blocks"})
        payload = {"token": "verification-token", "team_id": TEAM_ID, "api_app_id": APP_ID, "event": event,
                   "type": "event_callback", "event_id": event_id or f"Ev{next(self._ids):010d}",
                   "event_time": int(float(event.get("event_ts") or time.time())),
                   "authorizations": [{"enterprise_id": None, "team_id": TEAM_ID, "user_id": BOT_USER_ID,
                                       "is_bot": True, "is_enterprise_install": False}],
                   "is_ext_shared_channel": False, "event_context": f"4-standin-{uuid.uuid4().hex[:12]}"}
        return self.push("events_api", payload, retry_attempt=retry_attempt, retry_reason=retry_reason)

    def redeliver(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Slack's at-least-once retry: same event_id/event, new envelope, ``retry_attempt`` + 1."""
        payload = envelope["payload"]
        return self.deliver(payload["event"], event_id=payload["event_id"],
                            retry_attempt=int(envelope.get("retry_attempt") or 0) + 1, retry_reason="timeout")

    def dm(self, user_id: str, text: str, **extra: Any) -> Dict[str, Any]:
        event = self.message_event(self.dm_channel(user_id), user_id, text, channel_type="im", **extra)
        return self.deliver(event)

    def channel_post(self, channel: str, user_id: str, text: str, *, mention: bool,
                     thread_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        """A human posts in a channel. With ``mention`` Slack emits BOTH ``app_mention`` and the
        ``message`` (channel_type=channel) event for the same ts, as separate envelopes."""
        if mention:
            text = f"<@{BOT_USER_ID}> {text}"
        msg = self.message_event(channel, user_id, text, channel_type="channel", thread_ts=thread_ts)
        out = [self.deliver(msg)]
        if mention:
            app_mention = dict(msg, type="app_mention")
            app_mention.pop("channel_type", None)
            out.insert(0, self.deliver(app_mention))
        return out

    def dm_file(self, user_id: str, filename: str, data: bytes, mime: str, caption: str = "") -> Dict[str, Any]:
        fid = f"F{next(self._ids):08d}"
        path = f"{TEAM_ID}-{fid}/{filename}"
        url = f"{self.base_url}/files-pri/{path}"
        self.files[fid] = {"id": fid, "name": filename, "title": filename, "mimetype": mime,
                           "filetype": filename.rsplit(".", 1)[-1], "size": len(data), "url_private": url,
                           "url_private_download": url, "mode": "hosted", "_path": path, "_data": data}
        public = {k: v for k, v in self.files[fid].items() if not k.startswith("_")}
        event = self.message_event(self.dm_channel(user_id), user_id, caption, channel_type="im",
                                   subtype="file_share", files=[public])
        return self.deliver(event)

    def block_action(self, user_id: str, channel: str, message_ts: str, action: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            vis = self._visible.get((channel, message_ts))
        message = {"type": "message", "user": BOT_USER_ID, "bot_id": BOT_ID, "ts": message_ts,
                   "text": vis.text if vis else "", "blocks": (vis.extra.get("blocks") if vis else None) or []}
        if vis and vis.extra.get("thread_ts"):
            message["thread_ts"] = vis.extra["thread_ts"]
        act = {"action_id": action["action_id"], "block_id": action.get("block_id") or "blk",
               "type": action.get("type", "button"), "value": action.get("value", ""),
               "text": action.get("text") or {"type": "plain_text", "text": action.get("value", "")},
               "action_ts": f"{time.time():.6f}"}
        payload = {"type": "block_actions", "team": {"id": TEAM_ID, "domain": "standin"},
                   "user": {"id": user_id, "username": f"user{user_id.lower()}", "name": f"user{user_id.lower()}",
                            "team_id": TEAM_ID},
                   "api_app_id": APP_ID, "token": "verification-token", "trigger_id": f"trig-{uuid.uuid4().hex[:10]}",
                   "container": {"type": "message", "message_ts": message_ts, "channel_id": channel,
                                 "is_ephemeral": False},
                   "channel": {"id": channel, "name": self._channel(channel).get("name", "directmessage")},
                   "message": message, "state": {"values": {}},
                   "response_url": self._new_response_url(channel, user_id), "actions": [act]}
        return self.push("interactive", payload, accepts_response_payload=False)

    def slash(self, user_id: str, channel: str, command: str, text: str = "") -> Dict[str, Any]:
        payload = {"token": "verification-token", "team_id": TEAM_ID, "team_domain": "standin",
                   "channel_id": channel, "channel_name": self._channel(channel).get("name", "directmessage"),
                   "user_id": user_id, "user_name": f"user{user_id.lower()}", "command": command, "text": text,
                   "api_app_id": APP_ID, "is_enterprise_install": "false",
                   "response_url": self._new_response_url(channel, user_id),
                   "trigger_id": f"trig-{uuid.uuid4().hex[:10]}"}
        return self.push("slash_commands", payload, accepts_response_payload=True)

    # ground truth ------------------------------------------------------------------------------
    def visible(self, channel: str) -> List[Visible]:
        with self._lock:
            return [v for (c, _), v in self._visible.items() if c == str(channel) and not v.deleted]

    def buttons(self, channel: str) -> List[Dict[str, Any]]:
        """Every button element in blocks the bot currently shows in ``channel`` (with ``message_id``)."""
        out = []
        with self._lock:
            items = [(ts, v) for (c, ts), v in self._visible.items() if c == str(channel) and not v.deleted]
        for ts, vis in items:
            for block in vis.extra.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                elements = block.get("elements") or ([block["accessory"]] if block.get("accessory") else [])
                for el in elements:
                    if isinstance(el, dict) and el.get("type") == "button":
                        out.append({**el, "block_id": block.get("block_id"), "message_id": ts})
        return out


class _ApiError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
