"""Telegram Bot API stand-in (https://core.telegram.org/bots/api).

The real ``plugins/platforms/telegram`` adapter reaches it through python-telegram-bot's own
``base_url``/``base_file_url`` (``platforms.telegram.extra.base_url``), so every request crosses the
real PTB HTTP stack: ``POST {base}/bot<token>/<method>`` with form/JSON/multipart parameters, and
long-poll ``getUpdates`` that returns queued ``Update`` objects. Only the methods the adapter calls
are implemented; any other method answers ``404 Not Found`` like the real Bot API (recorded as
faulted), so a call the stand-in does not model can never be mistaken for a success.

Payload fidelity kept to the Bot API: message text must be 1..4096 chars after entity parsing
(``message text is empty`` / ``message is too long``), and with ``parse_mode=MarkdownV2`` every
reserved character outside an entity must be backslash-escaped, else ``400 can't parse entities``.
"""

from __future__ import annotations

import asyncio
import itertools
import re
import time
from typing import Any, Dict, List, Optional

from aiohttp import web

from tests.fakes.platforms._standin import StandinServer, Visible, decode_value

BOT_ID = 7000000001
BOT_USERNAME = "hermes_standin_bot"
MAX_TEXT = 4096
# MarkdownV2 (https://core.telegram.org/bots/api#markdownv2-style): these must be escaped outside
# entities; inside ``code``/```pre``` only ` and \ are special.
_MDV2_RESERVED = set("_*[]()~`>#+-=|{}.!")
_MDV2_LINK = re.compile(r"\[((?:\\.|[^\]\\])*)\]\(((?:\\.|[^)\\])*)\)")


def mdv2_error(text: str) -> str | None:
    """The Bot API's ``can't parse entities`` reason for ``text`` in MarkdownV2, or None if it parses.

    A faithful subset of the server's parser: escapes, code/pre spans, links, the paired style
    markers (``*`` ``_`` ``__`` ``~`` ``||``) and line-leading ``>`` quotes. Any other reserved
    character must carry a preceding backslash.
    """
    i, n, open_marks = 0, len(text), []
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if text.startswith("```", i):
            end = text.find("```", i + 3)
            if end < 0:
                return "Can't find end of Pre entity at byte offset %d" % i
            i = end + 3
            continue
        if ch == "`":
            end = text.find("`", i + 1)
            if end < 0:
                return "Can't find end of Code entity at byte offset %d" % i
            i = end + 1
            continue
        if ch == "[":
            m = _MDV2_LINK.match(text, i)
            if not m:
                return "Character '[' is reserved and must be escaped with the preceding '\\'"
            i = m.end()
            continue
        if ch == ">" and (i == 0 or text[i - 1] == "\n"):
            i += 1
            continue
        mark = next((mk for mk in ("||", "__", "*", "_", "~") if text.startswith(mk, i)), None)
        if mark is not None:
            if open_marks and open_marks[-1] == mark:
                open_marks.pop()
            else:
                open_marks.append(mark)
            i += len(mark)
            continue
        if ch in _MDV2_RESERVED:
            return f"Character '{ch}' is reserved and must be escaped with the preceding '\\'"
        i += 1
    if open_marks:
        return f"Can't find end of the entity starting with '{open_marks[-1]}'"
    return None


class BotApiError(Exception):
    """A Bot API ``ok: false`` reply (``description`` + ``error_code``) from a method handler."""

    def __init__(self, description: str, code: int = 400) -> None:
        super().__init__(description)
        self.description, self.code = description, code


def _check_text(p: Dict[str, Any], field: str = "text") -> str:
    text = str(p.get(field) or "")
    if p.get("parse_mode") == "MarkdownV2":
        why = mdv2_error(text)
        if why:
            raise BotApiError(f"Bad Request: can't parse entities: {why}")
        text = re.sub(r"\\(.)", r"\1", text)  # length and emptiness count the parsed text
    if not text.strip():
        raise BotApiError("Bad Request: message text is empty")
    if len(text) > MAX_TEXT:
        raise BotApiError("Bad Request: message is too long")
    return str(p.get(field) or "")


class TelegramStandin(StandinServer):
    def __init__(self, token: str = "123456:standin-token") -> None:
        super().__init__()
        self.token = token
        self._update_ids = itertools.count(1000)
        self._message_ids = itertools.count(1)
        self._inbound_ids = itertools.count(50_000)
        self._updates: List[Dict[str, Any]] = []
        self._update_event: Optional[asyncio.Event] = None
        self.files: Dict[str, bytes] = {}
        self.file_paths: Dict[str, str] = {}
        # (chat_id, message_id) -> Visible for BOT messages only
        self._visible: Dict[tuple, Visible] = {}
        self.chats: Dict[str, Dict[str, Any]] = {}
        # (chat_id, draft_id) -> latest draft preview text (sendMessageDraft: ephemeral, not a message)
        self.drafts: Dict[tuple, str] = {}

    @property
    def api_base(self) -> str:
        return f"{self.base_url}/bot"

    @property
    def file_base(self) -> str:
        return f"{self.base_url}/file/bot"

    # server ------------------------------------------------------------------------------------
    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app.router.add_route("*", "/bot{token}/{method}", self._api)
        app.router.add_get("/file/bot{token}/{path:.*}", self._file)
        return app

    async def _params(self, request: web.Request) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(request.query)
        ctype = request.content_type or ""
        if ctype == "application/json":
            body = await request.json()
            params.update(body or {})
        elif ctype.startswith("multipart/"):
            reader = await request.multipart()
            async for part in reader:
                if part.filename:
                    params[part.name] = {"filename": part.filename, "size": len(await part.read())}
                else:
                    params[part.name] = decode_value(await part.text())
        elif request.can_read_body:
            for k, v in (await request.post()).items():
                params[k] = decode_value(v) if isinstance(v, str) else v
        return params

    async def _api(self, request: web.Request) -> web.Response:
        method = request.match_info["method"]
        if request.match_info["token"] != self.token:
            return web.json_response({"ok": False, "error_code": 401, "description": "Unauthorized"}, status=401)
        params = await self._params(request)
        if method == "getUpdates":
            result = await self._get_updates(params)
            self.record(method, params, result)
            return web.json_response({"ok": True, "result": result})
        fault = self.take_fault(method, params)
        if fault is not None:
            self.record(method, params, fault.body, faulted=True)
            return web.json_response(fault.body, status=fault.status)
        handler = getattr(self, f"_m_{method}", None)
        if handler is None:
            body = {"ok": False, "error_code": 404, "description": "Not Found"}
            self.record(method, params, body, faulted=True)
            return web.json_response(body, status=404)
        try:
            result = handler(params)
        except BotApiError as exc:
            body = {"ok": False, "error_code": exc.code, "description": exc.description}
            self.record(method, params, body, faulted=True)
            return web.json_response(body, status=exc.code)
        self.record(method, params, result)
        return web.json_response({"ok": True, "result": result})

    async def _file(self, request: web.Request) -> web.Response:
        path = request.match_info["path"]
        self.record("file_download", {"path": path}, None)
        for file_id, fpath in self.file_paths.items():
            if fpath == path:
                return web.Response(body=self.files[file_id])
        return web.Response(status=404)

    async def _get_updates(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self._update_event is None:
            self._update_event = asyncio.Event()
        offset = int(params.get("offset") or 0)
        timeout = min(float(params.get("timeout") or 0), 2.0)
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                # An update id below the confirmed offset is gone (Bot API semantics); a replay
                # (``redeliver``) re-queues the very same update object with a fresh position.
                self._updates = [u for u in self._updates if u["_replay"] or u["update_id"] >= offset]
                ready = self._updates[:]
                for u in ready:
                    u["_replay"] = False
            if ready:
                return [{k: v for k, v in u.items() if k != "_replay"} for u in ready]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            self._update_event.clear()
            try:
                await asyncio.wait_for(self._update_event.wait(), remaining)
            except asyncio.TimeoutError:
                return []

    def _push(self, update: Dict[str, Any], replay: bool = False) -> None:
        with self._lock:
            self._updates.append({**update, "_replay": replay})
        loop = self._loop
        if loop is not None and self._update_event is not None:
            loop.call_soon_threadsafe(self._update_event.set)

    # Bot API methods ---------------------------------------------------------------------------
    def _bot_user(self) -> Dict[str, Any]:
        return {"id": BOT_ID, "is_bot": True, "first_name": "Hermes", "username": BOT_USERNAME,
                "can_join_groups": True, "can_read_all_group_messages": True, "supports_inline_queries": False}

    def _chat(self, chat_id: Any) -> Dict[str, Any]:
        return self.chats.get(str(chat_id)) or {"id": int(chat_id), "type": "private", "first_name": "User"}

    def _bot_message(self, params: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
        chat_id = str(params["chat_id"])
        mid = next(self._message_ids)
        msg = {"message_id": mid, "date": int(time.time()), "chat": self._chat(chat_id), "from": self._bot_user(),
               **fields}
        if params.get("message_thread_id"):
            msg["message_thread_id"] = int(params["message_thread_id"])
        with self._lock:
            self._visible[(chat_id, str(mid))] = Visible(str(mid), fields.get("text") or fields.get("caption") or "",
                                                         extra={"kind": fields.get("_kind", "text")})
        return msg

    def _m_getMe(self, _p: Dict[str, Any]) -> Dict[str, Any]:
        return self._bot_user()

    def _m_getMyCommands(self, _p: Dict[str, Any]) -> List[Any]:
        return []

    def _m_getChat(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return {**self._chat(p["chat_id"]), "accent_color_id": 0, "max_reaction_count": 11}

    def _m_sendMessage(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self._bot_message(p, text=_check_text(p))

    def _m_sendMessageDraft(self, p: Dict[str, Any]) -> bool:
        """Animate a private-chat draft preview; not a message (no message_id)."""
        if not int(p.get("draft_id") or 0):
            raise BotApiError("Bad Request: draft_id must be non-zero")
        text = _check_text(p)
        with self._lock:
            self.drafts[(str(p["chat_id"]), int(p["draft_id"]))] = text
        return True

    def _m_sendRichMessageDraft(self, p: Dict[str, Any]) -> bool:
        if not int(p.get("draft_id") or 0):
            raise BotApiError("Bad Request: draft_id must be non-zero")
        with self._lock:
            self.drafts[(str(p["chat_id"]), int(p["draft_id"]))] = str(p.get("text") or p.get("content") or "")
        return True

    def _ok_true(self, _p: Dict[str, Any]) -> bool:
        return True

    # Side-effect-only methods the adapter calls (reactions, typing, command menu, webhook reset).
    _m_setMessageReaction = _m_sendChatAction = _m_setMyCommands = _m_deleteMyCommands = _ok_true
    _m_setMyShortDescription = _m_setMyDescription = _m_deleteWebhook = _m_answerCallbackQuery = _ok_true

    def _m_sendPhoto(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self._bot_message(p, caption=p.get("caption", ""), _kind="photo",
                                 photo=[{"file_id": "out-photo", "file_unique_id": "op", "width": 1, "height": 1}])

    def _m_sendDocument(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self._bot_message(p, caption=p.get("caption", ""), _kind="document",
                                 document={"file_id": "out-doc", "file_unique_id": "od"})

    def _m_editMessageText(self, p: Dict[str, Any]) -> Dict[str, Any]:
        _check_text(p)
        chat_id, mid = str(p["chat_id"]), str(p["message_id"])
        with self._lock:
            vis = self._visible.get((chat_id, mid))
            if vis is not None:
                vis.text = p.get("text", "")
                vis.edits += 1
        return {"message_id": int(mid), "date": int(time.time()), "chat": self._chat(chat_id),
                "from": self._bot_user(), "text": p.get("text", ""), "edit_date": int(time.time())}

    def _m_editMessageReplyMarkup(self, p: Dict[str, Any]) -> Any:
        return {"message_id": int(p["message_id"]), "date": int(time.time()), "chat": self._chat(p["chat_id"]),
                "from": self._bot_user(), "text": ""}

    def _m_deleteMessage(self, p: Dict[str, Any]) -> bool:
        with self._lock:
            vis = self._visible.get((str(p["chat_id"]), str(p["message_id"])))
            if vis is not None:
                vis.deleted = True
        return True

    def _m_getFile(self, p: Dict[str, Any]) -> Dict[str, Any]:
        fid = p["file_id"]
        return {"file_id": fid, "file_unique_id": f"u-{fid}", "file_size": len(self.files.get(fid, b"")),
                "file_path": self.file_paths.get(fid, f"documents/{fid}")}

    # test-facing driver ------------------------------------------------------------------------
    def _user(self, user_id: int, name: str = "Tester") -> Dict[str, Any]:
        return {"id": int(user_id), "is_bot": False, "first_name": name, "username": f"user{user_id}",
                "language_code": "en"}

    def _inbound(self, chat: Dict[str, Any], user_id: int, **fields: Any) -> Dict[str, Any]:
        self.chats[str(chat["id"])] = chat
        mid = next(self._inbound_ids)
        update = {"update_id": next(self._update_ids),
                  "message": {"message_id": mid, "date": int(time.time()), "chat": chat,
                              "from": self._user(user_id), **fields}}
        self._push(update)
        return update

    def dm(self, user_id: int, text: str) -> Dict[str, Any]:
        chat = {"id": int(user_id), "type": "private", "first_name": "Tester", "username": f"user{user_id}"}
        return self._inbound(chat, user_id, text=text)

    def group(self, chat_id: int, user_id: int, text: str, *, mention: bool) -> Dict[str, Any]:
        chat = {"id": int(chat_id), "type": "supergroup", "title": "Standin Group"}
        fields: Dict[str, Any] = {"text": text}
        if mention:
            handle = f"@{BOT_USERNAME}"
            fields["text"] = f"{handle} {text}"
            fields["entities"] = [{"type": "mention", "offset": 0, "length": len(handle)}]
        return self._inbound(chat, user_id, **fields)

    def dm_document(self, user_id: int, filename: str, data: bytes, mime: str, caption: str = "") -> Dict[str, Any]:
        fid = f"doc-{len(self.files) + 1}"
        self.files[fid] = data
        self.file_paths[fid] = f"documents/{filename}"
        chat = {"id": int(user_id), "type": "private", "first_name": "Tester"}
        fields: Dict[str, Any] = {"document": {"file_id": fid, "file_unique_id": f"u-{fid}", "file_name": filename,
                                               "mime_type": mime, "file_size": len(data)}}
        if caption:
            fields["caption"] = caption
        return self._inbound(chat, user_id, **fields)

    def callback(self, user_id: int, chat_id: int, message_id: int, data: str) -> Dict[str, Any]:
        """A user clicks an inline button under bot message ``message_id``."""
        chat = self._chat(chat_id)
        with self._lock:
            vis = self._visible.get((str(chat_id), str(message_id)))
        update = {"update_id": next(self._update_ids),
                  "callback_query": {"id": f"cb-{next(self._inbound_ids)}", "from": self._user(user_id),
                                     "chat_instance": f"ci-{chat_id}", "data": data,
                                     "message": {"message_id": int(message_id), "date": int(time.time()),
                                                 "chat": chat, "from": self._bot_user(),
                                                 "text": vis.text if vis else ""}}}
        self._push(update)
        return update

    def redeliver(self, update: Dict[str, Any]) -> None:
        """Serve an already-delivered update again (a Bot API replay after a lost offset ack)."""
        self._push(dict(update), replay=True)

    def visible(self, chat_id: Any) -> List[Visible]:
        with self._lock:
            return [v for (c, _), v in self._visible.items() if c == str(chat_id) and not v.deleted]

    def buttons(self, chat_id: Any) -> List[Dict[str, Any]]:
        """Every inline button the bot attached to a message in ``chat_id`` (with its message id)."""
        out = []
        for call in self.calls_of("sendMessage", "editMessageText", "editMessageReplyMarkup"):
            if str(call.params.get("chat_id")) != str(chat_id) or call.faulted:
                continue
            markup = call.params.get("reply_markup") or {}
            mid = call.params.get("message_id") or (call.response or {}).get("message_id")
            for row in (markup.get("inline_keyboard") or []) if isinstance(markup, dict) else []:
                for btn in row:
                    out.append({**btn, "message_id": mid})
        return out
