"""WhatsApp over WAHA (WhatsApp HTTP API) — third WhatsApp transport.

Shares ``WhatsAppBehaviorMixin`` with the Baileys bridge adapter and the Meta
Cloud API adapter (gating, mention rules, allow-lists, markdown formatting);
owns only the WAHA transport: REST outbound, session-webhook inbound.

Layout follows ``gateway/platforms/ADDING_A_PLATFORM.md``'s sibling-adapter
pattern (see the WhatsApp section). Zero core changes: ``Platform("waha")``
resolves through the enum's bundled-plugin ``_missing_`` hook.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.whatsapp_identity import to_whatsapp_jid

logger = logging.getLogger(__name__)


def _waha_chat_id(chat_id: str) -> str:
    """Outbound chatId in the form WAHA/NOWEB documents (see waha.devlike.pro chat-ids).

    ``gateway.whatsapp_identity.to_whatsapp_jid`` renders bare phones as
    ``<digits>@s.whatsapp.net`` (Baileys-bridge form); the WAHA docs are explicit that
    internal ``@s.whatsapp.net`` JIDs must be converted to ``@c.us`` when used as a
    ``chatId``. ``@lid`` targets are passed through unchanged (WAHA accepts them and
    routes by the hidden id); groups/broadcasts/newsletters are returned as-is."""
    jid = to_whatsapp_jid(chat_id)
    if jid.endswith("@s.whatsapp.net"):
        jid = jid.split("@", 1)[0] + "@c.us"
    return jid


AIOHTTP_AVAILABLE = True
try:
    import aiohttp
    from aiohttp import web
except ImportError:  # pragma: no cover - aiohttp is a core dependency
    AIOHTTP_AVAILABLE = False

_TRUTHY = {"true", "1", "yes", "on"}

# WAHA session states that mean "this transport can deliver".
_WORKING_STATES = {"WORKING"}


def _wenv(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _bool_env(name: str, default: str = "false") -> bool:
    return (_wenv(name, default) or default).strip().lower() in _TRUTHY


def _mentioned_ids_from_data(payload: Dict[str, Any]) -> List[str]:
    """Mentioned JIDs from the engine-specific ``_data`` blob (NOWEB/GOWS keep the
    raw Baileys message there: ``message.<kind>.contextInfo.mentionedJid``)."""
    data = payload.get("_data")
    if not isinstance(data, dict):
        return []
    message = data.get("message")
    if not isinstance(message, dict):
        return []
    ids: List[str] = []
    for kind in message.values():
        if not isinstance(kind, dict):
            continue
        ctx = kind.get("contextInfo")
        if isinstance(ctx, dict):
            ids.extend(str(j) for j in (ctx.get("mentionedJid") or []) if j)
    return ids


def _waha_media_kind(mimetype: str) -> Optional[MessageType]:
    mime = (mimetype or "").lower()
    if mime.startswith("image/"):
        return MessageType.PHOTO
    if mime.startswith("video/"):
        return MessageType.VIDEO
    if mime in ("audio/ogg", "audio/opus") or "ogg" in mime:
        return MessageType.VOICE
    if mime.startswith("audio/"):
        return MessageType.AUDIO
    return MessageType.DOCUMENT


class WahaAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """WhatsApp transport backed by a self-hosted WAHA instance."""

    # WAHA/WhatsApp render the same dialect; the mixin's formatter applies as-is.
    FALLBACK_ON_FINAL_EDIT_FLOOD = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("waha"))
        extra = config.extra
        self._base_url = str(
            extra.get("base_url") or _wenv("WAHA_BASE_URL", "")).rstrip("/")
        self._api_key = str(extra.get("api_key") or os.getenv("WAHA_API_KEY", "")).strip()
        self._session = str(extra.get("session") or _wenv("WAHA_SESSION", "default") or "default")
        self._webhook_port = int(extra.get("webhook_port") or _wenv("WAHA_WEBHOOK_PORT", "8655") or 8655)
        self._webhook_secret = str(
            extra.get("webhook_secret") or os.getenv("WAHA_WEBHOOK_SECRET", "")).strip()
        self._reply_prefix: Optional[str] = extra.get("reply_prefix")
        self._dm_policy = str(extra.get("dm_policy") or _wenv("WAHA_DM_POLICY", "pairing")).strip().lower()
        self._allow_from = self._coerce_allow_list(
            self._select_dm_allowlist(extra, ("WAHA_ALLOWED_USERS",), _wenv))
        self._group_policy = str(extra.get("group_policy") or _wenv("WAHA_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = self._coerce_allow_list(
            extra.get("group_allow_from") or _wenv("WAHA_GROUP_ALLOW_FROM", ""))
        self._mention_patterns = self._compile_mention_patterns()
        rr = extra.get("send_read_receipts", False)
        self._send_read_receipts = rr if isinstance(rr, bool) else str(rr or "").strip().lower() in _TRUTHY
        self._bot_ids: set[str] = set()
        # Learned LID→phone-JID pairs (@c.us form) from observed alt fields; consulted
        # when a LID arrives without its alt (WAHA's Lids API needs the NOWEB store,
        # which is off in minimal deployments — see waha.devlike.pro contacts/lids).
        self._lid_pn_cache: dict[str, str] = {}
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._message_handler = None
        self._last_seen_message_ids: Dict[str, set] = {}

    # ------------------------------------------------------------------ transport helpers
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None,
                       timeout: float = 30) -> tuple[int, Any]:
        """``(status, json_or_text)`` against the WAHA REST API."""
        assert self._http_session is not None
        async with self._http_session.request(
                method, self._url(path), json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            try:
                return resp.status, await resp.json()
            except Exception:
                return resp.status, await resp.text()

    # ------------------------------------------------------------------ lifecycle
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._base_url:
            logger.error("[waha] WAHA_BASE_URL is not configured")
            return False
        self._http_session = aiohttp.ClientSession(headers=self._headers())
        status, body = await self._request("GET", f"/api/sessions/{self._session}", timeout=15)
        if status != 200:
            logger.error("[waha] session %r not found (HTTP %s): %s", self._session, status, body)
            await self._http_session.close()
            self._http_session = None
            return False
        state = str((body or {}).get("status") or "").upper() if isinstance(body, dict) else ""
        if state not in _WORKING_STATES:
            logger.error("[waha] session %r is %s (need WORKING) — scan the QR in the WAHA dashboard",
                         self._session, state or "unknown")
            await self._http_session.close()
            self._http_session = None
            return False
        me = (body or {}).get("me") or {}
        if isinstance(me, dict) and (me.get("id") or me.get("lid")):
            self._bot_ids = {str(v) for v in (me.get("id"), me.get("lid")) if v}
        self._running = True
        await self._start_webhook_receiver()
        self._wire_plugin_handlers()
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("[waha] connected: session=%s base=%s (webhook :%s)", self._session,
                    self._base_url, self._webhook_port)
        return True

    async def disconnect(self) -> None:
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def _health_loop(self) -> None:
        """Periodic session-state check: log (and surface) a dropped WhatsApp session."""
        while self._running:
            try:
                await asyncio.sleep(60)
                status, body = await self._request("GET", f"/api/sessions/{self._session}", timeout=15)
                state = str((body or {}).get("status") or "").upper() if status == 200 and isinstance(body, dict) else ""
                if state and state not in _WORKING_STATES:
                    logger.warning("[waha] session %r state is %s — messages will not flow until it is WORKING",
                                   self._session, state)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("[waha] health check failed", exc_info=True)

    # ------------------------------------------------------------------ inbound webhook receiver
    async def _start_webhook_receiver(self) -> None:
        app = web.Application()
        app.router.add_post("/webhooks/waha", self._handle_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, None, self._webhook_port)  # dual-stack, all interfaces
        await site.start()

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        if self._webhook_secret:
            provided = request.headers.get("X-Hermes-Token", "")
            import hmac as _hmac
            if not _hmac.compare_digest(provided, self._webhook_secret):
                return web.Response(status=401)
        try:
            body = await request.json()
        except Exception:
            return web.Response(status=400)
        event = str(body.get("event") or "")
        if event != "message":
            return web.Response(status=200)  # ack everything else silently
        payload = body.get("payload")
        if not isinstance(payload, dict):
            return web.Response(status=200)
        me = body.get("me") or {}
        if isinstance(me, dict) and (me.get("id") or me.get("lid")):
            # Both forms: lid-addressed groups quote/mention the bot by LID while
            # me.id is the phone JID — the reply/mention gates must match either.
            self._bot_ids = {str(v) for v in (me.get("id"), me.get("lid")) if v}
        if payload.get("fromMe"):
            return web.Response(status=200)  # bot mode: own messages are echoes
        try:
            event_obj = await self._build_message_event(payload)
        except Exception:
            logger.warning("[waha] failed to build event", exc_info=True)
            return web.Response(status=200)
        if event_obj and self._message_handler:
            # The handler's return carries replies the agent path didn't deliver
            # itself (slash-command results, drain/limit notices). Route it through
            # the shared inline-reply helper — same contract as the other adapters —
            # instead of dropping it: agent turns already delivered return None.
            await self._dispatch_inline_reply(event_obj, log_cmd=event_obj.get_command())
        return web.Response(status=200)

    def _lid_alt_jid(self, payload: Dict[str, Any]) -> str:
        """Phone JID for a LID-addressed message (``addressingMode: "lid"``).

        WhatsApp increasingly delivers DMs keyed by a privacy LID (``<n>@lid``);
        NOWEB exposes the phone form alongside it as ``_data.key.remoteJidAlt``
        (``<n>@s.whatsapp.net``). Allowlists hold phone JIDs, so prefer the alt
        whenever the primary id is a LID. Every resolved pair is remembered in
        ``_lid_pn_cache`` for LIDs that later arrive without an alt field."""
        data = payload.get("_data") if isinstance(payload.get("_data"), dict) else {}
        key = data.get("key") if isinstance(data.get("key"), dict) else {}
        alt = str(key.get("remoteJidAlt") or "")
        primary = str(key.get("remoteJid") or payload.get("from") or "")
        if alt and primary.endswith("@lid"):
            # Canonical phone form is @c.us (docs: don't use @s.whatsapp.net as chatId)
            alt_cus = alt.split("@", 1)[0] + "@c.us"
            self._lid_pn_cache[primary] = alt_cus
            return alt_cus
        return ""

    @staticmethod
    def _participant_alt_jid(payload: Dict[str, Any]) -> str:
        """Phone JID (@c.us) for a LID-addressed group sender (``key.participantAlt``)."""
        data = payload.get("_data") if isinstance(payload.get("_data"), dict) else {}
        key = data.get("key") if isinstance(data.get("key"), dict) else {}
        alt = str(key.get("participantAlt") or "")
        participant = str(key.get("participant") or payload.get("participant") or "")
        if alt and participant.endswith("@lid"):
            return alt.split("@", 1)[0] + "@c.us"
        return ""

    def _resolve_lid(self, lid: str) -> str:
        """Best-effort phone JID for a bare LID: learned cache first, else the LID
        unchanged (WAHA accepts ``@lid`` chatIds; the Lids API needs the NOWEB store)."""
        if not lid.endswith("@lid"):
            return lid
        return self._lid_pn_cache.get(lid, lid)

    def _map_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """WAHA webhook payload → the bridge-shaped dict the shared mixin gates on."""
        chat_id = str(payload.get("chatId") or payload.get("from") or "")
        alt_jid = self._lid_alt_jid(payload)
        if chat_id.endswith("@lid"):
            chat_id = alt_jid or self._resolve_lid(chat_id)
        is_group = chat_id.endswith("@g.us")
        sender_id = str(payload.get("participant") or payload.get("from") or "")
        if sender_id.endswith("@lid"):
            participant_alt = self._participant_alt_jid(payload)
            if participant_alt:
                self._lid_pn_cache[sender_id] = participant_alt.split("@", 1)[0] + "@c.us"
                sender_id = participant_alt
            else:
                sender_id = alt_jid or self._resolve_lid(sender_id)
        sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
        media = payload.get("media") if isinstance(payload.get("media"), dict) else {}
        reply_to = payload.get("replyTo") if isinstance(payload.get("replyTo"), dict) else {}
        return {
            "chatId": chat_id,
            "chatName": payload.get("chatName") or chat_id,
            "senderId": sender_id,
            "senderName": sender.get("pushName") or payload.get("pushname") or "",
            "isGroup": is_group,
            "body": str(payload.get("body") or ""),
            "messageId": str(payload.get("id") or ""),
            "timestamp": datetime.fromtimestamp(
                float(payload.get("timestamp") or 0) or datetime.now(timezone.utc).timestamp(),
                tz=timezone.utc).isoformat(),
            "botIds": sorted(self._bot_ids),
            "mentionedIds": _mentioned_ids_from_data(payload),
            "hasQuotedMessage": bool(reply_to),
            "quotedMessageId": str(reply_to.get("id") or "") or None,
            "quotedParticipant": str(reply_to.get("participant") or "") or None,
            "quotedText": str(reply_to.get("body") or "") or None,
            "hasMedia": bool(payload.get("hasMedia")),
            "mediaUrls": [media["url"]] if media.get("url") else [],
            "mime": str(media.get("mimetype") or ""),
        }

    async def _build_message_event(self, payload: Dict[str, Any]) -> Optional[MessageEvent]:
        """WAHA payload → MessageEvent (or None when the shared gate skips it)."""
        data = self._map_payload(payload)
        if not self._should_process_message(data):
            if self._should_observe_unmentioned_group_message(data):
                await self._observe_bridge_group_message(data)
            else:
                logger.debug("[waha] gate rejected chat=%s sender=%s body=%.60r",
                             data.get("chatId"), data.get("senderId"), data.get("body"))
            return None
        msg_type = self._classify_waha_message(data)
        source = self.build_source(
            chat_id=data["chatId"], chat_name=data.get("chatName"),
            chat_type="group" if data["isGroup"] else "dm",
            user_id=data["senderId"], user_name=data.get("senderName") or None)
        cached_urls, media_types = await self._collect_waha_media(data, msg_type)
        body = data["body"]
        if data["isGroup"]:
            body = self._clean_bot_mention_text(body, data)
        quoted = bool(data.get("hasQuotedMessage"))
        return MessageEvent(
            text=body, message_type=msg_type, source=source, raw_message=payload,
            message_id=data.get("messageId"), media_urls=cached_urls, media_types=media_types,
            reply_to_message_id=data.get("quotedMessageId"),
            reply_to_text=data.get("quotedText"),
            reply_to_author_id=(self._normalize_whatsapp_id(data.get("quotedParticipant")) or None) if quoted else None,
            reply_to_is_own_message=self._message_is_reply_to_bot(data) if quoted else False,
        )

    @staticmethod
    def _classify_waha_message(data: Dict[str, Any]) -> MessageType:
        if data.get("hasMedia") and data.get("mediaUrls"):
            return _waha_media_kind(data.get("mime", "")) or MessageType.DOCUMENT
        return MessageType.TEXT

    async def _collect_waha_media(self, data: Dict[str, Any], msg_type: MessageType) -> tuple[list, list]:
        """Download WAHA-hosted media to a local cache so the agent gets real files."""
        urls = data.get("mediaUrls") or []
        if not urls:
            return [], []
        mime = data.get("mime", "")
        ext = mimetypes.guess_extension(mime.split(";")[0]) or ".bin"
        try:
            assert self._http_session is not None
            async with self._http_session.get(urls[0], timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.warning("[waha] media download failed (HTTP %s)", resp.status)
                    return [], []
                content = await resp.read()
            tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            tmp.write(content)
            tmp.close()
            return [tmp.name], [msg_type]
        except Exception:
            logger.warning("[waha] media download failed", exc_info=True)
            return [], []

    async def _observe_bridge_group_message(self, data: Dict[str, Any]) -> None:
        """Observe one skipped group message: transcript-only, media downloaded to the
        local cache so the model can inspect it on demand at trigger time."""
        msg_type = self._classify_waha_message(data)
        body = str(data.get("body") or "").strip()
        try:
            cached_urls, _media_types = await self._collect_waha_media(data, msg_type)
        except Exception:
            logger.warning("[waha] observe media download failed", exc_info=True)
            cached_urls = []
        refs = self._whatsapp_observe_media_references(cached_urls, msg_type)
        if refs:
            ref_block = "\n".join(refs)
            body = f"{body}\n{ref_block}" if body else ref_block
        self._observe_unmentioned_group_message(data, msg_type, body)

    # ------------------------------------------------------------------ outbound
    @staticmethod
    def _serialized_message_id(body: Any, chat_id: str) -> str:
        """Baileys serialized id (``<fromMe>_<chatJid>_<id>``) from a WAHA send response.

        NOWEB sendText returns ``{"key": {"remoteJid", "fromMe", "id"}}`` with no
        top-level ``id``; WAHA's edit endpoint rejects bare ids (HTTP 500 "Message id be
        in format false_...@c.us_..."), so edits need the serialized form. ``remoteJid``
        (``@s.whatsapp.net`` for DMs) is the Baileys store key and is preferred over the
        ``@c.us`` chat id when present."""
        key = (body or {}).get("key") if isinstance(body, dict) else None
        mid = str((key or {}).get("id") or "") if isinstance(body, dict) else ""
        if not mid:
            return ""
        if "_" in mid:
            return mid  # already serialized (defensive)
        remote = str((key or {}).get("remoteJid") or chat_id)
        from_me = bool((key or {}).get("fromMe", True))
        return f"{'true' if from_me else 'false'}_{remote}_{mid}"

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Format markdown for WhatsApp, chunk preserving code blocks, send sequentially."""
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        chat_id = _waha_chat_id(chat_id)
        try:
            chunks = self.truncate_message(self.format_message(content), self._outgoing_chunk_limit())
            sent_ids: list[str] = []
            last_id = None
            for idx, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {"session": self._session, "chatId": chat_id, "text": chunk}
                if reply_to and idx == 0:
                    payload["reply_to"] = reply_to
                status, body = await self._request("POST", "/api/sendText", payload, timeout=30)
                if status not in (200, 201):
                    return SendResult(success=False, error=f"WAHA sendText HTTP {status}: {body}")
                last_id = self._serialized_message_id(body, chat_id) or None
                if last_id:
                    sent_ids.append(last_id)
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)
            return SendResult(success=True, message_id=last_id,
                              continuation_message_ids=tuple(sent_ids[:-1]),
                              raw_response={"message_ids": sent_ids})
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> Any:
        """Edit via WAHA's chat-message endpoint (NOWEB/WEBJS/GOWS all support it)."""
        chat_id = _waha_chat_id(chat_id)
        mid = str(message_id or "")
        if "_" not in mid:
            mid = f"true_{chat_id}_{mid}"  # bare id → serialize as own message
        try:
            path = f"/api/{self._session}/chats/{chat_id}/messages/{mid}"
            status, body = await self._request(
                "PUT", path, {"text": self.format_message(content)}, timeout=30)
            if status not in (200, 201):
                return SendResult(success=False, error=f"WAHA edit HTTP {status}: {body}")
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        try:
            await self._request("POST", "/api/startTyping",
                                {"session": self._session, "chatId": _waha_chat_id(chat_id)}, timeout=10)
        except Exception:
            logger.debug("[waha] startTyping failed", exc_info=True)

    async def _stop_typing(self, chat_id: str) -> None:
        try:
            await self._request("POST", "/api/stopTyping",
                                {"session": self._session, "chatId": _waha_chat_id(chat_id)}, timeout=10)
        except Exception:
            logger.debug("[waha] stopTyping failed", exc_info=True)

    async def _send_media(self, chat_id: str, path_or_url: str, kind: str,
                          caption: Optional[str] = None, file_name: Optional[str] = None) -> Any:
        from gateway.platforms.base import SendResult
        chat_id = _waha_chat_id(chat_id)
        endpoint = {"image": "sendImage", "video": "sendVideo", "voice": "sendVoice",
                    "audio": "sendAudio", "document": "sendFile"}.get(kind, "sendFile")
        payload: Dict[str, Any] = {"session": self._session, "chatId": chat_id}
        if caption:
            payload["caption"] = self.format_message(caption)
        if path_or_url.startswith(("http://", "https://")):
            payload["file"] = {"url": path_or_url}
            if kind == "document" and file_name:
                payload["file"]["filename"] = file_name
        else:
            data = Path(path_or_url).read_bytes()
            mime = mimetypes.guess_type(path_or_url)[0] or "application/octet-stream"
            payload["file"] = {"mimetype": mime, "data": base64.b64encode(data).decode("ascii"),
                               "filename": file_name or Path(path_or_url).name}
        status, body = await self._request("POST", f"/api/{endpoint}", payload, timeout=120)
        if status not in (200, 201):
            return SendResult(success=False, error=f"WAHA {endpoint} HTTP {status}: {body}")
        return SendResult(success=True, message_id=str((body or {}).get("id") or "") or None)

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Any:
        return await self._send_media(chat_id, image_url, "image", caption=caption)

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, **kwargs) -> Any:
        return await self._send_media(chat_id, image_path, "image", caption=caption)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> Any:
        return await self._send_media(chat_id, video_path, "video", caption=caption)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> Any:
        return await self._send_media(chat_id, file_path, "document", caption=caption, file_name=file_name)

    async def send_voice(self, chat_id: str, voice_path: str, reply_to: Optional[str] = None, **kwargs) -> Any:
        return await self._send_media(chat_id, voice_path, "voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        is_group = chat_id.endswith("@g.us")
        return {"name": chat_id, "type": "group" if is_group else "dm", "chat_id": chat_id}

    async def _send_read_receipt(self, chat_id: str, message_id: str) -> None:
        if not self._send_read_receipts or not message_id:
            return
        try:
            await self._request("POST", "/api/sendSeen",
                                {"session": self._session, "chatId": _waha_chat_id(chat_id),
                                 "messageId": message_id}, timeout=10)
        except Exception:
            logger.debug("[waha] sendSeen failed", exc_info=True)


# ------------------------------------------------------------------ plugin glue

def check_requirements() -> bool:
    """PASSIVE probe: aiohttp importable (core dep) — WAHA itself is external."""
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(str(extra.get("base_url") or _wenv("WAHA_BASE_URL", "")).strip())


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env so env-only setups show in gateway status."""
    base_url = _wenv("WAHA_BASE_URL", "").strip()
    if not base_url:
        return None
    seed: dict = {"base_url": base_url}
    if api_key := os.getenv("WAHA_API_KEY", "").strip():
        seed["api_key"] = api_key
    if session := _wenv("WAHA_SESSION", "").strip():
        seed["session"] = session
    if port := _wenv("WAHA_WEBHOOK_PORT", "").strip():
        try:
            seed["webhook_port"] = int(port)
        except ValueError:
            pass
    if secret := os.getenv("WAHA_WEBHOOK_SECRET", "").strip():
        seed["webhook_secret"] = secret
    if home := _wenv("WAHA_HOME_CHANNEL", "").strip():
        seed["home_channel"] = {"chat_id": home, "name": "WAHA Home"}
    return seed


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None,
                           force_document=False, caption=None):
    """Out-of-process cron delivery through the WAHA REST API."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed"}
    try:
        extra = (getattr(pconfig, "extra", {}) or {})
        base_url = str(extra.get("base_url") or _wenv("WAHA_BASE_URL", "")).rstrip("/")
        api_key = str(extra.get("api_key") or os.getenv("WAHA_API_KEY", "")).strip()
        session = str(extra.get("session") or _wenv("WAHA_SESSION", "default") or "default")
        if not base_url:
            return {"error": "WAHA_BASE_URL not configured"}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Api-Key"] = api_key
        chat_id = _waha_chat_id(chat_id)
        media = media_files or []
        media_caption = caption if (caption and len(media) == 1) else None
        text = message or ""
        try:
            text = WhatsAppBehaviorMixin.format_message(
                object.__new__(WhatsAppBehaviorMixin), text)
        except Exception:  # noqa: BLE001 - delivery must never fail
            logger.warning("[waha] format_message failed in _standalone_send; sending raw", exc_info=True)
        last_id = None
        async with aiohttp.ClientSession(headers=headers) as http:
            async def _post(path, payload, total=30):
                async with http.post(f"{base_url}{path}", json=payload,
                                     timeout=aiohttp.ClientTimeout(total=total)) as resp:
                    if resp.status in (200, 201):
                        body = await resp.json(content_type=None) or {}
                        key = body.get("key") if isinstance(body, dict) else None
                        return (str((key or {}).get("id") or body.get("id") or "") or None), None
                    return None, {"error": f"WAHA {path} HTTP {resp.status}: {await resp.text()}"}
            if text.strip() and not media_caption:
                last_id, err = await _post("/api/sendText",
                                           {"session": session, "chatId": chat_id, "text": text})
                if err:
                    return err
            for media_path, is_voice in media:
                if not os.path.exists(media_path):
                    if media_caption:
                        await _post("/api/sendText",
                                    {"session": session, "chatId": chat_id, "text": media_caption})
                    return {"error": f"WAHA media file not found: {media_path}"}
                kind = "voice" if is_voice else "document"
                mime = mimetypes.guess_type(media_path)[0] or "application/octet-stream"
                payload = {"session": session, "chatId": chat_id,
                           "file": {"mimetype": mime,
                                    "data": base64.b64encode(Path(media_path).read_bytes()).decode("ascii"),
                                    "filename": Path(media_path).name}}
                if media_caption:
                    payload["caption"] = media_caption
                endpoint = "/api/sendVoice" if is_voice else "/api/sendFile"
                last_id, err = await _post(endpoint, payload, total=120)
                if err:
                    return err
        return {"success": True, "platform": "waha", "chat_id": chat_id, "message_id": last_id}
    except Exception as e:
        return {"error": f"WAHA send failed: {e}"}


def register(ctx) -> None:
    ctx.register_platform(
        name="waha", label="WhatsApp via WAHA",
        adapter_factory=lambda cfg: WahaAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config, is_connected=is_connected,
        required_env=["WAHA_BASE_URL"],
        setup_fn=None,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="WAHA_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="WAHA_ALLOWED_USERS",
        max_message_length=65536,
        emoji="💬",
        platform_hint=(
            "You are chatting via WhatsApp. Write standard markdown freely "
            "(**bold**, *italic*, headings, lists, code fences) — the gateway "
            "converts it to WhatsApp formatting automatically. No tables; prefer "
            "bullets or labeled lines. Keep replies concise and mobile-friendly."))
