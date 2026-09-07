"""SimpleX Chat adapter: persistent WebSocket to a simplex-chat daemon (``simplex-chat -p 5225``,
or ``docker run -p 5225:5225 simplexchat/simplex-chat-cli -p 5225``); JSON commands out, events in.

Env: SIMPLEX_WS_URL (required; default ws://127.0.0.1:5225) · SIMPLEX_ALLOWED_USERS (numeric
contactIds — stable across renames, see ``/contacts`` — or display names) · SIMPLEX_ALLOW_ALL_USERS ·
SIMPLEX_AUTO_ACCEPT ('false' disables contact-request auto-accept; default true) ·
SIMPLEX_GROUP_ALLOWED (group IDs or '*'; omit to ignore groups) · SIMPLEX_HOME_CHANNEL[_NAME] ·
HERMES_SIMPLEX_TEXT_BATCH_DELAY (quiet seconds, default 0.8, merging rapid-fire inbound text).
``websockets`` is imported lazily so the plugin stays discoverable when the package is missing.
"""

import asyncio
import base64
import contextlib
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from urllib.parse import unquote

from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult, cache_image_from_url

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 8000  # SimpleX has no hard limit; chunk for sanity
WS_RETRY_DELAY_INITIAL = 2.0
WS_RETRY_DELAY_MAX = 60.0
HEALTH_CHECK_INTERVAL = 30.0
HEALTH_CHECK_STALE_THRESHOLD = 300.0
_CORR_PREFIX = "hermes-"  # marks requests we sent so our own echoes can be ignored
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".opus"}
_VOICE_TAG_EXTS = {".ogg", ".mp3", ".wav", ".m4a", ".opus"}  # MEDIA: tags sent as voice notes
_TEXT_BEARING_TYPES = ("text", "file", "image", "voice", "link", "video")
_THUMB_URI_PREFIX = "data:image/jpg;base64,"
_MEDIA_KIND_PRECEDENCE = (("audio/", MessageType.VOICE), ("image/", MessageType.PHOTO))  # first match wins


def _parse_comma_list(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _redact_id(contact_id: str) -> str:
    if not contact_id:
        return "<none>"
    s = str(contact_id)
    return s if len(s) <= 4 else s[:2] + "**" + s[-2:]


def _is_image_ext(ext: str) -> bool:
    return ext.lower() in _IMAGE_EXTS


def _is_audio_ext(ext: str) -> bool:
    return ext.lower() in _AUDIO_EXTS


def _mime_for_ext(ext: str) -> str:
    if _is_image_ext(ext):
        return f"image/{ext.lstrip('.')}"
    if _is_audio_ext(ext):
        return f"audio/{ext.lstrip('.')}"
    return "application/octet-stream"


def _display_name(obj: dict, profile_key: str, default: str = "") -> str:
    """``localDisplayName`` falling back to the nested profile's ``displayName``."""
    return obj.get("localDisplayName", "") or (obj.get(profile_key, {}) or {}).get("displayName", default)


def _send_cmd(chat_id: str, items: list) -> str:
    """Structured ``/_send <target> json …`` addressing *chat_id* by ID. The bare ``@name`` / ``#[id]``
    syntax is a display-name lookup that silently drops unresolved names; json.dumps escapes text."""
    target = f"#{chat_id[6:]}" if chat_id.startswith("group:") else f"@{chat_id}"
    return f"/_send {target} json {json.dumps(items)}"


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class SimplexAdapter(BasePlatformAdapter):
    """SimpleX Chat adapter using the simplex-chat daemon WebSocket API."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig, **kwargs):
        super().__init__(config=config, platform=Platform("simplex"))
        extra = getattr(config, "extra", {}) or {}
        self.ws_url = extra.get("ws_url", "ws://127.0.0.1:5225").rstrip("/")
        # Auto-accept is on by default; env wins over the ``_env_enablement`` seed.
        env_auto = _get_scoped_secret("SIMPLEX_AUTO_ACCEPT")
        if env_auto is not None:
            self.auto_accept = env_auto.strip().lower() not in {"0", "false", "no", ""}
        else:
            self.auto_accept = bool(extra.get("auto_accept", True))
        # Without SIMPLEX_GROUP_ALLOWED group messages are ignored (safer default); ``*`` = any group.
        group_allowed_str = _get_scoped_secret("SIMPLEX_GROUP_ALLOWED", "") or extra.get("group_allowed", "")
        # Parse allowlists — group policy is derived from presence of group allowlist Scoped reads (#93522):
        # allowlists are per-profile authorization config; raw os.getenv misses secondary profiles' .env
        # values and leaks the default profile's list into them.
        self.group_allow_from = set(_parse_comma_list(group_allowed_str))
        self._ws = None  # websockets connection
        self._ws_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_ws_activity = 0.0
        self._pending_corr_ids: set = set()  # cosmetic echo filter: corrIds we minted, bounded
        self._max_pending_corr = 200
        self._pending_file_transfers: Dict[int, dict] = {}  # awaiting rcvFileComplete, by fileId
        self._pending_responses: Dict[str, asyncio.Future] = {}  # awaited command replies
        self._corr_counter = 0
        # Text batching state consumed by BasePlatformAdapter._enqueue_text_event.
        self._text_batch_delay = float(os.getenv("HERMES_SIMPLEX_TEXT_BATCH_DELAY", "0.8"))
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        logger.info(
            "SimpleX adapter initialized: url=%s auto_accept=%s groups=%s",
            self.ws_url, self.auto_accept, "enabled" if self.group_allow_from else "disabled")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        try:
            import websockets as _wsclient
        except ImportError:
            logger.error("SimpleX: 'websockets' package not installed. Run: pip install websockets")
            return False
        if not self.ws_url:
            logger.error("SimpleX: SIMPLEX_WS_URL is required")
            return False
        try:  # quick connectivity check — open and immediately close
            async with _wsclient.connect(self.ws_url, open_timeout=10):
                pass
        except Exception as e:
            logger.error("SimpleX: cannot reach daemon at %s: %s", self.ws_url, e)
            return False
        self._running = True
        self._last_ws_activity = time.time()
        self._ws_task = asyncio.create_task(self._ws_listener())
        self._health_task = asyncio.create_task(self._health_monitor())
        self._mark_connected()
        logger.info("SimpleX: connected to %s", self.ws_url)
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        self._running = False
        await _cancel_task(self._ws_task)
        await _cancel_task(self._health_task)
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        for pending in (self._pending_text_batch_tasks.values(), self._pending_responses.values()):
            for item in list(pending):
                if not item.done():
                    item.cancel()
        self._pending_text_batch_tasks.clear()
        self._pending_text_batches.clear()
        self._pending_responses.clear()
        self._mark_disconnected()
        logger.info("SimpleX: disconnected")

    async def _ws_listener(self) -> None:
        import websockets as _wsclient
        from websockets.exceptions import ConnectionClosed
        backoff = WS_RETRY_DELAY_INITIAL
        while self._running:
            try:
                logger.debug("SimpleX WS: connecting to %s", self.ws_url)
                async with _wsclient.connect(self.ws_url, ping_interval=20, ping_timeout=20, close_timeout=10) as ws:
                    self._ws = ws
                    backoff = WS_RETRY_DELAY_INITIAL
                    self._last_ws_activity = time.time()
                    logger.info("SimpleX WS: connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_ws_activity = time.time()
                        try:
                            await self._handle_event(json.loads(raw))
                        except json.JSONDecodeError:
                            logger.debug("SimpleX WS: invalid JSON: %.100s", raw)
                        except Exception:
                            logger.exception("SimpleX WS: error handling event")
            except asyncio.CancelledError:
                break
            except ConnectionClosed as e:
                if self._running:
                    logger.warning("SimpleX WS: connection closed: %s (reconnecting in %.0fs)", e, backoff)
            except Exception as e:
                if self._running:
                    logger.warning("SimpleX WS: unexpected error: %s (reconnecting in %.0fs)", e, backoff)
            finally:
                self._ws = None
            if self._running:
                await asyncio.sleep(backoff + backoff * 0.2 * random.random())
                backoff = min(backoff * 2, WS_RETRY_DELAY_MAX)

    async def _health_monitor(self) -> None:
        """Log (never reconnect on) WebSocket idleness: simplex-chat legitimately stays
        application-silent for long periods and the client already sends protocol pings."""
        while self._running:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            if not self._running:
                break
            elapsed = time.time() - self._last_ws_activity
            if elapsed > HEALTH_CHECK_STALE_THRESHOLD:
                logger.debug("SimpleX: WS application-idle for %.0fs", elapsed)

    async def _handle_event(self, event: dict) -> None:
        # Usually {"corrId": ..., "resp": {"type": ...}}, but some daemons put the
        # response fields at top level — normalize both.
        resp = event.get("resp") if isinstance(event.get("resp"), dict) else event
        corr_id = event.get("corrId")
        if corr_id and corr_id in self._pending_responses:
            fut = self._pending_responses.pop(corr_id)
            if not fut.done():
                fut.set_result(resp)
            return
        if corr_id and isinstance(corr_id, str) and corr_id.startswith(_CORR_PREFIX):
            self._pending_corr_ids.discard(corr_id)  # our own un-awaited echo
            return
        resp_type = resp.get("type") or event.get("type", "")
        handler = self._EVENT_HANDLERS.get(resp_type)
        if handler is not None:
            await handler(self, resp)
        elif resp_type:
            logger.debug("SimpleX: unhandled event type: %s", resp_type)

    async def _on_contact_request(self, resp: dict) -> None:
        if not self.auto_accept:
            logger.debug("SimpleX: unhandled event type: %s", "contactRequest")
            return
        contact_req_id = (resp.get("contactRequest", {}) or {}).get("contactRequestId")
        if contact_req_id is not None:
            logger.info("SimpleX: auto-accepting contact request %s", _redact_id(str(contact_req_id)))
            await self._send_command(f"/accept {contact_req_id}")

    async def _on_rcv_file_descr_ready(self, resp: dict) -> None:
        """XFTP files fire this before newChatItems; start the download now, the chat item arrives later."""
        rcv_file = resp.get("rcvFileTransfer", {}) or {}
        if (file_id := rcv_file.get("fileId") if isinstance(rcv_file, dict) else None) is not None:
            logger.debug("SimpleX: rcvFileDescrReady for fileId=%s — sending /freceive", file_id)
            await self._send_fire_and_forget(f"/freceive {file_id}")

    async def _on_new_chat_items(self, resp: dict) -> None:
        chat_items = resp.get("chatItems", []) or []
        for item in chat_items if isinstance(chat_items, list) else [chat_items]:
            await self._safe_handle_chat_item(item, "SimpleX: error processing chat item")

    async def _on_new_chat_item(self, resp: dict) -> None:  # singular variant from some daemon versions
        await self._safe_handle_chat_item(resp, "SimpleX: error processing chat item")

    async def _on_rcv_file_complete(self, resp: dict) -> None:
        """Deliver a chat item deferred until its file transfer completed."""
        chat_item_data = (resp.get("chatItem", {}) or {}).get("chatItem", {}) or {}
        file_info = chat_item_data.get("file", {}) or {}
        file_id = file_info.get("fileId") if isinstance(file_info, dict) else None
        if file_id is None or file_id not in self._pending_file_transfers:
            return
        pending = self._pending_file_transfers.pop(file_id)
        file_source = file_info.get("fileSource", {}) or {}
        file_path = file_source.get("filePath") if isinstance(file_source, dict) else None
        if file_path:
            pending_item_data = pending.get("chatItem", {}) or {}
            pending_item_data.setdefault("file", {})["fileSource"] = {"filePath": file_path}
            pending["chatItem"] = pending_item_data
            await self._safe_handle_chat_item(pending, "SimpleX: error processing deferred file message")

    _EVENT_HANDLERS = {
        "contactRequest": _on_contact_request, "rcvFileDescrReady": _on_rcv_file_descr_ready,
        "newChatItems": _on_new_chat_items, "newChatItem": _on_new_chat_item, "rcvFileComplete": _on_rcv_file_complete,
    }

    async def _safe_handle_chat_item(self, item: dict, err_msg: str) -> None:
        try:
            await self._handle_chat_item(item)
        except Exception:
            logger.exception(err_msg)

    async def _handle_chat_item(self, chat_item: dict) -> None:
        chat_info = chat_item.get("chatInfo", {}) or {}
        chat_item_data = chat_item.get("chatItem", {}) or {}
        chat_type = chat_info.get("type", "")
        meta = chat_item_data.get("meta", {}) or {}
        content = chat_item_data.get("content", {}) or {}
        msg_content = content.get("msgContent", {}) or {}
        item_direction = chat_item_data.get("chatDir", {}) or {}
        direction_type = item_direction.get("type", "") if isinstance(item_direction, dict) else ""
        if direction_type in ("directSnd", "groupSnd"):  # our own messages
            return
        content_type = content.get("type", "") if isinstance(content, dict) else ""
        if content_type != "rcvMsgContent":
            return
        msg_type_str = msg_content.get("type", "") if isinstance(msg_content, dict) else ""
        text = msg_content.get("text", "") if msg_type_str in _TEXT_BEARING_TYPES else ""
        if not text and msg_type_str not in ("image", "file", "voice"):
            return
        is_group = chat_type == "group"
        if chat_type == "direct":
            contact = chat_info.get("contact", {}) or {}
            sender_id = chat_id = str(contact.get("contactId", ""))
            sender_name = chat_name = _display_name(contact, "profile")
        elif is_group:
            group_info = chat_info.get("groupInfo", {}) or {}
            group_id = str(group_info.get("groupId", ""))
            chat_id = f"group:{group_id}"
            member = item_direction.get("groupMember", {}) or {}
            sender_id = str(member.get("memberId", ""))
            sender_name = _display_name(member, "memberProfile")
            chat_name = _display_name(group_info, "groupProfile", chat_id)
            if not self.group_allow_from:
                logger.debug("SimpleX: ignoring group message (no SIMPLEX_GROUP_ALLOWED)")
                return
            if "*" not in self.group_allow_from and group_id not in self.group_allow_from:
                logger.debug("SimpleX: group %s not in allowlist", _redact_id(group_id))
                return
        else:
            logger.debug("SimpleX: unhandled chat type: %s", chat_type)
            return
        if not sender_id:
            logger.debug("SimpleX: ignoring message with no sender")
            return
        # Attachment: chatItem.chatItem.file (sibling of meta/content/chatDir).
        media_urls: List[str] = []
        media_types: List[str] = []
        file_info = chat_item_data.get("file")
        if file_info and isinstance(file_info, dict):
            file_source = file_info.get("fileSource", {}) or {}
            file_path = file_source.get("filePath") if isinstance(file_source, dict) else None
            file_id = file_info.get("fileId")
            ext = Path(file_path).suffix.lower() if file_path else ""
            if not ext and file_info.get("fileName", ""):
                ext = Path(file_info["fileName"]).suffix.lower()
            # Voice notes typically arrive before the file finishes downloading; defer until
            # rcvFileComplete. /freceive gets no corrId reply, so awaiting one would block the loop.
            if not file_path and _is_audio_ext(ext) and file_id is not None:
                logger.info("SimpleX: voice file %d not yet received, accepting transfer", file_id)
                self._pending_file_transfers[file_id] = chat_item
                await self._send_fire_and_forget(f"/freceive {file_id}")
                return
            if file_path:
                media_urls.append(file_path)
                media_types.append(_mime_for_ext(ext))
        source = self.build_source(
            chat_id=chat_id, chat_name=chat_name, chat_type="group" if is_group else "dm",
            user_id=sender_id, user_name=sender_name or sender_id)
        # Non-image/non-audio files are DOCUMENT so run.py's document-context injection surfaces them.
        msg_type = MessageType.TEXT
        if media_types:
            msg_type = next((t for prefix, t in _MEDIA_KIND_PRECEDENCE if any(mt.startswith(prefix) for mt in media_types)),
                            MessageType.DOCUMENT)
        ts_str = meta.get("itemTs") or meta.get("createdAt", "")
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(tz=timezone.utc)
        except (ValueError, AttributeError):
            timestamp = datetime.now(tz=timezone.utc)
        msg_event = MessageEvent(
            source=source, text=text or "", message_type=msg_type, media_urls=media_urls,
            media_types=media_types, timestamp=timestamp, raw_message=chat_item)
        logger.debug("SimpleX: message from %s in %s: %s", _redact_id(sender_id), chat_id[:20], (text or "")[:50])
        if msg_type == MessageType.TEXT and text:  # batch rapid-fire text into one combined message
            self._enqueue_text_event(msg_event)
        else:
            await self.handle_message(msg_event)

    # Text batching: enqueue lives on BasePlatformAdapter.

    def _text_batch_key(self, event: MessageEvent) -> str:
        return f"{event.source.platform.value}:{event.source.chat_id}"

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text."""
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(self._text_batch_delay)
            event = self._pending_text_batches.pop(key, None)
            if event:
                logger.info("[SimpleX] Flushing text batch %s (%d chars)", key, len(event.text or ""))
                await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    def _make_corr_id(self) -> str:
        """Mint a correlation ID and remember it for echo-filtering; the set is bounded by
        ``_max_pending_corr`` (overflow evicted in one sweep)."""
        self._corr_counter += 1
        corr_id = f"{_CORR_PREFIX}{self._corr_counter}-{int(time.time() * 1000)}"
        self._pending_corr_ids.add(corr_id)
        for _ in range(len(self._pending_corr_ids) - self._max_pending_corr):
            self._pending_corr_ids.pop()
        return corr_id

    async def _send_ws(self, payload: dict) -> None:
        """Fire-and-forget JSON write; drops cleanly when the WS is missing/closed."""
        if not self._ws:
            logger.debug("SimpleX: WS send dropped (not connected)")
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.warning("SimpleX: WS send error: %s", e)

    async def _send_command(self, command: str, timeout: float = 30.0) -> Optional[dict]:
        ws = self._ws
        if not ws:
            logger.warning("SimpleX: command sent but WebSocket not connected")
            return None

        corr_id = self._make_corr_id()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[corr_id] = fut
        try:
            await ws.send(json.dumps({"corrId": corr_id, "cmd": command}))
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("SimpleX: command timed out: %s", command[:50])
        except Exception as e:
            logger.warning("SimpleX: command failed: %s — %s", command[:50], e)
        self._pending_responses.pop(corr_id, None)
        return None

    async def _send_fire_and_forget(self, command: str) -> None:
        """Send a command the daemon never replies to with a corrId (e.g. ``/freceive``)."""
        await self._send_ws({"corrId": self._make_corr_id(), "cmd": command})

    async def _send_items(self, chat_id: str, items: list, error: str) -> SendResult:
        """Send a structured ``/_send`` payload and await the reply."""
        result = await self._send_command(_send_cmd(chat_id, items))
        return SendResult(success=True) if result is not None else SendResult(success=False, error=error)

    async def send(
        self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send text; ``MEDIA:<path>`` tags (TTS / audio tools) are stripped and sent as native voice
        notes or documents. The text send is fire-and-forget: the daemon doesn't always return a corrId
        reply for chat commands, and waiting would serialise all outbound traffic behind a 30s timeout."""
        media_paths = re.findall(r"MEDIA:(\S+)", content)
        if media_paths:
            content = re.sub(r"MEDIA:\S+", "", content).strip()
        if content:
            cmd_str = _send_cmd(chat_id, [{"msgContent": {"type": "text", "text": content}}])
            await self._send_ws({"corrId": self._make_corr_id(), "cmd": cmd_str})
        for path in media_paths:
            if os.path.splitext(path)[1].lower() in _VOICE_TAG_EXTS:
                media_result = await self.send_voice(chat_id, path)
            else:
                media_result = await self.send_document(chat_id, path)
            if not media_result.success:
                return media_result
        return SendResult(success=True)

    async def list_channels(self) -> Optional[List[Dict[str, Any]]]:
        """Enumerate contacts and allowed groups for the channel directory.

        Returns ``None`` (not ``[]``) when the WebSocket is down or the daemon is unresponsive so
        the directory falls back to session-history discovery instead of wiping known targets.
        Entry ``id`` values match the send targets: display name for DMs, ``group:<id>`` for groups.
        """
        if not self._ws:
            return None
        resp = await self._send_command("/contacts", timeout=10.0)
        if resp is None:
            return None
        channels: List[Dict[str, Any]] = []
        for contact in resp.get("contacts") or []:
            if not isinstance(contact, dict):
                continue
            contact_id = contact.get("contactId")
            name = _display_name(contact, "profile")
            if contact_id is None and not name:
                continue
            # Display name is what the DM send path addresses; fall back to contactId.
            channels.append({"id": str(name or contact_id), "name": str(name or contact_id), "type": "dm"})
        resp = await self._send_command("/groups", timeout=10.0)
        for group in (resp.get("groups") or []) if resp is not None else []:
            if isinstance(group, list) and group:  # groupInfo dict or [groupInfo, groupSummary] pair
                group = group[0]
            if not isinstance(group, dict) or group.get("groupId") is None:
                continue
            group_id = group["groupId"]
            name = _display_name(group, "groupProfile") or str(group_id)
            channels.append({"id": f"group:{group_id}", "name": str(name), "type": "group"})
        return channels

    @staticmethod
    def _prepare_image(file_path: str) -> tuple[str, str]:
        """Ensure *file_path* is PNG/JPEG and return ``(png_path, thumb_data_uri)``. SimpleX clients
        can't show WebP etc. inline, so convert to PNG when needed and build a 128px JPEG thumbnail for
        the ``image`` field. Uses Pillow when available, else ImageMagick ``convert``."""
        import subprocess
        import tempfile
        p = Path(file_path)
        needs_png = p.suffix.lower() not in (".png", ".jpg", ".jpeg")
        png_path = str(p.with_suffix(".png")) if needs_png else file_path
        thumb_uri = ""
        try:
            from PIL import Image
            import io
            img = Image.open(file_path)
            if needs_png:
                img.save(png_path, "PNG")
            thumb = img.copy()
            thumb.thumbnail((128, 128))
            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=70)
            thumb_uri = _THUMB_URI_PREFIX + base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            try:
                if needs_png:
                    subprocess.run(["convert", file_path, png_path],
                                   check=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(["convert", file_path, "-resize", "128x128", "-quality", "70", tmp_path],
                               check=True, capture_output=True, timeout=30, stdin=subprocess.DEVNULL)
                with open(tmp_path, "rb") as f:
                    thumb_uri = _THUMB_URI_PREFIX + base64.b64encode(f.read()).decode()
                os.remove(tmp_path)
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                logger.warning("SimpleX: image conversion unavailable: %s", exc)
        return png_path, thumb_uri

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None, **kwargs) -> SendResult:
        """Send an image. Supports ``file://`` URLs and ``http(s)://`` URLs."""
        if image_url.startswith("file://"):
            file_path = unquote(image_url[7:])
        else:
            try:
                file_path = await cache_image_from_url(image_url)
            except Exception as e:
                logger.warning("SimpleX: failed to download image: %s", e)
                return SendResult(success=False, error=str(e))
        if not file_path or not Path(file_path).exists():
            return SendResult(success=False, error="Image file not found")
        png_path, thumb_uri = self._prepare_image(file_path)
        # /_send addresses by numeric ID; /f only accepts display names.
        item = {"filePath": png_path, "msgContent": {"type": "image", "image": thumb_uri, "text": caption or ""}}
        return await self._send_items(chat_id, [item], "Failed to send image")

    async def send_image_file(self, chat_id: str, image_path: str, caption: Optional[str] = None,
                              reply_to: Optional[str] = None, **kwargs) -> SendResult:
        return await self.send_image(chat_id, f"file://{image_path}", caption=caption, **kwargs)

    async def send_video(self, chat_id: str, video_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, **kwargs) -> SendResult:
        """Videos go as file attachments."""
        return await self.send_document(chat_id, video_path, caption=caption)

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            filename: Optional[str] = None, **kwargs) -> SendResult:
        if not Path(file_path).exists():
            return SendResult(success=False, error="File not found")
        item = {"filePath": file_path, "msgContent": {"type": "file", "text": caption or ""}}
        return await self._send_items(chat_id, [item], "Failed to send document")

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, duration: int = 0, **kwargs) -> SendResult:
        """Send an audio file as an inline SimpleX voice note (``msgContent.type == "voice"``)."""
        if not Path(audio_path).exists():
            return SendResult(success=False, error="Voice file not found")
        item = {"msgContent": {"type": "voice", "text": caption or "", "duration": duration},
                "fileSource": {"filePath": audio_path}}
        return await self._send_items(chat_id, [item], "Failed to send voice message")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """SimpleX has no typing-indicator API — no-op."""

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        is_group = chat_id.startswith("group:")
        return {"chat_id": chat_id, "type": "group" if is_group else "dm", "name": chat_id[6:] if is_group else chat_id}


def check_requirements() -> bool:
    """Plugin gate: require SIMPLEX_WS_URL AND the websockets package."""
    if not _get_scoped_secret("SIMPLEX_WS_URL"):
        return False
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_get_scoped_secret("SIMPLEX_WS_URL") or extra.get("ws_url", ""))


def is_connected(config) -> bool:
    """Configured (env or config.yaml) ⇒ shown as connected in status."""
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env BEFORE adapter construction so ``gateway status``
    reflects env-only setups. ``None`` when not minimally configured; ``home_channel`` becomes
    a ``HomeChannel`` via the core hook."""
    ws_url = _get_scoped_secret("SIMPLEX_WS_URL", "").strip()
    if not ws_url:
        return None
    seed: dict = {"ws_url": ws_url}
    if auto_accept := _get_scoped_secret("SIMPLEX_AUTO_ACCEPT", "").strip().lower():
        seed["auto_accept"] = auto_accept not in {"0", "false", "no"}
    if group_allowed := _get_scoped_secret("SIMPLEX_GROUP_ALLOWED", "").strip():
        seed["group_allowed"] = group_allowed
    if home := _get_scoped_secret("SIMPLEX_HOME_CHANNEL", "").strip():
        seed["home_channel"] = {"chat_id": home, "name": _get_scoped_secret("SIMPLEX_HOME_CHANNEL_NAME", "").strip() or home}
    return seed


async def _standalone_send(
    pconfig, chat_id: str, message: str, *,
    thread_id: Optional[str] = None, media_files: Optional[List[str]] = None, force_document: bool = False,
) -> Dict[str, Any]:
    """Ephemeral WebSocket send for ``tools/send_message_tool`` when the gateway runner is not in
    this process (``hermes cron``). ``thread_id``/``force_document`` are signature parity only;
    ``media_files`` is accepted but only the text body is delivered — SimpleX file transfers need
    the daemon's filesystem-backed flow, which an ephemeral connection cannot drive safely."""
    try:
        import websockets as _wsclient
    except ImportError:
        return {"error": "websockets not installed. Run: pip install websockets"}
    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = _get_scoped_secret("SIMPLEX_WS_URL") or extra.get("ws_url", "ws://127.0.0.1:5225")
    if not ws_url:
        return {"error": "SimpleX standalone send: SIMPLEX_WS_URL is required"}
    try:
        payload = {
            "corrId": f"{_CORR_PREFIX}snd-{int(time.time() * 1000)}",
            "cmd": _send_cmd(chat_id, [{"msgContent": {"type": "text", "text": message}}])}
        async with _wsclient.connect(ws_url, open_timeout=10, close_timeout=5) as ws:
            await ws.send(json.dumps(payload))
            await asyncio.sleep(0.5)  # let the daemon process the command before closing
        return {"success": True, "platform": "simplex", "chat_id": chat_id}
    except Exception as e:
        return {"error": f"SimpleX send failed: {e}"}


_SETUP_PROMPTS = (
    ("SIMPLEX_WS_URL", "Daemon WebSocket URL (default ws://127.0.0.1:5225)"),
    ("SIMPLEX_ALLOWED_USERS", "Allowed contactIds or display names (comma-separated; blank=skip)"),
    ("SIMPLEX_GROUP_ALLOWED", "Allowed group IDs (comma-separated, or '*' for any; blank=disable groups)"),
    ("SIMPLEX_AUTO_ACCEPT", "Auto-accept incoming contact requests? (true/false, default true)"),
    ("SIMPLEX_HOME_CHANNEL", "Home channel contact/group ID (or empty)"))


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup gateway`` → SimpleX; writes ``~/.hermes/.env``."""
    print(
        "\nSimpleX Chat setup\n------------------\nRequirements:\n"
        "  1. simplex-chat daemon running (e.g. `simplex-chat -p 5225`).\n"
        "  2. Python package `websockets` installed (`pip install websockets`).\n")
    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        print("hermes_cli.config not available; set SIMPLEX_* vars manually in ~/.hermes/.env")
        return

    for var, prompt in _SETUP_PROMPTS:
        existing = get_env_value(var) if callable(get_env_value) else None
        suffix = " [keep current]" if existing else ""
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            continue
        if value:
            save_env_value(var, value)
    print("Done. Make sure the simplex-chat daemon is running before starting the gateway.")


def register(ctx) -> None:
    ctx.register_platform(
        name="simplex", label="SimpleX Chat", adapter_factory=lambda cfg: SimplexAdapter(cfg),
        check_fn=check_requirements, validate_config=validate_config, is_connected=is_connected,
        required_env=["SIMPLEX_WS_URL"],
        install_hint=("pip install websockets   # SimpleX adapter requires the websockets package"),
        setup_fn=interactive_setup, env_enablement_fn=_env_enablement, cron_deliver_env_var="SIMPLEX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send, allowed_users_env="SIMPLEX_ALLOWED_USERS",
        allow_all_env="SIMPLEX_ALLOW_ALL_USERS", max_message_length=MAX_MESSAGE_LENGTH, emoji="🔒",
        pii_safe=True,  # SimpleX uses opaque contact IDs only — nothing to redact
        allow_update_command=True,
        platform_hint=(
            "You are chatting via SimpleX Chat, a private decentralised "
            "messenger. Contacts are identified by opaque internal IDs, "
            "not phone numbers or usernames. SimpleX supports standard "
            "markdown formatting. There is no typing indicator and no "
            "hard message length limit, but keep responses conversational. "
            "You can attach native images, voice notes, and arbitrary "
            "files; the adapter handles MEDIA:<path> tags by sending them "
            "as inline voice notes (audio extensions) or documents."))
