"""
VK Messenger Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that connects to a VK community via
Long Poll API and relays messages to/from the Hermes agent.

Uses httpx for async HTTP.
"""

import asyncio
import json
import logging
import os
import random
import time
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from the Hermes core — lazy to avoid import errors at discovery
# ---------------------------------------------------------------------------

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.config import Platform

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VK_API_BASE = "https://api.vk.com/method"
VK_DEFAULT_VERSION = "5.199"
VK_POLL_INTERVAL = 5  # Rest API poll interval in seconds

# VK event types (group Long Poll)
EVENT_MESSAGE_NEW = 4
EVENT_MESSAGE_FLAGS_SET = 0
EVENT_MESSAGE_FLAGS_CLEAR = 1
EVENT_USER_TYPING = 8
EVENT_USER_TYPING_AUDIO = 9

# Message flags
FLAG_IMPORTANT = 1
FLAG_UNREAD = 2
FLAG_OUTBOX = 4
FLAG_REPLIED = 8
FLAG_CHAT = 256
FLAG_FRIENDS = 512
FLAG_SPAM = 1024
FLAG_DELETED = 2048
FLAG_FIXED = 4096
FLAG_MEDIA = 8192

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_vk_mention(text: str) -> str:
    """Strip VK mention prefix from inline button clicks in group chats.

    When a user clicks an inline keyboard button in a VK multi-chat, VK
    prepends an @mention of the bot to the button label.  E.g. the label
    '/approve' arrives as '[club239455766|@club239455766] /approve',
    which breaks the gateway's get_command() parsing.

    Returns the text without the leading VK mention, if present.
    """
    import re
    match = re.match(r'^\[[^\]]+\]\s*(.*)', text)
    if match:
        return match.group(1)
    return text


def _vk_format_text(text: str) -> str:
    """Strip Markdown formatting for VK (which doesn't render it for bots).

    VK community bots show **bold** and *italic* literally, not rendered.
    Clean formatting and use plain text structure instead.
    """
    import re

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', '').strip(), text)

    # Remove inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Convert Markdown links [text](url) → plain text with URL
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)

    # Remove **bold** markers
    text = text.replace('**', '')

    # Remove *italic* markers
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)(?<!\*)\*(?!\*)', r'\1', text)

    # Remove ~~strikethrough~~ markers
    text = text.replace('~~', '')

    # Convert headers to ALL CAPS
    text = re.sub(r'^#{1,6}\s+(.+)$', lambda m: m.group(1).upper(), text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Convert bullet lists: "- /command desc" → "  /command — desc"
    text = re.sub(r'^[-*]\s+(/\w[\w-]*)(.*)$', r'  \1\2', text, flags=re.MULTILINE)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading blank line
    text = re.sub(r'^\n+', '', text)

    return text.strip()


def _vk_api(method: str, params: dict) -> dict:
    """Call VK API method synchronously."""
    url = f"{VK_API_BASE}/{method}"
    resp = httpx.post(url, data=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"VK API error [{data['error']['error_code']}]: {data['error']['error_msg']}")
    return data.get("response", {})


async def _vk_api_async(method: str, params: dict) -> dict:
    """Call VK API method asynchronously."""
    url = f"{VK_API_BASE}/{method}"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, data=params)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"VK API error [{data['error']['error_code']}]: {data['error']['error_msg']}")
        return data.get("response", {})


def _parse_vk_message(event: list) -> Optional[dict]:
    """Parse a VK Long Poll event into a dict.

    Event type 4 = new message.
    Returns dict with peer_id, text, from_id, message_id, or None.
    """
    if not event or len(event) < 6:
        return None

    event_type = event[0]
    if event_type != EVENT_MESSAGE_NEW:
        return None

    message_id = event[1]
    flags = event[2]
    peer_id = event[3]
    timestamp = event[4]
    text = event[5] or ""

    # If text is not a string, it might be JSON object (with attachments)
    if not isinstance(text, str):
        try:
            text = ""
        except Exception:
            text = "[Attachment]"

    # Ignore outbox (our own messages)
    if flags & FLAG_OUTBOX:
        return None

    from_id = peer_id
    # For group chats (peer_id > 2e9), from_id is in the extra object
    # We'll extract from_id from event[7] if available
    if len(event) > 7 and isinstance(event[7], dict):
        from_id = event[7].get("from", peer_id)

    return {
        "message_id": message_id,
        "peer_id": peer_id,
        "from_id": from_id,
        "text": text,
        "timestamp": timestamp,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# VK Adapter
# ---------------------------------------------------------------------------


class VKAdapter(BasePlatformAdapter):
    """Async VK adapter using Long Poll API."""

    def __init__(self, config, **kwargs):
        platform = Platform("vk")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # Auth
        self.token = os.getenv("VK_TOKEN") or extra.get("token", "")
        group_id_str = os.getenv("VK_GROUP_ID") or extra.get("group_id", "")
        self.group_id = int(group_id_str) if group_id_str else 0
        self.api_version = os.getenv("VK_API_VERSION") or extra.get("api_version", VK_DEFAULT_VERSION)

        # Runtime state
        self._poll_task: Optional[asyncio.Task] = None
        self._last_message_id: Optional[int] = None  # Track last processed message
        self._http_client: Optional[httpx.AsyncClient] = None

        # Max message length
        self.max_message_length = extra.get("max_message_length", 4096)

    async def _resolve_lp_server(self, raw_url: str) -> str:
        """Resolve LP server hostname to IP to avoid async DNS issues."""
        from urllib.parse import urlparse
        parsed = urlparse(raw_url)
        hostname = parsed.hostname
        if not hostname:
            return raw_url
        try:
            import socket
            ips = await asyncio.get_event_loop().getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
            if ips:
                ip = ips[0][4][0]
                resolved = raw_url.replace(hostname, ip)
                logger.debug("VK: resolved %s → %s (%s)", hostname, ip, resolved)
                return resolved
        except Exception as e:
            logger.warning("VK: DNS resolution for %s failed: %s, using raw", hostname, e)
        return raw_url

    @property
    def name(self) -> str:
        return "VK"

    @property
    def _common_params(self) -> dict:
        return {
            "access_token": self.token,
            "v": self.api_version,
        }

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        if not self.token or not self.group_id:
            logger.error("VK: token and group_id must be configured")
            self._set_fatal_error(
                "config_missing",
                "VK_TOKEN and VK_GROUP_ID must be set",
                retryable=False,
            )
            return False

        self._http_client = httpx.AsyncClient(timeout=30)

        # Verify connection by getting conversations
        try:
            convs = await _vk_api_async("messages.getConversations", {
                **self._common_params,
                "group_id": self.group_id,
                "count": 1,
            })
            items = convs.get("items", [])
            if items:
                last_msg = items[0].get("last_message", {})
                self._last_message_id = last_msg.get("id")
                # Fetch full history for this conversation since connection time
                conv = items[0].get("conversation", {})
                peer = conv.get("peer", {})
                peer_id = peer.get("id")
                if peer_id:
                    await self._fetch_and_process_history(str(peer_id))
            logger.info("VK: connected (group %d, last_msg_id=%s)", self.group_id, self._last_message_id)
        except Exception as e:
            logger.error("VK: connection verification failed: %s", e)
            self._set_fatal_error("connect_failed", str(e), retryable=True)
            return False

        self._mark_connected()
        self._poll_task = asyncio.create_task(self._poll_loop())
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ── Poll loop ─────────────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        """Main poll loop using REST API (messages.getHistory)."""
        logger.info("VK: REST poll loop started (interval=%ds)", VK_POLL_INTERVAL)
        while self.is_connected:
            try:
                await self._check_new_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("VK: poll error: %s", e)
            await asyncio.sleep(VK_POLL_INTERVAL)

    async def _check_new_messages(self) -> None:
        """Check for new messages via REST API.

        Uses messages.getHistory on active conversations to fetch ALL new
        messages since _last_message_id, not just the last one. This is
        critical because when the bot sends response messages, the user's
        follow-up (e.g. /approve) is no longer the "last" message in the
        conversation and would be missed by getConversations alone.
        """
        # Step 1: find active conversations (has new messages since we last checked)
        params = {
            **self._common_params,
            "group_id": self.group_id,
            "count": 5,
        }
        if self._last_message_id:
            params["last_message_id"] = self._last_message_id

        try:
            convs = await _vk_api_async("messages.getConversations", params)
        except Exception as e:
            return  # Silently retry on next poll

        items = convs.get("items", [])
        if not items:
            return

        # Step 2: for each conversation, fetch ALL messages since last_message_id
        for item in reversed(items):
            conv = item.get("conversation", {})
            peer = conv.get("peer", {})
            peer_id = peer.get("id")
            if not peer_id:
                continue
            await self._fetch_and_process_history(str(peer_id))

    async def _fetch_and_process_history(self, peer_id: str) -> None:
        """Fetch all new messages in a conversation since _last_message_id and dispatch them.

        Each message is dispatched as a separate asyncio task so the poll loop
        can continue to run and pick up follow-up messages (e.g. the user's
        ``/approve`` or ``/deny`` response to a dangerous-command approval)
        without waiting for the previous message's agent processing to finish.
        """
        if not self._message_handler:
            return

        try:
            hist = await _vk_api_async("messages.getHistory", {
                **self._common_params,
                "peer_id": int(peer_id),
                "count": 200,
                "rev": 0,  # newest first — we only care about unprocessed msgs
            })
        except Exception as e:
            logger.debug("VK: getHistory error for peer %s: %s", peer_id, e)
            return

        msgs = hist.get("items", [])
        if not msgs:
            return

        for msg in msgs:
            msg_id = msg.get("id")
            if msg_id is None:
                continue

            # Skip already-processed messages
            if self._last_message_id is not None and msg_id <= self._last_message_id:
                continue

            # Update last_message_id BEFORE dispatching so we never re-process
            # this message, even if the task races with the next poll cycle.
            self._last_message_id = msg_id

            text = msg.get("text", "") or ""
            # Strip VK mention prefix from inline button clicks in group chats
            text = _strip_vk_mention(text)
            from_id = msg.get("from_id")
            attachments = msg.get("attachments", [])

            logger.info("VK: new message id=%d from=%s text=%r atts=%d", msg_id, from_id, text[:60], len(attachments))

            # Skip messages FROM the community (our own outgoing messages).
            # VK uses negative from_id for communities.
            if from_id is not None and from_id < 0:
                continue

            # Extract media attachments and add MEDIA: references to text
            media_text = ""
            has_attachment = False
            for att in attachments:
                att_type = att.get("type")
                if att_type == "photo":
                    has_attachment = True
                    sizes = att.get("photo", {}).get("sizes", [])
                    if sizes:
                        best = max(sizes, key=lambda s: s.get("width", 0) * s.get("height", 0))
                        url = best.get("url", "")
                        if url:
                            media_text += f"\nMEDIA:{url}"
                elif att_type == "doc":
                    has_attachment = True
                    doc = att.get("doc", {})
                    url = doc.get("url", "")
                    if url:
                        media_text += f"\nMEDIA:{url}"
                elif att_type == "audio_message":
                    has_attachment = True
                    am = att.get("audio_message", {})
                    url = am.get("link_mp3") or am.get("link_ogg", "")
                    if url:
                        media_text += f"\nMEDIA:{url}"

            if media_text:
                text = text + media_text

            if not text.strip() and not has_attachment:
                continue

            # Add a default description for attachment-only messages
            if not text.strip() and has_attachment:
                text = "[Media message]" + (media_text or "")

            int_peer = int(peer_id)

            # Determine chat type and name
            if int_peer > 2000000000:
                chat_type = "group"
                chat_name = f"chat_{int_peer}"
                try:
                    info = await _vk_api_async("messages.getConversationsById", {
                        **self._common_params,
                        "peer_ids": int_peer,
                    })
                    ci = info.get("items", [])
                    if ci and ci[0].get("chat_settings"):
                        chat_name = ci[0]["chat_settings"].get("title", chat_name)
                except Exception:
                    pass
            else:
                chat_type = "dm"
                chat_name = f"user_{from_id}"
                if from_id:
                    try:
                        ui = await _vk_api_async("users.get", {
                            **self._common_params,
                            "user_ids": from_id,
                        })
                        if ui:
                            chat_name = f"{ui[0].get('first_name', '')} {ui[0].get('last_name', '')}".strip()
                    except Exception:
                        pass

            source = self.build_source(
                chat_id=peer_id,
                chat_name=chat_name or peer_id,
                chat_type=chat_type,
                user_id=str(from_id) if from_id else peer_id,
                user_name=str(from_id) if from_id else peer_id,
            )

            # Dispatch message processing as a task so the poll loop can
            # continue to pick up follow-up messages (e.g. /approve, /deny).
            # This prevents a deadlock where the agent blocks waiting for
            # user input (approval) while the poll loop is stuck waiting
            # for the agent to finish.
            asyncio.create_task(
                self._process_one_message(peer_id, str(msg_id), text, source)
            )

    async def _process_one_message(
        self, peer_id: str, msg_id: str, text: str, source
    ) -> None:
        """Process a single message event: typing, dispatch, respond.

        Runs as an independent asyncio task so the poll loop can continue.
        """
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            self._keep_typing_simple(peer_id, stop_typing)
        )

        try:
            response_text = await self._message_handler(
                MessageEvent(
                    source=source,
                    message_id=msg_id,
                    text=text,
                )
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("VK: _process_one_message error: %s", exc)
            return
        finally:
            # Stop typing indicator
            stop_typing.set()
            typing_task.cancel()
            try:
                await typing_task
            except (asyncio.CancelledError, Exception):
                pass

        # Send the model's response back to the VK chat
        if response_text and isinstance(response_text, str) and response_text.strip():
            try:
                await self.send(peer_id, response_text.strip())
            except Exception as exc:
                logger.warning("VK: send error in _process_one_message: %s", exc)

    async def _keep_typing_simple(self, peer_id: str, stop_event: asyncio.Event) -> None:
        """Simple typing indicator loop — refresh setActivity every 3s until stopped.

        Respects ``_typing_paused`` set from the approval-flow pause mechanism.
        """
        try:
            while not stop_event.is_set():
                if str(peer_id) not in self._typing_paused:
                    await self.send_typing(peer_id)
                await asyncio.sleep(3)
        except asyncio.CancelledError:
            pass

    # ── Sending ───────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Send a message to a VK chat.

        Auto-splits messages > 4096 chars into multiple sends.
        """
        if not self.token:
            logger.warning("VK: send blocked — no token")
            return SendResult(success=False, error="Not connected")

        peer_id = int(chat_id)

        # Convert Markdown to VK-compatible formatting
        content = _vk_format_text(content)

        # VK message limit is 4096 chars; split if needed
        max_len = 4096
        if len(content) <= max_len:
            return await self._send_single(peer_id, content, reply_to)

        # Split into multiple messages
        logger.info("VK: splitting %d chars into parts", len(content))
        parts = []
        while content:
            if len(content) <= max_len:
                parts.append(content)
                break
            split_at = content.rfind('\n\n', 0, max_len)
            if split_at < max_len // 2:
                split_at = content.rfind('\n', 0, max_len)
            if split_at < max_len // 2:
                split_at = max_len
            parts.append(content[:split_at].strip())
            content = content[split_at:].strip()

        last_result = None
        for i, part in enumerate(parts):
            suffix = f" ({i+1}/{len(parts)})" if len(parts) > 1 else ""
            last_result = await self._send_single(peer_id, part + suffix, reply_to if i == 0 else None)
        return last_result or SendResult(success=True)

    async def _send_single(
        self, peer_id: int, content: str, reply_to: Optional[str] = None
    ) -> SendResult:
        """Send a single message (must be <= 4096 chars)."""
        params = {
            **self._common_params,
            "peer_id": peer_id,
            "message": content,
            "random_id": random.randint(-1_000_000, 1_000_000),
        }

        if reply_to:
            params["reply_to"] = int(reply_to)

        try:
            logger.info("VK: sending %d chars to peer %s", len(content), peer_id)
            result = await _vk_api_async("messages.send", params)
            logger.info("VK: send ok to peer %s, msg_id=%s", peer_id, result)
            return SendResult(success=True, message_id=str(result))
        except Exception as e:
            logger.error("VK: send error: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator via messages.setActivity."""
        try:
            result = await _vk_api_async("messages.setActivity", {
                **self._common_params,
                "peer_id": int(chat_id),
                "type": "typing",
                "group_id": self.group_id,
            })
            logger.debug("VK: send_typing ok for chat %s: %s", chat_id, result)
        except Exception as e:
            logger.warning("VK: send_typing failed for chat %s: %s", chat_id, e)

    async def send_image(self, chat_id: str, image_url: str, caption: str = None):
        """Send an image as a native VK photo attachment.

        Uses the VK upload server flow:
        1. photos.getMessagesUploadServer → upload URL
        2. Upload file to URL (multipart)
        3. photos.saveMessagesWallPhoto → save
        4. Send with attachment=photo{owner_id}_{id}
        """
        try:
            # Step 1: Get upload server
            upload_data = await _vk_api_async("photos.getMessagesUploadServer", {
                **self._common_params,
                "peer_id": int(chat_id),
            })
            upload_url = upload_data.get("upload_url")
            if not upload_url:
                logger.error("VK: send_image — no upload_url from server")
                return await super().send_image(chat_id, image_url, caption)

            # Step 2: Download the image locally first
            tmp_path = None
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.get(image_url)
                    resp.raise_for_status()
                    # Save to temp file
                    import tempfile
                    suffix = ".jpg"
                    for ext in [".png", ".gif", ".webp", ".jpeg"]:
                        if ext in image_url.lower():
                            suffix = ext
                            break
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                    tmp.close()

                # Step 3: Upload to VK server
                async with httpx.AsyncClient(timeout=30) as client:
                    with open(tmp_path, "rb") as f:
                        upload_resp = await client.post(
                            upload_url,
                            files={"photo": (f"image{suffix}", f, resp.headers.get("content-type", "image/jpeg"))},
                        )
                    upload_result = upload_resp.json()

                # Step 4: Save the photo
                saved = await _vk_api_async("photos.saveMessagesWallPhoto", {
                    **self._common_params,
                    "photo": upload_result.get("photo", ""),
                    "server": upload_result.get("server", 0),
                    "hash": upload_result.get("hash", ""),
                })
                if not saved:
                    raise RuntimeError("saveMessagesWallPhoto returned empty")

                photo = saved[0]
                attachment = f"photo{photo['owner_id']}_{photo['id']}"

                # Step 5: Send with attachment
                params = {
                    **self._common_params,
                    "peer_id": int(chat_id),
                    "attachment": attachment,
                    "random_id": random.randint(-1_000_000, 1_000_000),
                }
                if caption:
                    params["message"] = caption

                result = await _vk_api_async("messages.send", params)
                logger.info("VK: send_image ok, attachment=%s msg_id=%s", attachment, result)
                return SendResult(success=True, message_id=str(result))

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error("VK: send_image error: %s", e)
            # Fallback: send URL as text
            return await super().send_image(chat_id, image_url, caption)

    async def send_document(
        self,
        chat_id: str,
        file_path: str = "",
        *,
        file_url: str = "",
        caption: str = None,
        doc_type: str = "doc",
        reply_to: str = None,
        file_name: str = None,
        metadata: dict = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as a native VK document attachment.

        Uses docs.getMessagesUploadServer → upload → docs.save → send.
        Accepts ``file_path`` (local file) or ``file_url`` (HTTP URL).
        doc_type can be "doc" (generic file), "audio_message" (voice), or "graffiti".
        """
        # Normalise: gateway calls with file_path (local), but we also accept file_url for direct use.
        source = file_path or file_url or ""
        if not source:
            return SendResult(success=False, error="No file_path or file_url provided")

        try:
            # Step 1: Get upload server
            upload_data = await _vk_api_async("docs.getMessagesUploadServer", {
                **self._common_params,
                "peer_id": int(chat_id),
                "type": doc_type,
            })
            upload_url = upload_data.get("upload_url")
            if not upload_url:
                raise RuntimeError("No upload_url from docs.getMessagesUploadServer")

            # Step 2: Read the file (local path or HTTP URL)
            tmp_path = None
            cleanup_tmp = False
            try:
                import aiofiles
                if file_path and os.path.isfile(file_path):
                    # Local file — read directly
                    source_path = file_path
                else:
                    # Remote URL — download first
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.get(source, follow_redirects=True)
                    resp.raise_for_status()
                    import tempfile
                    content_type = resp.headers.get("content-type", "application/octet-stream")
                    ext = ".bin"
                    if "audio" in content_type or doc_type == "audio_message":
                        ext = ".ogg"
                    elif "pdf" in content_type:
                        ext = ".pdf"
                    elif "text" in content_type:
                        ext = ".txt"
                    elif "zip" in content_type:
                        ext = ".zip"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                    tmp.close()

                # VK blocks .html/.htm files — rename to .txt for the upload
                if ext.lower() in (".html", ".htm"):
                    ext = ".txt"
                    content_type = "text/plain"
                    logger.info("VK: .html blocked by VK, renaming to .txt for upload")

                # Step 3: Upload to VK
                async with httpx.AsyncClient(timeout=60) as client:
                    with open(tmp_path, "rb") as f:
                        field_name = "file" if doc_type != "audio_message" else "file"
                        upload_resp = await client.post(
                            upload_url,
                            files={field_name: (f"file{ext}", f, content_type)},
                        )
                    upload_result = upload_resp.json()

                file_param = upload_result.get("file", "")
                if not file_param:
                    raise RuntimeError("No 'file' in upload result")

                # Step 4: Save the document
                save_params = {
                    **self._common_params,
                    "file": file_param,
                    "title": caption or f"file{ext}",
                }
                if doc_type == "audio_message":
                    save_params["type"] = "audio_message"

                saved = await _vk_api_async("docs.save", save_params)
                if not saved:
                    raise RuntimeError("docs.save returned empty")

                doc = saved.get("audio_message") if doc_type == "audio_message" else saved.get("doc", saved)
                if isinstance(saved, list):
                    doc = saved[0]
                elif isinstance(saved, dict) and "doc" in saved:
                    doc = saved["doc"]

                owner_id = doc.get("owner_id", doc.get("doc", {}).get("owner_id", ""))
                doc_id = doc.get("id", doc.get("doc", {}).get("id", ""))
                attachment = f"doc{owner_id}_{doc_id}"

                # Step 5: Send with attachment
                params = {
                    **self._common_params,
                    "peer_id": int(chat_id),
                    "attachment": attachment,
                    "random_id": random.randint(-1_000_000, 1_000_000),
                }
                if caption and doc_type != "audio_message":
                    params["message"] = caption

                result = await _vk_api_async("messages.send", params)
                logger.info("VK: send_document ok, attachment=%s msg_id=%s", attachment, result)
                return SendResult(success=True, message_id=str(result))

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error("VK: send_document error: %s", e)
            # Fallback: send URL as text
            text = caption or ""
            text = f"{text}\n{file_url}" if text else file_url
            return await self.send(chat_id, text.strip())

    async def send_voice(self, chat_id: str, file_url: str, caption: str = None) -> SendResult:
        """Send a voice message (audio_message) to VK."""
        return await self.send_document(chat_id, file_url, caption, doc_type="audio_message")

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata=None,
    ) -> SendResult:
        """Send a confirmation prompt with inline buttons (VK keyboard)."""
        try:
            keyboard = json.dumps({
                "inline": True,
                "buttons": [[
                    {
                        "action": {
                            "type": "text",
                            "label": "/approve",
                            "payload": f'{{"confirm_id":"{confirm_id}","choice":"once"}}',
                        },
                        "color": "positive",
                    },
                    {
                        "action": {
                            "type": "text",
                            "label": "/cancel",
                            "payload": f'{{"confirm_id":"{confirm_id}","choice":"cancel"}}',
                        },
                        "color": "negative",
                    },
                ]],
            }, ensure_ascii=False)
            params = {
                **self._common_params,
                "peer_id": int(chat_id),
                "message": _vk_format_text(message[:4096]),
                "keyboard": keyboard,
                "random_id": random.randint(-1_000_000, 1_000_000),
            }
            result = await _vk_api_async("messages.send", params)
            logger.info("VK: send_slash_confirm ok msg_id=%s", result)
            return SendResult(success=True, message_id=str(result))
        except Exception as e:
            logger.warning("VK: send_slash_confirm failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata=None,
    ) -> SendResult:
        """Send a clarify prompt with inline buttons (VK keyboard)."""
        if not choices:
            return SendResult(success=False, error="No choices")
        try:
            buttons = []
            for choice in choices[:4]:
                buttons.append([{
                    "action": {"type": "text", "label": str(choice)[:40], "payload": "{}"},
                    "color": "primary",
                }])
            buttons.append([{
                "action": {"type": "text", "label": "✏ Other", "payload": "{}"},
                "color": "secondary",
            }])
            keyboard = json.dumps({"inline": True, "buttons": buttons}, ensure_ascii=False)
            params = {
                **self._common_params,
                "peer_id": int(chat_id),
                "message": question[:4096],
                "keyboard": keyboard,
                "random_id": random.randint(-1_000_000, 1_000_000),
            }
            result = await _vk_api_async("messages.send", params)
            logger.info("VK: send_clarify ok msg_id=%s", result)
            return SendResult(success=True, message_id=str(result))
        except Exception as e:
            logger.warning("VK: send_clarify failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str,
        description: str = "dangerous command",
        metadata=None,
    ) -> SendResult:
        """Send an inline-keyboard approval prompt for dangerous commands."""
        try:
            cmd_preview = command[:3800] + "..." if len(command) > 3800 else command
            text = (
                f"⚠️ **Dangerous command requires approval:**\n"
                f"```\n{cmd_preview}\n```"
            )
            keyboard = json.dumps({
                "inline": True,
                "buttons": [[
                    {"action": {"type": "text", "label": "/approve", "payload": "{}"}, "color": "positive"},
                    {"action": {"type": "text", "label": "/deny", "payload": "{}"}, "color": "negative"},
                    {"action": {"type": "text", "label": "/always", "payload": "{}"}, "color": "primary"},
                ]],
            }, ensure_ascii=False)
            params = {
                **self._common_params,
                "peer_id": int(chat_id),
                "message": _vk_format_text(text[:4096]),
                "keyboard": keyboard,
                "random_id": random.randint(-1_000_000, 1_000_000),
            }
            result = await _vk_api_async("messages.send", params)
            logger.info("VK: send_exec_approval ok msg_id=%s", result)
            return SendResult(success=True, message_id=str(result))
        except Exception as e:
            logger.warning("VK: send_exec_approval failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        peer_id = int(chat_id)
        is_group = peer_id > 2000000000

        try:
            info = await _vk_api_async("messages.getConversationsById", {
                **self._common_params,
                "peer_ids": peer_id,
            })
            items = info.get("items", [])
            if items:
                chat = items[0]
                name = chat.get("chat_settings", {}).get("title") if is_group else \
                    chat.get("peer", {}).get("local_id", str(peer_id))
                return {
                    "name": name or str(peer_id),
                    "type": "group" if is_group else "dm",
                }
        except Exception:
            pass

        return {
            "name": chat_id,
            "type": "group" if is_group else "dm",
        }


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------


def check_requirements() -> bool:
    """Check if httpx is available."""
    return HTTPX_AVAILABLE


def validate_config(config) -> bool:
    """Validate configuration, return True if valid."""
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("VK_TOKEN") or extra.get("token", "")
    group_id_str = os.getenv("VK_GROUP_ID") or extra.get("group_id", "")

    if not token or not group_id_str:
        return False
    try:
        int(group_id_str)
        return True
    except ValueError:
        return False


def is_connected(config) -> bool:
    """Quick check if the platform appears connected (called from status)."""
    token = os.getenv("VK_TOKEN") or getattr(config, "extra", {}).get("token", "")
    return bool(token)


async def interactive_setup(ctx) -> dict:
    """Interactive setup wizard for VK."""
    print("\n=== VK Messenger Bot Setup ===")
    print("1. Create a VK community (group)\n"
          "2. Go to Manage → Messages → Enable messages\n"
          "3. Go to Manage → API → Create token with 'messages' permission\n")
    token = input("VK API token: ").strip()
    group_id = input("VK community ID (numeric): ").strip()

    # Test token
    if token and group_id:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(f"{VK_API_BASE}/groups.getById", data={
                    "access_token": token,
                    "group_id": group_id,
                    "v": VK_DEFAULT_VERSION,
                })
                data = resp.json()
                if "response" in data:
                    group_name = data["response"]["groups"][0]["name"]
                    print(f"  ✓ Connected to community \"{group_name}\"")
                    return {
                        "token": token,
                        "group_id": group_id,
                    }
                else:
                    print(f"  ✗ API error: {data.get('error', {}).get('error_msg', 'unknown')}")
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")

    return {}


def _env_enablement() -> Optional[dict]:
    """Seed env vars into PlatformConfig.extra before adapter construction."""
    token = os.getenv("VK_TOKEN", "")
    group_id = os.getenv("VK_GROUP_ID", "")
    home = os.getenv("VK_HOME_CHANNEL", "")

    if not token or not group_id:
        return None

    result = {
        "token": token,
        "group_id": group_id,
    }
    if home:
        result["home_channel"] = home

    return result


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send a message without a running adapter (for cron / send_message tool).

    Must match the calling convention in tools/send_message_tool.py:
        (pconfig, chat_id, chunk, *, thread_id=..., media_files=..., force_document=...)
    """
    token = pconfig.get("token") if isinstance(pconfig, dict) else os.getenv("VK_TOKEN", "")
    if not token:
        return {"error": "VK standalone send: no token"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(f"{VK_API_BASE}/messages.send", data={
                "access_token": token,
                "peer_id": int(chat_id),
                "message": message,
                "random_id": random.randint(-1_000_000, 1_000_000),
                "v": VK_DEFAULT_VERSION,
            })
            data = resp.json()
            if "error" in data:
                return {"error": f"VK API error: {data['error']['error_msg']}"}
            return {"success": True, "message_id": str(data.get("response", ""))}
    except Exception as e:
        return {"error": f"VK standalone send failed: {e}"}


def register(ctx):
    """Plugin entry point: called by the Hermes plugin system."""
    ctx.register_platform(
        name="vk",
        label="VK",
        adapter_factory=lambda cfg: VKAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["VK_TOKEN", "VK_GROUP_ID"],
        install_hint="Requires httpx (pip install httpx)",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="VK_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="VK_ALLOWED_USERS",
        allow_all_env="VK_ALLOW_ALL_USERS",
        max_message_length=4096,
        emoji="💬",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via VK Messenger. "
            "VK supports basic text formatting. "
            "Messages are sent via VK API. "
            "Keep responses conversational and concise."
        ),
    )
