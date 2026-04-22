"""
Email platform adapter for the Hermes gateway.

Allows users to interact with Hermes by sending emails.
Uses IMAP to receive and SMTP to send messages.

Environment variables:
    EMAIL_IMAP_HOST     — IMAP server host (e.g., imap.gmail.com)
    EMAIL_IMAP_PORT     — IMAP server port (default: 993)
    EMAIL_SMTP_HOST     — SMTP server host (e.g., smtp.gmail.com)
    EMAIL_SMTP_PORT     — SMTP server port (default: 587)
    EMAIL_ADDRESS       — Email address for the agent
    EMAIL_PASSWORD      — Email password or app-specific password
    EMAIL_POLL_INTERVAL — Seconds between mailbox checks (default: 15)
    EMAIL_ALLOWED_USERS — Comma-separated list of allowed sender addresses
"""

import asyncio
import email as email_lib
import hashlib
import imaplib
import logging
import os
import re
import smtplib
import ssl
import time
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.config import Platform, PlatformConfig

logger = logging.getLogger(__name__)
# Automated sender patterns — emails from these are silently ignored
_NOREPLY_PATTERNS = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounce", "notifications@",
    "automated@", "auto-confirm", "auto-reply", "automailer",
)

# RFC headers that indicate bulk/automated mail
_AUTOMATED_HEADERS = {
    "Auto-Submitted": lambda v: v.lower() != "no",
    "Precedence": lambda v: v.lower() in ("bulk", "list", "junk"),
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}

# Gmail-safe max length per email body
MAX_MESSAGE_LENGTH = 50_000

# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def _is_automated_sender(address: str, headers: dict) -> bool:
    """Return True if this email is from an automated/noreply source."""
    addr = address.lower()
    if any(pattern in addr for pattern in _NOREPLY_PATTERNS):
        return True
    for header, check in _AUTOMATED_HEADERS.items():
        value = headers.get(header, "")
        if value and check(value):
            return True
    return False
    
def check_email_requirements() -> bool:
    """Check if email platform dependencies are available."""
    addr = os.getenv("EMAIL_ADDRESS")
    pwd = os.getenv("EMAIL_PASSWORD")
    imap = os.getenv("EMAIL_IMAP_HOST")
    smtp = os.getenv("EMAIL_SMTP_HOST")
    if not all([addr, pwd, imap, smtp]):
        return False
    return True


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 encoded email header into a plain string."""
    parts = decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            # Skip attachments
            if "attachment" in disposition:
                continue
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # Fallback: try text/html and strip tags
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            if content_type == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    return _strip_html(html)
        return ""
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                return _strip_html(text)
            return text
        return ""


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    if match:
        return match.group(1).strip().lower()
    return raw.strip().lower()


def _extract_attachments(
    msg: email_lib.message.Message,
    skip_attachments: bool = False,
) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally.

    When *skip_attachments* is True, all attachment/inline parts are ignored
    (useful for malware protection or bandwidth savings).
    """
    attachments = []
    if not msg.is_multipart():
        return attachments

    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", ""))
        if skip_attachments and ("attachment" in disposition or "inline" in disposition):
            continue
        if "attachment" not in disposition and "inline" not in disposition:
            continue
        # Skip text/plain and text/html body parts
        content_type = part.get_content_type()
        if content_type in ("text/plain", "text/html") and "attachment" not in disposition:
            continue

        filename = part.get_filename()
        if filename:
            filename = _decode_header_value(filename)
        else:
            ext = part.get_content_subtype() or "bin"
            filename = f"attachment.{ext}"

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        ext = Path(filename).suffix.lower()
        if ext in _IMAGE_EXTS:
            try:
                cached_path = cache_image_from_bytes(payload, ext)
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
            attachments.append({
                "path": cached_path,
                "filename": filename,
                "type": "image",
                "media_type": content_type,
            })
        else:
            cached_path = cache_document_from_bytes(payload, filename)
            attachments.append({
                "path": cached_path,
                "filename": filename,
                "type": "document",
                "media_type": content_type,
            })

    return attachments


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        self._address = os.getenv("EMAIL_ADDRESS", "")
        self._password = os.getenv("EMAIL_PASSWORD", "")
        self._imap_host = os.getenv("EMAIL_IMAP_HOST", "")
        self._imap_port = int(os.getenv("EMAIL_IMAP_PORT", "993"))
        self._smtp_host = os.getenv("EMAIL_SMTP_HOST", "")
        self._smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self._poll_interval = int(os.getenv("EMAIL_POLL_INTERVAL", "15"))

        # Skip attachments — configured via config.yaml:
        #   platforms:
        #     email:
        #       skip_attachments: true
        extra = config.extra or {}
        self._skip_attachments = extra.get("skip_attachments", False)

        # Session-keying mode — controls whether inbound messages from the
        # same sender share a session or are split per-thread.
        #   "sender"           (default): one session per sender address,
        #                                  preserving pre-existing behavior.
        #   "gmail_thread_id": one session per Gmail thread, keyed by Gmail's
        #                      X-GM-THRID IMAP extension. Requires an
        #                      imap.gmail.com-style server that advertises the
        #                      X-GM-EXT-1 capability (Google Workspace / Gmail).
        # Configure via:
        #   platforms:
        #     email:
        #       extra:
        #         session_keying: gmail_thread_id
        mode = (extra.get("session_keying") or "sender").lower()
        if mode not in ("sender", "gmail_thread_id"):
            logger.warning(
                "[Email] Unknown session_keying=%r, falling back to 'sender'",
                extra.get("session_keying"),
            )
            mode = "sender"
        self._session_keying: str = mode
        # Populated in connect() from imap.capability(); stays False until a
        # successful login confirms the server advertises X-GM-EXT-1.
        self._has_gmail_ext: bool = False

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None

        # Map (sender_addr, thread_key) -> subject/message-id/last_seen_ts for threading.
        # Keyed by a tuple so two concurrent inbound messages from the same
        # sender can coexist without one clobbering the other's reply headers.
        # In sender mode, thread_key is the inbound message_id. In
        # gmail_thread_id mode, thread_key is the derived thread identifier
        # (e.g. "gthr-1234567890").
        self._thread_context: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._thread_context_max: int = 500

        logger.info(
            "[Email] Adapter initialized for %s (session_keying=%s)",
            self._address, self._session_keying,
        )

    def _trim_seen_uids(self) -> None:
        """Keep only the most recent UIDs to prevent unbounded memory growth.

        IMAP UIDs are monotonically increasing integers. When the set grows
        beyond the cap, we keep only the highest half — old UIDs are safe to
        drop because new messages always have higher UIDs and IMAP's UNSEEN
        flag prevents re-delivery regardless.
        """
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            # UIDs are bytes like b'1234' — sort numerically and keep top half
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))
            keep = self._seen_uids_max // 2
            self._seen_uids = set(sorted_uids[-keep:])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            # Fallback: just clear old entries if sort fails
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2:])

    def _trim_thread_context(self) -> None:
        """Bound `_thread_context` size to prevent unbounded memory growth.

        When the dict exceeds `_thread_context_max`, drop the oldest half by
        insertion order (Python 3.7+ preserves dict insertion order). Snapshot
        keys via `list(items())` before iterating so callers can safely invoke
        this from inside a write path without tripping "dict changed size
        during iteration".
        """
        if len(self._thread_context) <= self._thread_context_max:
            return
        items = list(self._thread_context.items())
        keep = self._thread_context_max // 2
        self._thread_context = dict(items[-keep:])
        logger.debug("[Email] Trimmed thread context to %d entries", len(self._thread_context))

    def _lookup_thread_context(
        self,
        to_addr: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up reply-threading context for an outbound send.

        If `thread_id` is provided, try an exact `(to_addr, thread_id)` match
        — this is how PR B's session-keyed callers resolve to the correct
        thread. Otherwise, fall back to the most-recent entry (by
        `last_seen_ts`) whose key's first element is `to_addr` — preserves the
        pre-tuple-rekey "reply with this sender's latest subject" behavior
        for callers that haven't been plumbed `thread_id` through yet.
        """
        if thread_id is not None:
            ctx = self._thread_context.get((to_addr, thread_id))
            if ctx is not None:
                return ctx
        latest: Optional[Dict[str, Any]] = None
        latest_ts: float = -1.0
        for (sender, _mid), ctx in self._thread_context.items():
            if sender != to_addr:
                continue
            ts = ctx.get("last_seen_ts", 0.0)
            if ts > latest_ts:
                latest_ts = ts
                latest = ctx
        return latest or {}

    def _detect_gmail_extension(self, imap: imaplib.IMAP4_SSL) -> bool:
        """Return True when the IMAP server advertises X-GM-EXT-1.

        X-GM-EXT-1 is Gmail's IMAP extension capability — see
        https://developers.google.com/gmail/imap/imap-extensions. Presence
        guarantees `X-GM-THRID`, `X-GM-MSGID`, and `X-GM-LABELS` are
        available as FETCH data items.
        """
        try:
            status, data = imap.capability()
            if status != "OK" or not data:
                return False
            caps_raw = b" ".join(data) if isinstance(data, list) else bytes(data)
            return b"X-GM-EXT-1" in caps_raw.upper()
        except Exception as e:
            logger.warning("[Email] Capability probe failed: %s", e)
            return False

    async def connect(self) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        try:
            # Test IMAP connection
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
            imap.login(self._address, self._password)
            # Probe for Gmail IMAP extension (X-GM-EXT-1). If the caller asked
            # for gmail_thread_id mode but the server doesn't advertise the
            # extension, degrade to sender mode with a warning so we fail
            # loudly in logs rather than silently mis-route sessions.
            self._has_gmail_ext = self._detect_gmail_extension(imap)
            if self._session_keying == "gmail_thread_id" and not self._has_gmail_ext:
                logger.warning(
                    "[Email] session_keying=gmail_thread_id requested but "
                    "server %s does not advertise X-GM-EXT-1; falling back to "
                    "sender mode", self._imap_host,
                )
                self._session_keying = "sender"
            # Mark all existing messages as seen so we only process new ones
            imap.select("INBOX")
            status, data = imap.uid("search", None, "ALL")
            if status == "OK" and data and data[0]:
                for uid in data[0].split():
                    self._seen_uids.add(uid)
            # Keep only the most recent UIDs to prevent unbounded growth
            self._trim_seen_uids()
            imap.logout()
            logger.info("[Email] IMAP connection test passed. %d existing messages skipped.", len(self._seen_uids))
        except Exception as e:
            logger.error("[Email] IMAP connection failed: %s", e)
            return False

        try:
            # Test SMTP connection
            smtp = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30)
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self._address, self._password)
            smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
        except Exception as e:
            logger.error("[Email] SMTP connection failed: %s", e)
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        logger.info("[Email] Disconnected.")

    async def _poll_loop(self) -> None:
        """Poll IMAP for new messages at regular intervals."""
        while self._running:
            try:
                await self._check_inbox()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        # Run IMAP operations in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(None, self._fetch_new_messages)
        for msg_data in messages:
            await self._dispatch_message(msg_data)

    @staticmethod
    def _parse_gm_thrid(fetch_response: Any) -> Optional[str]:
        """Extract X-GM-THRID from a Gmail IMAP FETCH response.

        Gmail returns the THRID inline with the RFC822 literal header, e.g.
            b'5 (UID 5 X-GM-THRID 1234567890123456789 RFC822 {12345}'
        Returns the decoded decimal THRID or None if not present.
        """
        try:
            if not fetch_response or not isinstance(fetch_response, tuple):
                return None
            header = fetch_response[0]
            if not isinstance(header, (bytes, bytearray)):
                return None
            m = re.search(rb"X-GM-THRID\s+(\d+)", header)
            if m:
                return m.group(1).decode("ascii")
        except Exception as e:
            logger.debug("[Email] THRID parse failed: %s", e)
        return None

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        # Extend the FETCH payload to include X-GM-THRID when Gmail session
        # keying is active. No-op server-side cost: THRID is already indexed
        # on the Gmail side, and it's returned on the same round-trip.
        use_gmail_thrid = (
            self._session_keying == "gmail_thread_id" and self._has_gmail_ext
        )
        fetch_items = "(RFC822 X-GM-THRID)" if use_gmail_thrid else "(RFC822)"
        try:
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
            try:
                imap.login(self._address, self._password)
                imap.select("INBOX")

                status, data = imap.uid("search", None, "UNSEEN")
                if status != "OK" or not data or not data[0]:
                    return results

                for uid in data[0].split():
                    if uid in self._seen_uids:
                        continue
                    self._seen_uids.add(uid)
                    # Trim periodically to prevent unbounded memory growth
                    if len(self._seen_uids) > self._seen_uids_max:
                        self._trim_seen_uids()

                    status, msg_data = imap.uid("fetch", uid, fetch_items)
                    if status != "OK":
                        continue

                    # msg_data[0] is (header_bytes, rfc822_body) for a tuple
                    # response, or a bare literal in edge cases.
                    gmail_thrid: Optional[str] = None
                    if use_gmail_thrid:
                        gmail_thrid = self._parse_gm_thrid(msg_data[0])

                    raw_email = msg_data[0][1]
                    msg = email_lib.message_from_bytes(raw_email)

                    sender_raw = msg.get("From", "")
                    sender_addr = _extract_email_address(sender_raw)
                    sender_name = _decode_header_value(sender_raw)
                    # Remove email from name if present
                    if "<" in sender_name:
                        sender_name = sender_name.split("<")[0].strip().strip('"')

                    subject = _decode_header_value(msg.get("Subject", "(no subject)"))
                    message_id = msg.get("Message-ID", "")
                    in_reply_to = msg.get("In-Reply-To", "")
                    # Skip automated/noreply senders before any processing
                    msg_headers = dict(msg.items())
                    if _is_automated_sender(sender_addr, msg_headers):
                        logger.debug("[Email] Skipping automated sender: %s", sender_addr)
                        continue
                    body = _extract_text_body(msg)
                    attachments = _extract_attachments(msg, skip_attachments=self._skip_attachments)

                    results.append({
                        "uid": uid,
                        "sender_addr": sender_addr,
                        "sender_name": sender_name,
                        "subject": subject,
                        "message_id": message_id,
                        "in_reply_to": in_reply_to,
                        "body": body,
                        "attachments": attachments,
                        "date": msg.get("Date", ""),
                        "gmail_thread_id": gmail_thrid,
                    })
            finally:
                try:
                    imap.logout()
                except Exception:
                    pass
        except Exception as e:
            logger.error("[Email] IMAP fetch error: %s", e)
        return results

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]

        # Skip self-messages
        if sender_addr == self._address.lower():
            return

        # Never reply to automated senders
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return

        subject = msg_data["subject"]
        body = msg_data["body"].strip()
        attachments = msg_data["attachments"]

        # Build message text: include subject as context
        text = body
        if subject and not subject.startswith("Re:"):
            text = f"[Subject: {subject}]\n\n{body}"

        # Determine message type and media
        media_urls = []
        media_types = []
        msg_type = MessageType.TEXT

        for att in attachments:
            media_urls.append(att["path"])
            media_types.append(att["media_type"])
            if att["type"] == "image":
                msg_type = MessageType.PHOTO

        # Derive the session thread_id based on keying mode.
        #   - sender mode: no thread_id, collapses all messages from this
        #     sender into one session (pre-existing behavior).
        #   - gmail_thread_id mode: use Gmail's X-GM-THRID prefixed with
        #     "gthr-" so the session key is deterministic across restarts.
        #     If THRID is missing (Gmail didn't return it — edge case on
        #     delegated mailboxes, etc.), fall back to hashing the inbound
        #     Message-ID so the message is treated as its own thread root.
        thread_id: Optional[str] = None
        if self._session_keying == "gmail_thread_id":
            thrid = msg_data.get("gmail_thread_id")
            if thrid:
                thread_id = f"gthr-{thrid}"
            else:
                own_mid = msg_data.get("message_id") or ""
                if own_mid:
                    digest = hashlib.sha1(own_mid.encode("utf-8", errors="ignore")).hexdigest()[:12]
                    thread_id = f"mid-{digest}"

        # Store thread context for reply threading.
        # In sender mode thread_id is None, so key on message_id (preserves
        # the PR A clobber-prevention guarantee).
        # In gmail_thread_id mode key on thread_id so multiple messages
        # within the same thread accumulate on a single context entry.
        ctx_key = (sender_addr, thread_id or msg_data["message_id"])
        self._thread_context[ctx_key] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
            "last_seen_ts": time.time(),
        }
        self._trim_thread_context()

        source = self.build_source(
            chat_id=sender_addr,
            chat_name=msg_data["sender_name"] or sender_addr,
            chat_type="dm",
            user_id=sender_addr,
            user_name=msg_data["sender_name"] or sender_addr,
            thread_id=thread_id,
        )

        event = MessageEvent(
            text=text or "(empty email)",
            message_type=msg_type,
            source=source,
            message_id=msg_data["message_id"],
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=msg_data["in_reply_to"] or None,
        )

        logger.info("[Email] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an email reply to the given address.

        `metadata["thread_id"]` (if present) is used to look up the exact
        reply-threading context for the originating inbound message, so
        concurrent threads from the same sender don't cross-pollinate reply
        headers.
        """
        thread_id = (metadata or {}).get("thread_id")
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None, self._send_email, chat_id, content, reply_to, thread_id
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send failed to %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    def _send_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        # Thread context for reply — helper resolves to the exact stored entry
        # when thread_id is present (PR B keying), otherwise falls back to
        # the most-recent entry for this sender (PR A behavior).
        ctx = self._lookup_thread_context(to_addr, thread_id)
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # Threading headers
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._address.split('@')[1]}>"
        msg["Message-ID"] = msg_id

        msg.attach(MIMEText(body, "plain", "utf-8"))

        smtp = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30)
        try:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Email has no typing indicator — no-op."""

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image URL as part of an email body.

        Accepts `metadata` so thread_id-scoped replies carry the correct
        In-Reply-To / Message-ID headers in gmail_thread_id mode.
        """
        text = caption or ""
        text += f"\n\nImage: {image_url}"
        return await self.send(chat_id, text.strip(), reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a file as an email attachment.

        Accepts `metadata` so thread_id-scoped attachment replies thread
        correctly in gmail_thread_id mode.
        """
        thread_id = (metadata or {}).get("thread_id")
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None,
                self._send_email_with_attachment,
                chat_id,
                caption or "",
                file_path,
                file_name,
                thread_id,
            )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send document failed: %s", e)
            return SendResult(success=False, error=str(e))

    def _send_email_with_attachment(
        self,
        to_addr: str,
        body: str,
        file_path: str,
        file_name: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Send an email with a file attachment via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._lookup_thread_context(to_addr, thread_id)
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._address.split('@')[1]}>"
        msg["Message-ID"] = msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attach file
        p = Path(file_path)
        fname = file_name or p.name
        with open(p, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={fname}")
            msg.attach(part)

        smtp = smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30)
        try:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

        return msg_id

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        ctx = self._lookup_thread_context(chat_id)
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": ctx.get("subject", ""),
        }
