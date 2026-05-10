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
import json
import logging
import os
import re
import smtplib
import ssl
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate
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
from gateway.email_challenge import (
    DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS,
    EmailChallengeStore,
    cleanup_challenge_cached_attachments,
)

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
MAX_EMAIL_CHALLENGE_ATTACHMENTS = 25
MAX_EMAIL_CHALLENGE_ATTACHMENT_BYTES = 10_000_000
MAX_EMAIL_CHALLENGE_TOTAL_ATTACHMENT_BYTES = 25_000_000
MAX_EMAIL_CHALLENGE_EVENT_BYTES = 200_000
MAX_EMAIL_CONFIRM_CODE_LENGTH = 256
_CONFIRM_RE = re.compile(r"^\s*/confirm\s+([^\s]+)\s*(?:\r?\n.*)?\Z", re.IGNORECASE | re.DOTALL)

# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

def _send_imap_id(imap: "imaplib.IMAP4") -> None:
    """Send RFC 2971 IMAP ID command identifying this client.

    Required by 163/NetEase mailbox after LOGIN: without it, every UID
    SEARCH/FETCH returns ``BYE Unsafe Login`` and disconnects.  Other
    IMAP servers either honor it silently or reject the unknown command;
    we swallow failures so non-supporting servers keep working.
    """
    try:
        try:
            from hermes_cli import __version__ as _hermes_version
        except Exception:  # noqa: BLE001 — keep ID best-effort if import fails
            _hermes_version = "0"
        imap.xatom(
            "ID",
            f'("name" "hermes-agent" "version" "{_hermes_version}" '
            '"vendor" "NousResearch" '
            '"support-email" "noreply@nousresearch.com")',
        )
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


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
    max_attachment_bytes: Optional[int] = None,
    max_total_attachment_bytes: Optional[int] = None,
    max_attachments: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally.

    When *skip_attachments* is True, all attachment/inline parts are ignored
    (useful for malware protection or bandwidth savings).
    """
    attachments = []
    if not msg.is_multipart():
        return attachments

    total_payload_bytes = 0
    try:
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
            if max_attachments is not None and len(attachments) >= max_attachments:
                raise ValueError("too many attachments")
            payload_size = len(payload)
            if max_attachment_bytes is not None and payload_size > max_attachment_bytes:
                raise ValueError("attachment payload too large")
            total_payload_bytes += payload_size
            if max_total_attachment_bytes is not None and total_payload_bytes > max_total_attachment_bytes:
                raise ValueError("attachment payload total too large")

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
    except Exception:
        cleanup_challenge_cached_attachments({"attachments": attachments})
        raise

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
        self._auth_mode_config = str(extra.get("auth_mode") or extra.get("email_auth_mode") or "")
        self._challenge_store_config = str(extra.get("challenge_store") or "")
        self._challenge_ttl_config = extra.get("challenge_ttl_seconds")
        self._email_challenge_store_cache = EmailChallengeStore(
            path=os.getenv("EMAIL_CHALLENGE_STORE", "").strip() or self._challenge_store_config or None,
            ttl_seconds=self._email_challenge_ttl_seconds(),
        )

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None

        # Map chat_id (sender email) -> last subject + message-id for threading
        self._thread_context: Dict[str, Dict[str, str]] = {}

        logger.info("[Email] Adapter initialized for %s", self._address)

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

    async def connect(self) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        try:
            # Test IMAP connection
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
            imap.login(self._address, self._password)
            _send_imap_id(imap)
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
        if self._email_auth_mode() == "challenge":
            try:
                await loop.run_in_executor(None, self._cleanup_expired_email_challenges)
            except Exception as exc:  # noqa: BLE001 - cleanup must not block polling
                logger.warning("[Email] Email challenge cleanup failed; continuing inbox poll: %s", exc)
        messages = await loop.run_in_executor(None, self._fetch_new_messages)
        for msg_data in messages:
            await self._dispatch_message(msg_data)

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        try:
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
            try:
                imap.login(self._address, self._password)
                _send_imap_id(imap)
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

                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        continue

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
                    auth_mode = self._email_auth_mode()
                    if auth_mode == "challenge" and not self._is_email_challenge_authorized(sender_addr):
                        logger.debug("[Email] Skipping unauthorized challenge sender before body/cache: %s", sender_addr)
                        continue
                    direct_pre_auth = "allow"
                    if auth_mode == "direct":
                        direct_pre_auth = self._email_direct_pre_auth(sender_addr)
                    if auth_mode == "direct" and direct_pre_auth == "deny":
                        logger.debug("[Email] Skipping unauthorized direct sender before body/cache: %s", sender_addr)
                        continue

                    # Challenge mode must parse the text body to detect /confirm
                    # replies and oversized originals, but attachment caching is
                    # deferred until after those pre-agent eligibility checks.
                    body = _extract_text_body(msg)
                    attachment_error = ""
                    if auth_mode == "challenge" and (
                        len(body.strip()) > MAX_MESSAGE_LENGTH or _CONFIRM_RE.match(body.strip())
                    ):
                        attachments = []
                    else:
                        try:
                            if auth_mode == "challenge":
                                attachments = _extract_attachments(
                                    msg,
                                    skip_attachments=self._skip_attachments,
                                    max_attachment_bytes=MAX_EMAIL_CHALLENGE_ATTACHMENT_BYTES,
                                    max_total_attachment_bytes=MAX_EMAIL_CHALLENGE_TOTAL_ATTACHMENT_BYTES,
                                    max_attachments=MAX_EMAIL_CHALLENGE_ATTACHMENTS,
                                )
                            elif direct_pre_auth == "allow":
                                attachments = _extract_attachments(msg, skip_attachments=self._skip_attachments)
                            else:
                                attachments = []
                        except ValueError as exc:
                            if auth_mode != "challenge":
                                raise
                            logger.warning("[Email] Rejecting challenge attachment payload for %s: %s", sender_addr, exc)
                            attachments = []
                            if str(exc) == "too many attachments":
                                attachment_error = "too_many"
                            else:
                                attachment_error = "too_large"

                    result = {
                        "uid": uid,
                        "sender_addr": sender_addr,
                        "sender_name": sender_name,
                        "subject": subject,
                        "message_id": message_id,
                        "in_reply_to": in_reply_to,
                        "body": body,
                        "attachments": attachments,
                        "date": msg.get("Date", ""),
                    }
                    if attachment_error:
                        result["_email_challenge_attachment_error"] = attachment_error
                    results.append(result)
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
            cleanup_challenge_cached_attachments(msg_data)
            return

        # Never reply to automated senders
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            cleanup_challenge_cached_attachments(msg_data)
            return

        # In direct mode, preserve the historical early EMAIL_ALLOWED_USERS
        # guard. In challenge mode, only challenge senders authorized by env/global
        # allowlists the central gateway will also accept after confirmation.
        if self._email_auth_mode() == "challenge":
            if not self._is_email_challenge_authorized(sender_addr):
                logger.debug("[Email] Dropping unauthorized challenge sender at dispatch: %s", sender_addr)
                cleanup_challenge_cached_attachments(msg_data)
                return
        else:
            direct_pre_auth = self._email_direct_pre_auth(sender_addr)
            if direct_pre_auth == "deny":
                logger.debug("[Email] Dropping unauthorized direct sender at dispatch: %s", sender_addr)
                cleanup_challenge_cached_attachments(msg_data)
                return
            if direct_pre_auth != "allow":
                cleanup_challenge_cached_attachments(msg_data)
                msg_data["attachments"] = []

        subject = msg_data["subject"]
        body = msg_data["body"].strip()
        attachments = msg_data["attachments"]

        if (
            self._email_auth_mode() == "challenge"
            and not msg_data.get("_email_challenge_confirmed")
        ):
            confirm_match = _CONFIRM_RE.match(body)
            if confirm_match:
                try:
                    await self._handle_challenge_confirmation(sender_addr, confirm_match.group(1), msg_data)
                finally:
                    cleanup_challenge_cached_attachments(msg_data)
                return
            await self._create_email_challenge(sender_addr, subject, body, msg_data)
            return

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

        # Store thread context for reply threading
        self._thread_context[sender_addr] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
        }

        source = self.build_source(
            chat_id=sender_addr,
            chat_name=msg_data["sender_name"] or sender_addr,
            chat_type="dm",
            user_id=sender_addr,
            user_name=msg_data["sender_name"] or sender_addr,
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

    def _email_auth_mode(self) -> str:
        raw_mode = os.getenv("EMAIL_AUTH_MODE", "").strip().lower() or self._auth_mode_config.strip().lower()
        if not raw_mode:
            return "direct"
        if raw_mode in {"direct", "challenge"}:
            return raw_mode
        logger.warning("[Email] Invalid EMAIL_AUTH_MODE %r; using challenge mode for safety", raw_mode)
        return "challenge"

    def _email_auth_list_matches(self, raw: Any, sender_addr: str, include_local_part: bool = False) -> bool:
        if isinstance(raw, (list, tuple, set)):
            entries = {str(addr).strip() for addr in raw if str(addr).strip()}
        else:
            raw_text = str(raw or "").strip()
            entries = {addr.strip() for addr in raw_text.split(",") if addr.strip()}
        if not entries:
            return False
        check_ids = {sender_addr}
        if include_local_part and "@" in sender_addr:
            check_ids.add(sender_addr.split("@", 1)[0])
        return "*" in entries or bool(check_ids & entries)

    def _is_email_direct_authorized(self, sender_addr: str) -> bool:
        return self._email_direct_pre_auth(sender_addr) != "deny"

    def _email_direct_pre_auth(self, sender_addr: str) -> str:
        """Return allow/deny/defer for direct-mode pre-central auth.

        ``allow`` means this adapter can safely cache attachments before the
        gateway's central authorization. ``defer`` preserves central pairing
        checks but strips attachments because this adapter cannot verify them.
        """
        if os.getenv("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in ("true", "1", "yes"):
            return "allow"

        email_allowlist = os.getenv("EMAIL_ALLOWED_USERS", "").strip()
        global_allowlist = os.getenv("GATEWAY_ALLOWED_USERS", "").strip()
        if not email_allowlist and not global_allowlist:
            if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in ("true", "1", "yes"):
                return "allow"
            return "defer"

        if email_allowlist:
            # Preserve direct-mode historical behavior: EMAIL_ALLOWED_USERS is
            # a case-insensitive literal full-address list, so "*" is not a
            # wildcard here. GATEWAY_ALLOWED_USERS keeps central wildcard rules.
            allowed = {addr.strip().lower() for addr in email_allowlist.split(",") if addr.strip()}
            if sender_addr.lower() in allowed:
                return "allow"

        if global_allowlist and self._email_auth_list_matches(global_allowlist, sender_addr, include_local_part=True):
            return "allow"
        if global_allowlist and not email_allowlist:
            # Let central gateway auth decide pairing-store approvals. The
            # adapter cannot see pairing state, so preserve the body-only event
            # path without caching attachments for non-allowlisted senders.
            return "defer"

        return "deny"

    def _is_email_challenge_authorized(self, sender_addr: str) -> bool:
        if os.getenv("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in ("true", "1", "yes"):
            return True

        if self._email_auth_list_matches(os.getenv("EMAIL_ALLOWED_USERS", ""), sender_addr):
            return True
        if self._email_auth_list_matches(os.getenv("GATEWAY_ALLOWED_USERS", ""), sender_addr, include_local_part=True):
            return True

        configured_allowlists = any([
            os.getenv("EMAIL_ALLOWED_USERS", "").strip(),
            os.getenv("GATEWAY_ALLOWED_USERS", "").strip(),
        ])
        if not configured_allowlists:
            return os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in ("true", "1", "yes")
        return False

    def _email_challenge_ttl_seconds(self) -> int:
        raw = os.getenv("EMAIL_CHALLENGE_TTL_SECONDS", "").strip()
        if not raw and self._challenge_ttl_config is not None:
            raw = str(self._challenge_ttl_config).strip()
        if not raw:
            return DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS
        try:
            ttl = int(raw)
        except ValueError:
            return DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS
        return ttl if ttl > 0 else DEFAULT_EMAIL_CHALLENGE_TTL_SECONDS

    def _email_challenge_store(self) -> EmailChallengeStore:
        return self._email_challenge_store_cache

    def _cleanup_expired_email_challenges(self) -> None:
        if self._email_auth_mode() != "challenge":
            return
        self._email_challenge_store().cleanup_expired()

    async def _create_email_challenge(
        self,
        sender_addr: str,
        subject: str,
        body: str,
        msg_data: Dict[str, Any],
    ) -> None:
        self._thread_context[sender_addr] = {
            "subject": subject,
            "message_id": msg_data.get("message_id", ""),
        }
        if len(body) > MAX_MESSAGE_LENGTH:
            cleanup_challenge_cached_attachments(msg_data)
            await self.send(
                sender_addr,
                "Your email request is too large to challenge or process. Please shorten it and resend.",
                reply_to=msg_data.get("message_id") or None,
            )
            return
        if msg_data.get("_email_challenge_attachment_error") == "too_large":
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request has an attachment that is too large to challenge or process. Please reduce attachments and resend.",
                msg_data,
            )
            return
        if msg_data.get("_email_challenge_attachment_error") == "too_many":
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request has too many attachments to challenge or process. Please reduce the attachments and resend.",
                msg_data,
            )
            return
        attachments = msg_data.get("attachments", [])
        if not isinstance(attachments, list):
            attachments = []
        if len(attachments) > MAX_EMAIL_CHALLENGE_ATTACHMENTS:
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request has too many attachments to challenge or process. Please reduce the attachments and resend.",
                msg_data,
            )
            return
        event_data = {
            "sender_addr": sender_addr,
            "sender_name": msg_data.get("sender_name", sender_addr),
            "subject": subject,
            "message_id": msg_data.get("message_id", ""),
            "in_reply_to": msg_data.get("in_reply_to", ""),
            "body": body,
            "attachments": attachments,
            "date": msg_data.get("date", ""),
            "_email_challenge_confirmed": True,
        }
        try:
            event_size = len(json.dumps(event_data, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            logger.warning("[Email] Email challenge metadata could not be serialized for %s: %s", sender_addr, exc)
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request is temporarily unavailable for confirmation. Please try again later.",
                msg_data,
            )
            return
        if event_size > MAX_EMAIL_CHALLENGE_EVENT_BYTES:
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request is too large to challenge or process. Please shorten it and resend.",
                msg_data,
            )
            return
        store = self._email_challenge_store()
        try:
            code = store.create(sender_addr, subject, msg_data.get("message_id", ""), event_data)
        except Exception as exc:  # noqa: BLE001 - fail closed per message on store I/O errors
            logger.warning("[Email] Email challenge store create failed for %s: %s", sender_addr, exc)
            cleanup_challenge_cached_attachments(msg_data)
            await self._send_email_challenge_notice(
                sender_addr,
                "Your email request is temporarily unavailable for confirmation. Please try again later.",
                msg_data,
            )
            return
        if code is None:
            logger.warning("[Email] Email challenge not created for %s; pending challenge cap reached", sender_addr)
            cleanup_challenge_cached_attachments(msg_data)
            await self.send(
                sender_addr,
                "The email challenge queue is busy. Please try again later.",
                reply_to=msg_data.get("message_id") or None,
            )
            return
        body_hash = hashlib.sha256(body.encode("utf-8", errors="replace")).hexdigest()[:12]
        ttl = store.ttl_seconds
        result = await self.send(
            sender_addr,
            "Your email request is pending confirmation.\n\n"
            f"Reply with: /confirm {code}\n\n"
            f"Original subject: {subject or '(no subject)'}\n"
            f"Body SHA256 prefix: {body_hash}\n"
            f"This code expires in {ttl} seconds.",
            reply_to=msg_data.get("message_id") or None,
        )
        if not result.success:
            logger.warning("[Email] Email challenge delivery failed for %s; removing pending original", sender_addr)
            try:
                store.remove(sender_addr, code)
            except Exception as exc:  # noqa: BLE001 - cleanup must not interrupt polling
                logger.warning("[Email] Email challenge removal failed for %s after send failure: %s", sender_addr, exc)
            finally:
                cleanup_challenge_cached_attachments(msg_data)

    async def _handle_challenge_confirmation(
        self,
        sender_addr: str,
        code: str,
        msg_data: Dict[str, Any],
    ) -> None:
        self._thread_context[sender_addr] = {
            "subject": msg_data.get("subject", "Hermes Agent"),
            "message_id": msg_data.get("message_id", ""),
        }
        if len(code) > MAX_EMAIL_CONFIRM_CODE_LENGTH:
            await self.send(
                sender_addr,
                "That confirmation code is invalid or already used.",
                reply_to=msg_data.get("message_id") or None,
            )
            return
        try:
            status, pending = self._email_challenge_store().confirm(sender_addr, code)
        except Exception as exc:  # noqa: BLE001 - fail closed per message on store I/O errors
            logger.warning("[Email] Email challenge confirmation failed for %s: %s", sender_addr, exc)
            await self._send_email_challenge_notice(
                sender_addr,
                "That confirmation code could not be checked temporarily. Please try again later.",
                msg_data,
            )
            return
        if status == "ok" and pending:
            await self._dispatch_message(pending)
            return
        responses = {
            "sender_mismatch": "That confirmation code is invalid or already used.",
            "expired": "That confirmation code has expired. Please resend the original request.",
            "used": "That confirmation code is invalid or already used.",
            "not_found": "That confirmation code is invalid or already used.",
        }
        await self.send(
            sender_addr,
            responses.get(status, "That confirmation code is invalid."),
            reply_to=msg_data.get("message_id") or None,
        )

    async def _send_email_challenge_notice(self, sender_addr: str, content: str, msg_data: Dict[str, Any]) -> None:
        try:
            await self.send(sender_addr, content, reply_to=msg_data.get("message_id") or None)
        except Exception as exc:  # noqa: BLE001 - challenge notices are best-effort
            logger.warning("[Email] Email challenge notice failed for %s: %s", sender_addr, exc)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an email reply to the given address."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None, self._send_email, chat_id, content, reply_to
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
    ) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        # Thread context for reply
        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # Threading headers
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
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
    ) -> SendResult:
        """Send an image URL as part of an email body."""
        text = caption or ""
        text += f"\n\nImage: {image_url}"
        return await self.send(chat_id, text.strip(), reply_to)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single email with multiple MIME attachments.

        Local files are attached directly. URL images have their URL
        appended to the body (email adapter does not download remote
        images). No hard cap — email clients handle dozens of
        attachments fine, subject to SMTP message size limits.
        """
        if not images:
            return

        from urllib.parse import unquote as _unquote

        body_parts: List[str] = []
        local_paths: List[str] = []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if image_url.startswith("file://"):
                local_path = _unquote(image_url[7:])
                if Path(local_path).exists():
                    local_paths.append(local_path)
                else:
                    logger.warning("[Email] Skipping missing image: %s", local_path)
            else:
                # Remote URLs just get linked in the body (parity with send_image)
                body_parts.append(f"Image: {image_url}")

        if not local_paths and not body_parts:
            return

        body = "\n\n".join(body_parts)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._send_email_with_attachments,
                chat_id,
                body,
                local_paths,
            )
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(
        self,
        to_addr: str,
        body: str,
        file_paths: List[str],
    ) -> str:
        """Send an email with multiple file attachments via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._address.split('@')[1]}>"
        msg["Message-ID"] = msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        for file_path in file_paths:
            p = Path(file_path)
            try:
                with open(p, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={p.name}")
                    msg.attach(part)
            except Exception as e:
                logger.warning("[Email] Failed to attach %s: %s", file_path, e)

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

        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        return msg_id

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as an email attachment."""
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None,
                self._send_email_with_attachment,
                chat_id,
                caption or "",
                file_path,
                file_name,
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
    ) -> str:
        """Send an email with a file attachment via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = ctx.get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id

        msg["Date"] = formatdate(localtime=True)
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
        ctx = self._thread_context.get(chat_id, {})
        return {
            "name": chat_id,
            "type": "dm",
            "chat_id": chat_id,
            "subject": ctx.get("subject", ""),
        }
