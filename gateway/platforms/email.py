"""
Email platform adapter for the Hermes gateway.

Allows users to interact with Hermes by sending emails.
Uses IMAP to receive and SMTP to send messages.

Environment variables:
    EMAIL_PROVIDER      — Email provider backend: imap (default) or proton
    EMAIL_IMAP_HOST     — IMAP server host (e.g., imap.gmail.com)
    EMAIL_IMAP_PORT     — IMAP server port (default: 993)
    EMAIL_SMTP_HOST     — SMTP server host (e.g., smtp.gmail.com)
    EMAIL_SMTP_PORT     — SMTP server port (default: 587)
    EMAIL_ADDRESS       — Email address for the agent
    EMAIL_PASSWORD      — Email password or app-specific password
    EMAIL_POLL_INTERVAL — Seconds between mailbox checks (default: 15)
    EMAIL_ALLOWED_USERS — Comma-separated list of allowed sender addresses
    EMAIL_PROTON_SEEN_PATH — Optional persistent dedupe file for Proton ids
    PROTON_CLIENT_FACTORY — module:function returning a Proton API client
    PROTON_MAILBOX      — JSON mailbox path for dry-runs/tests
"""

import asyncio
import email as email_lib
import imaplib
import importlib
import json
import logging
import os
import re
import smtplib
import ssl
import uuid
from email.header import decode_header
from email.utils import parseaddr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate
from email import encoders
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple
from urllib import parse as urlparse
from urllib import request as urlrequest

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
    "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
    "X-Auto-Response-Suppress": lambda v: bool(v),
    "List-Unsubscribe": lambda v: bool(v),
}

# Gmail-safe max length per email body
MAX_MESSAGE_LENGTH = 50_000

# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_PROTON_INITIAL_RECONNECT_DELAY_SECONDS = 5
_PROTON_MAX_RECONNECT_DELAY_SECONDS = 300


class ProtonClient(Protocol):
    """Small protocol for the Proton runtime used in production.

    The actual Proton client is loaded by ``PROTON_CLIENT_FACTORY`` so Hermes
    does not need to depend on a provider-private package.  The production
    client must expose ``event_polling`` and ``get_message`` for inbound mail.
    Outbound email is optional but must be explicit: one of ``send_reply``,
    ``reply_message``, ``send_email``, or ``send_message`` must exist.
    """

    def event_polling(self) -> Iterable[dict[str, Any]]:
        ...

    def get_message(self, email_id: str) -> dict[str, Any]:
        ...


class JsonProtonMailboxClient:
    """Local Proton-shaped mailbox for tests and dry-runs."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.sent: list[dict[str, Any]] = self.data.setdefault("sent", [])

    def event_polling(self) -> Iterable[dict[str, Any]]:
        return self.data.get("events", [])

    def get_message(self, email_id: str) -> dict[str, Any]:
        for message in self.data.get("messages", []):
            if str(message.get("id") or message.get("ID") or message.get("MessageID")) == email_id:
                return message
        raise KeyError(f"Unknown email id: {email_id}")

    def send_reply(
        self,
        *,
        message_id: str | None,
        to: str,
        subject: str,
        body: str,
        **_: Any,
    ) -> str:
        sent_id = f"json-proton-{uuid.uuid4().hex[:12]}"
        self.sent.append({
            "id": sent_id,
            "message_id": message_id,
            "to": to,
            "subject": subject,
            "body": body,
        })
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return sent_id


def _load_proton_client() -> ProtonClient:
    mailbox = os.getenv("PROTON_MAILBOX")
    if mailbox:
        return JsonProtonMailboxClient(Path(mailbox).expanduser())

    factory = os.getenv("PROTON_CLIENT_FACTORY")
    if not factory:
        raise RuntimeError(
            "No Proton client configured. Set PROTON_CLIENT_FACTORY=module:function "
            "or PROTON_MAILBOX=/path/to/mailbox.json for dry-runs."
        )
    module_name, function_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)()


def _model_value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value.get(name)
        if hasattr(value, name):
            return getattr(value, name)
        lower = name[:1].lower() + name[1:]
        if hasattr(value, lower):
            return getattr(value, lower)
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sender_from_value(value: Any) -> tuple[str, str]:
    sender = _as_mapping(value)
    if sender:
        name = str(sender.get("Name") or sender.get("name") or "").strip()
        address = str(
            sender.get("Address")
            or sender.get("address")
            or sender.get("Email")
            or sender.get("email")
            or ""
        ).strip()
        return name, address

    name = str(_model_value(value, "Name", "DisplayName") or "").strip()
    address = str(_model_value(value, "Address", "Email") or "").strip()
    if address:
        return name, address

    parsed_name, parsed_address = parseaddr(str(value or ""))
    return parsed_name, parsed_address


def _sender_from_proton_message(value: Any) -> tuple[str, str]:
    for field in ("sender", "Sender", "from", "From", "from_", "FromAddress"):
        sender_name, sender_email = _sender_from_value(_model_value(value, field))
        if sender_email:
            return sender_name, sender_email

    headers = _as_mapping(_model_value(value, "headers", "Headers"))
    for header in ("From", "from", "Reply-To", "reply-to"):
        sender_name, sender_email = _sender_from_value(headers.get(header))
        if sender_email:
            return sender_name, sender_email

    sender_name = str(_model_value(value, "SenderName", "sender_name", "FromName") or "").strip()
    sender_email = str(
        _model_value(value, "SenderAddress", "sender_address", "Address", "Email", "FromEmail") or ""
    ).strip()
    return sender_name, sender_email


def _extract_proton_event_message(event: Mapping[str, Any]) -> dict[str, Any] | None:
    action = event.get("Action", event.get("action"))
    if action is not None and str(action) != "1":
        return None
    message = (
        event.get("message")
        or event.get("Message")
        or event.get("mail")
        or event.get("Mail")
        or event.get("payload")
    )
    if isinstance(message, dict):
        return message
    if event.get("id") or event.get("ID") or event.get("MessageID"):
        return dict(event)
    return None


def _proton_message_id(value: Mapping[str, Any]) -> str:
    return str(_model_value(value, "id", "ID", "message_id", "MessageID") or "")


def _proton_thread_id(value: Mapping[str, Any]) -> str:
    return str(
        _model_value(
            value,
            "thread_id",
            "ThreadID",
            "ConversationID",
            "conversation_id",
        )
        or ""
    )


def _proton_subject(value: Mapping[str, Any]) -> str:
    return str(_model_value(value, "subject", "Subject") or "(no subject)")


def _proton_body(value: Mapping[str, Any]) -> str:
    body = str(
        _model_value(value, "body_text", "BodyText", "text", "Text", "body", "Body")
        or ""
    )
    if body:
        return body
    html = _model_value(value, "body_html", "BodyHTML", "HTML", "Html")
    if html:
        return _strip_html(str(html))
    return ""


def _redact_email_for_log(address: str) -> str:
    local, _, domain = address.partition("@")
    if not domain:
        return "[redacted]"
    return f"{local[:2]}***@{domain}"


def _call_proton_outbound(
    method: Any,
    *,
    message_id: str | None,
    thread_id: str | None,
    to_addr: str,
    subject: str,
    body: str,
) -> Any:
    attempts = [
        {"message_id": message_id, "thread_id": thread_id, "to": to_addr, "subject": subject, "body": body},
        {"message_id": message_id, "to": to_addr, "subject": subject, "body": body},
        {"email_id": message_id, "thread_id": thread_id, "to": to_addr, "subject": subject, "body": body},
        {"email_id": message_id, "to": to_addr, "subject": subject, "body": body},
        {"to": to_addr, "subject": subject, "body": body},
    ]
    last_error: TypeError | None = None
    for kwargs in attempts:
        try:
            return method(**kwargs)
        except TypeError as e:
            last_error = e
    if last_error is not None:
        raise last_error
    raise RuntimeError("Proton outbound method could not be called")

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
    provider = os.getenv("EMAIL_PROVIDER", "imap").strip().lower()
    addr = os.getenv("EMAIL_ADDRESS")
    if provider == "proton":
        return bool(addr and (os.getenv("PROTON_CLIENT_FACTORY") or os.getenv("PROTON_MAILBOX")))
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
        if content_type in {"text/plain", "text/html"} and "attachment" not in disposition:
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
    """Email gateway adapter using IMAP/SMTP or Proton API transport."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        self._provider = os.getenv("EMAIL_PROVIDER", "imap").strip().lower() or "imap"
        self._address = os.getenv("EMAIL_ADDRESS", "")
        self._password = os.getenv("EMAIL_PASSWORD", "")
        self._imap_host = os.getenv("EMAIL_IMAP_HOST", "")
        self._imap_port = int(os.getenv("EMAIL_IMAP_PORT", "993"))
        self._smtp_host = os.getenv("EMAIL_SMTP_HOST", "")
        self._smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
        self._poll_interval = int(os.getenv("EMAIL_POLL_INTERVAL", "15"))
        allowed_raw = os.getenv("EMAIL_ALLOWED_USERS", "").strip()
        self._allowed_users = {
            addr.strip().lower()
            for addr in allowed_raw.split(",")
            if addr.strip()
        }

        # Skip attachments — configured via config.yaml:
        #   platforms:
        #     email:
        #       skip_attachments: true
        extra = config.extra or {}
        self._skip_attachments = extra.get("skip_attachments", False)

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._proton_client: Optional[ProtonClient] = None
        self._proton_reconnect_delay = _PROTON_INITIAL_RECONNECT_DELAY_SECONDS
        self._proton_seen_path = Path(
            os.getenv(
                "EMAIL_PROTON_SEEN_PATH",
                os.path.expanduser("~/.cache/hermes/proton-email-seen.json"),
            )
        )

        # Map chat_id (sender email) -> last subject + message-id for threading
        self._thread_context: Dict[str, Dict[str, str]] = {}

        logger.info("[Email] Adapter initialized for %s using %s provider", self._address, self._provider)

    def _load_persistent_seen_uids(self) -> None:
        if self._provider != "proton":
            return
        try:
            if not self._proton_seen_path.exists():
                return
            data = json.loads(self._proton_seen_path.read_text(encoding="utf-8"))
            ids = data.get("seen_ids", data if isinstance(data, list) else [])
            self._seen_uids.update(str(item) for item in ids if item)
            self._trim_seen_uids()
            logger.info("[Email] Loaded %d Proton seen ids", len(self._seen_uids))
        except Exception as e:
            logger.warning("[Email] Failed to load Proton seen ids: %s", e)

    def _save_persistent_seen_uids(self) -> None:
        if self._provider != "proton":
            return
        try:
            self._proton_seen_path.parent.mkdir(parents=True, exist_ok=True)
            items = sorted(str(item) for item in self._seen_uids)[-self._seen_uids_max:]
            self._proton_seen_path.write_text(json.dumps({"seen_ids": items}, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[Email] Failed to persist Proton seen ids: %s", e)

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
        """Connect to the configured email backend and start polling."""
        if self._provider == "proton":
            try:
                self._proton_client = _load_proton_client()
                self._load_persistent_seen_uids()
                logger.info("[Email] Proton client loaded for %s", self._address)
            except Exception as e:
                logger.error("[Email] Proton client initialization failed: %s", e)
                return False

            self._running = True
            self._poll_task = asyncio.create_task(self._poll_loop())
            print(f"[Email] Connected as {self._address} via Proton")
            return True

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
        """Poll the configured email backend for new messages."""
        while self._running:
            try:
                await self._check_inbox()
                self._proton_reconnect_delay = _PROTON_INITIAL_RECONNECT_DELAY_SECONDS
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._provider == "proton":
                    delay = self._proton_reconnect_delay
                    logger.warning(
                        "[Email] Proton poll failed with %s; retrying in %ss",
                        type(e).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    self._proton_reconnect_delay = min(
                        self._proton_reconnect_delay * 2,
                        _PROTON_MAX_RECONNECT_DELAY_SECONDS,
                    )
                    continue
                logger.error("[Email] Poll error: %s", e)
            await asyncio.sleep(self._poll_interval)

    async def _check_inbox(self) -> None:
        """Check INBOX for unseen messages and dispatch them."""
        # Run IMAP operations in a thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        if self._provider == "proton":
            messages = await loop.run_in_executor(None, self._fetch_new_proton_messages)
        else:
            messages = await loop.run_in_executor(None, self._fetch_new_messages)
        for msg_data in messages:
            await self._dispatch_message(msg_data)

    def _fetch_new_proton_messages(self) -> List[Dict[str, Any]]:
        """Fetch new Proton events. Runs in executor thread."""
        if self._proton_client is None:
            raise RuntimeError("Proton client is not initialized")

        results: List[Dict[str, Any]] = []
        for event in self._proton_client.event_polling():
            message = _extract_proton_event_message(event)
            if not message:
                continue
            message_id = _proton_message_id(message)
            if not message_id or message_id in self._seen_uids:
                continue

            raw_message = dict(message)
            sender_name, raw_sender = _sender_from_proton_message(raw_message)
            parsed_name, parsed_email = parseaddr(str(raw_sender))
            sender_addr = str(parsed_email or raw_sender).strip().lower()

            body = _proton_body(raw_message)
            should_hydrate = not sender_addr or not body
            if should_hydrate:
                try:
                    full_message = self._proton_client.get_message(message_id)
                    raw_message = {**raw_message, **full_message, "id": message_id}
                    full_sender_name, full_raw_sender = _sender_from_proton_message(raw_message)
                    full_parsed_name, full_parsed_email = parseaddr(str(full_raw_sender))
                    sender_name = full_sender_name or sender_name or full_parsed_name
                    sender_addr = str(full_parsed_email or full_raw_sender or sender_addr).strip().lower()
                    body = _proton_body(raw_message)
                except Exception as e:
                    logger.warning("[Email] Proton hydrate failed for id=%s: %s", message_id, type(e).__name__)

            if not sender_addr:
                logger.warning("[Email] Proton event id=%s has no sender after hydration", message_id)
                self._seen_uids.add(message_id)
                self._save_persistent_seen_uids()
                continue

            subject = _proton_subject(raw_message)
            msg_headers = dict(_as_mapping(_model_value(raw_message, "headers", "Headers")))
            if _is_automated_sender(sender_addr, msg_headers):
                logger.debug("[Email] Skipping automated Proton sender: %s", _redact_email_for_log(sender_addr))
                self._seen_uids.add(message_id)
                self._save_persistent_seen_uids()
                continue

            self._seen_uids.add(message_id)
            self._trim_seen_uids()
            self._save_persistent_seen_uids()

            results.append({
                "uid": message_id,
                "sender_addr": sender_addr,
                "sender_name": sender_name or parsed_name,
                "subject": subject,
                "message_id": message_id,
                "thread_id": _proton_thread_id(raw_message),
                "in_reply_to": str(_model_value(raw_message, "InReplyTo", "in_reply_to") or ""),
                "body": body,
                "attachments": [],
                "date": str(_model_value(raw_message, "Date", "Time", "date") or ""),
                "raw": raw_message,
            })
        return results

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

        # Skip senders not in EMAIL_ALLOWED_USERS — prevents the adapter
        # from creating a MessageEvent (and thus thread context) for senders
        # that the gateway will never authorize.  Without this early guard,
        # a race between dispatch and authorization can result in the adapter
        # sending a reply even though the handler returned None.
        if self._allowed_users:
            if sender_addr.lower() not in self._allowed_users:
                logger.debug("[Email] Dropping non-allowlisted sender at dispatch: %s", _redact_email_for_log(sender_addr))
                await self._send_passive_notification(msg_data)
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

        # Store thread context for reply threading
        self._thread_context[sender_addr] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
            "thread_id": msg_data.get("thread_id", ""),
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

        logger.info("[Email] New message from %s: %s", _redact_email_for_log(sender_addr), subject)
        await self.handle_message(event)

    async def _send_passive_notification(self, msg_data: Dict[str, Any]) -> None:
        """Notify the owner about non-trusted email without triggering an agent turn."""
        bot_token = (
            os.getenv("EMAIL_PASSIVE_TELEGRAM_BOT_TOKEN")
            or os.getenv("HERMES_TELEGRAM_BOT_TOKEN")
            or os.getenv("TELEGRAM_BOT_TOKEN")
        )
        chat_id = (
            os.getenv("EMAIL_PASSIVE_TELEGRAM_CHAT_ID")
            or os.getenv("HERMES_TELEGRAM_CHAT_ID")
            or os.getenv("TELEGRAM_CHAT_ID")
        )
        if not bot_token or not chat_id:
            return

        sender = msg_data.get("sender_addr", "")
        if msg_data.get("sender_name"):
            sender = f"{msg_data['sender_name']} <{sender}>"
        text = f"Nouveau mail De: {sender} Sujet: {msg_data.get('subject') or '(no subject)'}"

        def _post() -> None:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = urlparse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
            urlrequest.urlopen(url, data=data, timeout=10).read()

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _post)
        except Exception as e:
            logger.warning("[Email] Passive notification failed: %s", type(e).__name__)

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
            if self._provider == "proton":
                message_id = await loop.run_in_executor(
                    None, self._send_proton_email, chat_id, content, reply_to
                )
            else:
                message_id = await loop.run_in_executor(
                    None, self._send_email, chat_id, content, reply_to
                )
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send failed to %s: %s", _redact_email_for_log(chat_id), e)
            return SendResult(success=False, error=str(e))

    def _send_proton_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
    ) -> str:
        """Send an email through the configured Proton runtime."""
        client = self._proton_client or _load_proton_client()
        ctx = self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        thread_id = ctx.get("thread_id") or None

        for method_name in ("send_reply", "reply_message"):
            method = getattr(client, method_name, None)
            if callable(method):
                result = _call_proton_outbound(
                    method,
                    message_id=original_msg_id,
                    thread_id=thread_id,
                    to_addr=to_addr,
                    subject=subject,
                    body=body,
                )
                sent_id = str(result.get("id") if isinstance(result, Mapping) else result)
                logger.info("[Email] Sent Proton reply to %s (subject: %s)", _redact_email_for_log(to_addr), subject)
                return sent_id

        for method_name in ("send_email", "send_message"):
            method = getattr(client, method_name, None)
            if callable(method):
                result = _call_proton_outbound(
                    method,
                    message_id=original_msg_id,
                    thread_id=thread_id,
                    to_addr=to_addr,
                    subject=subject,
                    body=body,
                )
                sent_id = str(result.get("id") if isinstance(result, Mapping) else result)
                logger.info("[Email] Sent Proton email to %s (subject: %s)", _redact_email_for_log(to_addr), subject)
                return sent_id

        raise RuntimeError(
            "Proton outbound email is unavailable: configured client exposes no "
            "send_reply, reply_message, send_email, or send_message method"
        )

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

        logger.info("[Email] Sent reply to %s (subject: %s)", _redact_email_for_log(to_addr), subject)
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
