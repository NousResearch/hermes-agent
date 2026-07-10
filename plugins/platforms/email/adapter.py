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
import mimetypes
import os
import re
import smtplib
import socket
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
from utils import env_int, env_bool

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

SMTP_CONNECT_TIMEOUT = 30


def _create_ipv4_connection(
    host: str,
    port: int,
    timeout: float,
    source_address: Any = None,
) -> socket.socket:
    """Create a TCP connection using only IPv4 addresses.

    This mirrors ``socket.create_connection`` but constrains DNS resolution to
    ``AF_INET``.  It avoids mutating process-global socket functions, which
    matters because email sends run in executor threads.
    """
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(
            host,
            port,
            timeout,
            source_address=self.source_address,
        )


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        raw_sock = _create_ipv4_connection(
            host,
            port,
            timeout,
            source_address=self.source_address,
        )
        return self.context.wrap_socket(
            raw_sock,
            server_hostname=getattr(self, "_host", host),
        )

# Supported image extensions for inline detection
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_PROTON_INITIAL_RECONNECT_DELAY_SECONDS = 5
_PROTON_MAX_RECONNECT_DELAY_SECONDS = 300
# Emit a liveness line at most this often so a stalled poller is obvious in logs
# instead of silently going dark (prod once went 5 days with zero email lines).
_POLL_HEARTBEAT_INTERVAL_SECONDS = 900


def _poll_watchdog_timeout(poll_interval: float) -> float:
    """Maximum seconds a single inbox check may run before the poll loop treats
    it as hung and forces a retry.

    A blocking backend read with no socket timeout (notably the Proton
    ``event_polling`` path, which — unlike IMAP — has no timeout of its own) can
    otherwise leave the loop awaiting forever: no error, no retry, no logs. The
    watchdog bounds every check so the loop always recovers. Generously above
    the normal cadence so it only trips on a genuinely stuck read; override via
    ``EMAIL_POLL_TIMEOUT_SECONDS``.
    """
    raw = os.getenv("EMAIL_POLL_TIMEOUT_SECONDS")
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return max(120.0, poll_interval * 4)
_DEFAULT_BLOCKED_ATTACHMENT_EXTS = {
    ".env",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".sqlite",
    ".sqlite3",
    ".db",
}
_DEFAULT_BLOCKED_ATTACHMENT_MIME_TYPES = {
    "application/x-pem-file",
    "application/x-pkcs12",
}
_SECRET_LIKE_FILENAME_RE = re.compile(
    r"(^|[-_.])(secret|secrets|credential|credentials|token|password|passwd|private[_-]?key|id_rsa|auth)([-_.]|$)",
    re.IGNORECASE,
)


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
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        sent_id = f"json-proton-{uuid.uuid4().hex[:12]}"
        self.sent.append({
            "id": sent_id,
            "message_id": message_id,
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body": body,
            "attachments": attachments or [],
        })
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        return {"sent": True, "id": sent_id}

    def send_email(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.send_reply(
            message_id=kwargs.get("message_id"),
            to=to,
            subject=subject,
            body=body,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
        )


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


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _normalise_email_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts: list[str] = []
        for chunk in re.split(r"[,;\n]", value):
            _name, addr = parseaddr(chunk.strip())
            if addr:
                parts.append(addr.lower())
        return parts
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            raw = item.get("address") or item.get("Address") or item.get("email") or item.get("Email") or ""
        else:
            raw = str(item)
        _name, addr = parseaddr(raw)
        if addr:
            result.append(addr.lower())
    return result


def _recipient_policy(
    *,
    to_addr: str,
    cc: Any = None,
    bcc: Any = None,
) -> dict[str, Any]:
    recipients = _normalise_email_list([to_addr]) + _normalise_email_list(cc) + _normalise_email_list(bcc)
    unique = list(dict.fromkeys(recipients))
    max_recipients = int(os.getenv("EMAIL_MAX_RECIPIENTS", "10"))
    allowed = {"thomas@lfglabs.dev"}
    allowed.update(_normalise_email_list(os.getenv("EMAIL_ALLOWED_RECIPIENTS")))
    allowed.update(_normalise_email_list(os.getenv("EMAIL_ALLOWED_USERS")))
    allowed.update(_normalise_email_list(os.getenv("EMAIL_OWNER_RECIPIENTS")))
    allowed_domains = {d.strip().lower().lstrip("@") for d in _split_csv(os.getenv("EMAIL_ALLOWED_DOMAINS"))}
    allow_unknown = os.getenv("EMAIL_ALLOW_UNKNOWN_RECIPIENTS", "").strip().lower() in {"1", "true", "yes", "on"}
    kill_switch = os.getenv("EMAIL_OUTBOUND_DISABLED", "").strip().lower() in {"1", "true", "yes", "on"}

    rejected: list[str] = []
    reason = ""
    if kill_switch:
        rejected = unique
        reason = "outbound email is disabled by EMAIL_OUTBOUND_DISABLED"
    elif len(unique) > max_recipients:
        rejected = unique[max_recipients:]
        reason = f"too many recipients: {len(unique)} > {max_recipients}"
    else:
        for recipient in unique:
            domain = recipient.rsplit("@", 1)[-1] if "@" in recipient else ""
            if recipient in allowed or domain in allowed_domains or allow_unknown:
                continue
            rejected.append(recipient)
        if rejected:
            reason = "recipient is not approved by outbound email policy"

    accepted = [r for r in unique if r not in set(rejected)]
    return {
        "allowed": not rejected and bool(unique),
        "accepted_recipients": accepted,
        "rejected_recipients": rejected,
        "failure_reason": reason,
    }


def _attachment_allowed_roots() -> list[Path]:
    roots = [Path.cwd(), Path("/tmp")]
    roots.extend(Path(p).expanduser() for p in _split_csv(os.getenv("EMAIL_ATTACHMENT_ALLOWED_ROOTS")))
    return [root.resolve() for root in roots]


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _attachment_manifest(paths: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        return [], []
    if isinstance(paths, (str, os.PathLike)):
        raw_paths = [paths]
    else:
        raw_paths = list(paths)
    max_count = int(os.getenv("EMAIL_MAX_ATTACHMENTS", "5"))
    max_total = int(os.getenv("EMAIL_MAX_ATTACHMENT_BYTES", "10485760"))
    blocked_exts = {e.lower() for e in (_split_csv(os.getenv("EMAIL_BLOCKED_ATTACHMENT_EXTS")) or _DEFAULT_BLOCKED_ATTACHMENT_EXTS)}
    blocked_mimes = set(_split_csv(os.getenv("EMAIL_BLOCKED_ATTACHMENT_MIME_TYPES")) or _DEFAULT_BLOCKED_ATTACHMENT_MIME_TYPES)
    roots = _attachment_allowed_roots()

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total = 0
    for raw in raw_paths:
        path = Path(raw).expanduser().resolve()
        filename = re.sub(r"[^A-Za-z0-9_. -]+", "_", path.name).strip() or "attachment"
        guessed_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        item = {"path": str(path), "filename": filename, "content_type": guessed_type, "size": 0}
        if len(accepted) >= max_count:
            rejected.append({**item, "decision": "rejected", "reason": "too many attachments"})
            continue
        if not path.exists() or not path.is_file():
            rejected.append({**item, "decision": "rejected", "reason": "attachment not found"})
            continue
        size = path.stat().st_size
        item["size"] = size
        suffix = path.suffix.lower()
        if not any(_path_within(path, root) for root in roots):
            rejected.append({**item, "decision": "rejected", "reason": "attachment path is outside approved roots"})
            continue
        if suffix in blocked_exts or guessed_type in blocked_mimes:
            rejected.append({**item, "decision": "rejected", "reason": "attachment type is blocked"})
            continue
        if _SECRET_LIKE_FILENAME_RE.search(filename):
            rejected.append({**item, "decision": "rejected", "reason": "attachment filename looks secret-bearing"})
            continue
        if total + size > max_total:
            rejected.append({**item, "decision": "rejected", "reason": "total attachment size limit exceeded"})
            continue
        total += size
        accepted.append({**item, "decision": "accepted"})
    return accepted, rejected


def _structured_proton_result(
    raw_result: Any,
    *,
    accepted_recipients: list[str],
    rejected_recipients: list[str],
    attachments: list[dict[str, Any]],
    rejected_attachments: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
    message_id = (
        result.get("id")
        or result.get("message_id")
        or result.get("MessageID")
        or (str(raw_result) if raw_result is not None and not isinstance(raw_result, Mapping) else "")
    )
    sent = bool(result.get("sent", True)) and bool(message_id)
    return {
        "sent": sent,
        "id": str(message_id) if message_id else "",
        "message_id": str(message_id) if message_id else "",
        "accepted_recipients": result.get("accepted_recipients", accepted_recipients),
        "rejected_recipients": result.get("rejected_recipients", rejected_recipients),
        "attachments": result.get("attachments", attachments),
        "rejected_attachments": result.get("rejected_attachments", rejected_attachments),
        "failure_reason": result.get("failure_reason", "" if sent else "Proton did not return a sent message id"),
    }


def _call_proton_outbound(
    method: Any,
    *,
    message_id: str | None,
    thread_id: str | None,
    to_addr: str,
    subject: str,
    body: str,
    cc: list[str] | None = None,
    bcc: list[str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    is_html: bool = False,
) -> Any:
    attempts = [
        {"message_id": message_id, "thread_id": thread_id, "to": to_addr, "subject": subject, "body": body, "cc": cc or [], "bcc": bcc or [], "attachments": attachments or [], "is_html": is_html},
        {"message_id": message_id, "to": to_addr, "subject": subject, "body": body, "cc": cc or [], "bcc": bcc or [], "attachments": attachments or []},
        {"email_id": message_id, "thread_id": thread_id, "to": to_addr, "subject": subject, "body": body, "attachments": attachments or []},
        {"email_id": message_id, "to": to_addr, "subject": subject, "body": body, "attachments": attachments or []},
        {"to": to_addr, "subject": subject, "body": body, "attachments": attachments or []},
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
    """Check if email platform settings are available and non-blank.

    Treats blank/whitespace-only values as missing so an abandoned setup that
    left empty ``EMAIL_*`` keys in ``.env`` does not enable the platform (#40715).
    For the Proton provider a client factory / mailbox gates availability
    instead of the IMAP/SMTP host credentials.
    """
    provider = os.getenv("EMAIL_PROVIDER", "imap").strip().lower()
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
    if provider == "proton":
        return bool(addr and (os.getenv("PROTON_CLIENT_FACTORY") or os.getenv("PROTON_MAILBOX")))
    pwd = os.getenv("EMAIL_PASSWORD", "").strip()
    imap = os.getenv("EMAIL_IMAP_HOST", "").strip()
    smtp = os.getenv("EMAIL_SMTP_HOST", "").strip()
    return all([addr, pwd, imap, smtp])


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


def _domain_of(address: str) -> str:
    """Return the lowercased domain part of an email address, or ''."""
    _, _, domain = address.rpartition("@")
    return domain.strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Return True if two domains are equal or in an organizational
    parent/subdomain relationship (relaxed DMARC alignment).

    DMARC relaxed alignment treats ``mail.example.com`` as aligned with
    ``example.com``. We approximate organizational alignment by checking
    exact equality or that one domain is a dot-suffix of the other.
    """
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    if not a or not b:
        return False
    if a == b:
        return True
    return a.endswith("." + b) or b.endswith("." + a)


# Match a single "method=result" token in an Authentication-Results header,
# e.g. ``dmarc=pass`` or ``spf=fail``.
_AUTH_METHOD_RE = re.compile(
    r"\b(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE
)
# Match a property value like ``header.from=example.com`` or
# ``smtp.mailfrom=user@example.com``.
_AUTH_PROP_RE = re.compile(
    r"\b(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*([^\s;]+)",
    re.IGNORECASE,
)


def _verify_sender_authentication(
    msg: email_lib.message.Message,
    from_addr: str,
    *,
    authserv_id: str = "",
) -> Tuple[bool, str]:
    """Verify that the message's ``From:`` domain is authenticated.

    The ``From:`` header is attacker-controlled and is never authenticated by
    IMAP delivery, so an allowlist keyed on ``From:`` alone is trivially
    spoofable (GHSA-rxqh-5572-8m77). The only trustworthy signal is the
    ``Authentication-Results`` header that the *receiving* mail server (the one
    we IMAP into) stamps after running SPF/DKIM/DMARC. That header is prepended
    by our own server, so the topmost instance is the one we trust; any
    ``Authentication-Results`` an attacker injected into the body of their
    message sorts below it.

    Returns ``(authenticated, reason)``. ``authenticated`` is True when:
      * a DMARC pass is recorded for the From domain, OR
      * an SPF pass aligned with the From domain, OR
      * a DKIM pass aligned (``header.d``) with the From domain.

    When no ``Authentication-Results`` header is present at all, we return
    ``(False, "no Authentication-Results header")`` — fail-closed. Operators
    whose mail server does not stamp this header can opt out of the check
    (see ``EmailAdapter._require_authenticated_sender``).
    """
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"

    # get_all preserves header order; the receiving server prepends its result,
    # so the FIRST Authentication-Results is the trusted one. We pin to the
    # configured authserv-id when provided to defend against an injected header
    # that happens to sort first.
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return False, "no Authentication-Results header"

    trusted = None
    for raw in headers:
        value = " ".join(str(raw).split())
        if authserv_id:
            # authserv-id is the first token before the first ';'
            serv = value.split(";", 1)[0].strip().lower()
            if not _domains_aligned(serv, authserv_id) and serv != authserv_id.lower():
                continue
        trusted = value
        break
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"

    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}

    # 1) DMARC pass is the strongest signal — DMARC already enforces From
    #    alignment, so a pass means the From domain is authenticated.
    if methods.get("dmarc") == "pass":
        return True, "dmarc=pass"

    # 2) SPF pass aligned with the From domain (the envelope/MAIL FROM domain
    #    must match the From domain).
    if methods.get("spf") == "pass":
        spf_domain = _domain_of(props.get("smtp.mailfrom", "")) or props.get(
            "smtp.from", ""
        ) or props.get("envelope-from", "")
        spf_domain = _domain_of(spf_domain) if "@" in spf_domain else spf_domain
        if _domains_aligned(spf_domain, from_domain):
            return True, "spf=pass aligned"

    # 3) DKIM pass aligned with the From domain (the signing domain header.d
    #    must align with the From domain).
    if methods.get("dkim") == "pass":
        dkim_domain = props.get("header.d", "") or _domain_of(props.get("header.from", ""))
        if _domains_aligned(dkim_domain, from_domain):
            return True, "dkim=pass aligned"

    return False, f"authentication failed ({trusted[:120]})"


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
        # Resolve connection settings from the env vars first, then fall back to
        # PlatformConfig.extra (address/imap_host/smtp_host) — the canonical dict
        # gateway.config populates and that the "connected" check, the
        # send-helper, and `hermes config show` already read. Without the
        # fallback a config.yaml-only setup left these empty. Host/address values
        # are stripped: a stray space or newline made IMAP4_SSL raise the
        # misleading ``[Errno 8] nodename nor servname`` (an unresolvable name)
        # instead of an obvious "host not set" error.
        extra = config.extra or {}
        # `or ""` (not the .get default) because gateway.config can populate
        # these keys with an explicit None (e.g. a Proton setup leaves imap/smtp
        # host unset → None); .get(key, "") only defaults when the key is ABSENT,
        # so a present-but-None value would reach .strip() and crash.
        self._address = (os.getenv("EMAIL_ADDRESS", "") or extra.get("address") or "").strip()
        self._password = os.getenv("EMAIL_PASSWORD", "")
        self._imap_host = (os.getenv("EMAIL_IMAP_HOST", "") or extra.get("imap_host") or "").strip()
        self._imap_port = env_int("EMAIL_IMAP_PORT", 993)
        self._smtp_host = (os.getenv("EMAIL_SMTP_HOST", "") or extra.get("smtp_host") or "").strip()
        self._smtp_port = env_int("EMAIL_SMTP_PORT", 587)
        self._poll_interval = env_int("EMAIL_POLL_INTERVAL", 15)
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
        self._skip_attachments = extra.get("skip_attachments", False)

        # Require the sender's From: domain to be authenticated (SPF/DKIM/DMARC)
        # before trusting it for authorization. The From: header is
        # attacker-controlled and unauthenticated by IMAP, so an allowlist keyed
        # on it alone is spoofable (GHSA-rxqh-5572-8m77). Default ON (fail-closed).
        #
        # Operators whose receiving mail server does not stamp an
        # Authentication-Results header can opt out via config.yaml:
        #   platforms:
        #     email:
        #       require_authenticated_sender: false
        # or the EMAIL_TRUST_FROM_HEADER=true env mirror (parity with the other
        # EMAIL_* access-control vars). When allow-all is in effect the operator
        # has already chosen to accept any sender, so the check is moot and the
        # gate below is skipped.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        elif env_bool("EMAIL_TRUST_FROM_HEADER", False):
            self._require_authenticated_sender = False
        else:
            self._require_authenticated_sender = True

        # Optional authserv-id to pin Authentication-Results to the operator's
        # own receiving server (defends against an injected header that sorts
        # first). Defaults to the From-domain of the agent's own address.
        self._authserv_id = (
            extra.get("authserv_id", "") or os.getenv("EMAIL_AUTHSERV_ID", "")
        ).strip().lower()

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

    def _connect_smtp(self) -> smtplib.SMTP:
        """Create an SMTP connection, selecting the correct protocol for the port.

        Port 465 uses implicit TLS (``SMTP_SSL``).  All other ports use
        ``SMTP`` + ``STARTTLS``.

        When the host resolves to an IPv6 address that is unreachable
        (common on networks without IPv6 routing), the default connection can
        hang until the socket timeout expires.  We retry connection-level
        failures through an IPv4-only socket path, without mutating global
        resolver state.  TLS verification errors are not retried.

        Returns a connected SMTP object with TLS established — callers
        can proceed directly to ``login()``.
        """
        ctx = ssl.create_default_context()
        host = self._smtp_host
        port = self._smtp_port

        def _connect(*, ipv4_only: bool = False) -> smtplib.SMTP:
            """Attempt one SMTP connection."""
            smtp_cls = _IPv4SMTP if ipv4_only else smtplib.SMTP
            smtp_ssl_cls = _IPv4SMTP_SSL if ipv4_only else smtplib.SMTP_SSL
            if port == 465:
                return smtp_ssl_cls(host, port, timeout=SMTP_CONNECT_TIMEOUT, context=ctx)
            smtp = smtp_cls(host, port, timeout=SMTP_CONNECT_TIMEOUT)
            try:
                smtp.starttls(context=ctx)
            except Exception:
                smtp.close()
                raise
            return smtp

        try:
            return _connect()
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            # Connection-level failure (may be unreachable IPv6).
            # Retry with IPv4 only.
            return _connect(ipv4_only=True)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the configured email backend and start polling."""
        # Proton uses the API transport, not IMAP/SMTP — load its client and
        # start the poll loop directly.
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
            self._poll_task.add_done_callback(self._on_poll_task_done)
            print(f"[Email] Connected as {self._address} via Proton")
            return True

        # IMAP/SMTP path: validate up front so a missing host surfaces as an
        # actionable config error instead of IMAP4_SSL("") raising the cryptic
        # ``[Errno 8] nodename nor servname provided, or not known``.
        missing = [
            name
            for name, value in (
                ("EMAIL_ADDRESS", self._address),
                ("EMAIL_PASSWORD", self._password),
                ("EMAIL_IMAP_HOST", self._imap_host),
                ("EMAIL_SMTP_HOST", self._smtp_host),
            )
            if not value
        ]
        if missing:
            message = (
                "Not configured — missing "
                + ", ".join(missing)
                + ". Set it via `hermes gateway setup` (env) or platforms.email "
                "in config.yaml."
            )
            logger.error("[Email] %s", message)
            # Mark non-retryable so the gateway does NOT keep reconnecting against
            # an empty host. A blank-but-present env var (e.g. ``EMAIL_IMAP_HOST=``)
            # used to slip past the startup gate and drive an indefinite retry
            # loop that leaked memory until the host OOM-killed (#40715).
            self._set_fatal_error(
                "email_missing_configuration", message, retryable=False
            )
            return False

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
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
        except Exception as e:
            logger.error("[Email] SMTP connection failed: %s", e)
            return False

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._poll_task.add_done_callback(self._on_poll_task_done)
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

    def _on_poll_task_done(self, task: "asyncio.Task") -> None:
        """Surface an unexpected poll-loop exit loudly.

        The loop is meant to run until ``disconnect``. If it ever ends while
        ``_running`` is set, inbound mail has stopped — which must never happen
        silently (the prod symptom was exactly this: no logs, no alerts).
        """
        if not self._running:
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc is not None:
            logger.error(
                "[Email] poll loop crashed (%s); inbound mail has stopped: %r",
                type(exc).__name__,
                exc,
            )
        else:
            logger.error(
                "[Email] poll loop exited while running; inbound mail has stopped"
            )

    async def _poll_loop(self) -> None:
        """Poll the configured email backend for new messages.

        Hardened so the poller can never silently stall: every inbox check is
        bounded by a watchdog timeout (a hung backend read would otherwise block
        this loop forever — observed in prod as days of total silence), and a
        periodic heartbeat makes liveness visible in the logs.
        """
        watchdog = _poll_watchdog_timeout(self._poll_interval)
        loop = asyncio.get_running_loop()
        last_heartbeat = loop.time()
        while self._running:
            try:
                await asyncio.wait_for(self._check_inbox(), timeout=watchdog)
                self._proton_reconnect_delay = _PROTON_INITIAL_RECONNECT_DELAY_SECONDS
            except asyncio.CancelledError:
                break
            except Exception as e:
                # asyncio.TimeoutError (the watchdog firing) is an Exception, so
                # it lands here too and is treated like any transient failure.
                reason = (
                    "timed out (watchdog)"
                    if isinstance(e, asyncio.TimeoutError)
                    else f"failed with {type(e).__name__}"
                )
                if self._provider == "proton":
                    delay = self._proton_reconnect_delay
                    logger.warning(
                        "[Email] Proton poll %s; retrying in %ss", reason, delay
                    )
                    await asyncio.sleep(delay)
                    self._proton_reconnect_delay = min(
                        self._proton_reconnect_delay * 2,
                        _PROTON_MAX_RECONNECT_DELAY_SECONDS,
                    )
                    continue
                logger.error("[Email] Poll %s", reason)
            # Heartbeat so a prolonged lack of inbound is diagnosable from logs.
            now = loop.time()
            if now - last_heartbeat >= _POLL_HEARTBEAT_INTERVAL_SECONDS:
                logger.info(
                    "[Email] poller alive (provider=%s, %d ids seen)",
                    self._provider,
                    len(self._seen_uids),
                )
                last_heartbeat = now
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
                "sender_authenticated": True,
                "auth_reason": "proton-api",
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

                    # IMAP fetch can return unexpected structures (e.g. a
                    # single bytes item instead of a list of tuples). Guard
                    # against IndexError / TypeError so one malformed response
                    # doesn't abort the batch — the UID is already in
                    # _seen_uids, so an abort would permanently skip the
                    # remaining messages in this batch.
                    try:
                        raw_email = msg_data[0][1]
                    except (IndexError, TypeError):
                        logger.warning(
                            "[Email] Unexpected IMAP response structure for UID %s, skipping",
                            uid,
                        )
                        continue
                    if not isinstance(raw_email, (bytes, bytearray)):
                        logger.warning(
                            "[Email] Non-bytes IMAP payload for UID %s, skipping", uid
                        )
                        continue
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

                    # Verify the From: domain is authenticated (SPF/DKIM/DMARC)
                    # while the raw message — and its trusted
                    # Authentication-Results header — is still in scope. The
                    # verdict is consumed at dispatch where authorization is
                    # decided. From: is attacker-controlled, so this is the only
                    # place a spoof can be caught (GHSA-rxqh-5572-8m77).
                    sender_authenticated, auth_reason = _verify_sender_authentication(
                        msg, sender_addr, authserv_id=self._authserv_id
                    )

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
                        "sender_authenticated": sender_authenticated,
                        "auth_reason": auth_reason,
                    })
            finally:
                try:
                    imap.logout()
                except Exception:
                    pass
        except Exception as e:
            logger.error("[Email] IMAP fetch error: %s", e)
        return results

    @staticmethod
    def _allow_all_senders() -> bool:
        """Return True when the operator opted into accepting any sender.

        Mirrors the gateway authz allow-all resolution: the per-platform
        EMAIL_ALLOW_ALL_USERS flag or the global GATEWAY_ALLOW_ALL_USERS flag.
        When either is set, sender identity is moot, so the From: authentication
        gate is skipped.
        """
        truthy = {"true", "1", "yes"}
        return (
            os.getenv("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in truthy
            or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in truthy
        )

    @staticmethod
    def _allowlist_in_effect() -> bool:
        """Return True when a sender allowlist gates email access.

        Authorization keys on the From: address only when an allowlist is
        configured — the per-platform EMAIL_ALLOWED_USERS or the global
        GATEWAY_ALLOWED_USERS. When neither is set the gateway default-denies
        every sender regardless, so the spoofable From: identity grants nothing
        and the authentication gate is unnecessary.
        """
        return bool(
            os.getenv("EMAIL_ALLOWED_USERS", "").strip()
            or os.getenv("GATEWAY_ALLOWED_USERS", "").strip()
        )

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
        allowed_raw = os.getenv("EMAIL_ALLOWED_USERS", "").strip()
        allowed = self._allowed_users or {
            addr.strip().lower()
            for addr in allowed_raw.split(",")
            if addr.strip()
        }
        if not allowed:
            if os.getenv("EMAIL_ALLOW_ALL_USERS", "").strip().lower() not in {"true", "1", "yes"} and (
                os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() not in {"true", "1", "yes"}
            ):
                logger.debug(
                    "[Email] Dropping sender at dispatch — EMAIL_ALLOWED_USERS is unset "
                    "and open access is not opted in: %s",
                    sender_addr,
                )
                return
        else:
            if sender_addr.lower() not in allowed:
                logger.debug(
                    "[Email] Dropping non-allowlisted sender at dispatch: %s",
                    _redact_email_for_log(sender_addr),
                )
                await self._send_passive_notification(msg_data)
                return

        # Reject spoofed senders. The allowlist (and the gateway's own authz)
        # key on sender_addr, which comes straight from the attacker-controlled
        # From: header — so an attacker can forge From: an-allowlisted@addr to
        # get authorized (GHSA-rxqh-5572-8m77). This only matters when an
        # allowlist is actually being used to GRANT access: if no allowlist is
        # configured the gateway default-denies everyone anyway, and if allow-all
        # is on the operator already accepts any sender. So enforce From:
        # authentication exactly when an allowlist is in effect and allow-all is
        # off. Fail-closed: an unauthenticated From: is dropped before it can be
        # matched against the allowlist.
        if (
            self._require_authenticated_sender
            and self._allowlist_in_effect()
            and not self._allow_all_senders()
            and not msg_data.get("sender_authenticated", False)
        ):
            logger.warning(
                "[Email] Dropping sender with unauthenticated From: %s (%s). "
                "If your mail server does not stamp Authentication-Results, set "
                "platforms.email.require_authenticated_sender: false (or "
                "EMAIL_TRUST_FROM_HEADER=true) to accept the risk.",
                sender_addr,
                msg_data.get("auth_reason", "no verdict"),
            )
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
            if att["type"] == "image" and msg_type == MessageType.TEXT:
                msg_type = MessageType.PHOTO
            elif att["type"] == "document":
                # Document wins over PHOTO for mixed attachments: run.py's
                # image handling keys off the per-path image/* mime type
                # regardless of message_type, but document-context injection
                # gates strictly on MessageType.DOCUMENT — so DOCUMENT is the
                # only classification that surfaces both.
                msg_type = MessageType.DOCUMENT

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

    def _message_id_domain(self) -> str:
        """Domain part for generated Message-IDs.

        EMAIL_ADDRESS may lack an ``@`` (misconfiguration); fall back to
        ``localhost`` instead of crashing send with an IndexError.
        """
        if "@" in self._address:
            return self._address.rsplit("@", 1)[-1] or "localhost"
        return "localhost"

    def _send_proton_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
        attachments: Any = None,
        cc: Any = None,
        bcc: Any = None,
    ) -> str:
        """Send an email through the configured Proton runtime."""
        client = self._proton_client or _load_proton_client()
        policy = _recipient_policy(to_addr=to_addr, cc=cc, bcc=bcc)
        if not policy["allowed"]:
            raise RuntimeError(policy["failure_reason"])
        attachment_manifest, rejected_attachments = _attachment_manifest(attachments)
        if rejected_attachments:
            raise RuntimeError(f"attachment rejected: {rejected_attachments[0]['reason']}")
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
                    cc=_normalise_email_list(cc),
                    bcc=_normalise_email_list(bcc),
                    attachments=attachment_manifest,
                )
                structured = _structured_proton_result(
                    result,
                    accepted_recipients=policy["accepted_recipients"],
                    rejected_recipients=policy["rejected_recipients"],
                    attachments=attachment_manifest,
                    rejected_attachments=rejected_attachments,
                )
                if not structured["sent"]:
                    raise RuntimeError(structured["failure_reason"])
                sent_id = structured["message_id"]
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
                    cc=_normalise_email_list(cc),
                    bcc=_normalise_email_list(bcc),
                    attachments=attachment_manifest,
                )
                structured = _structured_proton_result(
                    result,
                    accepted_recipients=policy["accepted_recipients"],
                    rejected_recipients=policy["rejected_recipients"],
                    attachments=attachment_manifest,
                    rejected_attachments=rejected_attachments,
                )
                if not structured["sent"]:
                    raise RuntimeError(structured["failure_reason"])
                sent_id = structured["message_id"]
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
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        msg.attach(MIMEText(body, "plain", "utf-8"))

        smtp = self._connect_smtp()
        try:
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image URL as part of an email body.

        ``metadata`` is accepted to honor the base-class contract; the
        email body send doesn't use it.
        """
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
        if self._provider == "proton":
            return self._send_proton_email(to_addr, body, attachments=file_paths)

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
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
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

        smtp = self._connect_smtp()
        try:
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
        if self._provider == "proton":
            return self._send_proton_email(to_addr, body, attachments=[file_path])

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
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
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

        smtp = self._connect_smtp()
        try:
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


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the Email adapter moved from gateway/platforms/email.py into this
# bundled plugin. register() exposes the platform via the registry, replacing
# the Platform.EMAIL elif in gateway/run.py, the _PLATFORM_CONNECTED_CHECKERS
# entry in gateway/config.py, the _PLATFORMS["email"] static dict in
# hermes_cli/gateway.py, and the _send_email dispatch in
# tools/send_message_tool.py. EMAIL_* env→PlatformConfig seeding stays in core.
# ──────────────────────────────────────────────────────────────────────────


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process Email delivery via SMTP (one-shot). Implements the
    standalone_sender_fn contract; replaces the legacy _send_email helper."""
    import smtplib
    import ssl as _ssl
    from email.mime.text import MIMEText
    from email.utils import formatdate

    extra = getattr(pconfig, "extra", {}) or {}
    provider = os.getenv("EMAIL_PROVIDER", "imap").strip().lower() or "imap"
    address = extra.get("address") or os.getenv("EMAIL_ADDRESS", "")

    # Proton uses the API transport, not SMTP. The outbound helpers now live in
    # this module (post-#41112 plugin refactor), so call them directly rather
    # than importing from the legacy gateway.platforms.email path. Applies the
    # recipient allow-policy + attachment-safety manifest (blocks secret-like
    # files) before sending, and returns the structured Proton result.
    if provider == "proton":
        try:
            client = _load_proton_client()
            policy = _recipient_policy(to_addr=chat_id)
            if not policy["allowed"]:
                return {
                    "success": False,
                    "sent": False,
                    "platform": "email",
                    "chat_id": chat_id,
                    "accepted_recipients": policy["accepted_recipients"],
                    "rejected_recipients": policy["rejected_recipients"],
                    "error": policy["failure_reason"],
                }
            attachments, rejected_attachments = _attachment_manifest(media_files or [])
            if rejected_attachments:
                return {
                    "success": False,
                    "sent": False,
                    "platform": "email",
                    "chat_id": chat_id,
                    "accepted_recipients": policy["accepted_recipients"],
                    "rejected_recipients": policy["rejected_recipients"],
                    "rejected_attachments": rejected_attachments,
                    "error": rejected_attachments[0]["reason"],
                }
            for method_name in ("send_email", "send_message", "send_reply", "reply_message"):
                method = getattr(client, method_name, None)
                if callable(method):
                    raw_result = _call_proton_outbound(
                        method,
                        message_id=None,
                        thread_id=thread_id,
                        to_addr=chat_id,
                        subject="Hermes Agent",
                        body=message,
                        cc=_normalise_email_list(None),
                        bcc=_normalise_email_list(None),
                        attachments=attachments,
                    )
                    result = _structured_proton_result(
                        raw_result,
                        accepted_recipients=policy["accepted_recipients"],
                        rejected_recipients=policy["rejected_recipients"],
                        attachments=attachments,
                        rejected_attachments=rejected_attachments,
                    )
                    return {
                        "success": result["sent"],
                        "sent": result["sent"],
                        "platform": "email",
                        "chat_id": chat_id,
                        "message_id": result["message_id"],
                        "accepted_recipients": result["accepted_recipients"],
                        "rejected_recipients": result["rejected_recipients"],
                        "attachments": result["attachments"],
                        "rejected_attachments": result["rejected_attachments"],
                        **({"error": result["failure_reason"]} if not result["sent"] else {}),
                    }
            return {"error": "Email send failed: Proton outbound is not available in the configured client"}
        except Exception as e:
            return {"error": f"Email send failed: {e}"}

    password = os.getenv("EMAIL_PASSWORD", "")
    smtp_host = extra.get("smtp_host") or os.getenv("EMAIL_SMTP_HOST", "")
    try:
        smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))
    except (ValueError, TypeError):
        smtp_port = 587

    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}

    try:
        msg = MIMEText(message, "plain", "utf-8")
        msg["From"] = address
        msg["To"] = chat_id
        msg["Subject"] = "Hermes Agent"
        msg["Date"] = formatdate(localtime=True)

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls(context=_ssl.create_default_context())
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        return {"success": True, "platform": "email", "chat_id": chat_id}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e
            return _e(f"Email send failed: {e}")
        except Exception:
            return {"error": f"Email send failed: {e}"}


def _is_connected(config) -> bool:
    """Email is connected when an address is configured (in PlatformConfig.extra
    or via EMAIL_ADDRESS). Mirrors the legacy
    _PLATFORM_CONNECTED_CHECKERS[Platform.EMAIL] = bool(extra.get('address'))."""
    extra = getattr(config, "extra", {}) or {}
    if extra.get("address"):
        return True
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs EmailAdapter from a PlatformConfig."""
    return EmailAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="email",
        label="Email",
        adapter_factory=_build_adapter,
        check_fn=check_email_requirements,
        is_connected=_is_connected,
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib) — no extra deps",
        allowed_users_env="EMAIL_ALLOWED_USERS",
        allow_all_env="EMAIL_ALLOW_ALL_USERS",
        cron_deliver_env_var="EMAIL_HOME_ADDRESS",
        standalone_sender_fn=_standalone_send,
        max_message_length=50_000,
        pii_safe=True,
        emoji="📧",
        allow_update_command=True,
    )
