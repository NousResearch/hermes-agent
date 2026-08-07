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
import socket
import time

# Profile-scoped secret reader for multiplexing support (PR #50094)
from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret
import ssl
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate, formataddr
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
from utils import is_truthy_value

logger = logging.getLogger(__name__)


def _get_esecret(name: str, default: str = "") -> str:
    """Scope-aware ``EMAIL_*`` read with the default-profile startup fallback.

    Secondary profiles run under ``_profile_runtime_scope`` — the scope is
    authoritative and a scoped miss returns ``default`` (no cross-profile
    borrow). The DEFAULT profile's adapter constructs and sends *unscoped*
    under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash its email path; there ``os.environ``
    is that profile's own value, so fall back to it. Same pattern as the
    Slack ``SLACK_APP_TOKEN`` read (#59739) and the WhatsApp
    ``_get_wsecret`` fix (5438e9c629).
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


# Backwards-compatible alias for the name used by the original #59076 hunks.
_get_secret = _get_esecret


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read (``env_int`` variant of ``_get_esecret``)."""
    raw = str(_get_esecret(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read (``env_bool`` variant of ``_get_esecret``)."""
    return is_truthy_value(_get_esecret(name, ""), default=default)


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
    """
    addr = _get_secret("EMAIL_ADDRESS", "").strip()
    pwd = _get_secret("EMAIL_PASSWORD", "").strip()
    imap = _get_secret("EMAIL_IMAP_HOST", "").strip()
    smtp = _get_secret("EMAIL_SMTP_HOST", "").strip()
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
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)

        # Resolve connection settings from the env vars first, then fall back to
        # PlatformConfig.extra (address/imap_host/smtp_host) — the canonical dict
        # gateway.config populates and that the "connected" check, the
        # send-helper, and `hermes config show` already read. Without the
        # fallback a config.yaml-only setup left these empty. Host/address values
        # are stripped: a stray space or newline made IMAP4_SSL raise the
        # misleading ``[Errno 8] nodename nor servname`` (an unresolvable name)
        # instead of an obvious "host not set" error.
        extra = config.extra or {}
        self._address = (_get_secret("EMAIL_ADDRESS", "") or extra.get("address", "")).strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = (_get_secret("EMAIL_IMAP_HOST", "") or extra.get("imap_host", "")).strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        self._smtp_host = (_get_secret("EMAIL_SMTP_HOST", "") or extra.get("smtp_host", "")).strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)

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
        elif _esecret_bool("EMAIL_TRUST_FROM_HEADER", False):
            self._require_authenticated_sender = False
        else:
            self._require_authenticated_sender = True

        # Optional authserv-id to pin Authentication-Results to the operator's
        # own receiving server (defends against an injected header that sorts
        # first). Defaults to the From-domain of the agent's own address.
        self._authserv_id = (
            extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")
        ).strip().lower()

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None

        # Map context key -> last subject + message-id for threading.
        # Context key is "<chat_id>::<thread_id>" in thread mode, or the bare
        # chat_id (sender address) in sender mode. chat_id is ALWAYS a real
        # email address — never a subject or composite key.
        self._thread_context: Dict[str, Dict[str, Any]] = {}
        # Message-ID -> context key, so replies can anchor to the exact
        # inbound message even when send-time metadata lacks thread_id.
        self._msgid_context: Dict[str, str] = {}
        self._thread_context_max: int = 2000  # cap to prevent unbounded growth

        # Content hashes of files already attached per thread, so the agent
        # doesn't re-send the same file on every turn of a conversation.
        # ctx_key -> {sha256, ...}
        self._sent_attachments: Dict[str, set] = {}

        # ── Single-email coalescing ───────────────────────────────────────
        # One inbound turn should produce ONE outbound mail, the way a person
        # replies. The gateway dispatches a turn as several adapter calls:
        # send() for the body, send_multiple_images() once for an image batch,
        # and send_document() once PER remaining file, with human_delay sleeps
        # between them. Sent straight through, that is 1+N emails for one
        # answer — and on email (unlike chat) each one is a separate message in
        # the recipient's thread. Parts are buffered here and flushed as a
        # single MIME once the turn goes quiet.
        # ctx_key -> {to_addr, ctx, reply_to, body_parts, files, task, first_seen}
        self._pending: Dict[str, Dict[str, Any]] = {}
        # Idle debounce, re-armed by every new part. Observed body->attachment
        # gaps on real turns were 4.1-7.9s, so this must comfortably clear that.
        self._fold_window: float = float(
            extra.get("fold_window_seconds")
            or os.getenv("EMAIL_FOLD_WINDOW_SECONDS", "")
            or 12.0
        )
        # Hard ceiling measured from the first part, so a long attachment loop
        # can never stall a reply indefinitely.
        self._fold_max_wait: float = float(
            extra.get("fold_max_wait_seconds")
            or os.getenv("EMAIL_FOLD_MAX_WAIT_SECONDS", "")
            or 90.0
        )

        # Thread state is otherwise memory-only, so every gateway restart wipes
        # it: the next reply on a live thread goes out with a generic subject,
        # no In-Reply-To and no Cc. Persist it across restarts.
        #
        # Resolved through get_hermes_home() rather than Path.home() so the
        # file lands under the active HERMES_HOME (per-profile, and sandboxed
        # per-test), instead of leaking one shared file across profiles.
        _state_override = str(
            extra.get("thread_state_path")
            or os.getenv("EMAIL_THREAD_STATE_PATH", "")
        ).strip()
        if _state_override:
            self._state_path: str = _state_override
        else:
            try:
                from hermes_constants import get_hermes_home
                _home = Path(get_hermes_home())
            except Exception:
                _home = Path.home() / ".hermes"
            self._state_path = str(_home / "state" / "email_thread_state.json")
        self._load_state()

        # Session routing (opt-in thread isolation):
        #   "sender" (default) — one session per sender address (stock Hermes
        #     behaviour; safe for existing deployments — no session key change).
        #   "thread" — one session per email thread via SessionSource.thread_id
        #     (recommended for email-agent secretaries). Enable via
        #     platforms.email.session_routing or EMAIL_SESSION_ROUTING=thread.
        self._session_routing = (
            extra.get("session_routing", "")
            or os.getenv("EMAIL_SESSION_ROUTING", "")
            or "sender"
        ).strip().lower()
        if self._session_routing not in ("thread", "sender"):
            self._session_routing = "sender"

        # Display name for the From: header in outgoing emails.
        self._display_name = (
            extra.get("display_name", "")
            or os.getenv("EMAIL_DISPLAY_NAME", "")
            or ""
        ).strip()

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
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host surfaces as an actionable config
        # error instead of IMAP4_SSL("") raising the cryptic
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
        print(f"[Email] Connected as {self._address}")
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        # Emit anything still buffered by the fold: a restart mid-turn would
        # otherwise silently drop a reply, which is worse than sending two.
        try:
            self._flush_all_pending()
        except Exception as e:
            logger.error("[Email] Pending flush on disconnect failed: %s", e)
        try:
            self._save_state()
        except Exception:
            pass
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
                    # Kept so replies can EXTEND the References chain instead of
                    # resetting it to just the parent (long threads drift apart
                    # in Outlook otherwise), and so reply-all can preserve Cc.
                    references = msg.get("References", "")
                    to_header = _decode_header_value(msg.get("To", ""))
                    cc_header = _decode_header_value(msg.get("Cc", ""))
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
                        "references": references,
                        "to_header": to_header,
                        "cc_header": cc_header,
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
            _get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in truthy
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
            _get_secret("EMAIL_ALLOWED_USERS", "").strip()
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
        #
        # Precedence matches the gateway authz layer: allow-all wins FIRST;
        # only when it is off does the allowlist gate. Allowlist tokens may be
        # exact addresses or @domain wildcards (e.g. "@known.ltd").
        if not self._allow_all_senders():
            allowed_raw = os.getenv("EMAIL_ALLOWED_USERS", "").strip()
            if not allowed_raw:
                logger.debug(
                    "[Email] Dropping sender at dispatch — EMAIL_ALLOWED_USERS is unset "
                    "and open access is not opted in: %s",
                    sender_addr,
                )
                return
            tokens = {t.strip().lower() for t in allowed_raw.split(",") if t.strip()}
            sender_l = sender_addr.lower()
            if not any(
                (t.startswith("@") and sender_l.endswith(t)) or sender_l == t
                for t in tokens
            ):
                logger.debug("[Email] Dropping non-allowlisted sender at dispatch: %s", sender_addr)
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

        # Store thread context for reply threading.
        # chat_id is ALWAYS the sender address (a deliverable mailbox). In
        # "thread" mode, thread_id is a stable per-thread key so
        # build_session_key() isolates sessions per (sender, thread) the same
        # way Telegram topics / Discord threads do. In "sender" mode,
        # thread_id is None and each sender gets one session (legacy).
        if self._session_routing == "thread":
            _thread_subject = re.sub(
                r'^(?:(?:re|fwd|fw):\s*)+', '', subject.strip(), flags=re.IGNORECASE
            ).strip()
            _thread_key = (
                _thread_subject
                or msg_data.get("in_reply_to")
                or msg_data["message_id"]
            )
            thread_id = _thread_key or None
        else:
            thread_id = None

        _ctx_key = f"{sender_addr}::{thread_id}" if thread_id else sender_addr
        # Reply-all: remember every recipient on the inbound mail (To + Cc,
        # minus the sender and ourselves) so replies keep the chain instead of
        # silently dropping everyone who was cc'd.
        _recipients: List[str] = []
        for _hdr in (msg_data.get("to_header", ""), msg_data.get("cc_header", "")):
            if not _hdr:
                continue
            for _part in _hdr.split(","):
                _addr = _extract_email_address(_part)
                if (
                    _addr
                    and _addr != sender_addr
                    and _addr != self._address.lower()
                    and _addr not in _recipients
                ):
                    _recipients.append(_addr)
        self._thread_context[_ctx_key] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
            "references": msg_data.get("references", ""),
            "sender_addr": sender_addr,
            "recipients": list(_recipients),
            # Carried so every outbound path indexes itself against the same
            # key the inbound side used, rather than recomputing it.
            "ctx_key": _ctx_key,
        }
        if msg_data.get("message_id"):
            self._msgid_context[msg_data["message_id"]] = _ctx_key
        self._trim_thread_context()
        self._save_state()

        source = self.build_source(
            chat_id=sender_addr,
            thread_id=thread_id,
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

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an email reply to the given address.

        ``chat_id`` is always a real email address (the sender from the
        inbound dispatch). Thread context is resolved from ``metadata``
        (the gateway passes ``thread_id`` through) or from ``reply_to``
        (the inbound Message-ID), never from an arbitrary cached sender.
        """
        to_addr, ctx = self._resolve_recipient(chat_id, metadata=metadata, reply_to=reply_to)
        if not to_addr:
            logger.error("[Email] Cannot resolve recipient for chat_id=%r and no EMAIL_HOME_ADDRESS set", chat_id)
            return SendResult(success=False, error=f"Cannot resolve recipient for chat_id={chat_id!r}")
        # Buffer rather than send: any attachments for this same turn arrive in
        # later adapter calls and must land in the SAME email.
        return SendResult(
            success=True,
            message_id=self._queue_part(to_addr, ctx, reply_to=reply_to, body=content),
        )

    def _trim_thread_context(self) -> None:
        """Bound thread context + msgid index so long-lived gateways don't grow forever."""
        if len(self._thread_context) > self._thread_context_max:
            # Dicts preserve insertion order; drop the oldest half.
            drop = len(self._thread_context) - (self._thread_context_max // 2)
            stale_keys = list(self._thread_context.keys())[:drop]
            for key in stale_keys:
                del self._thread_context[key]
            stale = set(stale_keys)
            self._msgid_context = {
                mid: key for mid, key in self._msgid_context.items() if key not in stale
            }
        if len(self._msgid_context) > self._thread_context_max:
            drop = len(self._msgid_context) - (self._thread_context_max // 2)
            for mid in list(self._msgid_context.keys())[:drop]:
                del self._msgid_context[mid]

    # ── Persistent thread state ───────────────────────────────────────────
    def _load_state(self) -> None:
        """Restore thread/msgid/attachment state written by a previous run."""
        try:
            raw = json.loads(Path(self._state_path).read_text())
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning("[Email] Could not load thread state (%s) — starting empty", e)
            return
        try:
            tc = raw.get("thread_context") or {}
            mc = raw.get("msgid_context") or {}
            sa = raw.get("sent_attachments") or {}
            if isinstance(tc, dict):
                self._thread_context.update(
                    {k: v for k, v in tc.items() if isinstance(v, dict)}
                )
            if isinstance(mc, dict):
                self._msgid_context.update(
                    {k: v for k, v in mc.items() if isinstance(v, str)}
                )
            if isinstance(sa, dict):
                for k, v in sa.items():
                    if isinstance(v, list):
                        self._sent_attachments[k] = set(v)
            self._trim_thread_context()
            logger.info(
                "[Email] Restored thread state: %d thread(s), %d message-id(s)",
                len(self._thread_context),
                len(self._msgid_context),
            )
        except Exception as e:
            logger.warning("[Email] Malformed thread state (%s) — starting empty", e)

    def _save_state(self) -> None:
        """Atomically persist thread state. Never raises into a send path."""
        try:
            payload = {
                "thread_context": self._thread_context,
                "msgid_context": self._msgid_context,
                "sent_attachments": {
                    k: sorted(v) for k, v in self._sent_attachments.items()
                },
            }
            p = Path(self._state_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload))
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, p)
        except Exception as e:
            logger.warning("[Email] Could not save thread state: %s", e)

    def _ctx_key_for(self, to_addr: str, ctx: Optional[Dict[str, Any]]) -> str:
        """Key identifying the thread an outbound mail belongs to.

        Prefers the key stamped onto the context when the inbound message was
        stored, so the outbound side can never disagree with the inbound side
        about which thread this is.
        """
        ctx = ctx or {}
        key = str(ctx.get("ctx_key") or "").strip()
        if key:
            return key
        subject = str(ctx.get("subject") or "").strip()
        if subject:
            tid = re.sub(
                r"^(?:(?:re|fwd|fw):\s*)+", "", subject, flags=re.IGNORECASE
            ).strip()
            if tid:
                return f"{(ctx.get('sender_addr') or to_addr).lower()}::{tid}"
        return to_addr.lower()

    def _register_outbound_msgid(
        self, msg_id: str, to_addr: str, ctx: Optional[Dict[str, Any]]
    ) -> None:
        """Index an agent-generated Message-ID against its thread.

        Without this, a reply to one of the agent's own mails arrives with
        In-Reply-To=<hermes-...>, misses _msgid_context (which only ever held
        INBOUND ids), and falls through to the most-recent-thread scan in
        _context_for_send — answering on the wrong thread, with that thread's
        Cc list. Registering outbound ids closes that gap.
        """
        if not msg_id or msg_id.startswith(("suppressed-", "pending-", "folded-")):
            return
        key = self._ctx_key_for(to_addr, ctx)
        if not key:
            return
        self._msgid_context[msg_id] = key
        # Chain forward: the next reply on this thread should anchor to the
        # agent's latest mail and carry the earlier ids in References.
        stored = self._thread_context.get(key)
        if isinstance(stored, dict):
            chain = str(stored.get("references") or "").split()
            prior_id = str(stored.get("message_id") or "").strip()
            if prior_id and prior_id not in chain:
                chain.append(prior_id)
            stored["references"] = " ".join(chain)
            stored["message_id"] = msg_id
        self._trim_thread_context()
        self._save_state()

    def _apply_reply_headers(
        self,
        msg,
        to_addr: str,
        ctx: Optional[Dict[str, Any]] = None,
        reply_to_msg_id: Optional[str] = None,
        has_attachments: bool = False,
    ) -> None:
        """Stamp Subject / Cc / In-Reply-To / References on an outbound reply.

        Single source of truth for reply identity. Every send path (text,
        multi-attachment, single-attachment) goes through this. When the
        attachment paths built these headers themselves they silently diverged:
        a bare self._thread_context.get(to_addr) lookup always misses under
        session_routing="thread" (keys are "sender::thread_id"), so attachment
        mail went out as "Re: Hermes Agent" with no In-Reply-To and no Cc, and
        mail clients filed it as a brand-new conversation.
        """
        ctx = ctx or {}

        # Reply-all: preserve the inbound To/Cc chain.
        cc_addrs: List[str] = []
        for _addr in (ctx.get("recipients") or []):
            if _addr and _addr != to_addr.lower() and _addr != self._address.lower():
                if _addr not in cc_addrs:
                    cc_addrs.append(_addr)
        if cc_addrs:
            msg["Cc"] = ", ".join(cc_addrs)

        # Subject: inherit the thread's, prefixing a single "Re:".
        subject = str(ctx.get("subject") or "").strip()
        if not subject:
            # Fail loud rather than silently opening a new thread under a
            # generic subject — that is what cross-threaded attachment mail.
            logger.warning(
                "[Email] No thread context for %s (attachments=%s) — reply may not thread",
                to_addr,
                has_attachments,
            )
            subject = "Hermes Agent"
        if not re.match(r"^re:\s", subject, flags=re.IGNORECASE):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        # In-Reply-To is the parent; References is the whole chain (parent's
        # References + parent's Message-ID). Overwriting References with only
        # the parent made long threads drift apart in Outlook.
        parent_id = str(reply_to_msg_id or ctx.get("message_id") or "").strip()
        if parent_id:
            msg["In-Reply-To"] = parent_id
            chain = [r for r in str(ctx.get("references") or "").split() if r]
            if parent_id not in chain:
                chain.append(parent_id)
            if len(chain) > 20:
                # Keep the thread root plus the most recent ids.
                chain = chain[:1] + chain[-19:]
            msg["References"] = " ".join(chain)
        elif has_attachments:
            logger.error(
                "[Email] Sending attachment mail to %s with NO parent message-id "
                "— it will start a new thread",
                to_addr,
            )

    # ── Single-email coalescing ───────────────────────────────────────────
    def _queue_part(
        self,
        to_addr: str,
        ctx: Optional[Dict[str, Any]],
        reply_to: Optional[str] = None,
        body: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ) -> str:
        """Buffer one outbound part of a turn and (re)arm the idle flush.

        The gateway emits one answer as several adapter calls: send() for the
        body, send_multiple_images() once for an image batch, and
        send_document() once PER remaining file. Folding the body into just
        the first attachment would still leave one email per subsequent file,
        so parts accumulate here and are flushed together.
        """
        key = self._ctx_key_for(to_addr, ctx)
        # Tolerate a partially-constructed adapter (tests and some proactive
        # paths instantiate via object.__new__), so buffering never depends on
        # __init__ having run.
        pending = getattr(self, "_pending", None)
        if pending is None:
            pending = self._pending = {}
        entry = pending.get(key)
        if entry is None:
            entry = {
                "to_addr": to_addr,
                # Captured once. Deliberately NOT re-resolved at flush time:
                # re-resolving would reintroduce the wrong-thread fallback
                # this change exists to eliminate.
                "ctx": ctx or {},
                "reply_to": reply_to,
                "body_parts": [],
                "files": [],
                "task": None,
                "first_seen": time.monotonic(),
            }
            pending[key] = entry
        if body and body.strip():
            entry["body_parts"].append(body.strip())
        for fp in file_paths or []:
            if fp not in entry["files"]:
                entry["files"].append(fp)
        if reply_to and not entry.get("reply_to"):
            entry["reply_to"] = reply_to

        window = float(getattr(self, "_fold_window", 12.0))
        old = entry.get("task")
        if old is not None and not old.done():
            old.cancel()
        try:
            entry["task"] = asyncio.get_running_loop().create_task(
                self._flush_after_idle(key)
            )
        except RuntimeError:
            # No running loop — send immediately rather than swallow the reply.
            self._flush_now(key)
            return "folded-immediate"
        logger.info(
            "[Email] Buffered part for %s (body_parts=%d files=%d) — flushing in %.1fs",
            key,
            len(entry["body_parts"]),
            len(entry["files"]),
            window,
        )
        return f"pending-fold-{key}"

    async def _flush_after_idle(self, key: str) -> None:
        """Wait for the turn to go quiet, then emit exactly one mail."""
        try:
            entry = getattr(self, "_pending", {}).get(key)
            if not entry:
                return
            elapsed = time.monotonic() - entry["first_seen"]
            await asyncio.sleep(
                max(
                    0.0,
                    min(
                        float(getattr(self, "_fold_window", 12.0)),
                        float(getattr(self, "_fold_max_wait", 90.0)) - elapsed,
                    ),
                )
            )
        except asyncio.CancelledError:
            # A newer part arrived and re-armed the timer.
            return
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._flush_now, key)
        except Exception as e:
            logger.error("[Email] Fold flush failed for %s: %s", key, e, exc_info=True)

    def _flush_now(self, key: str) -> Optional[str]:
        """Build and send the buffered turn as a single MIME message."""
        entry = getattr(self, "_pending", {}).pop(key, None)
        if not entry:
            return None
        body = "\n\n".join(entry["body_parts"]).strip()
        files = [f for f in entry["files"] if f]
        to_addr = entry["to_addr"]
        ctx = entry["ctx"]
        reply_to = entry.get("reply_to")
        if not body and not files:
            return None
        logger.info(
            "[Email] Folding turn for %s into ONE email (body=%dch files=%d waited=%.1fs)",
            key,
            len(body),
            len(files),
            time.monotonic() - entry["first_seen"],
        )
        try:
            if files:
                return self._send_email_with_attachments(
                    to_addr, body, files, ctx=ctx, reply_to=reply_to
                )
            return self._send_email(to_addr, body, reply_to, ctx)
        except Exception as e:
            logger.error(
                "[Email] Folded send FAILED for %s (%d file(s)) — reply lost: %s",
                to_addr, len(files), e, exc_info=True,
            )
            return None

    def _flush_all_pending(self) -> None:
        """Emit every buffered turn — used on shutdown so nothing is dropped."""
        pending = getattr(self, "_pending", None) or {}
        for key in list(pending.keys()):
            entry = pending.get(key) or {}
            task = entry.get("task")
            if task is not None and not task.done():
                task.cancel()
            try:
                self._flush_now(key)
            except Exception as e:
                logger.error("[Email] Shutdown flush failed for %s: %s", key, e)

    def _context_for_send(
        self,
        chat_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> Dict[str, str]:
        """Resolve reply context (subject + threading headers) for a send.

        Lookup order:
        1. metadata.thread_id -> "<chat_id>::<thread_id>" (gateway thread passthrough)
        2. reply_to -> Message-ID index -> context key (exact inbound anchor)
        3. bare chat_id key (sender mode, or direct address)
        4. Most recent context stored for this sender (same mailbox only).

        Step 4 can only pick a subject/headers for a reply going to the SAME
        address — it never changes the recipient, so delivery stays fail-closed.
        """
        thread_id = (metadata or {}).get("thread_id")
        if thread_id:
            ctx = self._thread_context.get(f"{chat_id}::{thread_id}")
            if ctx:
                return ctx
        if reply_to:
            key = self._msgid_context.get(reply_to)
            if key:
                ctx = self._thread_context.get(key)
                if ctx:
                    return ctx
        ctx = self._thread_context.get(chat_id)
        if ctx:
            return ctx
        # Most recent context for this sender (insertion order = recency).
        best: Dict[str, str] = {}
        prefix = f"{chat_id}::"
        for key, val in self._thread_context.items():
            if key.startswith(prefix):
                best = val
        return best

    def _resolve_recipient(
        self,
        chat_id: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> tuple:
        """Resolve a chat_id to (recipient_address, thread_context).

        Single source of truth for all send paths. Never guesses — if the
        chat_id cannot be resolved to a specific recipient, returns ("", {})
        so the caller can fail closed or use EMAIL_HOME_ADDRESS explicitly.

        Resolution order:
        1. chat_id is itself an email address (the normal path: chat_id is
           always the sender's mailbox)
        2. Fall back to EMAIL_HOME_ADDRESS for proactive delivery only
        """
        ctx = self._context_for_send(chat_id, metadata=metadata, reply_to=reply_to)
        if "@" in chat_id:
            return chat_id, ctx
        # Proactive delivery (cron, kanban notifications) — use home address
        home = os.getenv("EMAIL_HOME_ADDRESS", "")
        if home:
            return home, {}
        return "", {}

    def _message_id_domain(self) -> str:
        """Domain part for generated Message-IDs.

        EMAIL_ADDRESS may lack an ``@`` (misconfiguration); fall back to
        ``localhost`` instead of crashing send with an IndexError.
        """
        if "@" in self._address:
            return self._address.rsplit("@", 1)[-1] or "localhost"
        return "localhost"

    def _send_email(
        self,
        to_addr: str,
        body: str,
        reply_to_msg_id: Optional[str] = None,
        ctx: Optional[Dict[str, str]] = None,
    ) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg = MIMEMultipart()
        msg["From"] = formataddr((self._display_name, self._address)) if self._display_name else self._address
        msg["To"] = to_addr

        ctx = ctx or {}
        self._apply_reply_headers(
            msg, to_addr, ctx=ctx, reply_to_msg_id=reply_to_msg_id, has_attachments=False
        )
        subject = msg["Subject"]

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

        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        self._register_outbound_msgid(msg_id, to_addr, ctx)
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

        to_addr, ctx = self._resolve_recipient(chat_id, metadata=metadata)
        if not to_addr:
            logger.error("[Email] Cannot resolve recipient for multi-image send, chat_id=%r", chat_id)
            return

        # Buffered, not sent: the body for this turn arrived in an earlier
        # send() call and the remaining files arrive in later send_document
        # calls. They all belong in one email.
        self._queue_part(to_addr, ctx, body=body, file_paths=local_paths)

    def _send_email_with_attachments(
        self,
        to_addr: str,
        body: str,
        file_paths: List[str],
        ctx: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> str:
        """Send an email with multiple file attachments via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = formataddr((self._display_name, self._address)) if self._display_name else self._address
        msg["To"] = to_addr

        ctx = ctx or {}
        self._apply_reply_headers(
            msg, to_addr, ctx=ctx, reply_to_msg_id=reply_to, has_attachments=True
        )

        msg["Date"] = formatdate(localtime=True)
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        msg["Message-ID"] = msg_id

        if body:
            msg.attach(MIMEText(body, "plain", "utf-8"))

        # Don't re-attach a file already delivered on this thread — a person
        # wouldn't resend the same attachments on every turn. Dedupe by content
        # hash, scoped to the thread so a genuinely revised file still sends.
        _dedupe_key = self._ctx_key_for(to_addr, ctx)
        _already = self._sent_attachments.setdefault(_dedupe_key, set())
        _fresh: List[str] = []
        _fresh_hashes: List[str] = []
        _skipped: List[str] = []
        for _fp in file_paths:
            try:
                _h = hashlib.sha256(Path(_fp).read_bytes()).hexdigest()
            except Exception:
                _fresh.append(_fp)
                continue
            if _h in _already or _h in _fresh_hashes:
                _skipped.append(Path(_fp).name)
            else:
                _fresh_hashes.append(_h)
                _fresh.append(_fp)
        if _skipped:
            logger.info(
                "[Email] Skipping %d already-sent attachment(s) for %s: %s",
                len(_skipped), to_addr, ", ".join(_skipped),
            )
        file_paths = _fresh
        if len(self._sent_attachments) > self._thread_context_max:
            for _k in list(self._sent_attachments.keys())[: len(self._sent_attachments) // 2]:
                del self._sent_attachments[_k]

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
        # Record hashes only AFTER a successful send: marking them earlier meant
        # a failed flush permanently suppressed those files on this thread.
        _already.update(_fresh_hashes)
        self._register_outbound_msgid(msg_id, to_addr, ctx)
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
        metadata = kwargs.get("metadata")
        to_addr, ctx = self._resolve_recipient(chat_id, metadata=metadata, reply_to=reply_to)
        if not to_addr:
            logger.error("[Email] Cannot resolve recipient for document send, chat_id=%r", chat_id)
            return SendResult(success=False, error=f"Cannot resolve recipient for chat_id={chat_id!r}")
        # The gateway calls this once PER file, so each call only contributes
        # its file to the buffered turn — it never sends an email of its own.
        return SendResult(
            success=True,
            message_id=self._queue_part(
                to_addr,
                ctx,
                reply_to=reply_to,
                body=caption or "",
                file_paths=[file_path],
            ),
        )

    def _send_email_with_attachment(
        self,
        to_addr: str,
        body: str,
        file_path: str,
        file_name: Optional[str] = None,
        ctx: Optional[Dict[str, Any]] = None,
        reply_to: Optional[str] = None,
    ) -> str:
        """Send an email with a file attachment via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = formataddr((self._display_name, self._address)) if self._display_name else self._address
        msg["To"] = to_addr

        ctx = ctx or {}
        self._apply_reply_headers(
            msg, to_addr, ctx=ctx, reply_to_msg_id=reply_to, has_attachments=True
        )

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
        if not ctx:
            # Thread mode: contexts live under "<chat_id>::<thread_id>"; report
            # the most recent subject for this mailbox.
            prefix = f"{chat_id}::"
            for key, val in self._thread_context.items():
                if key.startswith(prefix):
                    ctx = val
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
    address = extra.get("address") or _get_secret("EMAIL_ADDRESS", "")
    password = _get_secret("EMAIL_PASSWORD", "")
    smtp_host = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", "")
    try:
        smtp_port = int(_get_secret("EMAIL_SMTP_PORT", "587") or "587")
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


def interactive_setup() -> None:
    """Interactive `hermes gateway setup` flow for the Email platform.

    Collects IMAP/SMTP credentials, access control, session routing, and an
    optional From display name. Lazy-imports hermes_cli.setup helpers so the
    plugin stays importable in non-CLI contexts.
    """
    from hermes_cli.setup import (
        get_env_value,
        print_header,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_choice,
        prompt_yes_no,
        save_env_value,
    )

    print_header("Email")
    existing = get_env_value("EMAIL_ADDRESS")
    if existing:
        print_info(f"Email: already configured (address: {existing})")
        if not prompt_yes_no("Reconfigure Email?", False):
            return

    print_info("Talk to Hermes through a dedicated IMAP/SMTP mailbox.")
    print_info("  Use an app password for Gmail / Workspace — not the account password.")
    print_info("  Prefer a dedicated mailbox (not your personal inbox).")
    print()

    address = prompt("Email address", default=get_env_value("EMAIL_ADDRESS") or "")
    if not address:
        print_warning("Email address is required — skipping Email setup")
        return
    save_env_value("EMAIL_ADDRESS", address.strip())

    password = prompt("Email password / app password", password=True)
    if password:
        save_env_value("EMAIL_PASSWORD", password)
    elif not get_env_value("EMAIL_PASSWORD"):
        print_warning("Password is required — skipping Email setup")
        return

    imap_host = prompt(
        "IMAP host (e.g. imap.gmail.com)",
        default=get_env_value("EMAIL_IMAP_HOST") or "imap.gmail.com",
    )
    if imap_host:
        save_env_value("EMAIL_IMAP_HOST", imap_host.strip())

    smtp_host = prompt(
        "SMTP host (e.g. smtp.gmail.com)",
        default=get_env_value("EMAIL_SMTP_HOST") or "smtp.gmail.com",
    )
    if smtp_host:
        save_env_value("EMAIL_SMTP_HOST", smtp_host.strip())

    print()
    print_info("Access control — the gateway denies unknown senders by default.")
    allowed = prompt(
        "Allowed senders (comma-separated addresses or @domain tokens, empty = decide next)",
        default=get_env_value("EMAIL_ALLOWED_USERS") or "",
    )
    if allowed.strip():
        save_env_value("EMAIL_ALLOWED_USERS", allowed.replace(" ", ""))
        print_success("  Allowlist saved.")
    else:
        access_idx = prompt_choice(
            "How should unauthorized senders be handled?",
            [
                "Open access (any email sender can message the bot)",
                "Keep unknown senders silent (configure allowlist later)",
            ],
            1,
        )
        if access_idx == 0:
            save_env_value("EMAIL_ALLOW_ALL_USERS", "true")
            print_warning("  Open access enabled — anyone who can email this mailbox can talk to the bot.")
        else:
            save_env_value("EMAIL_ALLOW_ALL_USERS", "")
            print_info("  Unknown senders will be ignored until EMAIL_ALLOWED_USERS is set.")

    home = prompt(
        "Home address for cron/notifications (optional)",
        default=get_env_value("EMAIL_HOME_ADDRESS") or "",
    )
    if home.strip():
        save_env_value("EMAIL_HOME_ADDRESS", home.strip())

    print()
    print_info("Session routing controls how email conversations are isolated.")
    print_info("  Thread mode (recommended for email secretaries): one agent session")
    print_info("  per email thread — two subjects from the same person stay separate.")
    print_info("  Sender mode (stock default): one session per sender address.")
    routing_idx = prompt_choice(
        "Session routing?",
        [
            "One session per email thread (recommended for email agents)",
            "One session per sender (stock / legacy)",
        ],
        0,
    )
    routing = "thread" if routing_idx == 0 else "sender"
    save_env_value("EMAIL_SESSION_ROUTING", routing)
    # Also persist under platforms.email so config.yaml is the source of truth
    # for operators who prefer YAML over env.
    try:
        from hermes_cli.config import write_platform_config_field

        write_platform_config_field("email", "session_routing", routing, raw=True)
    except Exception:
        pass
    print_success(f"  Session routing: {routing}")

    print()
    display = prompt(
        "From display name (optional, e.g. Iris Sloane)",
        default=get_env_value("EMAIL_DISPLAY_NAME") or "",
    )
    if display.strip():
        save_env_value("EMAIL_DISPLAY_NAME", display.strip())
        try:
            from hermes_cli.config import write_platform_config_field

            write_platform_config_field("email", "display_name", display.strip(), raw=True)
        except Exception:
            pass
        print_success(f"  From header will show: {display.strip()} <{address.strip()}>")
    else:
        print_info("  From header will use the bare address.")

    print()
    print_success("Email platform configured.")
    print_info("  Start or restart the gateway: hermes gateway restart")
    print_info("  Docs: https://hermes-agent.nousresearch.com/docs/user-guide/messaging/email")


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
        setup_fn=interactive_setup,
    )
