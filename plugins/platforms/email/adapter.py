"""Email platform adapter for the Hermes gateway: users talk to Hermes by sending email; IMAP (polled)
receives, SMTP sends. Configured via EMAIL_* env vars or ``platforms.email`` in config.yaml (see website docs)."""

import asyncio
import email as email_lib
from contextlib import contextmanager, suppress
import imaplib
import json
import logging
import os
import re
import smtplib
import socket
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

from gateway.platforms.base import (BasePlatformAdapter, MessageEvent, MessageType, SendResult,
                                    cache_document_from_bytes, cache_image_from_bytes)
from gateway.config import Platform, PlatformConfig
from utils import is_truthy_value
from gateway.platforms._shared import get_scoped_secret as _get_secret, coerce_port

logger = logging.getLogger(__name__)

_SECURITY_ALIASES = {"tls": "tls", "ssl": "tls", "implicit": "tls", "starttls": "starttls", "plain": "plain", "none": "plain"}
# Automated senders (address substrings / bulk-mail headers) are silently ignored.
_NOREPLY_PATTERNS = ("noreply", "no-reply", "no_reply", "donotreply", "do-not-reply", "mailer-daemon", "postmaster",
                     "bounce", "notifications@", "automated@", "auto-confirm", "auto-reply", "automailer")
_AUTOMATED_HEADERS = {"Auto-Submitted": lambda v: v.lower() != "no",
                      "Precedence": lambda v: v.lower() in {"bulk", "list", "junk"},
                      "X-Auto-Response-Suppress": lambda v: bool(v), "List-Unsubscribe": lambda v: bool(v)}
MAX_MESSAGE_LENGTH = 50_000  # Gmail-safe max length per email body
SMTP_CONNECT_TIMEOUT = 30
_TRUTHY = {"true", "1", "yes"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
# Charset labels seen in the wild that Python's codec registry doesn't know: "unknown-8bit"/"x-unknown" are
# RFC 1428 placeholders (QQ Mail emits them); gb2312/gbk map to the gb18030 superset so GBK extensions decode.
_CHARSET_ALIASES = {"unknown-8bit": "utf-8", "unknown": "utf-8", "x-unknown": "utf-8", "default": "utf-8",
                    "ansi_x3.110-1983": "latin-1", "cp-850": "cp850",
                    "gb2312": "gb18030", "gbk": "gb18030", "ks_c_5601-1987": "cp949"}
# Ordered (pattern, replacement) substitutions for _strip_html.
_HTML_SUBS = ((re.compile(r"<br\s*/?>", re.IGNORECASE), "\n"), (re.compile(r"<p[^>]*>", re.IGNORECASE), "\n"),
              (re.compile(r"</p>", re.IGNORECASE), "\n"), (re.compile(r"<[^>]+>"), ""), (re.compile(r"&nbsp;"), " "),
              (re.compile(r"&amp;"), "&"), (re.compile(r"&lt;"), "<"), (re.compile(r"&gt;"), ">"), (re.compile(r"\n{3,}"), "\n\n"))
# "method=result" tokens (``dmarc=pass``) and property values (``header.from=x``) in Authentication-Results.
_AUTH_METHOD_RE = re.compile(r"\b(dmarc|dkim|spf)\s*=\s*([a-z]+)", re.IGNORECASE)
_AUTH_PROP_RE = re.compile(r"\b(header\.from|header\.d|smtp\.mailfrom|smtp\.from|envelope-from)\s*=\s*([^\s;]+)", re.IGNORECASE)


def _esecret_int(name: str, default: int) -> int:
    """Scope-aware integer read."""
    return coerce_port(str(_get_secret(name, "")).strip() or default, default)


def _esecret_bool(name: str, default: bool = False) -> bool:
    """Scope-aware boolean read."""
    return is_truthy_value(raw, default=default) if (raw := str(_get_secret(name, "")).strip()) else default


def _normalize_security(value: Any, default: str = "tls") -> str:
    """Map to ``tls`` | ``starttls`` | ``plain``; unknown values warn and fall back to *default* (a typo never downgrades to plaintext)."""
    raw = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if raw and raw not in _SECURITY_ALIASES:
        logger.warning("Unknown email security mode %r; using %r", value, default)
    return _SECURITY_ALIASES.get(raw, default)


def _tls_context(verify: bool, host: str) -> ssl.SSLContext:
    """Verified context by default; unverified only when explicitly opted out."""
    if verify:
        return ssl.create_default_context()
    if host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning("TLS verification disabled for non-loopback host %s", host)
    return ssl._create_unverified_context()


def _close_imap(imap: "imaplib.IMAP4") -> None:
    """Teardown that guarantees the socket closes: ``logout()`` only guards ``OSError``, so ``IMAP4.abort`` on a
    broken connection skipped ``shutdown()`` and leaked one fd per failed poll (fatal on macOS's 256 soft limit).

    ``IMAP4.logout()`` only guards against ``OSError`` internally: a broken connection makes
    ``_simple_command('LOGOUT')`` raise ``IMAP4.abort`` (which is *not* an ``OSError``), so ``logout()``
    propagates before its own ``shutdown()`` call and the TCP socket stays open. On macOS, where the default
    soft fd limit is 256 and pollers may run through a local proxy, these abandoned sockets accumulate one
    per failed poll until the gateway hits ``[Errno 24] Too many open files`` (#79889).
    """
    try:
        imap.logout()
    except Exception:
        with suppress(Exception):
            imap.shutdown()


def _create_ipv4_connection(host: str, port: int, timeout: float, source_address: Any = None) -> socket.socket:
    """``socket.create_connection`` constrained to ``AF_INET`` (no process-global socket mutation — sends run in executor threads)."""
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, sockaddr in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
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
    raise last_error if last_error is not None else OSError(f"No IPv4 address found for {host}:{port}")


class _IPv4SMTP(smtplib.SMTP):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return _create_ipv4_connection(host, port, timeout, source_address=self.source_address)


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    def _get_socket(self, host, port, timeout):  # type: ignore[override]
        return self.context.wrap_socket(_create_ipv4_connection(host, port, timeout, source_address=self.source_address), server_hostname=getattr(self, "_host", host))


def _open_smtp(host: str, port: int, security: str, ctx: ssl.SSLContext, smtp_cls: type, smtp_ssl_cls: type, **kwargs: Any) -> smtplib.SMTP:
    """Open one SMTP connection with TLS established per *security*; *kwargs* go to the constructor."""
    if security == "tls":
        return smtp_ssl_cls(host, port, context=ctx, **kwargs)
    smtp = smtp_cls(host, port, **kwargs)
    if security == "starttls":
        try:
            smtp.starttls(context=ctx)
        except Exception:
            smtp.close()
            raise
    return smtp


def _send_imap_id(imap: "imaplib.IMAP4") -> None:
    """Send RFC 2971 IMAP ID: 163/NetEase require it after LOGIN (else every UID command
    returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed."""
    try:
        try:
            from hermes_cli import __version__ as _hermes_version
        except Exception:  # noqa: BLE001 — keep ID best-effort if import fails
            _hermes_version = "0"
        imap.xatom("ID", f'("name" "hermes-agent" "version" "{_hermes_version}" '
                         '"vendor" "NousResearch" "support-email" "noreply@nousresearch.com")')
    except Exception as e:  # noqa: BLE001 — best-effort, never fatal
        logger.debug("[Email] IMAP ID command not accepted: %s", e)


def _is_automated_sender(address: str, headers: dict) -> bool:
    """True if this email is from an automated/noreply source."""
    addr = address.lower()
    return any(pattern in addr for pattern in _NOREPLY_PATTERNS) or any(
        (value := headers.get(header, "")) and check(value) for header, check in _AUTOMATED_HEADERS.items())


def check_email_requirements() -> bool:
    """True when all email settings are present and non-blank (blank keys left by an abandoned setup must not enable the platform).

    Treats blank/whitespace-only values as missing so an abandoned setup that left empty ``EMAIL_*`` keys in
    ``.env`` does not enable the platform (#40715).
    """
    return all(_get_secret(name, "").strip() for name in ("EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"))


def _safe_decode(payload: bytes, charset: "Optional[str]") -> str:
    """Decode without ever raising: ``errors="replace"`` does not guard a missing codec (``LookupError``), so fall back alias → UTF-8 → latin-1.

    Unknown or malformed charset labels (``unknown-8bit``, misspelled names, attacker-controlled garbage)
    previously raised ``LookupError`` from ``bytes.decode`` — ``errors="replace"`` only guards decode
    errors, not a missing codec — which aborted the whole IMAP fetch and dropped every message in the batch
    (#35901, #55381, #55383). Fall back through a small alias table, then UTF-8, then latin-1 (which never
    fails).
    """
    label = (charset or "utf-8").strip().strip("\"'").lower() or "utf-8"
    for candidate in (_CHARSET_ALIASES.get(label, label), "utf-8"):
        try:
            return payload.decode(candidate, errors="replace")
        except (LookupError, ValueError):
            continue
    return payload.decode("latin-1", errors="replace")


def _decode_header_value(raw: str) -> str:
    """Decode an RFC 2047 header into a plain string; never raises.

    Never raises: malformed encoded-words or unknown charsets degrade to replacement characters instead of
    crashing the fetch loop (#55381).
    """
    try:
        parts = decode_header(raw)
    except Exception:  # malformed RFC 2047 structure
        return raw
    return " ".join(_safe_decode(part, charset) if isinstance(part, bytes) else part for part, charset in parts)


def _first_body_part(msg: email_lib.message.Message, content_type: str) -> str:
    """Decoded text of the first non-attachment part of *content_type*, or ''."""
    for part in msg.walk():
        if "attachment" in str(part.get("Content-Disposition", "")) or part.get_content_type() != content_type:
            continue
        if payload := part.get_payload(decode=True):
            return _safe_decode(payload, part.get_content_charset())
    return ""


def _extract_text_body(msg: email_lib.message.Message) -> str:
    """Extract the plain-text body from a potentially multipart email."""
    if msg.is_multipart():
        html = _first_body_part(msg, "text/html")
        return _first_body_part(msg, "text/plain") or (_strip_html(html) if html else "")
    text = _safe_decode(payload, msg.get_content_charset()) if (payload := msg.get_payload(decode=True)) else ""
    return _strip_html(text) if msg.get_content_type() == "text/html" else text


def _strip_html(html: str) -> str:
    """Naive HTML tag stripper for fallback text extraction."""
    for pattern, repl in _HTML_SUBS:
        html = pattern.sub(repl, html)
    return html.strip()


def _extract_email_address(raw: str) -> str:
    """Extract bare email address from 'Name <addr>' format."""
    match = re.search(r"<([^>]+)>", raw)
    return (match.group(1) if match else raw).strip().lower()


def _domain_of(address: str) -> str:
    """Lowercased domain part of an email address, or ''."""
    return address.rpartition("@")[2].strip().lower()


def _domains_aligned(a: str, b: str) -> bool:
    """Relaxed DMARC alignment: equal, or one is a dot-suffix of the other."""
    a = (a or "").strip().lower().rstrip(".")
    b = (b or "").strip().lower().rstrip(".")
    return bool(a and b) and (a == b or a.endswith("." + b) or b.endswith("." + a))


def _verify_sender_authentication(msg: email_lib.message.Message, from_addr: str, *, authserv_id: str = "") -> Tuple[bool, str]:
    """Verify the ``From:`` domain is authenticated; returns ``(authenticated, reason)``.
    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy signal is the
    ``Authentication-Results`` header stamped by the *receiving* server. It prepends, so the FIRST
    instance is trusted and an injected copy sorts below it; pinned to *authserv_id* when given.
    True on DMARC pass, aligned SPF pass, or aligned DKIM (``header.d``) pass. No header → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``)."""
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    if not (headers := msg.get_all("Authentication-Results")):
        return False, "no Authentication-Results header"
    values = (" ".join(str(raw).split()) for raw in headers)  # authserv-id precedes the first ';'
    trusted = next((v for v in values if not authserv_id or (serv := v.split(";", 1)[0].strip().lower()) == authserv_id.lower()
                    or _domains_aligned(serv, authserv_id)), None)
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"
    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}
    if methods.get("dmarc") == "pass":  # DMARC already enforces From alignment
        return True, "dmarc=pass"
    if methods.get("spf") == "pass":  # envelope/MAIL FROM domain must align with From
        spf_domain = _domain_of(props.get("smtp.mailfrom", "")) or props.get("smtp.from", "") or props.get("envelope-from", "")
        if _domains_aligned(_domain_of(spf_domain) if "@" in spf_domain else spf_domain, from_domain):
            return True, "spf=pass aligned"
    if methods.get("dkim") == "pass":  # signing domain header.d must align with From
        dkim_domain = props.get("header.d", "") or _domain_of(props.get("header.from", ""))
        if _domains_aligned(dkim_domain, from_domain):
            return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _extract_attachments(msg: email_lib.message.Message, skip_attachments: bool = False) -> List[Dict[str, Any]]:
    """Extract attachment metadata and cache files locally (nothing when *skip_attachments*)."""
    attachments = []
    if not msg.is_multipart():
        return attachments
    for part in msg.walk():
        disposition, content_type = str(part.get("Content-Disposition", "")), part.get_content_type()
        if skip_attachments or ("attachment" not in disposition and (
                "inline" not in disposition or content_type in {"text/plain", "text/html"})):
            continue  # not an attachment, or an inline text/html body part
        filename = _decode_header_value(fn) if (fn := part.get_filename()) else f"attachment.{part.get_content_subtype() or 'bin'}"
        if not (payload := part.get_payload(decode=True)):
            continue
        if (ext := Path(filename).suffix.lower()) in _IMAGE_EXTS:
            try:
                cached_path, kind = cache_image_from_bytes(payload, ext), "image"
            except ValueError:
                logger.debug("Skipping non-image attachment %s (invalid magic bytes)", filename)
                continue
        else:
            cached_path, kind = cache_document_from_bytes(payload, filename), "document"
        attachments.append({"path": cached_path, "filename": filename, "type": kind, "media_type": content_type})
    return attachments


def _attach_file(msg: MIMEMultipart, path: Path, filename: str) -> None:
    """Attach *path* to *msg* as base64 application/octet-stream."""
    with open(path, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


class EmailAdapter(BasePlatformAdapter):
    """Email gateway adapter using IMAP (receive) and SMTP (send)."""

    splits_long_messages = True  # email has no practical length cap; the gateway must not truncate long sends

    # Per-account seen-UID snapshot surviving adapter recreation: the reconnect watcher builds a FRESH
    # adapter per retry; without this connect(is_reconnect=True) would re-mark the mailbox seen and skip
    # mail that arrived during the outage. Keyed by address (multiplex runs several accounts); same-process only.
    _seen_uids_snapshot: Dict[str, set] = {}

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.EMAIL)
        # Env first, then PlatformConfig.extra (config.yaml-only setups). Host/address are stripped: a stray
        # newline made IMAP4_SSL raise ``[Errno 8] nodename nor servname`` instead of "host not set".
        extra = config.extra or {}
        setting = lambda env, key: _get_secret(env, "") or extra.get(key, "")  # noqa: E731
        tls_verify = lambda env, key: _esecret_bool(env, is_truthy_value(extra.get(key), default=True))  # noqa: E731
        self._address = setting("EMAIL_ADDRESS", "address").strip()
        self._password = _get_secret("EMAIL_PASSWORD", "")
        self._imap_host = setting("EMAIL_IMAP_HOST", "imap_host").strip()
        self._imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
        self._imap_security = _normalize_security(setting("EMAIL_IMAP_SECURITY", "imap_security"))
        self._imap_tls_verify = tls_verify("EMAIL_IMAP_TLS_VERIFY", "imap_tls_verify")
        self._smtp_host = setting("EMAIL_SMTP_HOST", "smtp_host").strip()
        self._smtp_port = _esecret_int("EMAIL_SMTP_PORT", 587)
        self._smtp_security = _normalize_security(setting("EMAIL_SMTP_SECURITY", "smtp_security"), default="tls" if self._smtp_port == 465 else "starttls")
        self._smtp_tls_verify = tls_verify("EMAIL_SMTP_TLS_VERIFY", "smtp_tls_verify")
        self._poll_interval = _esecret_int("EMAIL_POLL_INTERVAL", 15)
        self._skip_attachments = extra.get("skip_attachments", False)  # platforms.email.skip_attachments
        # Require an authenticated From: domain (SPF/DKIM/DMARC) before trusting it for authorization
        # (GHSA-rxqh-5572-8m77). Default ON; opt out via require_authenticated_sender: false / EMAIL_TRUST_FROM_HEADER=true.
        if "require_authenticated_sender" in extra:
            self._require_authenticated_sender = bool(extra["require_authenticated_sender"])
        else:
            self._require_authenticated_sender = not _esecret_bool("EMAIL_TRUST_FROM_HEADER", False)
        # Optional authserv-id pinning Authentication-Results to the operator's own server (defeats an injected header sorting first).
        self._authserv_id = (extra.get("authserv_id", "") or _get_secret("EMAIL_AUTHSERV_ID", "")).strip().lower()
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._last_fetch_failed, self._last_fetch_error = False, ""  # "checked, nothing new" vs "the check itself failed"
        # Track the last IMAP fetch attempt so the poll loop can distinguish "checked, nothing new" from
        # "the check itself failed" (#80016).
        # Session lifecycle knob (config.yaml platforms.email.*): max_messages_per_session: 0 disables
        # auto-rotation (default 1000 — a thread rotates to a fresh session after this many inbound
        # messages so history re-processing cost stays bounded).
        try:
            self._max_messages_per_session = int(extra.get("max_messages_per_session", 1000) or 0)
        except (TypeError, ValueError):
            self._max_messages_per_session = 1000
        # Thread-context persistence (per-account path, multiplexing-safe): map of chat keys
        # (sender email / sender:slug / sender:thread_id) -> last subject + message-id (+ session
        # epoch/count) for reply threading. Persisted to JSON on every mutation and reloaded here so
        # replies keep their thread's subject/Message-ID across gateway restarts.
        try:
            from hermes_constants import get_hermes_home

            _state_dir = Path(get_hermes_home()) / "state"
        except Exception:
            _state_dir = Path.home() / ".hermes" / "state"
        _addr_slug = re.sub(r"[^a-z0-9]+", "_", self._address.lower()).strip("_") or "default"
        self._thread_context_path: Optional[Path] = _state_dir / f"email_thread_context_{_addr_slug}.json"
        self._thread_context: Dict[str, Dict[str, Any]] = self._load_thread_context()
        logger.info("[Email] Adapter initialized for %s", self._address)

    def _trim_seen_uids(self) -> None:
        """Keep only the highest half of UIDs once over the cap (UIDs are monotonic; UNSEEN prevents re-delivery)."""
        if len(self._seen_uids) <= self._seen_uids_max:
            return
        try:
            sorted_uids = sorted(self._seen_uids, key=lambda u: int(u))  # UIDs are bytes like b'1234'
            self._seen_uids = set(sorted_uids[-(self._seen_uids_max // 2):])
            logger.debug("[Email] Trimmed seen UIDs to %d entries", len(self._seen_uids))
        except (ValueError, TypeError):
            self._seen_uids = set(list(self._seen_uids)[-self._seen_uids_max // 2:])

    def _connect_imap(self) -> imaplib.IMAP4:
        """Create an IMAP connection using implicit TLS, STARTTLS, or plaintext."""
        if self._imap_security == "tls":
            return imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30, ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
        imap = imaplib.IMAP4(self._imap_host, self._imap_port, timeout=30)
        if self._imap_security == "starttls":
            try:
                imap.starttls(ssl_context=_tls_context(self._imap_tls_verify, self._imap_host))
            except Exception:
                _close_imap(imap)
                raise
        return imap

    @contextmanager
    def _inbox(self):
        """Logged-in IMAP handle on INBOX; always ``_close_imap``-ed on exit (a login/select failure used to leak one fd per reconnect)."""
        # Test IMAP connection. The handle is closed in ``finally`` — before this, a failure in
        # login/select/search left the TCP socket open with no owner, leaking one fd per connect attempt.
        # Under the gateway's reconnect watcher (fresh adapter instance per retry) against an
        # unreachable/proxied host this grew monotonically until fd exhaustion on macOS's 256 soft limit
        # (#79889).
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            imap.select("INBOX")
            yield imap
        finally:
            _close_imap(imap)

    # ── Thread-context persistence ─────────────────────────────────────────
    # _thread_context is in-memory; without persistence a gateway restart loses every thread's
    # subject/Message-ID and replies fall back to the generic "Hermes Agent" subject. Persist the map
    # to a small per-account JSON file on every mutation and reload on adapter construction.
    _thread_context_max_entries: int = 500

    def _load_thread_context(self) -> Dict[str, Dict[str, Any]]:
        """Load persisted thread context (subject/message_id per chat key)."""
        if not self._thread_context_path or not self._thread_context_path.exists():
            return {}
        try:
            raw = json.loads(self._thread_context_path.read_text("utf-8"))
            if not isinstance(raw, dict):
                return {}
            return {
                str(k): dict(v)
                for k, v in raw.items()
                if isinstance(v, dict) and isinstance(v.get("subject"), str)
            }
        except Exception as e:  # corrupt/partial file — start fresh
            logger.warning("[Email] Thread-context load failed (%s); starting fresh", e)
            return {}

    def _save_thread_context(self) -> None:
        """Persist the thread-context map atomically (tmp + rename)."""
        if not self._thread_context_path:
            return
        try:
            self._thread_context_path.parent.mkdir(parents=True, exist_ok=True)
            data = dict(self._thread_context)
            if len(data) > self._thread_context_max_entries:
                # Keep the most recent entries (dict preserves insertion order
                # — re-inserted keys move to the end, so drop the head).
                data = dict(list(data.items())[-self._thread_context_max_entries:])
            tmp = self._thread_context_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data), "utf-8")
            tmp.replace(self._thread_context_path)
        except Exception as e:
            logger.warning("[Email] Thread-context persist failed: %s", e)

    def _connect_smtp(self) -> smtplib.SMTP:
        """SMTP connection with TLS established (callers go straight to ``login()``). An unreachable IPv6 address can
        hang until the socket timeout, so connection-level failures retry through an IPv4-only socket path (no global
        resolver mutation); TLS verification errors are not retried."""
        host, port, security, ctx = self._smtp_host, self._smtp_port, self._smtp_security, _tls_context(self._smtp_tls_verify, self._smtp_host)
        try:
            return _open_smtp(host, port, security, ctx, smtplib.SMTP, smtplib.SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            if isinstance(exc, ssl.SSLError):
                raise
            return _open_smtp(host, port, security, ctx, _IPv4SMTP, _IPv4SMTP_SSL, timeout=SMTP_CONNECT_TIMEOUT)

    def _fail(self, log_fmt: str, err: object, code: str, detail: str, *, retryable: bool) -> bool:
        """Log *err*, record a fatal error for the gateway's reconnect machinery, return False."""
        logger.error(log_fmt, err)
        self._set_fatal_error(code, detail, retryable=retryable)
        return False

    def _probe_imap(self, is_reconnect: bool) -> bool:
        """Connection test + seen-UID baseline. Sets a fatal error and returns False on failure."""
        try:
            with self._inbox() as imap:
                snapshot = self._seen_uids_snapshot.get(self._address)
                if is_reconnect and snapshot is not None:
                    # Same-process reconnect: restore the previous adapter's baseline so mail that
                    # arrived during the outage stays eligible for the next poll.
                    self._seen_uids = set(snapshot)
                    passed = "[Email] IMAP reconnect test passed. Restored %d seen UIDs; messages received during the outage will be processed."
                else:  # first connect (or no snapshot): mark all existing messages seen
                    status, data = imap.uid("search", None, "ALL")
                    self._seen_uids.update(data[0].split() if status == "OK" and data and data[0] else ())
                    passed = "[Email] IMAP connection test passed. %d existing messages skipped."
                self._trim_seen_uids()
                logger.info(passed, len(self._seen_uids))
            self._seen_uids_snapshot[self._address] = set(self._seen_uids)
            return True
        except Exception as e:
            # Always set an explicit fatal code, else the gateway treats every failure as transient with zero
            # owner signal. retryable=True because imaplib raises the same generic IMAP4.error for bad credentials
            # AND transient NOs (Gmail "too many simultaneous connections"); loops surface via NEEDS_ATTENTION.
            return self._fail("[Email] IMAP connection failed: %s", e, "email_imap_connect_error",
                              f"IMAP connection to {self._imap_host}:{self._imap_port} failed: {e}", retryable=True)

    def _probe_smtp(self) -> bool:
        """SMTP connect + login test. Sets a fatal error and returns False on failure."""
        try:
            smtp = self._connect_smtp()
            try:
                smtp.login(self._address, self._password)
            finally:
                smtp.quit()
            logger.info("[Email] SMTP connection test passed.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # Typed auth failure (535 & friends) can never self-heal, so drop out of the reconnect queue — unambiguous, unlike IMAP4.error.
            return self._fail("[Email] SMTP authentication failed: %s", e, "email_auth_error",
                              f"SMTP authentication failed for {self._address}: {e}. Check EMAIL_PASSWORD (for Gmail/Outlook "
                              "this must be an app password, not the account password).", retryable=False)
        except Exception as e:
            return self._fail("[Email] SMTP connection failed: %s", e, "email_smtp_connect_error",
                              f"SMTP connection to {self._smtp_host} failed: {e}", retryable=True)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the IMAP server and start polling for new messages."""
        # Validate up front so a missing host is an actionable config error, not IMAP4_SSL("") raising ``[Errno 8]``.
        required = (("EMAIL_ADDRESS", self._address), ("EMAIL_PASSWORD", self._password), ("EMAIL_IMAP_HOST", self._imap_host), ("EMAIL_SMTP_HOST", self._smtp_host))
        if missing := [name for name, value in required if not value]:
            message = f"Not configured — missing {', '.join(missing)}. Set it via `hermes gateway setup` (env) or platforms.email in config.yaml."
            # Non-retryable: a blank-but-present env var used to drive an indefinite retry loop that leaked until OOM.
            return self._fail("[Email] %s", message, "email_missing_configuration", message, retryable=False)
        if not self._probe_imap(is_reconnect) or not self._probe_smtp():
            return False
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        print(f"[Email] Connected as {self._address}")
        self._wire_plugin_handlers(None)  # plugin-registered native handlers
        return True

    async def disconnect(self) -> None:
        """Stop polling and disconnect."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._poll_task
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
        messages = await asyncio.get_running_loop().run_in_executor(None, self._fetch_new_messages)
        # Dispatch partial results BEFORE escalating a failure — a mid-batch exception returns what was fetched (already marked seen).
        for msg_data in messages:
            await self._dispatch_message(msg_data)
        if self._last_fetch_failed:
            # The IMAP check itself failed (not an empty inbox): route through the fatal-error hook so the gateway's
            # reconnect/backoff re-establishes the mailbox. The handler runs detached (gateway/run.py), so awaiting it is safe.
            # The handler runs in a detached task (gateway/run.py), so awaiting it from our own poll task is
            # safe even though teardown cancels this task. See #80016.
            self._last_fetch_failed = False
            self._set_fatal_error("email_imap_fetch_failed", self._last_fetch_error or "IMAP fetch failed", retryable=True)
            await self._notify_fatal_error()

    def _fetch_new_messages(self) -> List[Dict[str, Any]]:
        """Fetch new (unseen) messages from IMAP. Runs in executor thread."""
        results = []
        try:
            with self._inbox() as imap:
                status, data = imap.uid("search", None, "UNSEEN")
                for uid in (data[0].split() if status == "OK" and data and data[0] else []):
                    if uid in self._seen_uids:
                        continue
                    status, msg_data = imap.uid("fetch", uid, "(RFC822)")
                    if status != "OK":
                        continue  # transient per-UID refusal: leave unseen so the next poll retries
                    # Mark seen once a response arrived (even malformed) so garbage is skipped once, not retried forever —
                    # but NOT before the fetch: a connection failure must leave the rest of the batch eligible for the next poll.
                    # IMAP fetch can return unexpected structures (e.g. a single bytes item instead of a
                    # list of tuples). See #80032.
                    self._seen_uids.add(uid)
                    self._trim_seen_uids()
                    try:
                        raw_email = msg_data[0][1]
                    except (IndexError, TypeError):
                        logger.warning("[Email] Unexpected IMAP response structure for UID %s, skipping", uid)
                        continue
                    if not isinstance(raw_email, (bytes, bytearray)):
                        logger.warning("[Email] Non-bytes IMAP payload for UID %s, skipping", uid)
                        continue
                    # One poison message (unparseable headers, pathological attachment, DNS hiccup) must not abort the batch or force a reconnect.
                    try:
                        # See #80032.
                        parsed = self._parse_fetched_message(uid, raw_email)
                    except Exception as parse_exc:
                        logger.error("[Email] Failed to process message UID %s, skipping: %s", uid, parse_exc)
                        continue
                    if parsed is not None:
                        results.append(parsed)
        except Exception as e:
            # _close_imap guarantees the socket dies even when logout() raises IMAP4.abort on a broken
            # connection (#79889).
            logger.error("[Email] IMAP fetch error: %s", e)
            self._last_fetch_failed, self._last_fetch_error = True, str(e)
        # Keep the reconnect snapshot current so a mid-outage adapter recreation does not re-dispatch messages already processed.
        self._seen_uids_snapshot[self._address] = set(self._seen_uids)
        return results

    def _parse_fetched_message(self, uid: bytes, raw_email: "bytes | bytearray") -> Optional[Dict[str, Any]]:
        """Parse one RFC822 payload into a dispatchable dict; ``None`` for automated senders. Raises on pathological input (caller logs + continues)."""
        msg = email_lib.message_from_bytes(raw_email)
        sender_addr, sender_name = _extract_email_address(msg.get("From", "")), _decode_header_value(msg.get("From", ""))
        if "<" in sender_name:
            sender_name = sender_name.split("<")[0].strip().strip('"')
        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        if _is_automated_sender(sender_addr, dict(msg.items())):
            logger.debug("[Email] Skipping automated sender: %s", sender_addr)
            return None
        # Verify From: while the trusted Authentication-Results header is in scope; the verdict is consumed at dispatch (GHSA-rxqh-5572-8m77).
        sender_authenticated, auth_reason = _verify_sender_authentication(msg, sender_addr, authserv_id=self._authserv_id)
        return {"uid": uid, "sender_addr": sender_addr, "sender_name": sender_name, "subject": subject,
                "message_id": msg.get("Message-ID", ""), "in_reply_to": msg.get("In-Reply-To", ""),
                "body": _extract_text_body(msg),
                "attachments": _extract_attachments(msg, skip_attachments=self._skip_attachments),
                "date": msg.get("Date", ""), "sender_authenticated": sender_authenticated, "auth_reason": auth_reason}

    @staticmethod
    def _allow_all_senders() -> bool:
        """True when the operator opted into any sender (EMAIL_ or GATEWAY_ALLOW_ALL_USERS)."""
        return (_get_secret("EMAIL_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY or os.getenv("GATEWAY_ALLOW_ALL_USERS", "").strip().lower() in _TRUTHY)

    @staticmethod
    def _allowlist_in_effect() -> bool:
        """True when EMAIL_/GATEWAY_ALLOWED_USERS gates access (without one the gateway default-denies, so the spoofable From: grants nothing)."""
        return bool(_get_secret("EMAIL_ALLOWED_USERS", "").strip() or os.getenv("GATEWAY_ALLOWED_USERS", "").strip())

    def _sender_accepted(self, sender_addr: str, msg_data: Dict[str, Any]) -> bool:
        """Pre-dispatch sender gate: self, automated, allowlist, From: authentication."""
        if sender_addr == self._address.lower():
            return False
        if _is_automated_sender(sender_addr, {}):
            logger.debug("[Email] Dropping automated sender at dispatch: %s", sender_addr)
            return False
        # Drop senders the gateway would never authorize before a MessageEvent (and thread context) exists —
        # otherwise a dispatch/authorization race can send a reply even though the handler returned None.
        allowed_raw = _get_secret("EMAIL_ALLOWED_USERS", "").strip()
        if not allowed_raw:
            if not self._allow_all_senders():
                logger.debug("[Email] Dropping sender at dispatch — EMAIL_ALLOWED_USERS is unset and open access is not opted in: %s", sender_addr)
                return False
        elif sender_addr.lower() not in {a.strip().lower() for a in allowed_raw.split(",") if a.strip()}:
            logger.debug("[Email] Dropping non-allowlisted sender at dispatch: %s", sender_addr)
            return False
        # Reject spoofed senders (GHSA-rxqh-5572-8m77): the allowlist keys on the attacker-controlled
        # From:. Only matters when an allowlist GRANTS access and allow-all is off; fail-closed.
        if (self._require_authenticated_sender and self._allowlist_in_effect()
                and not self._allow_all_senders() and not msg_data.get("sender_authenticated", False)):
            logger.warning("[Email] Dropping sender with unauthenticated From: %s (%s). If your mail server does not "
                           "stamp Authentication-Results, set platforms.email.require_authenticated_sender: false "
                           "(or EMAIL_TRUST_FROM_HEADER=true) to accept the risk.",
                           sender_addr, msg_data.get("auth_reason", "no verdict"))
            return False
        return True

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        if not self._sender_accepted(sender_addr, msg_data):
            return
        subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]

        # Subject-based session isolation (#26277/#27804): derive a stable slug from the normalized
        # subject and use it as thread_id so each distinct subject gets its own session, while replies
        # (Re:/Fwd:/Fw: prefixes stripped) rejoin the same thread. Empty subjects fall back to the
        # sender-only session (previous behaviour).
        _subject_slug = None
        if subject:
            _inner = re.sub(r"(?i)^(re|fwd|fw)[-:\s]*\d*[-:\s]*", "", subject)
            # Strip stacked prefixes (e.g. "Re: Fw: Meeting Notes")
            while re.match(r"(?i)^(re|fwd|fw)[-:\s]", _inner):
                _inner = re.sub(r"(?i)^(re|fwd|fw)[-:\s]*\d*[-:\s]*", "", _inner)
            _subject_slug = (
                re.sub(r"[^a-z0-9]+", "-", _inner.lower()).strip("-")[:80] or None
            )

        # ── Session lifecycle: /new in body + auto-rotation cap ──────────────
        # A per-thread epoch counter (persisted with the thread context) lets a "/new" email body — or
        # the message-count cap — start a FRESH session for this thread, while replies with the same
        # subject keep rejoining the current generation ("<slug>-<epoch>").
        _base_key = f"{sender_addr}:{_subject_slug}" if _subject_slug else sender_addr
        _cur_ctx = self._thread_context.get(_base_key, {}) or {}
        try:
            _epoch = int(_cur_ctx.get("epoch", 0) or 0)
        except (TypeError, ValueError):
            _epoch = 0
        try:
            _msg_count = int(_cur_ctx.get("msg_count", 0) or 0) + 1
        except (TypeError, ValueError):
            _msg_count = 1

        # "/new" (optionally followed by a fresh request) rotates the thread.
        _is_new_cmd = bool(re.match(r"^/new\b", body, re.IGNORECASE))
        if _is_new_cmd:
            _epoch += 1
            body = re.sub(r"^/new\b[ \t]*", "", body, flags=re.IGNORECASE).strip()

        # Message-count cap: rotate once this thread exceeds the threshold so history re-processing
        # cost stays bounded (the $2.69/4,175-call burst came from a 20K-message session
        # re-processing everything per reply).
        _cap_rotation = (
            not _is_new_cmd
            and self._max_messages_per_session > 0
            and _msg_count > self._max_messages_per_session
        )
        if _cap_rotation:
            _epoch += 1
            _msg_count = 1

        _thread_id = (
            f"{_subject_slug}-{_epoch}" if (_epoch and _subject_slug) else _subject_slug
        )
        if _epoch and not _subject_slug:
            _thread_id = f"new-{_epoch}"

        text = f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body  # subject unless reply
        # DOCUMENT wins over PHOTO for mixed attachments: run.py keys image handling off the per-path mime type regardless
        # of message_type, but document-context injection gates strictly on MessageType.DOCUMENT — so DOCUMENT surfaces both.
        kinds = {att["type"] for att in attachments}

        # Store thread context for reply threading — keyed by the compound session id (sender:thread_id)
        # so a reply carries THIS thread's subject, the base key (sender:slug) so the next dispatch
        # finds the current epoch/count, plus the plain sender for legacy sender-only sessions.
        # Persisted on every write so the mapping survives gateway restarts.
        _ctx_key = f"{sender_addr}:{_thread_id}" if _thread_id else sender_addr
        _ctx_entry = {
            "subject": subject,
            "message_id": msg_data["message_id"],
            "epoch": _epoch,
            "msg_count": _msg_count,
        }
        self._thread_context[_ctx_key] = _ctx_entry
        self._thread_context[_base_key] = dict(_ctx_entry)
        self._thread_context[sender_addr] = dict(_ctx_entry)
        self._save_thread_context()

        # A bare "/new" body resets the thread without running the agent — send a short confirmation
        # and skip dispatch (saves tokens; the fresh request in the same body, if any, is dispatched).
        if _is_new_cmd and not body:
            logger.info(
                "[Email] /new from %s — session reset for %s (epoch %d)",
                sender_addr, _base_key, _epoch,
            )
            await self.send(
                chat_id=_ctx_key,
                content=(
                    "✅ New session started for this thread. "
                    "Send your next email to begin."
                ),
            )
            return

        name = msg_data["sender_name"] or sender_addr
        event = MessageEvent(
            text=text or "(empty email)", message_id=msg_data["message_id"],
            message_type=MessageType.DOCUMENT if "document" in kinds else MessageType.PHOTO if "image" in kinds else MessageType.TEXT,
            source=self.build_source(chat_id=sender_addr, chat_name=name, chat_type="dm", user_id=sender_addr, user_name=name,
                                     thread_id=_thread_id),
            media_urls=[att["path"] for att in attachments], media_types=[att["media_type"] for att in attachments],
            reply_to_message_id=msg_data["in_reply_to"] or None)
        logger.info("[Email] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking SMTP sender in the executor; wrap its Message-ID in a SendResult."""
        try:
            return SendResult(success=True, message_id=await asyncio.get_running_loop().run_in_executor(None, fn, *args))
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an email reply to the given address."""
        thread_id = (metadata or {}).get("thread_id")
        explicit_subject = (metadata or {}).get("subject")
        return await self._run_send(self._send_email, (chat_id, content, reply_to, thread_id, explicit_subject), "[Email] Send failed to %s: %s", chat_id)

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, *,
                   thread_id: Optional[str] = None, explicit_subject: Optional[str] = None,
                   attach_empty_body: bool = False) -> Tuple[MIMEMultipart, str, str]:
        """Build a threaded reply skeleton. Returns ``(msg, msg_id, subject)``."""
        # Extract the actual email address from a compound chat_id (sender:thread_id).
        recipient = to_addr.split(":")[0] if ":" in to_addr else to_addr
        msg = MIMEMultipart()
        # Thread context for reply — resolve the compound (sender:thread_id) key via metadata
        # thread_id FIRST so a reply to THIS thread carries THIS thread's subject even when several of
        # the sender's threads were dispatched in the same poll cycle (the plain-sender key holds the
        # last dispatch's subject). The plain-sender key is consulted ONLY for genuine replies (a
        # reply-to message id is present): bare outbound sends (cron deliveries) must never inherit
        # the sender's last inbound subject, or Gmail threads the report into that conversation.
        ctx = {}
        if thread_id:
            ctx = self._thread_context.get(f"{recipient}:{thread_id}", {}) or {}
        if not ctx and reply_to_msg_id:
            ctx = self._thread_context.get(to_addr) or self._thread_context.get(recipient, {})
        if explicit_subject:
            # Outbound-only send (cron/standalone): verbatim subject, fresh conversation — no Re:
            # prefix, no thread-context threading headers.
            subject = explicit_subject  # keep the success log line bound
            original_msg_id = reply_to_msg_id
        else:
            if ctx:
                # Genuine reply context: keep the thread's subject with Re:.
                subject = ctx.get("subject", "Hermes Agent")
                if not subject.startswith("Re:"):
                    subject = f"Re: {subject}"
            else:
                # No thread context at all (bare outbound send): neutral.
                subject = "Hermes Agent"
            original_msg_id = reply_to_msg_id or ctx.get("message_id")
        threading = (("In-Reply-To", original_msg_id), ("References", original_msg_id)) if original_msg_id else ()
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        for key, value in (("From", self._address), ("To", recipient), ("Subject", subject), *threading,
                           ("Date", formatdate(localtime=True)), ("Message-ID", msg_id)):
            msg[key] = value
        if body or attach_empty_body:
            msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg, msg_id, subject

    def _smtp_send(self, msg: MIMEMultipart) -> None:
        """Login, send, and always release the SMTP connection (quit, else close)."""
        smtp = self._connect_smtp()
        try:
            smtp.login(self._address, self._password)
            smtp.send_message(msg)
        finally:
            try:
                smtp.quit()
            except Exception:
                smtp.close()

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None,
                    thread_id: Optional[str] = None, explicit_subject: Optional[str] = None) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, thread_id=thread_id,
                                               explicit_subject=explicit_subject, attach_empty_body=True)
        self._smtp_send(msg)
        logger.info("[Email] Sent reply to %s (subject: %s)", to_addr, subject)
        return msg_id

    def _send_with_files(self, to_addr: str, body: str, files: List[Tuple[Path, str]], *, lenient: bool,
                         thread_id: Optional[str] = None, explicit_subject: Optional[str] = None) -> str:
        """Send a reply with attachments; *lenient* logs-and-skips unattachable files instead of raising."""
        msg, msg_id, _ = self._new_reply(to_addr, body, thread_id=thread_id, explicit_subject=explicit_subject)
        for path, name in files:
            try:
                _attach_file(msg, path, name)
            except Exception as e:
                if not lenient:
                    raise
                logger.warning("[Email] Failed to attach %s: %s", path, e)
        self._smtp_send(msg)
        return msg_id

    def format_tool_event(self, event: Any, *, mode: str = "all",
                          preview_max_len: int = 40) -> None:
        """Suppress tool-progress chrome for email delivery.

        Email is a plain-text, non-editable medium — there is no way to
        update a previously sent message, so emitting a separate email per
        tool call would spam the user's inbox. Return None to drop all
        tool-progress events; the final assistant response is the only
        message sent (#27804).
        """
        return None

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image URL as part of an email body (``metadata`` unused)."""
        return await self.send(chat_id, f"{caption or ''}\n\nImage: {image_url}".strip(), reply_to)

    async def send_multiple_images(self, chat_id: str, images: List[Tuple[str, str]],
                                   metadata: Optional[Dict[str, Any]] = None, human_delay: float = 0.0) -> None:
        """One email per batch: local files attached, URL images linked in the body (no remote download); base-class fallback on failure."""
        if not images:
            return
        from urllib.parse import unquote as _unquote
        body_parts, local_paths = [], []
        for image_url, alt_text in images:
            if alt_text:
                body_parts.append(alt_text)
            if not image_url.startswith("file://"):
                body_parts.append(f"Image: {image_url}")  # parity with send_image
            elif Path(local_path := _unquote(image_url[7:])).exists():
                local_paths.append(local_path)
            else:
                logger.warning("[Email] Skipping missing image: %s", local_path)
        if not local_paths and not body_parts:
            return
        try:
            thread_id = (metadata or {}).get("thread_id")
            explicit_subject = (metadata or {}).get("subject")
            await asyncio.get_running_loop().run_in_executor(
                None, self._send_email_with_attachments, chat_id, "\n\n".join(body_parts), local_paths,
                thread_id, explicit_subject)
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: List[str],
                                     thread_id: Optional[str] = None, explicit_subject: Optional[str] = None) -> str:
        """Send an email with multiple file attachments via SMTP (unattachable files are skipped)."""
        msg_id = self._send_with_files(to_addr, body, [(Path(f), Path(f).name) for f in file_paths],
                                       lenient=True, thread_id=thread_id, explicit_subject=explicit_subject)
        logger.info("[Email] Sent multi-attachment email to %s (%d files)", to_addr, len(file_paths))
        return msg_id

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        """Send a file as an email attachment."""
        metadata = (kwargs or {}).get("metadata") or {}
        thread_id = metadata.get("thread_id")
        explicit_subject = metadata.get("subject")
        return await self._run_send(self._send_email_with_attachment, (chat_id, caption or "", file_path, file_name,
                                                                       thread_id, explicit_subject), "[Email] Send document failed: %s")

    def _send_email_with_attachment(self, to_addr: str, body: str, file_path: str, file_name: Optional[str] = None,
                                    thread_id: Optional[str] = None, explicit_subject: Optional[str] = None) -> str:
        """Send an email with a single file attachment via SMTP (raises if unattachable)."""
        return self._send_with_files(to_addr, body, [(Path(file_path), file_name or Path(file_path).name)],
                                     lenient=False, thread_id=thread_id, explicit_subject=explicit_subject)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        return {"name": chat_id, "type": "dm", "chat_id": chat_id, "subject": self._thread_context.get(chat_id, {}).get("subject", "")}


# Plugin glue: register() exposes the platform via the registry; EMAIL_* env → PlatformConfig seeding stays in core.
async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery via SMTP (one-shot); standalone_sender_fn contract."""
    extra = getattr(pconfig, "extra", {}) or {}
    address, password = extra.get("address") or _get_secret("EMAIL_ADDRESS", ""), _get_secret("EMAIL_PASSWORD", "")
    smtp_host, smtp_port = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", ""), _esecret_int("EMAIL_SMTP_PORT", 587)
    smtp_security = _normalize_security(_get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"), default="tls" if smtp_port == 465 else "starttls")
    smtp_tls_verify = _esecret_bool("EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True))
    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}
    try:
        msg = MIMEText(message, "plain", "utf-8")
        # Extract the actual email address from compound chat_id (cron delivery routes thread-scoped
        # origins like "user@gmail.com:thread_id" here — Gmail rejects with 555 5.5.2).
        recipient = chat_id.split(":")[0] if ":" in chat_id else chat_id
        for key, value in (("From", address), ("To", recipient), ("Subject", "Hermes Agent"), ("Date", formatdate(localtime=True))):
            msg[key] = value
        server = _open_smtp(smtp_host, smtp_port, smtp_security, _tls_context(smtp_tls_verify, smtp_host), smtplib.SMTP, smtplib.SMTP_SSL)
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
    """Connected when an address is configured (PlatformConfig.extra or EMAIL_ADDRESS)."""
    if (getattr(config, "extra", {}) or {}).get("address"):
        return True
    import hermes_cli.gateway as gateway_mod
    return bool((gateway_mod.get_env_value("EMAIL_ADDRESS") or "").strip())


def _build_adapter(config):
    """Factory wrapper that constructs EmailAdapter from a PlatformConfig."""
    return EmailAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="email", label="Email", adapter_factory=_build_adapter, check_fn=check_email_requirements, is_connected=_is_connected,
        required_env=["EMAIL_ADDRESS", "EMAIL_PASSWORD", "EMAIL_SMTP_HOST"],
        install_hint="Email uses the Python stdlib (smtplib/imaplib) — no extra deps", allowed_users_env="EMAIL_ALLOWED_USERS",
        allow_all_env="EMAIL_ALLOW_ALL_USERS", cron_deliver_env_var="EMAIL_HOME_ADDRESS", standalone_sender_fn=_standalone_send,
        max_message_length=50_000, pii_safe=True, emoji="📧", allow_update_command=True)
