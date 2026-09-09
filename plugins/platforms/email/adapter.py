"""Email platform adapter for the Hermes gateway: users talk to Hermes by sending email; IMAP (polled)
receives, SMTP sends. Configured via EMAIL_* env vars or ``platforms.email`` in config.yaml (see website docs)."""

import asyncio
import email as email_lib
from collections import OrderedDict
from contextlib import contextmanager, suppress
import imaplib
import logging
import os
import re
import smtplib
import socket
import ssl
import time
import uuid
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.utils import formatdate, parseaddr
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import (
    BasePlatformAdapter, SendResult,
    cache_document_from_bytes, cache_image_from_bytes,
)
from gateway.platforms.event import MessageEvent, MessageType
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

# Review-first outbound dispositions: "sent" went out via SMTP; "drafted" was appended to the IMAP
# drafts mailbox for human review and never touched SMTP.
DISPOSITION_SENT = "sent"
DISPOSITION_DRAFTED = "drafted"
# Metadata key stamped ONLY by trusted gateway-composed send paths (cron delivery via DeliveryRouter,
# lifecycle notifications via _non_conversational_metadata). Under review_first it authorizes
# transmission solely to the auto-send allowlist; never derived from message content.
GATEWAY_INTERNAL_SEND_KEY = "gateway_internal_send"


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
    returns ``BYE Unsafe Login``); other servers may reject it, so failures are swallowed.

    Sent only when the server advertises ``ID`` (RFC 2971 requires advertising it): a server
    without the extension can answer an untagged ``* BYE Unknown command.`` and close the
    connection, which imaplib cannot surface here — the failure appears one command later
    as a misleading SELECT error and the adapter retries forever (Purelymail, #39856).
    ``imap.capabilities`` is populated by imaplib at connect, so the check is free."""
    if "ID" not in imap.capabilities:
        logger.debug(
            "[Email] Server does not advertise IMAP ID capability; skipping ID"
        )
        return
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
    """Extract the bare email address from 'Name <addr>' format via RFC 2822 parsing.
    parseaddr honors quoting, so a display name containing an angle-bracketed address
    ('"x <victim@a>" <attacker@b>') can never shadow the real sender the way a first-match
    regex could; the regex remains only as a fallback for garbage parseaddr rejects."""
    parsed = parseaddr(raw or "")[1].strip().lower()
    if parsed:
        return parsed
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


def _normalize_addr(addr: Any) -> str:
    """Normalize one email address for comparison: parse, strip, lowercase."""
    return parseaddr(str(addr or ""))[1].strip().lower()


def _normalize_address_allowlist(raw: Any) -> frozenset:
    """Normalize ``auto_send_authenticated_senders`` fail-closed: a bare string is a one-element list
    (common YAML mistake); non-string or ``@``-less entries drop with a warning; any other shape yields
    an empty set — which under review_first drafts everything."""
    raw = [raw] if isinstance(raw, str) else raw
    if not isinstance(raw, (list, tuple, set, frozenset)):
        if raw is not None:
            logger.warning("[Email] auto_send_authenticated_senders must be a list of addresses; got %r — "
                           "treating as empty (fail closed)", type(raw).__name__)
        return frozenset()
    normalized = set()
    for entry in raw:
        addr = _normalize_addr(entry) if isinstance(entry, str) else ""
        if "@" in addr:
            normalized.add(addr)
        else:
            logger.warning("[Email] Ignoring malformed allowlist entry %r", entry)
    return frozenset(normalized)


def _positive_int(raw: Any, default: int) -> int:
    """Coerce a config value to a positive int, falling back to *default*."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _select_strict_auth_header(msg: email_lib.message.Message, authserv_id: str) -> "Tuple[Optional[str], str]":
    """Position-verified exact-pin Authentication-Results selection for review_first.

    An authserv-id string inside a header is itself attacker-controlled text, so pinning alone cannot
    establish provenance — position can: the receiving server prepends its trace headers, so only
    headers ABOVE the message's first transport ``Received:`` line were written by the receiving
    infrastructure. Requires an exact pin match (no subdomain alignment), exactly ONE matching header
    in that region, and at least one ``Received:`` header (every SMTP-delivered message carries the
    receiving server's own); anything else fails closed. Returns ``(header_value | None, reason)``."""
    if not authserv_id:
        return None, "review_first requires a pinned authserv_id (platforms.email.authserv_id) — failing closed"
    pin, first_received, candidates = authserv_id.strip().lower(), None, []
    for idx, (name, value) in enumerate(msg.items()):
        lname = name.lower()
        if lname == "received":
            first_received = idx if first_received is None else first_received
        elif lname == "authentication-results":
            v = " ".join(str(value).split())
            if v.split(";", 1)[0].strip().lower() == pin:
                candidates.append((idx, v))
    if first_received is None:
        return None, "no transport Received header — cannot verify Authentication-Results provenance, failing closed"
    candidates = [(i, v) for i, v in candidates if i < first_received]
    if not candidates:
        return None, "no Authentication-Results from the pinned authserv-id above the transport Received chain — failing closed"
    if len(candidates) > 1:
        return None, f"ambiguous Authentication-Results: {len(candidates)} headers claim the trusted authserv-id — failing closed"
    return candidates[0][1], ""


_STRICT_CLAUSE_METHOD_RE = re.compile(r"^\s*(dmarc|dkim|spf)\s*=\s*([a-z0-9]+)", re.IGNORECASE)


def _strict_authentication_verdict(trusted: str, from_domain: str) -> Tuple[bool, str]:
    """Method-scoped evaluation of one server-stamped Authentication-Results value (review_first).

    RFC 8601 resinfo units are ``;``-separated clauses, each opening with ``method=result``; a
    property binds only to the clause that produced it (a ``header.d`` sitting in an spf clause
    proves nothing about DKIM), and DKIM accepts ONLY an aligned ``header.d`` — never
    ``header.from``, which is the attacker-controlled visible identity, not a signing identity.
    Duplicate clauses for the same method are ambiguous and fail that method closed, as do
    clauses this parser cannot recognize as opening with a known method. Scoping and identity
    semantics follow the direction of #56608 (method-scoped AR properties, DKIM identity), which
    owns the repo-wide legacy parser under the #103093 tracker with authserv trust-topology in
    #88560; they are enforced here independently because under review_first this verdict becomes
    automatic SMTP send authority."""
    clauses: Dict[str, Optional[Tuple[str, Dict[str, str]]]] = {}
    for raw_clause in trusted.split(";")[1:]:  # segment 0 is the authserv-id
        m = _STRICT_CLAUSE_METHOD_RE.match(raw_clause)
        if not m:
            continue
        method, result = m.group(1).lower(), m.group(2).lower()
        props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(raw_clause)}
        clauses[method] = None if method in clauses else (result, props)

    def _clause(method: str) -> Tuple[str, Dict[str, str]]:
        entry = clauses.get(method)
        return entry if entry else ("", {})

    result, props = _clause("dmarc")
    if result == "pass":
        hf = props.get("header.from", "")
        dmarc_domain = _domain_of(hf) if "@" in hf else hf
        if dmarc_domain and _domains_aligned(dmarc_domain, from_domain):
            return True, "dmarc=pass"
    result, props = _clause("spf")
    if result == "pass":
        spf_domain = _domain_of(props.get("smtp.mailfrom", "")) or props.get("smtp.from", "") or props.get("envelope-from", "")
        if _domains_aligned(_domain_of(spf_domain) if "@" in spf_domain else spf_domain, from_domain):
            return True, "spf=pass aligned"
    result, props = _clause("dkim")
    if result == "pass" and _domains_aligned(props.get("header.d", ""), from_domain):
        return True, "dkim=pass aligned"
    return False, f"authentication failed ({trusted[:120]})"


def _verify_sender_authentication(msg: email_lib.message.Message, from_addr: str, *, authserv_id: str = "",
                                  strict: bool = False) -> Tuple[bool, str]:
    """Verify the ``From:`` domain is authenticated; returns ``(authenticated, reason)``.
    ``From:`` is attacker-controlled (GHSA-rxqh-5572-8m77); the only trustworthy signal is the
    ``Authentication-Results`` header stamped by the *receiving* server. It prepends, so the FIRST
    instance is trusted and an injected copy sorts below it; pinned to *authserv_id* when given.
    *strict* (review_first) hardens the selection via :func:`_select_strict_auth_header` and
    evaluates it with :func:`_strict_authentication_verdict` (method-scoped properties, no DKIM
    ``header.from`` fallback).
    True on DMARC pass, aligned SPF pass, or aligned DKIM (``header.d``) pass. No header → fail-closed
    (opt out via ``EmailAdapter._require_authenticated_sender``)."""
    from_domain = _domain_of(from_addr)
    if not from_domain:
        return False, "missing From domain"
    if not (headers := msg.get_all("Authentication-Results")):
        return False, "no Authentication-Results header"
    if strict:
        trusted, reason = _select_strict_auth_header(msg, authserv_id)
        if trusted is None:
            return False, reason
        return _strict_authentication_verdict(trusted, from_domain)
    values = (" ".join(str(raw).split()) for raw in headers)  # authserv-id precedes the first ';'
    trusted = next((v for v in values if not authserv_id or (serv := v.split(";", 1)[0].strip().lower()) == authserv_id.lower()
                    or _domains_aligned(serv, authserv_id)), None)
    if trusted is None:
        return False, "no Authentication-Results from trusted authserv-id"
    methods = {m.lower(): r.lower() for m, r in _AUTH_METHOD_RE.findall(trusted)}
    props = {p.lower(): v.strip().strip('"') for p, v in _AUTH_PROP_RE.findall(trusted)}
    if methods.get("dmarc") == "pass":
        # DMARC enforces alignment only for the identity it evaluated (header.from) — that
        # domain must itself align with the parsed From: domain, or a legitimately-passing
        # third-party message whose From: was mis-extracted (display-name spoof) would be
        # trusted. Legacy accepts a server that did not record header.from, but a recorded
        # misaligned domain never authenticates — aligned SPF/DKIM below may still vouch for
        # the actual From: domain.
        hf = props.get("header.from", "")
        dmarc_domain = _domain_of(hf) if "@" in hf else hf
        if not dmarc_domain or _domains_aligned(dmarc_domain, from_domain):
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
        # Review-first outbound policy (config.yaml only — behavior settings are not env vars; .env stays
        # credentials-only). Absent/empty preserves the legacy always-send behavior ("direct"); an unknown
        # value fails closed into review_first so a typo can never silently disable review.
        raw_policy = str(extra.get("outbound_policy") or "").strip().lower()
        if raw_policy not in ("", "direct", "review_first"):
            logger.error("[Email] Unknown outbound_policy %r — failing closed to review_first (valid: direct, review_first)", raw_policy)
        self._outbound_policy = "direct" if not raw_policy else ("direct" if raw_policy == "direct" else "review_first")
        # Auto-send allowlist: replies transmit via SMTP only for senders on this list whose inbound
        # message passed authentication. Independent of EMAIL_ALLOWED_USERS / EMAIL_ALLOW_ALL_USERS
        # (which gate access, not send authority). Empty/malformed ⇒ fail closed: every reply drafts.
        self._auto_send_allowlist = _normalize_address_allowlist(extra.get("auto_send_authenticated_senders"))
        self._drafts_mailbox = str(extra.get("drafts_mailbox") or "Drafts").strip() or "Drafts"
        self._sent_mailbox = str(extra.get("sent_mailbox") or "Sent").strip() or "Sent"
        # Sent archival via IMAP APPEND: default on under review_first, off under direct (legacy installs
        # stay byte-identical).
        self._save_sent = is_truthy_value(extra.get("save_sent"), default=(self._outbound_policy == "review_first"))
        # Under review_first EVERY email-triggered turn is tool-free — trusted senders included. Email
        # content must never trigger tools, and a per-sender toolset knob would be a widening lever.
        if extra.get("untrusted_sender_toolsets"):
            logger.warning("[Email] untrusted_sender_toolsets is not supported — review_first email turns "
                           "always run with zero tools; ignoring the configured value.")
        # Resolver contract: when toolsets_for_source raises, fail closed to an empty toolset instead of
        # falling back to platform defaults.
        self.toolsets_override_fail_closed = self._outbound_policy == "review_first"
        # Anchor-less sends without gateway-internal provenance in the OUT-OF-PROCESS standalone path:
        # "draft" reviews them, "send" lets them out. Unknown values fail closed. In-process anchor-less
        # sends draft unconditionally (see _decide_disposition).
        raw_agent_sends = str(extra.get("agent_initiated_sends") or "draft").strip().lower()
        if raw_agent_sends not in ("draft", "send"):
            logger.error("[Email] Unknown agent_initiated_sends %r — failing closed to 'draft' (valid: draft, send)", raw_agent_sends)
            raw_agent_sends = "draft"
        self._agent_initiated_sends = raw_agent_sends
        # Abuse limits for untrusted draft generation: every accepted stranger costs a model turn and a
        # Drafts entry. Enforced at dispatch, before any model work.
        self._untrusted_draft_limit_per_sender_hour = _positive_int(extra.get("untrusted_draft_limit_per_sender_hour"), 4)
        self._untrusted_draft_limit_global_hour = _positive_int(extra.get("untrusted_draft_limit_global_hour"), 20)
        self._untrusted_draft_times_by_sender: Dict[str, List[float]] = {}
        self._untrusted_draft_times_global: List[float] = []
        # Per-message trust records keyed (normalized sender, Message-ID): binding trust to the specific
        # inbound message (not the sender) prevents a later message from the same address — e.g. a forged
        # one that failed DMARC — overwriting trust for an in-flight reply. A colliding forged Message-ID
        # can only overwrite toward authenticated=False (one reply drafts instead of sending).
        self._msg_trust: "OrderedDict[Tuple[str, str], bool]" = OrderedDict()
        self._msg_trust_max = 500
        # Inbound Message-IDs already dispatched: a replayed message arriving under a fresh IMAP UID must
        # not trigger a second model turn or a second draft.
        self._seen_message_ids: "OrderedDict[str, bool]" = OrderedDict()
        self._seen_message_ids_max = 1000
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None
        self._last_fetch_failed, self._last_fetch_error = False, ""  # "checked, nothing new" vs "the check itself failed"
        # chat_id (sender email) -> last subject + message-id for threading
        # Track the last IMAP fetch attempt so the poll loop can distinguish "checked, nothing new" from
        # "the check itself failed" (#80016).
        self._thread_context: Dict[str, Dict[str, str]] = {}
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
        sender_authenticated, auth_reason = _verify_sender_authentication(
            msg, sender_addr, authserv_id=self._authserv_id, strict=self._outbound_policy == "review_first")
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

    def _untrusted_draft_allowed(self, sender_addr: str) -> bool:
        """Hourly rate limits for untrusted draft generation (per-sender + global), enforced at
        dispatch before any model work. False (with a clear log) when a budget is exhausted."""
        now = time.time()
        cutoff = now - 3600.0
        sender_norm = _normalize_addr(sender_addr)
        global_times = [t for t in self._untrusted_draft_times_global if t > cutoff]
        sender_times = [t for t in self._untrusted_draft_times_by_sender.get(sender_norm, []) if t > cutoff]
        allowed = True
        if len(global_times) >= self._untrusted_draft_limit_global_hour:
            logger.warning("[Email] Draft generation skipped for %s: global hourly limit reached (%d/h)",
                           sender_addr, self._untrusted_draft_limit_global_hour)
            allowed = False
        elif len(sender_times) >= self._untrusted_draft_limit_per_sender_hour:
            logger.warning("[Email] Draft generation skipped for %s: per-sender hourly limit reached (%d/h)",
                           sender_addr, self._untrusted_draft_limit_per_sender_hour)
            allowed = False
        else:
            sender_times.append(now)
            global_times.append(now)
        self._untrusted_draft_times_global = global_times
        if sender_times:
            self._untrusted_draft_times_by_sender[sender_norm] = sender_times
        else:
            self._untrusted_draft_times_by_sender.pop(sender_norm, None)
        while len(self._untrusted_draft_times_by_sender) > 500:
            self._untrusted_draft_times_by_sender.pop(next(iter(self._untrusted_draft_times_by_sender)))
        return allowed

    async def _dispatch_message(self, msg_data: Dict[str, Any]) -> None:
        """Convert a fetched email into a MessageEvent and dispatch it."""
        sender_addr = msg_data["sender_addr"]
        if not self._sender_accepted(sender_addr, msg_data):
            return
        # Replay suppression: the same inbound Message-ID under a fresh IMAP UID (resent/copied mail)
        # must not trigger a second model turn or a second draft.
        if inbound_message_id := msg_data.get("message_id") or "":
            if inbound_message_id in self._seen_message_ids:
                logger.warning("[Email] Dropping duplicate inbound Message-ID %s from %s", inbound_message_id, sender_addr)
                return
            self._seen_message_ids[inbound_message_id] = True
            while len(self._seen_message_ids) > self._seen_message_ids_max:
                self._seen_message_ids.popitem(last=False)
        # Review-first trust verdict, decided once — derived ONLY from the verified
        # Authentication-Results evaluation plus the configured allowlist, never message content.
        review_first = self._outbound_policy == "review_first"
        sender_trusted = (not review_first) or (
            _normalize_addr(sender_addr) in self._auto_send_allowlist
            and bool(msg_data.get("sender_authenticated", False)))
        if review_first and not sender_trusted and not self._untrusted_draft_allowed(sender_addr):
            return
        subject, body, attachments = msg_data["subject"], msg_data["body"].strip(), msg_data["attachments"]
        if len(body) > MAX_MESSAGE_LENGTH:
            logger.warning("[Email] Truncating oversized inbound body from %s (%d chars)", sender_addr, len(body))
            body = body[:MAX_MESSAGE_LENGTH] + "\n\n[... truncated by gateway]"
        text = f"[Subject: {subject}]\n\n{body}" if subject and not subject.startswith("Re:") else body  # subject unless reply
        # DOCUMENT wins over PHOTO for mixed attachments: run.py keys image handling off the per-path mime type regardless
        # of message_type, but document-context injection gates strictly on MessageType.DOCUMENT — so DOCUMENT surfaces both.
        kinds = {att["type"] for att in attachments}
        self._thread_context[sender_addr] = {"subject": subject, "message_id": msg_data["message_id"]}
        # Bind the authentication verdict to this specific message so the reply's delivery decision can
        # never be influenced by a later message from the same address.
        self._record_message_trust(_normalize_addr(sender_addr), msg_data["message_id"],
                                   msg_data.get("sender_authenticated", False))
        name = msg_data["sender_name"] or sender_addr
        source = self.build_source(chat_id=sender_addr, chat_name=name, chat_type="dm", user_id=sender_addr, user_name=name)
        # Review-first trust boundary: both flags are wire-invisible and stamped from verified metadata
        # only. email_zero_tools makes the gateway resolve this turn to ZERO callable tools regardless of
        # sender trust (email content must never trigger tools) and blocks proxy-mode delegation. The
        # "untrusted" thread_id segregates the session key so a forged-From message can never share
        # conversation history (or the cached agent) with the real sender's trusted session.
        source.email_sender_trusted = sender_trusted
        source.email_zero_tools = review_first
        if review_first and not sender_trusted:
            source.thread_id = "untrusted"
        event = MessageEvent(
            text=text or "(empty email)", message_id=msg_data["message_id"],
            message_type=MessageType.DOCUMENT if "document" in kinds else MessageType.PHOTO if "image" in kinds else MessageType.TEXT,
            source=source,
            media_urls=[att["path"] for att in attachments], media_types=[att["media_type"] for att in attachments],
            reply_to_message_id=msg_data["in_reply_to"] or None)
        logger.info("[Email] New message from %s: %s", sender_addr, subject)
        await self.handle_message(event)

    async def _run_send(self, fn, args: tuple, log_fmt: str, *log_args) -> SendResult:
        """Run a blocking sender in the executor; senders return ``(message_id, disposition)``."""
        try:
            msg_id, disposition = await asyncio.get_running_loop().run_in_executor(None, fn, *args)
            return SendResult(success=True, message_id=msg_id, disposition=disposition)
        except Exception as e:
            logger.error(log_fmt, *log_args, e)
            return SendResult(success=False, error=str(e))

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an email reply to the given address (or draft it for review)."""
        if self._outbound_policy == "review_first" and (metadata or {}).get("_interim_send"):
            # Mid-turn status/heartbeat sends (marked by the gateway's _interim_metadata) are
            # dropped under review_first: email is not a streaming surface, their authority is
            # the inbound turn's, and neither an SMTP send nor a Drafts entry per heartbeat is
            # acceptable. The turn-final reply still passes the full trust decision.
            logger.debug("[Email] Suppressing mid-turn interim send to %s under review_first", chat_id)
            return SendResult(success=True)
        return await self._run_send(self._send_email, (chat_id, content, reply_to, metadata), "[Email] Send failed to %s: %s", chat_id)

    def _message_id_domain(self) -> str:
        """Domain for generated Message-IDs; ``localhost`` when EMAIL_ADDRESS lacks ``@``."""
        return (self._address.rsplit("@", 1)[-1] if "@" in self._address else "") or "localhost"

    def _new_reply(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None, *,
                   attach_empty_body: bool = False) -> Tuple[MIMEMultipart, str, str]:
        """Build a threaded reply skeleton. Returns ``(msg, msg_id, subject)``."""
        msg, ctx = MIMEMultipart(), self._thread_context.get(to_addr, {})
        subject = ctx.get("subject", "Hermes Agent")
        if not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        original_msg_id = reply_to_msg_id or ctx.get("message_id")
        threading = (("In-Reply-To", original_msg_id), ("References", original_msg_id)) if original_msg_id else ()
        msg_id = f"<hermes-{uuid.uuid4().hex[:12]}@{self._message_id_domain()}>"
        for key, value in (("From", self._address), ("To", to_addr), ("Subject", subject), *threading,
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

    def _imap_append(self, mailbox: str, flags: str, msg: MIMEMultipart) -> None:
        """Append one message to an IMAP mailbox. Raises on failure."""
        imap = self._connect_imap()
        try:
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            status, data = imap.append(mailbox, flags, imaplib.Time2Internaldate(time.time()), msg.as_bytes())
            if status != "OK":
                raise RuntimeError(f"IMAP append to {mailbox!r} failed: {status} {data}")
        finally:
            _close_imap(imap)

    def _record_message_trust(self, sender_norm: str, message_id: str, authenticated: bool) -> None:
        """Record the authentication verdict for one inbound message, keyed (sender, Message-ID) so
        trust is bound to the specific message a reply anchors to — never the sender's latest."""
        if not message_id:
            return
        self._msg_trust[(sender_norm, message_id)] = bool(authenticated)
        while len(self._msg_trust) > self._msg_trust_max:
            self._msg_trust.popitem(last=False)

    @staticmethod
    def _explicit_reply_anchor(reply_to_msg_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Inbound Message-ID this outbound replies to: the explicit per-turn ``reply_to``, else the
        ``reply_to_message_id`` the gateway's thread metadata carries for send paths without a
        ``reply_to`` parameter (attachment batches). Deliberately NO thread-context fallback: that
        per-sender state tracks the sender's NEWEST message, so a reply to one message could resolve
        to another's trust record — the delivery decision only ever binds to an explicit anchor."""
        return reply_to_msg_id or str((metadata or {}).get("reply_to_message_id") or "") or None

    def _decide_disposition(self, to_addr: str, reply_to_msg_id: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """Sent or drafted, for one outbound message. Legacy ``direct`` always sends. Under
        ``review_first``: gateway-internal sends (gateway-stamped metadata, never message content)
        send only to the auto-send allowlist; a reply sends only when its EXPLICIT anchor's inbound
        message carries a live authenticated trust record AND the recipient is allowlisted (a missing
        or ambiguous record — including after a restart — drafts; permission is never inferred from
        the recipient address alone); any anchor-less in-process send drafts unconditionally (the
        ``agent_initiated_sends`` toggle applies only to out-of-process standalone sends)."""
        if self._outbound_policy != "review_first":
            return DISPOSITION_SENT
        to_norm = _normalize_addr(to_addr)
        if (metadata or {}).get(GATEWAY_INTERNAL_SEND_KEY) is True:
            # Gateway-internal provenance authorizes transmission ONLY to the allowlist:
            # EMAIL_HOME_ADDRESS is a notification destination, never send authority.
            if to_norm and to_norm in self._auto_send_allowlist:
                return DISPOSITION_SENT
            logger.info("[Email] Gateway-internal send to non-allowlisted recipient %s drafts under review_first", to_addr)
            return DISPOSITION_DRAFTED
        if (anchor := self._explicit_reply_anchor(reply_to_msg_id, metadata)) is not None:
            if self._msg_trust.get((to_norm, anchor)) is True and to_norm in self._auto_send_allowlist:
                return DISPOSITION_SENT
            return DISPOSITION_DRAFTED
        return DISPOSITION_DRAFTED

    def _deliver_message(self, msg: MIMEMultipart, to_addr: str, *, subject: str,
                         reply_to_msg_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Authoritative delivery for every outbound email; returns the disposition. ``sent`` = SMTP
        plus best-effort Sent-mailbox archival (an archival failure logs and still reports sent so
        callers never retry SMTP and duplicate the message). ``drafted`` = IMAP APPEND to the drafts
        mailbox; a failed append raises — it must never fall back to SMTP."""
        disposition = self._decide_disposition(to_addr, reply_to_msg_id, metadata)
        if disposition == DISPOSITION_SENT:
            self._smtp_send(msg)
            if self._save_sent:
                try:
                    self._imap_append(self._sent_mailbox, r"(\Seen)", msg)
                except Exception as e:
                    logger.warning("[Email] Sent-mailbox archival to %r failed (SMTP delivery already "
                                   "succeeded, not retrying): %s", self._sent_mailbox, e)
            logger.info("[Email] Disposition sent: %s (subject: %s)", to_addr, subject)
        else:
            self._imap_append(self._drafts_mailbox, r"(\Draft)", msg)
            logger.info("[Email] Disposition drafted: proposed reply for %s stored in %r (subject: %s)",
                        to_addr, self._drafts_mailbox, subject)
        return disposition

    def _send_email(self, to_addr: str, body: str, reply_to_msg_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Build a plain-text reply and deliver it (executor thread). Returns ``(message_id, disposition)``."""
        reply_to_msg_id = self._explicit_reply_anchor(reply_to_msg_id, metadata)
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id, attach_empty_body=True)
        disposition = self._deliver_message(msg, to_addr, subject=subject, reply_to_msg_id=reply_to_msg_id, metadata=metadata)
        return msg_id, disposition

    def _send_with_files(self, to_addr: str, body: str, files: List[Tuple[Path, str]], *, lenient: bool,
                         reply_to_msg_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Build a reply with attachments and deliver it; *lenient* logs-and-skips unattachable files
        instead of raising. Returns ``(message_id, disposition)``."""
        reply_to_msg_id = self._explicit_reply_anchor(reply_to_msg_id, metadata)
        msg, msg_id, subject = self._new_reply(to_addr, body, reply_to_msg_id)
        for path, name in files:
            try:
                _attach_file(msg, path, name)
            except Exception as e:
                if not lenient:
                    raise
                logger.warning("[Email] Failed to attach %s: %s", path, e)
        disposition = self._deliver_message(msg, to_addr, subject=subject, reply_to_msg_id=reply_to_msg_id, metadata=metadata)
        return msg_id, disposition

    async def send_image(self, chat_id: str, image_url: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Send an image URL as part of an email body. ``metadata`` carries delivery-security context
        (gateway_internal_send) and must survive every send path."""
        return await self.send(chat_id, f"{caption or ''}\n\nImage: {image_url}".strip(), reply_to, metadata)

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
            await asyncio.get_running_loop().run_in_executor(
                None, self._send_email_with_attachments, chat_id, "\n\n".join(body_parts), local_paths, metadata)
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(self, to_addr: str, body: str, file_paths: List[str],
                                     metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Deliver an email with multiple attachments (unattachable files are skipped). No ``reply_to``
        parameter on this path — the anchor rides in ``metadata["reply_to_message_id"]`` (stamped by the
        gateway's thread metadata); without one the send is anchor-less and drafts, staying fail-closed."""
        msg_id, disposition = self._send_with_files(
            to_addr, body, [(Path(f), Path(f).name) for f in file_paths], lenient=True, metadata=metadata)
        logger.info("[Email] Multi-attachment email to %s (%d files): %s", to_addr, len(file_paths), disposition)
        return msg_id, disposition

    async def send_document(self, chat_id: str, file_path: str, caption: Optional[str] = None,
                            file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs) -> SendResult:
        """Send a file as an email attachment (or draft it for review)."""
        return await self._run_send(
            self._send_email_with_attachment, (chat_id, caption or "", file_path, file_name, reply_to, kwargs.get("metadata")),
            "[Email] Send document failed: %s")

    def _send_email_with_attachment(self, to_addr: str, body: str, file_path: str, file_name: Optional[str] = None,
                                    reply_to_msg_id: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Deliver an email with a single attachment (raises if unattachable). Returns ``(message_id, disposition)``."""
        return self._send_with_files(to_addr, body, [(Path(file_path), file_name or Path(file_path).name)],
                                     lenient=False, reply_to_msg_id=reply_to_msg_id, metadata=metadata)

    def toolsets_for_source(self, source) -> Optional[List[str]]:
        """Under ``review_first`` EVERY email-triggered turn — trusted, allowlisted senders included —
        resolves to an explicit empty toolset: zero callable tools, honored literally by the gateway
        (no platform-default or MCP fallback). Defense in depth behind the ``email_zero_tools`` source
        flag stamped at dispatch; ``toolsets_override_fail_closed`` tells the resolver that an
        exception here also means zero tools. ``direct`` keeps the platform's normal resolution."""
        return [] if self._outbound_policy == "review_first" else None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the email chat."""
        return {"name": chat_id, "type": "dm", "chat_id": chat_id, "subject": self._thread_context.get(chat_id, {}).get("subject", "")}


# Plugin glue: register() exposes the platform via the registry; EMAIL_* env → PlatformConfig seeding stays in core.
def _standalone_draft_append(extra: dict, address: str, password: str, msg, chat_id: str) -> Dict[str, Any]:
    """Append a standalone message to the drafts mailbox (review_first). Raises on IMAP failure."""
    imap_host = (extra.get("imap_host") or _get_secret("EMAIL_IMAP_HOST", "")).strip()
    if not imap_host:
        return {"error": "Email review_first requires IMAP to store drafts, but EMAIL_IMAP_HOST is not configured — refusing to send"}
    drafts_mailbox = str(extra.get("drafts_mailbox") or "Drafts").strip() or "Drafts"
    imap_port = _esecret_int("EMAIL_IMAP_PORT", 993)
    imap_security = _normalize_security(_get_secret("EMAIL_IMAP_SECURITY", "") or extra.get("imap_security", ""))
    ctx = _tls_context(_esecret_bool("EMAIL_IMAP_TLS_VERIFY", is_truthy_value(extra.get("imap_tls_verify"), default=True)), imap_host)
    if imap_security == "tls":
        imap = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=30, ssl_context=ctx)
    else:
        imap = imaplib.IMAP4(imap_host, imap_port, timeout=30)
        if imap_security == "starttls":
            try:
                imap.starttls(ssl_context=ctx)
            except Exception:
                _close_imap(imap)
                raise
    try:
        imap.login(address, password)
        _send_imap_id(imap)
        status, data = imap.append(drafts_mailbox, r"(\Draft)", imaplib.Time2Internaldate(time.time()), msg.as_bytes())
        if status != "OK":
            raise RuntimeError(f"IMAP append to {drafts_mailbox!r} failed: {status} {data}")
    finally:
        _close_imap(imap)
    logger.info("[Email] Disposition drafted (standalone): proposed message for %s stored in %r under review_first",
                chat_id, drafts_mailbox)
    return {"success": True, "platform": "email", "chat_id": chat_id, "disposition": DISPOSITION_DRAFTED,
            "note": f"review_first policy: message stored in {drafts_mailbox!r} for review, not transmitted"}


async def _standalone_send(pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False):
    """Out-of-process Email delivery (one-shot); standalone_sender_fn contract.

    Honors the review-first policy: a standalone send runs out-of-process with no gateway-authored
    provenance and no inbound reply anchor, so under ``review_first`` it transmits ONLY when the
    operator explicitly configured ``agent_initiated_sends: send``. Recipient membership in the
    allowlist or EMAIL_HOME_ADDRESS establishes nothing about the send's origin and grants no
    transmission authority here; everything else is drafted and never touches SMTP."""
    extra = getattr(pconfig, "extra", {}) or {}
    address, password = extra.get("address") or _get_secret("EMAIL_ADDRESS", ""), _get_secret("EMAIL_PASSWORD", "")
    smtp_host, smtp_port = extra.get("smtp_host") or _get_secret("EMAIL_SMTP_HOST", ""), _esecret_int("EMAIL_SMTP_PORT", 587)
    smtp_security = _normalize_security(_get_secret("EMAIL_SMTP_SECURITY", "") or extra.get("smtp_security"), default="tls" if smtp_port == 465 else "starttls")
    smtp_tls_verify = _esecret_bool("EMAIL_SMTP_TLS_VERIFY", is_truthy_value(extra.get("smtp_tls_verify"), default=True))
    if not all([address, password, smtp_host]):
        return {"error": "Email not configured (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_SMTP_HOST required)"}
    raw_policy = str(extra.get("outbound_policy") or "").strip().lower()
    review_first = raw_policy not in ("", "direct")  # unknown values fail closed to review_first
    agent_sends = str(extra.get("agent_initiated_sends") or "draft").strip().lower()
    try:
        msg = MIMEText(message, "plain", "utf-8")
        for key, value in (("From", address), ("To", chat_id), ("Subject", "Hermes Agent"), ("Date", formatdate(localtime=True))):
            msg[key] = value
        if review_first and agent_sends != "send":
            return _standalone_draft_append(extra, address, password, msg, chat_id)
        server = _open_smtp(smtp_host, smtp_port, smtp_security, _tls_context(smtp_tls_verify, smtp_host), smtplib.SMTP, smtplib.SMTP_SSL)
        server.login(address, password)
        server.send_message(msg)
        server.quit()
        return {"success": True, "platform": "email", "chat_id": chat_id, "disposition": DISPOSITION_SENT}
    except Exception as e:
        try:
            from tools.send_message_tool import _error as _e
            return _e(f"Email send failed: {e}")
        except Exception:
            return {"error": f"Email send failed: {e}"}


# Behavior keys bridged from a ``platforms.email:`` / ``email:`` YAML block into PlatformConfig.extra
# (config-only settings — deliberately no env mirrors; .env stays credentials-only).
_YAML_EXTRA_KEYS = (
    "skip_attachments", "require_authenticated_sender", "authserv_id",
    "outbound_policy", "auto_send_authenticated_senders", "drafts_mailbox", "sent_mailbox", "save_sent",
    "untrusted_sender_toolsets", "agent_initiated_sends",
    "untrusted_draft_limit_per_sender_hour", "untrusted_draft_limit_global_hour",
)


def _apply_yaml_config(yaml_cfg: dict, email_cfg: dict) -> "dict | None":
    """``apply_yaml_config_fn`` contract: seed behavior keys written at the top level of the email
    platform block (the documented shape) into ``PlatformConfig.extra``, where the adapter reads them.
    Keys nested under ``extra:`` already arrive without this bridge."""
    seeded = {k: email_cfg[k] for k in _YAML_EXTRA_KEYS if k in email_cfg}
    return seeded or None


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
        apply_yaml_config_fn=_apply_yaml_config,  # platforms.email.<behavior key> → PlatformConfig.extra
        max_message_length=50_000, pii_safe=True, emoji="📧", allow_update_command=True)
