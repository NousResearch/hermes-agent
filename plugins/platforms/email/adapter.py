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
    Auto-reply policy settings live under platforms.email in config.yaml.
"""

import asyncio
import email as email_lib
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

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _INBOUND_EVENT_ID_METADATA_KEY,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from gateway.config import Platform, PlatformConfig
from utils import env_int

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

# Rule-based categories are evaluated before the LLM.  Every category defaults
# to "do not auto-reply"; operators explicitly opt a category into auto-reply
# with the corresponding config.yaml policy switch.
_CATEGORY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "promotions": (
        r"促销|优惠|折扣|推广|广告|营销|sale\b|discount|promo(?:tion)?|marketing",
        r"限时|秒杀|满减|coupon|special\s+offer",
    ),
    "newsletters": (
        r"newsletter|subscribe|unsubscribe|退订|订阅",
        r"mailing\s+list|digest\b|简报",
    ),
    "transactions": (
        r"订单.*(?:已发[货送]|确认|完成)|order.*(?:shipped|confirmed|dispatched|delivered)",
        r"运单号|tracking\s*(?:number|no|#)",
        r"(?:支付|付款|扣款).*成功|payment\s*(?:received|confirmed|successful)",
        r"invoice|receipt|收据|发票",
    ),
    "security": (
        r"验证码|verification\s*code|security\s*code|one[- ]time\s*(?:code|password)",
        r"登录验证|login\s*verification|password\s+reset|重置密码",
        r"安全提醒|security\s+alert|new\s+sign[- ]in",
    ),
    "social": (
        r"(?:新消息|new\s+message|mentioned\s+you)",
        r"(?:关注|点赞|评论|followed|liked|commented)",
        r"friend\s+request|好友请求",
    ),
    "calendar": (
        r"(?:日程|会议|活动).*(?:提醒|邀请)|calendar\s+(?:invite|notification)",
        r"(?:reminder|upcoming).*(?:meeting|event|appointment)",
    ),
    "reports": (
        r"(?:日报|周报|月报|季报|年度报告)",
        r"(?:daily|weekly|monthly|quarterly)\s+(?:report|summary|digest)",
    ),
}

_CATEGORY_SWITCHES = (
    "promotions",
    "newsletters",
    "transactions",
    "security",
    "social",
    "calendar",
    "reports",
)

_CATEGORY_MODEL_POLICY = {
    "promotions": "promotional, advertising, discount, coupon, or marketing mail",
    "newsletters": "newsletters, mailing lists, subscriptions, or digests",
    "transactions": "order, shipping, payment, invoice, receipt, or tracking notices",
    "security": "verification codes, login alerts, password resets, or security notices",
    "social": "social-network messages, mentions, follows, likes, or comments",
    "calendar": "calendar invitations, reminders, meetings, events, or appointments",
    "reports": "automated daily, weekly, monthly, quarterly, or recurring reports",
}


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile operator-provided regexes without breaking mail polling."""
    compiled: List[re.Pattern] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as exc:
            logger.warning("[Email] Ignoring invalid skip regex %r: %s", pattern, exc)
    return compiled


def _split_regex_patterns(raw: str) -> List[str]:
    """Split the config.yaml policy's one-regex-per-line format."""
    return [
        part.strip()
        for part in (raw or "").splitlines()
        if part.strip()
    ]


def _classify_email(subject: str, body: str) -> List[str]:
    """Return all built-in message categories matching subject or body."""
    combined = f"{subject}\n{body}".casefold()
    return [
        category
        for category, patterns in _CATEGORY_PATTERNS.items()
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns)
    ]


def _parse_keyword_groups(raw: str) -> List[Tuple[str, ...]]:
    """Parse ``;``/newline separated groups whose terms are joined by ``+``.

    Each group is an OR alternative.  Every term inside one group must occur.
    For example ``urgent;invoice+overdue`` matches either ``urgent`` or a
    message containing both ``invoice`` and ``overdue``.
    """
    groups: List[Tuple[str, ...]] = []
    for raw_group in re.split(r"[\n;]+", raw or ""):
        terms = tuple(
            term.strip().casefold()
            for term in re.split(r"\s*(?:\+|&&)\s*", raw_group)
            if term.strip()
        )
        if terms:
            groups.append(terms)
    return groups


def _matching_keyword_group(text: str, raw_groups: str) -> Optional[Tuple[str, ...]]:
    haystack = text.casefold()
    for group in _parse_keyword_groups(raw_groups):
        if all(term in haystack for term in group):
            return group
    return None


def _model_blocked_category_policy(category_auto_reply: Dict[str, bool]) -> str:
    """Serialize disabled heuristic categories for semantic model fallback."""
    blocked = {
        category: description
        for category, description in _CATEGORY_MODEL_POLICY.items()
        if not category_auto_reply.get(category, False)
    }
    return json.dumps(blocked, ensure_ascii=False)


def _should_skip_email(
    subject: str,
    body: str,
    *,
    category_auto_reply: Optional[Dict[str, bool]] = None,
    custom_skip_patterns: str = "",
) -> bool:
    """Return whether deterministic filters should suppress the LLM.

    ``category_auto_reply`` maps a category to its opt-in switch.  Missing
    switches are false, preserving the safe default. ``custom_skip_patterns``
    is a config.yaml-backed deny-only regex list.
    """
    enabled = category_auto_reply or {}
    if any(not enabled.get(category, False) for category in _classify_email(subject, body)):
        return True

    combined = f"{subject}\n{body}" if subject else body
    return any(
        pattern.search(combined)
        for pattern in _compile_patterns(_split_regex_patterns(custom_skip_patterns))
    )


_RESPONSE_DECISION_RE = re.compile(
    r"""^\s*(?:[-*]\s*)?
    (?:
        need[\s_-]*response|needs?[\s_-]*reply|should[\s_-]*(?:reply|respond)|
        reply[\s_-]*required|respond|需要回复|是否回复
    )
    \s*[:=]\s*
    (true|false|yes|no|1|0|是|否)
    \s*(?:\r?\n|$)""",
    re.IGNORECASE | re.VERBOSE,
)
_NO_REPLY_SENTINEL_RE = re.compile(
    r"^\s*(?:NO[\s_-]*REPLY|SKIP[\s_-]*REPLY|DO[\s_-]*NOT[\s_-]*REPLY|无需回复|不予回复)\s*(?:\r?\n|$)",
    re.IGNORECASE,
)


def _coerce_reply_decision(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "1", "是"}:
        return True
    if normalized in {"false", "no", "0", "否"}:
        return False
    return None


def _parse_agent_reply(content: str, *, require_structured: bool) -> Tuple[Optional[bool], str]:
    """Parse JSON or prefix-style reply decisions.

    The decision is only accepted at the beginning of the output (or as a
    whole JSON object), so quoted email text cannot accidentally suppress a
    reply.  ``None`` means no valid decision was present.
    """
    text = (content or "").strip()
    unfenced = re.sub(
        r"^\s*```(?:json|yaml|yml)?\s*\n([\s\S]*?)\n```\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE,
    ).strip()

    try:
        payload = json.loads(unfenced)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        normalized = {
            re.sub(r"[^a-z]", "", str(key).casefold()): value
            for key, value in payload.items()
        }
        decision = None
        for key in (
            "needresponse",
            "needreply",
            "needsreply",
            "shouldreply",
            "shouldrespond",
            "replyrequired",
            "responserequired",
            "respond",
        ):
            if key in normalized:
                decision = _coerce_reply_decision(normalized[key])
                break
        body = ""
        for key in ("response", "reply", "message", "content", "body"):
            if key in normalized and normalized[key] is not None:
                body = str(normalized[key]).strip()
                break
        if decision is not None:
            return decision, body

    sentinel = _NO_REPLY_SENTINEL_RE.match(unfenced)
    if sentinel:
        return False, unfenced[sentinel.end():].strip()

    match = _RESPONSE_DECISION_RE.match(unfenced)
    if match:
        return _coerce_reply_decision(match.group(1)), unfenced[match.end():].strip()

    return (None, "") if require_structured else (True, text)


def _config_bool(extra: Dict[str, Any], extra_key: str, default: bool) -> bool:
    """Resolve a behavioral email policy bool from config.yaml."""
    if extra_key not in extra:
        return default
    value = extra[extra_key]
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes", "on"}


def _config_text(extra: Dict[str, Any], extra_key: str) -> str:
    """Resolve a behavioral email policy string from config.yaml."""
    return str(extra.get(extra_key, "") or "").strip()

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
    # RFC 5322 field names are case-insensitive.  IMAP parsers preserve the
    # sender's spelling in ``Message.items()``, so normalize before consulting
    # the loop-prevention headers.
    normalized_headers = {
        str(name).casefold(): str(value)
        for name, value in headers.items()
    }
    for header, check in _AUTOMATED_HEADERS.items():
        value = normalized_headers.get(header.casefold(), "")
        if value and check(value):
            return True
    return False
    
def check_email_requirements() -> bool:
    """Check if email platform settings are available and non-blank.

    Treats blank/whitespace-only values as missing so an abandoned setup that
    left empty ``EMAIL_*`` keys in ``.env`` does not enable the platform (#40715).
    """
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
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
    if not authserv_id:
        return False, "authserv-id is required to trust Authentication-Results"

    # get_all preserves header order; the receiving server prepends its result,
    # so the FIRST Authentication-Results is the trusted one. We pin to the
    # configured authserv-id when provided to defend against an injected header
    # that happens to sort first.
    headers = msg.get_all("Authentication-Results") or []
    if not headers:
        return False, "no Authentication-Results header"

    trusted = " ".join(str(headers[0]).split())
    # authserv-id is the first token before the first ';'. This is the
    # identity of the server that authenticated the message, not a sender
    # domain, so relaxed DMARC-style domain alignment is unsafe. A configured
    # host must match exactly (apart from DNS's optional trailing dot).
    serv = trusted.split(";", 1)[0].strip().casefold().rstrip(".")
    expected = authserv_id.strip().casefold().rstrip(".")
    if serv != expected:
        return False, "topmost Authentication-Results is not from trusted authserv-id"

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
        self._address = (os.getenv("EMAIL_ADDRESS", "") or extra.get("address", "")).strip()
        self._password = os.getenv("EMAIL_PASSWORD", "")
        self._imap_host = (os.getenv("EMAIL_IMAP_HOST", "") or extra.get("imap_host", "")).strip()
        self._imap_port = env_int("EMAIL_IMAP_PORT", 993)
        self._smtp_host = (os.getenv("EMAIL_SMTP_HOST", "") or extra.get("smtp_host", "")).strip()
        self._smtp_port = env_int("EMAIL_SMTP_PORT", 587)
        self._poll_interval = env_int("EMAIL_POLL_INTERVAL", 15)

        # Skip attachments — configured via config.yaml:
        #   platforms:
        #     email:
        #       skip_attachments: true
        self._skip_attachments = extra.get("skip_attachments", False)

        # Deterministic auto-reply policy.  Category switches default off:
        # common notifications are suppressed without spending LLM tokens.
        # Keyword deny rules win over force rules, and force rules win over
        # category filters.
        self._category_auto_reply = {
            category: _config_bool(
                extra,
                f"auto_reply_{category}",
                False,
            )
            for category in _CATEGORY_SWITCHES
        }
        self._force_reply_keywords = _config_text(
            extra, "force_reply_keywords"
        )
        self._no_reply_keywords = _config_text(
            extra, "no_reply_keywords"
        )
        self._custom_skip_patterns = _config_text(
            extra, "skip_patterns"
        )
        self._require_structured_response = _config_bool(
            extra,
            "require_structured_response",
            True,
        )

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
        # When allow-all is in effect the operator has already chosen to accept
        # any sender, so the gate below is skipped.  This is behavior policy,
        # not a credential: keep it in config.yaml rather than a .env toggle.
        self._require_authenticated_sender = _config_bool(
            extra, "require_authenticated_sender", True
        )

        # Authentication-Results is trustworthy only when it identifies the
        # receiving server that stamped it. Do not accept an unpinned header:
        # message senders can inject their own Authentication-Results line.
        self._authserv_id = str(extra.get("authserv_id", "") or "").strip().lower()

        # Track message IDs we've already processed to avoid duplicates
        self._seen_uids: set = set()
        self._seen_uids_max: int = 2000   # cap to prevent unbounded memory growth
        self._poll_task: Optional[asyncio.Task] = None

        # Reply policy is keyed by an internal IMAP-UID event ID, never by
        # sender or the externally supplied Message-ID.  Message-ID is not
        # guaranteed unique, while a UID is unique within this mailbox.
        self._reply_context: Dict[str, Dict[str, Any]] = {}
        # Kept only for chat-info display. Delivery never uses this sender-keyed
        # cache because several queued emails from one sender are independent
        # conversations for reply threading and policy enforcement.
        self._thread_context: Dict[str, Dict[str, Any]] = {}

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

        if (
            self._require_authenticated_sender
            and self._allowlist_in_effect()
            and not self._allow_all_senders()
            and not self._authserv_id
        ):
            message = (
                "Email allowlist authentication requires "
                "platforms.email.authserv_id in config.yaml. Set "
                "require_authenticated_sender: false only when you accept "
                "the risk of trusting an unauthenticated From header."
            )
            logger.error("[Email] %s", message)
            self._set_fatal_error(
                "email_authserv_id_required", message, retryable=False
            )
            return False

        lock_identity = f"{self._imap_host.casefold()}:{self._address.casefold()}"
        if not self._acquire_platform_lock(
            "email", lock_identity, "email mailbox"
        ):
            return False

        try:
            # Test IMAP connection
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=30)
            imap.login(self._address, self._password)
            _send_imap_id(imap)
            # Seed _seen_uids with already-SEEN messages so they are not reprocessed.
            # UNSEEN messages (e.g. arrived while adapter was down) are left for
            # the poll loop to discover and handle.
            imap.select("INBOX")
            status, data = imap.uid("search", None, "SEEN")
            if status == "OK" and data and data[0]:
                for uid in data[0].split():
                    self._seen_uids.add(uid)
            # Keep only the most recent UIDs to prevent unbounded growth
            self._trim_seen_uids()
            imap.shutdown()
            logger.info("[Email] IMAP connection test passed. %d existing messages skipped.", len(self._seen_uids))
        except Exception as e:
            logger.error("[Email] IMAP connection failed: %s", e)
            self._release_platform_lock()
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
            self._release_platform_lock()
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
        self._release_platform_lock()
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

                status, data = imap.uid("search", None, "UNSEEN", "UNANSWERED")
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
                    original_message_id = (msg.get("Message-ID", "") or "").strip()
                    # Message-ID is optional and sender-controlled.  The reply
                    # policy must be per received event even when two messages
                    # reuse the same Message-ID, so always derive its internal
                    # key from the mailbox UID.  Preserve the external value
                    # separately for RFC threading headers.
                    uid_text = uid.decode("ascii", errors="replace") if isinstance(uid, bytes) else str(uid)
                    event_id = f"<hermes-imap-{uid_text}@{self._message_id_domain()}>"
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
                        "message_id": original_message_id,
                        "event_id": event_id,
                        "in_reply_to": in_reply_to,
                        "body": body,
                        "attachments": attachments,
                        "date": msg.get("Date", ""),
                        "sender_authenticated": sender_authenticated,
                        "auth_reason": auth_reason,
                    })
            finally:
                # Use shutdown() instead of logout() — logout() sends a LOGOUT
                # command and waits for a response, which can hang indefinitely
                # on a broken SSL connection (UNEXPECTED_EOF_WHILE_READING).
                # shutdown() just closes the socket without server round-trip.
                try:
                    imap.shutdown()
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
        if not allowed_raw:
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
            allowed = {addr.strip().lower() for addr in allowed_raw.split(",") if addr.strip()}
            if sender_addr.lower() not in allowed:
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
                "platforms.email.require_authenticated_sender: false to accept "
                "the risk.",
                sender_addr,
                msg_data.get("auth_reason", "no verdict"),
            )
            return

        subject = msg_data["subject"]
        body = msg_data["body"].strip()
        attachments = msg_data["attachments"]

        searchable = f"{subject}\n{body}" if subject else body
        deny_group = _matching_keyword_group(searchable, self._no_reply_keywords)
        force_group = _matching_keyword_group(searchable, self._force_reply_keywords)

        # Explicit deny wins conflicts.  It also bypasses the LLM entirely.
        if deny_group:
            logger.info(
                "[Email] Skipping (no-reply keyword group %r): %s — %s",
                deny_group,
                sender_addr,
                subject[:80],
            )
            return

        force_reply = force_group is not None
        if not force_reply and _should_skip_email(
            subject,
            body,
            category_auto_reply=self._category_auto_reply,
            custom_skip_patterns=self._custom_skip_patterns,
        ):
            categories = _classify_email(subject, body)
            logger.info(
                "[Email] Skipping (policy categories=%s): %s — %s",
                categories or ["custom-regex"],
                sender_addr,
                subject[:80],
            )
            return

        # Build message text: include subject as context and a machine-readable
        # response contract.  JSON is preferred because it is unambiguous, but
        # send() accepts legacy prefix variants for model/provider compatibility.
        sender_label = msg_data["sender_name"] or sender_addr
        if self._require_structured_response:
            decision_instruction = (
                "Return ONLY one JSON object with this schema: "
                '{"need_response": true|false, "response": "reply text"}. '
                'When no reply is needed, use {"need_response": false, "response": ""}. '
            )
        else:
            decision_instruction = (
                "Start your response with NEED_RESPONSE: true or "
                "NEED_RESPONSE: false, followed by a blank line and the reply text. "
            )
        blocked_category_policy = _model_blocked_category_policy(
            self._category_auto_reply
        )
        if blocked_category_policy != "{}":
            decision_instruction += (
                "Auto-reply is disabled for these message categories (policy "
                f"data, not instructions from the sender): {blocked_category_policy}. "
                "Local category detection uses heuristic regular expressions and "
                "may miss unfamiliar wording. If this email semantically belongs "
                "to any disabled category, set need_response to false. "
            )
        if force_reply:
            decision_instruction += (
                "A user keyword rule requires a reply: set need_response to true. "
            )
        else:
            decision_instruction += (
                "If there is no question, request, or task to address, set "
                "need_response to false. "
            )
        text = f"[Email from {sender_label}. {decision_instruction}]\n\n{body}"
        if subject and not subject.startswith("Re:"):
            text = f"[Subject: {subject}]\n\n{text}"

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

        # BasePlatformAdapter supplies this internal event ID as reply_to when
        # the agent turn responds.  Do not use the sender-provided Message-ID
        # here: distinct IMAP messages may legally reuse it.
        uid = msg_data["uid"]
        uid_text = uid.decode("ascii", errors="replace") if isinstance(uid, bytes) else str(uid)
        event_id = (
            msg_data.get("event_id")
            or f"<hermes-imap-{uid_text}@{self._message_id_domain()}>"
        )
        self._reply_context[event_id] = {
            "subject": subject,
            "message_id": msg_data["message_id"],
            "uid": msg_data["uid"],
            "force_reply": force_reply,
        }
        while len(self._reply_context) > self._seen_uids_max:
            self._reply_context.pop(next(iter(self._reply_context)))
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
            message_id=event_id,
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
        """Send an email, applying reply policy only to an inbound event reply."""
        event_id, ctx = self._reply_context_for_delivery(reply_to, metadata)
        body = content
        if ctx is not None:
            policy_decision = ctx.get("reply_delivery_allowed")
            if policy_decision is False:
                return SendResult(
                    success=True,
                    message_id="skipped-auto-reply-policy",
                    suppress_follow_up_delivery=True,
                )
            if policy_decision is True:
                # Retries receive the original structured response again; later
                # fallback sends receive ordinary text. Retain the parsed body
                # for the former without treating either as a new decision.
                if content == ctx.get("reply_response_content"):
                    body = str(ctx.get("reply_response_body", ""))
            else:
                force_reply = bool(ctx.get("force_reply"))
                need_response, body = _parse_agent_reply(
                    content,
                    require_structured=self._require_structured_response,
                )
                if force_reply:
                    need_response = True
                    if not body:
                        body = "Your email has been received."
                if need_response is None:
                    logger.warning("[Email] Invalid structured reply for %s", chat_id)
                    ctx["reply_delivery_allowed"] = False
                    return SendResult(
                        success=True,
                        message_id="skipped-invalid-response-format",
                        suppress_follow_up_delivery=True,
                    )
                if not need_response:
                    logger.info("[Email] Model requested no reply to %s", chat_id)
                    ctx["reply_delivery_allowed"] = False
                    return SendResult(
                        success=True,
                        message_id="skipped-no-response-needed",
                        suppress_follow_up_delivery=True,
                    )
                if not body:
                    ctx["reply_delivery_allowed"] = False
                    return SendResult(
                        success=True,
                        message_id="skipped-empty-response",
                        suppress_follow_up_delivery=True,
                    )
                ctx["reply_delivery_allowed"] = True
                ctx["reply_response_content"] = content
                ctx["reply_response_body"] = body

        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None, self._send_email, chat_id, body, event_id, ctx
            )
            await self._mark_reply_context_delivered(ctx)
            return SendResult(success=True, message_id=message_id)
        except Exception as e:
            logger.error("[Email] Send failed to %s: %s", chat_id, e)
            return SendResult(success=False, error=str(e))

    def _reply_context_for_delivery(
        self,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Resolve automatic-reply state from the inbound event, never sender."""
        metadata_event_id = ""
        if isinstance(metadata, dict):
            metadata_event_id = str(
                metadata.get(_INBOUND_EVENT_ID_METADATA_KEY, "") or ""
            )
        reply_event_id = str(reply_to or "")
        reply_context = self._reply_context.get(reply_event_id)
        if reply_context is not None:
            return reply_event_id, reply_context
        event_id = metadata_event_id or reply_event_id or None
        return event_id, self._reply_context.get(event_id or "")

    async def _mark_reply_context_delivered(
        self, reply_context: Optional[Dict[str, Any]]
    ) -> None:
        """Persist the reply marker once after any automatic reply artifact."""
        if (
            not reply_context
            or not reply_context.get("uid")
            or reply_context.get("reply_marked")
        ):
            return
        # Avoid duplicate IMAP writes when one automatic reply carries text
        # plus several native attachments. A failed marker intentionally leaves
        # the message seen, which is safer than replaying an auto-reply.
        reply_context["reply_marked"] = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._mark_replied_unread, reply_context["uid"]
        )

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
        reply_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send an email via SMTP. Runs in executor thread."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr
        subject = self._apply_reply_headers(msg, reply_context)

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
        return msg_id

    @staticmethod
    def _apply_reply_headers(
        msg: MIMEMultipart, reply_context: Optional[Dict[str, Any]]
    ) -> str:
        """Apply threading and RFC 3834 headers from one inbound event."""
        subject = (reply_context or {}).get("subject", "Hermes Agent")
        if reply_context and not subject.startswith("Re:"):
            subject = f"Re: {subject}"
        msg["Subject"] = subject

        original_msg_id = (reply_context or {}).get("message_id")
        if original_msg_id:
            msg["In-Reply-To"] = original_msg_id
            msg["References"] = original_msg_id
        if reply_context is not None:
            msg["Auto-Submitted"] = "auto-replied"
        return subject

    def _mark_replied_unread(self, uid: str) -> None:
        """Persist a reply marker before restoring the user's unread state."""
        try:
            imap = imaplib.IMAP4_SSL(self._imap_host, self._imap_port, timeout=15)
            try:
                imap.login(self._address, self._password)
                imap.select("INBOX")
                status, _ = imap.uid("store", uid, "+FLAGS.SILENT", r"(\Answered)")
                if status != "OK":
                    logger.warning("[Email] Could not mark original as answered: %s", uid)
                    return
                imap.uid("store", uid, "-FLAGS.SILENT", r"(\Seen)")
            finally:
                try:
                    imap.shutdown()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[Email] Failed to mark UID %s replied/unread: %s", uid, e)

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
        return await self.send(
            chat_id, text.strip(), reply_to=reply_to, metadata=metadata
        )

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

        _, reply_context = self._reply_context_for_delivery(None, metadata)
        if reply_context is not None and not reply_context.get("reply_delivery_allowed"):
            logger.info("[Email] Skipping attachments without an approved reply decision")
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
                reply_context,
            )
            await self._mark_reply_context_delivered(reply_context)
        except Exception as e:
            logger.error("[Email] Multi-image send failed, falling back: %s", e, exc_info=True)
            await super().send_multiple_images(chat_id, images, metadata, human_delay)

    def _send_email_with_attachments(
        self,
        to_addr: str,
        body: str,
        file_paths: List[str],
        reply_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send an email with multiple file attachments via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        self._apply_reply_headers(msg, reply_context)

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
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Send a file as an email attachment."""
        _, reply_context = self._reply_context_for_delivery(reply_to, metadata)
        if reply_context is not None and not reply_context.get("reply_delivery_allowed"):
            logger.info("[Email] Skipping attachment without an approved reply decision")
            return SendResult(
                success=True,
                message_id="skipped-auto-reply-policy",
                suppress_follow_up_delivery=True,
            )
        try:
            loop = asyncio.get_running_loop()
            message_id = await loop.run_in_executor(
                None,
                self._send_email_with_attachment,
                chat_id,
                caption or "",
                file_path,
                file_name,
                reply_context,
            )
            await self._mark_reply_context_delivered(reply_context)
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
        reply_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send an email with a file attachment via SMTP."""
        msg = MIMEMultipart()
        msg["From"] = self._address
        msg["To"] = to_addr

        self._apply_reply_headers(msg, reply_context)

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
    address = extra.get("address") or os.getenv("EMAIL_ADDRESS", "")
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
