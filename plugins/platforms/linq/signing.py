"""
Pure, dependency-free helpers for the Linq iMessage platform plugin.

This module deliberately imports **nothing** from the host Hermes package
(``gateway.*``) and nothing beyond the Python standard library.  Keeping the
webhook-signature verification, group-mention gating, and small parsing
helpers here means they can be unit-tested in isolation — ``adapter.py``
(which *does* import ``gateway.platforms.base``) only runs inside a real
Hermes runtime, but the security-critical bits below are testable anywhere.

Linq Blue v3 signs outbound webhooks as::

    signature = hex( HMAC_SHA256(secret, f"{timestamp}.{raw_body}") )

delivered in the ``X-Webhook-Signature`` header alongside an
``X-Webhook-Timestamp`` (unix seconds) header.  This mirrors the scheme used
by the OpenClaw Linq channel plugin so a single Linq webhook secret verifies
against both runtimes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from typing import Any, Optional

# Reject deliveries whose timestamp drifts more than this from now (replay
# protection).  Matches Photon/BlueBubbles' 5-minute window.
TIMESTAMP_DRIFT_SECONDS = 300

# Group-chat mention wake words.  When ``require_mention`` is enabled, group
# messages are ignored unless they match one of these — identical defaults to
# the Photon and BlueBubbles iMessage adapters so all three iMessage channels
# gate group chats the same way.
DEFAULT_MENTION_PATTERNS = [
    r"(?<![\w@])@?hermes\s+agent\b[,:\-]?",
    r"(?<![\w@])@?hermes\b[,:\-]?",
]


def coerce_port(value: Any, default: int) -> int:
    """Best-effort int coercion for port-like env/config values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse the truthy strings Hermes accepts in env/config."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def verify_signature(
    *,
    body: bytes,
    timestamp_header: str,
    signature_header: str,
    signing_secret: str,
    now: Optional[float] = None,
    drift: int = TIMESTAMP_DRIFT_SECONDS,
) -> bool:
    """Constant-time verify a Linq Blue v3 webhook signature.

    Returns ``True`` iff the timestamp is within ``drift`` seconds of *now*
    AND ``signature_header == hex(hmac_sha256(secret, f"{ts}.{body}"))``.

    The signature header may optionally carry a ``sha256=`` prefix; both
    shapes are accepted.  Exposed at module scope so tests can exercise it
    without an adapter instance.
    """
    if not timestamp_header or not signature_header or not signing_secret:
        return False
    try:
        ts = int(timestamp_header)
    except (TypeError, ValueError):
        return False
    if abs((now if now is not None else time.time()) - ts) > drift:
        return False
    provided = signature_header[7:] if signature_header.startswith("sha256=") else signature_header
    message = f"{ts}.".encode("utf-8") + body
    expected = hmac.new(signing_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, provided)
    except Exception:
        return False


def compile_mention_patterns(raw: Any) -> "list[re.Pattern]":
    """Compile group-mention wake words from config/env.

    ``raw`` may be a list (config or env JSON), a string (env var: a JSON
    list, or comma/newline-separated), or ``None`` (use Hermes defaults).
    Mirrors the Photon/BlueBubbles implementations so every iMessage channel
    accepts the same configuration shapes.
    """
    if raw is None:
        patterns: list = list(DEFAULT_MENTION_PATTERNS)
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            loaded = json.loads(text) if text else []
        except Exception:
            loaded = None
        if isinstance(loaded, list):
            patterns = loaded
        else:
            patterns = [
                part.strip()
                for line in text.splitlines()
                for part in line.split(",")
            ]
    elif isinstance(raw, list):
        patterns = raw
    else:
        patterns = [raw]

    compiled: "list[re.Pattern]" = []
    for pattern in patterns:
        token = str(pattern).strip()
        if not token:
            continue
        try:
            compiled.append(re.compile(token, re.IGNORECASE))
        except re.error:
            # Skip invalid patterns rather than crashing the adapter; the
            # adapter logs these at warning level when it compiles them.
            continue
    return compiled


def message_matches_mention(text: str, patterns: "list[re.Pattern]") -> bool:
    if not text or not patterns:
        return False
    return any(p.search(text) for p in patterns)


def clean_mention_text(text: str, patterns: "list[re.Pattern]") -> str:
    """Strip a single leading wake word before dispatch.

    Custom mention patterns are regexes, so we only strip a *leading* match
    to avoid deleting ordinary words later in the prompt.
    """
    if not text:
        return text
    stripped = text.lstrip()
    for pattern in patterns:
        match = pattern.match(stripped)
        if match:
            cleaned = stripped[match.end():].lstrip(" ,:-")
            return cleaned or text
    return text


def extract_text(parts: list) -> str:
    """Join the ``value`` of every ``{"type": "text"}`` part."""
    out = []
    for part in parts or []:
        if isinstance(part, dict) and part.get("type") == "text":
            value = part.get("value")
            if value:
                out.append(str(value))
    return "\n".join(out)


def extract_media(parts: list) -> "list[dict]":
    """Return ``[{"url": ..., "mime_type": ...}]`` for every media part with a URL."""
    out = []
    for part in parts or []:
        if (
            isinstance(part, dict)
            and part.get("type") == "media"
            and part.get("url")
        ):
            out.append(
                {
                    "url": str(part["url"]),
                    "mime_type": str(part.get("mime_type") or ""),
                    "filename": part.get("filename"),
                }
            )
    return out


def is_group_chat(data: dict) -> bool:
    """Heuristically classify an inbound Linq message as group vs. direct.

    Linq's Blue v3 ``message.received`` payload does not (in the documented
    shape we mirror from the OpenClaw channel) carry an explicit chat-type
    discriminator, so we look at the fields a group delivery is known to add:
    an ``is_group`` flag, a ``group_id``/``group_name``, or a ``participants``
    list with more than two members.  Everything else is treated as a DM.

    NOTE: confirm against a live Linq group webhook and tighten if the real
    payload exposes a first-class type field.
    """
    if coerce_bool(data.get("is_group")):
        return True
    if data.get("group_id") or data.get("group_name"):
        return True
    participants = data.get("participants")
    if isinstance(participants, list) and len(participants) > 2:
        return True
    chat_type = str(data.get("chat_type") or "").lower()
    return chat_type in {"group", "groupchat", "group_chat"}
