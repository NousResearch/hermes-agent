"""``hermes debug`` debug tools for Hermes Agent.

Currently supports:
    hermes debug share    Upload debug report (system info + logs) to a
                          paste service and print a shareable URL.
                          By default, log content is run through
                          ``agent.redact.redact_sensitive_text`` with
                          ``force=True`` before upload so credentials in
                          ``~/.hermes/logs/*.log`` are not leaked into
                          the public paste service. Pass ``--no-redact``
                          to disable.
                          Pass ``--nous`` to upload instead to Nous-internal
                          storage (AWS S3) via a signed URL minted by the
                          Nous account service: the bundle is private
                          (viewable only by Nous staff / allowlisted mods via
                          a Google-login-gated viewer) and auto-deletes after
                          14 days, rather than going to a public paste.
"""

import datetime
import gzip
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Banner prepended to upload-bound log content when redaction is enabled.
# Visible in the public paste so reviewers know the content was sanitized.
# Kept short; the trailing newline guarantees the banner sits on its own line.
_REDACTION_BANNER = (
    "[hermes debug share: log content redacted at upload time. "
    "run with --no-redact to disable]\n"
)

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)

# Historical gateway logs included WhatsApp Cloud identities and message
# previews on the inbound-routing line.  Bare digit strings on those lines are
# structurally phone data; elsewhere they may be timestamps or diagnostic IDs
# and must retain the global redactor's default pass-through behavior.
_WHATSAPP_INBOUND_LOG_RE = re.compile(
    r"\binbound message:\s.*\bplatform=whatsapp(?:_cloud)?\b",
    re.IGNORECASE,
)
_WHATSAPP_CONVERSATION_LOG_RE = re.compile(
    r"\bconversation turn:\s.*\bplatform=whatsapp(?:_cloud)?\b",
    re.IGNORECASE,
)
_WHATSAPP_SESSION_KEY_LOG_RE = re.compile(
    r"\bagent:[^:\s]+:whatsapp(?:_cloud)?:",
    re.IGNORECASE,
)
_WHATSAPP_GENERIC_IDENTITY_LOG_RE = re.compile(
    r"\b(?:"
    r"Unauthorized user:.*\bon whatsapp(?:_cloud)?\b|"
    r"pre_gateway_dispatch skip:.*\bplatform=whatsapp(?:_cloud)?\b|"
    r"(?:Sent|Failed to send) shutdown notification (?:to active chat |to home channel )?"
    r"whatsapp(?:_cloud)?:|"
    r"Sent post-update notification to whatsapp(?:_cloud)?:|"
    r"(?:Sent restart notification to|Restart notification to) whatsapp(?:_cloud)?:|"
    r"(?:Sent home-channel startup notification to|"
    r"Home-channel startup notification failed for) whatsapp(?:_cloud)?:|"
    r"No profile route matched:\s*platform=whatsapp(?:_cloud)?\b|"
    r"Profile route matching failed for (?:Platform\.)?whatsapp(?:_cloud)?/|"
    r"(?:Profile .* does not exist for|Failed to resolve profile directory for) "
    r"source whatsapp(?:_cloud)?/|"
    r"Redelivered recovered final response to whatsapp(?:_cloud)?:|"
    r"Slash command /[^\s]+ denied for whatsapp(?:_cloud)?:|"
    r"Auto voice reply skipped:.*\bplatform=whatsapp(?:_cloud)?\b|"
    r"Watch pattern notification .*\bfor whatsapp(?:_cloud)?\b|"
    r"Could not get WhatsApp chat info for\b|"
    r"Profile resolution failed for (?:Platform\.)?whatsapp(?:_cloud)?/"
    r")",
    re.IGNORECASE,
)
_WHATSAPP_SESSION_VALUE_RE = re.compile(
    r"(?P<prefix>\bagent:[^:\s]+:whatsapp(?:_cloud)?:)"
    r"(?P<value>[^\r\n]*)",
    re.IGNORECASE,
)
_WHATSAPP_DELIVERY_ID_RE = re.compile(
    r"(?P<prefix>\b(?:to|for)\s+whatsapp(?:_cloud)?:)"
    r"(?P<value>.*?)"
    r"(?P<suffix>\s+\(|\r?\n$|$)",
    re.IGNORECASE,
)
_WHATSAPP_AUTO_VOICE_FIELDS_RE = re.compile(
    r"(?P<prefix>\bchat=)(?P<chat>.*?)"
    r"(?P<suffix>\s+platform=whatsapp(?:_cloud)?\b)",
    re.IGNORECASE,
)
_WHATSAPP_WATCH_FIELDS_RE = re.compile(
    r"(?P<prefix>\bchat=)(?P<chat>.*?)"
    r"(?P<middle>\s+thread=)(?P<thread>[^\r\n]*?)"
    r"(?P<ending>\r?\n)?$",
    re.IGNORECASE,
)
_WHATSAPP_DIRECT_IDENTITY_LOG_RE = re.compile(
    r"\[(?:whatsapp|whatsapp_cloud)\]\s+(?:"
    r"Authorization check raised for user\b|"
    r"Ephemeral delete failed for\b|"
    r"Handler returned empty/None response for\b|"
    r"Sending (?:command .* response|response|video attachment) .*\bto\b|"
    r"response_delivery_(?:recovered|dropped):.*\b(?:for|to)\b"
    r")",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_IDENTIFIER_LOG_RE = re.compile(
    r"\[whatsapp_cloud\].*(?:"
    r"\bwamid\b|"
    r"\bmedia(?:[_ ]id| metadata| bytes)\b|"
    r"\bcached inbound .*\bmedia\b|"
    r"\bstatus\s+\S+\s+for\s+"
    r")",
    re.IGNORECASE,
)
_WHATSAPP_DIRECT_USER_RE = re.compile(
    r"(?P<prefix>\bAuthorization check raised for user\s+)"
    r"(?P<value>[^;\r\n]+)"
    r"(?P<suffix>;|\r?\n|$)",
    re.IGNORECASE,
)
_WHATSAPP_EPHEMERAL_IDS_RE = re.compile(
    r"(?P<prefix>\bEphemeral delete failed for\s+)"
    r"(?P<chat>[^/]+?)(?P<middle>/)"
    r"(?P<message>[^:\s]+)(?P<suffix>\s*:\s*|\s*\r?\n|$)",
    re.IGNORECASE,
)
_WHATSAPP_DIRECT_CHAT_FOR_RE = re.compile(
    r"(?P<prefix>\b(?:Handler returned empty/None response|"
    r"response_delivery_(?:recovered|dropped):[^\r\n]*?)\s+(?:for|to)\s+)"
    r"(?P<value>[^\s,(]+)"
    r"(?P<suffix>\s*(?:\(|,|\r?\n|$))",
    re.IGNORECASE,
)
_WHATSAPP_DIRECT_CHAT_TO_RE = re.compile(
    r"(?P<prefix>\bto\s+)"
    r"(?P<value>[^\s,\r\n]+)"
    r"(?P<suffix>\s*(?:,|\r?\n|$))",
    re.IGNORECASE,
)
_WHATSAPP_CHAT_INFO_RE = re.compile(
    r"(?P<prefix>\bCould not get WhatsApp chat info for\s+)"
    r"(?P<value>[^:\r\n]+)"
    r"(?P<suffix>:|\r?\n|$)",
    re.IGNORECASE,
)
_WHATSAPP_PROFILE_CHAT_RE = re.compile(
    r"(?P<prefix>\bProfile resolution failed for\s+(?:Platform\.)?"
    r"whatsapp(?:_cloud)?/)"
    r"(?P<value>[^,\r\n]+)"
    r"(?P<suffix>,|\r?\n|$)",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_WAMID_RE = re.compile(
    r"(?P<prefix>\bwamid\s+)"
    r"(?P<value>[^\s,;:)]+)",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_ID_FIELD_RE = re.compile(
    r"(?P<prefix>\bid=)"
    r"(?P<value>[^,\s)]+)",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_STATUS_ID_RE = re.compile(
    r"(?P<prefix>\bstatus\s+\S+\s+for\s+)"
    r"(?P<value>[^\s,;)]+)",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_MEDIA_ID_RE = re.compile(
    r"(?P<prefix>\bmedia[_ ]id\s*(?:=|:)\s*|"
    r"\brefusing malformed media id\s+)"
    r"(?P<value>[^\s,;)]+)",
    re.IGNORECASE,
)
_WHATSAPP_CLOUD_CACHED_MEDIA_RE = re.compile(
    r"(?P<prefix>\bcached inbound [^:\r\n]+media:\s+)"
    r"(?P<value>[^\r\n]+)",
    re.IGNORECASE,
)
_LEGACY_MESSAGE_PREVIEW_LOG_RE = re.compile(
    r"(\b(?:Processing queued message after agent completion|"
    r"Processing pending message|Delivering leftover /steer as next turn):\s*).*$",
    re.IGNORECASE,
)
_LEGACY_LOG_MESSAGE_FIELD_RE = re.compile(r"\bmsg=(.*)$", re.IGNORECASE)
_SAFE_WHATSAPP_INBOUND_METADATA_RE = re.compile(
    r"\bmsg_len=\d+\b.*\breply_to_id_present=(?:True|False)\b"
    r".*\breply_to_text_len=\d+\b",
    re.IGNORECASE,
)
_SAFE_WHATSAPP_INBOUND_METADATA_FIELDS_RE = re.compile(
    r"\binbound message:\s+platform=whatsapp(?:_cloud)?\b"
    r".*?\bmsg_len=(?P<msg_len>\d{1,9})\b"
    r".*?\breply_to_id_present=(?P<reply_to_id_present>True|False)\b"
    r".*?\breply_to_text_len=(?P<reply_to_text_len>\d{1,9})\b",
    re.IGNORECASE,
)
_WHATSAPP_EXCEPTION_LOG_RE = re.compile(
    r"\[(?:whatsapp|whatsapp_cloud)\].*\b(?:"
    r"raised|failed|exception|error"
    r")\b",
    re.IGNORECASE,
)
_SAFE_WHATSAPP_LOG_ID = r"(?:absent|present(?:\(len=\d+\))?|[0-9][0-9]\*{4}[0-9][0-9])"
_SAFE_WHATSAPP_ERROR_BODY_RE = re.compile(
    r"(?:"
    r"webhook server cleanup failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"http client close failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"send failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"interactive send failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"media upload failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"media send failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"ffmpeg opus conversion failed \(returncode=-?\d+, stderr_present=(?:True|False)\)|"
    r"ffmpeg subprocess raised \(error_type=[A-Za-z_][\w.]*\)|"
    r"media metadata fetch failed \(id=(?:absent|present\(len=\d+\)), status=\d+\)|"
    r"media metadata fetch raised \(id=present\(len=\d+\), "
    r"error_type=[A-Za-z_][\w.]*\)|"
    r"media bytes fetch failed \(id=(?:absent|present\(len=\d+\)), status=\d+\)|"
    r"media bytes fetch raised \(id=present\(len=\d+\), "
    r"error_type=[A-Za-z_][\w.]*\)|"
    r"failed to write cached media \(id=(?:absent|present\(len=\d+\)), "
    r"error_type=[A-Za-z_][\w.]*\)|"
    r"failed to build event for wamid (?:absent|present\(len=\d+\)) "
    r"\(error_type=[A-Za-z_][\w.]*\)|"
    r"handle_message raised for wamid (?:absent|present\(len=\d+\)) "
    r"\(error_type=[A-Za-z_][\w.]*\)|"
    r"mark_awaiting_text failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"clarify other-prompt failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"approval confirm failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"slash_confirm\.resolve failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"slash_confirm reply failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"WhatsApp read receipt failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"failed to download inbound (?:image|video|audio|voice|document|sticker) "
    r"\(id=(?:absent|present\(len=\d+\))\) — agent will see message "
    r"metadata but not the binary|"
    r"failed to read document text \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to read document text \(error_type=[A-Za-z_][\w.]*\)|"
    r"Could not acquire session lock \(non-fatal; error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to start bridge \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to install dependencies \(error_type=[A-Za-z_][\w.]*\)|"
    r"Error stopping bridge \(error_type=[A-Za-z_][\w.]*\)|"
    r"Poll error \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to cache (?:image|audio) \(error_type=[A-Za-z_][\w.]*\)|"
    r"Error building event \(error_type=[A-Za-z_][\w.]*\)|"
    r"Native WhatsApp clarify poll failed; falling back to text "
    r"\(error_detail_present=(?:True|False)\)|"
    r"WhatsApp read receipt failed with HTTP \d+|"
    r"Authorization check raised for user "
    + _SAFE_WHATSAPP_LOG_ID
    + r" \(error_type=[A-Za-z_][\w.]*\); treating as unknown|"
    r"Ephemeral delete failed for "
    + _SAFE_WHATSAPP_LOG_ID
    + "/"
    + _SAFE_WHATSAPP_LOG_ID
    + r" \(error_type=[A-Za-z_][\w.]*\)|"
    r"Error sending image \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to send image \(error_detail=(?:present|absent)\)|"
    r"Error batching images \(error_type=[A-Za-z_][\w.]*\)|"
    r"Error sending media \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to send media \([.A-Za-z0-9_-]+\) "
    r"\(error_detail=(?:present|absent)\)|"
    r"Failed to send local file \([.A-Za-z0-9_-]+\) "
    r"\(error_detail=(?:present|absent)\)|"
    r"Error sending local file present \(error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to send error notification to user \(error_type=[A-Za-z_][\w.]*\)|"
    r"Auto-TTS failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"Busy-session handler failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"Clarify text-intercept dispatch failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"[A-Za-z0-9_.-]+ hook failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"Command '/[A-Za-z0-9_-]+' dispatch failed \(error_type=[A-Za-z_][\w.]*\)|"
    r"Send failed \(attempt \d+/\d+, retrying in [0-9.]+s; "
    r"error_detail=(?:present|absent)\)|"
    r"Failed to deliver response after \d+ retries "
    r"\(error_detail=(?:present|absent)\)|"
    r"Send failed \(error_detail=(?:present|absent)\) — "
    r"trying plain-text fallback|"
    r"Fallback send also failed \(error_detail=(?:present|absent)\)|"
    r"Could not send delivery-failure notice \(error_detail=(?:present|absent)\)|"
    r"send_typing error \(non-fatal; error_type=[A-Za-z_][\w.]*\)|"
    r"Failed to resolve live adapter for final delivery|"
    r"send_private_notice failed, falling back to public "
    r"\(error_detail=(?:present|absent)\)|"
    r"Post-stream image batch delivery failed: (?:present|absent)|"
    r"Post-stream media delivery failed: (?:present|absent)|"
    r"Error handling message \(error_type=[A-Za-z_][\w.]*\)"
    r")"
)
_EXCEPTION_TYPE_LINE_RE = re.compile(
    r"^(?P<indent>\s*)(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Failure))"
    r"(?::[^\r\n]*)?(?P<ending>\r?\n)?$"
)


@dataclass
class _WhatsAppLogRedactionState:
    """Record state carried across upload-bound log fragments."""

    record: bool = False
    message_continuation: bool = False
    message_quote: Optional[str] = None
    message_legacy: bool = False
    # A quoted WhatsApp message that crosses a physical log line has no
    # authenticated terminator in the historical text format.  Keep this
    # separate from ``message_legacy`` so callers can distinguish the two
    # sources while applying the same fail-closed EOF policy.
    message_untrusted: bool = False
    exception_continuation: bool = False
    # A current type/metadata-only error line is held for one look-ahead line.
    # Historical logger.exception records can have the exact same header, so a
    # following traceback must reclassify the header as an opener.
    safe_error_pending: Optional[str] = None
    # The retained view begins after a discarded prefix that was too large to
    # replay.  No textual boundary can prove that an older selected record
    # ended, so snapshot capture replaces the view with a safe fragment.
    prefix_unresolved: bool = False


def _has_unescaped_quote(text: str, quote: str) -> bool:
    """Return whether *text* contains a non-backslash-escaped *quote*."""
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            return True
    return False


def _is_safe_whatsapp_error_log_line(text: str) -> bool:
    """Recognize current type/metadata-only WhatsApp error records."""
    # ``splitlines(keepends=True)`` retains CRLF's ``\r``.  Normalize only the
    # physical line ending; the body whitelist remains exact and fail-closed.
    text = text.rstrip("\r\n")
    match = re.search(
        r"\[(?:whatsapp|whatsapp_cloud)\]\s+(?P<body>[^\r\n]*)$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return False
    return bool(_SAFE_WHATSAPP_ERROR_BODY_RE.fullmatch(match.group("body")))


def _redact_exception_traceback_line(line: str) -> str:
    """Keep exception type metadata while removing traceback payloads."""
    match = _EXCEPTION_TYPE_LINE_RE.match(line)
    if match:
        return (
            f"{match.group('indent')}{match.group('type')}: "
            f"[REDACTED_EXCEPTION_DETAIL]{match.group('ending') or ''}"
        )
    line_ending = line[len(line.rstrip("\r\n")) :]
    return f"[REDACTED_EXCEPTION_TRACEBACK]{line_ending}"


def _looks_like_exception_continuation(line: str) -> bool:
    """Recognize traceback framing without trusting a record terminator."""
    stripped = line.lstrip()
    if stripped.startswith("Traceback (most recent call last):"):
        return True
    if re.match(r"File\s+['\"].*['\"],\s+line\s+\d+", stripped):
        return True
    if stripped.startswith((
        "During handling of the above exception, another exception occurred:",
        "The above exception was the direct cause of the following exception:",
    )):
        return True
    return bool(_EXCEPTION_TYPE_LINE_RE.match(line))


def _redact_whatsapp_log_identity(value: str, redact_sensitive_text) -> str:
    """Return a safe representation for an identity in a WhatsApp log line."""
    if value in {"", "None", "none"}:
        return "absent"
    masked = redact_sensitive_text(
        value,
        force=True,
        redact_bare_phone_numbers=True,
    )
    if masked != value:
        return masked
    return "present"


def _redact_whatsapp_cloud_identifier(value: str) -> str:
    """Return bounded metadata for a WAMID/media/status identifier."""
    text = str(value or "")
    sentinel = re.fullmatch(r"present\(len=(\d+)\)?", text)
    if sentinel:
        return f"present(len={sentinel.group(1)})"
    return f"present(len={len(text)})" if text else "absent"


def _redact_whatsapp_log_fields(line: str, redact_sensitive_text) -> str:
    """Remove identities from historical gateway log formats."""
    line = _WHATSAPP_SESSION_VALUE_RE.sub(
        lambda match: (
            f"{match.group('prefix')}[REDACTED_WHATSAPP_SESSION]"
        ),
        line,
    )
    line = _WHATSAPP_DELIVERY_ID_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_log_identity(match.group('value'), redact_sensitive_text)}"
            f"{match.group('suffix')}"
        ),
        line,
    )
    line = _WHATSAPP_AUTO_VOICE_FIELDS_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_log_identity(match.group('chat'), redact_sensitive_text)}"
            f"{match.group('suffix')}"
        ),
        line,
    )
    line = _WHATSAPP_WATCH_FIELDS_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_log_identity(match.group('chat'), redact_sensitive_text)}"
            f"{match.group('middle')}"
            f"{_redact_whatsapp_log_identity(match.group('thread'), redact_sensitive_text)}"
            f"{match.group('ending') or ''}"
        ),
        line,
    )
    line = _WHATSAPP_DIRECT_USER_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_log_identity(match.group('value'), redact_sensitive_text)}"
            f"{match.group('suffix')}"
        ),
        line,
    )
    line = _WHATSAPP_EPHEMERAL_IDS_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_log_identity(match.group('chat'), redact_sensitive_text)}"
            f"{match.group('middle')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('message'))}"
            f"{match.group('suffix')}"
        ),
        line,
    )
    for pattern in (_WHATSAPP_DIRECT_CHAT_FOR_RE, _WHATSAPP_DIRECT_CHAT_TO_RE,
                    _WHATSAPP_CHAT_INFO_RE, _WHATSAPP_PROFILE_CHAT_RE):
        line = pattern.sub(
            lambda match: (
                f"{match.group('prefix')}"
                f"{_redact_whatsapp_log_identity(match.group('value'), redact_sensitive_text)}"
                f"{match.group('suffix')}"
            ),
            line,
        )
    line = _WHATSAPP_CLOUD_WAMID_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('value'))}"
        ),
        line,
    )
    line = _WHATSAPP_CLOUD_ID_FIELD_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('value'))}"
        ),
        line,
    )
    line = _WHATSAPP_CLOUD_STATUS_ID_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('value'))}"
        ),
        line,
    )
    line = _WHATSAPP_CLOUD_MEDIA_ID_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('value'))}"
        ),
        line,
    )
    line = _WHATSAPP_CLOUD_CACHED_MEDIA_RE.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_redact_whatsapp_cloud_identifier(match.group('value'))}"
        ),
        line,
    )
    return line


# ---------------------------------------------------------------------------
# Paste services — try paste.rs first, dpaste.com as fallback.
# ---------------------------------------------------------------------------

_PASTE_RS_URL = "https://paste.rs/"
_DPASTE_COM_URL = "https://dpaste.com/api/"

# Maximum bytes to read from a single log file for upload.
# paste.rs caps at ~1 MB; we stay under that with headroom.
_MAX_LOG_BYTES = 512_000
# Prefix redaction state is reconstructed from candidate record lines only.
# The exact discarded bytes are still hashed on the open descriptor for the
# snapshot race check, but ordinary diagnostic lines are never decoded and
# passed through the state machine.  This bound limits the expensive
# state-machine portion even when a selected record is very large.
_WHATSAPP_STATE_SCAN_BYTES = 256 * 1024

# Auto-delete pastes after this many seconds (6 hours).
_AUTO_DELETE_SECONDS = 21600


# ---------------------------------------------------------------------------
# Pending-deletion tracking (replaces the old fork-and-sleep subprocess).
# ---------------------------------------------------------------------------

def _pending_file() -> Path:
    """Path to ``~/.hermes/pastes/pending.json``.

    Each entry: ``{"url": "...", "expire_at": <unix_ts>}``.  Scheduled
    DELETEs used to be handled by spawning a detached Python process per
    paste that slept for 6 hours; those accumulated forever if the user
    ran ``hermes debug share`` repeatedly.

    Deletion is now driven by the gateway's cron ticker
    (``gateway/run.py::_start_cron_ticker``) which calls
    ``_sweep_expired_pastes`` once per hour.  ``hermes debug share`` also
    runs an opportunistic sweep on entry as a fallback for CLI-only users
    who never start the gateway.
    """
    return get_hermes_home() / "pastes" / "pending.json"


def _load_pending() -> list[dict]:
    path = _pending_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Filter to well-formed entries only
            return [
                e for e in data
                if isinstance(e, dict) and "url" in e and "expire_at" in e
            ]
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return []


def _save_pending(entries: list[dict]) -> None:
    path = _pending_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        atomic_replace(tmp, path)
    except OSError:
        # Non-fatal — worst case the user has to run ``hermes debug delete``
        # manually.
        pass


def _record_pending(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS) -> None:
    """Record *urls* for deletion at ``now + delay_seconds``.

    Only paste.rs URLs are recorded (dpaste.com auto-expires).  Entries
    are merged into any existing pending.json.
    """
    paste_rs_urls = [u for u in urls if _extract_paste_id(u)]
    if not paste_rs_urls:
        return

    entries = _load_pending()
    # Dedupe by URL: keep the later expire_at if same URL appears twice
    by_url: dict[str, float] = {e["url"]: float(e["expire_at"]) for e in entries}
    expire_at = time.time() + delay_seconds
    for u in paste_rs_urls:
        by_url[u] = max(expire_at, by_url.get(u, 0.0))
    merged = [{"url": u, "expire_at": ts} for u, ts in by_url.items()]
    _save_pending(merged)


def _sweep_expired_pastes(now: Optional[float] = None) -> tuple[int, int]:
    """Synchronously DELETE any pending pastes whose ``expire_at`` has passed.

    Returns ``(deleted, remaining)``.  Best-effort: failed deletes stay in
    the pending file and will be retried on the next sweep.  Silent —
    intended to be called from every ``hermes debug`` invocation with
    minimal noise.
    """
    entries = _load_pending()
    if not entries:
        return (0, 0)

    current = time.time() if now is None else now
    deleted = 0
    remaining: list[dict] = []

    for entry in entries:
        try:
            expire_at = float(entry.get("expire_at", 0))
        except (TypeError, ValueError):
            continue  # drop malformed entries
        if expire_at > current:
            remaining.append(entry)
            continue

        url = entry.get("url", "")
        try:
            if delete_paste(url):
                deleted += 1
                continue
        except Exception:
            # Network hiccup, 404 (already gone), etc. — drop the entry
            # after a grace period; don't retry forever.
            pass

        # Retain failed deletes for up to 24h past expiration, then give up.
        if expire_at + 86400 > current:
            remaining.append(entry)
        else:
            deleted += 1  # count as reaped (paste.rs will GC eventually)

    if deleted:
        _save_pending(remaining)

    return (deleted, len(remaining))


def _best_effort_sweep_expired_pastes() -> None:
    """Attempt pending-paste cleanup without letting /debug fail offline."""
    try:
        _sweep_expired_pastes()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Privacy / delete helpers
# ---------------------------------------------------------------------------

_PRIVACY_NOTICE = """\
⚠️  This will upload system info + logs to a PUBLIC paste service.

Cryptographic secrets (API keys, tokens, passwords) are redacted before
upload, but the following personal data is NOT redacted and will be public:
  • Your display name and persistent platform user ID
  • Verbatim content of your recent messages (prompts, responses, tool output)
  • Local filesystem paths
  • Any other PII present in the logs

The resulting URL is public to anyone who has the link. Pastes auto-delete
after 6 hours, but may be archived by third parties in the meantime.

Use --local to view the report without uploading.
"""

_GATEWAY_PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** This uploads system info + recent log tails "
    "(may contain conversation fragments) to a public paste service. "
    "Full logs are NOT included from the gateway — use `hermes debug share` "
    "from the CLI for full log uploads.\n"
    "Pastes auto-delete after 6 hours."
)


def _extract_paste_id(url: str) -> Optional[str]:
    """Extract the paste ID from a paste.rs or dpaste.com URL.

    Returns the ID string, or None if the URL doesn't match a known service.
    """
    url = url.strip().rstrip("/")
    for prefix in ("https://paste.rs/", "http://paste.rs/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return None


def delete_paste(url: str) -> bool:
    """Delete a paste from paste.rs.  Returns True on success.

    Only paste.rs supports unauthenticated DELETE.  dpaste.com pastes
    expire automatically but cannot be deleted via API.
    """
    paste_id = _extract_paste_id(url)
    if not paste_id:
        raise ValueError(
            f"Cannot delete: only paste.rs URLs are supported.  Got: {url}"
        )

    target = f"{_PASTE_RS_URL}{paste_id}"
    req = urllib.request.Request(
        target, method="DELETE",
        headers={"User-Agent": "hermes-agent/debug-share"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return 200 <= resp.status < 300


def _schedule_auto_delete(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS):
    """Record *urls* for deletion ``delay_seconds`` from now.

    Previously this spawned a detached Python subprocess per call that slept
    for 6 hours and then issued DELETE requests.  Those subprocesses leaked —
    every ``hermes debug share`` invocation added ~20 MB of resident Python
    interpreters that never exited until the sleep completed.

    The replacement is stateless: we append to ``~/.hermes/pastes/pending.json``
    and the gateway's cron ticker sweeps expired entries once per hour.
    ``hermes debug share`` also runs an opportunistic sweep as a fallback
    for CLI-only users.  If neither runs again, paste.rs's own retention
    policy handles cleanup.
    """
    _record_pending(urls, delay_seconds=delay_seconds)


def _upload_paste_rs(content: str) -> str:
    """Upload to paste.rs.  Returns the paste URL.

    paste.rs accepts a plain POST body and returns the URL directly.
    """
    data = content.encode("utf-8")
    req = urllib.request.Request(
        _PASTE_RS_URL, data=data, method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from paste.rs: {url[:200]}")
    return url


def _upload_dpaste_com(content: str, expiry_days: int = 7) -> str:
    """Upload to dpaste.com.  Returns the paste URL.

    dpaste.com uses multipart form data.
    """
    boundary = "----HermesDebugBoundary9f3c"

    def _field(name: str, value: str) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )

    body = (
        _field("content", content)
        + _field("syntax", "text")
        + _field("expiry_days", str(expiry_days))
        + f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        _DPASTE_COM_URL, data=body, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "hermes-agent/debug-share",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from dpaste.com: {url[:200]}")
    return url


def upload_to_pastebin(content: str, expiry_days: int = 7) -> str:
    """Upload *content* to a paste service, trying paste.rs then dpaste.com.

    Returns the paste URL on success, raises on total failure.
    """
    errors: list[str] = []

    # Try paste.rs first (simple, fast)
    try:
        return _upload_paste_rs(content)
    except Exception as exc:
        errors.append(f"paste.rs: {exc}")

    # Fallback: dpaste.com (supports expiry)
    try:
        return _upload_dpaste_com(content, expiry_days=expiry_days)
    except Exception as exc:
        errors.append(f"dpaste.com: {exc}")

    raise RuntimeError(
        "Failed to upload to any paste service:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Log file reading
# ---------------------------------------------------------------------------


@dataclass
class LogSnapshot:
    """Single-read snapshot of a log file used by debug-share."""

    path: Optional[Path]
    tail_text: str
    full_text: Optional[str]


def _primary_log_path(log_name: str) -> Optional[Path]:
    """Where *log_name* would live if present. Doesn't check existence."""
    from hermes_cli.logs import LOG_FILES

    filename = LOG_FILES.get(log_name)
    return (get_hermes_home() / "logs" / filename) if filename else None


# Logs written by a client process rather than by this backend. When the
# desktop app talks to a remote/docker/SSH backend, `hermes debug share` runs
# on the *backend* and can never see them — a bare "(file not found)" then
# reads as "the app logged nothing" and sends triage down a dead end, which is
# exactly the wrong answer when the client is the thing being debugged.
_CLIENT_SIDE_LOGS = {
    "desktop": (
        "written by Hermes Desktop on the machine running the app, not by this "
        "backend. If the desktop connects to a remote/docker/SSH backend, collect "
        "it on that client machine"
    ),
}


def _missing_log_note(log_name: str) -> str:
    """Explain a missing log instead of stating a bare absence.

    For a client-side log the absence is expected on a remote backend, so the
    note names the writer and the path to collect by hand.
    """
    reason = _CLIENT_SIDE_LOGS.get(log_name)
    if reason is None:
        return "(file not found)"

    primary = _primary_log_path(log_name)
    where = f" — expected at {primary}" if primary else ""
    return f"(not on this host: {reason}{where})"


def _resolve_log_path(log_name: str) -> Optional[Path]:
    """Find the log file for *log_name*, falling back to the .1 rotation.

    Returns the first non-empty candidate (primary, then .1), or None.
    Callers distinguish 'empty primary' from 'truly missing' via
    :func:`_primary_log_path`.
    """
    primary = _primary_log_path(log_name)
    if primary is None:
        return None

    if primary.exists() and primary.stat().st_size > 0:
        return primary

    rotated = primary.parent / f"{primary.name}.1"
    if rotated.exists() and rotated.stat().st_size > 0:
        return rotated

    return None


def _redact_log_text_with_state(
    text: str,
    state: Optional[_WhatsAppLogRedactionState] = None,
    *,
    redact_output: bool = True,
    finalize: bool = False,
) -> tuple[str, _WhatsAppLogRedactionState]:
    """Transform one log fragment while preserving WhatsApp record state."""
    current = state or _WhatsAppLogRedactionState()
    if not text:
        return text, current

    if redact_output:
        from agent.redact import redact_sensitive_text

        # Keep the general force-mode pass over the complete blob so multiline
        # credentials (for example private keys) retain their existing coverage.
        text = redact_sensitive_text(text, force=True)
    else:
        redact_sensitive_text = None

    redacted_lines = []
    pending_safe_error = current.safe_error_pending
    current.safe_error_pending = None
    for line in text.splitlines(keepends=True):
        if pending_safe_error is not None:
            if _looks_like_exception_continuation(line):
                # The exact safe-looking header is also emitted by older
                # logger.exception paths.  A traceback continuation proves
                # that this occurrence is historical/ambiguous, so retain
                # the sanitized header but enter fail-closed traceback state.
                if redact_output:
                    redacted_lines.append(pending_safe_error)
                pending_safe_error = None
                current.exception_continuation = True
            else:
                if redact_output:
                    redacted_lines.append(pending_safe_error)
                pending_safe_error = None

        if current.message_continuation:
            if current.message_legacy or current.message_untrusted:
                # Historical queued/pending/leftover previews were emitted
                # with ``%s`` and have no trusted length or framing metadata.
                # A quote, literal ``...'`` suffix, timestamp, or logger
                # prefix can all be supplied by the message itself.  The same
                # is true of a quoted non-legacy WhatsApp message once it
                # crosses a physical line: an apostrophe in a continuation is
                # message data, not an authenticated closing delimiter.  Once
                # either selected record starts, remain redacted through EOF
                # rather than trusting an in-band terminator.  This
                # deliberately over-redacts later diagnostics, but preserves
                # the upload privacy invariant.
                if current.message_untrusted:
                    safe_metadata = _SAFE_WHATSAPP_INBOUND_METADATA_FIELDS_RE.search(
                        line
                    )
                    if safe_metadata and redact_output:
                        line_ending = line[len(line.rstrip("\r\n")) :]
                        redacted_lines.append(
                            "[WHATSAPP_INBOUND_METADATA] "
                            f"msg_len={safe_metadata.group('msg_len')} "
                            "reply_to_id_present="
                            f"{safe_metadata.group('reply_to_id_present')} "
                            "reply_to_text_len="
                            f"{safe_metadata.group('reply_to_text_len')}"
                            f"{line_ending}"
                        )
                        continue
                if redact_output:
                    line_ending = line[len(line.rstrip("\r\n")) :]
                    redacted_lines.append(
                        f"[REDACTED_MESSAGE_PREVIEW]{line_ending}"
                    )
                continue

            if redact_output:
                line_ending = line[len(line.rstrip("\r\n")) :]
                redacted_lines.append(
                    f"[REDACTED_MESSAGE_PREVIEW]{line_ending}"
                )
            continue

        if current.exception_continuation:
            # The exception message and traceback can forge both an exception
            # type line and a complete timestamp/level/logger prefix.  No
            # textual boundary is therefore trustworthy; keep the remainder
            # of the upload-bound view redacted through EOF.
            if redact_output:
                redacted_lines.append(_redact_exception_traceback_line(line))
            continue

        whatsapp_inbound = bool(_WHATSAPP_INBOUND_LOG_RE.search(line))
        whatsapp_conversation = bool(_WHATSAPP_CONVERSATION_LOG_RE.search(line))
        whatsapp_session_key = bool(_WHATSAPP_SESSION_KEY_LOG_RE.search(line))
        whatsapp_generic_identity = bool(
            _WHATSAPP_GENERIC_IDENTITY_LOG_RE.search(line)
        )
        whatsapp_direct_identity = bool(
            _WHATSAPP_DIRECT_IDENTITY_LOG_RE.search(line)
        )
        whatsapp_cloud_identifier = bool(
            _WHATSAPP_CLOUD_IDENTIFIER_LOG_RE.search(line)
        )
        legacy_preview_match = _LEGACY_MESSAGE_PREVIEW_LOG_RE.search(line)
        legacy_preview = bool(legacy_preview_match)
        safe_whatsapp_error = _is_safe_whatsapp_error_log_line(line)
        whatsapp_exception = bool(
            _WHATSAPP_EXCEPTION_LOG_RE.search(line)
        ) and not safe_whatsapp_error
        if safe_whatsapp_error:
            # Hold one line of look-ahead: a historical logger.exception
            # record can use the exact same sanitized header as a current
            # warning, and its following traceback must remain fail-closed.
            whatsapp_cloud_identifier = False
        if whatsapp_inbound:
            current.record = True
        message_field = _LEGACY_LOG_MESSAGE_FIELD_RE.search(line)
        selected = (
            current.record
            or whatsapp_conversation
            or whatsapp_session_key
            or whatsapp_generic_identity
            or whatsapp_direct_identity
            or whatsapp_cloud_identifier
            or legacy_preview
            or whatsapp_exception
        )

        if redact_output and selected:
            line = redact_sensitive_text(
                line,
                force=True,
                redact_bare_phone_numbers=True,
            )
            if (
                whatsapp_session_key
                or whatsapp_generic_identity
                or whatsapp_direct_identity
                or whatsapp_cloud_identifier
            ):
                line = _redact_whatsapp_log_fields(line, redact_sensitive_text)

        if redact_output and (current.record or whatsapp_conversation):
            line = _LEGACY_LOG_MESSAGE_FIELD_RE.sub(
                "msg=[REDACTED_MESSAGE_PREVIEW]",
                line,
            )
        if (current.record or whatsapp_conversation) and message_field:
            message_value = message_field.group(1).rstrip("\r\n")
            if message_value[:1] in {"'", '"'}:
                quote = message_value[0]
                message_complete = _has_unescaped_quote(message_value[1:], quote)
            else:
                quote = None
                message_complete = False

            if message_complete:
                current.record = False
            else:
                current.message_continuation = True
                current.message_quote = quote
                current.message_legacy = False
                # A quote can delimit a complete message only while it is
                # contained in the same physical record line.  After a
                # newline, all subsequent quotes are attacker-controlled
                # message bytes and cannot close this state safely.
                current.message_untrusted = True
        elif whatsapp_inbound and _SAFE_WHATSAPP_INBOUND_METADATA_RE.search(line):
            # The current gateway's body-free inbound record is self-contained.
            current.record = False

        if redact_output and legacy_preview:
            line = _LEGACY_MESSAGE_PREVIEW_LOG_RE.sub(
                r"\1[REDACTED_MESSAGE_PREVIEW]",
                line,
            )

        if legacy_preview_match:
            # These legacy preview logs used ``%s`` and can contain arbitrary
            # newlines.  Their literal quote/ellipsis suffix is forgeable by
            # the message, so remain redacted through EOF for every selected
            # legacy record, including one-line values.
            current.message_continuation = True
            current.message_quote = None
            current.message_legacy = True
            current.message_untrusted = False

        if whatsapp_exception:
            # Older logger.exception records may carry sensitive exception
            # messages and complete tracebacks after an otherwise safe header.
            # Keep only the bounded header and sanitize every ambiguous
            # continuation line.  Current runtime sinks log exception type
            # metadata instead, but historical debug-share inputs remain in
            # scope.
            current.exception_continuation = True

        if safe_whatsapp_error:
            pending_safe_error = line
        elif redact_output:
            redacted_lines.append(line)

    if pending_safe_error is not None:
        if finalize:
            if redact_output:
                redacted_lines.append(pending_safe_error)
        else:
            current.safe_error_pending = pending_safe_error

    redacted = "".join(redacted_lines) if redact_output else ""
    if redact_output:
        redacted = _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", redacted)
    return redacted, current


def _redact_log_text(text: str) -> str:
    """Run ``redact_sensitive_text`` with ``force=True`` over upload-bound text.

    Uses ``force=True`` so redaction fires regardless of the operator's
    ``security.redact_secrets`` setting. The local on-disk log file is
    not modified; only the in-memory copy headed for the public paste
    service is sanitized. Returns the redacted text (or the original
    when empty / non-string).
    """
    # The bare-phone option is intentionally not safe for arbitrary text.
    # Select only structured WhatsApp records and exact historical
    # message-preview records. Historical queued/pending/leftover previews and
    # exception tracebacks have no trustworthy textual terminator, so their
    # selected continuations remain redacted through EOF.
    redacted, _state = _redact_log_text_with_state(text, finalize=True)
    return redacted


def _whatsapp_log_state_at(
    log_file: BinaryIO,
    byte_offset: int,
    *,
    content_hash: Optional[Any] = None,
) -> _WhatsAppLogRedactionState:
    """Recover redaction state from a bounded suffix of the discarded prefix.

    The exact discarded bytes still come from the same open descriptor and
    are fed into ``content_hash`` when requested, preserving the later
    append/overwrite/truncate verification.  Selector presence is detected
    with a bounded-memory byte scan rather than replaying every ordinary line
    through the Python state machine.  A selected continuation found inside
    the bounded replay window is preserved as redacted message state; an
    older selected record that cannot be replayed is marked unresolved and
    replaced with a safe fragment instead of risking a leak.
    """
    state = _WhatsAppLogRedactionState()
    if byte_offset <= 0:
        return state

    # Use the already-required complete-prefix pass to look for selectors
    # without decoding every ordinary diagnostic line in Python.  If no
    # selector occurs anywhere in
    # the discarded bytes, a large ordinary log is known not to contain an
    # older selected record and can retain its useful diagnostics.  A selector
    # does not prove that its record ended before the retained view, so keep
    # the fail-closed fragment in that case.
    selector_markers = (
        b" whatsapp",
        b"platform=whatsapp",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
        b":whatsapp:",
        b":whatsapp_cloud:",
        b"for whatsapp",
        b"wamid",
        b"media_id",
        b"media id",
        b"processing queued message after agent completion:",
        b"processing pending message:",
        b"delivering leftover /steer as next turn:",
    )
    selector_seen = False
    state_candidate_markers = (
        b"processing queued message after agent completion:",
        b"processing pending message:",
        b"delivering leftover /steer as next turn:",
        b"conversation turn:",
        b"inbound message:",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
    )
    state_opener_before_window = False
    last_candidate_line_start: Optional[int] = None

    def _classify_prefix_candidate(candidate_at: int, resume_at: int) -> None:
        """Classify one pre-window candidate without retaining offsets."""
        nonlocal state_opener_before_window, last_candidate_line_start
        try:
            line_start = _physical_line_start(log_file, candidate_at)
            if line_start == last_candidate_line_start:
                log_file.seek(resume_at)
                return
            last_candidate_line_start = line_start
            log_file.seek(line_start)
            candidate_line = log_file.readline(1_048_576)
        except Exception:
            state_opener_before_window = True
            log_file.seek(resume_at)
            return
        finally:
            # The caller's sequential scan must continue from the end of the
            # chunk even when candidate classification seeks elsewhere.
            if log_file.tell() != resume_at:
                log_file.seek(resume_at)

        if not candidate_line or (
            b"\n" not in candidate_line and len(candidate_line) >= 1_048_576
        ):
            # An unbounded candidate line cannot be classified safely.
            state_opener_before_window = True
            return
        candidate_text = candidate_line.decode("utf-8", errors="replace")
        if _LEGACY_MESSAGE_PREVIEW_LOG_RE.search(candidate_text):
            state_opener_before_window = True
            return
        if _is_safe_whatsapp_error_log_line(candidate_text):
            # A current type/metadata-only warning and an older
            # logger.exception header can be byte-for-byte identical.  Read
            # the next physical line from this same descriptor before
            # declaring the prefix self-contained; a traceback continuation
            # means the retained suffix must remain fail-closed.
            next_at = line_start + len(candidate_line)
            log_file.seek(next_at)
            next_line = log_file.readline(1_048_576)
            next_text = next_line.decode("utf-8", errors="replace")
            if next_line and _looks_like_exception_continuation(next_text):
                state_opener_before_window = True
            log_file.seek(resume_at)
            return
        if (
            _WHATSAPP_EXCEPTION_LOG_RE.search(candidate_text)
            and not _is_safe_whatsapp_error_log_line(candidate_text)
        ):
            state_opener_before_window = True
            return
        if not (
            _WHATSAPP_CONVERSATION_LOG_RE.search(candidate_text)
            or _WHATSAPP_INBOUND_LOG_RE.search(candidate_text)
        ):
            return
        message_field = _LEGACY_LOG_MESSAGE_FIELD_RE.search(candidate_text)
        if not message_field:
            return
        message_value = message_field.group(1).rstrip("\r\n")
        if message_value[:1] in {"'", '"'}:
            if not _has_unescaped_quote(message_value[1:], message_value[0]):
                state_opener_before_window = True
        else:
            state_opener_before_window = True

    overlap = b""
    max_marker_len = max(map(len, selector_markers))
    remaining = byte_offset
    processed = 0
    scan_start = max(0, byte_offset - _WHATSAPP_STATE_SCAN_BYTES)
    log_file.seek(0)
    while remaining > 0:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            break
        if content_hash is not None:
            content_hash.update(chunk)
        haystack = overlap + chunk.lower()
        if any(marker in haystack for marker in selector_markers):
            selector_seen = True
        for marker in state_candidate_markers:
            search_from = 0
            while True:
                marker_at = haystack.find(marker, search_from)
                if marker_at < 0:
                    break
                absolute_at = max(0, processed + marker_at - len(overlap))
                if absolute_at < scan_start:
                    _classify_prefix_candidate(absolute_at, processed + len(chunk))
                    if state_opener_before_window:
                        break
                search_from = marker_at + 1
            if state_opener_before_window:
                break
        overlap = haystack[-(max_marker_len - 1) :]
        remaining -= len(chunk)
        processed += len(chunk)

    if not selector_seen:
        return state

    # Replay only the final bounded window when a selector is present.  This
    # keeps the useful redacted-message view for the common case where the
    # selected record is recent, while avoiding the old O(file-size) Python
    # line replay for selector-free logs.
    scan_start = max(0, byte_offset - _WHATSAPP_STATE_SCAN_BYTES)
    log_file.seek(scan_start)
    scan_remaining = byte_offset - scan_start
    window_selector_seen = False
    if scan_start > 0:
        log_file.seek(scan_start - 1)
        at_line_boundary = log_file.read(1) == b"\n"
        log_file.seek(scan_start)
        if not at_line_boundary:
            fragment = log_file.readline(scan_remaining)
            scan_remaining -= len(fragment)

    while scan_remaining > 0:
        line = log_file.readline(scan_remaining)
        if not line:
            break
        scan_remaining -= len(line)
        lowered = line.lower()
        if any(marker in lowered for marker in selector_markers):
            window_selector_seen = True
        _unused, state = _redact_log_text_with_state(
            line.decode("utf-8", errors="replace"),
            state,
            redact_output=False,
        )
        if state.message_continuation or state.exception_continuation:
            return state

    if state_opener_before_window:
        # The selected marker predates the bounded replay, so no textual line
        # can prove that an attacker-controlled multiline record ended before
        # the retained view.
        state.prefix_unresolved = True
    return state


def _descriptor_digest(
    log_file: BinaryIO,
    byte_offset: int,
    byte_count: int,
) -> Optional[bytes]:
    """Hash an exact range from an already-open log descriptor."""
    content_hash = hashlib.sha256()
    remaining = byte_count
    log_file.seek(byte_offset)
    while remaining > 0:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            return None
        remaining -= len(chunk)
        content_hash.update(chunk)
    return content_hash.digest()


def _physical_line_start(log_file: BinaryIO, byte_offset: int) -> int:
    """Find the start of the physical line containing ``byte_offset``."""
    cursor = max(0, byte_offset)
    while cursor:
        start = max(0, cursor - 65536)
        log_file.seek(start)
        block = log_file.read(cursor - start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        cursor = start
    return 0


def _split_line_is_whatsapp(
    log_file: BinaryIO,
    byte_offset: int,
    file_size: int,
) -> bool:
    """Classify a retained no-newline suffix using its complete line.

    The retained suffix can begin in the middle of a physical line.  Looking
    only at that suffix would miss a selector split across the byte cap, while
    unconditionally replacing every such suffix destroys ordinary diagnostics.
    Scan the complete line on the already-open descriptor and fail closed when
    it contains WhatsApp-related markers.  The scan keeps bounded overlap and
    memory, and a non-WhatsApp line remains available for sharing.
    """
    line_start = _physical_line_start(log_file, byte_offset)
    remaining = max(0, file_size - line_start)
    log_file.seek(line_start)
    overlap = b""
    markers = (
        b" whatsapp",
        b"platform=whatsapp",
        b"[whatsapp]",
        b"[whatsapp_cloud]",
        b":whatsapp:",
        b":whatsapp_cloud:",
        b"whatsapp chat info",
        b"platform.whatsapp/",
        b"wamid ",
        b"wamid=",
        b"wamid:",
        b"for wamid",
        b"media_id=",
        b"media id=",
        b"media id ",
        # Historical watch-pattern notifications identify the platform with
        # ``for whatsapp[_cloud]`` rather than a ``platform=`` field.  Keep
        # this selector in the complete-line classifier so a byte-cap split
        # cannot expose the retained identity suffix.
        b"for whatsapp",
    )
    while remaining:
        chunk = log_file.read(min(65536, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        haystack = overlap + chunk.lower()
        if any(marker in haystack for marker in markers):
            return True
        overlap = haystack[-128:]
    return False


def _decode_capped_utf8(data: bytes, max_bytes: int) -> str:
    """Decode a byte-capped view without invalid UTF-8 expansion.

    ``errors='replace'`` can turn an orphaned continuation byte at a suffix
    cut into a three-byte replacement character, making the returned string
    exceed ``max_bytes`` after re-encoding.  Decode the selected suffix while
    ignoring only incomplete/invalid leading bytes, then enforce the physical
    line boundary on the decoded text.  The result is valid UTF-8 and its
    encoded size is always at most the requested cap.
    """
    if max_bytes <= 0:
        return ""
    truncated = len(data) > max_bytes
    on_boundary = True
    selected = data
    if truncated:
        cut = len(data) - max_bytes
        on_boundary = cut > 0 and data[cut - 1 : cut] == b"\n"
        selected = data[cut:]

    text = selected.decode("utf-8", errors="ignore")
    if truncated and not on_boundary and "\n" in text:
        text = text.split("\n", 1)[1]

    # The ignore decode above normally makes this a no-op.  Keep the final
    # invariant explicit for callers handling unusual decoder input.
    while text and len(text.encode("utf-8")) > max_bytes:
        text = text[1:]
    return text


def _capture_log_snapshot(
    log_name: str,
    *,
    tail_lines: int,
    max_bytes: int = _MAX_LOG_BYTES,
    redact: bool = True,
) -> LogSnapshot:
    """Capture a log once and derive summary/full-log views from it.

    The report tail and standalone log upload must come from the same file
    snapshot. Otherwise a rotation/truncate between reads can make the report
    look newer than the uploaded ``agent.log`` paste.

    When ``redact`` is True (the default), both ``tail_text`` and
    ``full_text`` are run through ``_redact_log_text`` so the snapshot
    returned is upload-safe. The on-disk log file is never modified.
    Pass ``redact=False`` to capture original log content (used by
    ``hermes debug share --no-redact``).
    """
    log_path = _resolve_log_path(log_name)
    if log_path is None:
        primary = _primary_log_path(log_name)
        tail = (
            "(file empty)"
            if primary and primary.exists()
            else _missing_log_note(log_name)
        )
        return LogSnapshot(path=None, tail_text=tail, full_text=None)

    try:
        with open(log_path, "rb") as f:
            initial_stat = os.fstat(f.fileno())
            initial_fingerprint = (
                initial_stat.st_dev,
                initial_stat.st_ino,
                initial_stat.st_size,
                initial_stat.st_mtime_ns,
            )
            size = initial_stat.st_size
            if size == 0:
                # The file was truncated or replaced before the open completed.
                return LogSnapshot(
                    path=log_path,
                    tail_text="(file empty)",
                    full_text=None,
                )

            if size <= max_bytes:
                # Bind the view to the size observed at open time.  An
                # ordinary append must not leak into this point-in-time view.
                raw = f.read(size)
            else:
                # Read from the end until we have enough bytes for the
                # standalone upload and enough newline context to render the
                # summary tail from the same snapshot.
                chunk_size = 8192
                pos = size
                chunks: list[bytes] = []
                total = 0
                newline_count = 0

                while (
                    pos > 0
                    and (total < max_bytes or newline_count <= tail_lines + 1)
                    and total < max_bytes * 2
                ):
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    chunk = f.read(read_size)
                    chunks.insert(0, chunk)
                    total += len(chunk)
                    newline_count += chunk.count(b"\n")
                    chunk_size = min(chunk_size * 2, 65536)

                raw = b"".join(chunks)

            raw_start = pos if size > max_bytes else 0
            split_physical_line = False
            if raw_start > 0 and raw:
                # Chunk reads begin at an arbitrary byte. Drop the incomplete
                # first physical line, then scan the discarded prefix through
                # this exact boundary so multiline state remains trustworthy.
                first_newline = raw.find(b"\n")
                if first_newline >= 0:
                    raw_start += first_newline + 1
                    raw = raw[first_newline + 1 :]
                else:
                    # No retained newline means the selected suffix may be
                    # the continuation of one physical record.  Its marker
                    # and platform selector can therefore be split across
                    # the discarded prefix and retained bytes.  Do not parse
                    # those fragments as independent records: redact the
                    # entire retained fragment below.
                    split_physical_line = _split_line_is_whatsapp(
                        f, raw_start, size
                    )

            initial_content_hash = hashlib.sha256()
            if redact and not split_physical_line:
                state = _whatsapp_log_state_at(
                    f,
                    raw_start,
                    content_hash=initial_content_hash,
                )
                if state.prefix_unresolved:
                    split_physical_line = True
            else:
                if redact:
                    # Preserve the exact initial descriptor range for the
                    # append-race check without reconstructing state from a
                    # partial physical line.
                    remaining = raw_start
                    f.seek(0)
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        initial_content_hash.update(chunk)
                state = _WhatsAppLogRedactionState()
            initial_content_hash.update(raw)
            initial_content_digest = initial_content_hash.digest()
            final_stat = os.fstat(f.fileno())
            final_fingerprint = (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
            )
            if final_fingerprint != initial_fingerprint:
                same_descriptor = (
                    final_stat.st_dev == initial_stat.st_dev
                    and final_stat.st_ino == initial_stat.st_ino
                )
                append_candidate = (
                    same_descriptor and final_stat.st_size > size
                )
                verification_offset = 0 if redact else raw_start
                verification_size = size - verification_offset
                verified_digest = (
                    _descriptor_digest(
                        f,
                        verification_offset,
                        verification_size,
                    )
                    if append_candidate
                    else None
                )
                verified_stat = os.fstat(f.fileno())
                stable_initial_range = (
                    verified_digest == initial_content_digest
                    and verified_stat.st_dev == initial_stat.st_dev
                    and verified_stat.st_ino == initial_stat.st_ino
                    and verified_stat.st_size >= size
                )
                if not append_candidate or not stable_initial_range:
                    raise RuntimeError("log changed during snapshot capture")

        full_raw = raw
        full_was_truncated = raw_start > 0 or len(full_raw) > max_bytes
        if len(full_raw) > max_bytes:
            cut = len(full_raw) - max_bytes
            # Check whether the cut lands exactly on a line boundary.  If the
            # byte just before the cut position is a newline the first retained
            # byte starts a complete line and we should keep it.  Only drop a
            # partial first line when we're genuinely mid-line.
            on_boundary = cut > 0 and full_raw[cut - 1 : cut] == b"\n"
            full_raw = full_raw[cut:]
            if not on_boundary and b"\n" in full_raw:
                full_raw = full_raw.split(b"\n", 1)[1]

        if redact:
            if split_physical_line:
                safe_text = "[REDACTED_LOG_FRAGMENT]\n"
            else:
                safe_text, _state = _redact_log_text_with_state(
                    raw.decode("utf-8", errors="replace"),
                    state,
                    finalize=True,
                )
            tail_text = "".join(
                safe_text.splitlines(keepends=True)[-tail_lines:]
            ).rstrip("\n")

            safe_full_raw = safe_text.encode("utf-8")
            if len(safe_full_raw) > max_bytes:
                # Redaction can expand a selected view (for example, a bare
                # seven-digit WhatsApp identity becomes ``12****67``).  The
                # same line-boundary cap below then omits one or more source
                # records, so preserve the existing truncation marker rather
                # than presenting the shortened view as complete.
                full_was_truncated = True
            full_text = _decode_capped_utf8(safe_full_raw, max_bytes)
        else:
            all_text = raw.decode("utf-8", errors="replace")
            tail_text = "".join(
                all_text.splitlines(keepends=True)[-tail_lines:]
            ).rstrip("\n")
            full_text = _decode_capped_utf8(full_raw, max_bytes)

        if full_was_truncated:
            full_text = (
                f"[... truncated — showing last ~{max_bytes // 1024}KB ...]\n"
                f"{full_text}"
            )

        return LogSnapshot(path=log_path, tail_text=tail_text, full_text=full_text)
    except Exception as exc:
        return LogSnapshot(path=log_path, tail_text=f"(error reading: {exc})", full_text=None)


def _capture_default_log_snapshots(
    log_lines: int, *, redact: bool = True
) -> dict[str, LogSnapshot]:
    """Capture all logs used by debug-share exactly once.

    ``redact`` is forwarded to each ``_capture_log_snapshot`` call so all
    captured logs share the same redaction policy for a given run.
    """
    errors_lines = min(log_lines, 100)
    return {
        "agent": _capture_log_snapshot(
            "agent", tail_lines=log_lines, redact=redact
        ),
        "errors": _capture_log_snapshot(
            "errors", tail_lines=errors_lines, redact=redact
        ),
        "gateway": _capture_log_snapshot(
            "gateway", tail_lines=errors_lines, redact=redact
        ),
        "gui": _capture_log_snapshot(
            "gui", tail_lines=errors_lines, redact=redact
        ),
        "desktop": _capture_log_snapshot(
            "desktop", tail_lines=errors_lines, redact=redact
        ),
    }


# ---------------------------------------------------------------------------
# Debug report collection
# ---------------------------------------------------------------------------

def _capture_dump() -> str:
    """Run ``hermes dump`` and return its stdout as a string."""
    from hermes_cli.dump import run_dump

    class _FakeArgs:
        show_keys = False

    old_stdout = sys.stdout
    sys.stdout = capture = io.StringIO()
    try:
        run_dump(_FakeArgs())
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout

    return capture.getvalue()


def collect_debug_report(
    *,
    log_lines: int = 200,
    dump_text: str = "",
    log_snapshots: Optional[dict[str, LogSnapshot]] = None,
) -> str:
    """Build the summary debug report: system dump + log tails.

    Parameters
    ----------
    log_lines
        Number of recent lines to include per log file.
    dump_text
        Pre-captured dump output.  If empty, ``hermes dump`` is run
        internally.

    Returns the report as a plain-text string ready for upload.
    """
    buf = io.StringIO()

    if not dump_text:
        dump_text = _capture_dump()
    buf.write(dump_text)

    if log_snapshots is None:
        log_snapshots = _capture_default_log_snapshots(log_lines)

    # ── Recent log tails (summary only) ──────────────────────────────────
    buf.write("\n\n")
    buf.write(f"--- agent.log (last {log_lines} lines) ---\n")
    buf.write(log_snapshots["agent"].tail_text)
    buf.write("\n\n")

    errors_lines = min(log_lines, 100)
    buf.write(f"--- errors.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["errors"].tail_text)
    buf.write("\n\n")

    buf.write(f"--- gateway.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["gateway"].tail_text)
    buf.write("\n\n")

    buf.write(f"--- gui.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["gui"].tail_text)
    buf.write("\n\n")

    buf.write(f"--- desktop.log (last {errors_lines} lines) ---\n")
    buf.write(log_snapshots["desktop"].tail_text)
    buf.write("\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared bundle collection (used by both the paste.rs and Nous-S3 paths)
# ---------------------------------------------------------------------------

# Bundle format identifier embedded in the Nous-S3 JSON envelope. The
# discord-support viewer keys off this string to parse the bundle.
_NOUS_BUNDLE_FORMAT = "hermes-debug-share/1"


def collect_share_bundle(
    log_lines: int = 200,
    redact: bool = True,
) -> dict[str, str]:
    """Collect the debug report + full logs as a label→text mapping.

    Returns ``{"report": ..., "agent.log": ..., "gateway.log": ...,
    "desktop.log": ...}`` where each value is the already-redacted (when
    ``redact`` is True) text that would be uploaded.  Keys for logs that are
    absent/empty are simply omitted.

    This is the single source of collection + redaction shared by both
    destinations: the paste.rs path (:func:`build_debug_share`) and the
    Nous-S3 path (``--nous``).  Centralising it guarantees the Nous bundle is
    built from the *same* force-redacted snapshots as the public paste path —
    redaction is the safety boundary, so the Nous path must never see raw
    logs.

    The dump header is prepended to each full log (mirroring the historical
    paste behaviour) so every file is self-contained, and the redaction
    banner is prepended when ``redact`` is True.
    """
    dump_text = _capture_dump()
    log_snapshots = _capture_default_log_snapshots(log_lines, redact=redact)

    report = collect_debug_report(
        log_lines=log_lines,
        dump_text=dump_text,
        log_snapshots=log_snapshots,
    )
    agent_log = log_snapshots["agent"].full_text
    gateway_log = log_snapshots["gateway"].full_text
    gui_log = log_snapshots["gui"].full_text
    desktop_log = log_snapshots["desktop"].full_text

    # Prepend dump header to each full log so every file is self-contained.
    if agent_log:
        agent_log = dump_text + "\n\n--- full agent.log ---\n" + agent_log
    if gateway_log:
        gateway_log = dump_text + "\n\n--- full gateway.log ---\n" + gateway_log
    if gui_log:
        gui_log = dump_text + "\n\n--- full gui.log ---\n" + gui_log
    if desktop_log:
        desktop_log = dump_text + "\n\n--- full desktop.log ---\n" + desktop_log

    # Visible banner so reviewers know redaction was applied at upload time.
    if redact:
        report = _REDACTION_BANNER + report
        if agent_log:
            agent_log = _REDACTION_BANNER + agent_log
        if gateway_log:
            gateway_log = _REDACTION_BANNER + gateway_log
        if gui_log:
            gui_log = _REDACTION_BANNER + gui_log
        if desktop_log:
            desktop_log = _REDACTION_BANNER + desktop_log

    bundle: dict[str, str] = {"report": report}
    if agent_log:
        bundle["agent.log"] = agent_log
    if gateway_log:
        bundle["gateway.log"] = gateway_log
    if gui_log:
        bundle["gui.log"] = gui_log
    if desktop_log:
        bundle["desktop.log"] = desktop_log
    return bundle


def build_nous_bundle(bundle: dict[str, str], redact: bool = True) -> bytes:
    """Gzip-compress a :func:`collect_share_bundle` mapping into the Nous envelope.

    The JSON shape is what the discord-support viewer (Repo 3) parses::

        {"format": "hermes-debug-share/1",
         "redacted": <bool>,
         "created": <iso8601>,
         "files": {"report": ..., "agent.log": ..., ...}}
    """
    created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    envelope = {
        "format": _NOUS_BUNDLE_FORMAT,
        "redacted": bool(redact),
        "created": created,
        "files": bundle,
    }
    return gzip.compress(json.dumps(envelope).encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

@dataclass
class DebugShareResult:
    """Structured outcome of a ``debug share`` upload.

    Returned by :func:`build_debug_share` so non-CLI callers (the dashboard
    web server, gateway) can render the uploaded paste URLs as real links
    instead of scraping printed text.
    """

    urls: dict  # label -> paste URL (e.g. {"Report": "...", "agent.log": "..."})
    failures: list  # human-readable "label: error" strings for optional uploads
    redacted: bool  # whether force-mode redaction was applied before upload
    auto_delete_seconds: int  # how long until the pastes auto-delete
    report: str = ""  # the summary report text (kept for local fallback)


def build_debug_share(
    *,
    log_lines: int = 200,
    expiry: int = 7,
    redact: bool = True,
) -> DebugShareResult:
    """Collect the debug report + full logs, upload each, return the URLs.

    This is the shared core behind ``hermes debug share`` (CLI) and the
    dashboard ``POST /api/ops/debug-share`` endpoint. It performs blocking
    network I/O (paste uploads) — callers inside an event loop must run it in
    a worker thread.

    The summary report upload is required: on failure this raises
    ``RuntimeError``. Full-log uploads are best-effort; their errors are
    collected into ``failures`` rather than raised.
    """
    _best_effort_sweep_expired_pastes()

    # Collect the report + full logs (force-redacted when redact=True) via the
    # shared collector so the paste.rs and Nous-S3 paths build identical,
    # identically-redacted bundles. The dump header + redaction banner are
    # applied inside collect_share_bundle.
    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)

    if redact:
        logger.info(
            "hermes debug share: applied force-mode redaction to log snapshots before upload"
        )

    report = bundle["report"]

    urls: dict[str, str] = {}
    failures: list[str] = []

    # 1. Summary report (required — raises on failure so callers can fall back)
    urls["Report"] = upload_to_pastebin(report, expiry_days=expiry)

    # 2-5. Full logs (optional — failures are collected, not raised)
    for label in ("agent.log", "gateway.log", "gui.log", "desktop.log"):
        content = bundle.get(label)
        if not content:
            continue
        try:
            urls[label] = upload_to_pastebin(content, expiry_days=expiry)
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    # Schedule auto-deletion after 6 hours.
    _schedule_auto_delete(list(urls.values()))

    return DebugShareResult(
        urls=urls,
        failures=failures,
        redacted=redact,
        auto_delete_seconds=_AUTO_DELETE_SECONDS,
        report=report,
    )


def _confirm_upload(args) -> bool:
    """Require explicit consent before any debug-share upload.

    The privacy notice is printed by the caller. This gates the actual
    upload: with ``--yes`` (or ``-y``) we proceed unprompted; otherwise we
    ask an interactive ``[y/N]`` question. In a non-interactive context
    (no TTY on stdin — scripts, CI, piped input) we refuse rather than
    hang or upload silently, so debug data can't be exposed without a
    deliberate ``--yes``.

    Returns True to proceed with the upload, False to abort.
    """
    if bool(getattr(args, "yes", False)):
        return True
    if not sys.stdin.isatty():
        print(
            "ERROR: Non-interactive mode requires --yes to confirm upload.\n"
            "       This prevents accidental exposure of personal data.\n"
            "       Use --local to view the report without uploading.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        answer = input("Upload debug report? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in ("y", "yes"):
        print("Aborted.")
        return False
    return True


def run_debug_share(args):
    """Collect debug report + full logs, upload each, print URLs."""
    log_lines = getattr(args, "lines", 200)
    expiry = getattr(args, "expire", 7)
    local_only = getattr(args, "local", False)
    nous = getattr(args, "nous", False)
    redact = not getattr(args, "no_redact", False)

    if local_only:
        # Local-only path never uploads — render the report to stdout and bail
        # before any network I/O. Reuses the shared collector so the rendered
        # output matches exactly what would be uploaded.
        _best_effort_sweep_expired_pastes()
        print("Collecting debug report...")
        bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
        print(bundle["report"])
        for title, label in (
            ("FULL agent.log", "agent.log"),
            ("FULL gateway.log", "gateway.log"),
            ("FULL gui.log", "gui.log"),
            ("FULL desktop.log", "desktop.log"),
        ):
            body = bundle.get(label)
            if body:
                print(f"\n\n{'=' * 60}")
                print(title)
                print(f"{'=' * 60}\n")
                print(body)
        return

    if nous:
        _run_debug_share_nous(args, log_lines=log_lines, redact=redact)
        return

    print(_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    print("Collecting debug report...")
    print("Uploading...")

    try:
        result = build_debug_share(
            log_lines=log_lines,
            expiry=expiry,
            redact=redact,
        )
    except RuntimeError as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        print("\nRun `hermes debug share --local` to print the report instead.\n")
        sys.exit(1)

    # Print results
    label_width = max(len(k) for k in result.urls)
    print("\nDebug report uploaded:")
    for label, url in result.urls.items():
        print(f"  {label:<{label_width}}  {url}")

    if result.failures:
        print(f"\n  (failed to upload: {', '.join(result.failures)})")

    hours = result.auto_delete_seconds // 3600
    print(f"\n⏱  Pastes will auto-delete in {hours} hours.")

    # Manual delete fallback
    print("To delete now:  hermes debug delete <url>")

    print("\nShare these links with the Hermes team for support.")


_NOUS_PRIVACY_NOTICE = """\
⚠️  --nous: This uploads your debug bundle to Nous-INTERNAL storage (AWS S3),
    NOT a public paste service. The following is included:
  • System info (OS, Python/Hermes version, provider, which API keys are
    configured — NOT the actual keys)
  • Full agent.log, gateway.log, and desktop.log (up to 512 KB each — likely
    contains conversation content, tool outputs, and file paths)

  • The bundle is viewable only by Nous staff (and allowlisted Discord mods)
    via a Google-login-gated viewer.
  • It is NOT a public paste — there is no public URL to the contents.
  • It auto-deletes after 14 days.
"""


def _run_debug_share_nous(args, *, log_lines: int, redact: bool) -> None:
    """Handle ``hermes debug share --nous``: upload the bundle to Nous-S3.

    Collects the same force-redacted bundle as the paste path, gzips it into
    the Nous envelope, requests a signed URL from NAS, uploads, and prints the
    private viewer link. On any failure falls back to a clear error that
    suggests ``--local``.
    """
    from hermes_cli.diagnostics_upload import share_to_nous

    print(_NOUS_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    if not redact:
        print(
            "⚠️  --no-redact is set: secrets in your logs will NOT be redacted "
            "before upload.\n"
        )
    print("Collecting debug report...")
    _best_effort_sweep_expired_pastes()

    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
    if redact:
        logger.info(
            "hermes debug share --nous: applied force-mode redaction before upload"
        )
    blob = build_nous_bundle(bundle, redact=redact)

    print("Uploading to Nous diagnostics storage...")
    try:
        res = share_to_nous(blob)
    except Exception as exc:
        print(
            f"\nNous upload failed: {exc}\n"
            "\nThe Nous diagnostics service may be unavailable or not yet "
            "provisioned.\n"
            "Run `hermes debug share --local` to print the report instead, "
            "or `hermes debug share` to upload to a public paste service.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    view_url = res.get("viewUrl") or res.get("view_url")
    print("\nDebug bundle uploaded to Nous (private):")
    if view_url:
        print(f"  View URL  {view_url}")
    else:
        print(f"  (no view URL returned; upload id: {res.get('id', '?')})")

    expires_at = res.get("expiresAt") or res.get("expires_at")
    if expires_at:
        print(f"\n⏱  Auto-deletes at {expires_at} (14-day retention).")
    else:
        print("\n⏱  Auto-deletes after 14 days.")

    print(
        "\nShare this private link with the Nous team — only Nous staff "
        "(via Google login) can open it."
    )


def run_debug_delete(args):
    """Delete one or more paste URLs uploaded by /debug."""
    urls = getattr(args, "urls", [])
    if not urls:
        print("Usage: hermes debug delete <url> [<url> ...]")
        print("  Deletes paste.rs pastes uploaded by 'hermes debug share'.")
        return

    for url in urls:
        try:
            ok = delete_paste(url)
            if ok:
                print(f"  ✓ Deleted: {url}")
            else:
                print(f"  ✗ Failed to delete: {url} (unexpected response)")
        except ValueError as exc:
            print(f"  ✗ {exc}")
        except Exception as exc:
            print(f"  ✗ Could not delete {url}: {exc}")


def run_debug(args):
    """Route debug subcommands."""
    # Opportunistic sweep of expired pastes on every ``hermes debug`` call.
    # Replaces the old per-paste sleeping subprocess that used to leak as
    # one orphaned Python interpreter per scheduled deletion.  Silent and
    # best-effort — any failure is swallowed so ``hermes debug`` stays
    # reliable even when offline.
    try:
        _sweep_expired_pastes()
    except Exception:
        pass

    subcmd = getattr(args, "debug_command", None)
    if subcmd == "share":
        run_debug_share(args)
    elif subcmd == "delete":
        run_debug_delete(args)
    else:
        # Default: show help
        print("Usage: hermes debug <command>")
        print()
        print("Commands:")
        print("  share    Upload debug report to a paste service and print URL")
        print("  delete   Delete a previously uploaded paste")
        print()
        print("Options (share):")
        print("  --lines N    Number of log lines to include (default: 200)")
        print("  --expire N   Paste expiry in days (default: 7)")
        print("  --local      Print report locally instead of uploading")
        print("  --nous       Upload to Nous-internal storage (private, staff-only,")
        print("               auto-deletes in 14 days) instead of a public paste")
        print("  --no-redact  Disable upload-time secret redaction (default: redact)")
        print()
        print("Options (delete):")
        print("  <url> ...    One or more paste URLs to delete")
