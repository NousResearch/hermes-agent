"""Privacy projections for historical diagnostic copies, never source log mutation.

Historical message and traceback records have no authenticated multiline
terminator. Once one is selected, keep its ambiguous continuation redacted
through the end of the captured view. Selection always examines original text.
"""

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)
_MESSAGE_FIELD_SINGLE_RE = re.compile(
    r"\b(?P<key>msg|message|prompt|response|content|text|answer|body|quote|root|"
    r"target|title|url|reply_to_text|tool_output)="
    r"'(?:\\[^\r\n]|[^'\\\r\n])*'"
)
_MESSAGE_FIELD_DOUBLE_RE = re.compile(
    r"\b(?P<key>msg|message|prompt|response|content|text|answer|body|quote|root|"
    r'target|title|url|reply_to_text|tool_output)="'
    r'(?:\\[^\r\n]|[^"\\\r\n])*"'
)
_MESSAGE_FIELD_BARE_RE = re.compile(
    r"\b(?P<key>msg|message|prompt|response|content|text|answer|body|quote|root|"
    r"target|title|url|reply_to_text|tool_output)="
    r"(?!['\"]).*?(?=\s+[A-Za-z_][\w.-]*=|$)",
    re.MULTILINE,
)
_USER_FIELD_RE = re.compile(
    r"\b(?P<key>user|user_id|username|display_name|sender)="
    r".*?"
    r"(?=\s+[A-Za-z_][\w.-]*=|$)",
    re.MULTILINE,
)
_IDENTITY_TOKEN_FIELD_RE = re.compile(
    r"\b(?P<key>chat|chat_id|room|room_id|peer|peer_id|session|session_id|"
    r"thread|thread_id|reply_to_id|file_token|comment_id|reply_id|event_id|"
    r"from_open_id|to_open_id|wiki_token|obj_token|file|comment|reply|from|to|"
    r"token)=[^\s]+"
)
_UNAUTHORIZED_USER_RE = re.compile(
    r"(?m)(?P<prefix>\bUnauthorized user:\s*)[^\s(]+"
    r"(?:\s+\((?:[^\r\n)]|\)(?!\s+on\s+))*\))?"
    r"(?P<suffix>\s+on\s+[^\s\r\n]+)"
)
_USER_PROMPT_RE = re.compile(r"(?im)(?P<prefix>\bUser prompt:\s*)[^\r\n]*")
_WEBHOOK_RESPONSE_RE = re.compile(
    r"(?im)(?P<prefix>\[(?:webhook|msgraph_webhook)\]\s+Response for\s*)[^\r\n]*?"
    r"(?P<separator>:\s+)[^\r\n]*"
)
_WEBHOOK_DIRECT_DELIVER_RE = re.compile(
    r"(?im)(?P<prefix>\[webhook\]\s+direct-deliver log-only:\s*)[^\r\n]*"
)
_TELEGRAM_BLOCKED_RE = re.compile(
    r"(?im)(?P<prefix>\[Telegram\]\s+Blocked\s+(?:media from\s+)?"
    r"unauthorized user\s+)[^\s\r\n]+(?P<middle>\s+in chat\s+)[^\s\r\n]+"
)
_VOICE_INPUT_RE = re.compile(
    r"(?im)(?P<prefix>\bVoice input from user\s+)\d+(?P<separator>:\s*)[^\r\n]*"
)
_EMAIL_MESSAGE_RE = re.compile(
    r"(?im)(?P<prefix>\[Email\]\s+New message from\s+)[^\s\r\n]+"
    r"(?P<separator>:\s*)[^\r\n]*"
)
_EMAIL_SENT_REPLY_RE = re.compile(
    r"(?im)(?P<prefix>\[Email\]\s+Sent reply to\s+)[^\s\r\n]+"
    r"(?P<middle>\s+\(subject:\s*)[^\r\n)]*(?P<suffix>\))"
)
_TELEGRAM_UPDATE_ANSWER_RE = re.compile(
    r"(?im)(?P<prefix>\bTelegram update prompt answered\s+)"
    r"[^\r\n]*?(?P<middle>\s+by user\s+)[^\s\r\n]+"
)
_FEISHU_RAW_RESPONSE_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+API FAIL raw response:\s*)[^\r\n]*"
)
_FEISHU_API_REQUEST_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+API >>>\s+\S+\s+\S+)"
    r"[^\r\n]*"
)
_FEISHU_META_VALUE_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+query_document_meta:"
    r"[^\r\n]*?\svalue=)[^\r\n]*"
)
_FEISHU_FULL_PROMPT_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+Full prompt:\s*)[^\r\n]*"
)
_FEISHU_AGENT_RESPONSE_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+Agent response"
    r"\s+\(\d+\s+chars\):\s*)[^\r\n]*"
)
_FEISHU_SESSION_RE = re.compile(
    r"(?im)(?P<prefix>\[Feishu-Comment\]\s+Session\s+(?:expired|saved):\s*)"
    r"[^\s\r\n]+"
)
_IMAGE_PROMPT_RE = re.compile(
    r"(?im)(?P<prefix>\b(?:Editing|Generating) image[^\r\n]*?"
    r"(?:prompt|prompt text):\s*)[^\r\n]*"
)
_FORWARDED_UPDATE_PROMPT_RE = re.compile(
    r"(?im)(?P<prefix>\bForwarded update prompt to\s+)[^\s\r\n]+"
    r"(?P<separator>:\s*)[^\r\n]*"
)
_DUPLICATE_VOICE_RE = re.compile(
    r"(?im)(?P<prefix>\bSuppressing duplicate voice transcript for\s+)"
    r"guild=[^\s\r\n]+\s+user=[^\s\r\n]+(?P<separator>:\s*)[^\r\n]*"
)
_IPV4_ADDRESS_RE = re.compile(
    r"(?<![0-9A-Fa-f.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9A-Fa-f.])"
)
_IPV6_ADDRESS_RE = re.compile(
    r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
    r"(?:%[A-Za-z0-9_.-]+)?(?![0-9A-Fa-f:])"
)
_VERSION_VALUE_PREFIX_RE = re.compile(r"\b(?:version|ver)=\s*$", re.IGNORECASE)
_UNIX_HOME_COMPONENT_RE = re.compile(r"(?<![\w.-])/(?:home|Users)/[^/\s:'\"]+")
_WINDOWS_HOME_COMPONENT_RE = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s:'\"]+")
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


_GENERIC_MESSAGE_PATTERNS = (
    _USER_PROMPT_RE,
    _WEBHOOK_RESPONSE_RE,
    _WEBHOOK_DIRECT_DELIVER_RE,
    _VOICE_INPUT_RE,
    _EMAIL_MESSAGE_RE,
    _EMAIL_SENT_REPLY_RE,
    _TELEGRAM_UPDATE_ANSWER_RE,
    _FEISHU_RAW_RESPONSE_RE,
    _FEISHU_API_REQUEST_RE,
    _FEISHU_META_VALUE_RE,
    _FEISHU_FULL_PROMPT_RE,
    _FEISHU_AGENT_RESPONSE_RE,
    _IMAGE_PROMPT_RE,
    _FORWARDED_UPDATE_PROMPT_RE,
    _DUPLICATE_VOICE_RE,
    _MESSAGE_FIELD_BARE_RE,
)
# Prefix recovery uses these only as candidate markers; the raw-text parser
# classifies the complete physical line before deciding whether it opens state.
_GENERIC_MESSAGE_MARKERS = (
    b"user prompt:",
    b"response for",
    b"direct-deliver log-only:",
    b"voice input from user",
    b"new message from",
    b"sent reply to",
    b"telegram update prompt answered",
    b"api fail raw response:",
    b"api >>>",
    b"query_document_meta:",
    b"full prompt:",
    b"agent response",
    b"image",
    b"forwarded update prompt to",
    b"suppressing duplicate voice transcript",
    b"msg=",
    b"message=",
    b"prompt=",
    b"response=",
    b"content=",
    b"text=",
    b"answer=",
    b"body=",
    b"quote=",
    b"root=",
    b"target=",
    b"title=",
    b"url=",
    b"reply_to_text=",
    b"tool_output=",
)


_QUOTED_MESSAGE_FIELD_START_RE = re.compile(
    r"\b(?P<key>msg|message|prompt|response|content|text|answer|body|quote|root|"
    r"target|title|url|reply_to_text|tool_output)=(?P<quote>[\"'])"
)


def _generic_message_opener(line: str) -> bool:
    if any(pattern.search(line) for pattern in _GENERIC_MESSAGE_PATTERNS):
        return True
    return any(
        not _has_unescaped_quote(line[match.end() :], match.group("quote"))
        for match in _QUOTED_MESSAGE_FIELD_START_RE.finditer(line)
    )


@dataclass
class _WhatsAppLogRedactionState:
    """Record state carried across diagnostic log fragments."""

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


_SAFE_TRACEBACK_EXCEPTION_TYPES = frozenset({
    "Exception",
    "RuntimeError",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "OSError",
    "PermissionError",
    "FileNotFoundError",
    "TimeoutError",
    "ConnectionError",
    "ConnectionResetError",
})


def _redact_exception_traceback_line(line: str) -> str:
    """Keep exception type metadata while removing traceback payloads."""
    match = _EXCEPTION_TYPE_LINE_RE.match(line)
    # A continuation can forge a type-looking line. Only fixed builtin
    # names are useful metadata; arbitrary identifiers can encode payload.
    if match and match.group("type") in _SAFE_TRACEBACK_EXCEPTION_TYPES:
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
    from gateway.log_redaction import log_safe_gateway_identity

    if value in {"", "None", "none"}:
        return "absent"
    return log_safe_gateway_identity("whatsapp", value)


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
        lambda match: f"{match.group('prefix')}[REDACTED_WHATSAPP_SESSION]",
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
    for pattern in (
        _WHATSAPP_DIRECT_CHAT_FOR_RE,
        _WHATSAPP_DIRECT_CHAT_TO_RE,
        _WHATSAPP_CHAT_INFO_RE,
        _WHATSAPP_PROFILE_CHAT_RE,
    ):
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
                    redacted_lines.append(f"[REDACTED_MESSAGE_PREVIEW]{line_ending}")
                continue

            if redact_output:
                line_ending = line[len(line.rstrip("\r\n")) :]
                redacted_lines.append(f"[REDACTED_MESSAGE_PREVIEW]{line_ending}")
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
        whatsapp_generic_identity = bool(_WHATSAPP_GENERIC_IDENTITY_LOG_RE.search(line))
        whatsapp_direct_identity = bool(_WHATSAPP_DIRECT_IDENTITY_LOG_RE.search(line))
        whatsapp_cloud_identifier = bool(_WHATSAPP_CLOUD_IDENTIFIER_LOG_RE.search(line))
        legacy_preview_match = _LEGACY_MESSAGE_PREVIEW_LOG_RE.search(line)
        generic_preview = _generic_message_opener(line)
        legacy_preview = bool(legacy_preview_match) or generic_preview
        safe_whatsapp_error = _is_safe_whatsapp_error_log_line(line)
        safe_inbound = bool(_SAFE_WHATSAPP_INBOUND_METADATA_RE.search(line))
        whatsapp_exception = (
            bool(_WHATSAPP_EXCEPTION_LOG_RE.search(line)) and not safe_whatsapp_error
        )
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
            or whatsapp_exception
        )

        if redact_output and selected:
            if (
                whatsapp_session_key
                or whatsapp_generic_identity
                or whatsapp_direct_identity
                or whatsapp_cloud_identifier
            ):
                line = _redact_whatsapp_log_fields(line, redact_sensitive_text)
            line = redact_sensitive_text(
                line, force=True, redact_bare_phone_numbers=True
            )

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
        elif whatsapp_inbound and safe_inbound:
            # The current gateway's body-free inbound record is self-contained.
            current.record = False

        if redact_output and legacy_preview:
            line = _LEGACY_MESSAGE_PREVIEW_LOG_RE.sub(
                r"\1[REDACTED_MESSAGE_PREVIEW]",
                line,
            )

        if generic_preview and redact_output:
            line = _redact_debug_share_privacy(line)

        if legacy_preview and not current.message_continuation:
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
            # The opener itself may contain an arbitrary error body. Only
            # known safe metadata formats bypass this branch; an otherwise
            # selected line is as untrusted as its continuation.
            if redact_output:
                line_ending = line[len(line.rstrip("\r\n")) :]
                line = f"[REDACTED_WHATSAPP_EXCEPTION]{line_ending}"
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
        # Keep the force-mode pass over the complete selected output so
        # multiline credentials retain coverage without influencing selection.
        redacted = redact_sensitive_text(redacted, force=True)
        redacted = _redact_debug_share_privacy(redacted)
    return redacted, current


def _redact_log_text(text: str) -> str:
    """Project diagnostic text for local/private collection and previews."""
    # The bare-phone option is intentionally not safe for arbitrary text.
    # Select structured WhatsApp records and historical diagnostic
    # message-preview records. Historical queued/pending/leftover previews and
    # exception tracebacks have no trustworthy textual terminator, so their
    # selected continuations remain redacted through EOF.
    redacted, _state = _redact_log_text_with_state(text, finalize=True)
    return redacted


def _redact_unclosed_message_fields(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        for match in _QUOTED_MESSAGE_FIELD_START_RE.finditer(line):
            if not _has_unescaped_quote(line[match.end() :], match.group("quote")):
                ending = line[len(line.rstrip("\r\n")) :]
                line = (
                    line[: match.start()]
                    + match.group("key")
                    + "=[REDACTED_MESSAGE]"
                    + ending
                )
                break
        lines.append(line)
    return "".join(lines)


def _redact_debug_share_privacy(text: str) -> str:
    """Remove personal context that is not covered by secret redaction."""
    text = _redact_unclosed_message_fields(text)
    text = _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", text)
    text = _DUPLICATE_VOICE_RE.sub(
        lambda m: (
            f"{m.group('prefix')}guild=[REDACTED_ID] user=[REDACTED_ID]"
            f"{m.group('separator')}[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _MESSAGE_FIELD_SINGLE_RE.sub(
        lambda m: f"{m.group('key')}='[REDACTED_MESSAGE]'", text
    )
    text = _MESSAGE_FIELD_DOUBLE_RE.sub(
        lambda m: f'{m.group("key")}="[REDACTED_MESSAGE]"', text
    )
    text = _MESSAGE_FIELD_BARE_RE.sub(
        lambda m: (
            m.group(0)
            if m.group(0).partition("=")[2] == "[REDACTED_MESSAGE_PREVIEW]"
            else f"{m.group('key')}=[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _USER_FIELD_RE.sub(lambda m: f"{m.group('key')}=[REDACTED_ID]", text)
    text = _IDENTITY_TOKEN_FIELD_RE.sub(
        lambda m: f"{m.group('key')}=[REDACTED_ID]", text
    )
    text = _UNAUTHORIZED_USER_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_ID] ([REDACTED_NAME]){m.group('suffix')}"
        ),
        text,
    )
    text = _USER_PROMPT_RE.sub(lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text)
    text = _WEBHOOK_RESPONSE_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_ID]{m.group('separator')}[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _WEBHOOK_DIRECT_DELIVER_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _TELEGRAM_BLOCKED_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_ID]{m.group('middle')}[REDACTED_ID]",
        text,
    )
    text = _VOICE_INPUT_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_ID]{m.group('separator')}[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _EMAIL_MESSAGE_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_EMAIL]"
            f"{m.group('separator')}[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _EMAIL_SENT_REPLY_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_EMAIL]"
            f"{m.group('middle')}[REDACTED_MESSAGE]{m.group('suffix')}"
        ),
        text,
    )
    text = _TELEGRAM_UPDATE_ANSWER_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_MESSAGE]{m.group('middle')}[REDACTED_ID]"
        ),
        text,
    )
    text = _FEISHU_RAW_RESPONSE_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _FEISHU_API_REQUEST_RE.sub(
        lambda m: f"{m.group('prefix')} [REDACTED_REQUEST_METADATA]", text
    )
    text = _FEISHU_META_VALUE_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _FEISHU_FULL_PROMPT_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _FEISHU_AGENT_RESPONSE_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _FEISHU_SESSION_RE.sub(lambda m: f"{m.group('prefix')}[REDACTED_ID]", text)
    text = _IMAGE_PROMPT_RE.sub(
        lambda m: f"{m.group('prefix')}[REDACTED_MESSAGE]", text
    )
    text = _FORWARDED_UPDATE_PROMPT_RE.sub(
        lambda m: (
            f"{m.group('prefix')}[REDACTED_ID]{m.group('separator')}[REDACTED_MESSAGE]"
        ),
        text,
    )
    text = _IPV4_ADDRESS_RE.sub(_redact_ipv4_address, text)
    text = _IPV6_ADDRESS_RE.sub(_redact_ipv6_address, text)

    for raw_path, replacement in _privacy_path_replacements():
        text = text.replace(raw_path, replacement)
        text = text.replace(raw_path.replace("\\", "/"), replacement.replace("\\", "/"))

    text = _UNIX_HOME_COMPONENT_RE.sub("~", text)
    return _WINDOWS_HOME_COMPONENT_RE.sub("~", text)


def _redact_ipv4_address(match: re.Match[str]) -> str:
    """Redact valid IPv4 addresses without treating version-like text as IP."""
    prefix = match.string[max(0, match.start() - 24) : match.start()]
    if _VERSION_VALUE_PREFIX_RE.search(prefix):
        return match.group(0)
    try:
        ipaddress.ip_address(match.group(0))
    except ValueError:
        return match.group(0)
    return "[REDACTED_IP]"


def _redact_ipv6_address(match: re.Match[str]) -> str:
    """Redact valid IPv6 addresses, including scoped link-local forms."""
    candidate = match.group(0)
    address = candidate.split("%", 1)[0]
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return candidate
    return "[REDACTED_IP]" if parsed.version == 6 else candidate


def _privacy_path_replacements() -> list[tuple[str, str]]:
    """Return local paths that should not be exposed in public debug pastes."""
    replacements: list[tuple[str, str]] = []
    seen: set[str] = set()
    candidates = [
        (get_hermes_home(), "~/.hermes"),
        (Path.home(), "~"),
    ]

    for path, replacement in candidates:
        raw = str(path)
        if raw and raw not in seen:
            seen.add(raw)
            replacements.append((raw, replacement))

    return replacements
