"""Quarantined Discord BOT_MSG v1 protocol helpers.

Discord bot-to-bot control routing is decommissioned. Normal runtime files must
not import this module; it remains only so old references/tests can fail or be
removed deliberately without deleting tracked history in this change.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

DISCORD_BOT_MSG_KINDS = {
    "status",
    "action_required",
    "approval_decision",
    "approval_request",
    "decision_request",
    "handoff",
    "review_request",
    "audit",
}
DISCORD_BOT_MSG_CORRELATION_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
DISCORD_BOT_MSG_PROTOCOL_VERSION = "v1"
DISCORD_MENTION_RE = re.compile(r"<@!?(\d+)>")
DISCORD_BOT_MSG_REQUIRED_ERROR_RE = re.compile(
    r"Outbound raw mention of allowed bot \d+ requires send_bot_message\(\.\.\.\)"
)
DISCORD_BOT_ROUTING_GUARD_ERROR_RE = re.compile(r"BOT_ROUTING_GUARD:")
DISCORD_BOT_MSG_DEFAULT_MAX_BODY_CHARS = 1800


def discord_bot_reply_false_reaction() -> str:
    """Reaction used to acknowledge BOT_MSG v1 messages that need no reply."""
    return (os.getenv("DISCORD_BOT_REPLY_FALSE_REACTION", "👀") or "👀").strip() or "👀"


def discord_bot_msg_protocol_version() -> str:
    return (os.getenv("DISCORD_BOT_MSG_PROTOCOL", DISCORD_BOT_MSG_PROTOCOL_VERSION) or "").strip()


def discord_bot_msg_max_body_chars() -> int:
    """Maximum inbound BOT_MSG body size dispatched into a model turn."""
    raw = (os.getenv("DISCORD_BOT_MSG_MAX_BODY_CHARS", "") or "").strip()
    if not raw:
        return DISCORD_BOT_MSG_DEFAULT_MAX_BODY_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DISCORD_BOT_MSG_DEFAULT_MAX_BODY_CHARS
    return max(0, value)


def discord_mention_id(mention: str) -> Optional[str]:
    match = re.fullmatch(r"<@!?(\d+)>", (mention or "").strip())
    return match.group(1) if match else None


def build_discord_bot_msg_v1(
    *,
    recipient_bot_id: str,
    body: str,
    reply_expected: bool,
    kind: str,
    correlation_id: str,
) -> str:
    recipient = str(recipient_bot_id or "").strip()
    if not recipient.isdigit():
        raise ValueError("Invalid BOT_MSG v1 recipient_bot_id")
    if kind not in DISCORD_BOT_MSG_KINDS:
        raise ValueError(f"Invalid BOT_MSG v1 kind: {kind}")
    if not DISCORD_BOT_MSG_CORRELATION_RE.fullmatch(correlation_id or ""):
        raise ValueError("Invalid BOT_MSG v1 correlation_id")
    if kind != "status" and body == "":
        raise ValueError("BOT_MSG v1 body is required for non-status kinds")
    return "\n".join(
        [
            f"<@{recipient}>",
            "BOT_MSG v1",
            f"reply_expected: {'true' if reply_expected else 'false'}",
            f"kind: {kind}",
            f"correlation_id: {correlation_id}",
            "---",
            body or "",
        ]
    )


def parse_discord_bot_msg_v1(content: str, recipient_bot_id: Any) -> Optional[Dict[str, Any]]:
    """Parse the strict Discord BOT_MSG v1 envelope.

    Header grammar is deliberately narrow. Body lines are opaque and may
    contain strings that look like protocol headers.
    """
    sid = str(recipient_bot_id or "")
    if not sid:
        return None
    text = (content or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if len(lines) < 6:
        return None

    def h(line: str) -> str:
        return line.strip(" \t")

    mention = h(lines[0])
    if mention not in {f"<@{sid}>", f"<@!{sid}>"}:
        return None
    if h(lines[1]) != "BOT_MSG v1":
        return None

    expected_fields = ["reply_expected", "kind", "correlation_id"]
    values: Dict[str, str] = {}
    for idx, field in enumerate(expected_fields, start=2):
        line = h(lines[idx])
        prefix = f"{field}:"
        if not line.startswith(prefix):
            return None
        values[field] = line[len(prefix):].strip(" \t")

    if values["reply_expected"] not in {"true", "false"}:
        return None
    if values["kind"] not in DISCORD_BOT_MSG_KINDS:
        return None
    if not DISCORD_BOT_MSG_CORRELATION_RE.fullmatch(values["correlation_id"]):
        return None
    if h(lines[5]) != "---":
        return None

    body = "\n".join(lines[6:])
    if values["kind"] != "status" and body == "":
        return None

    return {
        "reply_expected": values["reply_expected"] == "true",
        "kind": values["kind"],
        "correlation_id": values["correlation_id"],
        "body": body,
    }


def parse_discord_bot_approval_decision_body(body: str) -> Optional[Dict[str, str]]:
    """Parse the narrow approval_decision BOT_MSG body grammar.

    Body format is line-oriented and deliberately not free-form:
      approval_id: <opaque live request id>
      decision: approve|deny
      scope: once|session|always        # optional for deny; defaults once
    """
    values: Dict[str, str] = {}
    for raw_line in (body or "").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key in {"approval_id", "decision", "scope"}:
            values[key] = value.strip().lower() if key != "approval_id" else value.strip()
    approval_id = values.get("approval_id", "")
    decision = values.get("decision", "")
    scope = values.get("scope", "once") or "once"
    if not approval_id:
        return None
    if decision not in {"approve", "deny"}:
        return None
    if scope not in {"once", "session", "always"}:
        return None
    return {"approval_id": approval_id, "decision": decision, "scope": scope}


def discord_content_mentioned_allowed_bots(content: str, allowed_bot_ids: set[str]) -> list[str]:
    """Return allowed bot IDs raw-mentioned in content, in content order."""
    if not content or not allowed_bot_ids:
        return []
    mentioned: list[str] = []
    for match in DISCORD_MENTION_RE.finditer(content):
        bot_id = match.group(1)
        if bot_id in allowed_bot_ids and bot_id not in mentioned:
            mentioned.append(bot_id)
    return mentioned


def discord_content_mentions_allowed_bot(content: str, allowed_bot_ids: set[str]) -> Optional[str]:
    mentioned = discord_content_mentioned_allowed_bots(content, allowed_bot_ids)
    return mentioned[0] if mentioned else None


def is_bot_msg_required_error(error: Any) -> bool:
    """Return True for terminal raw-allowed-bot-mention guard failures."""
    text = str(error or "")
    return bool(DISCORD_BOT_MSG_REQUIRED_ERROR_RE.search(text))


def is_discord_bot_routing_guard_error(error: Any) -> bool:
    """Return True for terminal Discord bot-routing final-response guard failures."""
    return bool(DISCORD_BOT_ROUTING_GUARD_ERROR_RE.search(str(error or "")))


def discord_bot_routing_guard_error() -> str:
    return "BOT_ROUTING_GUARD: blocked ordinary bot-to-bot final response"


def bot_msg_required_error(bot_id: str) -> str:
    return (
        f"Outbound raw mention of allowed bot {bot_id} requires "
        "send_bot_message(...) to create a BOT_MSG v1 envelope"
    )
