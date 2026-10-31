"""Conservative recognition of automatic Slack status and heartbeat roster footers.

Call only after Slack's ``allow_bots=mentions`` gate has established that the sender is a
peer bot and the current event contains a target mention. This helper never rewrites or
returns replacement message text: it decides only whether the event is an unaddressed,
structured heartbeat or status/ack notice whose final standalone line is an agent roster.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Optional


_ROSTER_ID = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>\s]*)?>\Z")
_SELF_MENTION = re.compile(r"<@([UW][A-Z0-9]+)(?:\|[^>\s]*)?>")
_ROOM_MENTION = re.compile(r"<!\s*(?:here|channel|everyone)(?:\|[^>]*)?>|(?<!\w)@(here|channel|everyone)\b", re.IGNORECASE)
_QUESTION_START = re.compile(
    r"(?im)^\s*(?:who|what|when|where|why|how|can|could|would|will|should|do|does|did|is|are)\b")
_HEARTBEAT_FIELDS = ("Goal", "Plan", "Alarm", "Trigger", "Loop")
_MUSE_PREFIX = re.compile(r"^\s*(?:\[MUSE\]\s*)?", re.IGNORECASE)
_STATUS_NOTICE = re.compile(r"\b(?:status|ack(?:nowledg(?:e)?ment)?|received|updated|update)\b", re.IGNORECASE)
_INLINE_SIGNATURE = re.compile(r"\s+[—–-]\s*[A-Za-z][A-Za-z0-9 ._'’-]{0,63}\s*$")


class _UnrecognizedSlackBlock(ValueError):
    """An unparsed block could contain addressing data, so roster suppression must fail open."""


def _render_rich_text(node: object) -> str:
    """Render known native Slack rich-text elements; reject shapes this detector cannot inspect."""
    if isinstance(node, list):
        return "".join(_render_rich_text(item) for item in node)
    if not isinstance(node, Mapping):
        raise _UnrecognizedSlackBlock
    kind = node.get("type")
    if kind == "text":
        value = node.get("text")
        if not isinstance(value, str):
            raise _UnrecognizedSlackBlock
        return value
    if kind == "user":
        user_id = node.get("user_id")
        if not isinstance(user_id, str) or not re.fullmatch(r"[UW][A-Z0-9]+", user_id):
            raise _UnrecognizedSlackBlock
        return f"<@{user_id}>"
    if kind == "broadcast":
        value = node.get("range")
        if value not in {"here", "channel", "everyone"}:
            raise _UnrecognizedSlackBlock
        return f"<!{value}>"
    if kind == "channel":
        channel_id = node.get("channel_id")
        if not isinstance(channel_id, str):
            raise _UnrecognizedSlackBlock
        return f"<#{channel_id}>"
    if kind == "emoji":
        name = node.get("name")
        if not isinstance(name, str):
            raise _UnrecognizedSlackBlock
        return f":{name}:"
    if kind in {"rich_text_section", "rich_text_preformatted", "rich_text_quote"}:
        elements = node.get("elements")
        if not isinstance(elements, list):
            raise _UnrecognizedSlackBlock
        return _render_rich_text(elements)
    if kind == "rich_text_list":
        elements = node.get("elements")
        if not isinstance(elements, list):
            raise _UnrecognizedSlackBlock
        # List items are separate lines in Slack's rendered message.
        return "\n".join(_render_rich_text(item) for item in elements)
    raise _UnrecognizedSlackBlock


def _event_surfaces(event: Mapping[str, object]) -> Optional[list[str]]:
    """Return inspected text projections, or None when a relevant representation is unknown."""
    text = event.get("text", "")
    if not isinstance(text, str):
        return None
    # Attachments/files can carry fallback text or mentions the native rich-text walker cannot see.
    if event.get("attachments") or event.get("files"):
        return None
    surfaces = [text] if text.strip() else []
    blocks = event.get("blocks")
    if blocks:
        if not isinstance(blocks, list):
            return None
        try:
            rendered = []
            for block in blocks:
                if not isinstance(block, Mapping) or block.get("type") != "rich_text":
                    return None
                elements = block.get("elements")
                if not isinstance(elements, list):
                    return None
                rendered.append(_render_rich_text(elements))
        except _UnrecognizedSlackBlock:
            return None
        block_text = "\n".join(rendered)
        if block_text.strip():
            surfaces.append(block_text)
    return surfaces or None


def _split_footer(surface: str) -> Optional[tuple[str, list[str]]]:
    """Split a final standalone roster with at least 15 distinct Slack user IDs."""
    lines = surface.splitlines()
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    if end and re.fullmatch(r"[—–-]\s*[A-Za-z][A-Za-z0-9 ._'’-]{0,63}", lines[end - 1].strip()):
        end -= 1
        while end and not lines[end - 1].strip():
            end -= 1
    if not end:
        return None
    footer = lines[end - 1].strip()
    footer = re.sub(r"^Agent roster:\s*", "", footer, flags=re.IGNORECASE)
    footer = _INLINE_SIGNATURE.sub("", footer).strip()
    tokens = footer.split()
    matches = [_ROSTER_ID.fullmatch(token) for token in tokens]
    if len(tokens) < 15 or any(match is None for match in matches):
        return None
    ids = [match.group(1) for match in matches if match is not None]
    if len(set(ids)) < 15:
        return None
    return "\n".join(lines[:end - 1]), ids


def _has_heartbeat_fields(body: str) -> bool:
    """Require a compact, ordered five-field heartbeat, not merely a message mentioning those words."""
    found: list[str] = []
    for line in body.splitlines():
        value = line.strip()
        match = re.match(r"^\*{0,2}(Goal|Plan|Alarm|Trigger|Loop)\*{0,2}(.*)$", value, re.IGNORECASE)
        if not match:
            continue
        label, suffix = match.group(1).title(), match.group(2).strip()
        if label == "Plan":
            # Native heartbeat template uses a bold standalone Plan heading; colon/value form is
            # accepted too. A colonless Plan line with trailing prose is not a structural heading.
            if suffix and not suffix.startswith(":"):
                return False
        elif not suffix.startswith(":"):
            return False
        found.append(label)
    return found == list(_HEARTBEAT_FIELDS)


def _has_heartbeat_marker(body: str) -> bool:
    """Require the explicit automatic-heartbeat marker on the first meaningful line."""
    first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
    first_line = _MUSE_PREFIX.sub("", first_line, count=1)
    if ":heartbeat:" in first_line.casefold():
        return True
    # Slack may render the native heart emoji directly; the accompanying word disambiguates
    # it from ordinary prose or a decorative emoji elsewhere in the message.
    return bool(re.search(r"(?:❤️|💓).*\bheartbeat\b", first_line, re.IGNORECASE))


def is_roster_only_automatic_heartbeat(
    event: Mapping[str, object],
    bot_user_id: str,
    *,
    has_wake_match: Optional[Callable[[str], bool]] = None,
) -> bool:
    """True only for an unaddressed heartbeat or status/ack notice with a large roster.

    The caller is responsible for restricting this check to a resolved peer bot in the
    existing ``allow_bots=mentions`` branch. Unknown Block Kit/event representations return
    False so they remain eligible under existing routing.
    """
    if not isinstance(event, Mapping) or not re.fullmatch(r"[UW][A-Z0-9]+", bot_user_id or ""):
        return False
    if event.get("attachments") or event.get("files"):
        return False
    surfaces = _event_surfaces(event)
    if not surfaces:
        return False
    splits = [_split_footer(surface) for surface in surfaces]
    if not splits or any(split is None for split in splits):
        return False
    # All available representations must agree that the target mention belongs to a roster
    # footer. Any question, room ping, self mention, or configured wake pattern in the body
    # makes this a live request and therefore fails open.
    for surface, split in zip(surfaces, splits):
        assert split is not None
        body, footer_ids = split
        is_structured_heartbeat = _has_heartbeat_marker(body) and _has_heartbeat_fields(body)
        if bot_user_id not in footer_ids or not (is_structured_heartbeat or _STATUS_NOTICE.search(body)):
            return False
        if "?" in body or _QUESTION_START.search(body) or _ROOM_MENTION.search(body):
            return False
        if any(match.group(1) == bot_user_id for match in _SELF_MENTION.finditer(body)):
            return False
        if has_wake_match is not None and has_wake_match(body):
            return False
    # Every rendered representation must agree that this is an undirected heartbeat/status notice.
    return True


async def should_drop_peer_bot(
    adapter, event: dict, user_id: str, bot_uid: Optional[str], channel_id: str,
    team_id: str, is_mentioned: bool,
) -> bool:
    """Preserve the peer-bot gate, excluding proved automatic roster-only heartbeats."""
    if not user_id or user_id == bot_uid:
        return False
    sender_is_bot = adapter._event_declares_bot_sender(event)
    if not sender_is_bot:
        sender_is_bot = await adapter._resolve_user_is_bot(
            user_id, chat_id=channel_id, team_id=team_id)
    if not sender_is_bot:
        return False
    allow_bots = adapter._slack_allow_bots()
    return allow_bots == "none" or (allow_bots == "mentions" and (
        not is_mentioned or is_roster_only_automatic_heartbeat(
            event, bot_uid, has_wake_match=adapter._slack_message_matches_mention_patterns)))
