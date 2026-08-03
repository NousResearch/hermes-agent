"""Zulip message search tool — fetch and search message history.

Provides a single flexible tool that wraps Zulip's ``/messages`` API.
The agent can search by stream+topic, full-text query, message anchor,
and paginate through results — all through one interface.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_zulip_credentials(platform_config: Any = None) -> tuple[str, str, str]:
    """Return Zulip ``(site_url, bot_email, api_key)`` from env or config."""
    site_url = os.getenv("ZULIP_SITE_URL", "").rstrip("/")
    bot_email = os.getenv("ZULIP_BOT_EMAIL", "")
    api_key = os.getenv("ZULIP_API_KEY", "")

    if platform_config is None:
        try:
            from gateway.config import Platform, load_gateway_config

            platform_config = load_gateway_config().platforms.get(Platform("zulip"))
        except Exception:
            platform_config = None

    if platform_config:
        extra = platform_config.extra or {}
        site_url = site_url or str(extra.get("site_url") or "").rstrip("/")
        bot_email = bot_email or str(extra.get("bot_email") or "")
        api_key = api_key or (platform_config.token or platform_config.api_key or "")

    return site_url, bot_email, api_key


def _check_zulip_search_requirements() -> bool:
    """Check that the zulip_search_messages tool is usable.

    The tool is available on Zulip sessions (gateway context) or when
    Zulip credentials are explicitly configured.  Follows the same
    pattern as ``_check_send_message`` in ``send_message_tool.py``.
    """
    site_url, bot_email, api_key = _get_zulip_credentials()

    # 1. Session-context check (gateway-side: agent knows the platform).
    try:
        from gateway.session_context import get_session_env
        platform = get_session_env("HERMES_SESSION_PLATFORM", "")
        if platform == "zulip":
            return bool(site_url and bot_email and api_key)
    except Exception:
        pass

    # 2. Explicit Zulip credential check (env or config-backed usage).
    # Unlike send_message, this tool calls Zulip's API directly; a running
    # gateway process is neither required nor sufficient without credentials.
    if site_url and bot_email and api_key:
        return True

    return False


def _get_session_narrow() -> Optional[List[List[str]]]:
    """Build a narrow filter from the current Zulip session context.

    When the tool is invoked from within a Zulip gateway session (i.e. the
    agent is handling a message that arrived from Zulip), this restricts
    the search to the *current conversation only* — the stream+topic or DM
    that the user is talking to the bot in.  This prevents a user in a
    private DM from asking the bot to exfiltrate messages from streams or
    other DMs the bot is subscribed to.

    Returns ``None`` when the tool is called from CLI or other platforms,
    in which case the caller's own credentials/permissions apply.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return None

    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    if platform != "zulip":
        return None

    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    chat_name = get_session_env("HERMES_SESSION_CHAT_NAME", "")
    if not chat_id:
        return None

    # DM: "dm:alice@example.com"
    if chat_id.startswith("dm:") and "@" in chat_id:
        email = chat_id[3:]  # strip "dm:" prefix
        return [["pm-with", email]]

    # Group DM: "group_dm:alice@example.com,bob@example.com"
    if chat_id.startswith("group_dm:"):
        emails = chat_id[9:]  # strip "group_dm:" prefix
        if emails:
            return [["pm-with", emails]]

    # Stream: "{stream_id}:{topic}"
    colon = chat_id.find(":")
    if colon > 0:
        stream_part = chat_id[:colon]
        if stream_part.isdigit():
            topic = chat_id[colon + 1 :] or "(no topic)"
            stream_name = chat_name or ""
            if stream_name:
                return [["stream", stream_name], ["topic", topic]]

    return None


def zulip_search_messages(
    stream: Optional[str] = None,
    topic: Optional[str] = None,
    query: Optional[str] = None,
    anchor: Optional[str] = None,
    num_before: int = 20,
    num_after: int = 0,
    *,
    task_id: Optional[str] = None,
) -> str:
    """Search Zulip message history.

    Fetches messages from a Zulip organization using the bot's credentials.
    Supports narrowing by stream, topic, full-text search, and pagination
    via message ID anchors.

    **Security note:** When this tool is called from within a Zulip gateway
    session (i.e. the user is talking to the bot via Zulip), the search is
    automatically restricted to the *current conversation only*.  A user in
    a DM cannot ask the bot to search streams or other DMs.  When called
    from CLI or other platforms, the full search scope is available.

    Args:
        stream: Stream name to narrow to (e.g. ``"general"``). Optional.
                Ignored when called from a Zulip session (restricted to
                current conversation).
        topic: Topic name to narrow to (e.g. ``"database"``). Optional.
               Ignored when called from a Zulip session.
        query: Full-text search using Zulip's search syntax.
               Supports operators like ``sender:alice@example.com``,
               ``has:link``, ``is:starred``, ``near:<id>``, etc. Optional.
               ``stream:`` and ``pm-with:`` operators are stripped when
               called from a Zulip session to prevent scope escalation.
        anchor: Message ID to anchor around, or ``"newest"`` / ``"oldest"``.
                Defaults to ``"newest"`` (most recent messages).
        num_before: Number of messages to fetch before the anchor. Default 20.
        num_after: Number of messages to fetch after the anchor. Default 0.
        task_id: Internal task ID (injected by framework).

    Returns:
        A JSON string with search results including messages and pagination
        info (the oldest message ID for continued pagination).

    **Common usage patterns:**

    - Recent context: ``stream="general", topic="database", anchor="newest", num_before=20``
    - Around a specific message: ``anchor="<msg_id>", num_before=5, num_after=5``
    - Text search: ``stream="general", query="postgresql"``
    - Find by sender: ``query="sender:alice@example.com"``
    - Older page: ``stream="general", topic="db", anchor="<oldest_id>", num_before=20``
    """
    try:
        import zulip
    except ImportError:
        return json.dumps({"error": "zulip package not installed"})

    site_url, bot_email, api_key = _get_zulip_credentials()

    if not site_url or not bot_email or not api_key:
        return json.dumps({
            "error": "Zulip credentials not configured. "
                     "Set ZULIP_SITE_URL, ZULIP_BOT_EMAIL, and ZULIP_API_KEY."
        })

    # Build the narrow filter.
    narrow: List[List[str]] = []

    # When in a Zulip session, restrict to current conversation only.
    session_narrow = _get_session_narrow()
    if session_narrow is not None:
        narrow = list(session_narrow)
        # Sanitize query to prevent scope escalation via search operators.
        if query:
            import re
            sanitized = re.sub(r"\b(stream|pm-with):\S+", "", query).strip()
            if sanitized:
                narrow.append(["search", sanitized])
    else:
        # CLI / non-Zulip session — caller controls scope.
        if stream:
            narrow.append(["stream", stream])
        if topic:
            narrow.append(["topic", topic])
        if query:
            narrow.append(["search", query])

    # Resolve anchor.
    anchor_value: Any = anchor if anchor else "newest"

    client = zulip.Client(site=site_url, email=bot_email, api_key=api_key)
    try:
        result = client.get_messages({
            "anchor": anchor_value,
            "num_before": num_before,
            "num_after": num_after,
            "narrow": narrow or None,
            "apply_markdown": False,
        })
    except Exception as exc:
        logger.warning("Zulip search failed: %s", exc)
        return json.dumps({"error": f"Zulip API error: {exc}"})

    if result.get("result") != "success":
        return json.dumps({
            "error": result.get("msg", "Unknown Zulip error"),
        })

    messages = result.get("messages", [])
    if not messages:
        return json.dumps({
            "messages": [],
            "count": 0,
            "found_newest": result.get("found_newest", True),
            "found_oldest": result.get("found_oldest", True),
            "note": "No messages matched the search criteria.",
        })

    # Format messages for readability.
    formatted: List[Dict[str, Any]] = []
    for msg in messages:
        formatted.append({
            "id": msg.get("id"),
            "sender": msg.get("sender_full_name") or msg.get("sender_email", "?"),
            "timestamp": msg.get("timestamp", 0),
            "content": (msg.get("content") or "").strip(),
            "is_bot": msg.get("sender_email") == bot_email,
        })

    # Pagination cues.
    oldest_id = None
    newest_id = None
    if formatted:
        oldest_id = min(m["id"] for m in formatted if m["id"])
        newest_id = max(m["id"] for m in formatted if m["id"])

    return json.dumps({
        "messages": formatted,
        "count": len(formatted),
        "requested_before": num_before,
        "requested_after": num_after,
        "oldest_message_id": oldest_id,
        "newest_message_id": newest_id,
        "found_oldest": result.get("found_oldest", False),
        "found_newest": result.get("found_newest", False),
        "pagination_hint": (
            f"To get older messages, call again with "
            f"anchor={oldest_id}, num_before={num_before}, num_after=0. "
            f"To get newer messages, call with "
            f"anchor={newest_id}, num_before=0, num_after={num_after or 20}."
        ) if formatted else "",
    })


_ZULIP_SEARCH_SCHEMA = {
    "name": "zulip_search_messages",
    "description": (
        "Search Zulip message history. Fetches messages from streams, "
        "topics, or by full-text search. Supports pagination via "
        "message ID anchors. Use this to get context about what was "
        "discussed before your @mention, to search for specific "
        "information in past conversations, or to find messages "
        "by a specific sender."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "stream": {
                "type": "string",
                "description": (
                    "Stream name to narrow search to. "
                    "Example: 'general', 'engineering', 'announce'."
                ),
            },
            "topic": {
                "type": "string",
                "description": (
                    "Topic name within the stream. "
                    "Example: 'database', 'onboarding'."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Full-text search using Zulip's search syntax. "
                    "Supports operators like "
                    "sender:alice@example.com, has:link, is:starred, "
                    "near:12345, pm-with:alice@example.com. "
                    "Combine with stream/topic for focused search."
                ),
            },
            "anchor": {
                "type": "string",
                "description": (
                    "Message ID to anchor pagination around, or "
                    "'newest' (most recent) or 'oldest'. "
                    "Default: 'newest'. For pagination, use the "
                    "'oldest_message_id' from a previous response."
                ),
            },
            "num_before": {
                "type": "integer",
                "description": (
                    "Number of messages to fetch before the anchor. "
                    "Default: 20. Max: 5000."
                ),
                "default": 20,
            },
            "num_after": {
                "type": "integer",
                "description": (
                    "Number of messages to fetch after the anchor. "
                    "Default: 0. Set to >0 to see context after a "
                    "specific message (e.g., 5 messages after a reply)."
                ),
                "default": 0,
            },
        },
        "required": [],
    },
}


def _handle_zulip_search_messages(args, **kw):
    return zulip_search_messages(
        stream=args.get("stream"),
        topic=args.get("topic"),
        query=args.get("query"),
        anchor=args.get("anchor"),
        num_before=args.get("num_before", 20),
        num_after=args.get("num_after", 0),
        task_id=kw.get("task_id"),
    )


def register_zulip_search_tool(ctx) -> None:
    ctx.register_tool(
        name="zulip_search_messages",
        toolset="zulip-history",
        schema=_ZULIP_SEARCH_SCHEMA,
        handler=_handle_zulip_search_messages,
        check_fn=_check_zulip_search_requirements,
        emoji="🔎",
    )
