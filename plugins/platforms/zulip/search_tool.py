"""Zulip message search tool — fetch and search message history.

Provides a single flexible tool that wraps Zulip's ``/messages`` API.
The agent can search by stream+topic, full-text query, message anchor,
and paginate through results — all through one interface.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hermes outbound long replies are split by BasePlatformAdapter.truncate_message
# into multiple Zulip messages ending with " (i/n)" (see gateway/platforms/base.py).
# Zulip's 10k code-point cap means a single agent answer can become several
# consecutive bot messages; search must rejoin them or full-text queries for
# content that only appears past the first chunk look like a miss.
_CHUNK_MARKER_RE = re.compile(r" \((\d+)/(\d+)\)\s*$")

# Zulip FTS is updated asynchronously (process_fts_updates). Large bot replies
# can lag behind short messages in the index, and pure-numeric / marker tokens
# often fail to hit. When a text query is present we always do a client-side
# content scan of recent history as a safety net (in addition to server FTS).
_CLIENT_SCAN_WINDOW = 200
_CLIENT_SCAN_MAX_PAGES = 3
# Structural Zulip search operators — not usable as literal content needles.
_SEARCH_OPERATOR_RE = re.compile(
    r"\b(?:"
    r"sender|from|stream|channel|topic|pm-with|dm|dm-including|"
    r"has|is|near|id|group-id|streams|channels"
    r"):\S+",
    re.IGNORECASE,
)


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
    #
    # ``scope_narrow`` is structural only (stream/topic/DM) — used both for
    # server FTS and for the client-side content scan fallback.
    # ``fts_narrow`` adds Zulip's ``search`` operator when a text query is set.
    scope_narrow: List[List[str]] = []
    text_query = (query or "").strip() or None
    fts_text: Optional[str] = None

    # When in a Zulip session, restrict to current conversation only.
    session_narrow = _get_session_narrow()
    if session_narrow is not None:
        scope_narrow = list(session_narrow)
        # Sanitize query to prevent scope escalation via search operators.
        if text_query:
            sanitized = re.sub(r"\b(stream|pm-with):\S+", "", text_query).strip()
            fts_text = sanitized or None
    else:
        # CLI / non-Zulip session — caller controls scope.
        if stream:
            scope_narrow.append(["stream", stream])
        if topic:
            scope_narrow.append(["topic", topic])
        fts_text = text_query

    fts_narrow: List[List[str]] = list(scope_narrow)
    if fts_text:
        fts_narrow.append(["search", fts_text])

    # Resolve anchor.
    anchor_value: Any = anchor if anchor else "newest"

    client = zulip.Client(site=site_url, email=bot_email, api_key=api_key)
    try:
        result = client.get_messages({
            "anchor": anchor_value,
            "num_before": num_before,
            "num_after": num_after,
            "narrow": fts_narrow or None,
            "apply_markdown": False,
        })
    except Exception as exc:
        logger.warning("Zulip search failed: %s", exc)
        return json.dumps({"error": f"Zulip API error: {exc}"})

    if result.get("result") != "success":
        return json.dumps({
            "error": result.get("msg", "Unknown Zulip error"),
        })

    messages = list(result.get("messages") or [])
    used_client_scan = False
    found_oldest = result.get("found_oldest", False)
    found_newest = result.get("found_newest", False)

    # Client-side content scan: Zulip FTS indexes asynchronously and often
    # misses (or ranks below short noise) content inside large multi-chunk
    # Hermes replies. Scan recent history in the same scope and merge hits.
    content_needles = _content_needles_from_query(fts_text)
    if content_needles:
        scanned, scan_meta = _client_content_scan(
            client,
            scope_narrow=scope_narrow or None,
            needles=content_needles,
            anchor=anchor_value,
            bot_email=bot_email,
        )
        used_client_scan = bool(scanned) or bool(scan_meta.get("scanned"))
        if scanned:
            messages = _merge_messages_by_id(messages, scanned)
            # Prefer the scan window's pagination edges when we scanned.
            if scan_meta.get("found_oldest") is not None:
                found_oldest = scan_meta["found_oldest"]
            if scan_meta.get("found_newest") is not None:
                found_newest = scan_meta["found_newest"]

    if not messages:
        note = "No messages matched the search criteria."
        if content_needles and used_client_scan:
            note += (
                " Client-side scan of recent history also found no match "
                f"(window up to {_CLIENT_SCAN_WINDOW * _CLIENT_SCAN_MAX_PAGES} "
                "messages). Try a wider stream/topic scope or a later anchor."
            )
        return json.dumps({
            "messages": [],
            "count": 0,
            "found_newest": found_newest if found_newest is not None else True,
            "found_oldest": found_oldest if found_oldest is not None else True,
            "note": note,
            "client_content_scan": used_client_scan,
        })

    # Format messages for readability, then rejoin Hermes multi-chunk replies
    # so content past the 10k Zulip cap is searchable/readable as one body.
    formatted: List[Dict[str, Any]] = [
        _format_raw_message(msg, bot_email) for msg in messages
    ]

    # Full-text search often returns only the matching chunk of a long
    # Hermes reply. Expand isolated chunk hits so we can rejoin the series.
    # Expand against scope_narrow (no search op) so siblings are fetchable.
    formatted = _expand_partial_hermes_chunks(
        client,
        formatted,
        narrow=scope_narrow or None,
        bot_email=bot_email,
    )
    formatted = _reassemble_hermes_chunks(formatted)

    # After reassembly, drop messages that still don't contain any needle
    # when the caller asked for a text query (avoids FTS false friends that
    # only matched a short plan/approval mentioning the marker).
    if content_needles:
        filtered = [
            m for m in formatted
            if _content_matches_needles(m.get("content") or "", content_needles)
        ]
        # Keep structural-only results if filtering wiped everything that
        # FTS returned but scan already injected real hits — otherwise keep
        # filtered list (may be empty → honest miss).
        if filtered or used_client_scan:
            formatted = filtered

    # Drop internal sender_email used only for chunk grouping.
    for entry in formatted:
        entry.pop("sender_email", None)

    # Pagination cues.
    oldest_id = None
    newest_id = None
    if formatted:
        oldest_id = min(m["id"] for m in formatted if m["id"])
        newest_id = max(m["id"] for m in formatted if m["id"])

    payload: Dict[str, Any] = {
        "messages": formatted,
        "count": len(formatted),
        "requested_before": num_before,
        "requested_after": num_after,
        "oldest_message_id": oldest_id,
        "newest_message_id": newest_id,
        "found_oldest": found_oldest,
        "found_newest": found_newest,
        "pagination_hint": (
            f"To get older messages, call again with "
            f"anchor={oldest_id}, num_before={num_before}, num_after=0. "
            f"To get newer messages, call with "
            f"anchor={newest_id}, num_before=0, num_after={num_after or 20}."
        ) if formatted else "",
    }
    if used_client_scan:
        payload["client_content_scan"] = True
        payload["note"] = (
            "Results include a client-side content scan of recent history. "
            "Zulip server full-text search can lag or miss long multi-chunk "
            "bot replies; the scan matches literal text in message bodies."
        )
    if not formatted:
        payload["note"] = (
            "No messages matched the search criteria after content filtering."
        )
    return json.dumps(payload)


def _parse_chunk_marker(content: str) -> Optional[tuple[str, int, int]]:
    """Return ``(body, index, total)`` when *content* ends with `` (i/n)``."""
    if not content:
        return None
    match = _CHUNK_MARKER_RE.search(content)
    if not match:
        return None
    index = int(match.group(1))
    total = int(match.group(2))
    if index < 1 or total < 2 or index > total:
        return None
    body = content[: match.start()]
    return body, index, total


def _content_needles_from_query(query: Optional[str]) -> List[str]:
    """Extract literal content needles from a Zulip search query string.

    Strips structural operators (``sender:``, ``has:``, …) and quoted phrases
    become exact needles. Remaining whitespace-separated tokens become
    case-insensitive substrings. Pure operator queries yield no needles
    (client scan is skipped — FTS alone is appropriate).
    """
    if not query or not str(query).strip():
        return []
    text = str(query).strip()
    needles: List[str] = []

    # Quoted phrases first (preserve order, allow spaces).
    for match in re.finditer(r'"([^"]+)"', text):
        phrase = match.group(1).strip()
        if phrase:
            needles.append(phrase)
    text_no_quotes = re.sub(r'"[^"]*"', " ", text)

    # Drop Zulip operators; keep the rest as free-text tokens.
    text_no_ops = _SEARCH_OPERATOR_RE.sub(" ", text_no_quotes)
    for token in text_no_ops.split():
        token = token.strip()
        if not token:
            continue
        # Skip lone boolean-ish noise.
        if token.lower() in {"and", "or", "not", "-"}:
            continue
        needles.append(token)

    # De-dupe while preserving order (case-insensitive key).
    seen: set[str] = set()
    unique: List[str] = []
    for n in needles:
        key = n.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(n)
    return unique


def _content_matches_needles(content: str, needles: List[str]) -> bool:
    """True when *content* contains every needle (case-insensitive substring)."""
    if not needles:
        return True
    haystack = content.casefold()
    return all(n.casefold() in haystack for n in needles)


def _merge_messages_by_id(
    primary: List[Dict[str, Any]],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union raw Zulip message dicts by id (prefer *primary* on conflict)."""
    by_id: Dict[Any, Dict[str, Any]] = {}
    for msg in extra:
        mid = msg.get("id")
        if mid is not None:
            by_id[mid] = msg
    for msg in primary:
        mid = msg.get("id")
        if mid is not None:
            by_id[mid] = msg
    return list(by_id.values())


def _client_content_scan(
    client: Any,
    *,
    scope_narrow: Optional[List[List[str]]],
    needles: List[str],
    anchor: Any,
    bot_email: str,
    window: int = _CLIENT_SCAN_WINDOW,
    max_pages: int = _CLIENT_SCAN_MAX_PAGES,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch recent history and keep messages whose body contains *needles*.

    Bypasses Zulip FTS entirely — used when server search lags or fails to
    index large multi-chunk bot replies. Returns ``(raw_messages, meta)``.
    """
    meta: Dict[str, Any] = {"scanned": False, "pages": 0}
    if not needles:
        return [], meta

    hits: List[Dict[str, Any]] = []
    seen_ids: set[Any] = set()
    page_anchor: Any = anchor if anchor not in (None, "") else "newest"
    found_oldest: Optional[bool] = None
    found_newest: Optional[bool] = None

    for _page in range(max_pages):
        try:
            result = client.get_messages({
                "anchor": page_anchor,
                "num_before": window,
                "num_after": 0,
                "narrow": scope_narrow,
                "apply_markdown": False,
            })
        except Exception as exc:
            logger.debug("Zulip client content scan failed: %s", exc)
            break
        if result.get("result") != "success":
            break
        meta["scanned"] = True
        meta["pages"] = int(meta["pages"]) + 1
        if result.get("found_oldest") is not None:
            found_oldest = bool(result.get("found_oldest"))
        if result.get("found_newest") is not None:
            found_newest = bool(result.get("found_newest"))

        page_msgs = list(result.get("messages") or [])
        if not page_msgs:
            break

        for raw in page_msgs:
            mid = raw.get("id")
            if mid is None or mid in seen_ids:
                continue
            seen_ids.add(mid)
            content = (raw.get("content") or "")
            if _content_matches_needles(content, needles):
                hits.append(raw)

        # Page older when we still need more coverage.
        oldest = min(
            (m.get("id") for m in page_msgs if m.get("id") is not None),
            default=None,
        )
        if oldest is None or result.get("found_oldest"):
            break
        # Stop early once we have hits — one window of neighbors is enough
        # for expand/reassemble to finish the job on long multi-chunk replies.
        if hits:
            break
        page_anchor = oldest

    meta["found_oldest"] = found_oldest
    meta["found_newest"] = found_newest
    # bot_email unused for filtering but kept for API symmetry / future flags.
    _ = bot_email
    return hits, meta


def _format_raw_message(msg: Dict[str, Any], bot_email: str) -> Dict[str, Any]:
    return {
        "id": msg.get("id"),
        "sender": msg.get("sender_full_name") or msg.get("sender_email", "?"),
        "sender_email": msg.get("sender_email") or "",
        "timestamp": msg.get("timestamp", 0),
        "content": (msg.get("content") or "").strip(),
        "is_bot": msg.get("sender_email") == bot_email,
    }


def _expand_partial_hermes_chunks(
    client: Any,
    messages: List[Dict[str, Any]],
    *,
    narrow: Optional[List[List[str]]],
    bot_email: str = "",
    max_expansions: int = 5,
) -> List[Dict[str, Any]]:
    """Fetch sibling chunks when the window only contains part of a series.

    A full-text ``query`` hit often returns only the matching chunk of a
    multi-message Hermes reply.  Pull a tight window around that message so
    ``_reassemble_hermes_chunks`` can join the full body.  Caps expansions to
    avoid hammering the API when many partial hits appear.
    """
    if not messages:
        return messages

    by_id: Dict[Any, Dict[str, Any]] = {
        m["id"]: m for m in messages if m.get("id") is not None
    }
    expansions = 0

    # Snapshot ids up front — we mutate by_id as we expand.
    candidates = list(by_id.values())
    for msg in candidates:
        if expansions >= max_expansions:
            break
        parsed = _parse_chunk_marker(msg.get("content") or "")
        if parsed is None:
            continue
        _, index, total = parsed
        # Already have a complete run nearby? Skip.
        sender = msg.get("sender_email") or msg.get("sender") or ""
        have = 0
        for other in by_id.values():
            other_sender = other.get("sender_email") or other.get("sender") or ""
            if other_sender != sender:
                continue
            other_parsed = _parse_chunk_marker(other.get("content") or "")
            if other_parsed and other_parsed[2] == total:
                have += 1
        if have >= total:
            continue

        msg_id = msg.get("id")
        if msg_id is None:
            continue
        try:
            # Fetch enough neighbors to cover the full series around this hit.
            num_before = max(0, index - 1)
            num_after = max(0, total - index)
            # Small pad for interleaved non-bot messages in the same topic.
            num_before = min(num_before + 2, 20)
            num_after = min(num_after + 2, 20)
            result = client.get_messages({
                "anchor": msg_id,
                "num_before": num_before,
                "num_after": num_after,
                "narrow": narrow,
                "apply_markdown": False,
            })
        except Exception as exc:
            logger.debug("Zulip chunk expand failed for %s: %s", msg_id, exc)
            continue
        if result.get("result") != "success":
            continue
        expansions += 1
        for raw in result.get("messages") or []:
            rid = raw.get("id")
            if rid is None or rid in by_id:
                continue
            by_id[rid] = _format_raw_message(raw, bot_email)

    return list(by_id.values())


def _reassemble_hermes_chunks(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge consecutive Hermes multi-chunk replies into single messages.

    ``truncate_message`` emits N consecutive messages from the same sender,
    each ending with `` (i/n)``.  Zulip stores them as independent messages, so
    a full-text search hit on chunk 2 alone returns only a fragment.  When a
    complete ``1..n`` run is present in the result window we collapse it into
    one entry with the concatenated body and ``chunk_ids`` for traceability.
    Incomplete runs (pagination window cut mid-series) are left as-is so the
    caller can widen the window.
    """
    if len(messages) < 2:
        return messages

    # Work in chronological order (oldest first) so chunk 1 precedes chunk n.
    ordered = sorted(
        messages,
        key=lambda m: (
            m.get("id") is None,
            m.get("id") if m.get("id") is not None else 0,
            m.get("timestamp") or 0,
        ),
    )

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(ordered):
        msg = ordered[i]
        parsed = _parse_chunk_marker(msg.get("content") or "")
        if parsed is None or parsed[1] != 1:
            out.append(msg)
            i += 1
            continue

        body, index, total = parsed
        sender_key = (
            msg.get("sender_email")
            or msg.get("sender")
            or ""
        )
        group = [(msg, body, index)]
        j = i + 1
        expected = 2
        while j < len(ordered) and expected <= total:
            nxt = ordered[j]
            nxt_key = nxt.get("sender_email") or nxt.get("sender") or ""
            if nxt_key != sender_key:
                break
            nxt_parsed = _parse_chunk_marker(nxt.get("content") or "")
            if nxt_parsed is None:
                break
            nxt_body, nxt_index, nxt_total = nxt_parsed
            if nxt_total != total or nxt_index != expected:
                break
            group.append((nxt, nxt_body, nxt_index))
            expected += 1
            j += 1

        if len(group) != total:
            # Incomplete series — keep every collected chunk as-is so the
            # agent can widen the pagination window and retry.
            for piece, _, _ in group:
                out.append(piece)
            i = j
            continue

        combined_content = "\n".join(part_body for _, part_body, _ in group)
        first = group[0][0]
        last = group[-1][0]
        merged = {
            "id": first.get("id"),
            "sender": first.get("sender"),
            "sender_email": first.get("sender_email", ""),
            "timestamp": first.get("timestamp", 0),
            "content": combined_content,
            "is_bot": first.get("is_bot", False),
            "chunk_ids": [g[0].get("id") for g in group],
            "chunk_count": total,
            "newest_chunk_id": last.get("id"),
        }
        out.append(merged)
        i = j

    # Oldest-first by message id — clearer for agents reconstructing long
    # multi-chunk answers. Pagination still exposes oldest/newest ids.
    out.sort(
        key=lambda m: (
            m.get("id") is None,
            m.get("id") if m.get("id") is not None else 0,
        )
    )
    return out


_ZULIP_SEARCH_SCHEMA = {
    "name": "zulip_search_messages",
    "description": (
        "Search Zulip message history. Fetches messages from streams, "
        "topics, or by full-text search. Supports pagination via "
        "message ID anchors. Long Hermes replies that were split across "
        "multiple Zulip messages (over the ~10k limit) are rejoined into "
        "one result when the full chunk series is in the window. Use this "
        "to get context about what was discussed before your @mention, to "
        "search for specific information in past conversations, or to find "
        "messages by a specific sender."
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
