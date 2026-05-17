"""
Feishu/Lark drive document comment handling.

Processes ``drive.notice.comment_add_v1`` events and interacts with the
Drive v2 comment reaction API.  Kept in a separate module so that the
main ``feishu.py`` adapter does not grow further and comment-related
logic can evolve independently.

Flow:
  1. Parse event -> extract file_token, comment_id, reply_id, etc.
  2. Add OK reaction
  3. Parallel fetch: doc meta + comment details (batch_query)
  4. Branch on is_whole:
       Whole -> list whole comments timeline
       Local -> list comment thread replies
  5. Build prompt (local or whole)
  6. Create AIAgent with feishu_doc + feishu_drive tools -> agent generates reply
  7. Route reply:
       Whole -> add_whole_comment
       Local -> reply_to_comment (fallback to add_whole_comment on 1069302)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lark SDK helpers (lazy-imported)
# ---------------------------------------------------------------------------


def _build_request(method: str, uri: str, paths=None, queries=None, body=None):
    """Build a lark_oapi BaseRequest."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    http_method = HttpMethod.GET if method == "GET" else HttpMethod.POST

    builder = (
        BaseRequest.builder()
        .http_method(http_method)
        .uri(uri)
        .token_types({AccessTokenType.TENANT})
    )
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)
    return builder.build()


async def _exec_request(client, method, uri, paths=None, queries=None, body=None):
    """Execute a lark API request and return (code, msg, data_dict)."""
    logger.info("[Feishu-Comment] API >>> %s %s paths=%s queries=%s body=%s",
                 method, uri, paths, queries,
                 json.dumps(body, ensure_ascii=False)[:500] if body else None)
    request = _build_request(method, uri, paths, queries, body)
    response = await asyncio.to_thread(client.request, request)

    code = getattr(response, "code", None)
    msg = getattr(response, "msg", "")

    data: dict = {}
    raw = getattr(response, "raw", None)
    if raw and hasattr(raw, "content"):
        try:
            body_json = json.loads(raw.content)
            data = body_json.get("data", {})
        except (json.JSONDecodeError, AttributeError):
            pass
    if not data:
        resp_data = getattr(response, "data", None)
        if isinstance(resp_data, dict):
            data = resp_data
        elif resp_data and hasattr(resp_data, "__dict__"):
            data = vars(resp_data)

    logger.info("[Feishu-Comment] API <<< %s %s code=%s msg=%s data_keys=%s",
                 method, uri, code, msg, list(data.keys()) if data else "empty")
    if code != 0:
        # Log raw response for debugging failed API calls
        raw = getattr(response, "raw", None)
        raw_content = ""
        if raw and hasattr(raw, "content"):
            raw_content = raw.content[:500] if isinstance(raw.content, (str, bytes)) else str(raw.content)[:500]
        logger.warning("[Feishu-Comment] API FAIL raw response: %s", raw_content)
    return code, msg, data


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def parse_drive_comment_event(data: Any) -> Optional[Dict[str, Any]]:
    """Extract structured fields from a ``drive.notice.comment_add_v1`` payload.

    *data* may be a ``CustomizedEvent`` (WebSocket) whose ``.event`` is a dict,
    or a ``SimpleNamespace`` (Webhook) built from the full JSON body.

    Returns a flat dict with the relevant fields, or ``None`` when the
    payload is malformed.
    """
    logger.debug("[Feishu-Comment] parse_drive_comment_event: data type=%s", type(data).__name__)
    event = getattr(data, "event", None)
    if event is None:
        logger.debug("[Feishu-Comment] parse_drive_comment_event: no .event attribute, returning None")
        return None

    evt: dict = event if isinstance(event, dict) else (
        vars(event) if hasattr(event, "__dict__") else {}
    )
    logger.debug("[Feishu-Comment] parse_drive_comment_event: evt keys=%s", list(evt.keys()))

    notice_meta = evt.get("notice_meta") or {}
    if not isinstance(notice_meta, dict):
        notice_meta = vars(notice_meta) if hasattr(notice_meta, "__dict__") else {}

    from_user = notice_meta.get("from_user_id") or {}
    if not isinstance(from_user, dict):
        from_user = vars(from_user) if hasattr(from_user, "__dict__") else {}

    to_user = notice_meta.get("to_user_id") or {}
    if not isinstance(to_user, dict):
        to_user = vars(to_user) if hasattr(to_user, "__dict__") else {}

    return {
        "event_id": str(evt.get("event_id") or ""),
        "comment_id": str(evt.get("comment_id") or ""),
        "reply_id": str(evt.get("reply_id") or ""),
        "is_mentioned": bool(evt.get("is_mentioned")),
        "timestamp": str(evt.get("timestamp") or ""),
        "file_token": str(notice_meta.get("file_token") or ""),
        "file_type": str(notice_meta.get("file_type") or ""),
        "notice_type": str(notice_meta.get("notice_type") or ""),
        "from_open_id": str(from_user.get("open_id") or ""),
        "to_open_id": str(to_user.get("open_id") or ""),
    }


# ---------------------------------------------------------------------------
# Comment reaction API
# ---------------------------------------------------------------------------

_REACTION_URI = "/open-apis/drive/v2/files/:file_token/comments/reaction"


async def add_comment_reaction(
    client: Any,
    *,
    file_token: str,
    file_type: str,
    reply_id: str,
    reaction_type: str = "OK",
) -> bool:
    """Add an emoji reaction to a document comment reply.

    Uses the Drive v2 ``update_reaction`` endpoint::

        POST /open-apis/drive/v2/files/{file_token}/comments/reaction?file_type=...

    Returns ``True`` on success, ``False`` on failure (errors are logged).
    """
    try:
        from lark_oapi import AccessTokenType  # noqa: F401
    except ImportError:
        logger.error("[Feishu-Comment] lark_oapi not available")
        return False

    body = {
        "action": "add",
        "reply_id": reply_id,
        "reaction_type": reaction_type,
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REACTION_URI,
        paths={"file_token": file_token},
        queries=[("file_type", file_type)],
        body=body,
    )

    succeeded = code == 0
    if succeeded:
        logger.info(
            "[Feishu-Comment] Reaction '%s' added: file=%s:%s reply=%s",
            reaction_type, file_type, file_token, reply_id,
        )
    else:
        logger.warning(
            "[Feishu-Comment] Reaction API failed: code=%s msg=%s "
            "file=%s:%s reply=%s",
            code, msg, file_type, file_token, reply_id,
        )
    return succeeded


async def delete_comment_reaction(
    client: Any,
    *,
    file_token: str,
    file_type: str,
    reply_id: str,
    reaction_type: str = "OK",
) -> bool:
    """Remove an emoji reaction from a document comment reply.

    Best-effort — errors are logged but not raised.
    """
    body = {
        "action": "delete",
        "reply_id": reply_id,
        "reaction_type": reaction_type,
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REACTION_URI,
        paths={"file_token": file_token},
        queries=[("file_type", file_type)],
        body=body,
    )

    succeeded = code == 0
    if succeeded:
        logger.info(
            "[Feishu-Comment] Reaction '%s' deleted: file=%s:%s reply=%s",
            reaction_type, file_type, file_token, reply_id,
        )
    else:
        logger.warning(
            "[Feishu-Comment] Reaction API failed: code=%s msg=%s "
            "file=%s:%s reply=%s",
            code, msg, file_type, file_token, reply_id,
        )
    return succeeded


# ---------------------------------------------------------------------------
# API call layer
# ---------------------------------------------------------------------------

_BATCH_QUERY_META_URI = "/open-apis/drive/v1/metas/batch_query"
_BATCH_QUERY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/batch_query"
_LIST_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_LIST_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_REPLY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"


async def query_document_meta(
    client: Any, file_token: str, file_type: str,
) -> Dict[str, Any]:
    """Fetch document title and URL via batch_query meta API.

    Returns ``{"title": "...", "url": "...", "doc_type": "..."}`` or empty dict.
    """
    body = {
        "request_docs": [{"doc_token": file_token, "doc_type": file_type}],
        "with_url": True,
    }
    logger.debug("[Feishu-Comment] query_document_meta: file_token=%s file_type=%s", file_token, file_type)
    code, msg, data = await _exec_request(
        client, "POST", _BATCH_QUERY_META_URI, body=body,
    )
    if code != 0:
        logger.warning("[Feishu-Comment] Meta batch_query failed: code=%s msg=%s", code, msg)
        return {}

    metas = data.get("metas", [])
    logger.debug("[Feishu-Comment] query_document_meta: raw metas type=%s value=%s",
                 type(metas).__name__, str(metas)[:300])
    if not metas:
        # Try alternate response shape: metas may be a dict keyed by token
        if isinstance(data.get("metas"), dict):
            meta = data["metas"].get(file_token, {})
        else:
            logger.debug("[Feishu-Comment] query_document_meta: no metas found")
            return {}
    else:
        meta = metas[0] if isinstance(metas, list) else {}

    result = {
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "doc_type": meta.get("doc_type", file_type),
    }
    logger.info("[Feishu-Comment] query_document_meta: title=%s url=%s",
                result["title"], result["url"][:80] if result["url"] else "")
    return result


_COMMENT_RETRY_LIMIT = 6
_COMMENT_RETRY_DELAY_S = 1.0


async def batch_query_comment(
    client: Any, file_token: str, file_type: str, comment_id: str,
) -> Dict[str, Any]:
    """Fetch comment details via batch_query comment API.

    Retries up to 6 times on failure (handles eventual consistency).

    Returns the comment dict with fields like ``is_whole``, ``quote``,
    ``reply_list``, etc.  Empty dict on failure.
    """
    logger.debug("[Feishu-Comment] batch_query_comment: file_token=%s comment_id=%s", file_token, comment_id)

    for attempt in range(_COMMENT_RETRY_LIMIT):
        code, msg, data = await _exec_request(
            client, "POST", _BATCH_QUERY_COMMENT_URI,
            paths={"file_token": file_token},
            queries=[
                ("file_type", file_type),
                ("user_id_type", "open_id"),
            ],
            body={"comment_ids": [comment_id]},
        )
        if code == 0:
            break
        if attempt < _COMMENT_RETRY_LIMIT - 1:
            logger.info(
                "[Feishu-Comment] batch_query_comment retry %d/%d: code=%s msg=%s",
                attempt + 1, _COMMENT_RETRY_LIMIT, code, msg,
            )
            await asyncio.sleep(_COMMENT_RETRY_DELAY_S)
        else:
            logger.warning(
                "[Feishu-Comment] batch_query_comment failed after %d attempts: code=%s msg=%s",
                _COMMENT_RETRY_LIMIT, code, msg,
            )
            return {}

    # Response: {"items": [{"comment_id": "...", ...}]}
    items = data.get("items", [])
    logger.debug("[Feishu-Comment] batch_query_comment: got %d items", len(items) if isinstance(items, list) else 0)
    if items and isinstance(items, list):
        item = items[0]
        logger.info("[Feishu-Comment] batch_query_comment: is_whole=%s quote=%s reply_count=%s",
                    item.get("is_whole"),
                    (item.get("quote", "") or "")[:60],
                    len(item.get("reply_list", {}).get("replies", [])) if isinstance(item.get("reply_list"), dict) else "?")
        return item
    logger.warning("[Feishu-Comment] batch_query_comment: empty items, raw data keys=%s", list(data.keys()))
    return {}


async def list_whole_comments(
    client: Any, file_token: str, file_type: str,
) -> List[Dict[str, Any]]:
    """List all whole-document comments (paginated, up to 500)."""
    logger.debug("[Feishu-Comment] list_whole_comments: file_token=%s", file_token)
    all_comments: List[Dict[str, Any]] = []
    page_token = ""

    for _ in range(5):  # max 5 pages
        queries = [
            ("file_type", file_type),
            ("is_whole", "true"),
            ("page_size", "100"),
            ("user_id_type", "open_id"),
        ]
        if page_token:
            queries.append(("page_token", page_token))

        code, msg, data = await _exec_request(
            client, "GET", _LIST_COMMENTS_URI,
            paths={"file_token": file_token},
            queries=queries,
        )
        if code != 0:
            logger.warning("[Feishu-Comment] List whole comments failed: code=%s msg=%s", code, msg)
            break

        items = data.get("items", [])
        if isinstance(items, list):
            all_comments.extend(items)
            logger.debug("[Feishu-Comment] list_whole_comments: page got %d items, total=%d",
                         len(items), len(all_comments))

        if not data.get("has_more"):
            break
        page_token = data.get("page_token", "")
        if not page_token:
            break

    logger.info("[Feishu-Comment] list_whole_comments: total %d whole comments fetched", len(all_comments))
    return all_comments


async def list_comment_replies(
    client: Any, file_token: str, file_type: str, comment_id: str,
    *, expect_reply_id: str = "",
) -> List[Dict[str, Any]]:
    """List all replies in a comment thread (paginated, up to 500).

    If *expect_reply_id* is set and not found in the first fetch,
    retries up to 6 times (handles eventual consistency).
    """
    logger.debug("[Feishu-Comment] list_comment_replies: file_token=%s comment_id=%s", file_token, comment_id)

    for attempt in range(_COMMENT_RETRY_LIMIT):
        all_replies: List[Dict[str, Any]] = []
        page_token = ""
        fetch_ok = True

        for _ in range(5):  # max 5 pages
            queries = [
                ("file_type", file_type),
                ("page_size", "100"),
                ("user_id_type", "open_id"),
            ]
            if page_token:
                queries.append(("page_token", page_token))

            code, msg, data = await _exec_request(
                client, "GET", _LIST_REPLIES_URI,
                paths={"file_token": file_token, "comment_id": comment_id},
                queries=queries,
            )
            if code != 0:
                logger.warning("[Feishu-Comment] List replies failed: code=%s msg=%s", code, msg)
                fetch_ok = False
                break

            items = data.get("items", [])
            if isinstance(items, list):
                all_replies.extend(items)

            if not data.get("has_more"):
                break
            page_token = data.get("page_token", "")
            if not page_token:
                break

        # If we don't need a specific reply, or we found it, return
        if not expect_reply_id or not fetch_ok:
            break
        found = any(r.get("reply_id") == expect_reply_id for r in all_replies)
        if found:
            break
        if attempt < _COMMENT_RETRY_LIMIT - 1:
            logger.info(
                "[Feishu-Comment] list_comment_replies: reply_id=%s not found, retry %d/%d",
                expect_reply_id, attempt + 1, _COMMENT_RETRY_LIMIT,
            )
            await asyncio.sleep(_COMMENT_RETRY_DELAY_S)
        else:
            logger.warning(
                "[Feishu-Comment] list_comment_replies: reply_id=%s not found after %d attempts",
                expect_reply_id, _COMMENT_RETRY_LIMIT,
            )

    logger.info("[Feishu-Comment] list_comment_replies: total %d replies fetched", len(all_replies))
    return all_replies


def _sanitize_comment_text(text: str) -> str:
    """Escape characters not allowed in Feishu comment text_run content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def reply_to_comment(
    client: Any, file_token: str, file_type: str, comment_id: str, text: str,
) -> Tuple[bool, int]:
    """Post a reply to a local comment thread.

    Returns ``(success, code)``.
    """
    text = _sanitize_comment_text(text)
    logger.info("[Feishu-Comment] reply_to_comment: comment_id=%s text=%s",
                comment_id, text[:100])
    body = {
        "content": {
            "elements": [
                {"type": "text_run", "text_run": {"text": text}},
            ]
        }
    }

    code, msg, _ = await _exec_request(
        client, "POST", _REPLY_COMMENT_URI,
        paths={"file_token": file_token, "comment_id": comment_id},
        queries=[("file_type", file_type)],
        body=body,
    )
    if code != 0:
        logger.warning(
            "[Feishu-Comment] reply_to_comment FAILED: code=%s msg=%s comment_id=%s",
            code, msg, comment_id,
        )
    else:
        logger.info("[Feishu-Comment] reply_to_comment OK: comment_id=%s", comment_id)
    return code == 0, code


async def add_whole_comment(
    client: Any, file_token: str, file_type: str, text: str,
) -> bool:
    """Add a new whole-document comment.

    Returns ``True`` on success.
    """
    text = _sanitize_comment_text(text)
    logger.info("[Feishu-Comment] add_whole_comment: file_token=%s text=%s",
                file_token, text[:100])
    body = {
        "file_type": file_type,
        "reply_elements": [
            {"type": "text", "text": text},
        ],
    }

    code, msg, _ = await _exec_request(
        client, "POST", _ADD_COMMENT_URI,
        paths={"file_token": file_token},
        body=body,
    )
    if code != 0:
        logger.warning("[Feishu-Comment] add_whole_comment FAILED: code=%s msg=%s", code, msg)
    else:
        logger.info("[Feishu-Comment] add_whole_comment OK")
    return code == 0


_REPLY_CHUNK_SIZE = 4000


def _chunk_text(text: str, limit: int = _REPLY_CHUNK_SIZE) -> List[str]:
    """Split text into chunks for delivery, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Find last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


async def deliver_comment_reply(
    client: Any,
    file_token: str,
    file_type: str,
    comment_id: str,
    text: str,
    is_whole: bool,
) -> bool:
    """Route agent reply to the correct API, chunking long text.

    - Whole comment -> add_whole_comment
    - Local comment -> reply_to_comment, fallback to add_whole_comment on 1069302
    """
    chunks = _chunk_text(text)
    logger.info("[Feishu-Comment] deliver_comment_reply: is_whole=%s comment_id=%s text_len=%d chunks=%d",
                is_whole, comment_id, len(text), len(chunks))

    all_ok = True
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info("[Feishu-Comment] deliver_comment_reply: sending chunk %d/%d (%d chars)",
                        i + 1, len(chunks), len(chunk))

        if is_whole:
            ok = await add_whole_comment(client, file_token, file_type, chunk)
        else:
            success, code = await reply_to_comment(client, file_token, file_type, comment_id, chunk)
            if success:
                ok = True
            elif code == 1069302:
                logger.info("[Feishu-Comment] Reply not allowed (1069302), falling back to add_whole_comment")
                ok = await add_whole_comment(client, file_token, file_type, chunk)
                is_whole = True  # subsequent chunks also use add_comment
            else:
                ok = False

        if not ok:
            all_ok = False
            break

    return all_ok


# ---------------------------------------------------------------------------
# Comment content extraction helpers
# ---------------------------------------------------------------------------


def _extract_reply_text(reply: Dict[str, Any]) -> str:
    """Extract plain text from a comment reply's content structure."""
    content = reply.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    elements = content.get("elements", [])
    parts = []
    for elem in elements:
        if elem.get("type") == "text_run":
            text_run = elem.get("text_run", {})
            parts.append(text_run.get("text", ""))
        elif elem.get("type") == "docs_link":
            docs_link = elem.get("docs_link", {})
            parts.append(docs_link.get("url", ""))
        elif elem.get("type") == "person":
            person = elem.get("person", {})
            parts.append(f"@{person.get('user_id', 'unknown')}")
    return "".join(parts)


def _get_reply_user_id(reply: Dict[str, Any]) -> str:
    """Extract user_id from a reply dict."""
    user_id = reply.get("user_id", "")
    if isinstance(user_id, dict):
        return user_id.get("open_id", "") or user_id.get("user_id", "")
    return str(user_id)


def _extract_semantic_text(reply: Dict[str, Any], self_open_id: str = "") -> str:
    """Extract semantic text from a reply, stripping self @mentions and extra whitespace."""
    content = reply.get("content", {})
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content

    elements = content.get("elements", [])
    parts = []
    for elem in elements:
        if elem.get("type") == "person":
            person = elem.get("person", {})
            uid = person.get("user_id", "")
            # Skip self @mention (it's routing, not content)
            if self_open_id and uid == self_open_id:
                continue
            parts.append(f"@{uid}")
        elif elem.get("type") == "text_run":
            text_run = elem.get("text_run", {})
            parts.append(text_run.get("text", ""))
        elif elem.get("type") == "docs_link":
            docs_link = elem.get("docs_link", {})
            parts.append(docs_link.get("url", ""))
    return " ".join("".join(parts).split()).strip()


# ---------------------------------------------------------------------------
# Document link parsing and wiki resolution
# ---------------------------------------------------------------------------

import re as _re

# Matches feishu/lark document URLs and extracts doc_type + token
_FEISHU_DOC_URL_RE = _re.compile(
    r"(?:feishu\.cn|larkoffice\.com|larksuite\.com|lark\.suite\.com)"
    r"/(?P<doc_type>wiki|doc|docx|sheet|sheets|slides|mindnote|bitable|base|file)"
    r"/(?P<token>[A-Za-z0-9_-]{10,40})"
)

_WIKI_GET_NODE_URI = "/open-apis/wiki/v2/spaces/get_node"


def _extract_docs_links(replies: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract unique document links from a list of comment replies.

    Returns list of ``{"url": "...", "doc_type": "...", "token": "..."}`` dicts.
    """
    seen_tokens = set()
    links = []
    for reply in replies:
        content = reply.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
        for elem in content.get("elements", []):
            if elem.get("type") not in {"docs_link", "link"}:
                continue
            link_data = elem.get("docs_link") or elem.get("link") or {}
            url = link_data.get("url", "")
            if not url:
                continue
            m = _FEISHU_DOC_URL_RE.search(url)
            if not m:
                continue
            doc_type = m.group("doc_type")
            token = m.group("token")
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            links.append({"url": url, "doc_type": doc_type, "token": token})
    return links


async def _reverse_lookup_wiki_token(
    client: Any, obj_type: str, obj_token: str,
) -> Optional[str]:
    """Reverse-lookup: given an obj_token, find its wiki node_token.

    Returns the wiki_token if the document belongs to a wiki space,
    or None if it doesn't or the API call fails.
    """
    code, msg, data = await _exec_request(
        client, "GET", _WIKI_GET_NODE_URI,
        queries=[("token", obj_token), ("obj_type", obj_type)],
    )
    if code == 0:
        node = data.get("node", {})
        wiki_token = node.get("node_token", "")
        return wiki_token if wiki_token else None
    # code != 0: either not a wiki doc or service error — log and return None
    logger.warning("[Feishu-Comment] Wiki reverse lookup failed: code=%s msg=%s obj=%s:%s", code, msg, obj_type, obj_token)
    return None


async def _resolve_wiki_nodes(
    client: Any,
    links: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Resolve wiki links to their underlying document type and token.

    Mutates entries in *links* in-place: replaces ``doc_type`` and ``token``
    with the resolved values for wiki links.  Non-wiki links are unchanged.
    """
    wiki_links = [l for l in links if l["doc_type"] == "wiki"]
    if not wiki_links:
        return links

    for link in wiki_links:
        wiki_token = link["token"]
        code, msg, data = await _exec_request(
            client, "GET", _WIKI_GET_NODE_URI,
            queries=[("token", wiki_token)],
        )
        if code == 0:
            node = data.get("node", {})
            resolved_type = node.get("obj_type", "")
            resolved_token = node.get("obj_token", "")
            if resolved_type and resolved_token:
                logger.info(
                    "[Feishu-Comment] Wiki resolved: %s -> %s:%s",
                    wiki_token, resolved_type, resolved_token,
                )
                link["resolved_type"] = resolved_type
                link["resolved_token"] = resolved_token
            else:
                logger.warning("[Feishu-Comment] Wiki resolve returned empty: %s", wiki_token)
        else:
            logger.warning("[Feishu-Comment] Wiki resolve failed: code=%s msg=%s token=%s", code, msg, wiki_token)

    return links


def _format_referenced_docs(
    links: List[Dict[str, str]], current_file_token: str = "",
) -> str:
    """Format resolved document links for prompt embedding."""
    if not links:
        return ""
    lines = ["", "Referenced documents in comments:"]
    for link in links:
        rtype = link.get("resolved_type", link["doc_type"])
        rtoken = link.get("resolved_token", link["token"])
        is_current = rtoken == current_file_token
        suffix = " (same as current document)" if is_current else ""
        lines.append(f"- {rtype}:{rtoken}{suffix} ({link['url'][:80]})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_PROMPT_TEXT_LIMIT = 220
_LOCAL_TIMELINE_LIMIT = 20
_WHOLE_TIMELINE_LIMIT = 12


def _truncate(text: str, limit: int = _PROMPT_TEXT_LIMIT) -> str:
    """Truncate text for prompt embedding."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _select_local_timeline(
    timeline: List[Tuple[str, str, bool]],
    target_index: int,
) -> List[Tuple[str, str, bool]]:
    """Select up to _LOCAL_TIMELINE_LIMIT entries centered on target_index.

    Always keeps first, target, and last entries.
    """
    if len(timeline) <= _LOCAL_TIMELINE_LIMIT:
        return timeline
    n = len(timeline)
    selected = set()
    selected.add(0)                            # first
    selected.add(n - 1)                        # last
    if 0 <= target_index < n:
        selected.add(target_index)             # current
    # Expand outward from target
    budget = _LOCAL_TIMELINE_LIMIT - len(selected)
    lo, hi = target_index - 1, target_index + 1
    while budget > 0 and (lo >= 0 or hi < n):
        if lo >= 0 and lo not in selected:
            selected.add(lo)
            budget -= 1
        lo -= 1
        if budget > 0 and hi < n and hi not in selected:
            selected.add(hi)
            budget -= 1
        hi += 1
    return [timeline[i] for i in sorted(selected)]


def _select_whole_timeline(
    timeline: List[Tuple[str, str, bool]],
    current_index: int,
    nearest_self_index: int,
) -> List[Tuple[str, str, bool]]:
    """Select up to _WHOLE_TIMELINE_LIMIT entries for whole-doc comments.

    Prioritizes current entry and nearest self reply.
    """
    if len(timeline) <= _WHOLE_TIMELINE_LIMIT:
        return timeline
    n = len(timeline)
    selected = set()
    if 0 <= current_index < n:
        selected.add(current_index)
    if 0 <= nearest_self_index < n:
        selected.add(nearest_self_index)
    # Expand outward from current
    budget = _WHOLE_TIMELINE_LIMIT - len(selected)
    lo, hi = current_index - 1, current_index + 1
    while budget > 0 and (lo >= 0 or hi < n):
        if lo >= 0 and lo not in selected:
            selected.add(lo)
            budget -= 1
        lo -= 1
        if budget > 0 and hi < n and hi not in selected:
            selected.add(hi)
            budget -= 1
        hi += 1
    if not selected:
        # Fallback: take last N entries
        return timeline[-_WHOLE_TIMELINE_LIMIT:]
    return [timeline[i] for i in sorted(selected)]


_COMMON_INSTRUCTIONS = """
This is a Feishu document comment thread, not an IM chat.
Do NOT call feishu_drive_add_comment or feishu_drive_reply_comment yourself.
Your reply will be posted automatically. Just output the reply text.
Use the thread timeline above as the main context.
If the quoted content is not enough, use feishu_doc_read to read nearby context.
The quoted content is your primary anchor — insert/summarize/explain requests are about it.
Do not guess document content you haven't read.
Reply in the same language as the user's comment unless they request otherwise.
Use plain text only. Do not use Markdown, headings, bullet lists, tables, or code blocks.
Do not show your reasoning process. Do not start with "I will", "Let me", or "I'll first".
Output only the final user-facing reply.
If no reply is needed, output exactly NO_REPLY.
""".strip()


def build_local_comment_prompt(
    *,
    doc_title: str,
    doc_url: str,
    file_token: str,
    file_type: str,
    comment_id: str,
    quote_text: str,
    root_comment_text: str,
    target_reply_text: str,
    timeline: List[Tuple[str, str, bool]],  # [(user_id, text, is_self)]
    self_open_id: str,
    target_index: int = -1,
    referenced_docs: str = "",
) -> str:
    """Build the prompt for a local (quoted-text) comment."""
    selected = _select_local_timeline(timeline, target_index)

    lines = [
        f'The user added a reply in "{doc_title}".',
        f'Current user comment text: "{_truncate(target_reply_text)}"',
        f'Original comment text: "{_truncate(root_comment_text)}"',
        f'Quoted content: "{_truncate(quote_text, 500)}"',
        "This comment mentioned you (@mention is for routing, not task content).",
        f"Document link: {doc_url}",
        "Current commented document:",
        f"- file_type={file_type}",
        f"- file_token={file_token}",
        f"- comment_id={comment_id}",
        "",
        f"Current comment card timeline ({len(selected)}/{len(timeline)} entries):",
    ]

    for user_id, text, is_self in selected:
        marker = " <-- YOU" if is_self else ""
        lines.append(f"[{user_id}] {_truncate(text)}{marker}")

    if referenced_docs:
        lines.append(referenced_docs)

    lines.append("")
    lines.append(_COMMON_INSTRUCTIONS)
    return "\n".join(lines)


def build_whole_comment_prompt(
    *,
    doc_title: str,
    doc_url: str,
    file_token: str,
    file_type: str,
    comment_text: str,
    timeline: List[Tuple[str, str, bool]],  # [(user_id, text, is_self)]
    self_open_id: str,
    current_index: int = -1,
    nearest_self_index: int = -1,
    referenced_docs: str = "",
) -> str:
    """Build the prompt for a whole-document comment."""
    selected = _select_whole_timeline(timeline, current_index, nearest_self_index)

    lines = [
        f'The user added a comment in "{doc_title}".',
        f'Current user comment text: "{_truncate(comment_text)}"',
        "This is a whole-document comment.",
        "This comment mentioned you (@mention is for routing, not task content).",
        f"Document link: {doc_url}",
        "Current commented document:",
        f"- file_type={file_type}",
        f"- file_token={file_token}",
        "",
        f"Whole-document comment timeline ({len(selected)}/{len(timeline)} entries):",
    ]

    for user_id, text, is_self in selected:
        marker = " <-- YOU" if is_self else ""
        lines.append(f"[{user_id}] {_truncate(text)}{marker}")

    if referenced_docs:
        lines.append(referenced_docs)

    lines.append("")
    lines.append(_COMMON_INSTRUCTIONS)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


def _resolve_model_and_runtime() -> Tuple[str, dict]:
    """Resolve model and provider credentials, same as gateway message handling."""
    from gateway.run import _load_gateway_config, _resolve_gateway_model

    user_config = _load_gateway_config()
    model = _resolve_gateway_model(user_config)

    from gateway.run import _resolve_runtime_agent_kwargs
    runtime_kwargs = _resolve_runtime_agent_kwargs()

    # Fall back to provider's default model if none configured
    if not model and runtime_kwargs.get("provider"):
        try:
            from hermes_cli.models import get_default_model_for_provider
            model = get_default_model_for_provider(runtime_kwargs["provider"])
        except Exception:
            pass

    return model, runtime_kwargs


# ---------------------------------------------------------------------------
# Session cache for cross-card memory within the same document
# ---------------------------------------------------------------------------

import threading
import time as _time

_SESSION_MAX_MESSAGES = 50  # keep last N messages per document session
_SESSION_TTL_S = 3600       # expire sessions after 1 hour of inactivity

_session_cache_lock = threading.Lock()
_session_cache: Dict[str, Dict] = {}  # key -> {"messages": [...], "last_access": float}


def _session_key(file_type: str, file_token: str) -> str:
    return f"comment-doc:{file_type}:{file_token}"


def _load_session_history(key: str) -> List[Dict[str, Any]]:
    """Load conversation history for a document session."""
    with _session_cache_lock:
        entry = _session_cache.get(key)
        if entry is None:
            return []
        # Check TTL
        if _time.time() - entry["last_access"] > _SESSION_TTL_S:
            del _session_cache[key]
            logger.info("[Feishu-Comment] Session expired: %s", key)
            return []
        entry["last_access"] = _time.time()
        return list(entry["messages"])


def _save_session_history(key: str, messages: List[Dict[str, Any]]) -> None:
    """Save conversation history for a document session (keeps last N messages)."""
    # Only keep user/assistant messages (strip system messages and tool internals)
    cleaned = [
        m for m in messages
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    # Keep last N
    if len(cleaned) > _SESSION_MAX_MESSAGES:
        cleaned = cleaned[-_SESSION_MAX_MESSAGES:]
    with _session_cache_lock:
        _session_cache[key] = {
            "messages": cleaned,
            "last_access": _time.time(),
        }
        logger.info("[Feishu-Comment] Session saved: %s (%d messages)", key, len(cleaned))


def _run_comment_agent(prompt: str, client: Any, session_key: str = "") -> str:
    """Create an AIAgent with feishu tools and run the prompt.

    If *session_key* is provided, loads/saves conversation history for
    cross-card memory within the same document.

    Returns the agent's final response text, or empty string on failure.
    """
    from run_agent import AIAgent

    logger.info("[Feishu-Comment] _run_comment_agent: injecting lark client into tool thread-locals")
    from tools.feishu_doc_tool import set_client as set_doc_client
    from tools.feishu_drive_tool import set_client as set_drive_client
    set_doc_client(client)
    set_drive_client(client)

    try:
        model, runtime_kwargs = _resolve_model_and_runtime()
        logger.info("[Feishu-Comment] _run_comment_agent: model=%s provider=%s base_url=%s",
                    model, runtime_kwargs.get("provider"), (runtime_kwargs.get("base_url") or "")[:50])

        # Load session history for cross-card memory
        history = _load_session_history(session_key) if session_key else []
        if history:
            logger.info("[Feishu-Comment] _run_comment_agent: loaded %d history messages from session %s",
                        len(history), session_key)

        agent = AIAgent(
            model=model,
            base_url=runtime_kwargs.get("base_url"),
            api_key=runtime_kwargs.get("api_key"),
            provider=runtime_kwargs.get("provider"),
            api_mode=runtime_kwargs.get("api_mode"),
            credential_pool=runtime_kwargs.get("credential_pool"),
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=15,
            enabled_toolsets=["feishu_doc", "feishu_drive"],
        )
        logger.info("[Feishu-Comment] _run_comment_agent: calling run_conversation (prompt=%d chars, history=%d)",
                    len(prompt), len(history))
        result = agent.run_conversation(prompt, conversation_history=history or None)
        response = (result.get("final_response") or "").strip()
        api_calls = result.get("api_calls", 0)
        logger.info("[Feishu-Comment] _run_comment_agent: done api_calls=%d response_len=%d response=%s",
                    api_calls, len(response), response[:200])

        # Save updated history
        if session_key:
            new_messages = result.get("messages", [])
            if new_messages:
                _save_session_history(session_key, new_messages)

        return response
    except Exception as e:
        logger.exception("[Feishu-Comment] _run_comment_agent: agent failed: %s", e)
        return ""
    finally:
        set_doc_client(None)
        set_drive_client(None)


# ---------------------------------------------------------------------------
# Event handler entry point
# ---------------------------------------------------------------------------

_NO_REPLY_SENTINEL = "NO_REPLY"


_ALLOWED_NOTICE_TYPES = {"add_comment", "add_reply"}


async def handle_drive_comment_event(
    client: Any, data: Any, *, self_open_id: str = "",
) -> None:
    """Full orchestration for a drive comment event.

    1. Parse event + filter (self-reply, notice_type)
    2. Add OK reaction
    3. Fetch doc meta + comment details in parallel
    4. Branch on is_whole: build timeline
    5. Build prompt, run agent
    6. Deliver reply
    """
    logger.info("[Feishu-Comment] ========== handle_drive_comment_event START ==========")
    parsed = parse_drive_comment_event(data)
    if parsed is None:
        logger.warning("[Feishu-Comment] Dropping malformed drive comment event")
        return
    logger.info("[Feishu-Comment] [Step 0/5] Event parsed successfully")

    file_token = parsed["file_token"]
    file_type = parsed["file_type"]
    comment_id = parsed["comment_id"]
    reply_id = parsed["reply_id"]
    from_open_id = parsed["from_open_id"]
    to_open_id = parsed["to_open_id"]
    notice_type = parsed["notice_type"]

    # Filter: self-reply, receiver check, notice_type
    if from_open_id and self_open_id and from_open_id == self_open_id:
        logger.debug("[Feishu-Comment] Skipping self-authored event: from=%s", from_open_id)
        return
    if not to_open_id or (self_open_id and to_open_id != self_open_id):
        logger.debug("[Feishu-Comment] Skipping event not addressed to self: to=%s", to_open_id or "(empty)")
        return
    if notice_type and notice_type not in _ALLOWED_NOTICE_TYPES:
        logger.debug("[Feishu-Comment] Skipping notice_type=%s", notice_type)
        return
    if not file_token or not file_type or not comment_id:
        logger.warning("[Feishu-Comment] Missing required fields, skipping")
        return

    logger.info(
        "[Feishu-Comment] Event: notice=%s file=%s:%s comment=%s from=%s",
        notice_type, file_type, file_token, comment_id, from_open_id,
    )

    # Access control
    from gateway.platforms.feishu_comment_rules import load_config, resolve_rule, is_user_allowed, has_wiki_keys

    comments_cfg = load_config()
    rule = resolve_rule(comments_cfg, file_type, file_token)

    # If no exact match and config has wiki keys, try reverse-lookup
    if rule.match_source in {"wildcard", "top"} and has_wiki_keys(comments_cfg):
        wiki_token = await _reverse_lookup_wiki_token(client, file_type, file_token)
        if wiki_token:
            rule = resolve_rule(comments_cfg, file_type, file_token, wiki_token=wiki_token)

    if not rule.enabled:
        logger.info("[Feishu-Comment] Comments disabled for %s:%s, skipping", file_type, file_token)
        return
    if not is_user_allowed(rule, from_open_id):
        logger.info("[Feishu-Comment] User %s denied (policy=%s, rule=%s)", from_open_id, rule.policy, rule.match_source)
        return

    logger.info("[Feishu-Comment] Access granted: user=%s policy=%s rule=%s", from_open_id, rule.policy, rule.match_source)
    if reply_id:
        asyncio.ensure_future(
            add_comment_reaction(
                client,
                file_token=file_token,
                file_type=file_type,
                reply_id=reply_id,
                reaction_type="OK",
            )
        )

    # Step 2: Parallel fetch -- doc meta + comment details
    logger.info("[Feishu-Comment] [Step 2/5] Parallel fetch: doc meta + comment batch_query")
    meta_task = asyncio.ensure_future(
        query_document_meta(client, file_token, file_type)
    )
    comment_task = asyncio.ensure_future(
        batch_query_comment(client, file_token, file_type, comment_id)
    )
    doc_meta, comment_detail = await asyncio.gather(meta_task, comment_task)

    doc_title = doc_meta.get("title", "Untitled")
    doc_url = doc_meta.get("url", "")
    is_whole = bool(comment_detail.get("is_whole"))

    logger.info(
        "[Feishu-Comment] Comment context: title=%s is_whole=%s",
        doc_title, is_whole,
    )

    # Step 3: Build timeline based on comment type
    logger.info("[Feishu-Comment] [Step 3/5] Building timeline (is_whole=%s)", is_whole)
    if is_whole:
        # Whole-document comment: fetch all whole comments as timeline
        logger.info("[Feishu-Comment] Fetching whole-document comments for timeline...")
        whole_comments = await list_whole_comments(client, file_token, file_type)

        timeline: List[Tuple[str, str, bool]] = []
        current_text = ""
        current_index = -1
        nearest_self_index = -1
        for wc in whole_comments:
            reply_list = wc.get("reply_list", {})
            if isinstance(reply_list, str):
                try:
                    reply_list = json.loads(reply_list)
                except (json.JSONDecodeError, TypeError):
                    reply_list = {}
            replies = reply_list.get("replies", [])
            for r in replies:
                uid = _get_reply_user_id(r)
                text = _extract_reply_text(r)
                is_self = (uid == self_open_id) if self_open_id else False
                idx = len(timeline)
                timeline.append((uid, text, is_self))
                if uid == from_open_id:
                    current_text = _extract_semantic_text(r, self_open_id)
                    current_index = idx
                if is_self:
                    nearest_self_index = idx

        if not current_text:
            for i, (uid, text, is_self) in reversed(list(enumerate(timeline))):
                if not is_self:
                    current_text = text
                    current_index = i
                    break

        logger.info("[Feishu-Comment] Whole timeline: %d entries, current_idx=%d, self_idx=%d, text=%s",
                    len(timeline), current_index, nearest_self_index,
                    current_text[:80] if current_text else "(empty)")

        # Extract and resolve document links from all replies
        all_raw_replies = []
        for wc in whole_comments:
            rl = wc.get("reply_list", {})
            if isinstance(rl, str):
                try:
                    rl = json.loads(rl)
                except (json.JSONDecodeError, TypeError):
                    rl = {}
            all_raw_replies.extend(rl.get("replies", []))
        doc_links = _extract_docs_links(all_raw_replies)
        if doc_links:
            doc_links = await _resolve_wiki_nodes(client, doc_links)
        ref_docs_text = _format_referenced_docs(doc_links, file_token)

        prompt = build_whole_comment_prompt(
            doc_title=doc_title,
            doc_url=doc_url,
            file_token=file_token,
            file_type=file_type,
            comment_text=current_text,
            timeline=timeline,
            self_open_id=self_open_id,
            current_index=current_index,
            nearest_self_index=nearest_self_index,
            referenced_docs=ref_docs_text,
        )

    else:
        # Local comment: fetch the comment thread replies
        logger.info("[Feishu-Comment] Fetching comment thread replies...")
        replies = await list_comment_replies(
            client, file_token, file_type, comment_id,
            expect_reply_id=reply_id,
        )

        quote_text = comment_detail.get("quote", "")

        timeline = []
        root_text = ""
        target_text = ""
        target_index = -1
        for i, r in enumerate(replies):
            uid = _get_reply_user_id(r)
            text = _extract_reply_text(r)
            is_self = (uid == self_open_id) if self_open_id else False
            timeline.append((uid, text, is_self))
            if i == 0:
                root_text = _extract_semantic_text(r, self_open_id)
            rid = r.get("reply_id", "")
            if rid and rid == reply_id:
                target_text = _extract_semantic_text(r, self_open_id)
                target_index = i

        if not target_text and timeline:
            for i, (uid, text, is_self) in reversed(list(enumerate(timeline))):
                if uid == from_open_id:
                    target_text = text
                    target_index = i
                    break

        logger.info("[Feishu-Comment] Local timeline: %d entries, target_idx=%d, quote=%s root=%s target=%s",
                    len(timeline), target_index,
                    quote_text[:60] if quote_text else "(empty)",
                    root_text[:60] if root_text else "(empty)",
                    target_text[:60] if target_text else "(empty)")

        # Extract and resolve document links from replies
        doc_links = _extract_docs_links(replies)
        if doc_links:
            doc_links = await _resolve_wiki_nodes(client, doc_links)
        ref_docs_text = _format_referenced_docs(doc_links, file_token)

        prompt = build_local_comment_prompt(
            doc_title=doc_title,
            doc_url=doc_url,
            file_token=file_token,
            file_type=file_type,
            comment_id=comment_id,
            quote_text=quote_text,
            root_comment_text=root_text,
            target_reply_text=target_text,
            timeline=timeline,
            self_open_id=self_open_id,
            target_index=target_index,
            referenced_docs=ref_docs_text,
        )

    logger.info("[Feishu-Comment] [Step 4/5] Prompt built (%d chars), running agent...", len(prompt))
    logger.debug("[Feishu-Comment] Full prompt:\n%s", prompt)

    # Step 4: Run agent in a thread (run_conversation is synchronous)
    # Session key groups all comment cards on the same document
    sess_key = _session_key(file_type, file_token)
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, _run_comment_agent, prompt, client, sess_key,
    )

    if not response or _NO_REPLY_SENTINEL in response:
        logger.info("[Feishu-Comment] Agent returned NO_REPLY, skipping delivery")
    else:
        logger.info("[Feishu-Comment] Agent response (%d chars): %s", len(response), response[:200])

        # Step 5: Deliver reply
        logger.info("[Feishu-Comment] [Step 5/5] Delivering reply (is_whole=%s, comment_id=%s)", is_whole, comment_id)
        success = await deliver_comment_reply(
            client, file_token, file_type, comment_id, response, is_whole,
        )
        if success:
            logger.info("[Feishu-Comment] Reply delivered successfully")
        else:
            logger.error("[Feishu-Comment] Failed to deliver reply")

    # Cleanup: remove OK reaction (best-effort, non-blocking)
    if reply_id:
        await delete_comment_reaction(
            client,
            file_token=file_token,
            file_type=file_type,
            reply_id=reply_id,
            reaction_type="OK",
        )

    logger.info("[Feishu-Comment] ========== handle_drive_comment_event END ==========")
