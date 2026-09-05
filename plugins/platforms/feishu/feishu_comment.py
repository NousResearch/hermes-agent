"""Feishu/Lark drive document comment handling (``drive.notice.comment_add_v1`` → Drive v1/v2 comment APIs).
Flow: parse event -> access check -> OK reaction -> parallel fetch (doc meta + comment) -> timeline (whole-doc comments
or local thread replies) -> prompt -> AIAgent with feishu_doc + feishu_drive tools -> deliver reply (whole ->
add_whole_comment; local -> reply_to_comment, falling back to add_whole_comment on 1069302)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


async def _exec_request(client, method, uri, paths=None, queries=None, body=None):
    """Execute a lark API request (tenant token) and return (code, msg, data_dict)."""
    logger.info("[Feishu-Comment] API >>> %s %s paths=%s queries=%s body=%s", method, uri, paths, queries, json.dumps(body, ensure_ascii=False)[:500] if body else None)
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest
    builder = BaseRequest.builder().http_method(HttpMethod.GET if method == "GET" else HttpMethod.POST).uri(uri).token_types({AccessTokenType.TENANT})
    for setter, value in (("paths", paths), ("queries", queries), ("body", body)):
        if value or (setter == "body" and value is not None):
            builder = getattr(builder, setter)(value)
    response = await asyncio.to_thread(client.request, builder.build())
    code, msg, raw = getattr(response, "code", None), getattr(response, "msg", ""), getattr(response, "raw", None)
    has_raw = bool(raw and hasattr(raw, "content"))
    try:
        data: dict = json.loads(raw.content).get("data", {}) if has_raw else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}
    resp_data = None if data else getattr(response, "data", None)
    if isinstance(resp_data, dict) or (resp_data and hasattr(resp_data, "__dict__")):
        data = _as_dict(resp_data)
    logger.info("[Feishu-Comment] API <<< %s %s code=%s msg=%s data_keys=%s", method, uri, code, msg, list(data.keys()) if data else "empty")
    if code != 0:
        shown = (raw.content if isinstance(raw.content, (str, bytes)) else str(raw.content))[:500] if has_raw else ""
        logger.warning("[Feishu-Comment] API FAIL raw response: %s", shown)
    return code, msg, data


def _log_outcome(code, fail_fmt: str, fail_args: tuple, ok_fmt: str, ok_args: tuple) -> None:
    """warning(fail_fmt) when *code* != 0 else info(ok_fmt)."""
    if code != 0:
        logger.warning(fail_fmt, *fail_args)
    else:
        logger.info(ok_fmt, *ok_args)


def _as_dict(obj: Any) -> dict:
    """Coerce a dict or SDK object (via ``vars()``) into a dict; anything else -> {}."""
    return obj if isinstance(obj, dict) else (vars(obj) if hasattr(obj, "__dict__") else {})


_BAD_JSON = object()


def _maybe_json(value: Any, fallback: Any = _BAD_JSON) -> Any:
    """Decode *value* when it is a JSON string (undecodable -> *fallback*); non-strings pass through."""
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return fallback


def parse_drive_comment_event(data: Any) -> Optional[Dict[str, Any]]:
    """Extract a flat field dict from a ``drive.notice.comment_add_v1`` payload, or None when malformed. *data* is a ``CustomizedEvent`` (WebSocket; ``.event`` is a
    dict) or a ``SimpleNamespace`` (Webhook body)."""
    logger.debug("[Feishu-Comment] parse_drive_comment_event: data type=%s", type(data).__name__)
    event = getattr(data, "event", None)
    if event is None:
        return logger.debug("[Feishu-Comment] parse_drive_comment_event: no .event attribute, returning None")
    evt = _as_dict(event)
    logger.debug("[Feishu-Comment] parse_drive_comment_event: evt keys=%s", list(evt.keys()))
    notice_meta = _as_dict(evt.get("notice_meta") or {})
    fields = {
        **{k: evt.get(k) for k in ("event_id", "comment_id", "reply_id", "timestamp")},
        **{k: notice_meta.get(k) for k in ("file_token", "file_type", "notice_type")},
        "from_open_id": _as_dict(notice_meta.get("from_user_id") or {}).get("open_id"),
        "to_open_id": _as_dict(notice_meta.get("to_user_id") or {}).get("open_id"),
    }
    return {**{k: str(v or "") for k, v in fields.items()}, "is_mentioned": bool(evt.get("is_mentioned"))}


_REACTION_URI = "/open-apis/drive/v2/files/:file_token/comments/reaction"
_BATCH_QUERY_META_URI = "/open-apis/drive/v1/metas/batch_query"
_BATCH_QUERY_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/comments/batch_query"
_LIST_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"
_WIKI_GET_NODE_URI = "/open-apis/wiki/v2/spaces/get_node"

_COMMENT_RETRY_LIMIT = 6
_COMMENT_RETRY_DELAY_S = 1.0
_MAX_PAGES = 5  # 5 x page_size 100
_REACTION_VERBS = {"add": "added", "delete": "deleted"}
_REPLY_CHUNK_SIZE = 4000
_ELEMENT_TEXT_KEYS = {"text_run": "text", "docs_link": "url"}  # reply element type -> key holding its plain text
_FEISHU_DOC_URL_RE = re.compile(
    r"(?:feishu\.cn|larkoffice\.com|larksuite\.com|lark\.suite\.com)"
    r"/(?P<doc_type>wiki|doc|docx|sheet|sheets|slides|mindnote|bitable|base|file)"
    r"/(?P<token>[A-Za-z0-9_-]{10,40})"
)
_PROMPT_TEXT_LIMIT = 220
_LOCAL_TIMELINE_LIMIT = 20
_WHOLE_TIMELINE_LIMIT = 12
_MENTION_NOTE = "This comment mentioned you (@mention is for routing, not task content)."
_NO_REPLY_SENTINEL = "NO_REPLY"
_ALLOWED_NOTICE_TYPES = {"add_comment", "add_reply"}
_SESSION_MAX_MESSAGES = 50  # cross-card memory within one document: keep last N messages per document session
_SESSION_TTL_S = 3600       # expire sessions after 1 hour of inactivity
_session_cache_lock = threading.Lock()
_session_cache: Dict[str, Dict] = {}  # key -> {"messages": [...], "last_access": float}
Timeline = List[Tuple[str, str, bool]]  # [(user_id, text, is_self)]


async def update_comment_reaction(client: Any, action: str, *, file_token: str, file_type: str, reply_id: str, reaction_type: str = "OK") -> bool:
    """Add (``action="add"``) or remove (``"delete"``) an emoji reaction on a comment reply (Drive v2); best-effort bool."""
    try:  # the add path is the first SDK touch per event: surface a missing lark_oapi cleanly
        if action == "add":
            from lark_oapi import AccessTokenType  # noqa: F401
    except ImportError:
        logger.error("[Feishu-Comment] lark_oapi not available")
        return False
    code, msg, _ = await _exec_request(client, "POST", _REACTION_URI, paths={"file_token": file_token}, queries=[("file_type", file_type)],
                                       body={"action": action, "reply_id": reply_id, "reaction_type": reaction_type})
    _log_outcome(code, "[Feishu-Comment] Reaction API failed: code=%s msg=%s file=%s:%s reply=%s", (code, msg, file_type, file_token, reply_id),
                 "[Feishu-Comment] Reaction '%s' %s: file=%s:%s reply=%s", (reaction_type, _REACTION_VERBS[action], file_type, file_token, reply_id))
    return code == 0


async def query_document_meta(client: Any, file_token: str, file_type: str) -> Dict[str, Any]:
    """Fetch ``{"title", "url", "doc_type"}`` via the batch_query meta API; empty dict on failure."""
    logger.debug("[Feishu-Comment] query_document_meta: file_token=%s file_type=%s", file_token, file_type)
    code, msg, data = await _exec_request(client, "POST", _BATCH_QUERY_META_URI, body={"request_docs": [{"doc_token": file_token, "doc_type": file_type}], "with_url": True})
    if code != 0:
        return logger.warning("[Feishu-Comment] Meta batch_query failed: code=%s msg=%s", code, msg) or {}
    metas = data.get("metas", [])
    logger.debug("[Feishu-Comment] query_document_meta: raw metas type=%s value=%s", type(metas).__name__, str(metas)[:300])
    if not metas and not isinstance(metas, dict):
        return logger.debug("[Feishu-Comment] query_document_meta: no metas found") or {}
    # alternate response shape: dict keyed by token
    meta = (metas[0] if isinstance(metas, list) else {}) if metas else metas.get(file_token, {})
    result = {"title": meta.get("title", ""), "url": meta.get("url", ""), "doc_type": meta.get("doc_type", file_type)}
    logger.info("[Feishu-Comment] query_document_meta: title=%s url=%s", result["title"], result["url"][:80] if result["url"] else "")
    return result


async def _retry_pause(attempt: int, retry_fmt: str, retry_args: tuple, fail_fmt: str, fail_args: tuple, lead: tuple = ()) -> bool:
    """Between retries: log ``retry_fmt % (*lead, attempt+1, LIMIT, *retry_args)`` and sleep -> True; on the last
    attempt log ``fail_fmt % (*lead, LIMIT, *fail_args)`` at warning -> False."""
    if attempt < _COMMENT_RETRY_LIMIT - 1:
        logger.info(retry_fmt, *lead, attempt + 1, _COMMENT_RETRY_LIMIT, *retry_args)
        await asyncio.sleep(_COMMENT_RETRY_DELAY_S)
        return True
    return logger.warning(fail_fmt, *lead, _COMMENT_RETRY_LIMIT, *fail_args) or False


async def batch_query_comment(client: Any, file_token: str, file_type: str, comment_id: str) -> Dict[str, Any]:
    """Fetch one comment's details (``is_whole``, ``quote``, ``reply_list``...); empty dict on failure. Retries up to ``_COMMENT_RETRY_LIMIT`` times: the comment
    may not be queryable yet when the notice arrives."""
    logger.debug("[Feishu-Comment] batch_query_comment: file_token=%s comment_id=%s", file_token, comment_id)
    for attempt in range(_COMMENT_RETRY_LIMIT):
        code, msg, data = await _exec_request(client, "POST", _BATCH_QUERY_COMMENT_URI, paths={"file_token": file_token},
                                              queries=[("file_type", file_type), ("user_id_type", "open_id")], body={"comment_ids": [comment_id]})
        if code == 0:
            break
        if not await _retry_pause(attempt, "[Feishu-Comment] batch_query_comment retry %d/%d: code=%s msg=%s", (code, msg),
                                  "[Feishu-Comment] batch_query_comment failed after %d attempts: code=%s msg=%s", (code, msg)):
            return {}
    items = data.get("items", [])
    logger.debug("[Feishu-Comment] batch_query_comment: got %d items", len(items) if isinstance(items, list) else 0)
    if not items or not isinstance(items, list):
        return logger.warning("[Feishu-Comment] batch_query_comment: empty items, raw data keys=%s", list(data.keys())) or {}
    item = items[0]
    logger.info("[Feishu-Comment] batch_query_comment: is_whole=%s quote=%s reply_count=%s", item.get("is_whole"), (item.get("quote", "") or "")[:60],
                len(item.get("reply_list", {}).get("replies", [])) if isinstance(item.get("reply_list"), dict) else "?")
    return item


async def _list_all_pages(client: Any, uri: str, paths: dict, queries: list, *, fail_msg: str, page_msg: str = "") -> Tuple[List[Dict[str, Any]], bool]:
    """GET up to ``_MAX_PAGES`` pages of ``items``; returns ``(items, fetch_ok)``. *fail_msg* is logged with ``(code, msg)`` on failure; *page_msg* (optional) at
    debug with ``(page_n, total)``."""
    items_out: List[Dict[str, Any]] = []
    page_token = ""
    for _ in range(_MAX_PAGES):
        code, msg, data = await _exec_request(client, "GET", uri, paths=paths, queries=queries + ([("page_token", page_token)] if page_token else []))
        if code != 0:
            return logger.warning(fail_msg, code, msg) or (items_out, False)
        items = data.get("items", [])
        items_out.extend(items if isinstance(items, list) else [])
        if isinstance(items, list) and page_msg:
            logger.debug(page_msg, len(items), len(items_out))
        page_token = data.get("page_token", "") if data.get("has_more") else ""
        if not page_token:
            break
    return items_out, True


async def list_whole_comments(client: Any, file_token: str, file_type: str) -> List[Dict[str, Any]]:
    """List all whole-document comments (paginated, up to 500)."""
    logger.debug("[Feishu-Comment] list_whole_comments: file_token=%s", file_token)
    all_comments, _ = await _list_all_pages(
        client, _LIST_COMMENTS_URI, {"file_token": file_token}, [("file_type", file_type), ("is_whole", "true"), ("page_size", "100"), ("user_id_type", "open_id")],
        fail_msg="[Feishu-Comment] List whole comments failed: code=%s msg=%s", page_msg="[Feishu-Comment] list_whole_comments: page got %d items, total=%d")
    logger.info("[Feishu-Comment] list_whole_comments: total %d whole comments fetched", len(all_comments))
    return all_comments


async def list_comment_replies(client: Any, file_token: str, file_type: str, comment_id: str, *, expect_reply_id: str = "") -> List[Dict[str, Any]]:
    """List all replies in a comment thread (paginated, up to 500). If *expect_reply_id* is set and absent from the fetched thread, retries up to
    ``_COMMENT_RETRY_LIMIT`` times (the new reply may not be listed yet)."""
    logger.debug("[Feishu-Comment] list_comment_replies: file_token=%s comment_id=%s", file_token, comment_id)
    for attempt in range(_COMMENT_RETRY_LIMIT):
        all_replies, fetch_ok = await _list_all_pages(
            client, _REPLIES_URI, {"file_token": file_token, "comment_id": comment_id}, [("file_type", file_type), ("page_size", "100"), ("user_id_type", "open_id")],
            fail_msg="[Feishu-Comment] List replies failed: code=%s msg=%s")
        if not expect_reply_id or not fetch_ok or any(r.get("reply_id") == expect_reply_id for r in all_replies):
            break
        await _retry_pause(attempt, "[Feishu-Comment] list_comment_replies: reply_id=%s not found, retry %d/%d", (),
                           "[Feishu-Comment] list_comment_replies: reply_id=%s not found after %d attempts", (), lead=(expect_reply_id,))
    logger.info("[Feishu-Comment] list_comment_replies: total %d replies fetched", len(all_replies))
    return all_replies


def _sanitize_comment_text(text: str) -> str:
    """Escape characters not allowed in Feishu comment text_run content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def reply_to_comment(client: Any, file_token: str, file_type: str, comment_id: str, text: str) -> Tuple[bool, int]:
    """Post a reply to a local comment thread. Returns ``(success, code)``."""
    text = _sanitize_comment_text(text)
    logger.info("[Feishu-Comment] reply_to_comment: comment_id=%s text=%s", comment_id, text[:100])
    code, msg, _ = await _exec_request(client, "POST", _REPLIES_URI, paths={"file_token": file_token, "comment_id": comment_id}, queries=[("file_type", file_type)],
                                       body={"content": {"elements": [{"type": "text_run", "text_run": {"text": text}}]}})
    _log_outcome(code, "[Feishu-Comment] reply_to_comment FAILED: code=%s msg=%s comment_id=%s", (code, msg, comment_id),
                 "[Feishu-Comment] reply_to_comment OK: comment_id=%s", (comment_id,))
    return code == 0, code


async def add_whole_comment(client: Any, file_token: str, file_type: str, text: str) -> bool:
    """Add a new whole-document comment. Returns ``True`` on success."""
    text = _sanitize_comment_text(text)
    logger.info("[Feishu-Comment] add_whole_comment: file_token=%s text=%s", file_token, text[:100])
    code, msg, _ = await _exec_request(client, "POST", _ADD_COMMENT_URI, paths={"file_token": file_token},
                                       body={"file_type": file_type, "reply_elements": [{"type": "text", "text": text}]})
    _log_outcome(code, "[Feishu-Comment] add_whole_comment FAILED: code=%s msg=%s", (code, msg), "[Feishu-Comment] add_whole_comment OK", ())
    return code == 0


def _chunk_text(text: str, limit: int = _REPLY_CHUNK_SIZE) -> List[str]:
    """Split text into chunks for delivery, preferring line breaks."""
    chunks = []
    while len(text) > limit:
        cut = nl if (nl := text.rfind("\n", 0, limit)) > 0 else limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks + ([text] if text or not chunks else [])


async def deliver_comment_reply(client: Any, file_token: str, file_type: str, comment_id: str, text: str, is_whole: bool) -> bool:
    """Route the agent reply to the right API, chunking long text. Whole comment -> add_whole_comment. Local comment -> reply_to_comment; on 1069302 (reply not
    allowed) fall back to add_whole_comment for this and all later chunks."""
    chunks = _chunk_text(text)
    logger.info("[Feishu-Comment] deliver_comment_reply: is_whole=%s comment_id=%s text_len=%d chunks=%d", is_whole, comment_id, len(text), len(chunks))
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info("[Feishu-Comment] deliver_comment_reply: sending chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        ok, code = (True, 0) if is_whole else await reply_to_comment(client, file_token, file_type, comment_id, chunk)
        if not is_whole and not ok and code == 1069302:
            logger.info("[Feishu-Comment] Reply not allowed (1069302), falling back to add_whole_comment")
            is_whole = True
        if is_whole:
            ok = await add_whole_comment(client, file_token, file_type, chunk)
        if not ok:
            return False
    return True


def _extract_reply_text(reply: Dict[str, Any], *, semantic: bool = False, self_open_id: str = "") -> str:
    """Plain text of a reply's content (text_run / docs_link / person elements). Person mentions render as ``@<user_id>``. In *semantic* mode (for the prompt's
    "current text"), the self @mention is dropped (routing, not content), an unknown mention renders as ``@`` and whitespace is collapsed."""
    raw = reply.get("content", {})
    content = _maybe_json(raw)
    if content is _BAD_JSON:
        return raw
    parts = []
    for elem in content.get("elements", []):
        etype = elem.get("type")
        uid = elem.get("person", {}).get("user_id", "" if semantic else "unknown") if etype == "person" else None
        if etype in _ELEMENT_TEXT_KEYS:
            parts.append(elem.get(etype, {}).get(_ELEMENT_TEXT_KEYS[etype], ""))
        elif etype == "person" and not (semantic and self_open_id and uid == self_open_id):
            parts.append(f"@{uid}")
    text = "".join(parts)
    return " ".join(text.split()).strip() if semantic else text


def _get_reply_user_id(reply: Dict[str, Any]) -> str:
    """Extract user_id from a reply dict."""
    user_id = reply.get("user_id", "")
    return (user_id.get("open_id", "") or user_id.get("user_id", "")) if isinstance(user_id, dict) else str(user_id)


def _reply_list_replies(whole_comment: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the ``reply_list.replies`` of a whole comment (``reply_list`` may be a JSON string)."""
    return _maybe_json(whole_comment.get("reply_list", {}), {}).get("replies", [])


def _extract_docs_links(replies: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Extract unique ``{"url", "doc_type", "token"}`` document links from comment replies."""
    seen_tokens = set()
    links = []
    for reply in replies:
        content = _maybe_json(reply.get("content", {}))
        if content is _BAD_JSON:
            continue
        for elem in content.get("elements", []):
            url = (elem.get("docs_link") or elem.get("link") or {}).get("url", "") if elem.get("type") in {"docs_link", "link"} else ""
            m = _FEISHU_DOC_URL_RE.search(url) if url else None
            if m and m.group("token") not in seen_tokens:
                seen_tokens.add(m.group("token"))
                links.append({"url": url, "doc_type": m.group("doc_type"), "token": m.group("token")})
    return links


async def _wiki_node(client: Any, queries: list, fail_msg: str, *fail_args) -> Optional[dict]:
    """GET a wiki node; logs *fail_msg* with ``(code, msg, *fail_args)`` and returns None on API failure."""
    code, msg, data = await _exec_request(client, "GET", _WIKI_GET_NODE_URI, queries=queries)
    return logger.warning(fail_msg, code, msg, *fail_args) if code != 0 else data.get("node", {})


async def _reverse_lookup_wiki_token(client: Any, obj_type: str, obj_token: str) -> Optional[str]:
    """Return the wiki node_token owning *obj_token*, or None if not a wiki doc / API failure."""
    node = await _wiki_node(client, [("token", obj_token), ("obj_type", obj_type)], "[Feishu-Comment] Wiki reverse lookup failed: code=%s msg=%s obj=%s:%s", obj_type, obj_token)
    return (node.get("node_token", "") or None) if node is not None else None


async def _resolve_wiki_nodes(client: Any, links: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Annotate wiki links in-place with ``resolved_type``/``resolved_token``; non-wiki links untouched."""
    for link in (l for l in links if l["doc_type"] == "wiki"):
        wiki_token = link["token"]
        node = await _wiki_node(client, [("token", wiki_token)], "[Feishu-Comment] Wiki resolve failed: code=%s msg=%s token=%s", wiki_token)
        if node is None:
            continue
        resolved_type, resolved_token = node.get("obj_type", ""), node.get("obj_token", "")
        if resolved_type and resolved_token:
            logger.info("[Feishu-Comment] Wiki resolved: %s -> %s:%s", wiki_token, resolved_type, resolved_token)
            link["resolved_type"], link["resolved_token"] = resolved_type, resolved_token
        else:
            logger.warning("[Feishu-Comment] Wiki resolve returned empty: %s", wiki_token)
    return links


def _format_referenced_docs(links: List[Dict[str, str]], current_file_token: str = "") -> str:
    """Format resolved document links for prompt embedding."""
    lines = ["", "Referenced documents in comments:"]
    for link in links:
        rtype, rtoken = link.get("resolved_type", link["doc_type"]), link.get("resolved_token", link["token"])
        lines.append(f"- {rtype}:{rtoken}{' (same as current document)' if rtoken == current_file_token else ''} ({link['url'][:80]})")
    return "\n".join(lines) if links else ""


async def _referenced_docs_text(client: Any, replies: List[Dict[str, Any]], file_token: str) -> str:
    """Extract, wiki-resolve and format the document links found in *replies*."""
    doc_links = _extract_docs_links(replies)
    return _format_referenced_docs(await _resolve_wiki_nodes(client, doc_links) if doc_links else doc_links, file_token)


def _truncate(text: str, limit: int = _PROMPT_TEXT_LIMIT) -> str:
    """Truncate text for prompt embedding."""
    return text if len(text) <= limit else text[:limit] + "..."


def _select_timeline(timeline: Timeline, limit: int, center: int, pinned: Tuple[int, ...] = ()) -> Timeline:
    """Select up to *limit* entries: *pinned* + *center*, then expand outward from *center*. Out-of-range indices are ignored; if nothing is selectable, falls back
    to the last *limit* entries."""
    n = len(timeline)
    if n <= limit:
        return timeline
    selected = {i for i in (*pinned, center) if 0 <= i < n}
    lo, hi = center - 1, center + 1
    while len(selected) < limit and (lo >= 0 or hi < n):
        if lo >= 0:
            selected.add(lo)
        if len(selected) < limit and hi < n:
            selected.add(hi)
        lo, hi = lo - 1, hi + 1
    return [timeline[i] for i in sorted(selected)] if selected else timeline[-limit:]


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


def _build_prompt(intro: List[str], doc_url: str, file_type: str, file_token: str, ids: List[str], label: str, timeline: Timeline, selected: Timeline, referenced_docs: str) -> str:
    """Intro lines + document block + ``label`` timeline header, the selected entries, referenced docs and common instructions."""
    lines = [*intro, _MENTION_NOTE, f"Document link: {doc_url}", "Current commented document:", f"- file_type={file_type}", f"- file_token={file_token}",
             *ids, "", f"{label} ({len(selected)}/{len(timeline)} entries):"]
    lines += [f"[{user_id}] {_truncate(text)}{' <-- YOU' if is_self else ''}" for user_id, text, is_self in selected]
    if referenced_docs:
        lines.append(referenced_docs)
    return "\n".join(lines + ["", _COMMON_INSTRUCTIONS])


def build_local_comment_prompt(
    *, doc_title: str, doc_url: str, file_token: str, file_type: str, comment_id: str, quote_text: str,
    root_comment_text: str, target_reply_text: str, timeline: Timeline, self_open_id: str,
    target_index: int = -1, referenced_docs: str = "",
) -> str:
    """Build the prompt for a local (quoted-text) comment."""
    intro = [f'The user added a reply in "{doc_title}".', f'Current user comment text: "{_truncate(target_reply_text)}"',
             f'Original comment text: "{_truncate(root_comment_text)}"', f'Quoted content: "{_truncate(quote_text, 500)}"']
    selected = _select_timeline(timeline, _LOCAL_TIMELINE_LIMIT, target_index, pinned=(0, len(timeline) - 1))
    return _build_prompt(intro, doc_url, file_type, file_token, [f"- comment_id={comment_id}"], "Current comment card timeline", timeline, selected, referenced_docs)


def build_whole_comment_prompt(
    *, doc_title: str, doc_url: str, file_token: str, file_type: str, comment_text: str, timeline: Timeline,
    self_open_id: str, current_index: int = -1, nearest_self_index: int = -1, referenced_docs: str = "",
) -> str:
    """Build the prompt for a whole-document comment."""
    intro = [f'The user added a comment in "{doc_title}".', f'Current user comment text: "{_truncate(comment_text)}"', "This is a whole-document comment."]
    selected = _select_timeline(timeline, _WHOLE_TIMELINE_LIMIT, current_index, pinned=(nearest_self_index,))
    return _build_prompt(intro, doc_url, file_type, file_token, [], "Whole-document comment timeline", timeline, selected, referenced_docs)


def _resolve_model_and_runtime() -> Tuple[str, dict]:
    """Resolve model and provider credentials, same as gateway message handling."""
    from gateway.run import _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs
    model = _resolve_gateway_model(_load_gateway_config())
    runtime_kwargs = _resolve_runtime_agent_kwargs()
    try:
        if not model and runtime_kwargs.get("provider"):  # fall back to the provider's default model
            from hermes_cli.models import get_default_model_for_provider
            model = get_default_model_for_provider(runtime_kwargs["provider"])
    except Exception:
        pass
    return model, runtime_kwargs


def _session_key(file_type: str, file_token: str) -> str:
    return f"comment-doc:{file_type}:{file_token}"


def _load_session_history(key: str) -> List[Dict[str, Any]]:
    """Load conversation history for a document session (expires after ``_SESSION_TTL_S``)."""
    with _session_cache_lock:
        entry = _session_cache.get(key)
        if entry is not None and time.time() - entry["last_access"] > _SESSION_TTL_S:
            del _session_cache[key]
            logger.info("[Feishu-Comment] Session expired: %s", key)
            return []
        if entry is not None:
            entry["last_access"] = time.time()
        return list(entry["messages"]) if entry is not None else []


def _save_session_history(key: str, messages: List[Dict[str, Any]]) -> None:
    """Save the last N user/assistant messages (system messages and tool internals stripped)."""
    cleaned = [m for m in messages if m.get("role") in {"user", "assistant"} and m.get("content")][-_SESSION_MAX_MESSAGES:]
    with _session_cache_lock:
        _session_cache[key] = {"messages": cleaned, "last_access": time.time()}
        logger.info("[Feishu-Comment] Session saved: %s (%d messages)", key, len(cleaned))


def _run_comment_agent(prompt: str, client: Any, session_key: str = "") -> str:
    """Create an AIAgent with feishu tools and run the prompt; empty string on failure. *session_key*, if given, loads/saves history for cross-card memory in the same document."""
    from run_agent import AIAgent
    from tools import feishu_doc_tool, feishu_drive_tool
    logger.info("[Feishu-Comment] _run_comment_agent: injecting lark client into tool thread-locals")
    tool_mods = (feishu_doc_tool, feishu_drive_tool)
    for mod in tool_mods:
        mod.set_client(client)
    try:
        model, runtime_kwargs = _resolve_model_and_runtime()
        logger.info("[Feishu-Comment] _run_comment_agent: model=%s provider=%s base_url=%s", model, runtime_kwargs.get("provider"), (runtime_kwargs.get("base_url") or "")[:50])
        history = _load_session_history(session_key) if session_key else []
        if history:
            logger.info("[Feishu-Comment] _run_comment_agent: loaded %d history messages from session %s", len(history), session_key)
        agent = AIAgent(model=model, **{k: runtime_kwargs.get(k) for k in ("base_url", "api_key", "provider", "api_mode", "credential_pool")},
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=15, enabled_toolsets=["feishu_doc", "feishu_drive"])
        logger.info("[Feishu-Comment] _run_comment_agent: calling run_conversation (prompt=%d chars, history=%d)", len(prompt), len(history))
        result = agent.run_conversation(prompt, conversation_history=history or None)
        response = (result.get("final_response") or "").strip()
        logger.info("[Feishu-Comment] _run_comment_agent: done api_calls=%d response_len=%d response=%s", result.get("api_calls", 0), len(response), response[:200])
        if session_key and result.get("messages", []):
            _save_session_history(session_key, result["messages"])
        return response
    except Exception as e:
        logger.exception("[Feishu-Comment] _run_comment_agent: agent failed: %s", e)
        return ""
    finally:
        for mod in tool_mods:
            mod.set_client(None)


def _last_index_where(timeline: Timeline, pred) -> Optional[Tuple[str, int]]:
    """Return ``(text, index)`` of the last timeline entry matching *pred*, or None."""
    return next(((timeline[i][1], i) for i in range(len(timeline) - 1, -1, -1) if pred(timeline[i])), None)


def _timeline_entry(r: Dict[str, Any], self_open_id: str) -> Tuple[str, str, bool]:
    uid = _get_reply_user_id(r)
    return uid, _extract_reply_text(r), (uid == self_open_id) if self_open_id else False


async def _whole_comment_prompt(client: Any, from_open_id: str, doc: dict) -> str:
    """Build the prompt for a whole-document comment from all whole comments on the doc. *doc* = build_*_prompt's shared kwargs (doc_title, doc_url, file_token,
    file_type, self_open_id)."""
    file_token, file_type, self_open_id = doc["file_token"], doc["file_type"], doc["self_open_id"]
    logger.info("[Feishu-Comment] Fetching whole-document comments for timeline...")
    whole_comments = await list_whole_comments(client, file_token, file_type)
    all_raw_replies: List[Dict[str, Any]] = [r for wc in whole_comments for r in _reply_list_replies(wc)]
    timeline: Timeline = [_timeline_entry(r, self_open_id) for r in all_raw_replies]
    current_text, current_index, nearest_self_index = "", -1, -1
    for idx, (r, (uid, _, is_self)) in enumerate(zip(all_raw_replies, timeline)):
        if uid == from_open_id:
            current_text, current_index = _extract_reply_text(r, semantic=True, self_open_id=self_open_id), idx
        if is_self:
            nearest_self_index = idx
    if not current_text and (found := _last_index_where(timeline, lambda e: not e[2])):
        current_text, current_index = found
    logger.info("[Feishu-Comment] Whole timeline: %d entries, current_idx=%d, self_idx=%d, text=%s",
                len(timeline), current_index, nearest_self_index, current_text[:80] if current_text else "(empty)")
    return build_whole_comment_prompt(comment_text=current_text, timeline=timeline, current_index=current_index, nearest_self_index=nearest_self_index,
                                      referenced_docs=await _referenced_docs_text(client, all_raw_replies, file_token), **doc)


async def _local_comment_prompt(client: Any, comment_id: str, reply_id: str, from_open_id: str, quote_text: str, doc: dict) -> str:
    """Build the prompt for a local comment from its thread replies (*doc* as in _whole_comment_prompt)."""
    file_token, file_type, self_open_id = doc["file_token"], doc["file_type"], doc["self_open_id"]
    logger.info("[Feishu-Comment] Fetching comment thread replies...")
    replies = await list_comment_replies(client, file_token, file_type, comment_id, expect_reply_id=reply_id)
    timeline: Timeline = [_timeline_entry(r, self_open_id) for r in replies]
    root_text = _extract_reply_text(replies[0], semantic=True, self_open_id=self_open_id) if replies else ""
    hits = [(_extract_reply_text(r, semantic=True, self_open_id=self_open_id), i) for i, r in enumerate(replies) if reply_id and r.get("reply_id", "") == reply_id]
    target_text, target_index = hits[-1] if hits else ("", -1)
    if not target_text and (found := _last_index_where(timeline, lambda e: e[0] == from_open_id)):
        target_text, target_index = found
    logger.info("[Feishu-Comment] Local timeline: %d entries, target_idx=%d, quote=%s root=%s target=%s",
                len(timeline), target_index, *(t[:60] if t else "(empty)" for t in (quote_text, root_text, target_text)))
    return build_local_comment_prompt(comment_id=comment_id, quote_text=quote_text, root_comment_text=root_text, target_reply_text=target_text, timeline=timeline,
                                      target_index=target_index, referenced_docs=await _referenced_docs_text(client, replies, file_token), **doc)


async def handle_drive_comment_event(client: Any, data: Any, *, self_open_id: str = "") -> None:
    """Full orchestration for a drive comment event. Parse + filter (self-reply, receiver, notice_type) -> access rules -> OK reaction -> parallel fetch (doc meta +
    comment) -> build timeline/prompt by is_whole -> run agent -> deliver reply -> remove OK reaction."""
    logger.info("[Feishu-Comment] ========== handle_drive_comment_event START ==========")
    parsed = parse_drive_comment_event(data)
    if parsed is None:
        return logger.warning("[Feishu-Comment] Dropping malformed drive comment event")
    logger.info("[Feishu-Comment] [Step 0/5] Event parsed successfully")
    file_token, file_type, comment_id, reply_id, from_open_id, to_open_id, notice_type = (
        parsed[k] for k in ("file_token", "file_type", "comment_id", "reply_id", "from_open_id", "to_open_id", "notice_type"))
    for skip, level, fmt, arg in (  # ordered early-exit filters
        (from_open_id and self_open_id and from_open_id == self_open_id, logging.DEBUG, "[Feishu-Comment] Skipping self-authored event: from=%s", from_open_id),
        (not to_open_id or (self_open_id and to_open_id != self_open_id), logging.DEBUG, "[Feishu-Comment] Skipping event not addressed to self: to=%s", to_open_id or "(empty)"),
        (notice_type and notice_type not in _ALLOWED_NOTICE_TYPES, logging.DEBUG, "[Feishu-Comment] Skipping notice_type=%s", notice_type),
        (not file_token or not file_type or not comment_id, logging.WARNING, "[Feishu-Comment] Missing required fields, skipping", None),
    ):
        if skip:
            return logger.log(level, fmt, *([arg] if arg is not None else []))
    logger.info("[Feishu-Comment] Event: notice=%s file=%s:%s comment=%s from=%s", notice_type, file_type, file_token, comment_id, from_open_id)
    # Access control. Wiki-hosted docs report their underlying obj token, so when no exact rule
    # matched and the config has wiki: keys, reverse-lookup the wiki node.
    from plugins.platforms.feishu.feishu_comment_rules import load_config, resolve_rule, is_user_allowed, has_wiki_keys
    comments_cfg = load_config()
    rule = resolve_rule(comments_cfg, file_type, file_token)
    if rule.match_source in {"wildcard", "top"} and has_wiki_keys(comments_cfg) and (wiki_token := await _reverse_lookup_wiki_token(client, file_type, file_token)):
        rule = resolve_rule(comments_cfg, file_type, file_token, wiki_token=wiki_token)
    if not rule.enabled:
        return logger.info("[Feishu-Comment] Comments disabled for %s:%s, skipping", file_type, file_token)
    if not is_user_allowed(rule, from_open_id):
        return logger.info("[Feishu-Comment] User %s denied (policy=%s, rule=%s)", from_open_id, rule.policy, rule.match_source)
    logger.info("[Feishu-Comment] Access granted: user=%s policy=%s rule=%s", from_open_id, rule.policy, rule.match_source)
    reaction_kwargs = dict(file_token=file_token, file_type=file_type, reply_id=reply_id, reaction_type="OK")
    if reply_id:
        asyncio.ensure_future(update_comment_reaction(client, "add", **reaction_kwargs))
    logger.info("[Feishu-Comment] [Step 2/5] Parallel fetch: doc meta + comment batch_query")
    doc_meta, comment_detail = await asyncio.gather(asyncio.ensure_future(query_document_meta(client, file_token, file_type)),
                                                    asyncio.ensure_future(batch_query_comment(client, file_token, file_type, comment_id)))
    doc = dict(doc_title=doc_meta.get("title", "Untitled"), doc_url=doc_meta.get("url", ""), file_token=file_token, file_type=file_type, self_open_id=self_open_id)
    is_whole = bool(comment_detail.get("is_whole"))
    logger.info("[Feishu-Comment] Comment context: title=%s is_whole=%s", doc["doc_title"], is_whole)
    logger.info("[Feishu-Comment] [Step 3/5] Building timeline (is_whole=%s)", is_whole)
    prompt = await (_whole_comment_prompt(client, from_open_id, doc) if is_whole
                    else _local_comment_prompt(client, comment_id, reply_id, from_open_id, comment_detail.get("quote", ""), doc))
    logger.info("[Feishu-Comment] [Step 4/5] Prompt built (%d chars), running agent...", len(prompt))
    logger.debug("[Feishu-Comment] Full prompt:\n%s", prompt)
    # run_conversation is synchronous -> thread. Session key groups all comment cards on one doc.
    response = await asyncio.get_running_loop().run_in_executor(None, _run_comment_agent, prompt, client, _session_key(file_type, file_token))
    if not response or _NO_REPLY_SENTINEL in response:
        logger.info("[Feishu-Comment] Agent returned NO_REPLY, skipping delivery")
    else:
        logger.info("[Feishu-Comment] Agent response (%d chars): %s", len(response), response[:200])
        logger.info("[Feishu-Comment] [Step 5/5] Delivering reply (is_whole=%s, comment_id=%s)", is_whole, comment_id)
        delivered = await deliver_comment_reply(client, file_token, file_type, comment_id, response, is_whole)
        logger.log(logging.INFO if delivered else logging.ERROR, "[Feishu-Comment] Reply delivered successfully" if delivered else "[Feishu-Comment] Failed to deliver reply")
    if reply_id:  # best-effort cleanup of the OK reaction
        await update_comment_reaction(client, "delete", **reaction_kwargs)
    logger.info("[Feishu-Comment] ========== handle_drive_comment_event END ==========")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
