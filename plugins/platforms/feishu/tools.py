"""The Feishu comment agent's model tools: ``feishu_doc_read`` (toolset ``feishu_doc``) and the four
``feishu_drive_*`` comment tools (toolset ``feishu_drive``).

The plugin loader imports this file at discovery and calls ``register_tools(ctx)`` because
``plugin.yaml`` declares ``provides_tools``, so the tools exist in every process while the adapter stays
deferred. Requests go through the generic tenant-token ``BaseRequest`` path with a lazy ``lark_oapi`` import,
the same pattern as feishu_comment.py.
"""

import json

from tools.registry import tool_error, tool_result


def get_client():
    """Return the lark client of the comment being answered, or None outside a comment run."""
    # By its absolute name: the loader imports this file as ``hermes_plugins.<slug>.tools``, and only the
    # absolute name reaches the module object whose context variable _run_comment_agent sets.
    from plugins.platforms.feishu.feishu_comment import current_client
    return current_client.get()


def _check_feishu():
    # find_spec checks importability without executing lark_oapi's __init__, which
    # eagerly loads websockets/dispatcher/every api model (~5s). This probe fires at
    # every ``hermes`` startup; the handlers still do the real import when invoked.
    import importlib.util
    try:
        return importlib.util.find_spec("lark_oapi") is not None
    except (ImportError, ValueError):
        return False


def build_request(method, uri, paths=None, queries=None, body=None):
    """Build a tenant-token BaseRequest. Raises ImportError if lark_oapi is missing."""
    from lark_oapi import AccessTokenType
    from lark_oapi.core.enum import HttpMethod
    from lark_oapi.core.model.base_request import BaseRequest

    builder = (
        BaseRequest.builder()
        .http_method(HttpMethod.GET if method == "GET" else HttpMethod.POST)
        .uri(uri)
        .token_types({AccessTokenType.TENANT}))
    if paths:
        builder = builder.paths(paths)
    if queries:
        builder = builder.queries(queries)
    if body is not None:
        builder = builder.body(body)
    return builder.build()


def lark_call(client, method, uri, paths=None, queries=None, body=None):
    """Build + execute a BaseRequest; returns (code, msg, data_dict).

    Tool handlers run synchronously in a worker thread (no running event loop), so the
    blocking lark client is called directly.
    """
    response = client.request(build_request(method, uri, paths, queries, body))
    return getattr(response, "code", None), getattr(response, "msg", ""), response_data(response)


def raw_body(response):
    """Parsed JSON object of the raw HTTP body, or None when absent/unparseable/not a dict."""
    raw = getattr(response, "raw", None)
    try:
        body = json.loads(raw.content) if raw and hasattr(raw, "content") else None
    except (json.JSONDecodeError, AttributeError):
        return None
    return body if isinstance(body, dict) else None


def response_data(response) -> dict:
    """``data`` of a lark response: prefer the raw JSON body, fall back to typed .data."""
    body = raw_body(response)
    data = body.get("data", {}) if body is not None else {}
    if data:
        return data
    resp_data = getattr(response, "data", None)
    if isinstance(resp_data, dict):
        return resp_data
    return vars(resp_data) if resp_data and hasattr(resp_data, "__dict__") else data


# --- feishu_doc_read ---

_RAW_CONTENT_URI = "/open-apis/docx/v1/documents/:document_id/raw_content"

FEISHU_DOC_READ_SCHEMA = {
    "name": "feishu_doc_read",
    "description": (
        "Read the full content of a Feishu/Lark document as plain text. "
        "Useful when you need more context beyond the quoted text in a comment."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc_token": {
                "type": "string",
                "description": "The document token (from the document URL or comment context).",
            },
        },
        "required": ["doc_token"],
    },
}


def _handle_feishu_doc_read(args: dict, **kwargs) -> str:
    doc_token = args.get("doc_token", "").strip()
    if not doc_token:
        return tool_error("doc_token is required")
    client = get_client()
    if client is None:
        return tool_error("Feishu client not available (not in a Feishu comment context)")
    try:
        request = build_request("GET", _RAW_CONTENT_URI, paths={"document_id": doc_token})
    except ImportError:
        return tool_error("lark_oapi not installed")

    # Handlers run synchronously in a worker thread (no event loop): call the blocking client.
    response = client.request(request)
    code = getattr(response, "code", None)
    if code != 0:
        return tool_error(f"Failed to read document: code={code} msg={getattr(response, 'msg', 'unknown error')}")

    body = raw_body(response)
    if body is not None:
        try:
            return tool_result(success=True, content=body.get("data", {}).get("content", ""))
        except AttributeError:
            pass
    data = getattr(response, "data", None)  # fallback: the typed response.data
    if data:
        content = data.get("content", "") if isinstance(data, dict) else getattr(data, "content", str(data))
        return tool_result(success=True, content=content)
    return tool_error("No content returned from document API")


# --- feishu_drive_* comment tools ---

def _comment_op(keys, missing_msg, label, method, uri, queries=None, body=None, flag_success=False):
    """Handler factory: client check, then required ``keys`` (stripped) → lark_call → result.

    All keys except ``content`` are URI path params. ``queries(args)`` / ``body(args, values)``
    build the request pieces; ``flag_success`` adds ``success=True`` to the result (writes).
    """
    def _handler(args: dict, **kwargs) -> str:
        client = get_client()
        values = tuple(args.get(k, "").strip() for k in keys)
        if client is None:
            return tool_error("Feishu client not available")
        if not all(values):
            return tool_error(missing_msg)
        code, msg, data = lark_call(
            client, method, uri, paths={k: v for k, v in zip(keys, values) if k != "content"},
            queries=queries and queries(args), body=body and body(args, values))
        if code != 0:
            return tool_error(f"{label} failed: code={code} msg={msg}")
        return tool_result(success=True, data=data) if flag_success else tool_result(data)
    return _handler


def _file_type(args: dict) -> str:
    return args.get("file_type", "docx") or "docx"


def _paged_queries(args: dict, *extra) -> list:
    """Query params shared by the listing endpoints; page_token goes last (after any extra)."""
    queries = [
        ("file_type", _file_type(args)), ("user_id_type", "open_id"),
        ("page_size", str(args.get("page_size", 100))), *extra]
    page_token = args.get("page_token", "")
    if page_token:
        queries.append(("page_token", page_token))
    return queries


_FILE_TOKEN_PROP = {"type": "string", "description": "The document file token."}
_FILE_TYPE_PROP = {"type": "string", "description": "File type (default: docx).", "default": "docx"}
_PAGE_TOKEN_PROP = {"type": "string", "description": "Pagination token for next page."}
_COMMENTS_URI = "/open-apis/drive/v1/files/:file_token/comments"
_REPLIES_URI = "/open-apis/drive/v1/files/:file_token/comments/:comment_id/replies"
_ADD_COMMENT_URI = "/open-apis/drive/v1/files/:file_token/new_comments"


FEISHU_DRIVE_LIST_COMMENTS_SCHEMA = {
    "name": "feishu_drive_list_comments",
    "description": (
        "List comments on a Feishu document. "
        "Use is_whole=true to list whole-document comments only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "file_type": _FILE_TYPE_PROP,
            "is_whole": {
                "type": "boolean",
                "description": "If true, only return whole-document comments.",
                "default": False,
            },
            "page_size": {
                "type": "integer",
                "description": "Number of comments per page (max 100).",
                "default": 100,
            },
            "page_token": _PAGE_TOKEN_PROP,
        },
        "required": ["file_token"],
    },
}


FEISHU_DRIVE_LIST_REPLIES_SCHEMA = {
    "name": "feishu_drive_list_comment_replies",
    "description": "List all replies in a comment thread on a Feishu document.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "comment_id": {
                "type": "string",
                "description": "The comment ID to list replies for.",
            },
            "file_type": _FILE_TYPE_PROP,
            "page_size": {
                "type": "integer",
                "description": "Number of replies per page (max 100).",
                "default": 100,
            },
            "page_token": _PAGE_TOKEN_PROP,
        },
        "required": ["file_token", "comment_id"],
    },
}


FEISHU_DRIVE_REPLY_SCHEMA = {
    "name": "feishu_drive_reply_comment",
    "description": (
        "Reply to a local comment thread on a Feishu document. "
        "Use this for local (quoted-text) comments. "
        "For whole-document comments, use feishu_drive_add_comment instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "comment_id": {
                "type": "string",
                "description": "The comment ID to reply to.",
            },
            "content": {
                "type": "string",
                "description": "The reply text content (plain text only, no markdown).",
            },
            "file_type": _FILE_TYPE_PROP,
        },
        "required": ["file_token", "comment_id", "content"],
    },
}


FEISHU_DRIVE_ADD_COMMENT_SCHEMA = {
    "name": "feishu_drive_add_comment",
    "description": (
        "Add a new whole-document comment on a Feishu document. "
        "Use this for whole-document comments or as a fallback when "
        "reply_comment fails with code 1069302."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_token": _FILE_TOKEN_PROP,
            "content": {
                "type": "string",
                "description": "The comment text content (plain text only, no markdown).",
            },
            "file_type": _FILE_TYPE_PROP,
        },
        "required": ["file_token", "content"],
    },
}


_handle_list_comments = _comment_op(
    ("file_token",), "file_token is required", "List comments", "GET", _COMMENTS_URI,
    queries=lambda a: _paged_queries(a, *([("is_whole", "true")] if a.get("is_whole", False) else [])))
_handle_list_replies = _comment_op(
    ("file_token", "comment_id"), "file_token and comment_id are required", "List replies", "GET",
    _REPLIES_URI, queries=_paged_queries)
# Replies use the rich "content.elements[text_run]" body shape; file_type is a query param.
_handle_reply_comment = _comment_op(
    ("file_token", "comment_id", "content"), "file_token, comment_id, and content are required",
    "Reply comment", "POST", _REPLIES_URI, queries=lambda a: [("file_type", _file_type(a))],
    body=lambda a, v: {"content": {"elements": [{"type": "text_run", "text_run": {"text": v[-1]}}]}},
    flag_success=True)
# new_comments takes the flat "reply_elements[text]" shape with file_type in the body.
_handle_add_comment = _comment_op(
    ("file_token", "content"), "file_token and content are required", "Add comment", "POST",
    _ADD_COMMENT_URI,
    body=lambda a, v: {"file_type": _file_type(a), "reply_elements": [{"type": "text", "text": v[-1]}]},
    flag_success=True)


_TOOLS = (
    # (schema, handler, toolset, description, emoji)
    (FEISHU_DOC_READ_SCHEMA, _handle_feishu_doc_read, "feishu_doc", "Read Feishu document content", "\U0001f4c4"),
    (FEISHU_DRIVE_LIST_COMMENTS_SCHEMA, _handle_list_comments, "feishu_drive", "List document comments",
     "\U0001f4ac"),
    (FEISHU_DRIVE_LIST_REPLIES_SCHEMA, _handle_list_replies, "feishu_drive", "List comment replies", "\U0001f4ac"),
    (FEISHU_DRIVE_REPLY_SCHEMA, _handle_reply_comment, "feishu_drive", "Reply to a document comment",
     "\u2709\ufe0f"),
    (FEISHU_DRIVE_ADD_COMMENT_SCHEMA, _handle_add_comment, "feishu_drive", "Add a whole-document comment",
     "\u2709\ufe0f"),
)


def register_tools(ctx) -> None:
    """Register the five comment-agent tools in the ``feishu_doc`` and ``feishu_drive`` toolsets."""
    for schema, handler, toolset, description, emoji in _TOOLS:
        ctx.register_tool(
            name=schema["name"], toolset=toolset, schema=schema, handler=handler,
            check_fn=_check_feishu, requires_env=[], is_async=False, description=description, emoji=emoji)
