"""QQ (OneBot) model-facing tools.

Model tools for the OneBot channel (ported from dsh-onebot, PORTING #1):
  - qq_send_image / voice / video / file / forward — outbound media
  - qq_napcat_api — whitelisted NapCat action proxy
  - qq_group_history — pull group message history

Transport is the adapter's local HTTP API (default 127.0.0.1:8643; override
with ONEBOT_TOOL_BASE), so these work from any process — CLI/TUI sessions
included, where the gateway adapter may not be running. Target chat resolves
from an explicit chat_id argument, falling back to the calling session's
HERMES_SESSION_CHAT_ID when this is a QQ-channel session.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BASE = os.environ.get("ONEBOT_TOOL_BASE", "http://127.0.0.1:8643").rstrip("/")

# OneBot actions a model may invoke through qq_napcat_api (read-only queries,
# file URL resolution, uploads, OCR, AI chat) — mirrors dsh-onebot's list.
NAPCAT_API_WHITELIST = [
    "get_group_member_list",
    "get_group_member_info",
    "get_stranger_info",
    "get_forward_msg",
    "get_record",
    "get_file",
    "upload_group_file",
    "upload_private_file",
    "get_group_root_files",
    "get_group_files_by_folder",
    "get_group_file_url",
    "get_private_file_url",
    "ocr_image",
    "get_ai_characters",
    "send_group_ai_record",
]

HTTP_TIMEOUT = 60


def _resolve_chat(chat_id: Optional[str]) -> str:
    """Explicit chat_id → session chat id (QQ-channel only) → error."""
    if chat_id:
        return chat_id
    platform = os.environ.get("HERMES_SESSION_PLATFORM", "") or ""
    if platform.lower() == "onebot":
        session_chat = os.environ.get("HERMES_SESSION_CHAT_ID", "") or ""
        if session_chat:
            return session_chat
    raise ValueError(
        "无法确定目标会话：请传 chat_id（格式 private:<qq> 或 group:<群号>）"
    )


def _http(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = _BASE + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", "replace")
            return json.loads(body or "{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        return {"ok": False, "error": f"HTTP {e.code}: {detail}"}
    except Exception as e:  # noqa: BLE001 - surface any transport failure
        return {"ok": False, "error": str(e)}


def _media_result(resp: Dict[str, Any]) -> str:
    if resp.get("ok"):
        mid = resp.get("message_id")
        return "已发送" + (f"（message_id={mid}）" if mid else "")
    return "发送失败：" + str(resp.get("error") or "未知错误")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def qq_send_image(args: Dict[str, Any], **kw) -> str:
    """Send one or more images (local paths or http(s) URLs, max 9)."""
    sources = args.get("sources") or []
    sources = [s for s in sources if isinstance(s, str) and s]
    if not sources:
        return "qq_send_image: sources 不能为空"
    if len(sources) > 9:
        return "qq_send_image: 一次最多 9 张图片"
    chat_id = _resolve_chat(args.get("chat_id"))
    resp = _http(
        "POST",
        "/api/send_media",
        {"chat_id": chat_id, "kind": "image", "sources": sources, "caption": args.get("caption") or ""},
    )
    return _media_result(resp)


def qq_send_voice(args: Dict[str, Any], **kw) -> str:
    path = args.get("path") or ""
    if not path:
        return "qq_send_voice: path 必填"
    chat_id = _resolve_chat(args.get("chat_id"))
    resp = _http("POST", "/api/send_media", {"chat_id": chat_id, "kind": "voice", "path": path})
    return _media_result(resp)


def qq_send_video(args: Dict[str, Any], **kw) -> str:
    path = args.get("path") or ""
    if not path:
        return "qq_send_video: path 必填"
    chat_id = _resolve_chat(args.get("chat_id"))
    resp = _http("POST", "/api/send_media", {"chat_id": chat_id, "kind": "video", "path": path})
    return _media_result(resp)


def qq_send_file(args: Dict[str, Any], **kw) -> str:
    path = args.get("path") or ""
    if not path:
        return "qq_send_file: path 必填"
    chat_id = _resolve_chat(args.get("chat_id"))
    resp = _http(
        "POST",
        "/api/send_media",
        {
            "chat_id": chat_id,
            "kind": "file",
            "path": path,
            "file_name": args.get("file_name") or "",
        },
    )
    return _media_result(resp)


def qq_send_forward(args: Dict[str, Any], **kw) -> str:
    """Send a merged forward (QQ chat bubble) with named node messages."""
    nodes = args.get("nodes") or []
    if not isinstance(nodes, list) or not nodes:
        return "qq_send_forward: nodes 不能为空（[{name, content}, ...]）"
    chat_id = _resolve_chat(args.get("chat_id"))
    cleaned = []
    for n in nodes:
        if isinstance(n, dict) and (n.get("name") or n.get("content")):
            cleaned.append(
                {"name": str(n.get("name") or "Hermes"), "content": str(n.get("content") or "")}
            )
    if not cleaned:
        return "qq_send_forward: nodes 为空"
    resp = _http(
        "POST",
        "/api/send_media",
        {"chat_id": chat_id, "kind": "forward", "nodes": cleaned},
    )
    return _media_result(resp)


def qq_napcat_api(args: Dict[str, Any], **kw) -> str:
    """Proxy a whitelisted NapCat OneBot action."""
    action = str(args.get("action") or "")
    if action not in NAPCAT_API_WHITELIST:
        return (
            f"action {action!r} 不在白名单。可用："
            + ", ".join(NAPCAT_API_WHITELIST)
        )
    params = args.get("params") or {}
    if not isinstance(params, dict):
        return "qq_napcat_api: params 必须是对象"
    qs = urllib.parse.urlencode({"action": action, "params": json.dumps(params)})
    resp = _http("GET", f"/api/napcat?{qs}")
    if resp.get("ok"):
        return json.dumps(resp.get("data"), ensure_ascii=False)[:4000]
    return "调用失败：" + str(resp.get("error") or "未知错误")


def qq_group_history(args: Dict[str, Any], **kw) -> str:
    """Pull recent messages from a group (summary / monitoring)."""
    group_id = args.get("group_id") or ""
    if not group_id:
        return "qq_group_history: group_id 必填"
    count = int(args.get("count") or 20)
    count = max(1, min(count, 50))
    qs = urllib.parse.urlencode(
        {"group_id": group_id, "count": count, "message_seq": args.get("message_seq") or ""}
    )
    resp = _http("GET", f"/api/group_history?{qs}")
    if resp.get("ok"):
        return json.dumps(resp.get("data"), ensure_ascii=False)[:8000]
    return "拉取失败：" + str(resp.get("error") or "未知错误")


_HANDLERS: Dict[str, Any] = {
    "qq_send_image": qq_send_image,
    "qq_send_voice": qq_send_voice,
    "qq_send_video": qq_send_video,
    "qq_send_file": qq_send_file,
    "qq_send_forward": qq_send_forward,
    "qq_napcat_api": qq_napcat_api,
    "qq_group_history": qq_group_history,
}


def _str_prop(name: str, desc: str, required: bool = False) -> Dict[str, Any]:
    schema: Dict[str, Any] = {"type": "string", "description": desc}
    if required:
        schema["required"] = True
    return schema


def _schemas() -> Dict[str, Dict[str, Any]]:
    chat_id = {
        "type": "string",
        "description": "目标会话，格式 private:<qq> 或 group:<群号>；省略时默认当前会话",
    }
    return {
        "qq_send_image": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "图片来源：本地绝对路径或 http(s) URL，最多 9 个",
                },
                "chat_id": chat_id,
                "caption": {"type": "string", "description": "图片说明文字（可选，作为图片前的文本）"},
            },
            "required": ["sources"],
        },
        "qq_send_voice": {
            "type": "object",
            "properties": {
                "path": _str_prop("path", "本地音频文件绝对路径（mp3/wav/silk 等）", required=True),
                "chat_id": chat_id,
            },
            "required": ["path"],
        },
        "qq_send_video": {
            "type": "object",
            "properties": {
                "path": _str_prop("path", "本地视频文件绝对路径", required=True),
                "chat_id": chat_id,
            },
            "required": ["path"],
        },
        "qq_send_file": {
            "type": "object",
            "properties": {
                "path": _str_prop("path", "本地文件绝对路径", required=True),
                "file_name": _str_prop("file_name", "发送时显示的文件名（可选）"),
                "chat_id": chat_id,
            },
            "required": ["path"],
        },
        "qq_send_forward": {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": _str_prop("name", "节点昵称"),
                            "content": _str_prop("content", "节点文本内容"),
                        },
                    },
                    "description": "合并转发节点列表，每个 {name, content}；仿 QQ 聊天记录",
                },
                "chat_id": chat_id,
            },
            "required": ["nodes"],
        },
        "qq_napcat_api": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "白名单 action："
                    + ", ".join(NAPCAT_API_WHITELIST),
                },
                "params": {"type": "object", "description": "action 参数对象"},
            },
            "required": ["action"],
        },
        "qq_group_history": {
            "type": "object",
            "properties": {
                "group_id": _str_prop("group_id", "群号", required=True),
                "count": {"type": "integer", "description": "拉取条数（1-50，默认 20）"},
                "message_seq": {"type": "integer", "description": "翻页游标（可选）"},
            },
            "required": ["group_id"],
        },
    }


_DESCRIPTIONS = {
    "qq_send_image": "向 QQ 会话发送一张或多张图片（本地文件路径或 http(s) URL，最多 9 张）。",
    "qq_send_voice": "向 QQ 会话发送一条语音消息（本地音频文件路径，NapCat 负责转码）。",
    "qq_send_video": "向 QQ 会话发送一条视频消息（本地视频文件绝对路径）。",
    "qq_send_file": "向 QQ 会话发送一个文件（本地文件绝对路径）。",
    "qq_send_forward": "向 QQ 会话发送一条合并转发（多条节点消息组成的聊天记录卡片）。",
    "qq_napcat_api": "调用白名单内的 NapCat OneBot action（成员查询/文件/OCR/AI 等）。",
    "qq_group_history": "拉取 QQ 群最近消息历史，用于群总结或监控。",
}


def register_tools(ctx) -> None:
    """Register the qq_* tools in the ``qq`` toolset."""
    for name, schema in _schemas().items():
        ctx.register_tool(
            name=name,
            toolset="qq",
            schema=schema,
            handler=_HANDLERS[name],
            description=_DESCRIPTIONS[name],
            emoji="\U0001f427",  # penguin
        )