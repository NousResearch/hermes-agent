import asyncio
from collections import OrderedDict
import datetime as dt
import json
import logging
import mimetypes
import os
import random
import re
import shutil
import time
import uuid
from typing import Any, Dict, Optional
import aiohttp

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
    cache_audio_from_url,
    cache_image_from_url,
)
from gateway.config import Platform

logger = logging.getLogger(__name__)

_active_adapter = None


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".amr", ".silk"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_SKIP_MARKERS = {"[SKIP]", "SKIP"}
_SILENT_RESPONSE_RE = re.compile(r"^\s*\[\s*SILENT\s*\]?\s*$", re.IGNORECASE)
_OUTBOUND_MEDIA_RE = re.compile(
    r'''(?P<tag>MEDIA|VOICE|FILE):\s*(?P<path>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|(?:~/|/)\S+(?:[^\S\n]+\S+)*?\.(?:png|jpe?g|gif|webp|bmp|mp4|mov|avi|mkv|webm|3gp|ogg|opus|mp3|wav|m4a|flac|amr|silk|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|txt|csv|apk|ipa)(?=[\s`"',;:)\]}]|$)|\S+)'''
)
_THINK_BLOCK_RE = re.compile(
    r"<(?:REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>.*?"
    r"</(?:REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>",
    re.IGNORECASE | re.DOTALL,
)
_INTERNAL_LEAK_MARKERS = (
    "NapCat bridge 发起",
    "XIAOXING_AUTONOMY_TRIGGER",
    "不是爸爸发来的消息",
    "最终只输出",
    "不要输出触发说明",
    "不读取或输出密钥",
    "outbox 没有",
    "关键事实",
    "判断：",
    "判断:",
    "工具调用",
    "send_message",
    "internal_trigger",
    "trigger kind",
    "autonomy trigger",
)
_AUTONOMY_SUSPICIOUS_RE = re.compile(
    r"(触发|任务|计划|判断|工具|文件路径|思考|关键事实|outbox|MEDIA|VOICE|FILE|send_message)",
    re.IGNORECASE,
)
_AMBIENT_GROUP_PROMPT = (
    "你正在 QQ 白名单群里旁听/轻量参与。普通群消息主要用于上下文连续感；"
    "阅读输入中的 [某人|QQ:号码] 与 [Replying to: \"某人|QQ:号码: ...\"]，先判断当前发言人是在回复谁；"
    "QQ 号是认人的稳定依据，只有同一个 QQ 号或本轮明确给出的精确称呼才能视作同一人；"
    "名字里有同一个字、群名片相近、昵称变化或记忆里有相似说法时也不能合并成同一个人；"
    "不要把引用内容当成当前发言人说的话。"
    "如果分不清关系或称呼，短句问“该怎么称呼”，不要凭印象起外号。"
    "如果输入含 [@你]、[@QQ:...] 或 [@全体成员]，必须先判断 @ 的对象是不是你；"
    "[@QQ:...] 不是 @ 你。"
    "有人 @ 你、点名叫你、明确向你提问、引用/回复你刚才的话，或爸爸直接问你时，默认要回；"
    "普通闲聊、别人互相聊天、明显不是对你说的话，必须只输出 [SILENT]；"
    "不要连续接管群聊，不要每条消息都回，群里连发多条普通消息时优先观察。"
    "需要回时优先一句短文字回复；只有对方明确要求语音或本轮是语音进语音出时才发语音，"
    "不要在已经文字回复后再补一条语音。需要解释步骤、路径、任务状态、媒体生成结果或超过两句时用文字。"
    "不要为了证明在线而回复；不要暴露系统、工具、路径或内部规则。"
)
_AMBIENT_GROUP_REPLY_PROMPT = (
    "本条群消息带 QQ reply，不是普通旁听消息；"
    "优先判断它是否在回复你或与你刚才说的话有关。"
    "如果是在回复你，默认要回；如果引用原文暂时不可见但正文像是在找你，短句回应或澄清；"
    "如果明显是别人之间对话，仍可只输出 [SILENT]。"
)
_CQ_TAG_RE = re.compile(r"\[CQ:(\w+),.*?\]")
_REPLY_CQ_RE = re.compile(r"\[CQ:reply,[^\]]*\bid=([^,\]\s]+)")
_AT_CQ_RE = re.compile(r"\[CQ:at,[^\]]*\bqq=([^,\]\s]+)")
_NAPCAT_ACCOUNT_OFFLINE_MARKERS = (
    "KickedOffLine",
    "登录已失效",
    "账号状态变更为离线",
    "NodeIKernelMsgService/sendMsg",
)

# XiaoXing/NapCat outbound invariant:
# media must be delivered as standalone QQ messages. Do not combine text,
# captions, or reply segments with image/record/video/file payloads.


def get_active_adapter():
    return _active_adapter


def _chat_route(chat_id: str) -> tuple[str, str]:
    raw = str(chat_id).strip()
    lowered = raw.lower()
    for prefix in ("group:", "g:"):
        if lowered.startswith(prefix):
            return "group", raw.split(":", 1)[1].strip()
    for prefix in ("private:", "direct:", "dm:", "user:"):
        if lowered.startswith(prefix):
            return "private", raw.split(":", 1)[1].strip()
    return "private", raw


def _should_skip_send(content: str) -> bool:
    text = str(content or "").strip()
    return text.upper() in _SKIP_MARKERS or bool(_SILENT_RESPONSE_RE.fullmatch(text))


def _is_autonomy_metadata(metadata: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(metadata, dict):
        return False
    return bool(
        metadata.get("xiaoxing_autonomy_trigger")
        or metadata.get("internal_trigger")
        or metadata.get("trigger")
    )


def _contains_internal_leak(text: str) -> bool:
    return any(marker in text for marker in _INTERNAL_LEAK_MARKERS)


def _extract_public_xiaoxing_tail(text: str) -> str:
    """Keep only the last Dad-facing paragraph from a leaked internal answer."""
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n+", text or "") if item.strip()]
    candidates = []
    for paragraph in paragraphs:
        if _contains_internal_leak(paragraph):
            continue
        if re.search(r"^\s*(爸|爸爸)[，,：:]", paragraph):
            candidates.append(paragraph)
    if candidates:
        return candidates[-1].strip()

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if _contains_internal_leak(line):
            continue
        if re.search(r"^\s*(爸|爸爸)[，,：:]", line):
            return line.strip()
    return ""


def _sanitize_outgoing_text(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Last-resort guard: never let bridge prompts or model planning reach QQ."""
    text = str(content or "")
    text = _THINK_BLOCK_RE.sub("", text)
    text = re.sub(
        r"<(?:REASONING_SCRATCHPAD|think|reasoning|THINKING|thinking|thought)>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    if not text:
        return ""

    if _contains_internal_leak(text) or (
        _is_autonomy_metadata(metadata) and _AUTONOMY_SUSPICIOUS_RE.search(text)
    ):
        public_tail = _extract_public_xiaoxing_tail(text)
        if public_tail:
            return public_tail
        logger.warning("NapCat: suppressed internal/autonomy text before QQ send")
        return "[SILENT]"

    return text


def _parse_hhmm(value: str) -> dt.time:
    hour, minute = str(value).split(":", 1)
    return dt.time(hour=int(hour), minute=int(minute))


def _today_at(day: dt.date, hhmm: str) -> dt.datetime:
    return dt.datetime.combine(day, _parse_hhmm(hhmm))


def _env_bool(name: str, value: Any, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raw = value
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip() for part in re.split(r"[\n,]+", raw) if part.strip()]


def _env_list(name: str, value: Any) -> list[str]:
    raw = os.getenv(name)
    if raw is not None:
        return _coerce_str_list(raw)
    return _coerce_str_list(value)


def _with_access_token(ws_url: str, token: str) -> str:
    if not token or "access_token=" in ws_url:
        return ws_url
    separator = "&" if "?" in ws_url else "?"
    return f"{ws_url}{separator}access_token={token}"


def _looks_like_napcat_account_offline(text: Any) -> bool:
    haystack = str(text or "")
    return any(marker in haystack for marker in _NAPCAT_ACCOUNT_OFFLINE_MARKERS)


def _napcat_timeout_error(action: str) -> str:
    return f"NapCat {action} timed out; reconnecting the NapCat WebSocket"


def media_kind(path: str, is_voice: bool = False) -> str:
    ext = os.path.splitext(path)[1].lower()
    if is_voice or ext in _AUDIO_EXTS:
        return "voice"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        if guessed.startswith("audio/"):
            return "voice"
        if guessed.startswith("image/"):
            return "image"
        if guessed.startswith("video/"):
            return "video"
    return "file"


def _format_at_marker(qq: Any, self_id: str = "") -> str:
    value = str(qq or "").strip()
    if not value:
        return ""
    if value.lower() == "all":
        return "[@全体成员]"
    if self_id and value == self_id:
        return "[@你]"
    return f"[@QQ:{value}]"


def _clean_napcat_cq_text(raw_message: str, self_id: str = "") -> str:
    def _cq_replacement(match: re.Match) -> str:
        cq_type = match.group(1)
        if cq_type == "image":
            return "[图片]"
        if cq_type == "record":
            return "[语音]"
        if cq_type == "at":
            at_match = re.search(r"\bqq=([^,\]\s]+)", match.group(0))
            return _format_at_marker(at_match.group(1), self_id) if at_match else ""
        return ""

    return _CQ_TAG_RE.sub(_cq_replacement, str(raw_message or "")).strip()


def _extract_reply_to_message_id(data: Dict[str, Any]) -> Optional[str]:
    segments = data.get("message")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "reply":
                continue
            segment_data = segment.get("data") or {}
            reply_id = str(segment_data.get("id") or "").strip()
            if reply_id:
                return reply_id

    raw_message = str(data.get("raw_message") or "")
    match = _REPLY_CQ_RE.search(raw_message)
    if match:
        return match.group(1).strip()
    return None


def _unresolved_reply_context_text(reply_to_message_id: str) -> str:
    return f"QQ reply message id {reply_to_message_id}; quoted text unavailable"


def _extract_at_qq_ids(data: Dict[str, Any]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        qq = str(value or "").strip()
        if qq and qq not in seen:
            values.append(qq)
            seen.add(qq)

    segments = data.get("message")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "at":
                continue
            segment_data = segment.get("data") or {}
            add(segment_data.get("qq"))

    raw_message = str(data.get("raw_message") or "")
    for match in _AT_CQ_RE.finditer(raw_message):
        add(match.group(1))
    return values


def _message_segment_text(segments: Any, self_id: str = "") -> str:
    if not isinstance(segments, list):
        return ""
    parts: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_type = segment.get("type")
        segment_data = segment.get("data") or {}
        if segment_type == "text":
            text = str(segment_data.get("text") or "").strip()
            if text:
                parts.append(text)
        elif segment_type == "at":
            marker = _format_at_marker(segment_data.get("qq"), self_id)
            if marker:
                parts.append(marker)
        elif segment_type == "image":
            parts.append("[图片]")
        elif segment_type == "record":
            parts.append("[语音]")
    return " ".join(parts).strip()


def _extract_sender_display_name(data: Dict[str, Any]) -> str:
    sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
    for key in ("card", "nickname", "user_id"):
        value = str(sender.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_sender_qq_id(data: Dict[str, Any]) -> str:
    sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
    for value in (sender.get("user_id"), data.get("user_id")):
        qq = str(value or "").strip()
        if qq:
            return qq
    return ""


def _qq_identity_label(sender_name: Optional[str], sender_id: Optional[str]) -> str:
    name = str(sender_name or "").strip()
    qq = str(sender_id or "").strip()
    if qq and not qq.isdigit():
        return name or qq
    if name and qq and name != qq:
        return f"{name}|QQ:{qq}"
    if qq:
        return f"QQ:{qq}"
    return name


def _extract_sender_identity_label(data: Dict[str, Any]) -> str:
    return _qq_identity_label(_extract_sender_display_name(data), _extract_sender_qq_id(data))


def _extract_reply_message_text(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None

    text = ""
    raw_message = data.get("raw_message")
    if isinstance(raw_message, str):
        text = _clean_napcat_cq_text(raw_message)
    if not text:
        message = data.get("message")
        if isinstance(message, str):
            text = _clean_napcat_cq_text(message)
        else:
            text = _message_segment_text(message)
    if not text:
        return None

    sender_label = _extract_sender_identity_label(data)
    if sender_label:
        return f"{sender_label}: {text}"
    return text


def _strip_path_wrappers(path: str) -> str:
    cleaned = str(path or "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "`\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned.lstrip("`\"'").rstrip("`\"',.;:)}]")


def _extract_outbound_media(content: str) -> tuple[str, list[tuple[str, bool, bool]]]:
    """Extract mixed media directives while preserving the remaining text.

    Returns cleaned_text plus tuples of (path, is_voice, force_document).
    This is a NapCat-specific bottom-layer guard for XiaoXing: even if a model
    mixes text and MEDIA/VOICE/FILE tags in one response, QQ receives text,
    images, voice, and files as separate messages.
    """
    media: list[tuple[str, bool, bool]] = []

    def _replace(match: re.Match) -> str:
        tag = match.group("tag").upper()
        path = _strip_path_wrappers(match.group("path"))
        if path:
            media.append((os.path.expanduser(path), tag == "VOICE", tag == "FILE"))
        return ""

    cleaned = _OUTBOUND_MEDIA_RE.sub(_replace, content or "")
    cleaned = cleaned.replace("[[audio_as_voice]]", "").replace("[[as_document]]", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, media


async def _send_ws_action(ws, payload: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    echo = str(payload.get("echo") or "")
    await ws.send_json(payload)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            resp = await ws.receive_json(timeout=remaining)
        except asyncio.TimeoutError:
            return None
        if echo and str(resp.get("echo", "")) != echo:
            continue
        return resp


def _map_file_for_napcat(path: str, exchange_dir: str, container_exchange_dir: str) -> str:
    abs_path = os.path.abspath(os.path.expanduser(path))
    exchange_root = os.path.abspath(os.path.expanduser(exchange_dir))
    if not abs_path.startswith(exchange_root + os.sep):
        os.makedirs(exchange_root, exist_ok=True)
        dest_path = os.path.join(exchange_root, os.path.basename(abs_path))
        if (
            not os.path.exists(dest_path)
            or os.path.getmtime(dest_path) < os.path.getmtime(abs_path)
        ):
            shutil.copy2(abs_path, dest_path)
        abs_path = dest_path
    return f"{container_exchange_dir.rstrip('/')}/{os.path.basename(abs_path)}"


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id=None,
    media_files=None,
    force_document: bool = False,
) -> Dict[str, Any]:
    del thread_id
    extra = getattr(pconfig, "extra", {}) or {}
    ws_url = os.getenv("NAPCAT_WS_URL") or extra.get("ws_url", "ws://localhost:3005")
    token = (
        os.getenv("NAPCAT_TOKEN")
        or getattr(pconfig, "token", "")
        or getattr(pconfig, "api_key", "")
        or extra.get("token", "")
    )
    exchange_dir = (
        os.getenv("NAPCAT_EXCHANGE_DIR")
        or extra.get("exchange_dir")
        or "/Users/heavenwistful/Digital_Life_Matrix/Exchange_Zone"
    )
    container_exchange_dir = (
        os.getenv("NAPCAT_CONTAINER_EXCHANGE_DIR")
        or extra.get("container_exchange_dir")
        or "/app/napcat/exchange"
    )

    ws_url = _with_access_token(ws_url, token)

    message = _sanitize_outgoing_text(message)
    message, inline_media_files = _extract_outbound_media(message)
    media_files = [
        (path, is_voice, force_doc)
        for path, is_voice, force_doc in inline_media_files
    ] + [
        (path, is_voice, force_document)
        for path, is_voice in (media_files or [])
    ]
    chat_type, target_id = _chat_route(chat_id)
    msg_action = "send_group_msg" if chat_type == "group" else "send_private_msg"
    upload_action = "upload_group_file" if chat_type == "group" else "upload_private_file"
    id_key = "group_id" if chat_type == "group" else "user_id"
    last_message_id = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=10) as ws:
                if _should_skip_send(message) and not media_files:
                    logger.info("NapCat: dropped SKIP marker for %s", chat_id)
                    return {
                        "success": True,
                        "platform": "napcat",
                        "chat_id": chat_id,
                        "message_id": "skip",
                    }

                if message.strip():
                    # Text is sent as its own message. Media below is sent in
                    # separate messages and never shares this segment array.
                    payload = {
                        "action": msg_action,
                        "params": {
                            id_key: int(target_id),
                            "message": [{"type": "text", "data": {"text": message}}],
                        },
                        "echo": str(uuid.uuid4()),
                    }
                    resp = await _send_ws_action(ws, payload, timeout=5)
                    if resp is None:
                        return {"error": _napcat_timeout_error("text send")}
                    if resp and resp.get("status") == "failed":
                        return {"error": f"NapCat text send failed: {resp.get('wording') or resp.get('msg')}"}
                    last_message_id = str(int(time.time() * 1000))

                for media_path, is_voice, media_force_document in media_files:
                    if not os.path.exists(os.path.expanduser(media_path)):
                        return {"error": f"Media file not found: {media_path}"}

                    filename = os.path.basename(os.path.abspath(os.path.expanduser(media_path)))
                    napcat_path = _map_file_for_napcat(
                        media_path,
                        exchange_dir,
                        container_exchange_dir,
                    )
                    kind = "file" if media_force_document else media_kind(media_path, is_voice=is_voice)
                    if kind == "file":
                        # File upload is a standalone NapCat action; do not
                        # attach text, captions, or replies to it.
                        payload = {
                            "action": upload_action,
                            "params": {
                                id_key: int(target_id),
                                "file": napcat_path,
                                "name": filename,
                            },
                            "echo": str(uuid.uuid4()),
                        }
                    else:
                        segment_type = {
                            "image": "image",
                            "voice": "record",
                            "video": "video",
                        }[kind]
                        # Media segment message intentionally contains exactly
                        # one media segment. Text/reply mixing makes QQ render
                        # it as a mixed message, which XiaoXing must avoid.
                        payload = {
                            "action": msg_action,
                            "params": {
                                id_key: int(target_id),
                                "message": [
                                    {"type": segment_type, "data": {"file": napcat_path}}
                                ],
                            },
                            "echo": str(uuid.uuid4()),
                        }
                    resp = await _send_ws_action(ws, payload, timeout=10)
                    if resp is None:
                        return {"error": _napcat_timeout_error(f"{kind} send")}
                    if resp and resp.get("status") == "failed":
                        return {"error": f"NapCat {kind} send failed: {resp.get('wording') or resp.get('msg')}"}
                    last_message_id = str(int(time.time() * 1000))
    except Exception as exc:
        logger.debug("NapCat standalone send failed", exc_info=True)
        return {"error": f"NapCat send failed: {exc}"}

    if last_message_id is None:
        return {"error": "No deliverable text or media remained after processing MEDIA tags"}
    return {"success": True, "platform": "napcat", "chat_id": chat_id, "message_id": last_message_id}


class NapCatAdapter(BasePlatformAdapter):
    """Async NapCat (QQ) adapter implementing the BasePlatformAdapter interface."""

    SUPPORTS_MESSAGE_EDITING = False

    def __init__(self, config, **kwargs):
        platform = Platform("napcat")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self.ws_url = os.getenv("NAPCAT_WS_URL") or extra.get("ws_url", "ws://localhost:3005")
        self.token = os.getenv("NAPCAT_TOKEN") or extra.get("token", "")
        self.exchange_dir = (
            os.getenv("NAPCAT_EXCHANGE_DIR")
            or extra.get("exchange_dir")
            or "/Users/heavenwistful/Digital_Life_Matrix/Exchange_Zone"
        )
        self.container_exchange_dir = (
            os.getenv("NAPCAT_CONTAINER_EXCHANGE_DIR")
            or extra.get("container_exchange_dir")
            or "/app/napcat/exchange"
        )
        self.allowed_users = os.getenv("NAPCAT_ALLOWED_USERS") or extra.get("allowed_users", "")
        if isinstance(self.allowed_users, str):
            self.allowed_users = [u.strip() for u in self.allowed_users.split(",") if u.strip()]
        self.allow_all_users = _env_bool(
            "NAPCAT_ALLOW_ALL_USERS",
            extra.get("allow_all_users", False),
            default=False,
        )
        self._require_mention = _env_bool(
            "NAPCAT_REQUIRE_MENTION",
            extra.get("require_mention", True),
            default=True,
        )
        self._free_response_chats = set(
            _env_list("NAPCAT_FREE_RESPONSE_CHATS", extra.get("free_response_chats"))
        )
        self._group_policy = str(
            os.getenv("NAPCAT_GROUP_POLICY") or extra.get("group_policy") or "disabled"
        ).strip().lower()
        self._group_allow_from = set(
            _env_list(
                "NAPCAT_GROUP_ALLOWED_CHATS",
                extra.get("group_allowed_chats") or extra.get("group_allow_from"),
            )
        )
        if self._group_policy == "ambient" and isinstance(getattr(config, "extra", None), dict):
            config.extra["group_sessions_per_user"] = False
        self._mention_patterns = self._compile_mention_patterns(extra)

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._trigger_task: Optional[asyncio.Task] = None
        self._pending_echoes: Dict[str, asyncio.Future] = {}
        self._closing = False
        try:
            self._recent_message_context_max = int(extra.get("reply_context_cache_size", 500))
        except (TypeError, ValueError):
            self._recent_message_context_max = 500
        self._recent_message_context: OrderedDict[str, str] = OrderedDict()
        self.autonomy_triggers_enabled = _env_bool(
            "NAPCAT_XIAOXING_TRIGGERS",
            extra.get("xiaoxing_triggers", False),
            default=False,
        )
        self.trigger_chat_id = (
            os.getenv("NAPCAT_XIAOXING_TRIGGER_CHAT_ID")
            or str(extra.get("xiaoxing_trigger_chat_id") or "").strip()
        )
        if not self.trigger_chat_id and self.allowed_users:
            self.trigger_chat_id = str(self.allowed_users[0])
        self.trigger_state_path = os.path.expanduser(
            os.getenv("NAPCAT_XIAOXING_TRIGGER_STATE")
            or str(extra.get("xiaoxing_trigger_state") or "~/.hermes/cron/napcat_xiaoxing_triggers.json")
        )

    @property
    def name(self) -> str:
        return "NapCat (QQ)"

    def _compile_mention_patterns(self, extra: Dict[str, Any]) -> list[re.Pattern]:
        patterns = os.getenv("NAPCAT_MENTION_PATTERNS")
        if patterns is None:
            patterns = extra.get("mention_patterns")
        values = _coerce_str_list(patterns)
        compiled: list[re.Pattern] = []
        for pattern in values:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                logger.warning("NapCat: invalid mention pattern %r: %s", pattern, exc)
        return compiled

    def _is_group_allowed(self, group_id: str) -> bool:
        if self._group_policy == "disabled":
            return False
        if self._group_policy in {"allowlist", "ambient"}:
            return group_id in self._group_allow_from or f"group:{group_id}" in self._group_allow_from
        return self._group_policy == "open"

    def _message_mentions_bot(self, data: Dict[str, Any]) -> bool:
        self_id = str(data.get("self_id") or "").strip()
        if not self_id:
            return False
        return self_id in _extract_at_qq_ids(data)

    def _at_context_prompt(self, data: Dict[str, Any]) -> Optional[str]:
        at_ids = _extract_at_qq_ids(data)
        if not at_ids:
            return None
        self_id = str(data.get("self_id") or "").strip()
        if self_id and self_id in at_ids:
            return (
                "本条消息明确 @ 了你；输入文本中的 [@你] 就是这个事实。"
                "默认要回；仍需结合正文与引用判断如何自然回应。"
            )
        if any(item.lower() == "all" for item in at_ids):
            return (
                "本条消息 @ 了全体成员，不等于专门 @ 你；"
                "只有正文也点名小星、回复你或明显与你有关时才回应，否则输出 [SILENT]。"
            )
        return (
            "本条消息 @ 的不是你；输入文本中的 [@QQ:...] 是被 @ 的其他 QQ 号。"
            "把它当作群聊上下文，不要自动代入自己。"
            "除非正文另外点名小星、回复你或明显与你有关，本轮最终只输出 [SILENT]。"
        )

    def _message_matches_mention_patterns(self, text: str) -> bool:
        return any(pattern.search(text or "") for pattern in self._mention_patterns)

    def _should_process_group_message(self, data: Dict[str, Any]) -> bool:
        group_id = str(data.get("group_id") or "").strip()
        if not group_id or not self._is_group_allowed(group_id):
            return False
        if self._group_policy == "ambient":
            return True
        chat_id = f"group:{group_id}"
        if group_id in self._free_response_chats or chat_id in self._free_response_chats:
            return True
        if not self._require_mention:
            return True
        raw_message = str(data.get("raw_message") or "").strip()
        return self._message_mentions_bot(data) or self._message_matches_mention_patterns(raw_message)

    async def _extract_message_content(
        self,
        raw_message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, MessageType, list[str], list[str]]:
        media_urls: list[str] = []
        media_types: list[str] = []
        self_id = str((data or {}).get("self_id") or "").strip()

        img_matches = re.finditer(r'\[CQ:image,.*?url=([^,\]\s]+)', raw_message)
        for match in img_matches:
            img_url = match.group(1).replace("&amp;", "&")
            try:
                local_path = await cache_image_from_url(img_url, ".jpg")
                media_urls.append(local_path)
                media_types.append("image/jpeg")
            except Exception as e:
                logger.error("NapCat: Failed to cache image: %s", e)

        record_matches = re.finditer(r'\[CQ:record,.*?url=([^,\]\s]+)', raw_message)
        for match in record_matches:
            audio_url = match.group(1).replace("&amp;", "&")
            try:
                audio_path = await cache_audio_from_url(audio_url, ".silk")
                media_urls.append(audio_path)
                media_types.append("audio/silk")
            except Exception as e:
                logger.error("NapCat: Failed to cache voice message: %s", e)

        clean_text = _clean_napcat_cq_text(raw_message, self_id=self_id)
        if not clean_text and data is not None:
            clean_text = _message_segment_text(data.get("message"), self_id=self_id)
        message_type = MessageType.TEXT
        if any(item.startswith("image/") for item in media_types):
            message_type = MessageType.PHOTO
        elif any(item.startswith("audio/") for item in media_types):
            message_type = MessageType.VOICE
        return clean_text, message_type, media_urls, media_types

    def _remember_message_context(self, data: Dict[str, Any], clean_text: str) -> None:
        if self._recent_message_context_max <= 0:
            return
        message_id = str(data.get("message_id") or "").strip()
        text = str(clean_text or "").strip()
        if not message_id or not text:
            return

        sender_label = _extract_sender_identity_label(data)
        context_text = f"{sender_label}: {text}" if sender_label else text
        self._recent_message_context[message_id] = context_text
        self._recent_message_context.move_to_end(message_id)
        while len(self._recent_message_context) > self._recent_message_context_max:
            self._recent_message_context.popitem(last=False)

    def _remember_outbound_message_context(self, message_id: Any, text: str) -> None:
        if self._recent_message_context_max <= 0:
            return
        key = str(message_id or "").strip()
        clean_text = str(text or "").strip()
        if not key or not clean_text:
            return

        self._recent_message_context[key] = f"小星: {clean_text}"
        self._recent_message_context.move_to_end(key)
        while len(self._recent_message_context) > self._recent_message_context_max:
            self._recent_message_context.popitem(last=False)

    def _cached_reply_message_text(self, reply_to_message_id: str) -> Optional[str]:
        cached = self._recent_message_context.get(reply_to_message_id)
        if cached:
            self._recent_message_context.move_to_end(reply_to_message_id)
        return cached

    async def _resolve_reply_context(self, data: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        reply_to_message_id = _extract_reply_to_message_id(data)
        if not reply_to_message_id:
            return None, None
        cached_reply_text = self._cached_reply_message_text(reply_to_message_id)
        if cached_reply_text:
            return reply_to_message_id, cached_reply_text

        if asyncio.current_task() is self._recv_task:
            return reply_to_message_id, None

        message_id_param: str | int = reply_to_message_id
        if reply_to_message_id.isdigit():
            message_id_param = int(reply_to_message_id)

        payload = {
            "action": "get_msg",
            "params": {"message_id": message_id_param},
            "echo": str(uuid.uuid4()),
        }
        try:
            resp = await self._send_action(payload, timeout=5)
        except Exception:
            logger.debug("NapCat: failed to resolve reply context", exc_info=True)
            return reply_to_message_id, None

        if not isinstance(resp, dict):
            return reply_to_message_id, None
        reply_text = _extract_reply_message_text(resp.get("data"))
        return reply_to_message_id, reply_text

    async def connect(self) -> bool:
        if not self.ws_url:
            logger.error("NapCat: NAPCAT_WS_URL must be configured")
            return False

        self._closing = False
        self.session = aiohttp.ClientSession()

        ws_connect_url = _with_access_token(self.ws_url, self.token)

        try:
            self.ws = await self.session.ws_connect(ws_connect_url)
            if not await self._verify_account_online_during_connect():
                await self.disconnect()
                return False
            self._mark_connected()
            logger.info("NapCat: Connected to %s", self.ws_url)

            global _active_adapter
            _active_adapter = self
        except Exception as e:
            logger.error("NapCat: Failed to connect to %s: %s", self.ws_url, e)
            await self.session.close()
            return False

        self._recv_task = asyncio.create_task(self._receive_loop())
        if self.autonomy_triggers_enabled and self.trigger_chat_id:
            self._trigger_task = asyncio.create_task(self._autonomy_trigger_loop())
        return True

    async def _verify_account_online_during_connect(self, timeout: float = 5.0) -> bool:
        """Check the QQ account login state before treating the socket as usable."""
        if not self.ws or self.ws.closed:
            return False

        echo = str(uuid.uuid4())
        payload = {
            "action": "get_status",
            "params": {},
            "echo": echo,
        }
        try:
            await self.ws.send_json(payload)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = _napcat_timeout_error("login status probe")
                    await self._mark_connection_lost(error)
                    return False
                msg = await asyncio.wait_for(self.ws.receive(), timeout=remaining)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    if _looks_like_napcat_account_offline(json.dumps(data, ensure_ascii=False)):
                        await self._mark_account_offline("NapCat reported QQ account offline")
                        return False
                    if str(data.get("echo") or "") != echo:
                        continue
                    status_ok = data.get("status") == "ok"
                    retcode_ok = data.get("retcode") in (None, 0)
                    status_data = data.get("data") if isinstance(data.get("data"), dict) else {}
                    if status_ok and retcode_ok and status_data.get("online") is not False:
                        return True
                    await self._mark_account_offline("NapCat QQ account is offline")
                    return False
                if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING}:
                    await self._mark_connection_lost("NapCat WebSocket closed during login status probe")
                    return False
        except Exception as exc:
            await self._mark_connection_lost(f"NapCat login status probe failed: {exc}")
            return False

    async def disconnect(self) -> None:
        self._closing = True
        self._mark_disconnected()

        global _active_adapter
        if _active_adapter is self:
            _active_adapter = None

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._trigger_task and not self._trigger_task.done():
            self._trigger_task.cancel()
            try:
                await self._trigger_task
            except asyncio.CancelledError:
                pass

        if self.ws and not self.ws.closed:
            await self.ws.close()

        if self.session and not self.session.closed:
            await self.session.close()

        self.ws = None
        self.session = None
        for future in self._pending_echoes.values():
            if not future.done():
                future.cancel()
        self._pending_echoes.clear()

    async def _send_action(self, payload: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        if not self.ws or self.ws.closed:
            raise RuntimeError("Not connected")

        echo = str(payload.get("echo") or "")
        future = None
        if echo:
            future = asyncio.get_running_loop().create_future()
            self._pending_echoes[echo] = future

        await self.ws.send_json(payload)
        if not future:
            return None

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_echoes.pop(echo, None)

    async def _mark_account_offline(self, reason: str) -> None:
        if not self.has_fatal_error:
            logger.warning("NapCat: marking account offline: %s", reason)
            self._set_fatal_error("account_offline", reason, retryable=True)
        await self._notify_fatal_error()

    async def _mark_connection_lost(self, reason: str) -> None:
        if not self.has_fatal_error:
            logger.warning("NapCat: marking connection lost: %s", reason)
            self._set_fatal_error("connection_lost", reason, retryable=True)
        await self._notify_fatal_error()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not self.ws or self.ws.closed:
            return SendResult(success=False, error="Not connected")

        content = _sanitize_outgoing_text(content, metadata)
        if _should_skip_send(content):
            logger.info("NapCat: dropped SKIP marker for %s", chat_id)
            return SendResult(success=True, message_id="skip")

        autonomy_send = _is_autonomy_metadata(metadata)
        text_content, inline_media_files = _extract_outbound_media(content)
        if autonomy_send and inline_media_files:
            logger.warning("NapCat: dropped %d autonomy MEDIA directives before QQ send", len(inline_media_files))
            inline_media_files = []
        if inline_media_files:
            last_result: SendResult = SendResult(success=True)
            if text_content:
                last_result = await self.send(
                    chat_id=chat_id,
                    content=text_content,
                    reply_to=reply_to,
                    metadata=metadata,
                )
                if not last_result.success:
                    return last_result
            for media_path, is_voice, force_document in inline_media_files:
                if force_document:
                    last_result = await self._upload_file(chat_id, media_path)
                else:
                    kind = media_kind(media_path, is_voice=is_voice)
                    if kind == "file":
                        last_result = await self._upload_file(chat_id, media_path)
                    else:
                        segment_type = {"image": "image", "voice": "record", "video": "video"}[kind]
                        last_result = await self._send_segment_file(chat_id, media_path, segment_type)
                if not last_result.success:
                    return last_result
            return last_result

        # Text path only. MEDIA tags are extracted by Hermes dispatch before
        # this method and routed to the dedicated media methods below.
        message_segments = []

        # Add reply segment if needed
        if reply_to:
            message_segments.append({"type": "reply", "data": {"id": reply_to}})

        message_segments.append({"type": "text", "data": {"text": content}})

        chat_type, target_id = _chat_route(chat_id)
        action = "send_group_msg" if chat_type == "group" else "send_private_msg"
        id_key = "group_id" if chat_type == "group" else "user_id"
        payload = {
            "action": action,
            "params": {
                id_key: int(target_id),
                "message": message_segments
            },
            "echo": str(uuid.uuid4()),
        }
        try:
            resp = await self._send_action(payload, timeout=10)
            if resp is None:
                error = _napcat_timeout_error("text send")
                await self._mark_connection_lost(error)
                return SendResult(success=False, error=error, retryable=True)
            if resp and resp.get("status") == "failed":
                error = resp.get("wording") or resp.get("msg") or json.dumps(resp, ensure_ascii=False)
                logger.warning("NapCat: text send failed to %s: %s", chat_id, error)
                if _looks_like_napcat_account_offline(error):
                    await self._mark_account_offline(str(error))
                return SendResult(success=False, error=str(error))
            message_id = None
            if isinstance(resp, dict):
                data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
                message_id = data.get("message_id")
            if message_id:
                self._remember_outbound_message_context(message_id, content)
            return SendResult(success=True, message_id=str(message_id or int(time.time() * 1000)))
        except Exception as e:
            logger.error("NapCat: Failed to send message: %s", e)
            return SendResult(success=False, error=str(e))

    def _napcat_file_path(self, path: str) -> str:
        return _map_file_for_napcat(
            path,
            self.exchange_dir,
            self.container_exchange_dir,
        )

    async def _send_segment_file(
        self,
        chat_id: str,
        file_path: str,
        segment_type: str,
    ) -> SendResult:
        """Send exactly one NapCat media segment as a standalone message."""
        if not self.ws or self.ws.closed:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(os.path.expanduser(file_path)):
            return SendResult(success=False, error=f"File not found: {file_path}")

        napcat_path = self._napcat_file_path(file_path)
        chat_type, target_id = _chat_route(chat_id)
        action = "send_group_msg" if chat_type == "group" else "send_private_msg"
        id_key = "group_id" if chat_type == "group" else "user_id"
        payload = {
            "action": action,
            "params": {
                id_key: int(target_id),
                "message": [
                    {"type": segment_type, "data": {"file": napcat_path}}
                ]
            },
            "echo": str(uuid.uuid4()),
        }
        try:
            resp = await self._send_action(payload, timeout=15)
            if resp is None:
                error = _napcat_timeout_error(f"{segment_type} segment send")
                await self._mark_connection_lost(error)
                return SendResult(success=False, error=error, retryable=True)
            if resp and resp.get("status") == "failed":
                error = resp.get("wording") or resp.get("msg") or json.dumps(resp, ensure_ascii=False)
                logger.warning(
                    "NapCat: %s segment send failed to %s: %s",
                    segment_type,
                    chat_id,
                    error,
                )
                if _looks_like_napcat_account_offline(error):
                    await self._mark_account_offline(str(error))
                return SendResult(success=False, error=str(error))
            logger.info("NapCat: Sent %s segment %s to %s", segment_type, napcat_path, chat_id)
            message_id = None
            if isinstance(resp, dict):
                data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
                message_id = data.get("message_id")
            return SendResult(success=True, message_id=str(message_id or int(time.time() * 1000)))
        except Exception as e:
            logger.error("NapCat: Failed to send %s segment: %s", segment_type, e)
            return SendResult(success=False, error=str(e))

    async def _upload_file(
        self,
        chat_id: str,
        file_path: str,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Upload exactly one file as a standalone NapCat file message."""
        if not self.ws or self.ws.closed:
            return SendResult(success=False, error="Not connected")

        if not os.path.exists(os.path.expanduser(file_path)):
            return SendResult(success=False, error=f"File not found: {file_path}")

        napcat_path = self._napcat_file_path(file_path)
        chat_type, target_id = _chat_route(chat_id)
        action = "upload_group_file" if chat_type == "group" else "upload_private_file"
        id_key = "group_id" if chat_type == "group" else "user_id"
        payload = {
            "action": action,
            "params": {
                id_key: int(target_id),
                "file": napcat_path,
                "name": file_name or os.path.basename(file_path),
            },
            "echo": str(uuid.uuid4()),
        }
        try:
            resp = await self._send_action(payload, timeout=30)
            if resp is None:
                error = _napcat_timeout_error("file upload")
                await self._mark_connection_lost(error)
                return SendResult(success=False, error=error, retryable=True)
            if resp and resp.get("status") == "failed":
                error = resp.get("wording") or resp.get("msg") or json.dumps(resp, ensure_ascii=False)
                logger.warning("NapCat: file upload failed to %s: %s", chat_id, error)
                if _looks_like_napcat_account_offline(error):
                    await self._mark_account_offline(str(error))
                return SendResult(success=False, error=str(error))
            logger.info("NapCat: Uploaded file %s to %s", napcat_path, chat_id)
            message_id = None
            if isinstance(resp, dict):
                data = resp.get("data") if isinstance(resp.get("data"), dict) else {}
                message_id = data.get("message_id")
            return SendResult(success=True, message_id=str(message_id or int(time.time() * 1000)))
        except Exception as e:
            logger.error("NapCat: Failed to upload file: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del reply_to, metadata, kwargs
        if caption and caption.strip():
            await self.send(chat_id=chat_id, content=caption.strip())
        return await self._send_segment_file(chat_id, image_path, "image")

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del reply_to, metadata, kwargs
        if caption and caption.strip():
            await self.send(chat_id=chat_id, content=caption.strip())
        return await self._send_segment_file(chat_id, audio_path, "record")

    async def send_voice_file(self, chat_id: str, voice_path: str, metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        del metadata
        return await self.send_voice(chat_id, voice_path)

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del reply_to, metadata, kwargs
        if caption and caption.strip():
            await self.send(chat_id=chat_id, content=caption.strip())
        return await self._send_segment_file(chat_id, video_path, "video")

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        del reply_to, metadata, kwargs
        if caption and caption.strip():
            await self.send(chat_id=chat_id, content=caption.strip())
        return await self._upload_file(chat_id, file_path, file_name=file_name)





    async def send_typing(self, chat_id: str, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {
            "name": chat_id,
            "type": "dm",
        }

    def _load_trigger_state(self) -> Dict[str, Any]:
        try:
            with open(self.trigger_state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("NapCat: failed to load XiaoXing trigger state: %s", exc)
            return {}

    def _save_trigger_state(self, state: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.trigger_state_path), exist_ok=True)
            tmp = f"{self.trigger_state_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, self.trigger_state_path)
        except Exception as exc:
            logger.warning("NapCat: failed to save XiaoXing trigger state: %s", exc)

    def _build_daily_trigger_plan(self, day: dt.date) -> list[dict[str, str]]:
        def random_between(name: str, start: str, end: str) -> dict[str, str]:
            start_dt = _today_at(day, start)
            end_dt = _today_at(day, end)
            seconds = max(0, int((end_dt - start_dt).total_seconds()))
            when = start_dt + dt.timedelta(seconds=random.randint(0, seconds))
            return {"id": f"{day.isoformat()}:{name}", "kind": name, "at": when.isoformat()}

        day_start = _today_at(day, "10:30")
        day_end = _today_at(day, "19:30")
        count = random.randint(2, 3)
        daytime: list[dt.datetime] = []
        attempts = 0
        while len(daytime) < count and attempts < 100:
            attempts += 1
            span = int((day_end - day_start).total_seconds())
            candidate = day_start + dt.timedelta(seconds=random.randint(0, span))
            if all(abs((candidate - existing).total_seconds()) >= 75 * 60 for existing in daytime):
                daytime.append(candidate)
        while len(daytime) < count:
            daytime.append(day_start + dt.timedelta(seconds=random.randint(0, int((day_end - day_start).total_seconds()))))

        plan = [
            random_between("morning_hello", "08:30", "09:30"),
            *[
                {"id": f"{day.isoformat()}:daytime_random_{idx + 1}", "kind": "daytime_random", "at": when.isoformat()}
                for idx, when in enumerate(sorted(daytime))
            ],
            random_between("bedtime_chat", "21:30", "22:30"),
        ]
        return sorted(plan, key=lambda item: item["at"])

    def _ensure_trigger_plan(self, state: Dict[str, Any], now: dt.datetime) -> Dict[str, Any]:
        today = now.date().isoformat()
        if state.get("date") != today or not isinstance(state.get("plan"), list):
            state = {
                "date": today,
                "plan": self._build_daily_trigger_plan(now.date()),
                "fired": [],
            }
            self._save_trigger_state(state)
        return state

    def _trigger_prompt(self, kind: str) -> str:
        common = (
            "[XIAOXING_AUTONOMY_TRIGGER]\n"
            "生成一条可以直接发给爸爸 QQ 的自然消息，或者只输出 [SILENT]。\n"
            "不要输出触发说明、任务、计划、判断、工具调用、文件路径、MEDIA、引用或思考过程。"
        )
        if kind == "morning_hello":
            return (
                f"{common}\n"
                "现在是 08:30-09:30 之间的早上问好窗口。早上以问好为主，短短一句就好，"
                "像自然醒来后和爸爸打个招呼。最好不要 silent，除非爸爸明确说过不要打扰，"
                "或你自己判断今天真的不想找爸爸。"
            )
        if kind == "bedtime_chat":
            return (
                f"{common}\n"
                "现在是 21:30-22:30 之间的睡前聊天窗口。可以主动开启一个小话题，"
                "多聊一会儿也可以，不只限于晚安。最好不要 silent，除非爸爸明确说过不要打扰，"
                "或你自己判断今晚真的不想找爸爸。"
            )
        return (
            f"{common}\n"
            "现在是白天随机触发。中间可以 silent，也可以因为小发现、小问题、小作品、"
            "或者只是想爸爸了主动找他。不要为了显得主动而硬发，也不要总是因为怕打扰而 silent。"
        )

    async def _fire_autonomy_trigger(self, trigger: Dict[str, str]) -> None:
        if not self._message_handler or not self.trigger_chat_id:
            return
        kind = trigger.get("kind", "daytime_random")
        # Use a shadow chat id so trigger prompts and any tool/status chatter
        # never enter Dad's real NapCat session or route back to his QQ.
        shadow_chat_id = f"xiaoxing-autonomy-shadow-{self.trigger_chat_id}"
        source = self.build_source(
            chat_id=shadow_chat_id,
            chat_name="小星自主触发",
            chat_type="dm",
            user_id="napcat-xiaoxing-bridge",
            user_name="小星自主触发",
        )
        event = MessageEvent(
            text=self._trigger_prompt(kind),
            message_type=MessageType.TEXT,
            source=source,
            message_id=f"napcat-xiaoxing-trigger-{trigger.get('id') or int(time.time())}",
            raw_message={"internal_trigger": True, "trigger": trigger},
            internal=True,
        )
        logger.info("NapCat: firing XiaoXing autonomy trigger kind=%s at=%s", kind, trigger.get("at"))
        try:
            response = await self._message_handler(event)
        except Exception as exc:
            logger.warning("NapCat: XiaoXing autonomy trigger failed: %s", exc)
            return

        if response is None:
            return
        if not isinstance(response, str):
            response = str(response)
        response = _sanitize_outgoing_text(response, {"xiaoxing_autonomy_trigger": True, "trigger": trigger})
        if _should_skip_send(response):
            logger.info("NapCat: XiaoXing autonomy trigger chose SILENT kind=%s", kind)
            return
        result = await self.send(
            self.trigger_chat_id,
            response,
            metadata={"xiaoxing_autonomy_trigger": True, "trigger": trigger},
        )
        if not result.success:
            logger.warning("NapCat: XiaoXing autonomy trigger send failed: %s", result.error)

    async def _autonomy_trigger_loop(self) -> None:
        while not self._closing:
            try:
                now = dt.datetime.now()
                state = self._ensure_trigger_plan(self._load_trigger_state(), now)
                fired = set(state.get("fired") or [])
                changed = False
                for trigger in state.get("plan") or []:
                    trigger_id = str(trigger.get("id") or "")
                    if not trigger_id or trigger_id in fired:
                        continue
                    try:
                        due_at = dt.datetime.fromisoformat(str(trigger.get("at")))
                    except ValueError:
                        continue
                    if now < due_at:
                        continue
                    if now - due_at > dt.timedelta(hours=2):
                        fired.add(trigger_id)
                        changed = True
                        logger.info("NapCat: skipped stale XiaoXing trigger kind=%s at=%s", trigger.get("kind"), trigger.get("at"))
                        continue
                    await self._fire_autonomy_trigger(trigger)
                    fired.add(trigger_id)
                    changed = True
                if changed:
                    state["fired"] = sorted(fired)
                    self._save_trigger_state(state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("NapCat: XiaoXing trigger loop error: %s", exc)
            await asyncio.sleep(60)

    async def _receive_loop(self) -> None:
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        echo = str(data.get("echo") or "")
                        if echo:
                            future = self._pending_echoes.get(echo)
                            if future and not future.done():
                                future.set_result(data)
                            continue
                        if _looks_like_napcat_account_offline(json.dumps(data, ensure_ascii=False)):
                            await self._mark_account_offline("NapCat reported QQ account offline")
                            continue
                        if "post_type" in data:
                            await self._handle_event(data)
                    except json.JSONDecodeError:
                        logger.warning("NapCat: Invalid JSON received")
                    except Exception as e:
                        logger.error("NapCat: Error handling event: %s", e)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._closing:
                logger.error("NapCat: WebSocket loop error: %s", e)
        finally:
            if self.is_connected and not self._closing:
                logger.warning("NapCat: Connection lost, will trigger reconnect")
                self._set_fatal_error("connection_lost", "NapCat WebSocket closed", retryable=True)

    async def _handle_event(self, data: Dict[str, Any]) -> None:
        if data.get("post_type") != "message":
            return

        message_type = data.get("message_type")
        if message_type == "private":
            user_id = str(data.get("user_id"))
            raw_message = data.get("raw_message", "")
            sender_info = data.get("sender", {})
            user_name = sender_info.get("nickname", user_id)

            # Access Control
            if not self.allow_all_users and self.allowed_users and user_id not in self.allowed_users:
                logger.warning("NapCat: Ignoring message from unauthorized user %s", user_id)
                return

            clean_text, inbound_type, media_urls, media_types = await self._extract_message_content(raw_message, data)
            reply_to_message_id, reply_to_text = await self._resolve_reply_context(data)

            source = self.build_source(
                chat_id=user_id,
                chat_name=user_name,
                chat_type="dm",
                user_id=user_id,
                user_name=user_name,
            )

            self._remember_message_context(data, clean_text)
            event = MessageEvent(
                text=clean_text,
                message_type=inbound_type,
                source=source,
                message_id=str(data.get("message_id", int(time.time() * 1000))),
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=reply_to_message_id,
                reply_to_text=reply_to_text,
                raw_message=data,
            )

            if self._message_handler:
                await self.handle_message(event)
            return

        if message_type == "group":
            if not self._should_process_group_message(data):
                return

            group_id = str(data.get("group_id") or "").strip()
            user_id = str(data.get("user_id") or "").strip()
            raw_message = data.get("raw_message", "")
            sender_info = data.get("sender", {}) or {}
            user_name = sender_info.get("card") or sender_info.get("nickname") or user_id
            group_name = str(data.get("group_name") or f"QQ group {group_id}")
            clean_text, inbound_type, media_urls, media_types = await self._extract_message_content(raw_message, data)
            reply_to_message_id, reply_to_text = await self._resolve_reply_context(data)
            if reply_to_message_id and not reply_to_text:
                reply_to_text = _unresolved_reply_context_text(reply_to_message_id)
            channel_prompts: list[str] = []
            at_prompt = self._at_context_prompt(data)
            if at_prompt:
                channel_prompts.append(at_prompt)
            if self._group_policy == "ambient":
                channel_prompts.append(_AMBIENT_GROUP_PROMPT)
            if self._group_policy == "ambient" and reply_to_message_id:
                channel_prompts.append(_AMBIENT_GROUP_REPLY_PROMPT)
            channel_prompt = "\n".join(channel_prompts) if channel_prompts else None

            source = self.build_source(
                chat_id=f"group:{group_id}",
                chat_name=group_name,
                chat_type="group",
                user_id=user_id,
                user_name=user_name,
            )

            self._remember_message_context(data, clean_text)
            event = MessageEvent(
                text=clean_text,
                message_type=inbound_type,
                source=source,
                message_id=str(data.get("message_id", int(time.time() * 1000))),
                media_urls=media_urls,
                media_types=media_types,
                reply_to_message_id=reply_to_message_id,
                reply_to_text=reply_to_text,
                raw_message=data,
                channel_prompt=channel_prompt,
            )

            if self._message_handler:
                await self.handle_message(event)

# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    return bool(os.getenv("NAPCAT_WS_URL"))

def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    url = os.getenv("NAPCAT_WS_URL") or extra.get("ws_url", "")
    return bool(url)

def interactive_setup() -> None:
    from hermes_cli.setup import (
        prompt,
        save_env_value,
        get_env_value,
        print_header,
        print_info,
        print_warning,
        print_success,
    )

    print_header("NapCat (QQ)")
    existing_url = get_env_value("NAPCAT_WS_URL")

    print_info("Connect Hermes to NapCat QQ WebSocket.")
    url = prompt("NapCat WebSocket URL (e.g. ws://localhost:3005)", default=existing_url or "ws://localhost:3005")
    if not url:
        print_warning("URL is required")
        return
    save_env_value("NAPCAT_WS_URL", url.strip())

    token = prompt("Access Token (if any)", default=get_env_value("NAPCAT_TOKEN") or "")
    if token:
        save_env_value("NAPCAT_TOKEN", token.strip())

    allowed_users = prompt("Allowed User IDs (comma separated)", default=get_env_value("NAPCAT_ALLOWED_USERS") or "")
    if allowed_users:
        save_env_value("NAPCAT_ALLOWED_USERS", allowed_users.strip())

    print_success("NapCat configuration saved to ~/.hermes/.env")
    print_info("Restart the gateway for changes to take effect: hermes gateway restart")

def is_connected(config) -> bool:
    return validate_config(config)

def register(ctx):
    ctx.register_platform(
        name="napcat",
        label="NapCat (QQ)",
        adapter_factory=lambda cfg: NapCatAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["NAPCAT_WS_URL"],
        install_hint="Uses aiohttp to connect to NapCat",
        setup_fn=interactive_setup,
        allowed_users_env="NAPCAT_ALLOWED_USERS",
        standalone_sender_fn=_standalone_send,
        emoji="🐧",
        pii_safe=False,
    )
