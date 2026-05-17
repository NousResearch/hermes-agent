"""Numbered mobile reply bridge for gateway handoffs.

This module implements the small, durable contract used when a desktop or
cloud session asks the user for a decision through Weixin/other mobile
channels:

    回复 HERMES-ASK-YYYYMMDD-HHMMSS-XXXXXX content

Replies are recorded under ``$HERMES_HOME/mobile_reply_bridge`` so the
originating session can poll for them without guessing which chat thread the
answer belongs to.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home


BRIDGE_DIR_NAME = "mobile_reply_bridge"
REQUEST_ID_PREFIX = "HERMES-ASK"
MAX_TITLE_CHARS = 200
MAX_QUESTION_CHARS = 8000
MAX_REPLY_CHARS = 20000

REQUEST_ID_RE = re.compile(
    r"^HERMES-ASK-\d{8}-\d{6}-[A-F0-9]{6,12}$",
    re.IGNORECASE,
)
_REPLY_WITH_ID_PATTERNS = (
    re.compile(
        r"^\s*(?:回复|回覆)\s*(?:[:：#-]|\s+)\s*"
        r"(?P<request_id>HERMES-ASK-\d{8}-\d{6}-[A-F0-9]{6,12})"
        r"(?:\s+|[:：-]\s*)(?P<content>[\s\S]*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:reply|re)\s+(?P<request_id>HERMES-ASK-\d{8}-\d{6}-[A-F0-9]{6,12})"
        r"(?:\s+|[:：-]\s*)(?P<content>[\s\S]*)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<request_id>HERMES-ASK-\d{8}-\d{6}-[A-F0-9]{6,12})"
        r"(?:\s+|[:：-]\s*)(?P<content>[\s\S]*)$",
        re.IGNORECASE,
    ),
)
_REPLY_MISSING_ID_RE = re.compile(
    r"^\s*(?:(?:回复|回覆)\s*(?:[:：#-]|\s+)|(?:reply|re)\s+)(?P<content>[\s\S]*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedMobileReply:
    request_id: str
    content: str


@dataclass(frozen=True)
class MobileBridgeResult:
    action: str
    message: str
    request_id: Optional[str] = None
    reply_path: Optional[Path] = None


def bridge_root(hermes_home: str | Path | None = None) -> Path:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return home / BRIDGE_DIR_NAME


def bridge_paths(hermes_home: str | Path | None = None) -> dict[str, Path]:
    root = bridge_root(hermes_home)
    return {
        "root": root,
        "requests": root / "requests",
        "replies": root / "replies",
        "events": root / "events",
    }


def ensure_bridge_dirs(hermes_home: str | Path | None = None) -> dict[str, Path]:
    paths = bridge_paths(hermes_home)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = str(value or "").replace("\x00", "").strip()
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _safe_json_value(value: Any, max_chars: int = 1000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _safe_text(value, max_chars)[0]
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item, max_chars=max_chars) for item in value[:50]]
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 50:
                break
            safe[str(key)[:100]] = _safe_json_value(item, max_chars=max_chars)
        return safe
    return _safe_text(repr(value), max_chars)[0]


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def normalize_request_id(request_id: str) -> str:
    normalized = str(request_id or "").strip().upper()
    if not REQUEST_ID_RE.fullmatch(normalized):
        raise ValueError("invalid mobile reply request id")
    return normalized


def generate_request_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{REQUEST_ID_PREFIX}-{stamp}-{secrets.token_hex(3).upper()}"


def parse_mobile_reply(text: str) -> Optional[ParsedMobileReply]:
    raw = str(text or "").strip()
    if not raw:
        return None
    for pattern in _REPLY_WITH_ID_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        request_id = normalize_request_id(match.group("request_id"))
        content = match.group("content") or ""
        content, _ = _safe_text(content, MAX_REPLY_CHARS)
        return ParsedMobileReply(request_id=request_id, content=content)
    return None


def looks_like_mobile_reply_missing_id(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if parse_mobile_reply(raw) is not None:
        return False
    return bool(_REPLY_MISSING_ID_RE.match(raw))


def create_mobile_ask(
    *,
    title: str,
    question: str,
    source_kind: str = "codex",
    source_session: str | None = None,
    metadata: dict[str, Any] | None = None,
    hermes_home: str | Path | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    paths = ensure_bridge_dirs(hermes_home)
    normalized_id = normalize_request_id(request_id) if request_id else generate_request_id()
    title_text, title_truncated = _safe_text(title, MAX_TITLE_CHARS)
    question_text, question_truncated = _safe_text(question, MAX_QUESTION_CHARS)
    record = {
        "type": "mobile_ask",
        "request_id": normalized_id,
        "status": "pending",
        "title": title_text,
        "title_truncated": title_truncated,
        "question": question_text,
        "question_truncated": question_truncated,
        "source_kind": _safe_text(source_kind, 80)[0],
        "source_session": _safe_text(source_session or "", 500)[0] or None,
        "metadata": _safe_json_value(metadata or {}),
        "created_at": _now_iso(),
        "reply_instruction": f"回复 {normalized_id} 你的指令",
    }
    _atomic_json_write(paths["requests"] / f"{normalized_id}.json", record)
    return record


def format_mobile_ask_card(record: dict[str, Any]) -> str:
    request_id = normalize_request_id(str(record.get("request_id", "")))
    title = str(record.get("title") or "Hermes 卡点请求").strip()
    question = str(record.get("question") or "").strip()
    source_kind = str(record.get("source_kind") or "").strip()
    source_session = str(record.get("source_session") or "").strip()

    lines = [
        "【Hermes 卡点请求】",
        f"编号：{request_id}",
    ]
    if title:
        lines.append(f"主题：{title}")
    if source_kind:
        source_line = f"来源：{source_kind}"
        if source_session:
            source_line += f" / {source_session}"
        lines.append(source_line)
    if question:
        lines.extend(["", question])
    lines.extend(
        [
            "",
            f"回复格式：回复 {request_id} 你的指令",
            "请保留编号，避免回错会话。",
        ]
    )
    return "\n".join(lines)


def record_mobile_reply(
    parsed: ParsedMobileReply,
    *,
    platform: str | None = None,
    chat_id: str | None = None,
    user_id: str | None = None,
    user_name: str | None = None,
    message_id: str | None = None,
    hermes_home: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    request_id = normalize_request_id(parsed.request_id)
    content, truncated = _safe_text(parsed.content, MAX_REPLY_CHARS)
    paths = ensure_bridge_dirs(hermes_home)
    request_path = paths["requests"] / f"{request_id}.json"
    known_request = request_path.exists()
    received_at = _now_iso()
    reply = {
        "type": "mobile_reply",
        "request_id": request_id,
        "content": content,
        "content_truncated": truncated,
        "received_at": received_at,
        "known_request": known_request,
        "source": {
            "platform": _safe_text(platform or "", 80)[0] or None,
            "chat_id": _safe_text(chat_id or "", 200)[0] or None,
            "user_id": _safe_text(user_id or "", 200)[0] or None,
            "user_name": _safe_text(user_name or "", 200)[0] or None,
            "message_id": _safe_text(message_id or "", 200)[0] or None,
        },
    }
    stamp = str(int(time.time() * 1000))
    reply_dir = paths["replies"] / request_id
    reply_path = reply_dir / f"{stamp}-{secrets.token_hex(3)}.json"
    _atomic_json_write(reply_path, reply)
    _atomic_json_write(reply_dir / "latest.json", reply | {"reply_path": str(reply_path)})

    request_record = _load_json(request_path)
    if request_record is not None:
        request_record["status"] = "answered"
        request_record["answered_at"] = received_at
        request_record["last_reply_path"] = str(reply_path)
        _atomic_json_write(request_path, request_record)

    _atomic_json_write(paths["events"] / f"{stamp}-{request_id}.json", reply | {"reply_path": str(reply_path)})
    return reply, reply_path


def handle_mobile_reply_text(
    text: str,
    *,
    platform: str | None = None,
    chat_id: str | None = None,
    user_id: str | None = None,
    user_name: str | None = None,
    message_id: str | None = None,
    hermes_home: str | Path | None = None,
) -> Optional[MobileBridgeResult]:
    parsed = parse_mobile_reply(text)
    if parsed is not None:
        if not parsed.content:
            return MobileBridgeResult(
                action="missing_content",
                request_id=parsed.request_id,
                message=f"请在编号 {parsed.request_id} 后写回复内容。",
            )
        reply, reply_path = record_mobile_reply(
            parsed,
            platform=platform,
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
            message_id=message_id,
            hermes_home=hermes_home,
        )
        if reply.get("known_request"):
            msg = f"已收到请求 {parsed.request_id} 的回复，原会话可以继续读取。"
        else:
            msg = f"已收到 {parsed.request_id} 的回复，但本地未找到对应请求卡，已按孤立回复保存。"
        return MobileBridgeResult(
            action="recorded",
            request_id=parsed.request_id,
            reply_path=reply_path,
            message=msg,
        )

    if looks_like_mobile_reply_missing_id(text):
        return MobileBridgeResult(
            action="missing_id",
            message=(
                "请带上请求编号，避免回错会话。格式："
                "回复 HERMES-ASK-YYYYMMDD-HHMMSS-XXXXXX 你的指令"
            ),
        )
    return None


def load_latest_reply(
    request_id: str,
    *,
    hermes_home: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    normalized_id = normalize_request_id(request_id)
    return _load_json(bridge_root(hermes_home) / "replies" / normalized_id / "latest.json")


def wait_for_mobile_reply(
    request_id: str,
    *,
    timeout_seconds: float = 0,
    poll_seconds: float = 2,
    hermes_home: str | Path | None = None,
) -> Optional[dict[str, Any]]:
    normalized_id = normalize_request_id(request_id)
    deadline = time.monotonic() + max(float(timeout_seconds), 0)
    while True:
        reply = load_latest_reply(normalized_id, hermes_home=hermes_home)
        if reply is not None:
            return reply
        if timeout_seconds <= 0 or time.monotonic() >= deadline:
            return None
        time.sleep(max(float(poll_seconds), 0.2))
