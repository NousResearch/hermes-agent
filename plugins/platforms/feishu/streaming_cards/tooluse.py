"""工具调用追踪与可视化."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypedDict


class ToolStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class ToolBlock(TypedDict):
    language: str
    content: str
    fenced: str


class ToolDisplayStep(TypedDict):
    name: str
    title: str
    status: str
    detail: str
    label: str  # 单行动作标签（"📖 Reading 幼儿园与学习.md"），空则退 title
    emoji: str  # 工具动作 emoji（折叠态快速识别）
    output: str
    error: str
    icon: str
    elapsed_ms: float
    result_block: ToolBlock | None
    error_block: ToolBlock | None


@dataclass
class ToolStep:
    name: str
    status: ToolStatus
    detail: str = ""
    output: str = ""
    error: str = ""
    result_block: ToolBlock | None = None
    error_block: ToolBlock | None = None
    started_at: float = 0.0
    elapsed_ms: float = 0.0


@dataclass
class ToolSession:
    steps: list[ToolStep] = field(default_factory=list)


_SENSITIVE_NAME_RE = re.compile(
    r"token|secret|password|api[_-]?key|authorization|cookie|credential"
    r"|bearer|session[_-]?id|client[_-]?secret|access[_-]?key",
    re.IGNORECASE,
)

_INLINE_ASSIGNMENT_RE = re.compile(r'(^|[\s"\'`])([A-Za-z_][A-Za-z0-9_]*)(=(?:"[^"]*"|\'[^\']*\'|[^\s"\'`]+))')
_AUTH_HEADER_RE = re.compile(
    r"(Authorization\s*:\s*(?:Bearer|Basic|Token)\s+)([^\'\"\s]+)",
    re.IGNORECASE,
)
_SECRET_FLAG_RE = re.compile(
    r'((?:^|[\s"\'`])(--?[A-Za-z0-9][A-Za-z0-9-]*)(=|\s+)("(?:[^"]*)"|\'(?:[^\']*)\'|[^\s"\'`]+))'
)


def redact_inline_secrets(value: str) -> str:
    """脱敏 key=secret、Authorization header、--flag secret 模式."""

    def _redact_assign(m: re.Match) -> str:
        key = str(m.group(2))
        if _SENSITIVE_NAME_RE.search(key):
            return f"{m.group(1)}{key}=[redacted]"
        return str(m.group(0))

    def _redact_flag(m: re.Match) -> str:
        flag = re.sub(r"^-+", "", str(m.group(2)))
        if _SENSITIVE_NAME_RE.search(flag):
            return f"{m.group(1)}{m.group(2)}{m.group(3)}[redacted]"
        return str(m.group(0))

    return _SECRET_FLAG_RE.sub(
        _redact_flag,
        _AUTH_HEADER_RE.sub(r"\1[redacted]", _INLINE_ASSIGNMENT_RE.sub(_redact_assign, value)),
    )


def _sanitize_detail(text: str, sanitizer: str | None) -> str:
    """根据 sanitizer 类型清洗 detail 文本."""
    if not text or not sanitizer:
        return text
    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    if not cleaned:
        return text
    if sanitizer == "command":
        cleaned = redact_inline_secrets(cleaned)
        return _redact_paths(cleaned)
    if sanitizer == "path":
        return _basename_only(re.sub(r"^(?:from|file|path)\s+", "", cleaned, flags=re.IGNORECASE).strip())
    if sanitizer == "search":
        return cleaned.strip("'\"")
    if sanitizer == "url":
        if cleaned.lower().startswith("from "):
            return cleaned.strip("'\"").replace("from ", "", 1)
        return cleaned.strip("'\"")
    return cleaned


def _redact_paths(text: str) -> str:
    """命令中路径只保留 basename."""
    return re.sub(
        r'(^|[\s=\'"()])([~./][^\s\'"()]+)',
        lambda m: f"{m.group(1)}{os.path.basename(m.group(2))}",
        text,
    )


def _basename_only(text: str) -> str:
    if not text:
        return text
    return os.path.basename(text.replace("\\", "/").rstrip("/"))


_TOOL_DESCRIPTORS: list[dict[str, Any]] = [
    {
        "aliases": ["skill"],
        "icon": "app-default_outlined",
        "title": "Load skill",
        "emoji": "📚",
        "verb_ing": "Loading skill",
        "sanitizer": None,
    },
    {
        "aliases": ["read", "open"],
        "icon": "file-link-text_outlined",
        "title": "Read",
        "emoji": "📖",
        "verb_ing": "Reading",
        "sanitizer": "path",
        "no_result": True,
    },
    {
        "aliases": ["write", "edit", "patch"],
        "icon": "edit_outlined",
        "title": "Edit",
        "emoji": "🔧",
        "verb_ing": "Editing",
        "sanitizer": "path",
        "no_result": True,
    },
    {
        "aliases": ["web_search", "web-search", "search"],
        "icon": "search_outlined",
        "title": "Search",
        "emoji": "🔍",
        "verb_ing": "Searching",
        "sanitizer": "search",
    },
    {
        "aliases": ["web_fetch", "web-fetch", "fetch", "extract"],
        "icon": "language_outlined",
        "title": "Fetch web page",
        "emoji": "🌐",
        "verb_ing": "Fetching",
        "sanitizer": "url",
        "no_result": True,
    },
    {
        "aliases": ["grep"],
        "icon": "doc-search_outlined",
        "title": "Search text",
        "emoji": "🔎",
        "verb_ing": "Searching text",
        "sanitizer": "search",
    },
    {
        "aliases": ["glob", "find"],
        "icon": "folder_outlined",
        "title": "Search files",
        "emoji": "🗂️",
        "verb_ing": "Searching files",
        "sanitizer": "path",
    },
    {
        "aliases": ["exec", "bash", "command", "run", "terminal"],
        "icon": "setting_outlined",
        "title": "Run command",
        "emoji": "⚡",
        "verb_ing": "Running",
        "sanitizer": "command",
    },
    {
        "aliases": ["browser", "playwright", "navigate", "computer_use"],
        "icon": "browser-mac_outlined",
        "title": "Browser",
        "emoji": "🧭",
        "verb_ing": "Browsing",
        "no_result": True,
    },
    {
        "aliases": ["agent", "task", "spawn", "delegate"],
        "icon": "robot_outlined",
        "title": "Run sub-agent",
        "emoji": "🤖",
        "verb_ing": "Delegating",
    },
    {
        "aliases": ["check", "determine", "verify", "inspect"],
        "icon": "list-check_outlined",
        "title": "Check",
        "emoji": "✅",
        "verb_ing": "Checking",
    },
    {
        "aliases": ["summarize", "analyze", "prepare", "reason"],
        "icon": "report_outlined",
        "title": "Analyze",
        "emoji": "🧠",
        "verb_ing": "Analyzing",
    },
    {
        "aliases": ["clarify"],
        "icon": "chat_outlined",
        "title": "Clarify",
        "emoji": "❓",
        "verb_ing": "Asking",
        "no_result": True,
    },
    {
        "aliases": ["vision", "look", "image_analyze"],
        "icon": "eyes_outlined",
        "title": "Looking at image",
        "emoji": "👁️",
        "verb_ing": "Looking at image",
    },
    {
        "aliases": ["image_generate", "generate"],
        "icon": "image_outlined",
        "title": "Generating image",
        "emoji": "🎨",
        "verb_ing": "Generating image",
    },
    {
        "aliases": ["memory"],
        "icon": "brain_outlined",
        "title": "Updating memory",
        "emoji": "💾",
        "verb_ing": "Updating memory",
    },
]


def _resolve_tool_descriptor(name: str | None) -> dict[str, Any] | None:
    if not name:
        return None
    normalized = name.strip().lower().replace("-", "_")
    for desc in _TOOL_DESCRIPTORS:
        for alias in desc["aliases"]:
            if normalized == alias or normalized.startswith(f"{alias}_"):
                return desc
    return None


def _humanize_tool_name(name: str) -> str:
    cleaned = name.replace("-", " ").replace("_", " ").strip()
    if not cleaned:
        return "Tool"
    return cleaned[0].upper() + cleaned[1:]


def _format_duration_label(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{(ms / 1000):.1f} s"


def _build_display_block(
    value: Any,
    fallback_lang: str = "json",
    *,
    sanitizer: str | None = None,
) -> ToolBlock | None:
    """构建结果/错误的显示块 — 返回 {language, content, fenced} 含 markdown 代码围栏."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n").strip()
        if not normalized:
            return None
        if sanitizer == "command":
            normalized = redact_inline_secrets(normalized)
        if normalized.startswith("{") or normalized.startswith("["):
            try:
                parsed = json.loads(normalized)
                pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                return _fenced_block("json", pretty)
            except json.JSONDecodeError:
                pass
        return _fenced_block("text" if fallback_lang == "json" else fallback_lang, normalized)
    if isinstance(value, (dict, list)):
        try:
            return _fenced_block("json", json.dumps(value, ensure_ascii=False, indent=2))
        except (TypeError, ValueError):
            pass
    normalized = str(value).strip()
    return _fenced_block("text", normalized) if normalized else None


_BLOCK_MAX_CHARS = 1200  # 单工具结果块上限：飞书卡片有 JSON 体积上限（200860），
# execute_code 等工具的完整输出会把卡撑爆——工具面板只是进度展示，全文在终端里


def _fenced_block(language: str, content: str) -> ToolBlock:
    if len(content) > _BLOCK_MAX_CHARS:
        head, tail = content[:900], content[-240:]
        omitted = len(content) - 1140
        content = f"{head}\n…（已截断 {omitted} 字符，完整输出见终端）…\n{tail}"
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", content)), default=0) + 1)
    return {"language": language, "content": content, "fenced": f"{fence}{language}\n{content}\n{fence}"}


class ToolUseTracker:
    """追踪当前消息中的工具调用步骤.

    按 session 隔离，每个会话独立生命周期.
    """

    def __init__(self, max_steps: int = 128) -> None:
        self._session: ToolSession | None = None
        self._max_steps = max_steps

    def record_start(self, name: str, detail: str = "") -> None:
        if self._session is None:
            self._session = ToolSession()
        if len(self._session.steps) >= self._max_steps:
            return
        self._session.steps.append(
            ToolStep(
                name=name,
                status=ToolStatus.RUNNING,
                detail=detail,
                started_at=time.time(),
            )
        )

    def record_end(self, name: str, *, error: str = "", output: str = "") -> None:
        """通过名字匹配最近的一个 running 步骤来结束."""
        if self._session is None:
            return
        desc = _resolve_tool_descriptor(name)
        sanitizer = desc.get("sanitizer") if desc else None
        for step in reversed(self._session.steps):
            if step.name == name and step.status == ToolStatus.RUNNING:
                step.status = ToolStatus.ERROR if error else ToolStatus.SUCCESS
                step.error = error
                step.output = output
                step.elapsed_ms = (time.time() - step.started_at) * 1000
                if error:
                    step.error_block = _build_display_block(error, "text", sanitizer=sanitizer)
                elif output:
                    step.result_block = _build_display_block(output, "json", sanitizer=sanitizer)
                return
        self._session.steps.append(
            ToolStep(
                name=name,
                status=ToolStatus.ERROR if error else ToolStatus.SUCCESS,
                detail=error or output,
                output=output,
                error=error,
                started_at=time.time(),
                error_block=_build_display_block(error, "text", sanitizer=sanitizer) if error else None,
                result_block=_build_display_block(output, "json", sanitizer=sanitizer) if output else None,
            )
        )

    def build_display_steps(self) -> list[ToolDisplayStep]:
        """构建用于卡片渲染的步骤列表."""
        if self._session is None:
            return []
        steps: list[ToolDisplayStep] = []
        for s in self._session.steps:
            desc = _resolve_tool_descriptor(s.name)
            base_title = desc["title"] if desc else _humanize_tool_name(s.name)
            if s.elapsed_ms > 0:
                base_title = f"{base_title} ({_format_duration_label(s.elapsed_ms)})"
            sanitizer = desc.get("sanitizer") if desc else None
            detail = _sanitize_detail(s.detail, sanitizer)
            emoji = (desc.get("emoji") if desc else None) or "⚙️"
            verb_ing = (desc.get("verb_ing") if desc else None) or base_title
            # 动作标签：emoji + 进行时动词 + 目标（detail 已脱敏/取 basename）。
            # 目标过长（如长 command/search）截断到 60 字符防标题爆炸。
            label = f"{emoji} {verb_ing}"
            if detail:
                label += f" {detail[:60]}"
            steps.append(
                {
                    "name": s.name,
                    "title": base_title,
                    "status": s.status.value,
                    "detail": detail[:200],
                    "label": label,
                    "emoji": emoji,
                    "output": s.output,
                    "error": s.error,
                    "icon": desc["icon"] if desc else "setting-inter_outlined",
                    "elapsed_ms": s.elapsed_ms,
                    "result_block": None if (desc and desc.get("no_result")) else s.result_block,
                    "error_block": s.error_block,
                }
            )
        return steps
