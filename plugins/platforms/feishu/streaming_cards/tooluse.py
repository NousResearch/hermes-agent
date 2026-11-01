"""Tool-call tracking and visualization."""

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
    label: str  # one-line action label (e.g. "📖 Reading foo.md"); falls back to title when empty
    emoji: str  # action emoji for quick recognition while collapsed
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
    """Redact key=secret, Authorization headers, and --flag secret patterns."""

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
    """Sanitize detail text according to sanitizer type."""
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
    """Reduce paths in commands to their basename."""
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
    """Build a result/error display block — returns {language, content, fenced} with a markdown fence."""
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


_BLOCK_MAX_CHARS = 1200  # per-tool result block cap: Feishu cards have a JSON
# size limit (200860) and full outputs from tools like execute_code would blow
# it — the panel is a progress view; the terminal holds the full text


def _fenced_block(language: str, content: str) -> ToolBlock:
    if len(content) > _BLOCK_MAX_CHARS:
        head, tail = content[:900], content[-240:]
        omitted = len(content) - 1140
        content = f"{head}\n…({omitted} chars truncated; full output in the terminal)…\n{tail}"
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", content)), default=0) + 1)
    return {"language": language, "content": content, "fenced": f"{fence}{language}\n{content}\n{fence}"}


class ToolUseTracker:
    """Track tool-call steps for the current message.

    Isolated per session; each session owns its lifecycle.
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
        """Close the most recent running step matching by name."""
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
        """Build the step list used for card rendering."""
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
            # Action label: emoji + progressive verb + target (detail already
            # redacted / basename-reduced). Overlong targets (long commands or
            # searches) truncate to 60 chars so titles cannot explode.
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
