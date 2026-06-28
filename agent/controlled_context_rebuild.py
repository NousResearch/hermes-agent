"""Deterministic controlled context rebuild sidecar for compression.

This module preserves exact operational anchors (paths, commands, errors, recent
user intent) before lossy LLM summarization rewrites the active transcript.
It deliberately stays pure/stdlib-ish so compression can call it best-effort
without adding provider calls or tool dependencies.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home

CONTROLLED_CONTEXT_REBUILD_HEADER = "[CONTROLLED CONTEXT REBUILD]"
_PACKET_END_MARKER = (
    "[END CONTROLLED CONTEXT REBUILD — continue from live messages below]"
)
_LLM_SUMMARY_HEADING = "## LLM Compaction Summary"

_NOISE_MARKERS = (
    "[CONTEXT COMPACTION",
    "CONTEXT COMPACTION — REFERENCE ONLY",
    "Historical Task Snapshot",
    "Historical In-Progress State",
    "Historical Pending User Asks",
    "Historical Remaining Work",
    "[Your active task list was preserved across context compression]",
    CONTROLLED_CONTEXT_REBUILD_HEADER,
)
_PATH_RE = re.compile(
    r"(?:(?:~|\$HOME|/)[\w.@%+=:,~/-]+|(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+)"
)
_ERROR_RE = re.compile(
    r"\b(?:error|failed|failure|exception|traceback|assertionerror|timeout|fatal|TS\d{3,5}|E\d{3,5})\b",
    re.I,
)
_TOKENISH_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|gh[pousr]_[A-Za-z0-9_]{2,}\.\.\.[A-Za-z0-9_]{2,}|sk-[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9-]{10,}|[A-Za-z0-9_]{24,}\.[A-Za-z0-9_]{6,}\.[A-Za-z0-9_]{20,})\b"
)


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    try:
        return {key: message[key] for key in message.keys()}
    except Exception:
        return {
            "role": getattr(message, "role", ""),
            "content": getattr(message, "content", ""),
        }


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _redact(text: str) -> str:
    text = redact_sensitive_text(str(text or ""), force=True)
    # redact_sensitive_text intentionally targets full tokens. Catch shortened
    # examples like ghp_ab...3456 that still should not fossilize into context.
    return _TOKENISH_RE.sub("[REDACTED]", text)


def _compact(text: Any, *, limit: int = 420) -> str:
    value = _redact(_content_text(text))
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 18)].rstrip() + " ...[truncated]"


def _is_noise(row: dict[str, Any]) -> bool:
    text = _content_text(row.get("content"))
    return any(marker in text for marker in _NOISE_MARKERS)


def _clean_rows(messages: Iterable[Any]) -> list[dict[str, Any]]:
    rows = [_message_dict(message) for message in messages]
    return [
        row
        for row in rows
        if (_content_text(row.get("content")) or row.get("tool_calls"))
        and not _is_noise(row)
    ]


def _dedupe_append(items: list[str], value: str, *, limit: int = 20) -> None:
    value = value.strip()
    if not value or value in items:
        return
    items.append(value)
    if len(items) > limit:
        del items[0]


def _extract_tool_calls(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    for tc in row.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            calls.append((
                str(tc.get("id") or ""),
                str(fn.get("name") or "unknown"),
                str(fn.get("arguments") or ""),
            ))
            continue
        fn = getattr(tc, "function", None)
        calls.append((
            str(getattr(tc, "id", "") or ""),
            str(getattr(fn, "name", "unknown") if fn else "unknown"),
            str(getattr(fn, "arguments", "") if fn else ""),
        ))
    return calls


def _collect_paths(text: str, paths: list[str]) -> None:
    for match in _PATH_RE.findall(text):
        cleaned = match.rstrip(".,);]'\"")
        if len(cleaned) >= 3:
            _dedupe_append(paths, cleaned, limit=24)


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id or "default").strip("._")
    return cleaned or "default"


def _trim_to_budget(text: str, budget: int) -> str:
    if budget <= 0 or len(text) <= budget:
        return text
    marker = "\n...[controlled rebuild truncated]\n"
    keep = max(0, budget - len(marker))
    head = int(keep * 0.70)
    tail = keep - head
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def build_controlled_rebuild_packet(
    session_id: str,
    messages: Iterable[Any],
    *,
    budget: int = 12_000,
) -> str:
    """Build a deterministic packet that survives lossy compression.

    The packet is intentionally factual and bounded. It is not a replacement
    for the LLM summary; it is a guardrail preserving exact literals.
    """
    rows = _clean_rows(messages)
    users = [row for row in rows if row.get("role") == "user"]
    assistants = [row for row in rows if row.get("role") == "assistant"]

    paths: list[str] = []
    commands: list[str] = []
    errors: list[str] = []
    tool_actions: list[str] = []
    call_id_to_tool: dict[str, tuple[str, str]] = {}

    for row in rows:
        text = _compact(row.get("content"), limit=900)
        _collect_paths(text, paths)
        if _ERROR_RE.search(text):
            _dedupe_append(errors, text, limit=10)
        for call_id, name, raw_args in _extract_tool_calls(row):
            args = _redact(raw_args)
            call_id_to_tool[call_id] = (name, args)
            _dedupe_append(
                tool_actions, f"{name}: {_compact(args, limit=320)}", limit=18
            )
            _collect_paths(args, paths)
            try:
                parsed = json.loads(args)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                cmd = parsed.get("command")
                if isinstance(cmd, str):
                    _dedupe_append(commands, _compact(cmd, limit=320), limit=18)
                for key in ("path", "workdir", "file_path", "output_path"):
                    val = parsed.get(key)
                    if isinstance(val, str):
                        _dedupe_append(paths, val, limit=24)

    for row in rows:
        if row.get("role") != "tool":
            continue
        text = _compact(row.get("content"), limit=600)
        tool_name, tool_args = call_id_to_tool.get(
            str(row.get("tool_call_id") or ""), ("unknown", "")
        )
        if text:
            _dedupe_append(tool_actions, f"{tool_name} result: {text}", limit=18)
            _collect_paths(text, paths)
        if _ERROR_RE.search(text):
            _dedupe_append(errors, text, limit=10)
        _collect_paths(tool_args, paths)

    recent_user = (
        _compact(users[-1].get("content") if users else "", limit=650)
        or "None detected."
    )
    recent_assistant = (
        _compact(assistants[-1].get("content") if assistants else "", limit=650)
        or "None detected."
    )
    tail = [
        f"- {row.get('role', 'unknown')}: {_compact(row.get('content'), limit=260)}"
        for row in rows[-8:]
        if _compact(row.get("content"), limit=260)
    ]

    def bullets(items: list[str], *, empty: str = "None.", limit: int = 12) -> str:
        return "\n".join(f"- {item}" for item in items[:limit]) if items else empty

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    packet = f"""{CONTROLLED_CONTEXT_REBUILD_HEADER}
Generated: {generated}
Session: {_safe_session_id(session_id)}

## Purpose
Deterministic rebuild packet produced before lossy compression. Treat it as exact-literal continuity: paths, commands, errors, current intent, and recent state. Do not treat historical/compression task-list noise as fresh user input.

## Current User Intent
{recent_user}

## Latest Assistant State
{recent_assistant}

## Exact Paths / Files / URLs
{bullets(paths, limit=18)}

## Commands / Operations
{bullets(commands, limit=12)}

## Tool Evidence
{bullets(tool_actions, limit=14)}

## Errors / Blockers
{bullets(errors, limit=8)}

## Recent Live Tail
{chr(10).join(tail) if tail else "None."}

{_PACKET_END_MARKER}"""
    return _trim_to_budget(_redact(packet), budget)


def checkpoint_path_for_session(
    session_id: str, *, hermes_home: str | Path | None = None
) -> Path:
    home = Path(hermes_home) if hermes_home is not None else get_hermes_home()
    return (
        home / "context" / "sessions" / _safe_session_id(session_id) / "checkpoint.md"
    )


def write_controlled_rebuild_checkpoint(
    session_id: str,
    messages: Iterable[Any],
    *,
    hermes_home: str | Path | None = None,
    budget: int = 16_000,
) -> Path:
    path = checkpoint_path_for_session(session_id, hermes_home=hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = build_controlled_rebuild_packet(session_id, messages, budget=budget)
    path.write_text(content, encoding="utf-8")
    return path


def append_controlled_rebuild_to_summary(
    summary: str,
    session_id: str,
    messages: Iterable[Any],
    *,
    budget: int = 12_000,
) -> str:
    """Prefix an LLM compaction summary with the deterministic rebuild packet."""
    summary = str(summary or "").strip()
    if CONTROLLED_CONTEXT_REBUILD_HEADER in summary:
        return summary
    packet = build_controlled_rebuild_packet(
        session_id, messages, budget=budget
    ).strip()
    if not summary:
        return packet
    return f"{packet}\n\n{_LLM_SUMMARY_HEADING}\n{summary}"
