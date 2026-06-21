"""Zero-inference context dump helpers for gateway sessions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def context_dump_slug(session_key: str | None) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_key or "unknown")).strip("._")
    return slug[:180] or "unknown"


def context_dump_path(dump_dir: Path, session_key: str | None) -> Path:
    return dump_dir / f"{context_dump_slug(session_key)}.json"


def context_dump_text_path(dump_dir: Path, session_key: str | None) -> Path:
    return dump_dir / f"{context_dump_slug(session_key)}.message.txt"


def write_context_dump_payload(
    dump_dir: Path, session_key: str | None, payload: Mapping[str, Any]
) -> Path:
    dump_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = context_dump_path(dump_dir, session_key)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        fh.write("\n")
    tmp_path.replace(path)
    _chmod_private(path)
    return path


def write_context_dump_text(
    dump_dir: Path, session_key: str | None, payload: Mapping[str, Any]
) -> Path:
    dump_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = context_dump_text_path(dump_dir, session_key)
    tmp_path = path.with_suffix(".txt.tmp")
    lines = render_context_dump_text(payload)
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    tmp_path.replace(path)
    _chmod_private(path)
    return path


def render_context_dump_text(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# Hermes Raw Context Dump",
        "",
        f"Captured: {payload.get('captured_at', '')}",
        f"Mode: {payload.get('capture_mode', payload.get('phase', ''))}",
        f"Session key: {payload.get('session_key', '')}",
        f"Session id: {payload.get('session_id', '')}",
        f"Model: {payload.get('resolved_model') or payload.get('model') or ''}",
        f"Provider: {payload.get('provider') or ''}",
        f"Estimated total tokens: {payload.get('estimated_total_tokens', '')}",
        "",
        "## Context Layers",
        "",
        json.dumps(
            payload.get("context_layers", []), ensure_ascii=False, indent=2, default=str
        ),
        "",
        "## Raw API Messages",
        "",
        json.dumps(
            payload.get("api_messages", []), ensure_ascii=False, indent=2, default=str
        ),
        "",
        "## Tool Schemas",
        "",
        json.dumps(payload.get("tools", []), ensure_ascii=False, indent=2, default=str),
        "",
        "## Debug Metadata",
        "",
        json.dumps(
            {
                k: v
                for k, v in payload.items()
                if k not in {"api_messages", "tools", "context_layers"}
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "",
    ]


def agent_history_for_context_dump(
    history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    agent_history: list[dict[str, Any]] = []
    for msg in history:
        role = msg.get("role")
        if not role or role == "session_meta" or role == "system":
            continue
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            agent_history.append({k: v for k, v in msg.items() if k != "timestamp"})
            continue
        content = msg.get("content")
        if content:
            if msg.get("mirror"):
                mirror_src = msg.get("mirror_source", "another session")
                content = f"[Delivered from {mirror_src}] {content}"
            entry: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                for key in ("reasoning", "reasoning_details", "codex_reasoning_items"):
                    value = msg.get(key)
                    if value:
                        entry[key] = value
            agent_history.append(entry)
    return agent_history


def estimate_context_dump_tokens(payload: Mapping[str, Any]) -> int:
    try:
        from agent.model_metadata import estimate_request_tokens_rough

        return int(
            estimate_request_tokens_rough(
                payload.get("api_messages", []) or [],
                tools=payload.get("tools", []) or None,
            )
        )
    except Exception:
        messages = payload.get("api_messages", []) or []
        tools = payload.get("tools", []) or []
        return (len(str(messages)) + len(str(tools)) + 3) // 4


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass
