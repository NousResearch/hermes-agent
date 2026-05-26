"""Opt-in tool-result compaction for noisy terminal outputs.

This plugin is a clean-room Hermes implementation of a narrow idea surfaced
while reviewing OpenHuman: compress large tool payloads before they re-enter
LLM context. It does not import or copy OpenHuman code.
"""

from __future__ import annotations

import json
from typing import Any, Optional


_DEFAULT_MODE = "observe"
_DEFAULT_THRESHOLD_CHARS = 12_000
_DEFAULT_HEAD_LINES = 40
_DEFAULT_TAIL_LINES = 80
_SUPPORTED_TOOLS = {"terminal"}
_ERROR_MARKERS = (
    "error",
    "failed",
    "failure",
    "traceback",
    "exception",
    "assertionerror",
    "syntaxerror",
    "typeerror",
    "valueerror",
    "modulenotfounderror",
)


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _load_settings() -> dict[str, Any]:
    """Read plugin settings from config.yaml, fail-closed to observe mode."""
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        settings = cfg_get(
            config,
            "plugins",
            "entries",
            "tool-result-compaction",
            default={},
        )
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _interesting_lines(lines: list[str], *, limit: int = 40) -> list[str]:
    selected: list[str] = []
    seen: set[int] = set()
    for idx, line in enumerate(lines):
        lower = line.lower()
        if any(marker in lower for marker in _ERROR_MARKERS):
            start = max(0, idx - 1)
            end = min(len(lines), idx + 2)
            for pos in range(start, end):
                if pos not in seen:
                    seen.add(pos)
                    selected.append(lines[pos])
                    if len(selected) >= limit:
                        return selected
    return selected


def _interesting_span(text: str, *, radius: int = 160) -> str:
    lower = text.lower()
    marker_positions = [
        lower.find(marker)
        for marker in _ERROR_MARKERS
        if lower.find(marker) >= 0
    ]
    if not marker_positions:
        return ""
    pos = min(marker_positions)
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return text[start:end]


def _compact_single_line(text: str) -> tuple[str, dict[str, Any]]:
    head_chars = 500
    tail_chars = 500
    diagnostic = _interesting_span(text)
    omitted = max(0, len(text) - head_chars - tail_chars - len(diagnostic))
    parts = [
        text[:head_chars],
        f"... [tool-result-compaction omitted {omitted} middle char(s)] ...",
    ]
    if diagnostic and diagnostic not in parts[0] and diagnostic not in text[-tail_chars:]:
        parts.extend(["--- preserved diagnostic span ---", diagnostic])
    parts.extend(["--- tail ---", text[-tail_chars:]])
    compacted = "\n".join(parts)
    return compacted, {
        "original_lines": 1,
        "compacted_lines": len(compacted.splitlines()),
        "strategy": "single-line-head-diagnostics-tail",
    }


def _compact_text(text: str, *, head_lines: int, tail_lines: int) -> tuple[str, dict[str, Any]]:
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        if len(lines) <= 1 and len(text) > 1_200:
            return _compact_single_line(text)
        return text, {
            "original_lines": len(lines),
            "compacted_lines": len(lines),
            "strategy": "unchanged-line-count",
        }

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    interesting = _interesting_lines(lines)

    omitted = max(0, len(lines) - len(head) - len(tail))
    parts: list[str] = []
    parts.extend(head)
    parts.append(f"... [tool-result-compaction omitted {omitted} middle line(s)] ...")
    if interesting:
        parts.append("--- preserved diagnostic lines ---")
        parts.extend(interesting)
    parts.append("--- tail ---")
    parts.extend(tail)

    compacted = "\n".join(parts)
    return compacted, {
        "original_lines": len(lines),
        "compacted_lines": len(compacted.splitlines()),
        "strategy": "head-diagnostics-tail",
    }


def compact_tool_result(
    *,
    tool_name: str,
    result: Any,
    threshold_chars: int = _DEFAULT_THRESHOLD_CHARS,
    head_lines: int = _DEFAULT_HEAD_LINES,
    tail_lines: int = _DEFAULT_TAIL_LINES,
    mode: str = _DEFAULT_MODE,
) -> Optional[str]:
    """Return a compacted replacement JSON string, or ``None`` to pass through.

    ``mode='observe'`` deliberately returns ``None`` so enabling the plugin is
    safe before users opt into actual compaction.
    """
    if tool_name not in _SUPPORTED_TOOLS:
        return None
    if mode != "compact":
        return None
    if not isinstance(result, str) or len(result) < threshold_chars:
        return None

    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    output = payload.get("output")
    if not isinstance(output, str) or len(output) < threshold_chars:
        return None

    compacted_output, metadata = _compact_text(
        output,
        head_lines=head_lines,
        tail_lines=tail_lines,
    )
    if len(compacted_output) >= len(output):
        return None

    compacted_payload = dict(payload)
    compacted_payload["output"] = compacted_output
    compacted_payload["tool_result_compaction"] = {
        "mode": "compact",
        "original_chars": len(result),
        "compacted_chars": len(json.dumps(compacted_payload, ensure_ascii=False)),
        **metadata,
    }
    return json.dumps(compacted_payload, ensure_ascii=False)


def _transform_tool_result(**kwargs: Any) -> Optional[str]:
    settings = _load_settings()
    mode = str(settings.get("mode", _DEFAULT_MODE)).strip().lower()
    return compact_tool_result(
        tool_name=str(kwargs.get("tool_name") or ""),
        result=kwargs.get("result"),
        threshold_chars=_as_int(
            settings.get("threshold_chars"),
            _DEFAULT_THRESHOLD_CHARS,
            minimum=1,
        ),
        head_lines=_as_int(settings.get("head_lines"), _DEFAULT_HEAD_LINES, minimum=1),
        tail_lines=_as_int(settings.get("tail_lines"), _DEFAULT_TAIL_LINES, minimum=1),
        mode=mode,
    )


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", _transform_tool_result)
