"""Presentation helpers for adaptive gateway tool-progress traces.

This module is intentionally presentation-only: traces are never written back to
agent history and never affect model context.  The gateway uses them to render a
compact progress summary while keeping redacted per-tool details available to
platform UIs that support an interactive drill-down (Telegram inline buttons).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


_STATUS_ORDER = {"running": 0, "completed": 1, "failed": 2}


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _value_preview(value: Any, *, max_chars: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clip(value, max_chars)
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except Exception:
        rendered = str(value)
    return _clip(rendered, max_chars)


def _status_symbol(status: str, ok: Optional[bool] = None) -> str:
    if status == "running":
        return "…"
    if status == "failed" or ok is False:
        return "✗"
    return "✓"


def _duration_text(duration: Optional[float]) -> str:
    if duration is None:
        return ""
    try:
        value = float(duration)
    except (TypeError, ValueError):
        return ""
    if value < 1:
        return f"{value * 1000:.0f}ms"
    if value < 60:
        return f"{value:.1f}s"
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}m{seconds:02d}s"


def _tool_emoji(tool_name: str) -> str:
    try:
        from agent.display import get_tool_emoji

        return get_tool_emoji(tool_name, default="⚙️")
    except Exception:
        return "⚙️"


@dataclass
class ToolTraceCall:
    """One presentation-layer tool invocation."""

    index: int
    name: str
    preview: str = ""
    args_preview: str = ""
    status: str = "running"
    ok: Optional[bool] = None
    duration: Optional[float] = None
    result_preview: str = ""
    call_id: str = ""
    started_at: float = field(default_factory=time.monotonic)
    completed_at: Optional[float] = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "preview": self.preview,
            "args_preview": self.args_preview,
            "status": self.status,
            "ok": self.ok,
            "duration": self.duration,
            "duration_text": _duration_text(self.duration),
            "result_preview": self.result_preview,
            "call_id": self.call_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ToolProgressTrace:
    """Collect redacted tool-progress events for one gateway turn."""

    def __init__(
        self,
        trace_id: str,
        *,
        max_args_chars: int = 1200,
        max_result_chars: int = 1600,
        redact: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.trace_id = str(trace_id)
        self.max_args_chars = max_args_chars
        self.max_result_chars = max_result_chars
        self._redact = redact or (lambda text: text)
        self._calls: list[ToolTraceCall] = []
        self._next_index = 1
        self.created_at = time.monotonic()
        self.updated_at = self.created_at

    def started(
        self,
        name: str,
        preview: Any = None,
        args: Any = None,
        *,
        call_id: Any = None,
    ) -> ToolTraceCall:
        call = ToolTraceCall(
            index=self._next_index,
            name=str(name or "tool"),
            preview=_clip(str(preview or ""), 240),
            args_preview=self._redacted_preview(args, self.max_args_chars),
            call_id=str(call_id or ""),
        )
        self._next_index += 1
        self._calls.append(call)
        self.updated_at = time.monotonic()
        return call

    def completed(
        self,
        name: str,
        *,
        duration: Any = None,
        is_error: Any = False,
        result: Any = None,
        call_id: Any = None,
    ) -> ToolTraceCall:
        call = self._find_open_call(str(name or "tool"), call_id=call_id)
        if call is None:
            call = self.started(str(name or "tool"), call_id=call_id)
        failed = bool(is_error)
        call.status = "failed" if failed else "completed"
        call.ok = not failed
        try:
            call.duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            call.duration = None
        call.result_preview = self._redacted_preview(result, self.max_result_chars)
        call.completed_at = time.monotonic()
        self.updated_at = call.completed_at
        return call

    def snapshot(self) -> dict[str, Any]:
        calls = [call.to_payload() for call in self._calls]
        counts = Counter(call["name"] for call in calls)
        running = sum(1 for call in calls if call.get("status") == "running")
        failed = sum(1 for call in calls if call.get("status") == "failed")
        completed = sum(1 for call in calls if call.get("status") == "completed")
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "calls": calls,
            "counts": dict(counts),
            "total": len(calls),
            "running": running,
            "completed": completed,
            "failed": failed,
        }

    def _find_open_call(self, name: str, *, call_id: Any = None) -> Optional[ToolTraceCall]:
        call_id_text = str(call_id or "")
        if call_id_text:
            for call in self._calls:
                if call.call_id == call_id_text and call.status == "running":
                    return call
            return None
        for call in self._calls:
            if call.name == name and call.status == "running":
                return call
        return None

    def _redacted_preview(self, value: Any, max_chars: int) -> str:
        rendered = _value_preview(value, max_chars=max_chars)
        if not rendered:
            return ""
        try:
            rendered = self._redact(rendered)
        except Exception:
            pass
        return _clip(rendered, max_chars)


def format_tool_progress_summary(
    snapshot: dict[str, Any],
    *,
    inline_limit: int = 2,
    max_chars: int = 3900,
) -> str:
    """Return the default compact/adaptive progress message."""

    calls = list(snapshot.get("calls") or [])
    total = len(calls)
    if total == 0:
        return "🛠️ Tools: starting…"

    completed = int(snapshot.get("completed") or 0)
    failed = int(snapshot.get("failed") or 0)
    running = int(snapshot.get("running") or 0)
    status_bits: list[str] = []
    if completed:
        status_bits.append(f"✓ {completed}")
    if running:
        status_bits.append(f"… {running}")
    if failed:
        status_bits.append(f"✗ {failed}")
    status = f" ({', '.join(status_bits)})" if status_bits else ""

    header = f"🛠️ Tools: {total} call{'s' if total != 1 else ''}{status}"
    inline_limit = max(0, int(inline_limit or 0))

    if total <= inline_limit:
        lines = [header]
        for call in calls:
            lines.extend(_summary_call_lines(call, include_args=True))
        return _clip("\n".join(lines), max_chars)

    counts = snapshot.get("counts") or {}
    count_line = ", ".join(
        f"{name}×{count}" if count > 1 else str(name)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    )
    sequence = " → ".join(str(call.get("name") or "tool") for call in calls[:10])
    if total > 10:
        sequence += " → …"

    lines = [header]
    if count_line:
        lines.append(f"Counts: {count_line}")
    lines.append(f"Sequence: {sequence}")
    lines.append("Tap Details for arguments and output previews.")
    return _clip("\n".join(lines), max_chars)


def format_tool_progress_detail(
    snapshot: dict[str, Any],
    *,
    max_chars: int = 3900,
) -> str:
    """Return an expanded all-calls detail view."""

    calls = list(snapshot.get("calls") or [])
    if not calls:
        return "🛠️ No tool calls yet."
    lines = [f"🛠️ Tool details: {len(calls)} call{'s' if len(calls) != 1 else ''}"]
    for call in calls:
        lines.extend(_summary_call_lines(call, include_args=True, include_output=True))
    return _clip("\n".join(lines), max_chars)


def format_tool_progress_call(
    snapshot: dict[str, Any],
    index: int,
    *,
    section: str = "call",
    max_chars: int = 3900,
) -> str:
    """Return a single-call drill-down view."""

    call = _find_payload_call(snapshot, index)
    if call is None:
        return f"🛠️ Tool call #{index} is no longer available."

    symbol = _status_symbol(str(call.get("status") or ""), call.get("ok"))
    duration = call.get("duration_text") or _duration_text(call.get("duration"))
    title = f"{symbol} #{call.get('index')} {_tool_emoji(str(call.get('name') or ''))} {call.get('name') or 'tool'}"
    if duration:
        title += f" · {duration}"

    args = str(call.get("args_preview") or "").strip()
    output = str(call.get("result_preview") or "").strip()
    preview = str(call.get("preview") or "").strip()
    lines = [title]
    if preview:
        lines.append(f"Preview: {preview}")

    if section == "args":
        lines.append("Args:")
        lines.append(args or "(empty)")
    elif section == "output":
        lines.append("Output preview:")
        lines.append(output or "(not available yet)")
    else:
        if args:
            lines.append("Args:")
            lines.append(args)
        if output:
            lines.append("Output preview:")
            lines.append(output)
        if not args and not output:
            lines.append("No details available yet.")
    return _clip("\n".join(lines), max_chars)


def _summary_call_lines(
    call: dict[str, Any],
    *,
    include_args: bool = False,
    include_output: bool = False,
) -> list[str]:
    symbol = _status_symbol(str(call.get("status") or ""), call.get("ok"))
    duration = call.get("duration_text") or _duration_text(call.get("duration"))
    suffix = f" · {duration}" if duration else ""
    lines = [f"{symbol} #{call.get('index')} {_tool_emoji(str(call.get('name') or ''))} {call.get('name') or 'tool'}{suffix}"]
    preview = str(call.get("preview") or "").strip()
    if preview:
        lines.append(f"  preview: {preview}")
    if include_args:
        args = str(call.get("args_preview") or "").strip()
        if args:
            lines.append(f"  args: {_clip(args, 500)}")
    if include_output:
        output = str(call.get("result_preview") or "").strip()
        if output:
            lines.append(f"  output: {_clip(output, 500)}")
    return lines


def _find_payload_call(snapshot: dict[str, Any], index: int) -> Optional[dict[str, Any]]:
    for call in snapshot.get("calls") or []:
        try:
            if int(call.get("index")) == int(index):
                return call
        except (TypeError, ValueError):
            continue
    return None
