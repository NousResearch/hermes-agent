"""Format and apply the runtime-owned route/depth status bar."""

from __future__ import annotations

import re
from typing import Tuple

from agent.turn_receipt import TurnReceipt

_ROUTE_BAR_RE = re.compile(r"^\s*路径：[^\n\r]*(?:\r?\n)?", re.UNICODE)


def _tool_field(receipt: TurnReceipt) -> str:
    if receipt.tool_total <= 0 and not receipt.tool_names:
        return "工具 none"
    names = receipt.tool_names[:3]
    if names:
        joined = "+".join(names)
        if receipt.tool_total > len(names):
            joined += f"+{receipt.tool_total - len(names)}"
    else:
        joined = str(receipt.tool_total)
    if receipt.tool_failed:
        joined += f"({receipt.tool_failed} failed)"
    return f"工具 {joined}"


def _count_field(label: str, value: int | None) -> str:
    return f"{label} unknown" if value is None else f"{label} {value}"


def _opencode_field(state: str) -> str:
    normalized = (state or "unknown").strip().lower()
    if normalized in {"called", "yes", "true", "1", "已调用"}:
        return "OpenCode 已调用"
    if normalized in {"not_called", "none", "false", "0", "未调用"}:
        return "OpenCode 未调用"
    return "OpenCode unknown"


def _human_field(state: str) -> str:
    normalized = (state or "unknown").strip().lower()
    if normalized in {"seen", "yes", "true", "1", "ok", "✓"}:
        return "人话 ✓"
    if normalized in {"not_seen", "none", "false", "0"}:
        return "人话 none"
    return "人话 unknown"


def _elapsed_field(receipt: TurnReceipt) -> str:
    if receipt.elapsed_seconds is None:
        return "用时 N/A"
    seconds = max(0, int(round(receipt.elapsed_seconds)))
    return f"用时 {seconds}s"


def _evidence_field(status: str) -> str:
    normalized = (status or "unknown").strip().lower()
    if normalized in {"ok", "true", "yes", "pass", "passed", "✓"}:
        return "证据 ✓"
    if normalized in {"failed", "fail", "error", "errored"}:
        return "证据 failed"
    if normalized in {"partial", "incomplete"}:
        return "证据 partial"
    return "证据 unknown"


def format_route_depth_bar(receipt: TurnReceipt) -> str:
    """Return the one-line runtime-owned status bar for ``receipt``."""

    fields = [
        f"路径：{receipt.route or 'native'}",
        f"原因：{receipt.reason or 'runtime_default'}",
        _opencode_field(receipt.opencode_state),
        _tool_field(receipt),
        _count_field("agents", receipt.agents_count),
        _count_field("subagents", receipt.subagents_count),
        _human_field(receipt.human_language_state),
        _elapsed_field(receipt),
        _evidence_field(receipt.evidence_status),
    ]
    return "｜".join(fields)


def strip_route_depth_bar(text: str) -> str:
    """Remove a leading model-authored/runtime route bar if present."""

    if not isinstance(text, str) or not text:
        return text or ""
    return _ROUTE_BAR_RE.sub("", text, count=1).lstrip("\n")


def apply_route_depth_bar(text: str, receipt: TurnReceipt) -> Tuple[str, bool]:
    """Prepend exactly one runtime route/depth bar to ``text``.

    Any leading model-authored ``路径：`` line is stripped/replaced.  Returns
    ``(new_text, changed)``.
    """

    original = text or ""
    body = strip_route_depth_bar(original)
    bar = format_route_depth_bar(receipt)
    if body:
        updated = f"{bar}\n{body}"
    else:
        updated = bar
    return updated, updated != original
