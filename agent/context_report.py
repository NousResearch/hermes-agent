"""Context-window composition diagnostics for CLI, gateway, and TUI."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from agent.model_metadata import estimate_messages_tokens_rough


def _text_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _value_tokens(value: Any) -> int:
    if value is None:
        return 0
    return _text_tokens(str(value))


def _pct(part: int, total: int) -> float:
    return (part / total * 100.0) if total else 0.0


def _bar(percent: float, width: int = 12) -> str:
    safe = max(0.0, min(100.0, percent))
    filled = round((safe / 100.0) * width)
    return "[" + ("#" * filled) + ("." * max(0, width - filled)) + "]"


def _fmt_tokens(value: int) -> str:
    return f"{int(value):,}"


def _format_row(label: str, tokens: int, total: int, note: str = "") -> str:
    percent = _pct(tokens, total)
    suffix = f"  {note}" if note else ""
    return f"  {label:<22} {_fmt_tokens(tokens):>10} tok  {_bar(percent)} {percent:5.1f}%{suffix}"


def _format_compact_row(label: str, tokens: int, denominator: int, note: str = "") -> str:
    percent = _pct(tokens, denominator)
    suffix = f", {note}" if note else ""
    return f"{label}: {_fmt_tokens(tokens)} tok ({percent:.1f}%{suffix})"


def _tool_function_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return "tool"
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "tool")
    return str(tool.get("name") or "tool")


def _message_tokens(message: Dict[str, Any]) -> int:
    try:
        return estimate_messages_tokens_rough([message])
    except Exception:
        return _value_tokens(message)


def _extract_tool_calls(message: Dict[str, Any]) -> Iterable[tuple[str, str, str]]:
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list):
        return []
    out: list[tuple[str, str, str]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id") or call.get("call_id") or "")
        fn = call.get("function")
        if isinstance(fn, dict):
            name = str(fn.get("name") or "")
            args = str(fn.get("arguments") or "")
        else:
            name = str(call.get("name") or "")
            args = str(call.get("arguments") or "")
        if call_id or name:
            out.append((call_id, name, args))
    return out


def _parse_skill_name(args_json: str, content: Any) -> Optional[str]:
    try:
        args = json.loads(args_json) if args_json else {}
        if isinstance(args, dict) and args.get("name"):
            return str(args.get("name"))
    except Exception:
        pass
    try:
        data = json.loads(content) if isinstance(content, str) else content
        if isinstance(data, dict) and data.get("name"):
            return str(data.get("name"))
    except Exception:
        pass
    return None


def _message_breakdown(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    role_totals: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    tool_call_by_id: dict[str, tuple[str, str]] = {}
    top_messages: list[dict[str, Any]] = []
    top_tool_results: list[dict[str, Any]] = []
    loaded_skills: list[dict[str, Any]] = []

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        tokens = _message_tokens(message)
        role_totals[role] = role_totals.get(role, 0) + tokens
        role_counts[role] = role_counts.get(role, 0) + 1
        top_messages.append({"index": idx, "role": role, "tokens": tokens})

        for call_id, name, args in _extract_tool_calls(message):
            if call_id:
                tool_call_by_id[call_id] = (name, args)

        if role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            tool_name, args_json = tool_call_by_id.get(call_id, ("tool", ""))
            entry = {
                "index": idx,
                "tool": tool_name or "tool",
                "tokens": tokens,
            }
            top_tool_results.append(entry)
            if tool_name == "skill_view":
                skill_name = _parse_skill_name(args_json, message.get("content"))
                loaded_skills.append(
                    {
                        "name": skill_name or "skill_view",
                        "tokens": tokens,
                        "index": idx,
                    }
                )

    top_messages.sort(key=lambda item: item["tokens"], reverse=True)
    top_tool_results.sort(key=lambda item: item["tokens"], reverse=True)
    loaded_skills.sort(key=lambda item: item["tokens"], reverse=True)
    return {
        "total": sum(role_totals.values()),
        "role_totals": role_totals,
        "role_counts": role_counts,
        "top_messages": top_messages[:8],
        "top_tool_results": top_tool_results[:8],
        "loaded_skills": loaded_skills[:8],
    }


def _skills_index_breakdown(system_prompt: str) -> Dict[str, Any]:
    start = system_prompt.find("<available_skills>")
    end = system_prompt.find("</available_skills>")
    if start == -1 or end == -1 or end <= start:
        return {"tokens": 0, "entries": []}

    block = system_prompt[start : end + len("</available_skills>")]
    entries: list[dict[str, Any]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:]
        name = body.split(":", 1)[0].strip() or "skill"
        entries.append({"name": name, "tokens": _text_tokens(stripped)})
    entries.sort(key=lambda item: item["tokens"], reverse=True)
    return {"tokens": _text_tokens(block), "entries": entries[:8]}


def build_context_report(
    agent: Any,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return an estimated prompt-composition report for the next request."""
    messages = list(messages or getattr(agent, "_session_messages", None) or [])

    parts: dict[str, str] = {}
    try:
        parts = agent._build_system_prompt_parts(None)
    except Exception:
        parts = {}
    cached_prompt = getattr(agent, "_cached_system_prompt", None) or ""
    if not cached_prompt and parts:
        cached_prompt = "\n\n".join(
            p for p in (parts.get("stable", ""), parts.get("context", ""), parts.get("volatile", "")) if p
        )
    if not cached_prompt:
        try:
            cached_prompt = agent._build_system_prompt(None)
        except Exception:
            cached_prompt = ""

    tools = list(getattr(agent, "tools", None) or [])
    tool_entries = [
        {"name": _tool_function_name(tool), "tokens": _value_tokens(tool)}
        for tool in tools
    ]
    tool_entries.sort(key=lambda item: item["tokens"], reverse=True)
    tools_total = _value_tokens(tools)

    system_total = _text_tokens(cached_prompt)
    system_parts = {
        "stable": _text_tokens(parts.get("stable", "")),
        "context": _text_tokens(parts.get("context", "")),
        "volatile": _text_tokens(parts.get("volatile", "")),
    }
    message_info = _message_breakdown(messages)
    skills_index = _skills_index_breakdown(cached_prompt)

    total = system_total + tools_total + message_info["total"]
    comp = getattr(agent, "context_compressor", None)
    context_length = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    threshold_tokens = int(getattr(comp, "threshold_tokens", 0) or 0) if comp else 0
    last_prompt_tokens = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0

    return {
        "model": getattr(agent, "model", "") or "",
        "provider": getattr(agent, "provider", "") or "",
        "context_length": context_length,
        "threshold_tokens": threshold_tokens,
        "compression_count": int(getattr(comp, "compression_count", 0) or 0) if comp else 0,
        "last_prompt_tokens": last_prompt_tokens,
        "estimated_total": total,
        "components": {
            "system_prompt": system_total,
            "tools_schema": tools_total,
            "messages": message_info["total"],
        },
        "system_parts": system_parts,
        "message_info": message_info,
        "tools": {
            "count": len(tools),
            "top": tool_entries[:8],
        },
        "skills_index": skills_index,
    }


def format_context_report_compact(report: Dict[str, Any]) -> str:
    """Return a Telegram-friendly context report without alignment bars."""
    total = int(report.get("estimated_total") or 0)
    context_length = int(report.get("context_length") or 0)
    denominator = context_length or total
    window_pct = _pct(total, context_length)
    lines: list[str] = []

    model = report.get("model") or "unknown"
    provider = report.get("provider") or ""
    title_model = f"{provider}:{model}" if provider and provider not in str(model) else str(model)

    lines.append("🧠 Context Window")
    lines.append(f"Model: {title_model}")
    if context_length:
        lines.append(f"Estimate: {_fmt_tokens(total)} / {_fmt_tokens(context_length)} tok ({window_pct:.1f}%)")
    else:
        lines.append(f"Estimate: {_fmt_tokens(total)} tok")

    threshold = int(report.get("threshold_tokens") or 0)
    if threshold:
        lines.append(f"Compression threshold: {_fmt_tokens(threshold)} tok")
    last_prompt = int(report.get("last_prompt_tokens") or 0)
    if last_prompt and abs(last_prompt - total) > max(1024, int(total * 0.05)):
        lines.append(f"Last provider prompt: {_fmt_tokens(last_prompt)} tok")
    compressions = int(report.get("compression_count") or 0)
    if compressions:
        lines.append(f"Compressions: {compressions}")

    components = report.get("components") or {}
    lines.append("")
    lines.append("📦 Buckets")
    lines.append(_format_compact_row("System prompt", int(components.get("system_prompt") or 0), denominator))
    lines.append(_format_compact_row(
        "Tools schema",
        int(components.get("tools_schema") or 0),
        denominator,
        f"{(report.get('tools') or {}).get('count', 0)} tools",
    ))
    lines.append(_format_compact_row("Messages", int(components.get("messages") or 0), denominator))

    system_parts = report.get("system_parts") or {}
    if any(system_parts.values()):
        lines.append("")
        lines.append("🧩 System prompt")
        for label, key in (("Stable", "stable"), ("Context files", "context"), ("Volatile", "volatile")):
            value = int(system_parts.get(key) or 0)
            if value:
                lines.append(_format_compact_row(label, value, denominator))

    message_info = report.get("message_info") or {}
    role_totals = message_info.get("role_totals") or {}
    role_counts = message_info.get("role_counts") or {}
    if role_totals:
        lines.append("")
        lines.append("💬 Messages")
        for role, tokens in sorted(role_totals.items(), key=lambda item: item[1], reverse=True):
            lines.append(_format_compact_row(f"{role} ({role_counts.get(role, 0)})", int(tokens), denominator))

    tools = report.get("tools") or {}
    if tools.get("top"):
        lines.append("")
        lines.append("🛠 Heaviest tool schemas")
        for item in tools["top"][:5]:
            lines.append(_format_compact_row(str(item["name"])[:28], int(item["tokens"]), denominator))

    if message_info.get("top_tool_results"):
        lines.append("")
        lines.append("📎 Heaviest tool results")
        for item in message_info["top_tool_results"][:5]:
            lines.append(_format_compact_row(
                str(item["tool"])[:28],
                int(item["tokens"]),
                denominator,
                f"msg #{item['index']}",
            ))

    skills_index = report.get("skills_index") or {}
    loaded_skills = message_info.get("loaded_skills") or []
    if int(skills_index.get("tokens") or 0) or loaded_skills:
        lines.append("")
        lines.append("🎯 Skills")
        if int(skills_index.get("tokens") or 0):
            lines.append(_format_compact_row("Skills index", int(skills_index["tokens"]), denominator))
        for item in loaded_skills[:5]:
            lines.append(_format_compact_row(
                str(item["name"])[:28],
                int(item["tokens"]),
                denominator,
                f"loaded, msg #{item['index']}",
            ))
        if skills_index.get("entries"):
            names = ", ".join(
                f"{item['name']} ~{_fmt_tokens(int(item['tokens']))} tok"
                for item in skills_index["entries"][:3]
            )
            if names:
                lines.append(f"Index entries: {names}")

    return "\n".join(lines)


def format_context_report(report: Dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        return format_context_report_compact(report)

    total = int(report.get("estimated_total") or 0)
    context_length = int(report.get("context_length") or 0)
    window_pct = _pct(total, context_length)
    lines: list[str] = []

    model = report.get("model") or "unknown"
    provider = report.get("provider") or ""
    title_model = f"{provider}:{model}" if provider and provider not in str(model) else str(model)
    lines.append("Context composition")
    lines.append(f"Model: {title_model}")
    if context_length:
        lines.append(f"Estimated next prompt: {_fmt_tokens(total)} / {_fmt_tokens(context_length)} tokens ({window_pct:.1f}% of window)")
    else:
        lines.append(f"Estimated next prompt: {_fmt_tokens(total)} tokens")

    threshold = int(report.get("threshold_tokens") or 0)
    if threshold:
        lines.append(f"Compression threshold: {_fmt_tokens(threshold)} tokens")
    last_prompt = int(report.get("last_prompt_tokens") or 0)
    if last_prompt and abs(last_prompt - total) > max(1024, int(total * 0.05)):
        lines.append(f"Last provider-reported prompt: {_fmt_tokens(last_prompt)} tokens")
    lines.append(f"Compressions: {int(report.get('compression_count') or 0)}")
    lines.append("")

    components = report.get("components") or {}
    lines.append("Top-level buckets")
    lines.append(_format_row("System prompt", int(components.get("system_prompt") or 0), total))
    lines.append(_format_row("Tools schema", int(components.get("tools_schema") or 0), total, f"({(report.get('tools') or {}).get('count', 0)} tools)"))
    lines.append(_format_row("Messages", int(components.get("messages") or 0), total))

    system_parts = report.get("system_parts") or {}
    if any(system_parts.values()):
        lines.append("")
        lines.append("System prompt tiers")
        for label, key in (("Stable", "stable"), ("Context files", "context"), ("Volatile", "volatile")):
            value = int(system_parts.get(key) or 0)
            if value:
                lines.append(_format_row(label, value, total))

    message_info = report.get("message_info") or {}
    role_totals = message_info.get("role_totals") or {}
    role_counts = message_info.get("role_counts") or {}
    if role_totals:
        lines.append("")
        lines.append("Messages by role")
        for role, tokens in sorted(role_totals.items(), key=lambda item: item[1], reverse=True):
            lines.append(_format_row(f"{role} ({role_counts.get(role, 0)})", int(tokens), total))

    tools = report.get("tools") or {}
    if tools.get("top"):
        lines.append("")
        lines.append("Largest tool schemas")
        for item in tools["top"]:
            lines.append(_format_row(str(item["name"])[:22], int(item["tokens"]), total))

    if message_info.get("top_tool_results"):
        lines.append("")
        lines.append("Largest tool results in messages")
        for item in message_info["top_tool_results"]:
            note = f"(message #{item['index']})"
            lines.append(_format_row(str(item["tool"])[:22], int(item["tokens"]), total, note))

    skills_index = report.get("skills_index") or {}
    loaded_skills = message_info.get("loaded_skills") or []
    if int(skills_index.get("tokens") or 0) or loaded_skills:
        lines.append("")
        lines.append("Skill-related context")
        if int(skills_index.get("tokens") or 0):
            lines.append(_format_row("Skills index", int(skills_index["tokens"]), total))
        for item in loaded_skills:
            note = f"(loaded, message #{item['index']})"
            lines.append(_format_row(str(item["name"])[:22], int(item["tokens"]), total, note))
        if skills_index.get("entries"):
            lines.append("  Largest skill index entries:")
            for item in skills_index["entries"][:5]:
                lines.append(f"    - {item['name']}: ~{_fmt_tokens(int(item['tokens']))} tok")

    return "\n".join(lines)
