"""Prompt/system-prompt size diagnostics.

Provides the existing ``hermes prompt-size`` CLI subcommand plus the shared
rendering used by the in-session ``/system_prompt`` slash command.  Diagnostics
measure prompt construction and tool schemas locally; they never make model API
calls and never print raw prompt content.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# The skills index is wrapped in this tag pair inside the stable tier.
_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def _approx_tokens_from_chars(chars: int) -> int:
    """Cheap display-only estimate.  Provider truth is shown without ``~``."""
    return max(0, (chars + 3) // 4)


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "unknown"
    fn = schema.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn.get("name"))
    if schema.get("name"):
        return str(schema.get("name"))
    return "unknown"


def _build_inspection_agent(platform: str) -> Any:
    """Construct an offline AIAgent for prompt inspection.

    Dummy ``api_key`` + ``base_url`` force the direct-construction path in
    ``run_agent.py`` (no provider auto-detection, no network). Toolsets and
    platform come from config so the breakdown matches a real fresh session.
    """
    from run_agent import AIAgent
    from hermes_cli.config import load_config

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    model = model_cfg.get("default") or model_cfg.get("model") or ""

    return AIAgent(
        model=model,
        api_key="inspect-only",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        save_trajectories=False,
        platform=platform,
    )


def _prompt_parts_for_agent(agent: Any) -> Tuple[Dict[str, str], str]:
    from agent.system_prompt import build_system_prompt, build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    cached = getattr(agent, "_cached_system_prompt", None)
    # If a resident agent already has a cached prompt, read it.  Do not assign or
    # invalidate anything: /system_prompt is strictly observational.
    full = cached if isinstance(cached, str) and cached else build_system_prompt(agent)
    return parts, full


def _toolset_breakdown(agent: Any) -> List[Dict[str, Any]]:
    try:
        from run_agent import get_toolset_for_tool
    except Exception:  # pragma: no cover - defensive for unusual embeddings
        get_toolset_for_tool = lambda _name: None  # noqa: E731

    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"tools": 0, "chars": 0, "bytes": 0})
    for schema in getattr(agent, "tools", None) or []:
        name = _tool_name(schema)
        toolset = get_toolset_for_tool(name) or "unknown"
        blob = json.dumps(schema, ensure_ascii=False)
        buckets[toolset]["tools"] += 1
        buckets[toolset]["chars"] += len(blob)
        buckets[toolset]["bytes"] += _bytes(blob)

    rows = [
        {"toolset": toolset, **vals}
        for toolset, vals in buckets.items()
    ]
    rows.sort(key=lambda r: (-int(r["chars"]), str(r["toolset"])))
    return rows


def compute_prompt_breakdown(platform: str = "cli") -> Dict[str, Any]:
    """Return a dict of prompt-size measurements for a fresh session.

    Backwards-compatible data shape used by ``hermes prompt-size``.
    """
    agent = _build_inspection_agent(platform)
    data = compute_system_prompt_breakdown(agent=agent, platform=platform, resident=False)
    return {
        "platform": platform,
        "model": data["model"],
        "system_prompt": {"chars": data["system_prompt"]["chars"], "bytes": data["system_prompt"]["bytes"]},
        "skills_index": {"chars": data["major_blocks"]["skills_catalog"]["chars"], "bytes": data["major_blocks"]["skills_catalog"]["bytes"]},
        "memory": {"chars": data["major_blocks"]["memory"]["chars"], "bytes": data["major_blocks"]["memory"]["bytes"]},
        "user_profile": {"chars": data["major_blocks"]["user_profile"]["chars"], "bytes": data["major_blocks"]["user_profile"]["bytes"]},
        "tools": {"count": data["tools"]["count"], "json_bytes": data["tools"]["bytes"]},
        "sections": [(r["label"], r["chars"], r["bytes"]) for r in data["tiers"]],
    }


def compute_system_prompt_breakdown(
    *,
    agent: Optional[Any] = None,
    platform: str = "cli",
    resident: Optional[bool] = None,
) -> Dict[str, Any]:
    """Measure system prompt + tool schemas without exposing raw content.

    Passing a resident ``agent`` reads its current cached prompt when available;
    passing no agent constructs an offline inspection agent.  Neither path makes
    model API calls or mutates session counters/cache state.
    """
    if agent is None:
        agent = _build_inspection_agent(platform)
        if resident is None:
            resident = False
    elif resident is None:
        resident = True

    parts, full = _prompt_parts_for_agent(agent)
    stable = parts.get("stable", "")
    context = parts.get("context", "")
    volatile = parts.get("volatile", "")

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_catalog = skills_match.group(0) if skills_match else ""

    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is not None:
        try:
            if getattr(agent, "_memory_enabled", True):
                memory_block = store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_user_profile_enabled", True):
                user_block = store.format_for_system_prompt("user") or ""
        except Exception:
            pass

    tool_rows = _toolset_breakdown(agent)
    tools_chars = sum(int(r["chars"]) for r in tool_rows)
    tools_bytes = sum(int(r["bytes"]) for r in tool_rows)
    tools_count = sum(int(r["tools"]) for r in tool_rows)

    tiers = [
        {"label": "stable", "description": "identity / guidance / skills catalog", "chars": len(stable), "bytes": _bytes(stable)},
        {"label": "context", "description": "cwd context files / caller system message", "chars": len(context), "bytes": _bytes(context)},
        {"label": "volatile", "description": "memory / user profile / timestamp", "chars": len(volatile), "bytes": _bytes(volatile)},
    ]

    major = {
        "skills_catalog": {"label": "skills catalog (<available_skills>)", "chars": len(skills_catalog), "bytes": _bytes(skills_catalog)},
        "memory": {"label": "MEMORY.md snapshot", "chars": len(memory_block), "bytes": _bytes(memory_block)},
        "user_profile": {"label": "USER.md profile", "chars": len(user_block), "bytes": _bytes(user_block)},
    }

    return {
        "platform": platform,
        "resident": bool(resident),
        "model": getattr(agent, "model", "") or "unset",
        "provider": getattr(agent, "provider", "") or "unset",
        "system_prompt": {"chars": len(full), "bytes": _bytes(full)},
        "tiers": tiers,
        "major_blocks": major,
        "tools": {"count": tools_count, "chars": tools_chars, "bytes": tools_bytes, "toolsets": tool_rows},
        "cache_note": "Hermes caches the whole system prompt as one provider prefix block; tiers are measurement labels, not separate cache controls.",
    }


def _fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def _fmt_measure(chars: int, byts: int) -> str:
    return f"~{_approx_tokens_from_chars(chars):,} tok · {chars:,} chars · {_fmt_kb(byts)}"


def render_system_prompt_breakdown(data: Dict[str, Any], *, markdown: bool = False, max_toolsets: int = 12) -> str:
    """Render /system_prompt output for CLI/gateway without raw prompt text."""
    bold_l = "**" if markdown else ""
    bold_r = "**" if markdown else ""
    lines: List[str] = []
    sp = data["system_prompt"]
    tools = data["tools"]
    fixed_chars = int(sp["chars"]) + int(tools["chars"])
    fixed_bytes = int(sp["bytes"]) + int(tools["bytes"])

    lines.append(f"🧠 {bold_l}System prompt breakdown{bold_r}")
    lines.append(f"Model: {data.get('provider', 'unset')}/{data.get('model', 'unset')}")
    lines.append(f"Source: {'resident agent (read-only)' if data.get('resident') else 'offline inspection agent'}")
    lines.append("")
    lines.append(f"System prompt: {_fmt_measure(sp['chars'], sp['bytes'])}")
    lines.append(f"Tool schemas: ~{_approx_tokens_from_chars(tools['chars']):,} tok · {tools['count']} tools · {_fmt_kb(tools['bytes'])}")
    lines.append(f"Fixed overhead: {_fmt_measure(fixed_chars, fixed_bytes)}")
    lines.append("")
    lines.append(f"{bold_l}System prompt tiers{bold_r}")
    for row in data["tiers"]:
        lines.append(f"- {row['label']}: {_fmt_measure(row['chars'], row['bytes'])} — {row['description']}")
    lines.append("")
    lines.append(f"{bold_l}Major blocks{bold_r}")
    for key in ("skills_catalog", "memory", "user_profile"):
        row = data["major_blocks"][key]
        lines.append(f"- {row['label']}: {_fmt_measure(row['chars'], row['bytes'])}")
    lines.append("- skill tools: counted under Tool schemas (skills_list / skill_view / skill_manage), not the skills catalog")
    lines.append("")
    lines.append(f"{bold_l}Tool schemas by toolset{bold_r}")
    for row in tools["toolsets"][:max_toolsets]:
        lines.append(f"- {row['toolset']}: ~{_approx_tokens_from_chars(row['chars']):,} tok · {row['tools']} tool(s)")
    if len(tools["toolsets"]) > max_toolsets:
        lines.append(f"- … {len(tools['toolsets']) - max_toolsets} more toolset(s)")
    lines.append("")
    lines.append(f"Cache: {data['cache_note']}")
    lines.append("Privacy: raw SOUL/MEMORY/USER/system-prompt text is never printed by this command.")
    return "\n".join(lines)


def render_breakdown(data: Dict[str, Any]) -> str:
    """Render the legacy prompt-size breakdown as plain terminal text."""
    lines: List[str] = []
    sp = data["system_prompt"]
    lines.append(f"Prompt-size breakdown (platform={data['platform']}, model={data['model'] or 'unset'})")
    lines.append("")
    lines.append(f"  System prompt total : {sp['bytes']:>8,} B  ({_fmt_kb(sp['bytes'])}, {sp['chars']:,} chars)")
    lines.append("")
    lines.append("  Major blocks:")
    si = data["skills_index"]
    mem = data["memory"]
    up = data["user_profile"]
    lines.append(f"    skills index       : {si['bytes']:>8,} B  ({_fmt_kb(si['bytes'])})")
    lines.append(f"    memory             : {mem['bytes']:>8,} B  ({_fmt_kb(mem['bytes'])})")
    lines.append(f"    user profile       : {up['bytes']:>8,} B  ({_fmt_kb(up['bytes'])})")
    lines.append("")
    lines.append("  Prompt tiers:")
    for label, chars, byts in data["sections"]:
        lines.append(f"    {label:<36}: {byts:>8,} B  ({_fmt_kb(byts)})")
    lines.append("")
    tools = data["tools"]
    lines.append(f"  Tool schemas         : {tools['json_bytes']:>8,} B  ({_fmt_kb(tools['json_bytes'])}, {tools['count']} tools)")
    return "\n".join(lines)


def cmd_prompt_size(args: Any) -> None:
    """Entry point for ``hermes prompt-size``."""
    platform = getattr(args, "platform", "cli") or "cli"
    as_json = getattr(args, "json", False)
    try:
        data = compute_prompt_breakdown(platform)
    except Exception as e:
        print(f"Could not compute prompt-size breakdown: {e}")
        return
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render_breakdown(data))
