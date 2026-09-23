"""Prompt-size diagnostic: ``hermes prompt-size``.

Builds a real inspection agent (so the numbers match what ships on the wire) but never makes a
network call: dummy credentials force ``AIAgent.__init__`` down the direct-construction path, then
``build_system_prompt_parts`` / ``agent.tools`` are inspected offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)

# A rendered skill entry is ``    - name: desc`` (or ``    - name``); category headers use two
# leading spaces, so the four-space + ``- `` prefix isolates skill lines.
_SKILL_LINE_PREFIX = "    - "

# Posture-demoted categories render all visible skill names on one shared line.
_NAMES_ONLY_LINE_RE = re.compile(r"^  .+ \[names only\]: (?P<names>.+)$")

# Cap the human-readable "Skills by size" table; ``--json`` always has them all.
_SKILLS_TABLE_LIMIT = 20


def _bytes(s: str) -> int:
    return len(s.encode("utf-8"))


def _size(text: str) -> Dict[str, int]:
    return {"chars": len(text), "bytes": _bytes(text)}


def _fmt_kb(n: int) -> str:
    return f"{n / 1024:.1f} KB"


def _tool_name(tool: Any) -> str:
    """Callable name of a tool schema (OpenAI ``function`` shape)."""
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    return str(fn["name"]) if isinstance(fn, dict) and fn.get("name") else str(tool.get("name", ""))


def _build_inspection_agent(platform: str) -> Any:
    """Offline AIAgent for prompt inspection: dummy ``api_key`` + ``base_url`` force the
    direct-construction path (no provider auto-detection, no network); toolsets resolve the way
    the gateway does so the breakdown matches a real session.
    """
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    from agent.skill_utils import parse_config_string_list

    cfg = load_config()
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    agent_cfg = cfg.get("agent") or {}
    return AIAgent(
        model=model_cfg.get("default") or model_cfg.get("model") or "",
        api_key="inspect-only", base_url="https://openrouter.ai/api/v1", quiet_mode=True, save_trajectories=False,
        platform=platform, enabled_toolsets=sorted(_get_platform_tools(cfg, platform)),
        disabled_toolsets=parse_config_string_list(agent_cfg.get("disabled_toolsets")) or None,
    )


def _skill_md_paths_by_name() -> Dict[str, Path]:
    """Map each installed skill's frontmatter ``name`` AND directory name to its ``SKILL.md``.
    Local skills win over external dirs (``get_all_skills_dirs`` yields local first), matching
    the index's own precedence.
    """
    from agent.skill_utils import get_all_skills_dirs, iter_skill_index_files, parse_frontmatter

    mapping: Dict[str, Path] = {}
    for skills_dir in get_all_skills_dirs():
        if not skills_dir.exists():
            continue
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            dir_name = skill_file.parent.name
            try:
                frontmatter, _ = parse_frontmatter(skill_file.read_text(encoding="utf-8"))
                frontmatter_name = str(frontmatter.get("name") or dir_name)
            except Exception:
                frontmatter_name = dir_name
            mapping.setdefault(frontmatter_name, skill_file)  # first (local) occurrence wins
            mapping.setdefault(dir_name, skill_file)
    return mapping


def _compute_skills_breakdown(skills_block: str) -> List[Dict[str, Any]]:
    """Per-skill byte breakdown parsed from the rendered ``<available_skills>``.

    ``index_line_bytes`` is the skill's attributed always-on index cost. For a compact
    ``[names only]`` line each name keeps its own bytes plus an even share of the shared prefix
    and separators.
    """
    name_to_path = _skill_md_paths_by_name()
    entries: List[Dict[str, Any]] = []

    def append_entry(name: str, **index_fields: int) -> None:  # kwarg order == output key order
        path = name_to_path.get(name)
        md_bytes: Optional[int] = None
        try:
            md_bytes = path.stat().st_size if path is not None else None
        except OSError:
            pass
        entries.append({"name": name, **index_fields, "skill_md_bytes": md_bytes, "path": str(path) if path is not None else ""})

    for line in skills_block.splitlines():
        line_bytes = _bytes(line)
        if (compact_match := _NAMES_ONLY_LINE_RE.match(line)) is not None:
            names = [n.strip() for n in compact_match.group("names").split(",") if n.strip()]
            name_bytes = [_bytes(name) for name in names]
            shared_base, shared_remainder = divmod(line_bytes - sum(name_bytes), len(names)) if names else (0, 0)
            for index, name in enumerate(names):
                shared = shared_base + (1 if index < shared_remainder else 0)
                append_entry(name, index_line_bytes=name_bytes[index] + shared, index_line_total_bytes=line_bytes,
                             index_line_shared_bytes=shared, index_line_skill_count=len(names))
        elif line.startswith(_SKILL_LINE_PREFIX):
            # Partition on ``": "`` (not ``:``) so namespaced names like ``codex:rescue`` stay intact.
            name = line[len(_SKILL_LINE_PREFIX):].partition(": ")[0].strip()
            if name:
                append_entry(name, index_line_bytes=line_bytes, index_line_total_bytes=line_bytes,
                             index_line_shared_bytes=0, index_line_skill_count=1)
    entries.sort(key=lambda e: (-(e["skill_md_bytes"] or 0), e["name"]))
    return entries


def _compute_toolsets_breakdown(tools: List[Any]) -> List[Dict[str, Any]]:
    """Per-toolset schema-byte breakdown, largest-first (tie-broken by name). Each tool is
    attributed to its single canonical toolset so ``json_bytes`` sums to the grand total.
    """
    from tools.registry import registry

    tool_to_toolset = registry.get_tool_to_toolset_map()
    groups: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        toolset = tool_to_toolset.get(_tool_name(tool)) or "(unknown)"
        group = groups.setdefault(toolset, {"toolset": toolset, "tool_count": 0, "json_bytes": 0})
        group["tool_count"] += 1
        group["json_bytes"] += _bytes(json.dumps(tool, ensure_ascii=False))
    return sorted(groups.values(), key=lambda g: (-g["json_bytes"], g["toolset"]))


def compute_prompt_breakdown(platform: str = "cli") -> Dict[str, Any]:
    """Prompt-size measurements for a fresh session: ``system_prompt``, ``skills_index``,
    ``memory``, ``user_profile``, ``tools``, ``sections`` (the three prompt tiers), and the
    largest-first ``skills_breakdown`` / ``toolsets_breakdown`` ("what should I disable?").
    """
    from agent.system_prompt import build_system_prompt, build_system_prompt_parts

    agent = _build_inspection_agent(platform)
    parts = build_system_prompt_parts(agent)
    full = build_system_prompt(agent)
    stable, context, volatile = (parts.get(k, "") for k in ("stable", "context", "volatile"))

    # The skills index lives in the volatile tier (moved from stable so skill edits don't
    # invalidate the cached identity prefix); fall back to stable for older layouts.
    skills_match = _SKILLS_BLOCK_RE.search(volatile) or _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""

    # Memory + user profile are joined into ``volatile``; re-derive them from the store so the
    # numbers stay attributable.
    memory_block = user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is not None:
        try:
            if getattr(agent, "_memory_enabled", True):
                memory_block = store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_user_profile_enabled", True):
                user_block = store.format_for_system_prompt("user") or ""
        except Exception:
            pass

    tools = getattr(agent, "tools", None) or []
    sections: List[Tuple[str, int, int]] = [
        (label, len(text), _bytes(text))
        for label, text in (("stable (identity/guidance/skills)", stable), ("context (AGENTS.md/cwd files)", context),
                            ("volatile (memory/profile/timestamp)", volatile))
    ]
    return {
        "platform": platform,
        "model": getattr(agent, "model", "") or "",
        "system_prompt": _size(full),
        "skills_index": _size(skills_index),
        "memory": _size(memory_block),
        "user_profile": _size(user_block),
        "tools": {"count": len(tools), "json_bytes": _bytes(json.dumps(tools, ensure_ascii=False))},
        "sections": sections,
        "skills_breakdown": _compute_skills_breakdown(skills_index),
        "toolsets_breakdown": _compute_toolsets_breakdown(tools),
    }


def render_breakdown(data: Dict[str, Any]) -> str:
    """Render the breakdown as plain text suitable for a terminal."""
    sp = data["system_prompt"]
    tools = data["tools"]
    lines: List[str] = [
        f"Prompt-size breakdown (platform={data['platform']}, model={data['model'] or 'unset'})", "",
        f"  System prompt total : {sp['bytes']:>8,} B  ({_fmt_kb(sp['bytes'])}, {sp['chars']:,} chars)", "",
        "  Major blocks:",
    ]
    for label, key in (("skills index", "skills_index"), ("memory", "memory"), ("user profile", "user_profile")):
        byts = data[key]["bytes"]
        lines.append(f"    {label:<19}: {byts:>8,} B  ({_fmt_kb(byts)})")
    lines += ["", "  Prompt tiers:"] + [f"    {label:<36}: {byts:>8,} B  ({_fmt_kb(byts)})" for label, _chars, byts in data["sections"]]
    lines += ["", f"  Tool schemas         : {tools['json_bytes']:>8,} B  ({_fmt_kb(tools['json_bytes'])}, {tools['count']} tools)"]

    if toolsets := data.get("toolsets_breakdown") or []:
        lines += ["", "  Toolsets by size (tool-schema JSON, largest first):", f"    {'toolset':<22} {'tools':>5}  {'schema':>10}"]
        lines += [f"    {ts['toolset']:<22} {ts['tool_count']:>5}  {ts['json_bytes']:>8,} B  ({_fmt_kb(ts['json_bytes'])})" for ts in toolsets]

    # Per-skill cost — index line (always shipped) vs SKILL.md (read on load).
    if skills := data.get("skills_breakdown") or []:
        lines += ["", "  Skills by size (SKILL.md on-disk = read cost; index cost = attributed always-on bytes, largest first):",
                  f"    {'skill':<28} {'SKILL.md':>10}  {'index cost':>10}"]
        shown = skills[:_SKILLS_TABLE_LIMIT]
        for sk in shown:
            md = sk["skill_md_bytes"]
            md_str = f"{md:>8,} B" if md is not None else f"{'n/a':>10}"
            name = sk["name"] if len(sk["name"]) <= 28 else sk["name"][:27] + "…"
            lines.append(f"    {name:<28} {md_str}  {sk['index_line_bytes']:>8,} B")
        if (remaining := len(skills) - len(shown)) > 0:
            lines.append(f"    … and {remaining} more (use --json for the full list)")
    return "\n".join(lines)


def cmd_prompt_size(args: Any) -> None:
    """Entry point for ``hermes prompt-size``."""
    try:
        data = compute_prompt_breakdown(getattr(args, "platform", "cli") or "cli")
    except Exception as e:
        print(f"Could not compute prompt-size breakdown: {e}")
        return
    print(json.dumps(data, ensure_ascii=False, indent=2) if getattr(args, "json", False) else render_breakdown(data))
