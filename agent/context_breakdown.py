"""Live session context-window breakdown for UI surfaces.

Estimates system prompt tiers, tool schemas, and conversation history for the
category breakdown. Overall occupancy retains its provider-usage or estimate
provenance; category estimates are not exact tokenizer counts or gate authority.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.i18n import t

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)
_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})

# A category at zero tokens is dropped from the payload, which reads as "not
# configured" - true for MCP, memory and skills, and false for the
# conversation. Every session has one, so hiding the row at zero makes an empty
# transcript indistinguishable from a breakdown that never measured it (#87903).
#
# The membership rule, stated so a later addition argues from the same
# principle rather than from "this one felt important": a category belongs
# here when zero is a MEASUREMENT of something every session has, not the
# ABSENCE of something optional. "conversation" qualifies because a session
# cannot not have a transcript, so zero means "nothing said yet" and is worth
# showing. "mcp", "memory", "skills" and "subagent_definitions" do not: zero
# there means the user configured none, which is what dropping the row already
# communicates, and a permanent 0-token row would be noise on most hosts.
# "system_prompt" and "tool_definitions" are always present too but are never
# zero in practice, so adding them would buy nothing.
_ALWAYS_REPORTED = frozenset({"conversation"})

# id -> (dashboard color, /context glyph); declaration order is display order. The label is
# ``gateway.context.category.<id>`` resolved when the payload is built (never at import).
_CATEGORIES = {
    "system_prompt": ("var(--context-usage-system)", "■"),
    "tool_definitions": ("var(--context-usage-tools)", "▣"),
    "rules": ("var(--context-usage-rules)", "▩"),
    "skills": ("var(--context-usage-skills)", "▤"),
    "mcp": ("var(--context-usage-mcp)", "▥"),
    "subagent_definitions": ("var(--context-usage-subagents)", "▦"),
    "memory": ("var(--context-usage-memory)", "▧"),
    "conversation": ("var(--context-usage-conversation)", "▨"),
}


def _category_label(category_id: str) -> str:
    return t(f"gateway.context.category.{category_id}")
_FREE_GLYPH = "·"
_CONTEXT_SOURCES = frozenset({"local_estimate", "provider_usage", "provider_usage_plus_estimate"})
_GRID_COLUMNS = 20
_GRID_ROWS = 5  # 100 cells → 1 cell per percent of the context window
_DETAILS_TABLE_LIMIT = 15  # display cap only; the underlying data keeps everything


def _chars_to_tokens(text: str) -> int:
    from agent.model_metadata import estimate_tokens_rough
    return estimate_tokens_rough(text)


def _json_tokens(value: Any) -> int:
    return _chars_to_tokens(json.dumps(value, ensure_ascii=False)) if value else 0


def _bytes_to_tokens(size: Optional[int]) -> Optional[int]:
    from agent.model_metadata import CHARS_PER_TOKEN
    return None if size is None else (int(size) + 3) // CHARS_PER_TOKEN


def _skills_block(stable: str) -> str:
    """The live ``<available_skills>`` block inside the stable tier, or ''."""
    m = _SKILLS_BLOCK_RE.search(stable)
    return m.group(0) if m else ""


def _split_tools(tools: Sequence[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    builtin: list[dict] = []
    mcp: list[dict] = []
    subagent: list[dict] = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        name = str((fn if isinstance(fn, dict) else tool).get("name") or "")
        bucket = mcp if name.startswith("mcp_") else subagent if name in _SUBAGENT_TOOL_NAMES else builtin
        bucket.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> tuple[str, str]:
    memory_block = user_block = ""
    store = getattr(agent, "_memory_store", None)
    try:
        if store is not None and getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if store is not None and getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        pass
    return memory_block, user_block


def _strip_blocks(text: str, *blocks: str) -> str:
    for block in blocks:
        if block:
            text = text.replace(block, "")
    return text.strip()


def _join(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part).strip()


def _glyph(cat: dict[str, Any]) -> str:
    return _CATEGORIES.get(str(cat.get("id") or ""), (None, "▪"))[1]


def context_display_source(compressor: Any) -> str:
    """Distinguish the built-in preflight display seed from a provider reading.

    Engines without the built-in real-usage ledger own their occupancy figure.
    A seed never updates that ledger, even if its number later matches real usage.
    """
    real = getattr(compressor, "last_real_prompt_tokens", None)
    shown = getattr(compressor, "last_prompt_tokens", 0) or 0
    return "local_estimate" if isinstance(real, (int, float)) and shown > 0 and shown != real else "provider_usage"


def context_usage_fields(compressor: Any) -> dict[str, Any]:
    """Current occupancy only; lifetime throughput is never a context fallback."""
    used = max(0, getattr(compressor, "last_prompt_tokens", 0) or 0)
    maximum = getattr(compressor, "context_length", 0) or 0
    if not used or not maximum:
        return {}
    used = min(used, maximum)
    source = context_display_source(compressor)
    return {"context_used": used, "context_max": maximum,
            "context_percent": max(0, min(100, round(used / maximum * 100))),
            "context_source": source, "context_estimated": source != "provider_usage"}


def compute_session_context_breakdown(agent: Any, messages: Optional[list[dict]] = None) -> dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.usage_anchor import anchored_context_tokens
    from agent.system_prompt import build_system_prompt_parts

    messages = messages or []
    parts = build_system_prompt_parts(agent)
    stable = parts.get("stable", "") or ""
    skills_index = _skills_block(stable)
    memory_block, user_block = _memory_blocks(agent)
    system_prompt_text = _join(
        _strip_blocks(stable, skills_index), _strip_blocks(parts.get("volatile", "") or "", memory_block, user_block)
    )
    builtin_tools, mcp_tools, subagent_tools = _split_tools(list(getattr(agent, "tools", None) or []))
    tokens_by_id = {
        "system_prompt": _chars_to_tokens(system_prompt_text),
        "tool_definitions": _json_tokens(builtin_tools),
        "rules": _chars_to_tokens(parts.get("context", "") or ""),
        "skills": _chars_to_tokens(skills_index),
        "mcp": _json_tokens(mcp_tools),
        "subagent_definitions": _json_tokens(subagent_tools),
        "memory": _chars_to_tokens(_join(memory_block, user_block)),
        "conversation": estimate_messages_tokens_rough(messages),
    }
    estimated_total = sum(tokens_by_id.values())

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    # Usage-anchored figure (provider-exact tokens of a response + delta of what was
    # appended since) beats last_prompt_tokens (lags) and the heuristic. Prefer the
    # turn-base anchor: on reasoning models later same-turn responses inflate
    # prompt_tokens with replayed thinking that evaporates at the turn boundary, so
    # anchoring on the LAST response makes the meter sawtooth. Fall back to the
    # last-response anchor, then measured, then estimated.
    anchor = getattr(agent, "_turn_base_usage_anchor", None)
    context_used = anchored_context_tokens(messages, anchor, charge_stale_thinking=False)
    if context_used is None:
        anchor = getattr(agent, "_usage_anchor", None)
        context_used = anchored_context_tokens(messages, anchor)
    if context_used is None:
        measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
        context_used = measured_used if measured_used > 0 else estimated_total
        source = context_display_source(comp) if measured_used > 0 else "local_estimate"
    else:
        delta = messages[int(anchor["base_count"]):]
        if delta and delta[0].get("role") == "assistant":
            delta = delta[1:]
        source = "provider_usage_plus_estimate" if delta else "provider_usage"
    # A single prompt can never exceed the model window; any excess is estimate drift.
    if context_max:
        context_used = min(context_used, context_max)

    return {
        "categories": [
            {"color": color, "id": category_id, "label": _category_label(category_id), "tokens": tokens_by_id[category_id]}
            for category_id, (color, _glyph_) in _CATEGORIES.items()
            if tokens_by_id[category_id] > 0 or category_id in _ALWAYS_REPORTED
        ],
        "context_max": context_max,
        "context_percent": max(0, min(100, round(context_used / context_max * 100))) if context_max else 0,
        "context_used": context_used,
        "context_source": source,
        "context_estimated": source != "provider_usage",
        "estimated_total": estimated_total,
        "model": getattr(agent, "model", "") or "",
    }


def compute_context_details(agent: Any) -> dict[str, Any]:
    """Expanded per-skill / per-toolset cost listing for ``/context all``.

    Reuses the ``hermes prompt-size`` attribution (index-line bytes from the
    live skills block; schema bytes via the registry's tool→toolset map).
    """
    from hermes_cli.prompt_size import _compute_skills_breakdown, _compute_toolsets_breakdown
    from agent.system_prompt import build_system_prompt_parts

    skills_block = _skills_block(build_system_prompt_parts(agent).get("stable", "") or "")
    tools = list(getattr(agent, "tools", None) or [])
    return {
        "skills": [
            {
                "name": entry.get("name", ""),
                "index_tokens": _bytes_to_tokens(entry.get("index_line_bytes")) or 0,
                "skill_md_tokens": _bytes_to_tokens(entry.get("skill_md_bytes")),
            }
            for entry in (_compute_skills_breakdown(skills_block) if skills_block else [])
        ],
        "toolsets": [
            {
                "toolset": group.get("toolset", ""),
                "tool_count": int(group.get("tool_count", 0) or 0),
                "schema_tokens": _bytes_to_tokens(group.get("json_bytes")) or 0,
            }
            for group in (_compute_toolsets_breakdown(tools) if tools else [])
        ],
    }


# ── /context rendering (CLI + gateway) ──────────────────────────────────────
# Pure text renderers over the payload above. The gateway skips the glyph grid
# (monospace is not guaranteed on messaging platforms).


def render_context_grid(payload: dict[str, Any]) -> list[str]:
    """Glyph grid: 100 cells, one per percent of the context window; categories
    fill in declaration order, the remainder is free space."""
    context_max = int(payload.get("context_max") or 0)
    total_cells = _GRID_COLUMNS * _GRID_ROWS
    cells: list[str] = []
    if context_max > 0:
        for cat in payload.get("categories") or []:
            tokens = int(cat.get("tokens") or 0)
            # never render a nonzero category as invisible
            n = round(tokens / context_max * total_cells) or (1 if tokens > 0 else 0)
            cells.extend([_glyph(cat)] * n)
        cells = cells[:total_cells]
    cells.extend([_FREE_GLYPH] * (total_cells - len(cells)))
    return [" ".join(cells[row * _GRID_COLUMNS:(row + 1) * _GRID_COLUMNS]) for row in range(_GRID_ROWS)]


def render_context_category_lines(payload: dict[str, Any]) -> list[str]:
    """Render the 'Estimated usage by category' table as plain-text lines."""
    categories = payload.get("categories") or []
    context_max = int(payload.get("context_max") or 0)
    estimated_total = int(payload.get("estimated_total") or 0)
    denom = context_max or estimated_total

    lines = [t("gateway.context.category_header")]
    if not categories:
        return [*lines, t("gateway.context.no_data_yet")]
    free_label = t("gateway.context.free_space")
    width = max(len(free_label), *(len(str(cat.get("label") or "")) for cat in categories))
    for cat in categories:
        tokens, label = int(cat.get("tokens") or 0), str(cat.get("label") or cat.get("id") or "")
        lines.append(t("gateway.context.category_row", glyph=_glyph(cat), label=f"{label:<{width}}",
                       tokens=f"{tokens:>9,}", pct=f"{tokens / denom * 100 if denom else 0.0:>5.1f}"))
    if context_max > 0:
        free = max(0, context_max - estimated_total)
        lines.append(t("gateway.context.category_row", glyph=_FREE_GLYPH, label=f"{free_label:<{width}}",
                       tokens=f"{free:>9,}", pct=f"{free / context_max * 100:>5.1f}"))
    return lines


def _toolset_row(group: dict[str, Any]) -> str:
    return t("gateway.context.toolset_row", toolset=f"{group['toolset']:<24}", count=f"{group['tool_count']:>3}",
             tokens=f"{group['schema_tokens']:>8,}")


def _skill_row(entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or "")
    if len(name) > 28:
        name = name[:27] + "…"
    md = entry.get("skill_md_tokens")
    md_str = f"~{md:>8,}" if md is not None else f"{t('gateway.context.not_available'):>8}"
    return t("gateway.context.skill_row", name=f"{name:<28}", index_tokens=f"{entry['index_tokens']:>6,}", md_tokens=md_str)


def _table(lines: list[str], title: str, rows: list[dict[str, Any]], fmt) -> None:
    """Append a titled, display-capped table (blank-separated from a preceding one)."""
    if not rows:
        return
    if lines:
        lines.append("")
    lines.append(title)
    lines.extend(fmt(row) for row in rows[:_DETAILS_TABLE_LIMIT])
    if len(rows) > _DETAILS_TABLE_LIMIT:
        lines.append(t("gateway.context.and_more", count=len(rows) - _DETAILS_TABLE_LIMIT))


def render_context_details_lines(details: dict[str, Any]) -> list[str]:
    """Render the expanded ``/context all`` per-skill / per-toolset tables."""
    lines: list[str] = []
    _table(lines, t("gateway.context.toolsets_title"), details.get("toolsets") or [], _toolset_row)
    _table(lines, t("gateway.context.skills_title"), details.get("skills") or [], _skill_row)
    return lines


def render_context_breakdown_lines(
    payload: dict[str, Any],
    *,
    details: Optional[dict[str, Any]] = None,
    grid: bool = True,
) -> list[str]:
    """Full /context view. ``grid`` prepends the glyph grid (CLI; the gateway
    keeps its own gauge); ``details`` appends the expanded listings."""
    lines: list[str] = [*render_context_grid(payload), ""] if grid else []
    lines.extend(render_context_category_lines(payload))

    context_max = int(payload.get("context_max") or 0)
    if context_max > 0:
        used, pct = int(payload.get("context_used") or 0), int(payload.get("context_percent") or 0)
        mark = "~" if payload.get("context_estimated") else ""
        lines.extend(["", t("gateway.context.window_line", mark=mark, used=f"{used:,}", max=f"{context_max:,}", pct=pct)])
        source = payload.get("context_source")
        if source:
            source_label = t(f"gateway.context.source.{source}") if source in _CONTEXT_SOURCES else source
            lines.append(t("gateway.context.source_line", source=source_label))

    if details is None:
        lines.extend(["", t("gateway.context.hint_all")])
    elif detail_lines := render_context_details_lines(details):
        lines.extend(["", *detail_lines])
    return lines
