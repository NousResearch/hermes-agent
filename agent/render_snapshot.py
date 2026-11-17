"""What the agent saw: a session's stored system prompt, model route and tools[], per compression segment.

Read-only over what the session already persisted: the prompt comes from the row's content-addressed
``system_prompts`` hash (never a rebuild, so the cached prefix is untouched) and the tools from the row's
frozen tools[] pin. A live agent contributes only attributes it already holds (``tools``, ``model``,
``provider``). A rotating compression (``compression.in_place: false``) or a fork leaves one row per
segment, and the lineage is diffed segment by segment into change records (prompt changed, tool
added/removed, model or provider changed). An in-place compaction (the default) rewrites the same row,
so such a session has one segment and no change records: nothing durable holds its earlier snapshot. Prompt text and tool
descriptions go through the same force-mode redaction ``hermes sessions export --redact`` uses.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Any, Optional

from agent.redact import redact_sensitive_text
from hermes_state_sessions import _parse_model_config
from tools.registry import registry

# A long-lived conversation can rotate many times; the inspector diffs the newest segments.
MAX_SEGMENTS = 20


def _function(tool: Any) -> dict:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return fn if isinstance(fn, dict) else tool if isinstance(tool, dict) else {}


def _tool_name(tool: Any) -> str:
    return tool if isinstance(tool, str) else str(_function(tool).get("name") or "")


def _pinned_tools(raw: Optional[str]) -> list:
    """The row's tools[] pin (``{"version", "tools"}``) or a legacy name list; [] when absent."""
    try:
        pin = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return []  # a pin hash whose row a cleanup swept resolves to itself
    tools = pin.get("tools") if isinstance(pin, dict) else pin
    return tools if isinstance(tools, list) else []


def _mcp_server(toolset: str) -> Optional[str]:
    """The server name from its registered ``mcp-<server>`` toolset. A tool name cannot be reversed into
    one (registration sanitizes the server name), so an unregistered MCP tool names no server."""
    return toolset[len("mcp-"):] if toolset.startswith("mcp-") else None


def _tool_rows(tools: list) -> list[dict]:
    toolsets = registry.get_tool_to_toolset_map()
    rows = []
    for tool in tools:
        if not (name := _tool_name(tool)):
            continue
        toolset = toolsets.get(name) or ""
        description = str(_function(tool).get("description") or "").strip()
        rows.append({"name": name, "toolset": toolset, "mcp_server": _mcp_server(toolset),
                     "description": redact_sensitive_text(description, force=True)})
    return rows


def _route(row: dict) -> tuple[str, str]:
    config = _parse_model_config(row.get("model_config"))
    return str(row.get("model") or ""), str(config.get("provider") or row.get("billing_provider") or "")


def _segment(row: dict) -> dict:
    model, provider = _route(row)
    prompt = row.get("system_prompt")
    return {"session_id": row["id"], "started_at": row.get("started_at"), "end_reason": row.get("end_reason"),
            "model": model, "provider": provider, "prompt_hash": row.get("system_prompt_hash"),
            "prompt_chars": len(prompt) if isinstance(prompt, str) else 0,
            "tool_names": [n for n in map(_tool_name, _pinned_tools(row.get("tool_names"))) if n]}


def _changes(before: dict, after: dict) -> list[dict]:
    """What differs between two consecutive segments, in a fixed reading order."""
    changes: list[dict] = []
    for field, kind in (("model", "model_changed"), ("provider", "provider_changed")):
        if before[field] != after[field]:
            changes.append({"kind": kind, "before": before[field], "after": after[field]})
    if before["prompt_hash"] != after["prompt_hash"]:
        changes.append({"kind": "prompt_changed", "before": before["prompt_hash"], "after": after["prompt_hash"]})
    old, new = set(before["tool_names"]), set(after["tool_names"])
    changes += [{"kind": "tool_added", "name": name} for name in sorted(new - old)]
    changes += [{"kind": "tool_removed", "name": name} for name in sorted(old - new)]
    return changes


def build_render_snapshot(db, session_id: str, *, agent: Any = None) -> dict:
    """The inspector payload for stored session *session_id* (its compression tip or any segment).
    *agent*, when the session has one resident, supplies the tip's tools and route as it holds them."""
    tip_id = db.get_compression_tip(session_id) or session_id
    lineage = (db.get_compression_lineage(tip_id) or [tip_id])[-MAX_SEGMENTS:]
    rows = [row for row in map(db.get_session, lineage) if row]
    segments = [_segment(row) for row in rows]
    for previous, segment in pairwise(segments):
        segment["changes"] = _changes(previous, segment)
    if segments:
        segments[0]["changes"] = []
    tip = rows[-1] if rows else {}
    prompt = tip.get("system_prompt")
    live_tools = list(getattr(agent, "tools", None) or []) if agent is not None else []
    tools = _tool_rows(live_tools or _pinned_tools(tip.get("tool_names")))
    model, provider = _route(tip)
    if agent is not None:
        model = str(getattr(agent, "model", "") or model)
        provider = str(getattr(agent, "provider", "") or provider)
    return {
        "session_id": tip.get("id") or tip_id,
        "system_prompt": redact_sensitive_text(prompt, force=True) if isinstance(prompt, str) else None,
        "model": model, "provider": provider,
        "tools_source": "live" if live_tools else "stored" if tools else "none",
        "tools": tools,
        "toolsets": sorted({tool["toolset"] for tool in tools if tool["toolset"]}),
        "mcp_servers": sorted({tool["mcp_server"] for tool in tools if tool["mcp_server"]}),
        "segments": [{key: value for key, value in segment.items() if key != "tool_names"}
                     | {"tool_count": len(segment["tool_names"])} for segment in segments],
    }
