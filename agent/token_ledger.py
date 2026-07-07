"""Operational token-budget attribution for Hermes context assembly.

This module is intentionally pure/report-only for the first rollout: it mirrors
``agent.context_breakdown`` categories but returns typed segments that future
compression policy can consume without scraping UI payloads.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

_SKILLS_BLOCK_RE = re.compile(r"<available_skills>.*?</available_skills>", re.DOTALL)
_SUBAGENT_TOOL_NAMES = frozenset({"delegate_task"})


@dataclass(frozen=True)
class TokenSegment:
    """One attributable chunk of the next model request."""

    source: str
    label: str
    token_count: int
    stable_prefix: bool
    prunable: bool
    priority: int
    content_hash: str
    ref_id: str | None = None


@dataclass(frozen=True)
class TokenLedger:
    """Token attribution report for one assembled request."""

    segments: tuple[TokenSegment, ...]
    context_max: int = 0
    measured_used: int = 0
    model: str = ""
    by_source: dict[str, int] = field(init=False)
    estimated_total: int = field(init=False)
    context_used: int = field(init=False)
    context_percent: int = field(init=False)

    def __post_init__(self) -> None:
        by_source: dict[str, int] = defaultdict(int)
        for segment in self.segments:
            by_source[segment.source] += int(segment.token_count)
        estimated_total = sum(by_source.values())
        context_used = self.measured_used if self.measured_used > 0 else estimated_total
        context_percent = (
            max(0, min(100, round(context_used / self.context_max * 100)))
            if self.context_max
            else 0
        )
        object.__setattr__(self, "by_source", dict(by_source))
        object.__setattr__(self, "estimated_total", estimated_total)
        object.__setattr__(self, "context_used", context_used)
        object.__setattr__(self, "context_percent", context_percent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.__dict__.copy() for segment in self.segments],
            "by_source": dict(self.by_source),
            "estimated_total": self.estimated_total,
            "context_max": self.context_max,
            "context_used": self.context_used,
            "context_percent": self.context_percent,
            "model": self.model,
        }


@dataclass(frozen=True)
class BudgetViolation:
    source: str
    tokens: int
    cap: int
    excess: int


def _chars_to_tokens(text: str) -> int:
    if not text:
        return 0
    return (len(text) + 3) // 4


def _json_tokens(value: Any) -> int:
    if not value:
        return 0
    return _chars_to_tokens(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _stable_hash(source: str, rendered: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"\0")
    h.update(rendered.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(fn, dict):
        return str(fn.get("name") or "")
    return str(tool.get("name") or "") if isinstance(tool, dict) else ""


def _split_tools(tools: Sequence[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    builtin: list[dict] = []
    mcp: list[dict] = []
    subagent: list[dict] = []
    for tool in tools:
        name = _tool_name(tool)
        if name.startswith("mcp_"):
            mcp.append(tool)
        elif name in _SUBAGENT_TOOL_NAMES:
            subagent.append(tool)
        else:
            builtin.append(tool)
    return builtin, mcp, subagent


def _memory_blocks(agent: Any) -> tuple[str, str]:
    memory_block = ""
    user_block = ""
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return memory_block, user_block
    try:
        if getattr(agent, "_memory_enabled", True):
            memory_block = store.format_for_system_prompt("memory") or ""
        if getattr(agent, "_user_profile_enabled", True):
            user_block = store.format_for_system_prompt("user") or ""
    except Exception:
        return "", ""
    return str(memory_block or ""), str(user_block or "")


def _strip_blocks(text: str, *blocks: str) -> str:
    out = text
    for block in blocks:
        if block:
            out = out.replace(block, "")
    return out.strip()


def _segment(
    *,
    source: str,
    label: str,
    rendered: str,
    tokens: int | None = None,
    stable_prefix: bool,
    prunable: bool,
    priority: int,
    ref_id: str | None = None,
) -> TokenSegment | None:
    token_count = _chars_to_tokens(rendered) if tokens is None else int(tokens)
    if token_count <= 0:
        return None
    return TokenSegment(
        source=source,
        label=label,
        token_count=token_count,
        stable_prefix=stable_prefix,
        prunable=prunable,
        priority=priority,
        content_hash=_stable_hash(source, rendered),
        ref_id=ref_id,
    )


def compute_token_ledger(agent: Any, messages: Optional[list[dict]] = None) -> TokenLedger:
    """Build a report-only token ledger for the next model request.

    The token estimate intentionally matches Hermes' existing rough accounting
    (char/4 plus message estimator) so thresholds and UI numbers remain aligned.
    """
    from agent.model_metadata import estimate_messages_tokens_rough
    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    stable = str(parts.get("stable", "") or "")
    context = str(parts.get("context", "") or "")
    volatile = str(parts.get("volatile", "") or "")

    skills_match = _SKILLS_BLOCK_RE.search(stable)
    skills_index = skills_match.group(0) if skills_match else ""
    memory_block, user_block = _memory_blocks(agent)
    memory_text = "\n\n".join(part for part in (memory_block, user_block) if part).strip()

    system_core = _strip_blocks(stable, skills_index)
    system_tail = _strip_blocks(volatile, memory_block, user_block)
    system_prompt_text = "\n\n".join(part for part in (system_core, system_tail) if part).strip()

    tools = list(getattr(agent, "tools", None) or [])
    builtin_tools, mcp_tools, subagent_tools = _split_tools(tools)
    conversation_tokens = estimate_messages_tokens_rough(messages or [])
    conversation_rendered = json.dumps(messages or [], ensure_ascii=False, sort_keys=True)

    segment_specs = [
        _segment(
            source="system_prompt",
            label="System prompt",
            rendered=system_prompt_text,
            stable_prefix=True,
            prunable=False,
            priority=100,
        ),
        _segment(
            source="rules",
            label="Rules",
            rendered=context,
            stable_prefix=True,
            prunable=False,
            priority=95,
        ),
        _segment(
            source="skills",
            label="Skills",
            rendered=skills_index,
            stable_prefix=True,
            prunable=False,
            priority=80,
        ),
        _segment(
            source="memory",
            label="Memory",
            rendered=memory_text,
            stable_prefix=True,
            prunable=False,
            priority=90,
        ),
        _segment(
            source="tool_definitions",
            label="Tool definitions",
            rendered=json.dumps(builtin_tools, ensure_ascii=False, sort_keys=True),
            tokens=_json_tokens(builtin_tools),
            stable_prefix=True,
            prunable=False,
            priority=70,
        ),
        _segment(
            source="mcp",
            label="MCP",
            rendered=json.dumps(mcp_tools, ensure_ascii=False, sort_keys=True),
            tokens=_json_tokens(mcp_tools),
            stable_prefix=True,
            prunable=False,
            priority=65,
        ),
        _segment(
            source="subagent_definitions",
            label="Subagent definitions",
            rendered=json.dumps(subagent_tools, ensure_ascii=False, sort_keys=True),
            tokens=_json_tokens(subagent_tools),
            stable_prefix=True,
            prunable=False,
            priority=60,
        ),
        _segment(
            source="conversation",
            label="Conversation",
            rendered=conversation_rendered,
            tokens=conversation_tokens,
            stable_prefix=False,
            prunable=True,
            priority=40,
        ),
    ]
    segments = tuple(segment for segment in segment_specs if segment is not None)

    comp = getattr(agent, "context_compressor", None)
    context_max = int(getattr(comp, "context_length", 0) or 0) if comp else 0
    measured_used = int(getattr(comp, "last_prompt_tokens", 0) or 0) if comp else 0
    return TokenLedger(
        segments=segments,
        context_max=context_max,
        measured_used=measured_used,
        model=getattr(agent, "model", "") or "",
    )


def find_budget_violations(
    ledger: TokenLedger,
    caps: Mapping[str, int],
) -> list[BudgetViolation]:
    """Return source caps exceeded by a ledger, preserving ledger source order."""
    seen: set[str] = set()
    violations: list[BudgetViolation] = []
    for segment in ledger.segments:
        source = segment.source
        if source in seen or source not in caps:
            continue
        seen.add(source)
        cap = int(caps[source])
        tokens = int(ledger.by_source.get(source, 0))
        if tokens > cap:
            violations.append(
                BudgetViolation(
                    source=source,
                    tokens=tokens,
                    cap=cap,
                    excess=tokens - cap,
                )
            )
    return violations
