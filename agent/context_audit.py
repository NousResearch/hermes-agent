"""Redacted startup context-budget audit helpers.

The audit reports labels, sizes, hashes, ranks, and recommendations only. It
must never retain raw prompt text, raw memory/profile content, or full tool
schemas inside report objects intended for display/logging.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class NecessityRank(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    SITUATIONAL = "situational"
    CANDIDATE_TO_TRIM = "candidate_to_trim"


@dataclass(frozen=True)
class ContextAuditEntry:
    label: str
    source_type: str
    tier: str = ""
    chars: int = 0
    bytes: int = 0
    estimated_tokens: int = 0
    content_hash: str = ""
    necessity: NecessityRank = NecessityRank.SITUATIONAL
    reason: str = ""
    tokenizer: str = "rough_chars_over_4"

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "source_type": self.source_type,
            "tier": self.tier,
            "chars": self.chars,
            "bytes": self.bytes,
            "estimated_tokens": self.estimated_tokens,
            "content_hash": self.content_hash,
            "necessity": self.necessity.value,
            "reason": self.reason,
            "tokenizer": self.tokenizer,
        }


@dataclass(frozen=True)
class SourceTypeTotals:
    source_type: str
    total_chars: int
    total_bytes: int
    estimated_tokens: int
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "total_chars": self.total_chars,
            "total_bytes": self.total_bytes,
            "estimated_tokens": self.estimated_tokens,
            "count": self.count,
        }


@dataclass(frozen=True)
class OptimizationOption:
    source_label: str
    title: str
    estimated_savings_chars: int
    risk: str
    action: str
    reason: str
    config_hint: str = ""
    command_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_label": self.source_label,
            "title": self.title,
            "estimated_savings_chars": self.estimated_savings_chars,
            "risk": self.risk,
            "action": self.action,
            "reason": self.reason,
            "config_hint": self.config_hint,
            "command_hint": self.command_hint,
        }


@dataclass(frozen=True)
class ContextAuditReport:
    entries: tuple[ContextAuditEntry, ...]
    optimization_options: tuple[OptimizationOption, ...] = field(default_factory=tuple)
    model: str = ""
    provider: str = ""
    tokenizer: str = "rough_chars_over_4"
    tool_count: int = 0

    @property
    def prompt_chars(self) -> int:
        return sum(entry.chars for entry in self.entries if entry.source_type == "prompt")

    @property
    def prompt_bytes(self) -> int:
        return sum(entry.bytes for entry in self.entries if entry.source_type == "prompt")

    @property
    def tool_schema_bytes(self) -> int:
        return sum(
            entry.bytes
            for entry in self.entries
            if entry.source_type == "tool_schema" and entry.label != "tool_schema.total"
        )

    @property
    def total_chars(self) -> int:
        return sum(entry.chars for entry in self.entries)

    @property
    def total_bytes(self) -> int:
        return self.prompt_bytes + self.tool_schema_bytes + sum(
            entry.bytes for entry in self.entries if entry.source_type not in {"prompt", "tool_schema"}
        )

    @property
    def estimated_tokens(self) -> int:
        return sum(entry.estimated_tokens for entry in self.entries if entry.label != "tool_schema.total")

    def entries_by_source_type(self) -> dict[str, SourceTypeTotals]:
        grouped: dict[str, list[ContextAuditEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.source_type, []).append(entry)
        return {
            source_type: SourceTypeTotals(
                source_type=source_type,
                total_chars=sum(e.chars for e in items if e.label != "tool_schema.total"),
                total_bytes=sum(e.bytes for e in items if e.label != "tool_schema.total"),
                estimated_tokens=sum(e.estimated_tokens for e in items if e.label != "tool_schema.total"),
                count=len(items),
            )
            for source_type, items in grouped.items()
        }

    def top_contributors(self, limit: int = 5) -> tuple[ContextAuditEntry, ...]:
        material = [entry for entry in self.entries if entry.label != "tool_schema.total"]
        return tuple(sorted(material, key=lambda entry: (entry.bytes, entry.chars), reverse=True)[:limit])

    def to_redacted_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "tokenizer": self.tokenizer,
            "prompt_chars": self.prompt_chars,
            "prompt_bytes": self.prompt_bytes,
            "tool_schema_bytes": self.tool_schema_bytes,
            "tool_count": self.tool_count,
            "total_bytes": self.total_bytes,
            "estimated_tokens": self.estimated_tokens,
            "entries": [entry.to_redacted_dict() for entry in self.entries],
            "optimization_options": [option.to_dict() for option in self.optimization_options],
        }


_CLASSIFICATION_RULES: tuple[tuple[tuple[str, ...], NecessityRank, str], ...] = (
    (("stable", "soul", "identity", "tool-use", "safety", "profile safety"), NecessityRank.CRITICAL, "Stable identity and safety posture are required for correct operation."),
    (("context", "project", "agents", "claude"), NecessityRank.HIGH, "Project context is high-value operating guidance for this workspace."),
    (("volatile", "memory", "user", "profile"), NecessityRank.HIGH, "Memory and user profile preserve preferences and session posture."),
    (("skills", "available-skills"), NecessityRank.SITUATIONAL, "Skill discovery is useful but broad indexes are often compressible."),
    (("tool_schema", "tool-schema", "tool schema"), NecessityRank.SITUATIONAL, "Tool schemas are needed only for enabled tools and can be scoped by task."),
    (("nous", "capability", "status"), NecessityRank.SITUATIONAL, "Managed capability inventory is useful but usually low-risk to compact."),
    (("duplicate", "personality overlay", "oversized"), NecessityRank.CANDIDATE_TO_TRIM, "Duplicated or oversized guidance is a safe trim candidate after review."),
)


def _estimate_tokens(text_or_bytes_len: int) -> int:
    if text_or_bytes_len <= 0:
        return 0
    return max(1, (text_or_bytes_len + 3) // 4)


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _classify(label: str, source_type: str, tier: str, size: int) -> tuple[NecessityRank, str]:
    haystack = f"{label} {source_type} {tier}".lower()
    for needles, rank, reason in _CLASSIFICATION_RULES:
        if any(needle in haystack for needle in needles):
            if source_type == "prompt" and tier == "stable":
                return NecessityRank.CRITICAL, reason
            return rank, reason
    if size >= 4096:
        return NecessityRank.CANDIDATE_TO_TRIM, "Large unclassified context source; classify it before letting it grow."
    return NecessityRank.SITUATIONAL, "Source is not classified yet; review if it becomes a top contributor."


def _entry_from_text(label: str, source_type: str, text: str, *, tier: str = "") -> ContextAuditEntry:
    safe_text = text or ""
    data = safe_text.encode("utf-8")
    rank, reason = _classify(label, source_type, tier, len(data))
    return ContextAuditEntry(
        label=label,
        source_type=source_type,
        tier=tier,
        chars=len(safe_text),
        bytes=len(data),
        estimated_tokens=_estimate_tokens(len(safe_text)),
        content_hash=_hash_bytes(data),
        necessity=rank,
        reason=reason,
    )


def _serialize_tool(tool: Mapping[str, Any]) -> bytes:
    return json.dumps(tool, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def _tool_name(tool: Mapping[str, Any], fallback: int) -> str:
    fn = tool.get("function") if isinstance(tool, Mapping) else None
    if isinstance(fn, Mapping) and fn.get("name"):
        return str(fn["name"])
    return f"tool_{fallback}"


def _optimization_options(entries: Sequence[ContextAuditEntry]) -> tuple[OptimizationOption, ...]:
    options: list[OptimizationOption] = []
    tool_entries = [entry for entry in entries if entry.source_type == "tool_schema" and entry.label != "tool_schema.total"]
    if tool_entries:
        total = sum(entry.bytes for entry in tool_entries)
        options.append(
            OptimizationOption(
                source_label="tool_schema.total",
                title="Scope enabled toolsets to the current task",
                estimated_savings_chars=max(total, 0),
                risk="medium",
                action="Review largest toolsets before disabling anything live.",
                reason="Tool definitions consume API context even outside the rendered system prompt.",
                command_hint="hermes tools list; use per-job/per-agent enabled_toolsets where possible",
            )
        )
    for entry in entries:
        if entry.source_type == "prompt" and entry.tier == "context" and entry.bytes:
            options.append(
                OptimizationOption(
                    source_label=entry.label,
                    title="Keep project context as a router",
                    estimated_savings_chars=max(entry.chars // 3, 0),
                    risk="low",
                    action="Move detail into skills/references; keep AGENTS.md short.",
                    reason="Project context is valuable but often grows into duplicated procedure text.",
                )
            )
        elif entry.necessity == NecessityRank.CANDIDATE_TO_TRIM:
            options.append(
                OptimizationOption(
                    source_label=entry.label,
                    title="Classify or trim oversized context source",
                    estimated_savings_chars=max(entry.chars // 2 or entry.bytes // 2, 1),
                    risk="low",
                    action="Classify the source and move non-routing detail behind an explicit command or skill.",
                    reason=entry.reason,
                    config_hint="agent.startup_context_audit: status",
                )
            )
        elif entry.source_type == "prompt" and entry.tier == "stable" and entry.bytes > 4096:
            options.append(
                OptimizationOption(
                    source_label=entry.label,
                    title="Review stable prompt bulk without disabling safety",
                    estimated_savings_chars=max(entry.chars // 5, 0),
                    risk="medium",
                    action="Condense duplicated operational wording; do not disable critical identity or safety guidance.",
                    reason="Stable prompt bytes are cache-friendly but still consume initial context budget.",
                )
            )
    return tuple(sorted(options, key=lambda opt: (-opt.estimated_savings_chars, opt.risk)))


def collect_context_audit(
    agent: Any,
    *,
    prompt_parts: Mapping[str, str],
    extra_sources: Iterable[tuple[str, str, str]] | None = None,
) -> ContextAuditReport:
    """Build a deterministic redacted report for prompt parts and tool schemas."""
    entries: list[ContextAuditEntry] = []
    for tier in ("stable", "context", "volatile"):
        entries.append(_entry_from_text(f"system_prompt.{tier}", "prompt", prompt_parts.get(tier, "") or "", tier=tier))

    tools = list(getattr(agent, "tools", None) or [])
    for index, tool in enumerate(tools):
        data = _serialize_tool(tool if isinstance(tool, Mapping) else {"tool": str(tool)})
        label = f"tool_schema.{_tool_name(tool, index) if isinstance(tool, Mapping) else f'tool_{index}'}"
        rank, reason = _classify(label, "tool_schema", "api_payload", len(data))
        entries.append(
            ContextAuditEntry(
                label=label,
                source_type="tool_schema",
                tier="api_payload",
                chars=0,
                bytes=len(data),
                estimated_tokens=_estimate_tokens(len(data)),
                content_hash=_hash_bytes(data),
                necessity=rank,
                reason=reason,
            )
        )
    if not tools:
        entries.append(
            ContextAuditEntry(
                label="tool_schema.total",
                source_type="tool_schema",
                tier="api_payload",
                chars=0,
                bytes=0,
                estimated_tokens=0,
                content_hash=_hash_bytes(b""),
                necessity=NecessityRank.SITUATIONAL,
                reason="No tool schemas loaded.",
            )
        )

    for label, source_type, text in extra_sources or ():
        entries.append(_entry_from_text(label, source_type, text))

    report_entries = tuple(entries)
    return ContextAuditReport(
        entries=report_entries,
        optimization_options=_optimization_options(report_entries),
        model=str(getattr(agent, "model", "") or ""),
        provider=str(getattr(agent, "provider", "") or ""),
        tool_count=len(tools),
    )


def with_prompt_summary_entry(report: ContextAuditReport, summary: str) -> ContextAuditReport:
    """Return a copy whose prompt totals include an injected audit summary."""
    if not summary:
        return report
    summary_entry = _entry_from_text("system_prompt.context_audit_summary", "prompt", summary, tier="volatile")
    entries = (*report.entries, summary_entry)
    return ContextAuditReport(
        entries=entries,
        optimization_options=_optimization_options(entries),
        model=report.model,
        provider=report.provider,
        tokenizer=report.tokenizer,
        tool_count=report.tool_count,
    )


def render_context_audit_summary(report: ContextAuditReport, *, max_lines: int = 10) -> str:
    """Render a compact redacted summary suitable for startup/status surfaces."""
    lines = [
        "Context audit",
        f"Prompt: {report.prompt_chars:,} chars (~{sum(e.estimated_tokens for e in report.entries if e.source_type == 'prompt'):,} tokens est)",
        f"API tool schemas: {report.tool_schema_bytes:,} bytes across {report.tool_count} tools",
        "Top contributors:",
    ]
    for entry in report.top_contributors(limit=max(1, max_lines - 6)):
        size = f"{entry.bytes:,} bytes" if entry.source_type == "tool_schema" else f"{entry.chars:,} chars"
        lines.append(f"- {entry.label}: {size}, {entry.necessity.value}")
    if report.optimization_options and len(lines) < max_lines:
        option = report.optimization_options[0]
        lines.append(f"Top option: {option.title} (~{option.estimated_savings_chars:,} chars/bytes, {option.risk} risk)")
    return "\n".join(lines[:max_lines])


def render_context_audit_report(report: ContextAuditReport, *, max_options: int = 5) -> str:
    """Render a detailed but still redacted report for slash commands."""
    lines = [render_context_audit_summary(report, max_lines=8), "", "Optimization options:"]
    for option in report.optimization_options[:max_options]:
        hint = option.command_hint or option.config_hint
        suffix = f" ({hint})" if hint else ""
        lines.append(f"- {option.title}: {option.reason} Risk: {option.risk}.{suffix}")
    return "\n".join(lines).strip()
