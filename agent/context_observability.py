"""Read-only context composition observability utilities.

This module intentionally reports sizes and labels, not raw prompt content.
It is safe to use from CLI/Gateway/WebUI surfaces without dumping secrets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ContextSection:
    """One high-level bucket in a provider request."""

    name: str
    estimated_tokens: int
    item_count: int = 0
    cacheability: str = "unknown"
    notes: list[str] = field(default_factory=list)

    @property
    def share(self) -> float:
        """Placeholder kept for callers that prefer section-local access.

        The true share depends on the full report total; use
        ``ContextBreakdown.section_shares`` for exact percentages.
        """
        return 0.0


@dataclass(frozen=True)
class CacheSummary:
    """Provider cache eligibility/configuration summary."""

    supported: bool = False
    configured: bool = False
    stable_estimated_tokens: int = 0
    volatile_estimated_tokens: int = 0
    status: str = "unknown"


@dataclass(frozen=True)
class ContextBreakdown:
    """Safe, structured summary of the active request context."""

    model: str = "unknown"
    provider: str = "unknown"
    context_window: int = 0
    total_estimated_tokens: int = 0
    sections: list[ContextSection] = field(default_factory=list)
    cache: CacheSummary = field(default_factory=CacheSummary)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    estimator: str = "rough_chars_div_4"
    message_count: int = 0
    tool_count: int = 0
    enabled_toolsets: list[str] = field(default_factory=list)

    def section_shares(self) -> dict[str, float]:
        if not self.total_estimated_tokens:
            return {section.name: 0.0 for section in self.sections}
        return {
            section.name: section.estimated_tokens / self.total_estimated_tokens
            for section in self.sections
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rough_tokens(value: Any) -> int:
    """Estimate tokens without provider-specific tokenizer dependencies."""
    if value is None or value == "":
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            text = str(value)
    return (len(text) + 3) // 4


def _message_role(message: Mapping[str, Any]) -> str:
    return str(message.get("role") or "unknown")


def _split_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    system_messages: list[Mapping[str, Any]] = []
    tool_results: list[Mapping[str, Any]] = []
    conversation_messages: list[Mapping[str, Any]] = []
    for message in messages:
        role = _message_role(message)
        if role == "system":
            system_messages.append(message)
        elif role == "tool":
            tool_results.append(message)
        else:
            conversation_messages.append(message)
    return system_messages, conversation_messages, tool_results


def _provider_cache_summary(
    provider: str,
    *,
    sections: Sequence[ContextSection],
    openrouter_cache_enabled: bool | None,
) -> CacheSummary:
    provider_key = (provider or "").lower()
    supported = provider_key in {"openrouter", "openrouter.ai"}
    configured = bool(openrouter_cache_enabled) if supported else False
    stable = sum(section.estimated_tokens for section in sections if section.cacheability == "stable")
    volatile = sum(section.estimated_tokens for section in sections if section.cacheability == "volatile")
    if supported and configured:
        status = "configured"
    elif supported:
        status = "supported_not_configured"
    else:
        status = "unknown"
    return CacheSummary(
        supported=supported,
        configured=configured,
        stable_estimated_tokens=stable,
        volatile_estimated_tokens=volatile,
        status=status,
    )


def _nonzero_section(name: str, value: Any, *, item_count: int, cacheability: str, notes: Iterable[str] = ()) -> ContextSection | None:
    tokens = _rough_tokens(value)
    if tokens <= 0 and item_count <= 0:
        return None
    return ContextSection(
        name=name,
        estimated_tokens=tokens,
        item_count=item_count,
        cacheability=cacheability,
        notes=list(notes),
    )


def build_context_breakdown(
    messages: Sequence[Mapping[str, Any]] | None,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    system_prompt: str = "",
    model: str = "unknown",
    provider: str = "unknown",
    context_window: int = 0,
    enabled_toolsets: Sequence[str] | None = None,
    openrouter_cache_enabled: bool | None = None,
) -> ContextBreakdown:
    """Build a safe breakdown of request context buckets.

    The function deliberately avoids returning prompt text, message content,
    tool descriptions, environment values, or any other raw payload data.
    """
    messages = list(messages or [])
    tools = list(tools or [])
    system_messages, conversation_messages, tool_results = _split_messages(messages)

    sections: list[ContextSection] = []
    for section in (
        _nonzero_section("system_prompt", system_prompt, item_count=1 if system_prompt else 0, cacheability="stable"),
        _nonzero_section("system_messages", system_messages, item_count=len(system_messages), cacheability="stable"),
        _nonzero_section("conversation_messages", conversation_messages, item_count=len(conversation_messages), cacheability="volatile"),
        _nonzero_section("tool_results", tool_results, item_count=len(tool_results), cacheability="volatile"),
        _nonzero_section("tool_schemas", tools, item_count=len(tools), cacheability="stable"),
    ):
        if section is not None:
            sections.append(section)

    total = sum(section.estimated_tokens for section in sections)
    cache = _provider_cache_summary(
        provider,
        sections=sections,
        openrouter_cache_enabled=openrouter_cache_enabled,
    )

    shares = {section.name: (section.estimated_tokens / total if total else 0.0) for section in sections}
    warnings: list[str] = []
    suggestions: list[str] = []
    tool_share = shares.get("tool_schemas", 0.0)
    if tool_share >= 0.30 and tools:
        warnings.append("Tool schemas dominate the estimated request context.")
        suggestions.append("Use narrower toolsets or route heavy MCP/native tools only when needed.")
    if context_window and total >= int(context_window * 0.50):
        warnings.append("Estimated request is over half the model context window.")
        suggestions.append("Compress, branch, or move raw data to files and reference paths/manifests only.")
    if cache.supported and not cache.configured:
        suggestions.append("OpenRouter cache appears supported but not configured for this request path.")

    return ContextBreakdown(
        model=model or "unknown",
        provider=provider or "unknown",
        context_window=int(context_window or 0),
        total_estimated_tokens=total,
        sections=sections,
        cache=cache,
        warnings=warnings,
        suggestions=suggestions,
        message_count=len(messages),
        tool_count=len(tools),
        enabled_toolsets=[str(toolset) for toolset in (enabled_toolsets or [])],
    )


def format_context_breakdown(report: ContextBreakdown) -> str:
    """Format a breakdown as a compact, content-free text report."""
    shares = report.section_shares()
    lines = [
        "  🧮 Context Composition",
        f"  {'─' * 40}",
        f"  Model:                  {report.model}",
        f"  Provider:               {report.provider}",
    ]
    for section in report.sections:
        share = shares.get(section.name, 0.0) * 100.0
        label = "item" if section.item_count == 1 else "items"
        lines.append(
            f"  {section.name + ':':<24} {section.estimated_tokens:>10,} tokens "
            f"({share:>4.1f}%, {section.item_count} {label}, {section.cacheability})"
        )
    lines.append(f"  Estimated request:      {report.total_estimated_tokens:>10,} tokens")
    if report.context_window:
        pct = min(100.0, report.total_estimated_tokens / report.context_window * 100.0)
        lines.append(f"  Context window:         {report.context_window:>10,} tokens ({pct:.1f}% estimated)")
    else:
        lines.append(f"  Context window:         {'unknown':>10}")
    lines.append(
        "  Cache:                  "
        f"{report.cache.status} "
        f"(stable ~{report.cache.stable_estimated_tokens:,}, volatile ~{report.cache.volatile_estimated_tokens:,})"
    )
    if report.enabled_toolsets:
        lines.append(f"  Enabled toolsets:       {', '.join(report.enabled_toolsets)}")
    lines.append(f"  Loaded tools:           {report.tool_count:>10,}")
    if report.warnings:
        lines.append("  Warnings:")
        lines.extend(f"    - {warning}" for warning in report.warnings)
    if report.suggestions:
        lines.append("  Suggestions:")
        lines.extend(f"    - {suggestion}" for suggestion in report.suggestions)
    lines.append("  Note: rough estimate; provider tokenizers and cache accounting differ.")
    return "\n".join(lines)
