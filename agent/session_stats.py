from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent.insights import _get_pricing, _has_known_pricing
from agent.model_metadata import resolve_model_capabilities


@dataclass(frozen=True)
class SessionStatsSnapshot:
    model: Optional[str] = None
    model_display_name: Optional[str] = None
    provider: Optional[str] = None
    api_mode: Optional[str] = None
    base_url: Optional[str] = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0

    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_hit_rate: Optional[float] = None

    context_current_tokens: Optional[int] = None
    context_completion_tokens: Optional[int] = None
    context_total_tokens: Optional[int] = None
    context_max_tokens: Optional[int] = None
    context_threshold_tokens: Optional[int] = None
    context_source: Optional[str] = None
    max_completion_tokens: Optional[int] = None

    message_count: int = 0
    user_message_count: int = 0
    assistant_message_count: int = 0
    system_message_count: int = 0
    tool_message_count: int = 0

    tool_calls_total: int = 0
    tool_calls_by_name: Dict[str, int] = field(default_factory=dict)
    top_tools: List[Tuple[str, int]] = field(default_factory=list)

    compression_count: int = 0
    summarization_count: int = 0
    estimated_tokens_saved: int = 0
    summary_model: Optional[str] = None

    pricing_known: bool = False
    input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    cache_read_cost_per_million: Optional[float] = None
    cache_write_cost_per_million: Optional[float] = None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _resolve_model_info(model: Optional[str], base_url: str = "", model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if isinstance(model_info, dict):
        return dict(model_info)
    if not model:
        return {}
    try:
        return dict(resolve_model_capabilities(model, base_url=base_url))
    except Exception:
        return {}


def collect_session_stats(
    *,
    agent: Any,
    compressor: Any = None,
    session_db: Any = None,
    session_id: Optional[str] = None,
    model_info: Optional[Dict[str, Any]] = None,
    top_n_tools: int = 5,
) -> SessionStatsSnapshot:
    model = _safe_getattr(agent, "model")
    provider = _safe_getattr(agent, "provider")
    api_mode = _safe_getattr(agent, "api_mode")
    base_url = _safe_getattr(agent, "base_url")

    prompt_tokens = int(_safe_getattr(agent, "session_prompt_tokens", 0) or 0)
    completion_tokens = int(_safe_getattr(agent, "session_completion_tokens", 0) or 0)
    total_tokens = int(_safe_getattr(agent, "session_total_tokens", 0) or 0)
    api_calls = int(_safe_getattr(agent, "session_api_calls", 0) or 0)
    cache_read_tokens = int(_safe_getattr(agent, "session_cache_read_tokens", 0) or 0)
    cache_write_tokens = int(_safe_getattr(agent, "session_cache_write_tokens", 0) or 0)
    cache_hit_rate = (cache_read_tokens / prompt_tokens) if prompt_tokens > 0 else None

    context_current_tokens = None
    context_completion_tokens = None
    context_total_tokens = None
    context_max_tokens = None
    context_threshold_tokens = None
    context_source = None
    compression_count = 0
    summarization_count = 0
    estimated_tokens_saved = 0
    summary_model = None
    if compressor is not None:
        context_current_tokens = _safe_getattr(compressor, "last_prompt_tokens")
        context_completion_tokens = _safe_getattr(compressor, "last_completion_tokens")
        context_total_tokens = _safe_getattr(compressor, "last_total_tokens")
        context_max_tokens = _safe_getattr(compressor, "context_length")
        context_threshold_tokens = _safe_getattr(compressor, "threshold_tokens")
        context_source = _safe_getattr(compressor, "context_source")
        compression_count = int(_safe_getattr(compressor, "compression_count", 0) or 0)
        summarization_count = int(_safe_getattr(compressor, "summarization_count", compression_count) or 0)
        estimated_tokens_saved = int(_safe_getattr(compressor, "estimated_tokens_saved", 0) or 0)
        summary_model = _safe_getattr(compressor, "summary_model")

    message_count = 0
    user_message_count = 0
    assistant_message_count = 0
    system_message_count = 0
    tool_message_count = 0
    tool_calls_total = 0
    tool_calls_by_name: Dict[str, int] = {}
    if session_db is not None and session_id:
        try:
            messages = session_db.get_messages(session_id) or []
            message_count = len(messages)
            tool_counts: Dict[str, int] = {}
            pending_tool_results = 0
            for msg in messages:
                role = msg.get("role")
                if role == "user":
                    user_message_count += 1
                elif role == "assistant":
                    assistant_message_count += 1
                    msg_tool_calls = msg.get("tool_calls")
                    if isinstance(msg_tool_calls, list) and msg_tool_calls:
                        for tc in msg_tool_calls:
                            name = None
                            if isinstance(tc, dict):
                                name = tc.get("tool_name") or tc.get("name") or ((tc.get("function") or {}).get("name") if isinstance(tc.get("function"), dict) else None)
                            if name:
                                tool_counts[name] = tool_counts.get(name, 0) + 1
                                tool_calls_total += 1
                elif role == "system":
                    system_message_count += 1
                elif role == "tool":
                    tool_message_count += 1
                    pending_tool_results += 1
                    tool_name = msg.get("tool_name") or "(unknown)"
                    if tool_name not in tool_counts:
                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            if tool_calls_total == 0 and pending_tool_results:
                tool_calls_total = sum(tool_counts.values())
            tool_calls_by_name = dict(sorted(tool_counts.items(), key=lambda item: (-item[1], item[0])))
        except Exception:
            pass

    top_tools = list(tool_calls_by_name.items())[:top_n_tools]

    resolved_model_info = _resolve_model_info(model, base_url=base_url or "", model_info=model_info)
    if context_max_tokens is None:
        context_max_tokens = resolved_model_info.get("context_length")
    max_completion_tokens = resolved_model_info.get("max_completion_tokens")
    model_display_name = resolved_model_info.get("name")
    pricing_blob = resolved_model_info.get("pricing") if isinstance(resolved_model_info.get("pricing"), dict) else {}

    pricing_known = _has_known_pricing(model) if model else False
    inferred_pricing = _get_pricing(model) if model else {"input": 0.0, "output": 0.0}
    input_cost_per_million = _coerce_float(pricing_blob.get("prompt"))
    output_cost_per_million = _coerce_float(pricing_blob.get("completion"))
    cache_read_cost_per_million = _coerce_float(pricing_blob.get("cache_read"))
    cache_write_cost_per_million = _coerce_float(pricing_blob.get("cache_write"))

    if input_cost_per_million is None:
        input_cost_per_million = inferred_pricing.get("input")
    if output_cost_per_million is None:
        output_cost_per_million = inferred_pricing.get("output")

    return SessionStatsSnapshot(
        model=model,
        model_display_name=model_display_name,
        provider=provider,
        api_mode=api_mode,
        base_url=base_url,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        api_calls=api_calls,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cache_hit_rate=cache_hit_rate,
        context_current_tokens=context_current_tokens,
        context_completion_tokens=context_completion_tokens,
        context_total_tokens=context_total_tokens,
        context_max_tokens=context_max_tokens,
        context_threshold_tokens=context_threshold_tokens,
        context_source=context_source,
        max_completion_tokens=max_completion_tokens,
        message_count=message_count,
        user_message_count=user_message_count,
        assistant_message_count=assistant_message_count,
        system_message_count=system_message_count,
        tool_message_count=tool_message_count,
        tool_calls_total=tool_calls_total,
        tool_calls_by_name=tool_calls_by_name,
        top_tools=top_tools,
        compression_count=compression_count,
        summarization_count=summarization_count,
        estimated_tokens_saved=estimated_tokens_saved,
        summary_model=summary_model,
        pricing_known=pricing_known,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
        cache_read_cost_per_million=cache_read_cost_per_million,
        cache_write_cost_per_million=cache_write_cost_per_million,
    )


def render_stats_terminal(stats: SessionStatsSnapshot) -> str:
    def fmt_int(value: Optional[int]) -> str:
        return "n/a" if value is None else f"{int(value):,}"

    def fmt_pct(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value * 100:.1f}%"

    def fmt_money(value: Optional[float]) -> str:
        return "n/a" if value is None else f"${value:.3f} / 1M"

    lines: List[str] = []
    lines.append("Session Diagnostics")
    lines.append("=" * 19)
    lines.append("")

    lines.append("Connection")
    lines.append(f"  Model:     {stats.model or 'n/a'}")
    if stats.model_display_name:
        lines.append(f"  Name:      {stats.model_display_name}")
    lines.append(f"  Provider:  {stats.provider or 'n/a'}")
    lines.append(f"  API Mode:  {stats.api_mode or 'n/a'}")
    lines.append(f"  Base URL:  {stats.base_url or 'n/a'}")
    lines.append("")

    lines.append("Token Usage")
    lines.append(f"  Prompt:      {fmt_int(stats.prompt_tokens)}")
    lines.append(f"  Completion:  {fmt_int(stats.completion_tokens)}")
    lines.append(f"  Total:       {fmt_int(stats.total_tokens)}")
    lines.append(f"  API Calls:   {fmt_int(stats.api_calls)}")
    lines.append("")

    lines.append("Prompt Cache")
    lines.append(f"  Cache Read:   {fmt_int(stats.cache_read_tokens)}")
    lines.append(f"  Cache Write:  {fmt_int(stats.cache_write_tokens)}")
    lines.append(f"  Hit Rate:     {fmt_pct(stats.cache_hit_rate)}")
    lines.append("")

    lines.append("Context Window")
    lines.append(f"  Current:     {fmt_int(stats.context_current_tokens)}")
    lines.append(f"  Completion:  {fmt_int(stats.context_completion_tokens)}")
    lines.append(f"  Last Total:  {fmt_int(stats.context_total_tokens)}")
    lines.append(f"  Max:         {fmt_int(stats.context_max_tokens)}")
    lines.append(f"  Threshold:   {fmt_int(stats.context_threshold_tokens)}")
    lines.append(f"  Source:      {stats.context_source or 'n/a'}")
    lines.append(f"  Max Compl.:  {fmt_int(stats.max_completion_tokens)}")
    lines.append("")

    lines.append("Messages")
    lines.append(
        f"  Total: {fmt_int(stats.message_count)}  "
        f"(user: {fmt_int(stats.user_message_count)}, assistant: {fmt_int(stats.assistant_message_count)}, "
        f"system: {fmt_int(stats.system_message_count)}, tool: {fmt_int(stats.tool_message_count)})"
    )
    lines.append("")

    lines.append("Tool Usage")
    lines.append(f"  Total Calls: {fmt_int(stats.tool_calls_total)}")
    if stats.top_tools:
        for tool_name, count in stats.top_tools:
            lines.append(f"    {tool_name:<16} {count:,}")
    else:
        lines.append("    n/a")
    lines.append("")

    lines.append("Compression")
    lines.append(f"  Compressions:   {fmt_int(stats.compression_count)}")
    lines.append(f"  Summaries:      {fmt_int(stats.summarization_count)}")
    lines.append(f"  Tokens Saved:   {fmt_int(stats.estimated_tokens_saved)}")
    lines.append(f"  Summary Model:  {stats.summary_model or 'n/a'}")
    lines.append("")

    lines.append("Model Pricing")
    lines.append(f"  Known:        {'yes' if stats.pricing_known else 'no'}")
    lines.append(f"  Input:        {fmt_money(stats.input_cost_per_million)}")
    lines.append(f"  Output:       {fmt_money(stats.output_cost_per_million)}")
    lines.append(f"  Cache Read:   {fmt_money(stats.cache_read_cost_per_million)}")
    lines.append(f"  Cache Write:  {fmt_money(stats.cache_write_cost_per_million)}")

    return "\n".join(lines)
