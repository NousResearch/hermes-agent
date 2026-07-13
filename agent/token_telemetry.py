"""Privacy-safe token-efficiency telemetry helpers.

This module is deliberately observe-only: it attributes request size to coarse
context blocks and records provider usage/cache metrics without storing raw
prompts, tool schemas, or tool results by default.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent.model_metadata import estimate_messages_tokens_rough, estimate_request_tokens_rough
from hermes_constants import get_hermes_home

_SCHEMA = "hermes.token_efficiency.v1"
_IMAGE_TOKEN_COST = 1500


def _hash_text(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()[:16]


def _message_tokens(message: Dict[str, Any]) -> int:
    return estimate_messages_tokens_rough([message])


def _tool_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    if not tools:
        return 0
    return estimate_request_tokens_rough([], tools=tools)


def _count_images(message: Dict[str, Any]) -> int:
    count = 0
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image", "image_url", "input_image"}:
                count += 1
    stashed = message.get("_anthropic_content_blocks") if isinstance(message, dict) else None
    if isinstance(stashed, list):
        for part in stashed:
            if isinstance(part, dict) and part.get("type") == "image":
                count += 1
    return count


def _looks_like_summary(message: Dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, str):
        return False
    head = content[:512].lower()
    return (
        "context compaction" in head
        or "conversation summary" in head
        or "## progress" in head and "## critical context" in content[:4000].lower()
        or "compressed context" in head
    )


def _block(kind: str, rough_tokens: int, **extra: Any) -> Dict[str, Any]:
    item = {"kind": kind, "rough_tokens": max(0, int(rough_tokens))}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def classify_token_blocks(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return coarse, prompt-safe token attribution blocks.

    Blocks contain sizes, counts, hashes, and booleans — never raw message text
    or raw tool schema strings.
    """
    blocks: List[Dict[str, Any]] = []
    system_tokens = 0
    system_hash_parts: List[str] = []
    summary_tokens = 0
    tool_result_tokens = 0
    attachment_tokens = 0
    image_count = 0
    history_tokens = 0
    history_count = 0

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        tokens = _message_tokens(msg)
        role = msg.get("role")
        images = _count_images(msg)
        if images:
            image_count += images
            attachment_tokens += images * _IMAGE_TOKEN_COST
        if role == "system":
            system_tokens += tokens
            system_hash_parts.append(_hash_text(msg.get("content", "")))
        elif role == "tool":
            tool_result_tokens += tokens
        elif _looks_like_summary(msg):
            summary_tokens += tokens
        else:
            history_tokens += tokens
            history_count += 1

    if system_tokens:
        blocks.append(
            _block(
                "system",
                system_tokens,
                cacheable=True,
                stable_hash=_hash_text(system_hash_parts),
                count=len(system_hash_parts),
            )
        )
    tool_schema_tokens = _tool_tokens(tools)
    if tool_schema_tokens or tools:
        blocks.append(
            _block(
                "tools_schema",
                tool_schema_tokens,
                count=len(tools or []),
                cacheable=True,
                stable_hash=_hash_text(tools or []),
            )
        )
    if summary_tokens:
        blocks.append(_block("summary", summary_tokens, cacheable=True))
    if history_tokens:
        blocks.append(_block("history_recent", history_tokens, count=history_count))
    if tool_result_tokens:
        blocks.append(_block("tool_results", tool_result_tokens))
    if attachment_tokens:
        blocks.append(_block("attachments", attachment_tokens, image_count=image_count))
    if not blocks:
        blocks.append(_block("empty", 0))
    return blocks


def build_token_efficiency_record(
    *,
    session_id: str,
    turn_id: str,
    api_request_id: str,
    platform: str,
    provider: str,
    model: str,
    api_mode: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    rough_request_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    blocks = classify_token_blocks(messages, tools)
    if rough_request_tokens is None:
        rough_request_tokens = estimate_request_tokens_rough(messages or [], tools=tools or None)
    return {
        "schema": _SCHEMA,
        "created_at": time.time(),
        "session_id": session_id or "",
        "turn_id": turn_id or "",
        "api_request_id": api_request_id or "",
        "platform": platform or "",
        "provider": provider or "",
        "model": model or "",
        "api_mode": api_mode or "",
        "message_count": len(messages or []),
        "tool_count": len(tools or []),
        "rough_request_tokens": int(rough_request_tokens or 0),
        "blocks": blocks,
        "privacy": {"raw_prompt_stored": False, "raw_tools_stored": False},
    }


def _usage_int(usage: Dict[str, Any], key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except Exception:
        return 0


def _diagnostics(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rough = max(1, int(record.get("rough_request_tokens") or 0))
    by_kind = {b.get("kind"): int(b.get("rough_tokens") or 0) for b in record.get("blocks", [])}
    items: List[Dict[str, Any]] = []
    tool_ratio = by_kind.get("tools_schema", 0) / rough
    if tool_ratio >= 0.30:
        items.append({"kind": "tool_schema_overhead", "severity": "warn", "ratio": round(tool_ratio, 4)})
    tool_result_ratio = by_kind.get("tool_results", 0) / rough
    if tool_result_ratio >= 0.25:
        items.append({"kind": "tool_result_bloat", "severity": "warn", "ratio": round(tool_result_ratio, 4)})
    cache = record.get("cache") or {}
    hit_ratio = cache.get("hit_ratio")
    if isinstance(hit_ratio, (int, float)) and hit_ratio and hit_ratio < 0.40:
        items.append({"kind": "low_cache_hit_ratio", "severity": "info", "ratio": round(float(hit_ratio), 4)})
    return items


def _empty_actual() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }


def finalize_token_efficiency_record(
    record: Dict[str, Any],
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    finalized = dict(record)
    finalized["status"] = "success"
    usage = usage or {}
    actual = {
        "input_tokens": _usage_int(usage, "input_tokens") or _usage_int(usage, "prompt_tokens"),
        "output_tokens": _usage_int(usage, "output_tokens") or _usage_int(usage, "completion_tokens"),
        "cache_read_tokens": _usage_int(usage, "cache_read_tokens"),
        "cache_write_tokens": _usage_int(usage, "cache_write_tokens"),
        "reasoning_tokens": _usage_int(usage, "reasoning_tokens"),
    }
    finalized["actual"] = actual
    denom = actual["input_tokens"] + actual["cache_read_tokens"] + actual["cache_write_tokens"]
    finalized["cache"] = {
        "provider_reported": bool(actual["cache_read_tokens"] or actual["cache_write_tokens"]),
        "hit_ratio": (actual["cache_read_tokens"] / denom) if denom else 0.0,
        "write_to_read_ratio": (actual["cache_write_tokens"] / actual["cache_read_tokens"]) if actual["cache_read_tokens"] else None,
    }
    finalized["diagnostics"] = _diagnostics(finalized)
    finalized["completed_at"] = time.time()
    return finalized


def finalize_token_efficiency_no_usage(record: Dict[str, Any], *, reason: str = "provider_missing_usage") -> Dict[str, Any]:
    finalized = dict(record)
    finalized["status"] = "no_usage"
    finalized["reason"] = reason
    finalized["actual"] = _empty_actual()
    finalized["cache"] = {"provider_reported": False, "hit_ratio": 0.0, "write_to_read_ratio": None}
    finalized["diagnostics"] = [
        {"kind": "missing_usage", "severity": "info", "reason": reason},
        *_diagnostics(finalized),
    ]
    finalized["completed_at"] = time.time()
    return finalized


def finalize_token_efficiency_error(
    record: Dict[str, Any],
    *,
    error: BaseException,
    retry_count: int = 0,
    will_retry: bool = False,
) -> Dict[str, Any]:
    finalized = dict(record)
    finalized["status"] = "error"
    finalized["actual"] = _empty_actual()
    finalized["cache"] = {"provider_reported": False, "hit_ratio": 0.0, "write_to_read_ratio": None}
    finalized["error"] = {
        "type": type(error).__name__,
        "retry_count": int(retry_count or 0),
        "will_retry": bool(will_retry),
    }
    diagnostics = [{"kind": "api_error", "severity": "warn", "error_type": type(error).__name__}]
    if retry_count or will_retry:
        diagnostics.append({"kind": "retry_overhead", "severity": "info", "retry_count": int(retry_count or 0), "will_retry": bool(will_retry)})
    finalized["diagnostics"] = [*diagnostics, *_diagnostics(finalized)]
    finalized["completed_at"] = time.time()
    return finalized


def build_compression_efficiency_event(
    *,
    session_id: str,
    turn_id: str,
    before_tokens: int,
    after_tokens: int,
    trigger: str = "unknown",
    summary_text: Optional[str] = None,
) -> Dict[str, Any]:
    before = max(0, int(before_tokens or 0))
    after = max(0, int(after_tokens or 0))
    saved = max(0, before - after)
    return {
        "schema": "hermes.token_efficiency.compression.v1",
        "created_at": time.time(),
        "session_id": session_id or "",
        "turn_id": turn_id or "",
        "status": "compression",
        "trigger": trigger,
        "before_tokens": before,
        "after_tokens": after,
        "saved_tokens": saved,
        "roi_ratio": round((saved / before) if before else 0.0, 4),
        "summary_hash": _hash_text(summary_text or ""),
        "privacy": {"raw_summary_stored": False},
    }


class TokenEfficiencyStore:
    """Append-only JSONL store for token-efficiency telemetry."""

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path) if path is not None else default_token_efficiency_path()

    def append(self, record: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    def read_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rows: List[Dict[str, Any]] = []
        for line in lines[-max(1, int(limit)):]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows


def default_token_efficiency_path() -> Path:
    return get_hermes_home() / "telemetry" / "token_efficiency.jsonl"


def summarize_token_efficiency_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    totals: Dict[str, int] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    no_usage_records = 0
    error_records = 0
    compression_events = 0
    compression_saved = 0
    for row in rows:
        status = row.get("status")
        if status == "no_usage":
            no_usage_records += 1
        elif status == "error":
            error_records += 1
        elif status == "compression":
            compression_events += 1
            compression_saved += int(row.get("saved_tokens") or 0)
        actual = row.get("actual") or {}
        input_tokens += int(actual.get("input_tokens") or 0)
        output_tokens += int(actual.get("output_tokens") or 0)
        cache_read += int(actual.get("cache_read_tokens") or 0)
        cache_write += int(actual.get("cache_write_tokens") or 0)
        for block in row.get("blocks") or []:
            kind = block.get("kind") or "unknown"
            totals[kind] = totals.get(kind, 0) + int(block.get("rough_tokens") or 0)
    top_blocks = [
        {"kind": kind, "rough_tokens": tokens}
        for kind, tokens in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]
    denom = input_tokens + cache_read + cache_write
    return {
        "records": len(rows),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cache_read_tokens": cache_read,
        "total_cache_write_tokens": cache_write,
        "cache_hit_ratio": (cache_read / denom) if denom else 0.0,
        "no_usage_records": no_usage_records,
        "error_records": error_records,
        "compression_events": compression_events,
        "compression_saved_tokens": compression_saved,
        "top_blocks": top_blocks,
    }


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _recommendations(records: List[Dict[str, Any]], summary: Dict[str, Any]) -> List[str]:
    kinds = {diag.get("kind") for row in records for diag in (row.get("diagnostics") or [])}
    top = {block.get("kind"): int(block.get("rough_tokens") or 0) for block in summary.get("top_blocks") or []}
    recommendations: List[str] = []
    if "tool_schema_overhead" in kinds or top.get("tools_schema", 0) > top.get("history_recent", 0):
        recommendations.append("Use narrower toolsets for this task; tool schemas are a large prompt block.")
    if "tool_result_bloat" in kinds or top.get("tool_results", 0) >= 10_000:
        recommendations.append("prune or summarize old tool outputs; prefer paginated reads/search snippets over large dumps.")
    if summary.get("cache_hit_ratio", 0.0) < 0.40 and (summary.get("total_cache_read_tokens", 0) or summary.get("total_cache_write_tokens", 0)):
        recommendations.append("preserve stable prefixes and avoid injecting volatile context before cached blocks.")
    if top.get("summary", 0) >= 8_000 or top.get("history_recent", 0) >= 30_000:
        recommendations.append("Estimate compression ROI; consider /compress or a lower threshold for long-running sessions.")
    if summary.get("no_usage_records", 0):
        recommendations.append("Investigate provider missing usage metadata; token-efficiency attribution is blind after these calls.")
    if summary.get("error_records", 0):
        recommendations.append("Reduce retry/fallback overhead by narrowing context before retrying large failing requests.")
    if summary.get("compression_events", 0):
        recommendations.append(f"compression saved {_fmt_int(summary.get('compression_saved_tokens'))} rough tokens across recent events; use ROI to tune thresholds.")
    if not recommendations:
        recommendations.append("No major token-efficiency hotspot detected in the recent records.")
    return recommendations


def render_token_efficiency_report(
    *,
    store: Optional[TokenEfficiencyStore] = None,
    limit: int = 20,
    session_id: Optional[str] = None,
) -> str:
    """Render a read-only token-efficiency report.

    The report is intentionally prescriptive: it turns safe local telemetry into
    concrete ways to reduce future token usage without mutating requests.
    """
    store = store or TokenEfficiencyStore()
    rows = store.read_recent(limit=limit)
    if session_id:
        rows = [row for row in rows if row.get("session_id") == session_id]
    lines = ["Token Efficiency Report (observe-only)", "────────────────────────────────────────"]
    if not rows:
        lines.extend([
            "No token efficiency records yet.",
            "Run a model call, then retry /token-efficiency to see block-level context attribution.",
        ])
        return "\n".join(lines)

    summary = summarize_token_efficiency_records(rows)
    last = rows[-1]
    lines.extend([
        f"Records analyzed:      {_fmt_int(summary['records'])}",
        f"Input tokens:          {_fmt_int(summary['total_input_tokens'])}",
        f"Output tokens:         {_fmt_int(summary['total_output_tokens'])}",
        f"Cache read tokens:     {_fmt_int(summary['total_cache_read_tokens'])}",
        f"Cache write tokens:    {_fmt_int(summary['total_cache_write_tokens'])}",
        f"Cache hit ratio:       {summary['cache_hit_ratio'] * 100:.0f}%",
        f"No-usage records:     {_fmt_int(summary.get('no_usage_records'))}",
        f"Error records:        {_fmt_int(summary.get('error_records'))}",
        f"Compression events:   {_fmt_int(summary.get('compression_events'))}",
        f"Compression saved:    {_fmt_int(summary.get('compression_saved_tokens'))} rough tokens",
        "",
        "Top prompt blocks:",
    ])
    for block in (summary.get("top_blocks") or [])[:6]:
        lines.append(f"  - {block['kind']:<16} {_fmt_int(block['rough_tokens'])} rough tokens")

    lines.extend(["", "Last request:"])
    lines.append(f"  model/provider: {last.get('model', '')} / {last.get('provider', '')}")
    lines.append(f"  request size:   {_fmt_int(last.get('rough_request_tokens'))} rough tokens")
    last_cache = last.get("cache") or {}
    if last_cache:
        lines.append(f"  cache hit:      {float(last_cache.get('hit_ratio') or 0.0) * 100:.0f}%")
    last_diagnostics = last.get("diagnostics") or []
    if last_diagnostics:
        lines.append("  diagnostics:")
        for diag in last_diagnostics[:6]:
            ratio = diag.get("ratio")
            suffix = f" ({float(ratio) * 100:.0f}%)" if isinstance(ratio, (int, float)) else ""
            lines.append(f"    - {diag.get('kind', 'unknown')}{suffix}")

    lines.extend(["", "Recommendations:"])
    for recommendation in _recommendations(rows, summary):
        lines.append(f"  - {recommendation}")
    lines.extend([
        "",
        "Mode: read-only. No tools, context, model, or cache policy were changed.",
    ])
    return "\n".join(lines)


def _proposal(
    *,
    key: str,
    title: str,
    rationale: str,
    estimated_savings_tokens: int,
    confidence: str,
    evidence: Optional[Dict[str, Any]] = None,
    action_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "rationale": rationale,
        "estimated_savings_tokens": max(0, int(estimated_savings_tokens or 0)),
        "confidence": confidence if confidence in {"low", "medium", "high"} else "low",
        "evidence": evidence or {},
        "action_spec": action_spec or {},
        "mutation_available": False,
        "requires_approval": True,
        "auto_applied": False,
    }


def _action_spec(key: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "key": key,
        "would_not_change": ["model", "provider", "system_prompt", "memory", "skills", "external_actions"],
        "requires_human_review": True,
        "apply_endpoint": None,
        "auto_apply_allowed": False,
        "fallback": "keep_current_context_strategy",
    }
    specs: Dict[str, Dict[str, Any]] = {
        "narrow_toolsets": {
            "kind": "toolset_scope",
            "would_change": ["enabled tool schema surface for future similar tasks"],
            "capability_risk": "medium",
            "preserve_capabilities": ["terminal", "file", "search/session_search when needed", "skills", "memory"],
            "lost_capabilities_to_review": ["browser", "image/video/media tools", "messaging", "cron/delegation when not needed"],
            "fallback": "restore_full_task_toolset_if_capability_needed",
        },
        "compress_now": {
            "kind": "compression",
            "would_change": ["compression timing recommendation", "candidate before/after context size"],
            "capability_risk": "medium",
            "preserve_recent_decisions": True,
            "preserve_open_tasks": True,
            "fallback": "keep_uncompressed_context_or_manual_compress_only",
        },
        "prune_tool_results": {
            "kind": "tool_result_policy",
            "would_change": ["summarize old tool results", "prefer paginated reads/search snippets"],
            "capability_risk": "low",
            "preserve_recent_tool_results": True,
            "fallback": "retain_full_tool_results_for_current_turn",
        },
        "stabilize_prefix": {
            "kind": "cache_prefix",
            "would_change": ["move volatile context after stable cacheable prefix", "keep system/tool/skill prefix stable"],
            "capability_risk": "low",
            "preserve_instruction_priority": True,
            "fallback": "keep_current_prompt_order",
        },
        "lazy_retrieve_history": {
            "kind": "lazy_retrieval",
            "would_change": ["move older history behind targeted retrieval", "inject only relevant recalled slices"],
            "capability_risk": "medium",
            "retrieval_fallback": "session_search_or_full_context",
            "fallback": "restore_full_history_when_uncertainty_is_high",
        },
    }
    spec = dict(base)
    spec.update(specs.get(key, {
        "kind": "generic_preview",
        "would_change": ["future context strategy"],
        "capability_risk": "medium",
    }))
    return spec


def build_token_efficiency_optimizer_preview(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Build preview-only optimizer proposals from token-efficiency telemetry.

    This function deliberately does not mutate config, toolsets, context,
    models, cache policy, or compression thresholds. It translates observed
    hotspots into possible future actions that must be explicitly approved.
    """
    rows = list(records)
    summary = summarize_token_efficiency_records(rows)
    top = {block.get("kind"): int(block.get("rough_tokens") or 0) for block in summary.get("top_blocks") or []}
    total_block_tokens = sum(top.values()) or 1
    diagnostics = {diag.get("kind") for row in rows for diag in (row.get("diagnostics") or [])}
    proposals: List[Dict[str, Any]] = []

    tools_tokens = top.get("tools_schema", 0)
    if tools_tokens and ("tool_schema_overhead" in diagnostics or tools_tokens / total_block_tokens >= 0.25):
        proposals.append(_proposal(
            key="narrow_toolsets",
            title="Preview narrower toolsets",
            rationale="Tool schemas are a large context block; a narrower task-specific toolset could reduce stable prompt overhead.",
            estimated_savings_tokens=int(tools_tokens * 0.35),
            confidence="high" if tools_tokens / total_block_tokens >= 0.35 else "medium",
            evidence={"tools_schema_tokens": tools_tokens, "share": round(tools_tokens / total_block_tokens, 3)},
            action_spec=_action_spec("narrow_toolsets"),
        ))

    tool_result_tokens = top.get("tool_results", 0)
    if tool_result_tokens and ("tool_result_bloat" in diagnostics or tool_result_tokens >= 2_500):
        proposals.append(_proposal(
            key="prune_tool_results",
            title="Preview pruning old tool results",
            rationale="Tool outputs should feed the immediate step, not become long-term prompt ballast; summarize or page older results.",
            estimated_savings_tokens=int(tool_result_tokens * 0.55),
            confidence="high" if tool_result_tokens >= 10_000 else "medium",
            evidence={"tool_result_tokens": tool_result_tokens},
            action_spec=_action_spec("prune_tool_results"),
        ))

    history_tokens = top.get("history_recent", 0) + top.get("summary", 0)
    compression_saved = int(summary.get("compression_saved_tokens") or 0)
    if history_tokens >= 8_000 or compression_saved:
        estimated = compression_saved if compression_saved else int(history_tokens * 0.45)
        proposals.append(_proposal(
            key="compress_now",
            title="Preview context compression timing",
            rationale="Long history/summary blocks suggest compression ROI may be positive; preview before/after thresholds before changing them.",
            estimated_savings_tokens=estimated,
            confidence="high" if compression_saved >= 10_000 else "medium",
            evidence={"history_summary_tokens": history_tokens, "recent_compression_saved_tokens": compression_saved},
            action_spec=_action_spec("compress_now"),
        ))

    cache_hit_ratio = float(summary.get("cache_hit_ratio") or 0.0)
    cache_observed = int(summary.get("total_cache_read_tokens") or 0) + int(summary.get("total_cache_write_tokens") or 0)
    if cache_observed and cache_hit_ratio < 0.40:
        proposals.append(_proposal(
            key="stabilize_prefix",
            title="Preview stable-prefix cleanup",
            rationale="Cache hit ratio is low despite cache traffic; volatile context may be appearing before cacheable blocks.",
            estimated_savings_tokens=int(cache_observed * max(0.0, 0.40 - cache_hit_ratio)),
            confidence="medium",
            evidence={"cache_hit_ratio": round(cache_hit_ratio, 3), "cache_observed_tokens": cache_observed},
            action_spec=_action_spec("stabilize_prefix"),
        ))

    if history_tokens >= 20_000 or (history_tokens > 0 and int(top.get("history_recent", 0)) >= int(top.get("tools_schema", 0) * 0.5)):
        proposals.append(_proposal(
            key="lazy_retrieve_history",
            title="Preview lazy history retrieval",
            rationale="History is large enough to consider moving older context behind targeted retrieval/session search instead of always injecting it.",
            estimated_savings_tokens=int(history_tokens * 0.30),
            confidence="medium" if history_tokens >= 20_000 else "low",
            evidence={"history_summary_tokens": history_tokens},
            action_spec=_action_spec("lazy_retrieve_history"),
        ))

    limited_proposals = proposals[:8]
    total_estimated_savings = sum(int(p.get("estimated_savings_tokens") or 0) for p in limited_proposals)
    needs_attention = bool(limited_proposals and total_estimated_savings >= 1_000)
    visibility = {
        "needs_attention": needs_attention,
        "recommended_surface": "mission_control_review" if needs_attention else "diagnostics_collapsed",
        "reason": "actionable_token_hotspots" if needs_attention else "no_actionable_token_hotspots",
        "cockpit_card": None,
    }
    if needs_attention:
        visibility["cockpit_card"] = {
            "title": "Token efficiency hotspot",
            "why_it_matters": "Recent context construction has actionable token hotspots; reviewing them may reduce overhead without changing model or capabilities.",
            "recommended_action": "Review optimizer preview; do not apply automatically.",
            "autonomy_gate": "review_only_no_auto_apply",
            "proposal_count": len(limited_proposals),
            "estimated_savings_tokens": total_estimated_savings,
            "function_first": True,
        }

    return {
        "ok": True,
        "label": "Optimizer Recommendation Preview",
        "mode": "preview_only",
        "mutation_available": False,
        "requires_approval": True,
        "auto_applied": False,
        "summary": summary,
        "proposals": limited_proposals,
        "visibility": visibility,
        "guardrail": {
            "no_request_mutation": True,
            "no_context_mutation": True,
            "no_toolset_mutation": True,
            "no_model_change": True,
            "no_cache_policy_change": True,
            "no_auto_apply": True,
            "approval_required_for_future_optimizer": True,
            "function_first": True,
            "no_capability_degradation_without_review": True,
            "avoid_over_engineering": True,
            "prompt_text_stored": False,
            "tool_output_text_stored": False,
        },
    }
