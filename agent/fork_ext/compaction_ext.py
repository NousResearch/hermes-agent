"""Fork-owned pure compaction announce helpers.

These helpers are kept out of ``agent.conversation_compression`` so fork-only
announce rendering changes stay behind a stable import seam during upstream
syncs. They are pure formatters/gates: no agent mutation, no DB I/O.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Allow-list gating (Invariant I8 / §5.5). Statuses that ALWAYS represent a real
# context reduction → announce unconditionally.
_ANNOUNCE_STATUS_UNCONDITIONAL = frozenset(
    {"compacted", "overflow_recovery", "degraded_fallback_compressed"}
)
# Statuses that announce ONLY when context tokens actually dropped (token
# reduction is monotonic across LCM node reassignment; message count is not).
_ANNOUNCE_STATUS_CONDITIONAL = frozenset({"degraded_fail_open", "sanitized"})
# Statuses carrying a degraded-summary caveat.
_DEGRADED_STATUSES = frozenset({"degraded_fallback_compressed", "degraded_fail_open"})


def _fmt_gross_frac(gross_tok: int, pre_tok: int) -> str:
    """Render the raw-kept-tail-vs-pre ratio TRUTHFULLY for the observability marker.

    ``gross_tok`` (``raw_tail_tokens``) is a documented UPPER BOUND on the kept-tail
    magnitude, estimated on the RAW (pre-sanitize) suffix; ``pre_tok`` is the
    pre-compaction total counted on the SANITIZED/in-context basis. The two are on
    DIFFERENT bases, so the raw bound can legitimately exceed ``pre`` — printing a bare
    ``101.3%`` reads as an impossible "kept more than existed". Cap the displayed
    fraction at 100% and mark it as a bound (``≥100%``) with a basis note when it
    exceeds pre, so the marker is honest; the underlying comparison still uses the raw
    ratio (a bound over threshold correctly triggers the two-line degrade)."""
    if pre_tok <= 0:
        return "n/a (pre=0)"
    frac = gross_tok / pre_tok
    if frac > 1.0:
        # raw upper-bound exceeds the sanitized pre-total: bases differ, not a real >100%
        return f"≥100% (raw-tail bound {gross_tok} ≥ pre {pre_tok}; raw vs sanitized basis)"
    return f"{frac:.1%}"


def _inturn_stats_render_eligible(status, pre_tokens, post_tokens) -> bool:
    """True iff the LCM announce will actually RENDER for ``status`` — the P1 gate
    for the in-turn stats block (spec 2026-07-02, D-1/§5A).

    Mirrors ``_format_compaction_announce``'s LCM gating exactly, by consuming the
    SAME module-level allow-lists (single source of truth — no copied literals):
    unconditional statuses always render; conditional statuses render only when
    the token render-condition (``post < pre``, both truthy) holds; everything
    else (noop/idle/running/bypassed/unknown) is default-denied. Building stats —
    and emitting COMPACTION_STATS_* degrade WARNINGs — for a non-rendering status
    is pure log noise: the ~100%-kept_tail APPROX_ATTRIBUTION markers on no-op
    compactions that polluted the daily watcher report (2026-07-02 #logs).
    """
    if status in _ANNOUNCE_STATUS_UNCONDITIONAL:
        return True
    if status in _ANNOUNCE_STATUS_CONDITIONAL:
        return bool(pre_tokens and post_tokens and post_tokens < pre_tokens)
    return False


def _compaction_window_label(tokens: "int | None") -> str:
    """Compact human label for a context window: 1M, 272K, 128K."""
    if not tokens or tokens <= 0:
        return ""
    if tokens >= 1_000_000:
        whole = tokens / 1_000_000
        return f"{whole:.0f}M" if whole == int(whole) else f"{whole:.1f}M"
    if tokens >= 1_000:
        return f"{tokens // 1000}K"
    return str(tokens)


def _abbrev_tokens(tokens: "int | None") -> str:
    """~323K / ~15K / ~1.2M style for the token-delta display."""
    if tokens is None or tokens <= 0:
        return "?"
    if tokens >= 1_000_000:
        whole = tokens / 1_000_000
        return f"~{whole:.0f}M" if whole == int(whole) else f"~{whole:.1f}M"
    if tokens >= 1_000:
        return f"~{tokens // 1000}K"
    return f"~{tokens}"


def _compaction_reason_clause(
    trigger_reason: "str | None", trigger_value: "int | None"
) -> str:
    """Render the parenthetical 'why this fired' clause for the announce head."""
    if not trigger_reason:
        return ""
    if trigger_reason == "hygiene_messages":
        n = trigger_value if trigger_value else "?"
        return f" (message-count safety limit: {n} messages)"
    if trigger_reason == "hygiene_tokens":
        if trigger_value:
            return f" (session-hygiene token threshold, ~{trigger_value:,} tokens)"
        return " (session-hygiene token threshold)"
    if trigger_reason == "threshold":
        if trigger_value:
            return f" (crossed the compaction threshold, ~{trigger_value:,} tokens)"
        return " (crossed the compaction threshold)"
    if trigger_reason == "overflow_413":
        return " (the API rejected an oversize request — 413)"
    if trigger_reason == "overflow_context":
        return " (context length exceeded)"
    if trigger_reason == "tier_reduction":
        return " (long-context tier window reduction)"
    if trigger_reason == "manual":
        return " (you ran /compress)"
    return ""


def _format_compaction_announce(
    engine_name: "str | None",
    status: "str | None",
    *,
    old_session_id: "str | None",
    new_session_id: "str | None",
    old_messages: int,
    new_messages: int,
    pre_tokens: "int | None",
    post_tokens: "int | None",
    model: "str | None",
    provider: "str | None",
    window_from: "int | None" = None,
    window_to: "int | None" = None,
    summary_snippet: "str | None" = None,
    raw_store_count: "int | None" = None,
    after_fallback: bool = False,
    trigger_reason: "str | None" = None,
    trigger_value: "int | None" = None,
    reasoning: "str | None" = None,
    stats: "Any | None" = None,
    recovery_hint: "str | None" = None,
    in_place: bool = False,
) -> "str | None":
    """Build the engine-aware announce line, or ``None`` if gating says skip."""
    is_lcm = engine_name == "lcm"

    if is_lcm:
        if status in _ANNOUNCE_STATUS_UNCONDITIONAL:
            pass
        elif status in _ANNOUNCE_STATUS_CONDITIONAL:
            if not (pre_tokens and post_tokens and post_tokens < pre_tokens):
                return None
        else:
            return None
    else:
        if not new_session_id:
            return None
        if not in_place and (not old_session_id or old_session_id == new_session_id):
            return None

    degraded = status in _DEGRADED_STATUSES

    if stats is not None:
        try:
            _ok, _reason = stats.validate()
        except Exception as _verr:  # pragma: no cover - defensive
            _ok, _reason = False, f"validate() raised: {_verr}"
        if not _ok:
            logger.warning("COMPACTION_STATS_RECONCILE_FAILED %s", _reason)
            stats = None

    head = "🗜️ Context compacted"
    head += _compaction_reason_clause(trigger_reason, trigger_value)
    if after_fallback:
        head += " after model fallback"
    if degraded:
        head += " (degraded)"

    from agent.provider_model_util import format_provider_model

    model_part = format_provider_model(provider, model) if model else ""
    _r = (reasoning or "").strip().lower()
    if _r and _r not in {"default", "none"}:
        model_part = f"{model_part} · r:{_r}" if model_part else f"r:{_r}"
    if is_lcm and model_part:
        model_part = f"{model_part} · engine: lcm"
    elif is_lcm:
        model_part = "engine: lcm"

    if stats is not None:
        line = _format_granular_announce(
            head, stats, model_part, after_fallback, window_from, window_to,
        )
    else:
        parts = [f"{head}: {old_messages}→{new_messages} messages"]
        parts.append(f"{_abbrev_tokens(pre_tokens)}→{_abbrev_tokens(post_tokens)} tokens")
        if model:
            parts.append(format_provider_model(provider, model))
            if _r and _r not in {"default", "none"}:
                parts.append(f"r:{_r}")
        if after_fallback:
            wf, wt = _compaction_window_label(window_from), _compaction_window_label(window_to)
            if wf and wt and wf != wt:
                parts.append(f"window {wf}→{wt}")
        if is_lcm:
            parts.append("engine: lcm")
        line = " · ".join(parts)

    if recovery_hint:
        ref = recovery_hint
    elif is_lcm:
        if raw_store_count and raw_store_count > 0:
            ref = (
                f"↩ nothing lost — {raw_store_count:,} raw turns from this session "
                "preserved in lcm.db · recover with lcm_grep / lcm_expand"
            )
        else:
            ref = (
                "↩ nothing lost — raw turns preserved in lcm.db · "
                "recover with lcm_grep / lcm_expand"
            )
    else:
        ref = f"↩ previous: {old_session_id} → current: {new_session_id}"
    line += "\n" + ref

    if summary_snippet:
        line += f"\nSummary: {summary_snippet}"
    elif degraded:
        line += "\nSummary: unavailable — summarizer degraded this pass; raw store is intact."

    return line


def _append_subsplit_lines(lines, *, tool_count, tool_tokens, other_count, other_tokens, other_desc):
    """Append tool/other sub-split lines for a bucket."""
    if tool_count > 0:
        lines.append(
            f"     • {tool_count} tool-result messages  →  {_abbrev_tokens(tool_tokens)} reclaimed"
            f"   (raw tool output)"
        )
    if other_count > 0:
        lines.append(
            f"     • {other_count} other messages  →  {_abbrev_tokens(other_tokens)} reclaimed"
            f"   ({other_desc})"
        )


def _format_granular_announce(
    head: str, stats: "Any", model_part: str,
    after_fallback: bool, window_from: "int | None", window_to: "int | None",
    *, basis: str = "live",
    wire_before: "int | None" = None,
    wire_after: "int | None" = None,
) -> str:
    """Render the multi-line single-unit breakdown from validated stats."""
    stored = basis == "stored"
    wire_mode = bool(stored and (wire_before or 0) > 0 and (wire_after or 0) > 0)
    ctx_label = "Stored transcript:" if (stored and not wire_mode) else "Context:  "
    freed_verb = "reclaimed" if (stored and not wire_mode) else "freed"
    removed_hdr = "stored transcript" if stored else "live context"
    kept_where = "transcript" if stored else "context"
    lines: list[str] = []
    kept_bits = [f"kept {stats.kept_messages} recent chat"]
    if stats.summary_messages:
        kept_bits.append(f"{stats.summary_messages} summary")
    if stats.anchor_messages:
        kept_bits.append(f"{stats.anchor_messages} anchor{'s' if stats.anchor_messages != 1 else ''}")
    lines.append(f"{head}")
    lines.append(f"   Messages:  {stats.pre_messages} → {stats.post_messages}   ({' + '.join(kept_bits)})")

    if wire_mode:
        _wb, _wa = int(wire_before or 0), int(wire_after or 0)
        _wfreed = _wb - _wa
        if _wfreed > 0:
            _wpct = round(_wfreed * 100 / _wb) if _wb else 0
            lines.append(
                f"   Context:   {_wb:,} → ~{_wa:,} tokens"
                f"   (freed {_abbrev_tokens(_wfreed)}, {_wpct}% smaller"
                f" · before measured, after next-request estimate)"
            )
        else:
            lines.append(
                f"   Context:   {_wb:,} → ~{_wa:,} tokens"
                f"   (no net reduction expected"
                f" · before measured, after next-request estimate)"
            )
    elif stats.freed_tokens > 0 and stats.freed_pct is not None:
        lines.append(
            f"   {ctx_label} {_abbrev_tokens(stats.pre_tokens)} → {_abbrev_tokens(stats.post_tokens)} tokens"
            f"   ({freed_verb} {_abbrev_tokens(stats.freed_tokens)}, {stats.freed_pct}% smaller)"
        )
    else:
        lines.append(
            f"   {ctx_label} {_abbrev_tokens(stats.pre_tokens)} → {_abbrev_tokens(stats.post_tokens)} tokens"
            f"   (no net token reduction this pass)"
        )

    removed = stats.cleared_count + stats.folded_count
    if removed > 0:
        if wire_mode:
            _arch_freed = max(int(stats.freed_tokens or 0), 0)
            lines.append(
                f"   Removed from {removed_hdr} ({removed} messages,"
                f" {_abbrev_tokens(_arch_freed)} token-est reclaimed from archive):"
            )
        else:
            lines.append(f"   Removed from {removed_hdr} ({removed} messages):")
        if stats.cleared_count > 0:
            if stats.cleared_tool_count is not None:
                _append_subsplit_lines(
                    lines,
                    tool_count=stats.cleared_tool_count or 0,
                    tool_tokens=stats.cleared_tool_tokens or 0,
                    other_count=stats.cleared_other_count or 0,
                    other_tokens=stats.cleared_other_tokens or 0,
                    other_desc="system + tool-call turns, cleared",
                )
            else:
                lines.append(
                    f"     • {stats.cleared_count} cleared messages  →  {_abbrev_tokens(stats.cleared_tokens)} reclaimed"
                    f"   (tool results + system + tool-call-only turns)"
                )
        if stats.folded_count > 0:
            if stats.folded_tool_count is not None:
                _append_subsplit_lines(
                    lines,
                    tool_count=stats.folded_tool_count or 0,
                    tool_tokens=stats.folded_tool_tokens or 0,
                    other_count=stats.folded_other_count or 0,
                    other_tokens=stats.folded_other_tokens or 0,
                    other_desc=f"chat + tool-call turns + system, folded into {stats.summary_messages or 1} summary",
                )
            else:
                lines.append(
                    f"     • {stats.folded_count} folded messages   →  {_abbrev_tokens(stats.folded_tokens)} reclaimed"
                    f"   (older chat condensed into {stats.summary_messages or 1} summary)"
                )
        replacement = stats.summary_tokens + stats.anchor_tokens
        if replacement > 0:
            lines.append(
                f"   Replacement cost: {_abbrev_tokens(replacement)} kept in {kept_where} (summary + anchors)"
            )

    if after_fallback:
        wf, wt = _compaction_window_label(window_from), _compaction_window_label(window_to)
        if wf and wt and wf != wt:
            lines.append(f"   Window: {wf} → {wt}")

    if model_part:
        lines.append(f"   Model: {model_part}")

    return "\n".join(lines)
