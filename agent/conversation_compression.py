"""Context compression — extract the AIAgent methods that drive summarisation.

Three concerns live here:

* :func:`check_compression_model_feasibility` — startup probe of the
  configured auxiliary compression model.  Warns when the aux context
  window can't fit the main model's compression threshold; auto-lowers
  the session threshold when possible; hard-rejects auxes below
  ``MINIMUM_CONTEXT_LENGTH``.

* :func:`replay_compression_warning` — re-emit a stored warning through
  the gateway ``status_callback`` once it's wired up (the callback is
  set after :class:`AIAgent` construction).

* :func:`compress_context` — the actual compression call.  Runs the
  configured compressor, splits the SQLite session, rotates the
  session_id, notifies plugin context engines / memory providers, and
  returns the compressed message list and freshly-built system prompt.

* :func:`try_shrink_image_parts_in_messages` — image-too-large recovery
  helper that re-encodes ``data:image/...;base64,...`` parts at a smaller
  size so retries can fit under provider ceilings (Anthropic's 5 MB).

``run_agent`` keeps thin wrappers for each so existing call sites
(``self._compress_context(...)``) keep working.  Tests that exercise
these paths see no behavioural change.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from agent.model_metadata import estimate_request_tokens_rough

logger = logging.getLogger(__name__)

# Stable marker the gateway matches on to re-tag the auto-compaction lifecycle
# status as ``kind="compacting"`` (tui_gateway/server.py::_status_update), so
# drivers like the desktop app can show an explicit "Summarizing…" indicator
# instead of the transcript appearing to silently reset. Keep the marker phrase
# intact if you reword COMPACTION_STATUS.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = (
    f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
)

# ── Compaction completion announce (engine-aware) ──────────────────────────
# Spec: ~/.hermes/plans/2026-06-20_compaction-announce-with-context-reference.md
# A persistent, in-chat marker emitted when context is actually compacted, for
# BOTH the built-in ContextCompressor (lossy, session-rotating) and the LCM/DAG
# engine (lossless raw store + lcm_grep/lcm_expand recovery). Additive to the
# fallback announce (never a replacement). Emitted out-of-band via _emit_status,
# never injected into model history.

# Markers stripped from a summary snippet before display.
_COMPACTION_SUMMARY_MARKERS = (
    "[CONTEXT COMPACTION — REFERENCE ONLY]",
    "[CONTEXT COMPACTION - REFERENCE ONLY]",
    "[CONTEXT COMPACTION]",
)

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

# A-floor approximate-attribution gross-error ceiling (in-turn granular announce).
# The A-floor (fallback single-walk partition in build_inturn_stats) reconciles
# TOTALS by construction but its kept/folded SPLIT is signature-approximate. The
# split error is provably bounded by the kept-tail fraction of pre (the folded bulk
# is a contiguous prefix that always classifies correctly), so when the kept tail
# exceeds this fraction the displayed split could be materially wrong and the render
# degrades to the honest two-line form instead. On real failing sessions the kept
# tail is ≤~7% of pre, so this never trips in practice — it is the honest backstop.
_APPROX_GROSS_MAX_FRAC = 0.10


def _warn_compaction_stats_once(agent, message: str, *, exc_info: bool = False) -> None:
    """Emit a compaction-stats degrade ``warning`` at most once per (cause, session).

    The granular compaction announce silently degrades to a two-line form when
    stats fail to build/reconcile; logging that at ``debug`` is how the PR #95
    regression stayed dark for weeks. This raises it to ``warning`` with a stable,
    greppable ``COMPACTION_STATS_*`` marker — but throttled per cause+session so a
    persistent reconcile bug can't flood the gateway log every turn. The throttle
    state lives on the agent (``_compaction_stats_warned``); if the agent can't
    hold it (no attribute), we still warn (fail-loud over fail-silent).
    """
    try:
        seen = getattr(agent, "_compaction_stats_warned", None)
        if seen is None:
            seen = set()
            try:
                agent._compaction_stats_warned = seen
            except Exception:
                seen = None
        # Key on the marker + path (first 2 tokens, e.g.
        # "COMPACTION_STATS_RECONCILE_FAILED in-turn"), NOT the full message, so a
        # varying reconcile reason doesn't defeat the throttle.
        cause = " ".join(message.split()[:2])
        key = (cause, getattr(agent, "session_id", None))
        if seen is not None:
            if key in seen:
                return
            seen.add(key)
    except Exception:
        pass  # never let throttle bookkeeping break the reply path
    logger.warning(message, exc_info=exc_info)


def _compaction_window_label(tokens: "int | None") -> str:
    """Compact human label for a context window: 1M, 272K, 128K (mirror of
    chat_completion_helpers._format_context_window; kept local to avoid a
    cross-module import cycle)."""
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


def _msg_text(content: Any) -> str:
    """Flatten a message ``content`` (str or list-of-blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _extract_compaction_summary_snippet(
    compressed_messages: list, *, max_chars: int = 160
) -> "str | None":
    """Deterministically pull a one-line snippet of what was summarised.

    Scans for the first message carrying a compaction summary marker, strips the
    marker boilerplate, collapses whitespace, and truncates at a word boundary.
    Returns ``None`` when no usable summary text exists (e.g. a placeholder-only
    marker). Not an LLM call.
    """
    if not compressed_messages:
        return None
    for msg in compressed_messages:
        if not isinstance(msg, dict):
            continue
        text = _msg_text(msg.get("content"))
        if not text:
            continue
        if not any(m in text for m in _COMPACTION_SUMMARY_MARKERS):
            continue
        for m in _COMPACTION_SUMMARY_MARKERS:
            text = text.replace(m, " ")
        # collapse all whitespace runs to single spaces
        cleaned = " ".join(text.split())
        if not cleaned:
            return None
        if len(cleaned) <= max_chars:
            return cleaned
        # truncate at a word boundary, then append an ellipsis
        cut = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()
        if not cut:
            cut = cleaned[:max_chars].rstrip()
        return cut + "…"
    return None


def _compaction_reason_clause(
    trigger_reason: "str | None", trigger_value: "int | None"
) -> str:
    """Render the parenthetical 'why this fired' clause for the announce head.

    Returns '' for None/unknown reasons (back-compat: no clause). The value is
    the real number that tripped the trigger (message count / threshold tokens).
    """
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
    return ""  # unknown/future reason → no clause (never echo a raw token)


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
    """Build the engine-aware announce line, or ``None`` if gating says skip.

    Gating (§5.5, Invariant I8):
    - LCM (``engine_name == "lcm"``): allow-list on ``status`` — unconditional
      set always announces; conditional set announces only on a real token drop;
      everything else (incl. unknown/future) is silent.
    - Built-in (no engine name / no status): announce only when a real rotation
      happened (``old_session_id != new_session_id``).

    ``trigger_reason``/``trigger_value`` (optional) render an honest 'why this
    fired' clause in the head (message-count valve, token valve, overflow, …);
    they NEVER affect gating.

    ``stats`` (optional ``CompactionStats``) renders the granular reconciling
    breakdown ("Removed from live context …"). If absent OR it fails
    ``validate()``, the announce degrades to the two-line Messages+Context form
    (a reconcile failure can never ship wrong math). ``reasoning`` adds the
    ``r:<level>`` segment to the model line (omitted for unset/default/none).
    ``recovery_hint`` overrides the default recovery line (per-path store).
    """
    is_lcm = engine_name == "lcm"

    if is_lcm:
        if status in _ANNOUNCE_STATUS_UNCONDITIONAL:
            pass
        elif status in _ANNOUNCE_STATUS_CONDITIONAL:
            if not (pre_tokens and post_tokens and post_tokens < pre_tokens):
                return None
        else:
            return None  # default-deny: noop/idle/running/bypassed/unknown
    else:
        # built-in compressor: a real compaction normally rotates the session id.
        # EXCEPTION (2026-06-29 upstream merge): upstream's in-place compaction
        # (compression.in_place=True, the config default) rewrites the transcript
        # WITHOUT rotating — old_session_id is None / equals new_session_id. That
        # is still a real compaction and must announce (otherwise the announce
        # campaign goes dark for every in-place compaction). Only gate out the
        # genuine no-op: no new session id at all.
        if not new_session_id:
            return None
        if not in_place and (not old_session_id or old_session_id == new_session_id):
            return None

    degraded = status in _DEGRADED_STATUSES

    # Validate stats; on any failure fall back to None (two-line form) and log a
    # loud, greppable marker — a reconcile bug must NEVER ship wrong numbers.
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
    # r:<level> — omit for unset/empty/default/none (match runtime footer skip set)
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
        # ── back-compat two-line form ──
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

    # Recovery reference (engine-correct, or caller-supplied per-path hint).
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
    """Append the tool/other sub-split sub-lines for a bucket, with zero-count
    suppression (CHANGE-C): a populated-zero count never headlines a `0 … → ~0K`
    line. The tool parenthetical is DESCRIPTIVE ("raw tool output"), not a
    superlative the numbers could falsify (BLOCKER-1)."""
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
) -> str:
    """Render the multi-line single-unit (messages) breakdown from a validated
    ``CompactionStats``. Every line leads with messages; tokens are the
    parenthetical secondary. Reconciles by construction (validate() passed)."""
    lines: list[str] = []
    # Headline: messages pre→post + what's kept
    kept_bits = [f"kept {stats.kept_messages} recent chat"]
    if stats.summary_messages:
        kept_bits.append(f"{stats.summary_messages} summary")
    if stats.anchor_messages:
        kept_bits.append(f"{stats.anchor_messages} anchor{'s' if stats.anchor_messages != 1 else ''}")
    lines.append(f"{head}")
    lines.append(f"   Messages:  {stats.pre_messages} → {stats.post_messages}   ({' + '.join(kept_bits)})")

    # Context line — guard freed<=0 (no net reduction)
    if stats.freed_tokens > 0 and stats.freed_pct is not None:
        lines.append(
            f"   Context:   {_abbrev_tokens(stats.pre_tokens)} → {_abbrev_tokens(stats.post_tokens)} tokens"
            f"   (freed {_abbrev_tokens(stats.freed_tokens)}, {stats.freed_pct}% smaller)"
        )
    else:
        lines.append(
            f"   Context:   {_abbrev_tokens(stats.pre_tokens)} → {_abbrev_tokens(stats.post_tokens)} tokens"
            f"   (no net token reduction this pass)"
        )

    # "Removed from live context" block — omit entirely when nothing cleared
    removed = stats.cleared_count + stats.folded_count
    if removed > 0:
        lines.append(f"   Removed from live context ({removed} messages):")
        # cleared bucket: render the tool/other sub-split when populated, else coarse line
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
        # folded bucket: same, with the in-turn "other" description
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
                f"   Replacement cost: {_abbrev_tokens(replacement)} kept in context (summary + anchors)"
            )

    if after_fallback:
        wf, wt = _compaction_window_label(window_from), _compaction_window_label(window_to)
        if wf and wt and wf != wt:
            lines.append(f"   Window: {wf} → {wt}")

    if model_part:
        lines.append(f"   Model: {model_part}")

    return "\n".join(lines)


def _emit_compaction_announce(agent: Any, *, dedupe_key, **fmt_kwargs) -> None:
    """Emit the compaction announce once per real compaction boundary.

    Dedupe: ``agent._last_compaction_announced`` holds an engine-namespaced
    ``dedupe_key``; a repeat key is skipped. The key is set ONLY after a
    successful emit (D7), so a swallowed emit failure does not suppress the next
    real compaction's announce. The caller holds the compression lock, so the
    read-then-write is serialized per session.

    Loud-fail (§3.5, B3): this site KNOWS the message is a compaction announce
    (so the marker is announce-specific, never tripped by a generic lifecycle
    status). When ``status_callback`` is absent the throwaway/gateway caller is
    expected to deliver → INFO ``PENDING_CALLER_DELIVERY``. When the in-turn
    ``status_callback`` exists but the send fails →
    WARNING ``STATUS_CALLBACK_FAILED`` (the in-turn silent hole, now loud).
    """
    line = _format_compaction_announce(**fmt_kwargs)
    if line is None:
        return  # gating skip — do not advance the key
    if getattr(agent, "_last_compaction_announced", None) == dedupe_key:
        return  # already announced this boundary
    _sid = getattr(agent, "session_id", None) or "?"
    emit = getattr(agent, "_emit_status", None)
    if not callable(emit):
        return
    _had_callback = bool(getattr(agent, "status_callback", None))
    try:
        delivered = emit(line)
    except Exception:
        logger.debug("compaction announce emit failed", exc_info=True)
        return  # key NOT advanced — next real compaction can still announce
    # ``_emit_status`` returns True only when a gateway status_callback existed
    # AND did not raise. Three cases (loud-fail §3.5 / B3):
    if delivered:
        agent._last_compaction_announced = dedupe_key  # in-turn live delivery OK
        return
    if _had_callback:
        # callback existed but the send leg raised → the in-turn announce was
        # LOST. Make it loud (was the one remaining silent-compaction hole).
        logger.warning("COMPACTION_ANNOUNCE_STATUS_CALLBACK_FAILED session=%s", _sid)
        return  # key NOT advanced — a retry can still announce
    # No gateway callback → either a CLI-only agent (the _vprint leg already
    # showed it) or a throwaway agent (hygiene/compress) whose CALLER delivers
    # from real facts. Either way it is NOT a silent failure. Mark pending so a
    # throwaway-path delivery gap is auditable, and advance the key (CLI delivery
    # via _vprint is complete; the throwaway caller dedupes structurally).
    logger.info(
        "COMPACTION_ANNOUNCE_PENDING_CALLER_DELIVERY reason=%s session=%s",
        fmt_kwargs.get("trigger_reason"), _sid,
    )
    agent._last_compaction_announced = dedupe_key


# Tight wall-clock window (seconds) used ONLY when no turn id is available to
# link a fallback to a following compaction. Real chained fallback→compaction
# happens within one turn (seconds), not minutes — keep this tight.
_POST_FALLBACK_WALLCLOCK_SECS = 75.0


def _compaction_after_fallback(
    agent: Any, *, now_monotonic: float, current_turn_id: "str | None"
) -> "Tuple[bool, Optional[int], Optional[int]]":
    """Decide whether this compaction follows a model fallback, turn-scoped.

    Returns ``(after_fallback, window_from, window_to)``. The causal signal is
    *same logical turn AND fallback-before-compaction* — NOT wall-clock
    proximity (the §0 incident had the fallback AFTER the compaction, which must
    NOT be labeled). Only when no turn id exists on either side does it fall back
    to a tight wall-clock window, still requiring fallback-before-compaction.
    """
    ev = getattr(agent, "_last_fallback_event", None)
    if not isinstance(ev, dict):
        return (False, None, None)
    fb_mono = ev.get("monotonic_time")
    if fb_mono is None or fb_mono > now_monotonic:
        # fallback happened AFTER this compaction (or unknown time) → not causal
        return (False, None, None)
    fb_turn = ev.get("turn_id")
    if fb_turn is not None and current_turn_id is not None:
        if fb_turn != current_turn_id:
            return (False, None, None)
    else:
        # no turn linkage available → tight wall-clock fallback
        if (now_monotonic - fb_mono) > _POST_FALLBACK_WALLCLOCK_SECS:
            return (False, None, None)
    return (True, ev.get("old_window"), ev.get("new_window"))


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique holder id for the lock: pid:tid:agent-instance:uuid.

    The pid+tid prefix lets ops tell crashed/abandoned holders apart from
    live ones (expiry-based recovery uses the timestamp, but ``holder``
    is what shows up in diagnostics + log lines). The agent instance id
    and a per-acquire uuid disambiguate two co-resident agents on the
    same thread (background_review forks run on a worker thread, but
    on machines where compression itself dispatches to a thread pool
    we want each acquire to be unique).
    """
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


class _CompressionLockLeaseRefresher:
    def __init__(
        self,
        db: Any,
        session_id: str,
        holder: str,
        ttl_seconds: float,
        refresh_interval_seconds: float | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one lease's worth of
        # time, so the give-up window is genuinely bounded by the TTL the
        # acquirer set (a single blip recovers on the next tick; a persistent
        # failure stops before the lease could outlive its TTL). Floor of 1 so a
        # degenerate interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(
            1, int(self._ttl_seconds / self._refresh_interval_seconds)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-lock-refresh",
            daemon=True,
        )

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() may time out while the refresher is mid-UPDATE; that's safe —
        # it's a daemon thread, and a late refresh on an already-released lock
        # matches rowcount 0 (a no-op). stop() returning does not guarantee the
        # thread has fully quiesced, only that we've signalled it and waited
        # briefly.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh must NOT permanently kill the lease: a
        # transient DB blip (write contention escaping _execute_write's retry
        # budget, a momentary "database is locked") returns False just like a
        # genuine lost-ownership, but only the latter should stop the loop.
        # Tolerate consecutive failures for at most one lease's worth of time
        # (_max_consecutive_failures = ttl / interval), so a one-off blip
        # recovers on the next tick while the total give-up window stays bounded
        # by the TTL the acquirer set — the lock can never be held past its TTL
        # by a stuck refresher.
        consecutive_failures = 0
        while not self._stop.wait(self._refresh_interval_seconds):
            try:
                refreshed = self._db.refresh_compression_lock(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as exc:
                logger.debug("compression lock refresh raised: %s", exc)
                refreshed = False
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; "
                    "stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def check_compression_model_feasibility(agent: Any) -> None:
    """Warn at session start if the auxiliary compression model's context
    window is smaller than the main model's compression threshold.

    When the auxiliary model cannot fit the content that needs summarising,
    compression will either fail outright (the LLM call errors) or produce
    a severely truncated summary.

    Called during ``AIAgent.__init__`` so CLI users see the warning
    immediately (via ``_vprint``).  The gateway sets ``status_callback``
    *after* construction, so :func:`replay_compression_warning` re-sends
    the stored warning through the callback on the first
    ``run_conversation()`` call.
    """
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        # Best-effort aux provider label for the warning message. The
        # configured provider may be "auto", in which case we fall back
        # to the client's base_url hostname so the user can still tell
        # where the compression model is actually being called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression",
                _aux_cfg_provider,
            )
            if fb_client is not None and fb_model:
                client, aux_model = fb_client, fb_model
                if "(" in fb_label and fb_label.endswith(")"):
                    _aux_cfg_provider = fb_label.rsplit("(", 1)[1][:-1]
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # ``client.api_key`` may be a callable (Azure Foundry Entra ID
        # bearer provider). The context-length resolver chain expects a
        # string, but it only needs a key for live catalogue probes
        # (provider model lists). For Entra clients the model-metadata
        # chain still resolves via models.dev + hardcoded family
        # fallbacks, which don't require auth — pass empty string rather
        # than minting a bearer JWT just to look up a context length.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Each model must be resolved with its own provider so that
            # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
            # are invoked for the correct client, not inherited from the main model.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # Hard floor: the auxiliary compression model must have at least
        # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
        # is already required to meet this floor (checked earlier in
        # __init__), so the compression model must too — otherwise it
        # cannot summarise a full threshold-sized window of main-model
        # content.  Mirrors the main-model rejection pattern.
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # Auto-correct: lower the live session threshold so
            # compression actually works this session.  The hard floor
            # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
            # so the new threshold is always >= 64K.
            #
            # The compression summariser sends a single user-role
            # prompt (no system prompt, no tools) to the aux model, so
            # new_threshold == aux_context is safe: the request is
            # the raw messages plus a small summarisation instruction.
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # Keep threshold_percent in sync so future main-model
            # context_length changes (update_model) re-derive from a
            # sensible number rather than the original too-high value.
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # Build human-readable "model (provider)" labels for both
            # the main model and the compression model so users can
            # tell at a glance which provider each side is actually
            # using. When the configured provider is empty or "auto",
            # fall back to the client's base_url hostname.
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
                f"  To make this permanent, edit config.yaml — either:\n"
                f"  1. Use a larger compression model:\n"
                f"       auxiliary:\n"
                f"         compression:\n"
                f"           model: <model-with-{old_threshold:,}+-context>\n"
                f"  2. Lower the compression threshold:\n"
                f"       compression:\n"
                f"         threshold: 0.{safe_pct:02d}"
            )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate
        # so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """Re-send the compression warning through ``status_callback``.

    During ``__init__`` the gateway's ``status_callback`` is not yet
    wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
    method is called once at the start of the first
    ``run_conversation()`` — by then the gateway has set the callback,
    so every platform (Telegram, Discord, Slack, etc.) receives the
    warning.
    """
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def conversation_history_after_compression(agent: Any, messages: list) -> Optional[list]:
    """Return the correct flush baseline after a compression boundary.

    Legacy compression rotates to a fresh child session. That child has not
    seen the compacted transcript through the normal same-turn flush path yet,
    so callers must clear ``conversation_history`` to ``None`` and let the next
    persistence call write the whole compacted list.

    In-place compaction is different: ``archive_and_compact()`` has already
    soft-archived the previous active rows and inserted ``messages`` as the new
    active live transcript under the same session id. If the same agent turn
    continues with ``conversation_history=None``, the identity-based flush path
    treats those already-persisted compacted dicts as new and appends them a
    second time, doubling the active context and retriggering compression.

    A shallow copy is intentional: it captures the current compacted dict
    identities as history while allowing later same-turn appends to remain new.
    """
    if bool(getattr(agent, "_last_compaction_in_place", False)):
        return list(messages)
    return None


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
    trigger_reason: Optional[str] = None,
) -> Tuple[list, str]:
    """Compress conversation context and split the session in SQLite.

    Args:
        agent: The owning :class:`AIAgent`.
        messages: Current message history (will be summarised).
        system_message: Current system prompt; rebuilt after compression.
        approx_tokens: Pre-compression token estimate, logged for ops.
        task_id: Tool task scope (used for clearing file-read dedup state).
        focus_topic: Optional focus string for guided compression — the
            summariser will prioritise preserving information related to
            this topic.  Inspired by Claude Code's ``/compact <focus>``.
        force: If True, bypass any active summary-failure cooldown.  Set
            by the manual ``/compress`` slash command so users can retry
            immediately after an auto-compress abort.  Auto-compress
            callers use the default ``False``.

    Returns:
        ``(compressed_messages, new_system_prompt)`` tuple.  When
        compression aborts (aux LLM failed to produce a usable summary),
        returns the original messages unchanged and the existing system
        prompt — the session is NOT rotated.  Callers should detect the
        no-op via ``len(returned) == len(input)`` and stop the retry loop.
    """
    # Lazy feasibility check — run the auxiliary-provider probe + context
    # length lookup just-in-time on the first compression attempt instead of
    # at AIAgent.__init__. Saves ~400ms cold off every short session that
    # never reaches the threshold (the vast majority of ``chat -q`` runs).
    # The check itself sets ``agent._compression_warning`` so the
    # status-callback replay machinery still emits the warning to the user
    # the first time it would matter.
    if not getattr(agent, "_compression_feasibility_checked", False):
        # Mark as checked only after the probe completes. If the check
        # raises (e.g. a fatal aux-context ValueError that aborts the
        # session), leaving the flag unset is harmless; a non-fatal
        # transient failure is swallowed inside the function so the flag
        # is set normally on the next successful pass.
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    # In-place compaction (config: compression.in_place, see #38763). When True,
    # this compaction rewrites the message list + rebuilds the system prompt but
    # keeps the SAME session_id — no end_session, no parent_session_id child, no
    # `name #N` renumber, no contextvar/env/logging re-sync, no memory/context-
    # engine session-switch. The conversation keeps one durable id for life,
    # eliminating the session-rotation bug cluster. Default False during rollout.
    in_place = bool(getattr(agent, "compression_in_place", False))
    # Set True once the in-place DB write actually completes (the DB block can
    # raise and skip it). Surfaced to the gateway via agent._last_compaction_in_place.
    compacted_in_place = False
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    agent._emit_status(COMPACTION_STATUS)

    # ── Compression lock ────────────────────────────────────────────────
    # Atomic, state.db-backed lock per session_id.  Without this, two
    # AIAgent instances that share the same session_id (most commonly the
    # parent-turn agent and its background-review fork — see
    # ``agent/background_review.py``: ``review_agent.session_id =
    # agent.session_id``) can each call compress() on overlapping
    # snapshots of the same conversation.  Both succeed, both rotate
    # ``agent.session_id`` to a fresh id, both create child sessions in
    # state.db parented to the same old id.  The gateway's SessionEntry
    # only catches one rotation, so the other child becomes an orphan
    # that silently accumulates writes — Damien's repro shape.
    #
    # Acquire keyed on the OLD session_id (the rotation target's parent),
    # because that's the id that competing paths see and read from
    # SessionEntry at the start of their own compression attempt.
    #
    # If we can't acquire the lock, another path is mid-compression on
    # this session.  Aborting is correct: the messages are unchanged, the
    # other path's rotation will produce the canonical new session_id,
    # and our caller's auto-compress loop sees ``len(returned) == len(input)``
    # and stops retrying for this cycle. The session is NOT corrupted —
    # we just sit out this round and let the winner finish.
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _lock_holder: Optional[str] = None
    # Probe whether the lock subsystem is actually available on this
    # SessionDB instance.  A process running mismatched module versions
    # (e.g. ``conversation_compression.py`` reloaded after a pull but the
    # long-lived ``hermes_state.SessionDB`` class still bound to the
    # pre-#34351 version in memory) has the call site but not the method.
    # In that case ``try_acquire_compression_lock`` raises AttributeError —
    # NOT a ``sqlite3.Error`` — so the method's own fail-open guard never
    # runs and the exception propagates to the outer agent loop, which
    # prints the error and retries.  Because compression never succeeds,
    # the token count never drops and the loop re-triggers compaction
    # forever (the "API call #47/#48/#49 ... has no attribute
    # try_acquire_compression_lock" spin).  Fail OPEN here: if the lock
    # subsystem is missing or broken in any unexpected way, skip locking
    # and proceed with compression.  Skipping the lock risks a rare
    # concurrent-compression session fork; an infinite no-progress loop
    # that never compresses at all is strictly worse.
    try:
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    except (TypeError, ValueError):
        _lock_ttl = 300.0
    _lock_refresh_interval = getattr(agent, "_compression_lock_refresh_interval", None)
    _lock_refresher: Optional[_CompressionLockLeaseRefresher] = None
    if _lock_db is not None and _lock_sid:
        _lock_holder = _compression_lock_holder(agent)
        try:
            _lock_acquired = _lock_db.try_acquire_compression_lock(
                _lock_sid, _lock_holder, ttl_seconds=_lock_ttl
            )
        except Exception as _lock_err:
            # Broken/absent lock subsystem (version skew, etc.).  Log once
            # per session and proceed WITHOUT the lock rather than letting
            # the exception spin the outer loop.
            _lock_holder = None  # we don't own anything to release
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "(%s: %s) — proceeding without lock. This usually means a "
                    "stale in-memory module after an update; restart the "
                    "process (or `hermes update`) to resync.",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
            _lock_acquired = True  # treat as acquired-but-unlocked; proceed
        if not _lock_acquired:
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            _lock_holder = None  # don't release a lock we don't own
            # Surface to the user once — quiet for downstream auto-compress loops
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _lock_holder is not None:
            _lock_refresher = _CompressionLockLeaseRefresher(
                _lock_db,
                _lock_sid,
                _lock_holder,
                _lock_ttl,
                _lock_refresh_interval,
            ).start()

    def _release_lock() -> None:
        """Release the lock keyed on the OLD session_id (before rotation)."""
        if _lock_refresher is not None:
            _lock_refresher.stop()
        if _lock_db is not None and _lock_sid and _lock_holder:
            try:
                _lock_db.release_compression_lock(_lock_sid, _lock_holder)
            except Exception as _rel_err:
                logger.debug("compression lock release failed: %s", _rel_err)

    # Notify external memory provider before compression discards context
    if agent._memory_manager:
        try:
            agent._memory_manager.on_pre_compress(messages)
        except Exception:
            pass

    try:
        compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic, force=force)
    except TypeError:
        # Plugin context engine with strict signature that doesn't accept
        # focus_topic / force — fall back to calling without them.
        try:
            compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens)
        except BaseException:
            _release_lock()
            raise
    except BaseException:
        # ANY exception during compress() must release the lock so the
        # session isn't permanently blocked from future compression.
        _release_lock()
        raise

    # If compression aborted (aux LLM failed to produce a usable summary)
    # the compressor returns the input messages unchanged.  Surface the
    # error to the user, skip the session-rotation work entirely (no
    # session has logically ended), and let auto-compress callers detect
    # the no-op via len(returned) == len(input).
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        try:
            _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
            if getattr(agent, "_last_compression_summary_warning", None) != _err:
                agent._last_compression_summary_warning = _err
                agent._emit_warning(
                    f"⚠ Compression aborted: {_err}. "
                    "No messages were dropped — conversation continues unchanged. "
                    "Run /compress to retry, or /new to start a fresh session."
                )
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        finally:
            _release_lock()

    try:
        summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
                agent._last_compression_summary_warning = summary_error
                agent._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # No hard failure — but did the configured aux model error out
            # and get recovered by retrying on main?  Surface that so users
            # know their auxiliary.compression.model setting is broken even
            # though compression succeeded.
            _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                    agent._last_aux_fallback_warning_key = _aux_key
                    agent._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = agent._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        agent._invalidate_system_prompt()
        new_system_prompt = agent._build_system_prompt(system_message)
        agent._cached_system_prompt = new_system_prompt

        if agent._session_db:
            try:
                # Trigger memory extraction on the current session before the
                # transcript is rewritten (runs in BOTH modes — the logical
                # conversation's pre-compaction turns are about to be summarized
                # away regardless of whether the id rotates).
                agent.commit_memory_session(messages)

                if in_place:
                    # ── In-place compaction: keep the same session_id ──────────
                    # No end_session, no new row, no parent_session_id, no title
                    # renumber, no contextvar/env/logging re-sync. The session's
                    # id, title, cwd, /goal, and gateway routing all stay put.
                    #
                    # Durable, NON-DESTRUCTIVE replace: soft-archive the
                    # pre-compaction turns (active=0, kept on disk + FTS-searchable +
                    # recoverable) and insert `compressed` as the new live (active=1)
                    # set, atomically. `compressed` already carries the surviving
                    # tail (current-turn messages the compressor kept via
                    # protect_last_n), so we DON'T pre-flush here — a flush would
                    # INSERT current-turn rows that archive_and_compact would then
                    # archive alongside the rest (harmless but wasted writes). The
                    # live-context load filters active=1, so a resume reloads ONLY
                    # the compacted set; the original turns remain under the SAME id
                    # for search/recovery (Teknium review — keep one durable id
                    # WITHOUT destroying history, unlike a hard replace_messages).
                    # See #38763.
                    agent._session_db.archive_and_compact(agent.session_id, compressed)
                    # Reset the flush identity set so the next turn's appends are
                    # diffed against the COMPACTED transcript: the compacted dicts
                    # are passed as conversation_history next turn and skipped by
                    # identity, so only genuinely new turn messages get appended
                    # (no dup of the summary, no resurrection of dropped turns).
                    agent._flushed_db_message_ids = set()
                    # Rotation-independent signal: the conversation was compacted in
                    # place (id unchanged). The gateway reads this (NOT an id-change
                    # diff) to re-baseline transcript handling.
                    compacted_in_place = True
                else:
                    # ── Rotation (legacy): end this session, fork a continuation ─
                    # Flush any un-persisted current-turn messages to the OLD
                    # session before ending it, so they survive in the preserved
                    # parent transcript (#47202). (In-place skips this — see above.)
                    try:
                        agent._flush_messages_to_session_db(messages)
                    except Exception:
                        pass  # best-effort — don't block compression on a flush error
                    # Propagate title to the new session with auto-numbering
                    old_title = agent._session_db.get_session_title(agent.session_id)
                    agent._session_db.end_session(agent.session_id, "compression")
                    old_session_id = agent.session_id
                    agent.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                    # Ordering contract: the agent thread updates the contextvar here;
                    # the gateway propagates to SessionEntry after run_in_executor returns.
                    try:
                        from gateway.session_context import set_current_session_id

                        set_current_session_id(agent.session_id)
                    except Exception:
                        os.environ["HERMES_SESSION_ID"] = agent.session_id
                    # The gateway/tools session context (ContextVar + env) and the
                    # logging session context are SEPARATE mechanisms. The call above
                    # moves the former; the ``[session_id]`` tag on log lines comes
                    # from ``hermes_logging._session_context`` (set once per turn in
                    # conversation_loop.py). Without this, post-rotation log lines in
                    # the same turn keep the STALE old id while the message/DB/gateway
                    # state carry the new one — breaking log correlation exactly at the
                    # compaction boundary (see #34089). Guarded separately so a logging
                    # failure can never regress the routing update above.
                    try:
                        from hermes_logging import set_session_context

                        set_session_context(agent.session_id)
                    except Exception:
                        pass
                    agent._session_db_created = False
                    try:
                        agent._session_db.create_session(
                            session_id=agent.session_id,
                            source=agent.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                            model=agent.model,
                            model_config=agent._session_init_model_config,
                            parent_session_id=old_session_id,
                        )
                    except Exception as _cs_err:
                        # The child row could not be created (e.g. FK constraint,
                        # contended write). Previously the outer handler simply
                        # warned and let the agent continue on the NEW id — which
                        # has no row in state.db, producing an orphan: the parent
                        # is ended, the child is never indexed, and every
                        # subsequent message is attributed to a session that
                        # doesn't exist (#33906/#33907). Roll the live id back to
                        # the parent so the conversation stays attached to a real,
                        # indexed session instead of a phantom.
                        logger.warning(
                            "Compression child session create failed (%s) — "
                            "rolling back to parent session %s to avoid an orphan.",
                            _cs_err, old_session_id,
                        )
                        agent.session_id = old_session_id
                        try:
                            from gateway.session_context import set_current_session_id
                            set_current_session_id(agent.session_id)
                        except Exception:
                            os.environ["HERMES_SESSION_ID"] = agent.session_id
                        try:
                            from hermes_logging import set_session_context
                            set_session_context(agent.session_id)
                        except Exception:
                            pass
                        # Re-open the parent: it was ended above, but we're
                        # continuing on it, so it must not stay closed.
                        try:
                            agent._session_db.reopen_session(old_session_id)
                        except Exception:
                            pass
                        old_session_id = None  # no rotation happened
                        # The parent row already exists in state.db, so mark the
                        # session as created — _ensure_db_session would otherwise
                        # retry a (harmless INSERT OR IGNORE) create next turn.
                        agent._session_db_created = True
                        raise
                    agent._session_db_created = True
                    # Carry a persistent /goal onto the continuation session.
                    # Compression mints a fresh child id; load_goal does a flat
                    # per-session lookup with no parent walk, so without this an
                    # active goal silently dies at the boundary (#33618).
                    try:
                        from hermes_cli.goals import migrate_goal_to_session
                        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
                    except Exception as _goal_err:
                        logger.debug("Could not migrate goal on compression: %s", _goal_err)
                    # Auto-number the title for the continuation session
                    if old_title:
                        try:
                            new_title = agent._session_db.get_next_title_in_lineage(old_title)
                            agent._session_db.set_session_title(agent.session_id, new_title)
                        except (ValueError, Exception) as e:
                            logger.debug("Could not propagate title on compression: %s", e)

                # Shared post-write steps (both modes target agent.session_id, which
                # in-place keeps and rotation has already reassigned to the new id):
                # refresh the stored system prompt and reset the flush cursor so the
                # next turn re-bases its append diff.
                agent._session_db.update_system_prompt(agent.session_id, new_system_prompt)
                agent._last_flushed_db_idx = 0
            except Exception as e:
                # If the rotation rolled back to the parent (orphan-avoidance
                # above), agent.session_id is the still-indexed parent and
                # old_session_id was cleared — so this is recovery, not an
                # un-indexed orphan. Otherwise an earlier step failed before the
                # child was created and the warning's original meaning holds.
                if locals().get("old_session_id") is None and not in_place:
                    logger.warning(
                        "Compression rotation aborted and rolled back to the "
                        "parent session (%s): %s", agent.session_id or "?", e,
                    )
                else:
                    logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Compaction-boundary bookkeeping, computed once. `old_session_id` is only
        # bound in the rotation branch; in-place leaves it unset. `_boundary_parent`
        # is the id the boundary notifications attribute the prior state to: the old
        # id on rotation, the (unchanged) current id in-place.
        _old_sid = locals().get("old_session_id")
        _is_boundary = bool(_old_sid) or in_place
        _boundary_parent = _old_sid or agent.session_id or ""

        # Notify the context engine that a compaction boundary occurred. Plugin
        # engines (e.g. hermes-lcm) use boundary_reason="compression" to preserve
        # DAG lineage / checkpoint per-session state across the boundary instead of
        # re-initializing fresh. See hermes-lcm#68. Built-in ContextCompressor
        # ignores kwargs. Fires in BOTH modes: rotation passes old→new ids; in-place
        # passes the SAME id (the boundary is real even though the id didn't move).
        try:
            if _is_boundary and hasattr(agent.context_compressor, "on_session_start"):
                agent.context_compressor.on_session_start(
                    agent.session_id or "",
                    boundary_reason="compression",
                    old_session_id=_boundary_parent,
                    platform=getattr(agent, "platform", None) or "cli",
                    conversation_id=getattr(agent, "_gateway_session_key", None),
                )
        except Exception as _ce_err:
            logger.debug("context engine on_session_start (compression): %s", _ce_err)

        # Notify memory providers of the compaction boundary so provider-cached
        # per-session state (Hindsight's _document_id, accumulated turn buffers,
        # counters) refreshes. reset=False because the logical conversation
        # continues. See #6672. Fires in BOTH modes: in-place uses the same id as
        # parent (the conversation didn't fork, but the buffer must still be told
        # the transcript was compacted so it doesn't double-count dropped turns).
        try:
            if _is_boundary and agent._memory_manager:
                agent._memory_manager.on_session_switch(
                    agent.session_id or "",
                    parent_session_id=_boundary_parent,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # Keep the post-compression rough estimate for diagnostics, but do not
        # treat it as provider-reported prompt usage. Schema-heavy rough estimates
        # can remain above threshold even after the next real API request fits.
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=agent.tools or None,
        )

        # Record the anti-thrash effectiveness verdict at the REQUEST level (the
        # same level should_compress uses), apples-to-apples: pre = the request-level
        # estimate of the messages that triggered this compaction, post =
        # _compressed_est. This is the SINGLE owner of the counter for the normal
        # compaction path and fixes the 2026-06-19 thrash where compress()'s
        # messages-only verdict reset the counter on every pass (the
        # 205,072 -> 297,723 "tokens went UP" case). Use the pre-compaction request
        # estimate of the ORIGINAL messages so a failed/placeholder summary that
        # leaves the request over threshold is correctly counted as ineffective.
        try:
            _pre_request_est = estimate_request_tokens_rough(
                messages,
                system_prompt=system_message or "",
                tools=agent.tools or None,
            )
            if hasattr(agent.context_compressor, "record_compaction_effectiveness"):
                agent.context_compressor.record_compaction_effectiveness(
                    pre_request_tokens=_pre_request_est,
                    post_request_tokens=_compressed_est,
                )
        except Exception as _eff_err:
            logger.debug("record_compaction_effectiveness failed: %s", _eff_err)

        agent.context_compressor.last_compression_rough_tokens = _compressed_est
        agent.context_compressor.last_prompt_tokens = -1
        agent.context_compressor.last_completion_tokens = 0
        agent.context_compressor.awaiting_real_usage_after_compression = True

        # Warn on repeated compressions (quality degrades with each pass).
        # Route through _emit_status (like the other compression warnings above)
        # so the warning reaches the TUI / Telegram / Discord via status_callback,
        # not just CLI stdout. _emit_status still _vprints for the CLI, and
        # storing it on _compression_warning lets replay_compression_warning
        # re-deliver it once a late-bound gateway status_callback is wired (#36908).
        _cc = agent.context_compressor.compression_count
        if _cc >= 2:
            _cc_msg = (
                f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh."
            )
            agent._compression_warning = _cc_msg
            agent._emit_status(_cc_msg)

        # Emit session:compress event so hooks (e.g. MemPalace sync) can ingest
        # the completed old session before its details are lost. In in-place mode
        # there is no old id (same session); ``in_place=True`` tells hooks the
        # transcript was compacted on the same id rather than rotated.
        if getattr(agent, "event_callback", None):
            try:
                agent.event_callback("session:compress", {
                    "platform": agent.platform or "",
                    "session_id": agent.session_id,
                    "old_session_id": _old_sid or "",
                    "in_place": in_place,
                    "compression_count": agent.context_compressor.compression_count,
                })
            except Exception as e:
                logger.debug("event_callback error on session:compress: %s", e)

        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=true",
            agent.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )

        # ── In-chat compaction announce (engine-aware; additive to fallback) ──────
        # Emitted out-of-band via _emit_status (persistent chat line, never injected
        # into model history). Engine-correct recovery reference: built-in → session
        # pointer; LCM → lossless-store + lcm_grep/lcm_expand guidance. Gating is the
        # allow-list in _format_compaction_announce. Runs INSIDE the lock hold (the
        # _release_lock() below) so the dedupe read-then-write is serialized.
        try:
            _cc = agent.context_compressor
            _engine_name = getattr(_cc, "name", None)
            _status = getattr(_cc, "_last_compression_status", None)
            _old_sid = locals().get("old_session_id")
            _new_sid = agent.session_id
            if _engine_name == "lcm":
                _dedupe_key = ("lcm", getattr(_cc, "compression_count", None))
            else:
                _dedupe_key = ("builtin", (_old_sid, _new_sid))
            _now_mono = time.monotonic()
            _after_fb, _win_from, _win_to = _compaction_after_fallback(
                agent,
                now_monotonic=_now_mono,
                current_turn_id=getattr(agent, "_current_turn_id", None),
            )
            # Granular stats (in-turn population = the whole messages list; cleared=0).
            # Built inside try/except; validate()+degrade so a reconcile bug never
            # ships wrong math or breaks the turn. Guarded by hasattr so built-in /
            # overflow / manual paths (no LCM marker shape) simply degrade.
            _inturn_stats = None
            try:
                from agent.compaction_stats import build_inturn_stats
                from agent.model_metadata import estimate_messages_tokens_rough as _est
                _why2 = "build raised"  # bound before build so the warning %s can't be unbound
                _cand = build_inturn_stats(
                    messages=messages,
                    compressed=compressed,
                    estimator=_est,
                    engine_is_lcm=(_engine_name == "lcm"),
                    sanitize=getattr(_cc, "_sanitize_active_context_messages", None),
                    fresh_tail_count=getattr(_cc, "protect_last_n", None),
                    on_tag_missing=lambda: _warn_compaction_stats_once(
                        agent, "COMPACTION_STATS_TAG_MISSING in-turn"
                    ),
                )
                _ok2, _why2 = _cand.validate()
                if _ok2:
                    # A-floor (approx_attribution) reconciles by construction but its
                    # kept/folded SPLIT is signature-approximate. The split error is
                    # bounded by the kept-tail fraction (the folded bulk is a contiguous
                    # prefix and always classifies correctly), so a kept-tail that is a
                    # large fraction of pre is the only case where the displayed split
                    # could be materially wrong. Degrade THAT render to two-line when the
                    # kept tail exceeds the gross-error threshold; otherwise show the
                    # granular split LABELED approximate + emit the observability marker.
                    if getattr(_cand, "approx_attribution", False):
                        # Gross-error magnitude = the RAW kept-tail size
                        # (estimator(messages[-fresh_tail_count:]) — match- AND
                        # sanitize-independent). kept_tokens (comp-side) is stripped small
                        # on a heavily-sanitized tail and _kept_pre_tokens is 0 when the
                        # signature match fails, so BOTH can under-report the true raw tail
                        # (Greptile P1 ×2, PR #109). Use raw_tail_tokens as the primary
                        # bound, with the other two as a floor in case it's unavailable.
                        _gross_tok = max(
                            _cand.raw_tail_tokens or 0,
                            _cand.kept_tokens or 0,
                            _cand._kept_pre_tokens or 0,
                        )
                        _pre_tok = _cand.pre_tokens or 0
                        _gross_frac = (_gross_tok / _pre_tok) if _pre_tok > 0 else 0.0
                        if _gross_frac > _APPROX_GROSS_MAX_FRAC:
                            # split could be materially wrong → honest two-line degrade
                            _warn_compaction_stats_once(
                                agent,
                                f"COMPACTION_STATS_APPROX_ATTRIBUTION in-turn "
                                f"degraded (kept_tail {_gross_tok} / pre {_pre_tok} "
                                f"= {_gross_frac:.1%} > {_APPROX_GROSS_MAX_FRAC:.0%}); two-line",
                            )
                            _inturn_stats = None
                        else:
                            _inturn_stats = _cand
                            # observability: the floor produced the numbers (not exact
                            # alignment / engine record). A heavy LCM session running the
                            # floor is now visible (watcher rate-alerts), never silent.
                            _warn_compaction_stats_once(
                                agent,
                                f"COMPACTION_STATS_APPROX_ATTRIBUTION in-turn "
                                f"(engine={_engine_name}; kept_tail {_gross_tok} / "
                                f"pre {_pre_tok} = {_gross_frac:.1%})",
                            )
                    else:
                        _inturn_stats = _cand
                else:
                    _warn_compaction_stats_once(
                        agent, f"COMPACTION_STATS_RECONCILE_FAILED in-turn {_why2}"
                    )
            except Exception:
                _warn_compaction_stats_once(
                    agent, "COMPACTION_STATS_BUILD_FAILED in-turn", exc_info=True
                )
            _reasoning_inturn = None
            try:
                from gateway.run import _load_gateway_config as _lgc
                _ac = (_lgc().get("agent") or {})
                _reasoning_inturn = str(_ac.get("reasoning_effort", "") or "").strip() or None
            except Exception:
                _reasoning_inturn = None
            _emit_compaction_announce(
                agent,
                dedupe_key=_dedupe_key,
                engine_name=_engine_name,
                status=_status,
                old_session_id=_old_sid,
                new_session_id=_new_sid,
                old_messages=_pre_msg_count,
                new_messages=len(compressed),
                pre_tokens=locals().get("_pre_request_est"),
                post_tokens=_compressed_est,
                model=getattr(agent, "model", None),
                provider=getattr(agent, "provider", None),
                window_from=_win_from,
                window_to=_win_to,
                summary_snippet=_extract_compaction_summary_snippet(compressed),
                raw_store_count=None,  # session-scoped count not cheap here; omit (N-NEW-3)
                after_fallback=_after_fb,
                trigger_reason=trigger_reason,
                trigger_value=(
                    getattr(_cc, "threshold_tokens", None)
                    if trigger_reason == "threshold" else None
                ),
                reasoning=_reasoning_inturn,
                stats=_inturn_stats,
                in_place=in_place,
            )
        except Exception:
            logger.debug("compaction announce skipped (non-fatal)", exc_info=True)

        # ── Option B provenance strip (load-bearing, MUST NOT be skipped) ──────────
        # The engine stamps ``_src_idx`` on kept rows so build_inturn_stats (above) can
        # read the EXACT pre-side partition. It MUST NOT reach the wire / prompt cache /
        # transcript (``compressed`` becomes the new session transcript), so strip it
        # here — the single point on the only path where ``compressed`` carries it (the
        # early abort/noop returns return the original ``messages``, never stamped).
        # Done inline (no import that could fail and silently leave the key — Greptile
        # #110); idempotent; the transport sanitizer also drops ``_``-prefixed keys as a
        # defense-in-depth backstop.
        for _m in compressed:
            if isinstance(_m, dict) and "_src_idx" in _m:
                try:
                    del _m["_src_idx"]
                except Exception:
                    _m.pop("_src_idx", None)

        # Surface the compaction mode to the caller (run_conversation / gateway)
        # via a rotation-independent flag. The gateway uses this — NOT an
        # id-change diff — to re-baseline transcript handling (history_offset=0 +
        # rewrite on the same id) when compaction happened in place. See #38763.
        agent._last_compaction_in_place = compacted_in_place

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        return compressed, new_system_prompt
    finally:
        # Release the lock on the OLD session_id only AFTER rotation completed
        # and all post-rotation bookkeeping (memory manager, context engine,
        # file dedup) ran. A concurrent path that wakes up the moment we
        # release will see the NEW session_id in state.db / SessionEntry and
        # acquire on that — no race against our just-finished work.
        _release_lock()


def try_shrink_image_parts_in_messages(
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Re-encode all native image parts at a smaller size to recover from
    image-too-large errors (Anthropic 5 MB, unknown other providers).

    Mutates ``api_messages`` in place. Returns True if any image part was
    actually replaced, False if there were no image parts to shrink or
    Pillow couldn't help (caller should surface the original error).

    Strategy: look for ``image_url`` / ``input_image`` parts carrying a
    ``data:image/...;base64,...`` payload, plus Anthropic-native
    ``{"type": "image", "source": {"type": "base64", ...}}`` blocks.
    For each one whose encoded size exceeds 4 MB (a safe target that slides
    under Anthropic's 5 MB ceiling with header overhead) or whose longest side
    exceeds ``max_dimension``, write the base64 to a tempfile, call
    ``vision_tools._resize_image_for_vision`` to produce a smaller data
    URL, and substitute it in place.

    Non-data-URL images (http/https URLs) are not touched — the provider
    fetches those itself and the size limit is different.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
    # Non-Anthropic providers we haven't observed rejecting are fine with
    # much larger; shrinking to 4 MB here loses quality but only fires
    # after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic enforces an 8000px per-side dimension cap independently of
    # the 5 MB byte cap.  In many-image requests, the provider can report a
    # lower cap (observed: 2000px).  The caller passes that parsed ceiling
    # when the rejection includes it.
    changed_count = 0
    # Track parts that are over the target but could NOT be shrunk under it.
    # If any survive, retrying is pointless — the same oversized payload will
    # be re-sent and rejected again, wasting the single retry budget.  We only
    # report success (caller retries) when every over-threshold image was
    # actually brought under the target.
    unshrinkable_oversized = 0

    def _decode_pixels(data_url: str) -> Optional[tuple]:
        """Return ``(width, height)`` of a base64 data URL, or None on failure.

        Soft-depends on Pillow; returns None (caller falls back to a
        bytes-only check) if Pillow is missing or the payload is corrupt.
        """
        try:
            import base64 as _b64_dim
            import io as _io_dim
            header_d, _, data_d = data_url.partition(",")
            if not data_d or not data_url.startswith("data:"):
                return None
            from PIL import Image as _PILImage
            with _PILImage.open(_io_dim.BytesIO(_b64_dim.b64decode(data_d))) as _img:
                return _img.size
        except Exception:
            return None

    def _shrink_data_url(url: str) -> tuple:
        """Return ``(resized_url, unshrinkable)`` for a data URL.

        ``resized_url`` is a smaller/dimension-correct data URL, or None when
        no rewrite was applied.  ``unshrinkable`` is True only when the image
        exceeded a constraint (byte-size or dimensions) and the resize failed
        to satisfy *that same* constraint — so the caller knows retrying is
        pointless even if a different image in the request shrank.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, False

        # Determine which constraint is binding.  The accept/reject gate below
        # MUST be checked against the same axis that triggered the shrink: a
        # downscaled screenshot PNG routinely re-encodes to *more* bytes than
        # the original (PNG compression is non-monotonic in image size — a
        # smaller raster with LANCZOS resampling noise compresses worse than a
        # larger smooth one).  Rejecting a pixel-correct downscale purely
        # because its bytes grew permanently wedges sessions on the Anthropic
        # many-image 2000px path (#48013).
        needs_shrink = len(url) > target_bytes  # over byte budget
        triggered_by = "bytes" if needs_shrink else None
        if not needs_shrink:
            # Bytes are fine — check pixel dimensions against the provider's
            # reported per-side cap.  A screenshot can be tiny in bytes yet
            # too large in pixels.
            dims = _decode_pixels(url)
            if dims is None:
                # Pillow missing or corrupt data — fall back to byte-only.
                return None, False
            if max(dims) <= max_dimension:
                return None, False  # both bytes and pixels are within limits
            needs_shrink = True
            triggered_by = "dimension"

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized:
                # Resize returned nothing — Pillow couldn't help.
                return None, True
            if triggered_by == "bytes":
                # Byte budget is the binding constraint — bytes must shrink.
                if len(resized) >= len(url):
                    return None, True  # re-encode made it bigger
                # The per-side dimension cap is ALSO an active provider
                # constraint on this request (the caller passes the parsed cap
                # to both this helper and the resizer).  _resize_image_for_vision
                # returns a best-effort, possibly-over-cap blob when it
                # exhausts its halving budget — it freezes the long side once
                # the short side hits its 64px floor, so a very-high-aspect
                # image can stay over the cap even after bytes shrank.  If the
                # output is still over the cap, retrying would re-400 on
                # dimensions; treat it as unshrinkable.  (Skip when dims can't
                # be decoded — preserves historical byte-only behaviour.)
                new_dims = _decode_pixels(resized)
                if new_dims is not None and max(new_dims) > max_dimension:
                    return None, True
                return resized, False
            # triggered_by == "dimension": the per-side cap is binding.  The
            # re-encode may have grown in bytes; accept it as long as it is now
            # within the dimension cap.  Verify the new dimensions when we can.
            new_dims = _decode_pixels(resized)
            if new_dims is not None:
                if max(new_dims) <= max_dimension:
                    return resized, False
                # Still over the per-side cap — the resize didn't satisfy it.
                return None, True
            # Couldn't verify the re-encode's dimensions (corrupt output or
            # Pillow gone mid-call).  Fall back to the historical "bytes must
            # shrink" gate so we never accept an unverifiable, byte-larger blob.
            if len(resized) >= len(url):
                return None, True
            return resized, False
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None, triggered_by is not None

    def _source_to_data_url(source: Any) -> Optional[str]:
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = str(source.get("media_type") or "image/jpeg").strip()
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return f"data:{media_type};base64,{data}"

    def _write_data_url_to_source(source: dict, data_url: str) -> None:
        header, _, data = data_url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            candidate = header[len("data:"):].split(";", 1)[0].strip()
            if candidate.startswith("image/"):
                media_type = candidate
        source["type"] = "base64"
        source["media_type"] = media_type
        source["data"] = data

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                source = part.get("source")
                url = _source_to_data_url(source)
                resized, unshrinkable = _shrink_data_url(url or "")
                if resized and isinstance(source, dict):
                    _write_data_url_to_source(source, resized)
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
                continue
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized, unshrinkable = _shrink_data_url(url)
                if resized:
                    image_value["url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized, unshrinkable = _shrink_data_url(image_value)
                if resized:
                    part["image_url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # At least one oversized image could not be shrunk under the target.
        # Retrying would re-send it and fail identically, so signal "no
        # progress" even if other parts shrank — the caller will surface the
        # original error rather than burning its single retry on a no-op.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS",
    "COMPACTION_STATUS_MARKER",
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
