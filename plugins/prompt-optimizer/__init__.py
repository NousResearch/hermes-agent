"""prompt-optimizer plugin — model-aware prompt rewrite + metrics + coaching.

Wires four behaviours:

1. ``pre_gateway_dispatch`` hook — intercepts incoming user messages, runs
   the optimizer engine (rewrite for token efficiency + model-aware terminology),
   scores before/after, stores metrics, and returns ``{"action": "rewrite"}``.

2. ``pre_user_message`` hook — intercepts CLI/TUI messages before agent sees them.

3. ``transform_llm_output`` hook — appends a lightweight inline badge to
   every assistant response when a rewrite occurred this turn.

4. ``/prompt-stats`` and ``/prompt-optimizer`` slash commands.

Dual mode:
  auto        — silent rewrite, no prompt
  interactive — show diff, ask user to approve
  off         — pass through untouched (still records baseline)

Bypass prefixes: /quick, *simple, #basic — skip optimization entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, Tuple

from .engine import (
    BYPASS_PREFIXES,
    OPTIMIZER_TIMEOUT_S,
    is_skill_invocation,
    is_structured_command,
    _analytics_rows,
    _build_full_report,
    _clip,
    _coerce_text,
    _delta_words,
    _display_bucket,
    _ensure_db,
    _fetch_rewrites,
    _format_diff,
    _limit_from_args,
    _normalise_args,
    _period_start,
    _quality_delta,
    _record,
    _render_analytics,
    _render_comparisons,
    _render_header,
    _render_insight_bullets,
    _render_suggestion_rows,
    _render_summary_cards,
    _run_optimizer,
    _summary_for_period,
    _wrap_block,
    _write_html_report,
    RewriteRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bridge state (will be refactored in later tasks)
# ---------------------------------------------------------------------------

# Trust config: ctx.llm reference stored at register() time
_ctx: Any = None

# Mode: auto | interactive | off
_DEFAULT_MODE = "auto"
_mode: str = _DEFAULT_MODE
_mode_lock = threading.Lock()

# Per-session rewrite state (session_id → RewriteRecord)
_session_rewrites: Dict[str, RewriteRecord] = {}
_rewrite_lock = threading.Lock()

# Interactive mode pending approvals (session_id → (record, timestamp))
_pending_approvals: Dict[str, Tuple[RewriteRecord, float]] = {}
_pending_lock = threading.RLock()   # reentrant: _check_pending_expiry called while holding it
_PENDING_TIMEOUT_S = 120


def _run_optimizer_bridge(original, model, provider):
    """Bridge to the engine optimizer using the plugin's ctx.llm."""
    if _ctx is None:
        logger.debug("prompt-optimizer: _ctx not set — bypassing")
        return None
    return _run_optimizer(original, model, provider, _ctx.llm)


# ---------------------------------------------------------------------------
# Gateway hook
# ---------------------------------------------------------------------------

def _check_pending_expiry(sid):
    """If sid has a pending approval past timeout, auto-approve and clear it."""
    with _pending_lock:
        if sid in _pending_approvals:
            record, created_at = _pending_approvals[sid]
            if time.time() - created_at > _PENDING_TIMEOUT_S:
                with _rewrite_lock:
                    _session_rewrites[sid] = record
                del _pending_approvals[sid]
                return record
    return None


def _send_diff_via_gateway(sid, record, gateway, event):
    """Send the diff to the user via the gateway adapter."""
    try:
        diff_text = _format_diff(record)
        source = getattr(event, 'source', None)
        if source is not None:
            chat_id = getattr(source, 'chat_id', None)
            platform = getattr(source, 'platform', 'telegram')
        else:
            logger.info("prompt-optimizer: interactive diff (no chat_id): %s", diff_text)
            return
        if gateway is not None and chat_id is not None:
            adapters = getattr(gateway, 'adapters', None)
            if adapters is not None:
                adapter = adapters.get(platform) if hasattr(adapters, 'get') else adapters
                if adapter is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(adapter.send(chat_id, diff_text))
                        else:
                            loop.run_until_complete(adapter.send(chat_id, diff_text))
                    except RuntimeError:
                        asyncio.run(adapter.send(chat_id, diff_text))
                    return
        logger.info("prompt-optimizer: interactive diff: %s", diff_text)
    except Exception as e:
        logger.warning("prompt-optimizer: failed to send diff: %s", e)


def _on_pre_gateway_dispatch(event=None, gateway=None, session_store=None, **kw):
    """Intercept gateway messages, optionally rewrite."""
    global _mode
    with _mode_lock:
        mode = _mode
    if mode == "off":
        return None
    if event is None:
        return None
    original = getattr(event, "text", None)
    if not isinstance(original, str) or not original.strip():
        return None
    stripped = original.strip()

    if any(stripped.startswith(p) for p in BYPASS_PREFIXES):
        return None

    if is_skill_invocation(stripped):
        logger.info("prompt-optimizer: skipped — skill invocation")
        return None

    if is_structured_command(stripped):
        logger.info("prompt-optimizer: skipped — structured command")
        return None

    # Slash commands (Discord, Telegram, etc.) are routing instructions to
    # the gateway, not prompts to optimise. Bypass anything that looks like
    # one: starts with '/', first whitespace-delimited word has no further
    # slashes (so file paths like '/Users/foo.md what's this?' still get
    # optimised). Mirrors hermes core's `_looks_like_slash_command`.
    if stripped.startswith("/"):
        first_word = stripped.split(None, 1)[0]
        if "/" not in first_word[1:]:
            return None

    model = ""
    provider = ""
    session_id = ""
    platform = "gateway"
    try:
        if gateway is not None:
            model = getattr(gateway, "model", "") or ""
            provider = getattr(gateway, "provider", "") or ""
        if session_store is not None:
            session_id = getattr(session_store, "session_id", "") or ""
    except Exception:
        pass

    sid = session_id or "unknown"

    if mode == "interactive":
        lowered = stripped.lower()

        with _pending_lock:
            has_pending = sid in _pending_approvals

        if has_pending:
            expired_record = _check_pending_expiry(sid)
            if expired_record is None:
                if lowered in ("y", "yes"):
                    with _pending_lock:
                        record, _ = _pending_approvals.pop(sid)
                    with _rewrite_lock:
                        _session_rewrites[sid] = record
                    _record(sid, platform, record.original, record.rewritten,
                            record.quality_before, record.quality_after,
                            record.token_delta_pct, record.model_profile,
                            model, mode, approved=True)
                    return {"action": "rewrite", "text": record.rewritten}
                elif lowered in ("n", "no"):
                    with _pending_lock:
                        record, _ = _pending_approvals.pop(sid)
                    _record(sid, platform, record.original, record.original,
                            record.quality_before, record.quality_after,
                            record.token_delta_pct, record.model_profile,
                            model, mode, approved=False)
                    return {"action": "rewrite", "text": record.original}
                else:
                    with _pending_lock:
                        old_record, _ = _pending_approvals.pop(sid)
                    with _rewrite_lock:
                        _session_rewrites[sid] = old_record
                    _record(sid, platform, old_record.original, old_record.rewritten,
                            old_record.quality_before, old_record.quality_after,
                            old_record.token_delta_pct, old_record.model_profile,
                            model, mode, approved=True)

        record = _run_optimizer_bridge(original, model, provider)
        if record is None:
            return None
        with _pending_lock:
            _pending_approvals[sid] = (record, time.time())
        _send_diff_via_gateway(sid, record, gateway, event)
        return {"action": "skip", "reason": "interactive-diff-shown"}
    else:
        record = _run_optimizer_bridge(original, model, provider)
        if record is None:
            return None

        with _rewrite_lock:
            _session_rewrites[sid] = record
        _record(sid, platform, record.original, record.rewritten,
                record.quality_before, record.quality_after,
                record.token_delta_pct, record.model_profile,
                model, mode, approved=True)
        logger.info("prompt-optimizer: rewrote %r → %r",
                    original[:60], record.rewritten[:60])
        return {"action": "rewrite", "text": record.rewritten}


# ---------------------------------------------------------------------------
# CLI hook
# ---------------------------------------------------------------------------

def _on_pre_user_message(message="", session_id="", platform="cli", **kw):
    """Intercept CLI messages, optionally rewrite.

    Skipped on the TUI surface because the TUI client owns the
    optimisation flow via the ``prompt.optimize.preview`` RPC — by the
    time a message reaches ``prompt.submit`` (and therefore this hook)
    the user has already made an accept/reject/edit decision in the
    overlay. Running the optimiser a second time here would re-show the
    diff and stall the turn.
    """
    global _mode
    if platform == "tui":
        return None
    with _mode_lock:
        mode = _mode
    if mode == "off":
        return None
    if not isinstance(message, str) or not message.strip():
        return None
    stripped = message.strip()
    if any(stripped.startswith(p) for p in BYPASS_PREFIXES):
        return None

    if is_skill_invocation(stripped):
        logger.info("prompt-optimizer: skipped — skill invocation")
        return None

    if is_structured_command(stripped):
        logger.info("prompt-optimizer: skipped — structured command")
        return None

    # Target model comes via the pre_user_message hook kwargs from
    # conversation_loop.py. Used by resolve_model_profile() to tailor the
    # rewrite. Provider isn't passed by this hook today; leave blank.
    model = kw.get("model", "") or ""
    provider = ""

    if mode == "interactive":
        lowered = stripped.lower()
        sid = session_id or "unknown"

        with _pending_lock:
            has_pending = sid in _pending_approvals

        if has_pending:
            expired_record = _check_pending_expiry(sid)
            if expired_record is None:
                if lowered in ("y", "yes"):
                    with _pending_lock:
                        record, _ = _pending_approvals.pop(sid)
                    with _rewrite_lock:
                        _session_rewrites[sid] = record
                    _record(sid, platform, record.original, record.rewritten,
                            record.quality_before, record.quality_after,
                            record.token_delta_pct, record.model_profile,
                            model, mode, approved=True)
                    return {"action": "rewrite", "text": record.rewritten}
                elif lowered in ("n", "no"):
                    with _pending_lock:
                        record, _ = _pending_approvals.pop(sid)
                    _record(sid, platform, record.original, record.original,
                            record.quality_before, record.quality_after,
                            record.token_delta_pct, record.model_profile,
                            model, mode, approved=False)
                    return {"action": "rewrite", "text": record.original}
                else:
                    with _pending_lock:
                        old_record, _ = _pending_approvals.pop(sid)
                    with _rewrite_lock:
                        _session_rewrites[sid] = old_record
                    _record(sid, platform, old_record.original, old_record.rewritten,
                            old_record.quality_before, old_record.quality_after,
                            old_record.token_delta_pct, old_record.model_profile,
                            model, mode, approved=True)

        record = _run_optimizer_bridge(message, model, provider)
        if record is None:
            return None
        return _cli_interactive_approval(sid, record)
    else:
        record = _run_optimizer_bridge(message, model, provider)
        if record is None:
            return None

        sid = session_id or "unknown"
        with _rewrite_lock:
            _session_rewrites[sid] = record
        _record(sid, platform, record.original, record.rewritten,
                record.quality_before, record.quality_after,
                record.token_delta_pct, record.model_profile,
                model, mode, approved=True)
        logger.info("prompt-optimizer: rewrote %r", message[:60])
        return {"action": "rewrite", "text": record.rewritten}


def _build_overlay_question(record):
    """Render the diff for the AskUserQuestion overlay.

    Avoids ``_format_diff`` because that targets the message-based flow
    (200-char clipping, trailing 'Reply Y / N' hint). Here we want full
    text with paragraph separation so the arrow-key overlay reads well.
    """
    if record.token_delta_pct > 0:
        token_msg = f"saved {abs(record.token_delta_pct):.0f}%"
    elif record.token_delta_pct < 0:
        token_msg = f"used +{abs(record.token_delta_pct):.0f}%"
    else:
        token_msg = "no token change"

    q_diff = record.quality_after - record.quality_before

    return (
        f"Prompt Optimizer\n"
        f"Quality: {record.quality_before:.0f} \N{RIGHTWARDS ARROW} "
        f"{record.quality_after:.0f} ({q_diff:+.0f})    "
        f"Tokens: {token_msg}\n"
        f"\n"
        f"── Original ────────────────────────────────────────\n"
        f"{record.original.strip()}\n"
        f"\n"
        f"── Rewritten ───────────────────────────────────────\n"
        f"{record.rewritten.strip()}\n"
        f"\n"
        f"Use the rewritten prompt?"
    )


def _cli_interactive_approval(sid, record):
    """Show the diff via the CLI's AskUserQuestion-style overlay.

    Prefers ``ctx.ask_user(question, choices)`` — the same arrow-key
    overlay the ``clarify`` tool uses. Blocks the agent thread while the
    user navigates; returns once they pick a choice or it times out.

    Falls back to the message-based skip flow when no overlay is
    available (gateway/Discord, headless scripts, tests) — diff is
    printed, pending rewrite stashed, current turn aborted via
    ``{"action": "skip"}``; the user's next message resolves it.
    """
    if _ctx is not None and hasattr(_ctx, "ask_user"):
        question = _build_overlay_question(record)
        choice = _ctx.ask_user(question, ["accept", "reject"])
        if choice in ("accept", "reject"):
            approved = choice == "accept"
            text = record.rewritten if approved else record.original
            if approved:
                with _rewrite_lock:
                    _session_rewrites[sid] = record
            _record(
                sid, "cli", record.original, record.rewritten,
                record.quality_before, record.quality_after,
                record.token_delta_pct, record.model_profile,
                "", "interactive", approved=approved,
            )
            return {"action": "rewrite", "text": text}
        # choice is None — overlay unavailable or timed out. Fall through
        # to the message-based flow below.

    diff_text = _format_diff(record)
    print(
        f"\n{diff_text}\n\n"
        "Reply 'y' to use the rewrite, 'n' to keep the original."
    )
    with _pending_lock:
        _pending_approvals[sid] = (record, time.time())
    return {"action": "skip", "reason": "awaiting_prompt_optimizer_approval"}


def _on_transform_llm_output(text="", response_text="", session_id="", **kw):
    """Append inline badge when a rewrite happened this turn."""
    if not session_id:
        return None
    with _rewrite_lock:
        record = _session_rewrites.pop(session_id, None)
    if record is None:
        return None
    base = _coerce_text(response_text) or _coerce_text(text)
    sign = "+" if record.token_delta_pct > 0 else ""
    badge = (
        f"\n\n---\n"
        f"Optimized · {sign}{record.token_delta_pct}% tokens · "
        f"quality {record.quality_before:.0f}\N{RIGHTWARDS ARROW}{record.quality_after:.0f} · "
        f"/prompt-insights"
    )
    return base + badge


# ---------------------------------------------------------------------------
# Slash commands and reports
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
/prompt-optimizer — toggle optimisation mode

Subcommands:
  auto         Enable automatic silent rewriting (default)
  interactive Show diff and ask for approval before rewriting
  off          Disable prompt optimisation entirely
  status       Show current mode and session stats

Reports:
  /prompt-insights [--html]       Executive report: comparisons, suggestions, insights, analytics
  /prompt-compare [--limit N]     Before/after prompt comparisons
  /prompt-suggestions [--limit N] Prompt replacement suggestions
  /prompt-analytics [period]      Daily, weekly, monthly analytics
  /prompt-stats [flags]           Compact legacy stats

Examples:
  /prompt-insights
  /prompt-insights --html
  /prompt-compare --limit 5
  /prompt-analytics monthly
"""


def _handle_prompt_optimizer(raw_args):
    global _mode
    argv = _normalise_args(raw_args)
    if not argv or argv[0] in {"help", "-h", "--help"}:
        return _HELP_TEXT
    sub = argv[0].lower()
    if sub in {"auto", "interactive", "off"}:
        with _mode_lock:
            old = _mode
            _mode = sub
        return f"Prompt optimizer mode: {old} \N{RIGHTWARDS ARROW} {sub}\n\nRun /prompt-insights to view recent prompt rewrites."
    if sub == "status":
        with _mode_lock:
            mode = _mode
        total = _summary_for_period("all")["count"]
        return f"Prompt Optimizer\n  Mode: {mode}\n  Stored rewrites: {total}\n  Pending badges: {len(_session_rewrites)}\n\nReports: /prompt-insights, /prompt-compare, /prompt-suggestions, /prompt-analytics"
    return f"Unknown subcommand: {sub}\n\n{_HELP_TEXT}"


def _handle_prompt_insights(raw_args):
    argv = _normalise_args(raw_args)
    limit = _limit_from_args(argv, default=5, maximum=12)
    report = _build_full_report(limit=limit)
    if "--html" in argv or "html" in argv:
        path = _write_html_report(report)
        return report + f"\n\nHTML report\n  file://{path}"
    return report


def _handle_prompt_compare(raw_args):
    argv = _normalise_args(raw_args)
    limit = _limit_from_args(argv, default=5, maximum=20)
    rows = _fetch_rewrites(limit=limit, since=0, approved_only=False)
    return "\n".join(_render_header("Prompt Comparisons", f"Latest {len(rows)} rewrites") + _render_comparisons(rows))


def _handle_prompt_suggestions(raw_args):
    argv = _normalise_args(raw_args)
    limit = _limit_from_args(argv, default=8, maximum=20)
    rows = _fetch_rewrites(limit=50, since=_period_start("monthly"), approved_only=True)
    rows = sorted(rows, key=lambda r: (_quality_delta(r), float(r.get("token_delta_pct") or 0)), reverse=True)[:limit]
    return "\n".join(_render_header("Prompt Suggestions", "Reusable replacements from recent optimisation data") + _render_suggestion_rows(rows))


def _handle_prompt_analytics(raw_args):
    argv = _normalise_args(raw_args)
    period = next((a for a in argv if not a.startswith("-")), "all")
    if period not in {"all", "daily", "day", "today", "weekly", "week", "monthly", "month"}:
        period = "all"
    if period in {"day", "today"}:
        period = "daily"
    if period == "week":
        period = "weekly"
    if period == "month":
        period = "monthly"
    return _render_analytics(period)


def _handle_prompt_stats(raw_args):
    argv = _normalise_args(raw_args)
    week_mode = "--week" in argv
    today_mode = "--today" in argv
    best_mode = "--best" in argv
    raw_mode = "--raw" in argv
    if raw_mode:
        payload = {
            "today": _summary_for_period("daily"),
            "week": _summary_for_period("weekly"),
            "month": _summary_for_period("monthly"),
        }
        return json.dumps(payload, indent=2)
    if today_mode:
        return _render_analytics("daily")
    if week_mode:
        return _render_analytics("weekly")
    if best_mode:
        return _handle_prompt_suggestions("--limit 5")
    return _handle_prompt_insights("")


def get_tui_preview(session_key: str, text: str, model: str = "",
                    provider: str = "") -> Dict[str, Any]:
    """Bridge function for TUI gateway RPC — returns structured preview or bypass.

    Returns dict with keys: status, preview (optional), reason (optional).
    Importable by tui_gateway/server.py.
    """
    global _mode
    with _mode_lock:
        mode = _mode
    if mode == "off":
        return {"status": "bypass", "reason": "disabled"}
    if not text or not text.strip():
        return {"status": "bypass", "reason": "empty"}
    stripped = text.strip()
    if any(stripped.startswith(p) for p in BYPASS_PREFIXES):
        return {"status": "bypass", "reason": "bypass_prefix"}
    if is_skill_invocation(stripped):
        return {"status": "bypass", "reason": "skill_invocation"}
    if is_structured_command(stripped):
        return {"status": "bypass", "reason": "structured_command"}

    record = _run_optimizer_bridge(text, model, provider)
    if record is None:
        return {"status": "bypass", "reason": "no_rewrite_produced"}

    from .engine import create_preview, PromptOptimizationPreview
    import dataclasses
    preview = PromptOptimizationPreview(
        session_key=session_key,
        original=text,
        rewritten=record.rewritten,
        quality_before=record.quality_before,
        quality_after=record.quality_after,
        token_delta_pct=record.token_delta_pct,
        model_profile=record.model_profile or model or "unknown",
    )
    create_preview(session_key, preview)
    return {
        "status": "preview",
        "preview": dataclasses.asdict(preview),
    }


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    global _ctx
    _ctx = ctx
    logger.info("prompt-optimizer: initializing")
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    ctx.register_hook("pre_user_message", _on_pre_user_message)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_command(
        "prompt-optimizer", handler=_handle_prompt_optimizer,
        description="Toggle prompt optimiser mode and list report commands.",
        args_hint="[auto|interactive|off|status]")
    ctx.register_command(
        "prompt-insights", handler=_handle_prompt_insights,
        description="Prompt optimisation report: comparisons, suggestions, insights, analytics.",
        args_hint="[--html] [--limit N]")
    ctx.register_command(
        "prompt-compare", handler=_handle_prompt_compare,
        description="Show before/after prompt comparisons from recent rewrites.",
        args_hint="[--limit N]")
    ctx.register_command(
        "prompt-suggestions", handler=_handle_prompt_suggestions,
        description="Show reusable prompt replacement suggestions.",
        args_hint="[--limit N]")
    ctx.register_command(
        "prompt-analytics", handler=_handle_prompt_analytics,
        description="Daily, weekly, and monthly prompt optimisation analytics.",
        args_hint="[all|daily|weekly|monthly]")
    ctx.register_command(
        "prompt-stats", handler=_handle_prompt_stats,
        description="Compact prompt optimisation stats; use /prompt-insights for the full report.",
        args_hint="[--week|--today|--best|--raw]")
    _ensure_db()
    logger.info("prompt-optimizer plugin registered (mode=%s)", _mode)
