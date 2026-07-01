"""Shared last-turn telemetry compute + render.

Extracted (PRD usage-format-codex) from the fleet ``blackbox-inspect`` plugin so
that BOTH the plugin's ``/context`` command AND the gateway ``/usage`` handler
render the *identical* last-turn card from ONE renderer — no two-renderer drift.

The functions here are byte-for-byte the same logic ``/context`` shipped, with
two deliberate changes (PRD D-5/D-7):

* ``compute_last_turn_record`` takes the invoking channel as EXPLICIT
  ``platform``/``chat_id`` arguments instead of reading the
  ``HERMES_SESSION_*`` env itself. Callers pass the channel they resolved
  (``/context``: from ``get_session_env``; ``/usage``: from ``event.source``).
* When a NON-EMPTY channel is supplied but no row matches it, the function
  returns ``{"found": False, ...}`` instead of silently falling back to the
  profile-global newest turn. The global-newest fallback is reserved for the
  EMPTY-channel case (CLI/cron), preserving ``/context``'s prior behavior there.
  This kills the silent cross-channel-leak failure mode: a caller that supplied
  a real channel never receives another channel's numbers.

This module lives in ``plugins/blackbox/`` (importable, no hyphen) so the
gateway can ``from plugins.blackbox.last_turn import ...`` — the fleet
``blackbox-inspect`` plugin (hyphenated dir, load-by-path) cannot be imported.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

_LOG = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Formatting helpers (verbatim from blackbox-inspect)
# --------------------------------------------------------------------------- #
def _humanize_tok(n) -> str:
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    if abs(n) >= 1_000_000:
        return f"{n // 1_000_000}M" if n % 1_000_000 == 0 else f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1000:
        return f"{n // 1000}k" if n % 1000 == 0 else f"{n / 1000:.1f}k"
    return str(n)


def _ctx_health(pct: float) -> str:
    return "🟢" if pct < 70 else ("🟡" if pct <= 90 else "🔴")


def _cache_health(pct: float) -> str:
    return "🟢" if pct > 80 else ("🟡" if pct >= 50 else "🔴")


def _session_label(platform: str, chat_id: str, chat_name: str) -> str:
    label = (platform or "").strip()
    key = label.lower()
    name = (chat_name or "").strip()
    cid = (chat_id or "").strip()
    if key == "discord":
        if cid and name:
            return f"Discord #{name} (<#{cid}>)"
        if cid:
            return f"Discord <#{cid}>"
        if name:
            return f"Discord #{name}"
        return "Discord"
    if key == "telegram":
        detail = name or cid
        return f"Telegram #{detail}".rstrip(" #") or "Telegram"
    detail = name or cid
    return f"{label} {detail}".strip() or (label or "—")


def _tools_summary_from_json(tools_json) -> str:
    import json as _json
    from collections import Counter
    try:
        tools = _json.loads(tools_json) if isinstance(tools_json, str) else (tools_json or [])
    except Exception:
        tools = []
    if not tools:
        return "0"
    counts = Counter(str(t) for t in tools)
    parts = [f"{name}×{n}" if n > 1 else name for name, n in counts.items()]
    return f"{len(tools)} ({', '.join(parts)})"


def _normalize_call(entry):
    # NEW shape: {"composition": {...}, "output_tokens": int, "reasoning_tokens": int}.
    # The top-level output_tokens key is the unambiguous OLD/NEW discriminator.
    if isinstance(entry, dict) and "output_tokens" in entry:
        return entry.get("composition"), entry.get("output_tokens"), entry.get("reasoning_tokens")
    # OLD shape: the entry is the composition dict (or None); per-call output is unknown.
    return entry, None, None


def turn_output_split(comp_calls, output_billed, *, turn_id=None):
    """Returns (finished, unfinished) or (None, None) if the per-call split is unknown."""
    if not isinstance(comp_calls, list) or not comp_calls:
        return (None, None)
    outs = []
    for entry in comp_calls:
        _composition, output_tokens, _reasoning_tokens = _normalize_call(entry)
        if output_tokens is None:
            return (None, None)
        try:
            outs.append(int(output_tokens))
        except (TypeError, ValueError):
            return (None, None)
    try:
        billed = int(output_billed or 0)
    except (TypeError, ValueError):
        billed = 0
    finished = outs[-1]
    if billed < finished:
        _LOG.warning(
            "output split clamp: turn_id=%s output_billed=%s finished=%s",
            turn_id,
            billed,
            finished,
        )
    unfinished = max(0, billed - finished)
    return (finished, unfinished)


def _comp_calls_from_json(value):
    if not value:
        return None
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except Exception:
        return None
    return decoded if isinstance(decoded, list) else None


# --------------------------------------------------------------------------- #
# Compute (channel-scoped read of the per-profile Blackbox store)
# --------------------------------------------------------------------------- #
def compute_last_turn_record(platform: str = "", chat_id: str = "") -> Dict[str, Any]:
    """Return the most-recent Blackbox turn record (rich: cost, api/tool calls,
    latency, session, model, datetime) or a reason it's unavailable.

    Reads the per-profile Blackbox SQLite store directly (the authoritative
    per-turn telemetry — strictly richer than the thin resident-agent
    ``last_turn_usage`` probe).

    CHANNEL SCOPING (PRD D-5/D-7): the invoking channel is passed EXPLICITLY by
    the caller (``/context`` from ``get_session_env``; ``/usage`` from
    ``event.source``), NOT read from the env here.

    * Non-empty ``(platform, chat_id)``: return THAT channel's most-recent
      non-subagent turn; if none exists, return ``found=False`` — do NOT fall
      back to the profile-global newest (that would leak another channel's
      numbers to a caller that asked for a specific channel).
    * Empty channel (CLI/cron, where no channel is bound): fall back to the
      profile-global most-recent row, preserving ``/context``'s prior behavior.
    """
    try:
        from plugins.blackbox import store as _bb_store  # type: ignore
    except Exception as e:
        return {"found": False, "reason": f"blackbox store unavailable: {e}"}
    try:
        import sqlite3

        db = _bb_store._db_path()
        if not db.exists():
            return {"found": False, "reason": "no blackbox turns recorded yet"}

        platform = (platform or "").strip()
        chat_id = (chat_id or "").strip()
        have_channel = bool(platform and chat_id)

        conn = sqlite3.connect(str(db), timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            row = None
            scoped = False
            if have_channel:
                # THIS channel's most-recent turn. Exclude subagent rows so the
                # card reflects the user-facing turn, not a background fork.
                row = conn.execute(
                    "SELECT * FROM turns "
                    "WHERE platform=? AND chat_id=? AND COALESCE(is_subagent,0)=0 "
                    "ORDER BY ts_end DESC LIMIT 1",
                    (platform, chat_id),
                ).fetchone()
                scoped = row is not None
                # D-7: a supplied channel that matches nothing must NOT fall
                # through to global-newest — that's the cross-channel leak.
                if row is None:
                    return {
                        "found": False,
                        "reason": "no blackbox turns for this channel yet",
                    }
            else:
                # Channel unknown (CLI/cron): global most-recent.
                row = conn.execute(
                    "SELECT * FROM turns ORDER BY ts_end DESC LIMIT 1"
                ).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"found": False, "reason": "no blackbox turns recorded yet"}
        rec = dict(row)
        rec["found"] = True
        rec["source"] = (
            "blackbox store (this channel)" if scoped else "blackbox store (latest, any channel)"
        )
        return rec
    except Exception as e:
        return {"found": False, "reason": f"blackbox store read error: {e}"}


# --------------------------------------------------------------------------- #
# Render (verbatim from blackbox-inspect — the /context last-turn card)
# --------------------------------------------------------------------------- #
def render_last_turn_record(rec: Dict[str, Any], compressions: "int | None" = None) -> List[str]:
    """Render the rich Blackbox last-turn block (matches the alert-card fields).

    ``compressions`` (optional): when provided (from the live resident agent's
    context_compressor.compression_count), a "• Compressions: N" row is appended
    right after the Cached row so /usage and /context show it inside the card
    instead of orphaned below the footer. None / 0 ⇒ row omitted.
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        _pt = ZoneInfo("America/Los_Angeles")
    except Exception:
        _pt = None

    lines: List[str] = ["", "**Last turn** (most recent recorded turn):"]
    cost = rec.get("cost_usd")
    status = rec.get("cost_status") or ""
    if cost is not None:
        lines.append(f"• Turn Cost: ${float(cost):.2f}" + (f" ({status})" if status else ""))
    else:
        lines.append(f"• Turn Cost: n/a" + (f" ({status})" if status else ""))
    lines.append(f"• API Calls: {int(rec.get('api_calls', 0) or 0)}")
    lines.append(f"• Tool Calls: {_tools_summary_from_json(rec.get('tools'))}")

    in_tok = int(rec.get("input_tokens", 0) or 0)
    out_tok = int(rec.get("output_tokens", 0) or 0)
    cache_r = int(rec.get("cache_read", 0) or 0)
    cache_w = int(rec.get("cache_write", 0) or 0)
    reasoning = int(rec.get("reasoning", 0) or 0)

    # Tokens in (billed) — the FULL billed input, summed across every API call of
    # the turn: cache_read + cache_write + uncached. This double-counts the
    # context by design (caching re-sends it each call) because the goal is to
    # surface SPEND, not window occupancy (that's the Context: line below).
    # The three inline parts make the cache split explicit and stop the
    # misleading bare-`input_tokens` headline (the "10 in" artifact — uncached
    # is just ~2 structural tokens/call). See Obsidian "Hermes Telemetry —
    # Token Terminology & Accounting".
    in_billed = cache_r + cache_w + in_tok
    if in_billed > 0:
        lines.append(
            f"• Tokens in: {_humanize_tok(in_billed)} billed "
            f"({_humanize_tok(cache_r)} cache-read + "
            f"{_humanize_tok(cache_w)} cache-write + "
            f"{_humanize_tok(in_tok)} uncached)"
        )
    # Tokens out (billed) — all generated tokens for the turn. Split by CALL
    # POSITION: finished = the last API call's output (the answer the user got),
    # unfinished = all earlier calls' output (tool-orchestration that didn't
    # reach the user). reasoning tokens are folded INTO this total, no longer
    # broken out (Ace 2026-06-14: /context reports finished/unfinished only, not
    # final/reasoning). When the per-call split is unknown (old/NULL/blackbox-off
    # blob) show the bare billed total — NEVER fall back to final/reasoning.
    if out_tok > 0:
        finished_output, unfinished_output = turn_output_split(
            _comp_calls_from_json(rec.get("comp_calls_json")),
            out_tok,
            turn_id=rec.get("turn_id"),
        )
        if (finished_output, unfinished_output) != (None, None):
            lines.append(
                f"• Tokens out: {_humanize_tok(out_tok)} billed "
                f"({_humanize_tok(finished_output)} finished + "
                f"{_humanize_tok(unfinished_output)} unfinished)"
            )
        else:
            lines.append(f"• Tokens out: {_humanize_tok(out_tok)} billed")

    used = int(rec.get("context_used", 0) or 0)
    length = int(rec.get("context_length", 0) or 0)
    # Last-call cache split decomposes the WINDOW (occupancy), distinct from the
    # billed sums above. These three (when present) sum to context_used. Columns
    # are nullable — old rows / blackbox-off fall back to the plain line. Read
    # both the _tokens alias and the raw column name (compute_last_turn_record
    # does a direct SELECT *, so it carries the raw `last_cache_read` keys).
    lc_read = rec.get("last_cache_read_tokens", rec.get("last_cache_read"))
    lc_write = rec.get("last_cache_write_tokens", rec.get("last_cache_write"))
    lc_unc = rec.get("last_uncached_tokens", rec.get("last_uncached"))
    _have_split = lc_read is not None and lc_write is not None and lc_unc is not None
    # Change 3 (Ace 2026-06-14): the last-call cache split gets its OWN line
    # framed as the final call's billed input, ABOVE the occupancy line. The
    # split sums to context_used, so "Last call: {used} billed (split)" uses the
    # same headline as the window line — two framings of one number. The
    # Context-window line keeps ONLY the occupancy (suffix removed). When the
    # split is absent (old rows / blackbox-off) the Last-call line is omitted.
    if _have_split and used > 0:
        lines.append(
            f"• Last call: {_humanize_tok(used)} billed "
            f"({_humanize_tok(int(lc_read or 0))} cache-read + "
            f"{_humanize_tok(int(lc_write or 0))} cache-write + "
            f"{_humanize_tok(int(lc_unc or 0))} uncached)"
        )

    # Cached row sits ABOVE the Context-window row (Ace 2026-06-30): cache-hit
    # rate then occupancy reads more naturally than the reverse.
    cache_r = int(rec.get("cache_read", 0) or 0)
    cache_w = int(rec.get("cache_write", 0) or 0)
    prompt_total = in_tok + cache_r + cache_w
    if prompt_total > 0 and cache_r:
        cpct = cache_r / prompt_total * 100
        lines.append(f"• Cached: {_humanize_tok(cache_r)}/{_humanize_tok(prompt_total)} {_cache_health(cpct)} {cpct:.0f}%")

    if length > 0:
        # Clamp at 100%: last_prompt tokens can transiently overshoot the model
        # max during streaming or before compression fires — users must never
        # see >100% "of model max" (mirrors the clamp in agent/display.py,
        # cli.py /stats, gateway status, and tools/memory_tool.py; see
        # tests/run_agent/test_percentage_clamp.py).
        pct = min(100, used / length * 100)
        lines.append(
            f"• Context window (last call): {_humanize_tok(used)}/{_humanize_tok(length)} "
            f"{_ctx_health(pct)} ({pct:.0f}% of model max)"
        )
    elif used:
        lines.append(f"• Context window (last call): {_humanize_tok(used)}")

    try:
        _comp = int(compressions or 0)
    except (TypeError, ValueError):
        _comp = 0
    if _comp > 0:
        lines.append(f"• Compressions: {_comp}")

    lines.append(f"• Agent: {rec.get('profile') or '—'}")
    _model = rec.get("model") or "—"
    _prov = (rec.get("provider") or "").strip()
    _model_disp = f"{_prov}/{_model}" if _prov and _model != "—" else _model
    lines.append(f"• Model: {_model_disp}")
    lines.append(f"• Session: {_session_label(rec.get('platform',''), rec.get('chat_id',''), rec.get('chat_name',''))}")

    ts_start = rec.get("ts_start")
    ts_end = rec.get("ts_end")
    try:
        if ts_start and ts_end:
            lines.append(f"• Latency: {round(float(ts_end) - float(ts_start))}s")
    except Exception:
        pass
    try:
        if ts_end and _pt is not None:
            dt = datetime.fromtimestamp(float(ts_end), tz=_pt)
            lines.append(f"• Datetime: {dt:%Y/%m/%d %H:%M:%S} PT")
    except Exception:
        pass

    tid = rec.get("turn_id")
    if tid:
        lines.append(f"• Investigate: /cost {tid}")
    src = rec.get("source")
    if src:
        lines.append(f"_(source: {src})_")
    return lines
