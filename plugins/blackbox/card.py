"""Alert card rendering for blackbox telemetry."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from plugins.blackbox.record import TurnRecord, tools_summary, turn_output_split


_PT = ZoneInfo("America/Los_Angeles")


def humanize_tokens(value: int | float | None) -> str:
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        n = 0
    if abs(n) >= 1_000:
        return f"{n // 1_000}k" if n % 1_000 == 0 else f"{n / 1_000:.1f}k"
    return str(n)


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:.2f}"


def cost_health(cost: float | None, threshold: float) -> str:
    if cost is None or threshold <= 0:
        return ""
    ratio = cost / threshold
    if ratio < 0.5:
        return "🟢"
    if ratio < 1:
        return "🟡"
    return "🔴"


def context_health(fill_pct: float) -> str:
    if fill_pct < 70:
        return "🟢"
    if fill_pct <= 90:
        return "🟡"
    return "🔴"


def cache_health(cache_pct: float) -> str:
    if cache_pct > 80:
        return "🟢"
    if cache_pct >= 50:
        return "🟡"
    return "🔴"


def _session_line(platform: str, chat_id: str, chat_name: str) -> str:
    label = (platform or "").strip()
    key = label.lower()
    name = (chat_name or "").strip()
    cid = (chat_id or "").strip()
    if key == "discord":
        # `<#id>` renders as a clickable channel mention in Discord; append the
        # human channel name when we have it so the card is readable elsewhere
        # too (and not just an empty `Discord <#>` when chat_id is missing).
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
    if key == "slack":
        if cid and name:
            return f"Slack <#{cid}|{name}>"
        if cid:
            return f"Slack <#{cid}>"
        return f"Slack #{name}" if name else "Slack"
    detail = name or cid
    return f"{label} {detail}".strip() or (label or "—")


def _context_line(record: TurnRecord) -> str:
    used = int(record.context_used or 0)
    length = int(record.context_length or 0)
    if length <= 0:
        return humanize_tokens(used)
    pct = used / length * 100
    return (
        f"{humanize_tokens(used)}/{humanize_tokens(length)} "
        f"{context_health(pct)} ({pct:.0f}% of model max)"
    )


def _tokens_out_line(record: TurnRecord) -> str:
    """Output side of the Tokens line — split finished/unfinished when known.

    finished = last API call's output (the answer the user got); unfinished =
    earlier calls' output (tool orchestration). Reasoning is folded into the
    total (no longer broken out). Falls back to the bare billed total when the
    per-call split is unknown (old/NULL/blackbox-off blob).
    """
    import json as _json
    out = int(record.output_tokens or 0)
    raw = getattr(record, "comp_calls_json", None)
    calls = None
    if raw:
        try:
            calls = _json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            calls = None
    finished, unfinished = turn_output_split(calls if isinstance(calls, list) else None, out)
    if finished is not None:
        return (
            f"{humanize_tokens(out)} out "
            f"({humanize_tokens(finished)} finished + {humanize_tokens(unfinished)} unfinished)"
        )
    return f"{humanize_tokens(out)} out"


def _prompt_total(record: TurnRecord) -> int:
    """Total input billed for the turn.

    `record.input_tokens` is only the FRESH (uncached) input remainder. Under
    prompt caching almost all input arrives as cache reads/writes, so the bare
    field reads as a tiny leftover (e.g. 12) while the real input is hundreds of
    thousands. The true total is fresh + cache_read + cache_write, mirroring
    agent.usage_pricing.CanonicalUsage.prompt_tokens.
    """
    return (
        int(record.input_tokens or 0)
        + int(record.cache_read_tokens or 0)
        + int(record.cache_write_tokens or 0)
    )


def _cache_line(record: TurnRecord) -> str:
    cache_read = int(record.cache_read_tokens or 0)
    # Cache hit rate = fraction of the full prompt served from cache. The
    # denominator is the TOTAL prompt (fresh input + cache read + cache write),
    # NOT the bare fresh-input count — otherwise a cache-heavy turn (12 fresh
    # input, 669k cached) divides 669k/12 and reports a nonsense 5,576,942%.
    prompt_total = _prompt_total(record)
    if prompt_total <= 0:
        return "n/a"
    pct = cache_read / prompt_total * 100
    return (
        f"{humanize_tokens(cache_read)}/{humanize_tokens(prompt_total)} "
        f"{cache_health(pct)} {pct:.0f}%"
    )


def render_card(record: TurnRecord, threshold_usd: float) -> str:
    """Render the spending alert card text."""
    dt = datetime.fromtimestamp(record.ts_end or record.ts_start or 0, tz=_PT)
    session = _session_line(record.platform, record.chat_id, record.chat_name)
    latency = round(record.latency_s)

    return "\n".join(
        [
            "💸 Spending Alert",
            f"• Turn Cost: {_money(record.cost_usd)}",
            f"• Threshold: {_money(threshold_usd)}",
            f"• API Calls: {record.api_calls}",
            f"• Tool Calls: {len(record.tools)} ({tools_summary(record.tools)})",
            f"• Tokens: {humanize_tokens(_prompt_total(record))} in + {_tokens_out_line(record)}",
            f"• Context: {_context_line(record)}",
            f"• Cached: {_cache_line(record)}",
            f"• Agent: {record.profile}",
            f"• Model: {record.model}",
            f"• Session: {session}",
            f"• Latency: {latency}s",
            f"• Datetime: {dt:%Y/%m/%d %H:%M:%S} PT",
            f"• Investigate: /cost {record.turn_id}",
        ]
    )


def _record_from_row(row: dict) -> TurnRecord:
    """Hydrate a TurnRecord from a stored DB row dict (only known fields).

    Rows come back from store.get_turn / store.get_last_turn as plain dicts
    whose keys are a superset of TurnRecord's fields (plus joined extras like
    a rendered tools summary). Filter to the dataclass fields so unknown keys
    don't blow up the constructor, and JSON-decode the tools list if needed.
    """
    import json
    from dataclasses import fields as _dc_fields

    valid = {f.name for f in _dc_fields(TurnRecord)}
    data = {k: v for k, v in (row or {}).items() if k in valid}
    tools = data.get("tools")
    if isinstance(tools, str):
        try:
            data["tools"] = json.loads(tools) or []
        except Exception:
            data["tools"] = []
    elif tools is None:
        data["tools"] = []
    # turn_id is the only required positional field on TurnRecord.
    data.setdefault("turn_id", str(row.get("turn_id", "")) if row else "")
    return TurnRecord(**data)


def render(record: "dict | TurnRecord", threshold_usd: float | None = None) -> str:
    """Dict-or-TurnRecord facade used by the /cost command path.

    store.get_turn / get_last_turn return dict rows; this hydrates them into a
    TurnRecord and renders. The threshold defaults to the configured alert
    threshold (so the Threshold line on a /cost lookup matches what would have
    fired an alert), falling back to the turn's own cost only if config is
    unreadable.
    """
    rec = record if isinstance(record, TurnRecord) else _record_from_row(record)
    if threshold_usd is None:
        threshold_usd = _configured_threshold()
        if threshold_usd is None:
            threshold_usd = float(rec.cost_usd) if rec.cost_usd is not None else 0.0
    return render_card(rec, threshold_usd)


def _configured_threshold() -> float | None:
    """Best-effort read of blackbox.cost_alert_threshold_usd from config.yaml."""
    try:
        import yaml
        from hermes_constants import get_hermes_home

        path = Path(get_hermes_home()) / "config.yaml"
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        block = data.get("blackbox")
        if isinstance(block, dict) and block.get("cost_alert_threshold_usd") is not None:
            return float(block["cost_alert_threshold_usd"])
    except Exception:
        return None
    return None
