"""Alert card rendering for blackbox telemetry."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from plugins.blackbox.record import TurnRecord, tools_summary


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
    if key == "discord":
        return f"Discord <#{chat_id}>"
    if key == "telegram":
        return f"Telegram #{chat_name}"
    if key == "slack":
        return f"Slack <#{chat_id}|{chat_name}>"
    detail = chat_name or chat_id
    return f"{label} {detail}".strip()


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


def _cache_line(record: TurnRecord) -> str:
    input_tokens = int(record.input_tokens or 0)
    cache_read = int(record.cache_read_tokens or 0)
    if input_tokens <= 0:
        return "n/a"
    pct = cache_read / input_tokens * 100
    return (
        f"{humanize_tokens(cache_read)}/{humanize_tokens(input_tokens)} "
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
            f"• Tokens: {humanize_tokens(record.input_tokens)} in + {humanize_tokens(record.output_tokens)} out",
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
