"""Rich Flight Deck renderer for the Context Cockpit.

Layout lock:
  1. Top state ribbon
  2. Middle 2x2: Context | LCM | Cost | Model
  3. Bottom next-action strip
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .metrics import HYGIENE_PCT, HOST_COMPRESS_PCT, QUALITY_WATCH_PCT
from .status import classify_status

GAUGE_WIDTH = 28
AUTHORITY_LINE = (
    "Authority: closeouts/commits > runtime proof > LCM > Hindsight > chat memory"
)


def _fmt_tokens(n: Optional[int]) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return f"{n:,}"


def _fmt_usd(n: Optional[float]) -> str:
    if n is None:
        return "—"
    if abs(n) < 0.01:
        return f"${n:.4f}"
    return f"${n:.2f}"


def _bar(pct: float, width: int = GAUGE_WIDTH, *, dim: bool = False) -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = round((pct / 100.0) * width)
    bar = Text()
    for i in range(width):
        ch = "█" if i < filled else "░"
        if dim:
            style = "dim"
        elif pct >= HYGIENE_PCT:
            style = "bold red" if i < filled else "dim red"
        elif pct >= HOST_COMPRESS_PCT:
            style = "yellow" if i < filled else "dim yellow"
        elif pct >= QUALITY_WATCH_PCT:
            style = "bold yellow" if i < filled else "dim yellow"
        else:
            style = "bright_green" if i < filled else "dim green"
        bar.append(ch, style=style)
    return bar


def _ribbon_panel(status: Dict[str, Any], metrics: Dict[str, Any]) -> Panel:
    ribbon = status["ribbon"]
    severity = status["severity"]
    summary = status["summary"]
    color = status["color"]

    if severity == "critical":
        style = "bold white on red"
        border = "red"
    elif severity in {"warn", "info"}:
        style = "bold black on yellow"
        border = "yellow"
    else:
        style = "bold white on green"
        border = "green"

    if status.get("dim_gauges"):
        style = "bold white on grey37"
        border = "grey50"

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(justify="center")
    grid.add_row(Text(f"  HERMES: {ribbon}  ", style=style))
    grid.add_row(Text(summary, style="bold white" if severity == "critical" else color))

    # Tiny freshness chip
    fresh = metrics.get("freshness", "?")
    hb = (metrics.get("liveness") or {}).get("heartbeat_age_sec")
    hb_s = f"{hb:.0f}s" if isinstance(hb, (int, float)) else "—"
    grid.add_row(
        Text(f"freshness={fresh}  heartbeat={hb_s}", style="dim")
    )

    return Panel(
        Align.center(grid),
        border_style=border,
        box=box.HEAVY,
        padding=(1, 2),
    )


def _kv_card(
    title: str,
    rows: list[tuple[str, str]],
    *,
    border: str,
    dim: bool = False,
) -> Panel:
    t = Table.grid(expand=True, padding=(0, 1))
    t.add_column(style="cyan" if not dim else "dim", ratio=2)
    t.add_column(style="white" if not dim else "dim", ratio=3, overflow="ellipsis")
    for k, v in rows:
        t.add_row(k, v)
    return Panel(
        t,
        title=f"[bold]{title}[/]" if not dim else f"[dim]{title}[/]",
        border_style=("grey50" if dim else border),
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _context_card(metrics: Dict[str, Any], status: Dict[str, Any]) -> Panel:
    dim = bool(status.get("dim_gauges"))
    pct = float(metrics.get("prompt_pct") or 0.0)
    real = int(metrics.get("prompt_tokens") or 0)
    window = int(metrics.get("window") or 0)
    lcm = metrics.get("lcm") or {}
    thresh = int(lcm.get("threshold_tokens") or 0)
    next_ctx = "wait"
    if status["ribbon"] in {"GETTING FULL", "MEMORY LINE HIT", "SHRINKING NOW", "CAN'T SHRINK YET", "SHRINK QUEUED"}:
        next_ctx = "compress"
    elif status["ribbon"] == "QUIET":
        next_ctx = "watch"

    body = Table.grid(expand=True, padding=(0, 0))
    body.add_row(Text(f"{_fmt_tokens(real)} / {_fmt_tokens(window)} tokens", style="bold" if not dim else "dim"))
    body.add_row(_bar(pct, dim=dim))
    body.add_row(Text(f"{pct:.1f}%   auto-shrink around: {_fmt_tokens(thresh)}", style="dim"))
    body.add_row(Text(f"Next: {next_ctx}", style="dim"))
    return Panel(
        body,
        title="[bold]How full?[/]" if not dim else "[dim]How full?[/]",
        border_style="grey50" if dim else ("red" if pct >= HOST_COMPRESS_PCT else "green"),
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _lcm_card(metrics: Dict[str, Any], status: Dict[str, Any]) -> Panel:
    dim = bool(status.get("dim_gauges"))
    lcm = metrics.get("lcm") or {}
    loaded = "Loaded" if lcm.get("loaded") else "Not loaded"
    ratio = float(lcm.get("threshold_ratio") or 0.0)
    rows = [
        ("state", loaded),
        ("auto-shrink line", f"{ratio*100:.0f}%"),
        ("auto-shrinks", str(lcm.get("compressions", 0))),
        ("turns since shrink", str(lcm.get("turns_since_leaf") if lcm.get("turns_since_leaf") is not None else "—")),
        ("cache", str(lcm.get("cache_state") or "—")),
        ("progress to line", f"{float(lcm.get('fill_of_lcm') or 0)*100:.0f}%"),
    ]
    border = "magenta"
    if status["ribbon"] in {"MEMORY LINE HIT", "SHRINKING NOW", "CAN'T SHRINK YET", "SHRINK QUEUED"}:
        border = "red"
    return _kv_card("LCM / Auto-shrink", rows, border=border, dim=dim)


def _cost_card(metrics: Dict[str, Any], status: Dict[str, Any]) -> Panel:
    dim = bool(status.get("dim_gauges"))
    cost = metrics.get("cost") or {}
    burn = cost.get("burn") or {}
    burn_label = "unknown"
    tpc = burn.get("tok_per_call")
    if tpc is not None:
        burn_label = "elevated" if tpc > 5000 else "moderate" if tpc > 1000 else "low"
    rows = [
        ("session", _fmt_usd(cost.get("estimated_usd"))),
        ("actual", _fmt_usd(cost.get("actual_usd"))),
        ("burn", burn_label),
        ("Δ tok/call", _fmt_tokens(int(tpc)) if tpc else "—"),
        ("billing", str(cost.get("billing_mode") or cost.get("billing_provider") or "—")),
        ("note", "Visor > /usage for OR cost"),
    ]
    border = "green"
    if status["ribbon"] == "COST WARNING":
        border = "red"
    return _kv_card("Cost", rows, border=border, dim=dim)


def _model_card(metrics: Dict[str, Any], status: Dict[str, Any]) -> Panel:
    dim = bool(status.get("dim_gauges"))
    model = str(metrics.get("model") or "unknown")
    short = model.split("/")[-1] if "/" in model else model
    alert = metrics.get("model_alert")
    rows = [
        ("model", short),
        ("full", model if len(model) < 40 else model[:37] + "…"),
        ("window", _fmt_tokens(int(metrics.get("window") or 0))),
        ("source", str(metrics.get("window_source") or "—")),
        ("fallback", "ALERT" if alert else "none"),
        ("alert", "stable" if not alert else "changed"),
    ]
    border = "red" if alert or status["ribbon"] == "MODEL CHANGED" else "blue"
    return _kv_card("Model", rows, border=border, dim=dim)


def _next_action_panel(status: Dict[str, Any]) -> Panel:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column()
    grid.add_row(Text(f"NEXT: {status['next_action']}", style="bold white"))
    cmds = " · ".join(status.get("useful_commands") or [])
    grid.add_row(
        Text.assemble(
            ("Useful: ", "dim"),
            (status.get("command") or "", "bold bright_cyan underline"),
            ("  |  ", "dim"),
            (cmds, "dim"),
        )
    )
    grid.add_row(Text(AUTHORITY_LINE, style="dim italic"))
    border = {"ok": "green", "info": "yellow", "warn": "yellow", "critical": "red"}.get(
        status.get("severity", "ok"), "cyan"
    )
    return Panel(
        grid,
        title="[bold]Next best action[/]",
        border_style=border,
        box=box.HEAVY,
        padding=(0, 1),
    )


def build_cockpit(metrics: Dict[str, Any], status: Optional[Dict[str, Any]] = None) -> Panel:
    status = status or classify_status(metrics)
    ribbon = _ribbon_panel(status, metrics)
    # 2x2 via two column rows
    top = Columns(
        [_context_card(metrics, status), _lcm_card(metrics, status)],
        equal=True,
        expand=True,
    )
    bottom = Columns(
        [_cost_card(metrics, status), _model_card(metrics, status)],
        equal=True,
        expand=True,
    )
    action = _next_action_panel(status)
    body = Group(ribbon, top, bottom, action)
    return Panel(
        body,
        title="[bold bright_cyan]⌬[/]  [bold white]Hermes Context Cockpit[/]",
        title_align="center",
        border_style=status.get("color", "bright_cyan"),
        box=box.DOUBLE_EDGE,
        padding=(0, 1),
    )
