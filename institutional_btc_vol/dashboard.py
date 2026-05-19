from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

RFQ_ROWS = [
    ("RFQ-001", "Treasury 45D 15% OTM covered call", "Internal template", "Review required"),
    ("RFQ-002", "Treasury 45D 25% OTM covered call", "Internal template", "Review required"),
    ("RFQ-003", "Treasury 135D 20% OTM covered call", "Internal template", "Review required"),
    ("RFQ-004", "Treasury 135D 90/80/115/130 put-spread collar", "Internal template", "Review required"),
    ("RFQ-005", "Treasury 1,000 BTC put-spread collar", "Internal template", "Review required"),
    ("RFQ-006", "Miner 90D 85% floor / 120% cap collar", "Internal template", "Review required"),
    ("RFQ-007", "Miner 90D 80–85% disaster put", "Internal template", "Review required"),
    ("RFQ-008", "Miner 135D 90/75/115/130 put-spread collar", "Internal template", "Review required"),
    ("RFQ-009", "IBIT vs Deribit 30D ATM straddle", "Internal template", "Review required"),
    ("RFQ-010", "IBIT vs Deribit 30D 25D risk reversal", "Internal template", "Review required"),
]


def _fmt_money(value: float | None) -> str:
    return "Missing" if value is None else f"${value:,.2f}"


def _fmt_btc_share(value: float | None) -> str:
    return "Missing" if value is None else f"{value:.12f}"


def _fmt_iv(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _status_class(label: str) -> str:
    lowered = label.lower()
    if "captured" in lowered or "available" in lowered or "ready" in lowered or "internal" in lowered:
        return "ok"
    if "pending" in lowered or "missing" in lowered:
        return "warn"
    return "muted"


def _render_curve_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<tr><td colspan="4" class="empty">No rows available</td></tr>'
    out = []
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{escape(str(row.get('dte', '')))}D</td>"
            f"<td>{escape(str(row.get('expiry', '')))}</td>"
            f"<td class='mono'>{escape(str(row.get('native_symbol', '')))}</td>"
            f"<td class='num'>{escape(_fmt_iv(row.get('iv_mark')))}</td>"
            "</tr>"
        )
    return "\n".join(out)


def _render_dislocation_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<tr><td colspan="4" class="empty">No screen-only candidates generated</td></tr>'
    out = []
    for row in rows:
        action = str(row.get("next_action", "watch"))
        action_class = "quote" if "quote" in action.lower() else "watch"
        out.append(
            "<tr>"
            f"<td>{escape(str(row.get('candidate', '')))}</td>"
            f"<td class='num'>{float(row.get('gross_iv_diff_vol_pts', 0)):.2f} vol pts</td>"
            f"<td><span class='pill screen'>{escape(str(row.get('confidence', 'screen-only')).upper())}</span></td>"
            f"<td><span class='pill {action_class}'>{escape(action.title())}</span></td>"
            "</tr>"
        )
    return "\n".join(out)


def _render_warning_rows(warnings: list[str]) -> str:
    if not warnings:
        return "<li>No warnings</li>"
    return "\n".join(f"<li>{escape(str(warning))}</li>" for warning in warnings)


def _render_rfq_rows() -> str:
    out = []
    for rfq_id, structure, status, blocker in RFQ_ROWS:
        out.append(
            "<tr>"
            f"<td class='mono'>{escape(rfq_id)}</td>"
            f"<td>{escape(structure)}</td>"
            f"<td><span class='pill watch'>{escape(status)}</span></td>"
            f"<td><span class='pill warn'>{escape(blocker)}</span></td>"
            "</tr>"
        )
    return "\n".join(out)


def _render_recent_runs(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<tr><td colspan="5" class="empty">No prior runs recorded</td></tr>'
    out = []
    for row in rows[:8]:
        quote_reviews = int(row.get("quote_review_candidates") or 0)
        qr_label = "quote-review" if quote_reviews == 1 else "quote-reviews"
        out.append(
            "<tr>"
            f"<td class='mono'>{escape(str(row.get('run_id', '')))}</td>"
            f"<td>{escape(str(row.get('as_of_cst', '')))}</td>"
            f"<td class='num'>{escape(_fmt_money(row.get('btc_spot')))}</td>"
            f"<td class='num'>{int(row.get('dislocations') or 0)}</td>"
            f"<td>{quote_reviews} {qr_label}</td>"
            "</tr>"
        )
    return "\n".join(out)


def _render_quality_score(score: dict[str, Any] | None) -> str:
    if not score:
        return "<p class='empty'>No quality score computed</p>"
    grade = str(score.get("grade", "unknown")).upper()
    grade_class = str(score.get("grade", "muted")).lower()
    numeric = int(score.get("score") or 0)
    core = "Core sources captured" if score.get("core_sources_live") else "Core source gap"
    warnings = int(score.get("material_warning_count") or 0)
    notes = score.get("notes") or []
    notes_html = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:5]) or "<li>No notes</li>"
    return (
        "<div class='quality-wrap'>"
        f"<div class='quality-score {escape(grade_class)}'>{numeric} / 100</div>"
        f"<div><span class='pill {escape(grade_class if grade_class in {'green', 'yellow', 'red'} else 'watch')}'>{escape(grade)}</span></div>"
        f"<div class='small'>{escape(core)} · {warnings} material warnings</div>"
        f"<ul>{notes_html}</ul>"
        "</div>"
    )


def _fmt_signed_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _fmt_signed_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+d}"


def _fmt_signed_vol(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f} vol pts"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def _render_sparkline(values: list[float | int]) -> str:
    if len(values) < 2:
        return "<span class='small'>not enough history</span>"
    clean = [float(value) for value in values]
    min_v, max_v = min(clean), max(clean)
    width, height, pad = 180, 46, 5
    if max_v == min_v:
        points = [(pad + i * (width - 2 * pad) / (len(clean) - 1), height / 2) for i, _ in enumerate(clean)]
    else:
        points = [
            (
                pad + i * (width - 2 * pad) / (len(clean) - 1),
                height - pad - ((value - min_v) / (max_v - min_v)) * (height - 2 * pad),
            )
            for i, value in enumerate(clean)
        ]
    point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<svg class='spark' viewBox='0 0 {width} {height}' role='img' aria-label='trend sparkline'><polyline points='{escape(point_str)}' /></svg>"


def _render_trend_summary(summary: dict[str, Any] | None) -> str:
    if not summary or int(summary.get("run_count") or 0) < 2:
        return "<p class='empty'>Not enough run history for trend deltas</p>"
    grade = str(summary.get("latest_quality_grade") or "unknown").upper()
    return f"""
      <div class="trend-grid">
        <div class="trend-card"><div class="label">BTC change</div><div class="trend-value">{escape(_fmt_signed_money(summary.get('btc_change')))}</div>{_render_sparkline(summary.get('btc_series') or [])}</div>
        <div class="trend-card"><div class="label">Quality change</div><div class="trend-value">{escape(_fmt_signed_int(summary.get('quality_change')))}</div>{_render_sparkline(summary.get('quality_series') or [])}</div>
        <div class="trend-card"><div class="label">Quote-review flags</div><div class="trend-value">{escape(_fmt_signed_int(summary.get('quote_review_change')))}</div><div class="small">Latest grade: {escape(grade)}</div></div>
      </div>
    """


def _render_iv_benchmark_trends(summary: dict[str, Any] | None) -> str:
    if not summary or int(summary.get("run_count") or 0) < 2:
        return "<p class='empty'>Not enough IV benchmark history</p>"
    return f"""
      <div class="trend-grid">
        <div class="trend-card"><div class="label">7D spread change</div><div class="trend-value">{escape(_fmt_signed_vol(summary.get('spread_7d_change_vol_pts')))}</div>{_render_sparkline(summary.get('spread_7d_series') or [])}</div>
        <div class="trend-card"><div class="label">30D Deribit</div><div class="trend-value">{escape(_fmt_pct(summary.get('latest_deribit_30d_iv')))}</div><div class="small">Nearest ATM benchmark · screen-only</div></div>
        <div class="trend-card"><div class="label">30D IBIT</div><div class="trend-value">{escape(_fmt_pct(summary.get('latest_ibit_30d_iv')))}</div><div class="small">Model-estimated from public bid/ask mid</div></div>
      </div>
    """


def _render_quote_evidence(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return "<p class='empty'>No quote evidence recorded. RFQ workflow remains legal-pending.</p>"
    summary = evidence.get("summary") or {}
    valid_records = evidence.get("valid_records") or []
    invalid_records = evidence.get("invalid_records") or []
    rows = []
    for row in valid_records[:6]:
        publishable = "Investor-publishable" if row.get("is_investor_publishable") else "Not investor-publishable"
        rows.append(
            "<tr>"
            f"<td class='mono'>{escape(str(row.get('rfq_id', '')))}</td>"
            f"<td>{escape(str(row.get('structure', '')))}</td>"
            f"<td>{escape(str(row.get('counterparty', '')))}</td>"
            f"<td><span class='pill screen'>{escape(str(row.get('execution_confidence', '')).lower())}</span></td>"
            f"<td class='num'>{escape(str(row.get('spread_vol_pts', 'n/a')))} vol pts</td>"
            f"<td><span class='pill warn'>{escape(publishable)}</span></td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6" class="empty">No valid quote evidence rows</td></tr>')
    return f"""
      <div class="evidence-summary">
        <span>Total: {int(summary.get('total') or 0)}</span>
        <span>Quote-verified: {int(summary.get('quote_verified') or 0)}</span>
        <span>Trade-verified: {int(summary.get('trade_verified') or 0)}</span>
        <span>Invalid: {len(invalid_records)}</span>
      </div>
      <table><thead><tr><th>RFQ</th><th>Structure</th><th>Counterparty</th><th>Confidence</th><th class="num">Spread</th><th>Publishability</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
    """


def _render_market_diagnostics(diagnostics: list[dict[str, Any]] | None) -> str:
    if not diagnostics:
        return "<p class='empty'>No market microstructure diagnostics computed</p>"
    rows = []
    for row in diagnostics:
        grade = str(row.get("grade", "unknown")).lower()
        grade_class = grade if grade in {"green", "yellow", "red"} else "watch"
        notes = row.get("notes") or []
        note_text = "; ".join(str(note) for note in notes[:3])
        rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('source', '')))}</td>"
            f"<td><span class='pill {escape(grade_class)}'>{escape(grade.upper())}</span></td>"
            f"<td class='num'>{int(row.get('option_markets') or 0)}</td>"
            f"<td class='num'>{int(row.get('valid_bid_ask') or 0)}</td>"
            f"<td class='num'>{int(row.get('missing_bid_ask') or 0)}</td>"
            f"<td class='num'>{int(row.get('crossed_markets') or 0)}</td>"
            f"<td class='num'>{int(row.get('wide_markets') or 0)}</td>"
            f"<td>{escape(note_text)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr><th>Source</th><th>Grade</th><th class='num'>Markets</th><th class='num'>Valid B/A</th><th class='num'>Missing B/A</th><th class='num'>Crossed</th><th class='num'>Wide</th><th>Notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_freshness(freshness: dict[str, Any] | None) -> str:
    if not freshness:
        return "<p class='empty'>No source freshness checks computed</p>"
    grade = str(freshness.get("grade", "unknown")).lower()
    grade_class = grade if grade in {"green", "yellow", "red"} else "watch"
    rows = []
    for name, row in (freshness.get("sources") or {}).items():
        status = str(row.get("status", "unknown")).lower()
        status_class = "green" if status == "fresh" else "yellow" if status == "stale" else "red"
        age = row.get("age_minutes")
        age_text = "n/a" if age is None else f"{float(age):.1f} min"
        rows.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td><span class='pill {escape(status_class)}'>{escape(status.upper())}</span></td>"
            f"<td class='num'>{escape(age_text)}</td>"
            f"<td class='num'>{int(row.get('max_age_minutes') or 0)} min</td>"
            f"<td class='mono'>{escape(str(row.get('captured_at') or 'missing'))}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="5" class="empty">No source rows</td></tr>'
    return (
        f"<div class='evidence-summary'><span>Grade: <span class='pill {escape(grade_class)}'>{escape(grade.upper())}</span></span>"
        f"<span>{escape(str(freshness.get('evidence_status', 'SCREEN-ONLY · NOT EXECUTABLE')))}</span></div>"
        f"<table><thead><tr><th>Source</th><th>Status</th><th class='num'>Age</th><th class='num'>Max Age</th><th>Captured At</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _render_candidate_triage(rows: list[dict[str, Any]] | None) -> str:
    if not rows:
        return "<p class='empty'>No ranked dislocation candidates</p>"
    out = []
    for row in rows[:8]:
        priority = str(row.get("priority", "low")).lower()
        priority_class = "red" if priority == "high" else "yellow" if priority == "medium" else "watch"
        out.append(
            "<tr>"
            f"<td class='num'>{int(row.get('rank') or 0)}</td>"
            f"<td>{escape(str(row.get('candidate', '')))}</td>"
            f"<td class='num'>{float(row.get('gross_iv_diff_vol_pts') or 0):.2f} vol pts</td>"
            f"<td>{escape(str(row.get('direction', '')))}</td>"
            f"<td><span class='pill {escape(priority_class)}'>{escape(priority.upper())}</span></td>"
            f"<td><span class='pill screen'>{escape(str(row.get('evidence_status', 'SCREEN-ONLY')))}</span></td>"
            f"<td>{escape(str(row.get('recommended_workflow', 'watch only')))}</td>"
            "</tr>"
        )
    return f"<table><thead><tr><th class='num'>Rank</th><th>Candidate</th><th class='num'>Spread</th><th>Direction</th><th>Priority</th><th>Evidence</th><th>Workflow</th></tr></thead><tbody>{''.join(out)}</tbody></table>"


def render_dashboard_html(
    *,
    run_id: str,
    as_of_cst: str,
    btc_spot: float | None,
    btc_per_share: float | None,
    deribit_atm_rows: list[dict[str, Any]],
    ibit_atm_rows: list[dict[str, Any]],
    dislocations: list[dict[str, Any]],
    quality_warnings: list[str],
    quality_score: dict[str, Any] | None = None,
    trend_summary: dict[str, Any] | None = None,
    quote_evidence: dict[str, Any] | None = None,
    candidate_triage: list[dict[str, Any]] | None = None,
    market_diagnostics: list[dict[str, Any]] | None = None,
    freshness: dict[str, Any] | None = None,
    recent_runs: list[dict[str, Any]] | None = None,
    cme_rows: int = 0,
) -> str:
    deribit_status = "Captured" if deribit_atm_rows else "Missing"
    ibit_status = "Captured" if ibit_atm_rows else "Missing"
    holdings_status = "Available" if btc_per_share is not None else "Missing"
    cme_status = "Available" if cme_rows > 0 else "Missing"
    cme_detail = f"{cme_rows:,} Databento rows" if cme_rows > 0 else "licensed/vendor feed required"
    rfq_status = "Legal pending"

    status_cards = [
        ("Deribit", deribit_status, f"{len(deribit_atm_rows)} ATM expiries"),
        ("IBIT / ETF", ibit_status, f"{len(ibit_atm_rows)} ATM expiries"),
        ("IBIT holdings", holdings_status, _fmt_btc_share(btc_per_share)),
        ("CME", cme_status, cme_detail),
        ("RFQ", rfq_status, "internal templates; no outreach"),
    ]
    cards_html = "\n".join(
        f"<section class='card status'><div class='label'>{escape(name)}</div><div class='status-line {escape(_status_class(status))}'>{escape(status)}</div><div class='small'>{escape(detail)}</div></section>"
        for name, status, detail in status_cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC Vol Desk Monitor — {escape(run_id)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0b1220; color: #e5eefb; }}
    .shell {{ max-width: 1420px; margin: 0 auto; padding: 28px; }}
    header {{ display: flex; justify-content: space-between; gap: 24px; align-items: flex-start; margin-bottom: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; color: #f8fafc; }}
    .sub {{ color: #93a4bd; font-size: 14px; }}
    .badge {{ display: inline-flex; align-items: center; border: 1px solid #f59e0b; color: #fbbf24; border-radius: 999px; padding: 8px 12px; font-weight: 800; letter-spacing: .08em; font-size: 12px; background: rgba(245,158,11,.08); }}
    .grid {{ display: grid; gap: 16px; }}
    .status-grid {{ grid-template-columns: repeat(5, minmax(0, 1fr)); margin-bottom: 16px; }}
    .two {{ grid-template-columns: 1fr 1fr; }}
    .card {{ background: linear-gradient(180deg, rgba(20,31,51,.96), rgba(15,23,42,.96)); border: 1px solid rgba(148,163,184,.18); border-radius: 18px; padding: 18px; box-shadow: 0 20px 60px rgba(0,0,0,.24); }}
    .status .label {{ color: #93a4bd; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .status-line {{ font-size: 24px; font-weight: 800; margin: 8px 0 4px; }}
    .ok {{ color: #34d399; }} .warn {{ color: #fbbf24; }} .muted {{ color: #94a3b8; }}
    .small {{ color: #8ea0bb; font-size: 13px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 20px 0; }}
    .kpi {{ border: 1px solid rgba(148,163,184,.14); border-radius: 14px; padding: 14px; background: rgba(15,23,42,.72); }}
    .kpi .name {{ color: #93a4bd; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .kpi .value {{ margin-top: 8px; font-size: 21px; font-weight: 800; }}
    .method-note {{ margin: 0 0 16px; color: #cbd5e1; font-size: 13px; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th {{ text-align: left; color: #93a4bd; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; border-bottom: 1px solid rgba(148,163,184,.18); padding: 10px 8px; }}
    td {{ border-bottom: 1px solid rgba(148,163,184,.1); padding: 10px 8px; vertical-align: middle; }}
    tr:hover td {{ background: rgba(59,130,246,.07); }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .pill {{ display: inline-flex; border-radius: 999px; padding: 4px 8px; font-size: 11px; font-weight: 800; }}
    .screen {{ color: #fbbf24; background: rgba(245,158,11,.12); }}
    .quote {{ color: #60a5fa; background: rgba(37,99,235,.18); }}
    .watch {{ color: #cbd5e1; background: rgba(148,163,184,.14); }}
    .pill.warn {{ color: #fbbf24; background: rgba(245,158,11,.12); }}
    .pill.green, .quality-score.green {{ color: #34d399; background: rgba(16,185,129,.12); }}
    .pill.yellow, .quality-score.yellow {{ color: #fbbf24; background: rgba(245,158,11,.12); }}
    .pill.red, .quality-score.red {{ color: #fb7185; background: rgba(244,63,94,.12); }}
    .quality-score {{ display: inline-flex; border-radius: 14px; padding: 10px 12px; font-size: 26px; font-weight: 900; margin-bottom: 10px; }}
    .quality-wrap ul {{ margin-top: 12px; }}
    .trend-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .trend-card {{ border: 1px solid rgba(148,163,184,.14); border-radius: 14px; padding: 14px; background: rgba(15,23,42,.72); }}
    .trend-card .label {{ color: #93a4bd; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    .trend-value {{ margin: 8px 0; font-size: 24px; font-weight: 900; }}
    .spark {{ width: 100%; max-width: 180px; height: 46px; }}
    .spark polyline {{ fill: none; stroke: #60a5fa; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .evidence-summary {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; color: #cbd5e1; font-size: 13px; }}
    .evidence-summary span {{ border: 1px solid rgba(148,163,184,.16); border-radius: 999px; padding: 5px 9px; background: rgba(15,23,42,.72); }}
    .empty {{ color: #64748b; text-align: center; padding: 28px; }}
    ul {{ margin: 0; padding-left: 20px; color: #cbd5e1; }}
    li {{ margin: 8px 0; }}
    footer {{ margin-top: 22px; color: #64748b; font-size: 12px; }}
    @media (max-width: 980px) {{ .status-grid, .two, .kpis, .trend-grid {{ grid-template-columns: 1fr; }} header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>BTC Vol Desk Monitor</h1>
        <div class="sub">Run <span class="mono">{escape(run_id)}</span> · {escape(as_of_cst)}</div>
      </div>
      <div class="badge">SCREEN-ONLY · NOT EXECUTABLE</div>
    </header>

    <section class="kpis">
      <div class="kpi"><div class="name">BTC reference</div><div class="value">{escape(_fmt_money(btc_spot))}</div></div>
      <div class="kpi"><div class="name">IBIT BTC/share</div><div class="value mono">{escape(_fmt_btc_share(btc_per_share))}</div></div>
      <div class="kpi"><div class="name">Dislocation candidates</div><div class="value">{len(dislocations)}</div></div>
    </section>

    <section class="grid status-grid">{cards_html}</section>

    <section class="card method-note">Internal evidence prototype. Public/API screen marks and model-estimated IV only; no RFQ is sent and no executable quote is implied. IBIT-vs-Deribit spreads are screening signals: tenor matching is approximate and excludes bid/ask, borrow, financing, tax/accounting, settlement, venue-hour, margin/collateral, and counterparty effects.</section>

    <section class="card" style="margin-bottom:16px"><h2>Trend Snapshot</h2>{_render_trend_summary(trend_summary)}</section>

    <section class="card" style="margin-bottom:16px"><h2>IV Benchmark Trends</h2>{_render_iv_benchmark_trends(trend_summary)}</section>

    <section class="grid two">
      <div class="card"><h2>Deribit ATM IV Curve</h2><table><thead><tr><th>DTE</th><th>Expiry</th><th>Symbol</th><th class="num">IV</th></tr></thead><tbody>{_render_curve_rows(deribit_atm_rows)}</tbody></table></div>
      <div class="card"><h2>IBIT / ETF ATM IV Curve</h2><table><thead><tr><th>DTE</th><th>Expiry</th><th>Symbol</th><th class="num">IV</th></tr></thead><tbody>{_render_curve_rows(ibit_atm_rows)}</tbody></table></div>
    </section>

    <section class="card" style="margin-top:16px"><h2>IBIT vs Deribit Dislocation Board</h2><div class="small" style="margin-bottom:10px">Screen-only comparison; not executable economics.</div><table><thead><tr><th>Candidate</th><th class="num">Gross IV Diff</th><th>Confidence</th><th>Next Action</th></tr></thead><tbody>{_render_dislocation_rows(dislocations)}</tbody></table></section>

    <section class="card" style="margin-top:16px"><h2>Candidate Triage</h2><div class="small" style="margin-bottom:10px">Internal review list; no submission or outreach implied.</div>{_render_candidate_triage(candidate_triage)}</section>

    <section class="card" style="margin-top:16px"><h2>RFQ Quote Evidence</h2><div class="small" style="margin-bottom:10px">Manual evidence ledger only; demo/manual rows do not count as real quote-verified evidence.</div>{_render_quote_evidence(quote_evidence)}</section>

    <section class="card" style="margin-top:16px"><h2>Market Diagnostics</h2>{_render_market_diagnostics(market_diagnostics)}</section>

    <section class="card" style="margin-top:16px"><h2>Source Freshness</h2>{_render_freshness(freshness)}</section>

    <section class="grid two" style="margin-top:16px">
      <div class="card"><h2>Internal RFQ Template Inventory</h2><div class="small" style="margin-bottom:10px">Templates are internal review aids, not a product offering.</div><table><thead><tr><th>ID</th><th>Structure</th><th>Status</th><th>Blocker</th></tr></thead><tbody>{_render_rfq_rows()}</tbody></table></div>
      <div class="card"><h2>Data Quality</h2>{_render_quality_score(quality_score)}<h2 style="margin-top:18px">Quality Warnings</h2><ul>{_render_warning_rows(quality_warnings)}</ul></div>
    </section>

    <section class="card" style="margin-top:16px"><h2>Recent Runs</h2><table><thead><tr><th>Run</th><th>Timestamp</th><th class="num">BTC</th><th class="num">Candidates</th><th>Review Flags</th></tr></thead><tbody>{_render_recent_runs(recent_runs or [])}</tbody></table></section>

    <footer>Evidence policy: public/API marks and model-estimated IVs remain screen-only until independently quote-verified or trade-verified. No client-specific RFQ or execution workflow is enabled.</footer>
  </main>
</body>
</html>
"""


def write_dashboard(path: str | Path, **kwargs: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard_html(**kwargs), encoding="utf-8")
    return target
