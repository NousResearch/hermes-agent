from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _money(value: Any) -> str:
    return "n/a" if value is None else f"${float(value):,.2f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _num(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _hash(value: Any) -> str:
    text = str(value or "missing")
    return text if len(text) <= 18 else f"{text[:12]}…{text[-10:]}"


def _review_language(value: Any) -> str:
    text = str(value or "review only")
    replacements = {
        "add to RFQ review queue": "add to candidate review list",
        "RFQ review queue": "candidate review list",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _candidate_cards(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "<p class='muted'>No current candidates. The engine will populate this after the next monitor run.</p>"
    cards = []
    for row in candidates[:4]:
        cards.append(
            "<article class='candidate-card'>"
            f"<div class='rank'>#{escape(str(row.get('rank') or '—'))}</div>"
            f"<h3>{escape(str(row.get('candidate') or 'Candidate'))}</h3>"
            f"<p>{escape(str(row.get('direction') or 'Review direction pending'))}</p>"
            f"<div class='metric'>{float(row.get('gross_iv_diff_vol_pts') or 0):+.2f} vol pts</div>"
            f"<span class='pill amber'>{escape(str(row.get('evidence_status') or 'SCREEN-ONLY · NOT EXECUTABLE'))}</span>"
            f"<p class='workflow'>{escape(_review_language(row.get('recommended_workflow')))}</p>"
            "</article>"
        )
    return "".join(cards)


def _evidence_link(label: str, value: Any) -> str:
    if not value:
        return ""
    return f"<li><span>{escape(label)}</span><code>{escape(str(value))}</code></li>"


def _claim_control_pill() -> str:
    return '<span class="metric-label">ILLUSTRATIVE MODEL OUTPUT · NOT A QUOTE · NOT EXECUTABLE · NOT INVESTOR-PUBLISHABLE</span>'


def _case_study_panels(case_studies: dict[str, Any]) -> str:
    if case_studies.get("available") is False:
        return (
            "<article class=\"panel\"><span class=\"pill amber\">Case studies suppressed</span>"
            "<h2>Case Studies Suppressed</h2>"
            f"<p class=\"control-note small\">{escape(str(case_studies.get('control_note') or 'BTC spot unavailable — case studies suppressed to avoid fallback economics.'))}</p>"
            "</article>"
        )
    treasury = case_studies.get("treasury") or {}
    miner = case_studies.get("miner") or {}
    control = _claim_control_pill()
    return f"""
      <article class="panel"><span class="pill amber">Case study · screen-only</span><h2>Treasury Hedge Case Study</h2><p>For corporate BTC holders: covered-call income, put-spread collars, and board-approved hedge policies that convert volatility into treasury risk management.</p><div class="case-metrics"><div>{control}<strong>{escape(_num(treasury.get('hedged_btc')))} BTC hedged</strong><span>Conservative policy sleeve</span></div><div>{control}<strong>{escape(_money(treasury.get('protected_value_at_floor_usd')))} illustrative floor-protected sleeve</strong><span>Illustrative protected value</span></div><div>{control}<strong>{escape(_money(treasury.get('floor_price')))} illustrative floor / {escape(_money(treasury.get('cap_price')))} cap</strong><span>Structure preview</span></div></div><p class="control-note small">{escape(str(treasury.get('quote_control') or 'Premium and executable levels require quote verification.'))}</p></article>
      <article class="panel"><span class="pill amber">Case study · screen-only</span><h2>Miner Runway Protection</h2><p>For miners: protect production runway without overhedging or creating collateral stress. The goal is survival and financing credibility, not speculative upside chasing.</p><div class="case-metrics"><div>{control}<strong>{escape(_num(miner.get('hedged_monthly_btc')))} BTC/month hedged</strong><span>Conservative production hedge</span></div><div>{control}<strong>{escape(_money(miner.get('monthly_floor_revenue_on_hedged_btc_usd')))} model-estimated monthly floor revenue</strong><span>Monthly floor revenue on hedged BTC</span></div><div>{control}<strong>{escape(_num(miner.get('cash_runway_months_before_hedge')))} months pre-hedge cash runway</strong><span>Baseline operating runway</span></div></div><p class="control-note small">{escape(str(miner.get('quote_control') or 'Indicative economics require quote verification before use.'))}</p></article>
    """


def _quote_verification_board(board: dict[str, Any]) -> str:
    rows = board.get("rows") or []
    summary = board.get("summary") or {}
    label_map = {
        "screen_only": "Screen-only",
        "reviewed": "Reviewed",
        "rfq_package_drafted": "Internal RFQ draft",
        "indicative_quote_1": "Indicative quote 1",
        "indicative_quote_2": "Indicative quote 2",
        "quote_verified": "Quote verified",
        "trade_verified": "Post-trade record verified",
    }
    row_html = "".join(
        "<article class='workflow-card'>"
        f"<div class='rank'>#{escape(str(row.get('rank') or '—'))}</div>"
        f"<h3>{escape(str(row.get('candidate') or 'Candidate'))}</h3>"
        f"<div class='stage'>{escape(label_map.get(str(row.get('stage') or 'screen_only'), str(row.get('stage') or 'screen_only')))}</div>"
        f"<p>{escape(str(row.get('next_action') or 'Internal review only.'))}</p>"
        f"<p class='workflow'>Quotes captured: {escape(str(row.get('quote_count', 0)))} / {escape(str(row.get('counterparty_quotes_required', 2)))} · Publishability: {escape(str(row.get('publishability') or 'internal-only'))}</p>"
        f"<span class='pill amber'>{escape(str(row.get('evidence_status') or 'SCREEN-ONLY · NOT EXECUTABLE'))}</span>"
        "</article>"
        for row in rows[:4]
    )
    if not row_html:
        row_html = "<p class='muted'>No quote verification rows yet. The board populates from candidate triage output.</p>"
    summary_html = "".join(
        f"<div class='kpi'><div class='label'>{escape(label_map.get(str(stage), str(stage)))}</div><div class='value'>{escape(str(count))}</div></div>"
        for stage, count in summary.items()
    )
    return f"""
    <section id="workflow" class="panel"><h2>{escape(str(board.get('title') or 'Quote Verification Demo Board'))}</h2><p>{escape(str(board.get('control') or 'Manual demo workflow only; no executable quote is implied.'))}</p><div class="quote-summary">{summary_html}</div><div class="quote-board-grid">{row_html}</div><div class="timeline"><div>Screen-only dislocation detected by the monitor.</div><div>Candidate triaged for materiality and structure fit.</div><div>RFQ package generated for review, not submission.</div><div>After approval and manual outreach, counterparties may provide indicative quotes.</div><div>Evidence ledger may mark candidate quote-verified after required indicative quote records are captured.</div><div>Post-trade verification only occurs after an actual external execution record exists; this demo has no trading capability.</div></div></section>
    """


def _backtest_research_panel(backtest: dict[str, Any]) -> str:
    if not backtest or not backtest.get("available"):
        return """
    <section id="backtest" class="panel"><h2>Backtest / Research Evidence</h2><span class="pill amber">SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE</span><p>No point-in-time backtest artifact is available yet. Any future outputs will use synthetic fills only and will not represent executable economics.</p></section>
    """
    evidence_status = str(backtest.get("evidence_status") or "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE")
    sample_gate = str(backtest.get("sample_gate") or "insufficient-history")
    scenarios = backtest.get("scenarios") or [backtest]
    scenario_cards = "".join(
        "<article class='workflow-card'>"
        f"<h3>{escape(str(row.get('tenor') or 'n/a'))} scenario</h3>"
        f"<span class='pill amber'>{escape(str(row.get('evidence_status') or evidence_status))}</span>"
        f"<div class='grid-3' style='margin-top:14px'><div class='kpi'><div class='label'>History</div><div class='value'>{escape(str(row.get('snapshot_count') or 0))} snapshots</div></div><div class='kpi'><div class='label'>Synthetic trades</div><div class='value'>{escape(str(row.get('trade_count') or 0))} synthetic trades</div></div><div class='kpi'><div class='label'>Gross PnL</div>{_claim_control_pill()}<div class='value'>{escape(_money(row.get('gross_pnl')))}</div></div><div class='kpi'><div class='label'>Max drawdown</div>{_claim_control_pill()}<div class='value'>{escape(_money(row.get('max_drawdown')))}</div></div><div class='kpi'><div class='label'>Win rate</div><div class='value'>{escape(_pct(row.get('win_rate')))}</div></div><div class='kpi'><div class='label'>Cost/trade</div>{_claim_control_pill()}<div class='value'>{escape(_money(row.get('cost_per_trade')))}</div></div></div>"
        f"<p class='workflow'>Sample gate: {escape(str(row.get('sample_gate') or sample_gate))} · {escape(str(row.get('slippage_vol_pts') if row.get('slippage_vol_pts') is not None else 'n/a'))} vol pts slippage · synthetic fills only.</p>"
        "</article>"
        for row in scenarios
    )
    controls_html = "".join(f"<li><span>Control</span><code>{escape(str(control))}</code></li>" for control in (backtest.get("controls") or []))
    if not controls_html:
        controls_html = "<li><span>Control</span><code>synthetic fills; not executable economics</code></li>"
    robustness = backtest.get("robustness_metrics") or {}
    cost = robustness.get("cost_sensitivity") or {}
    robustness_html = ""
    if robustness:
        robustness_html = f"""
        <div class="panel-lite"><h3>Backtest Robustness</h3><div class="grid-3"><div class="kpi"><div class="label">Sample gate pass count</div><div class="value">{escape(str(robustness.get('sample_gate_pass_count') or 0))} / {escape(str(robustness.get('scenario_count') or 0))}</div></div><div class="kpi"><div class="label">Minimum gates</div><div class="value">{escape(str(robustness.get('min_snapshot_gate') or 0))} snapshots / {escape(str(robustness.get('min_trade_gate') or 0))} synthetic trades</div></div><div class="kpi"><div class="label">Effective cost/trade</div>{_claim_control_pill()}<div class="value">{escape(_money(cost.get('effective_cost_per_trade')))}</div></div></div><p class="control-note wide">Tenors with synthetic trades: {escape(', '.join(robustness.get('tenors_with_trades') or []) or 'none')}. Sample gate ready: {escape(str(robustness.get('sample_gate_ready')))}. {escape(str(cost.get('control_note') or 'Synthetic fills only; not executable economics.'))}</p></div>
        """
    summary_json_link = ""
    if backtest.get("summary_json_path"):
        summary_json_link = f"<li><span>Backtest summary JSON</span><code>{escape(str(backtest.get('summary_json_path')))}</code></li><li><span>Summary JSON SHA-256</span><code>{escape(str(backtest.get('summary_json_sha256') or 'missing'))}</code></li>"
    return f"""
    <section id="backtest" class="panel"><h2>Backtest / Research Evidence</h2><span class="pill amber">{escape(evidence_status)}</span><p>{escape(str(backtest.get('control_note') or 'Point-in-time research only using synthetic fills; not executable economics and not an investment conclusion.'))}</p><div class="quote-board-grid">{scenario_cards}</div>{robustness_html}<p class="control-note wide">Sample gate: {escape(sample_gate)}. Backtest PnL is a scaffold for research hygiene, not an investment conclusion.</p><ul class="clean evidence-list"><li><span>Backtest report</span><code>{escape(str(backtest.get('path') or 'missing'))}</code></li><li><span>Backtest SHA-256</span><code>{escape(str(backtest.get('sha256') or 'missing'))}</code></li>{summary_json_link}{controls_html}</ul></section>
    """


def _source_diagnostics_panel(diagnostics: dict[str, Any]) -> str:
    evidence_status = str(diagnostics.get("evidence_status") or "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE")
    required = diagnostics.get("required_source_labels") or []
    required_html = "".join(f"<span class='pill violet'>{escape(str(label))}</span>" for label in required)
    coverage_matrix = diagnostics.get("coverage_matrix") or []
    coverage_html = "".join(
        f"<span class='pill {'green' if str(row.get('status')) == 'covered' else 'amber'}'>{escape(str(row.get('label')))} {escape(str(row.get('status')))}</span>"
        for row in coverage_matrix
    )
    missing = diagnostics.get("missing_required_source_labels") or []
    missing_html = ", ".join(escape(str(label)) for label in missing) if missing else "none"
    tracker_rows = diagnostics.get("source_coverage_tracker") or []
    tracker_html = "".join(
        "<article class='workflow-card'>"
        f"<h3>{escape(str(row.get('label') or 'Source'))} · {escape(str(row.get('status') or 'unknown'))}</h3>"
        f"<span class='pill {'green' if str(row.get('status')) == 'covered' else 'amber'}'>{escape(str(row.get('execution_confidence') or 'screen_only_not_executable'))}</span>"
        f"<p class='workflow'>Provider: {escape(str(row.get('provider') or 'missing'))} · License: {escape(str(row.get('license_label') or 'missing'))} · Rows: {escape(_num(row.get('row_count')))}</p>"
        f"<p class='workflow'>Coverage: {escape(str(row.get('coverage_start') or 'n/a'))} → {escape(str(row.get('coverage_end') or 'n/a'))} · Sources: {escape(str(row.get('source_count') or 0))}</p>"
        f"<ul class='clean evidence-list'><li><span>SHA-256</span><code>{escape(str(row.get('sha256') or 'missing'))}</code></li><li><span>Blocker</span><code>{escape(str(row.get('blocker') or 'missing'))}</code></li><li><span>Next action</span><code>{escape(str(row.get('next_action') or 'missing'))}</code></li></ul>"
        "</article>"
        for row in tracker_rows
    )
    if not tracker_html:
        tracker_html = "<p class='muted'>No source tracker rows available.</p>"
    sources = diagnostics.get("sources") or []
    if not diagnostics.get("available"):
        source_cards = "<p class='muted'>No historical source manifest available yet.</p>"
    else:
        source_cards = "".join(
            "<article class='workflow-card'>"
            f"<h3>{escape(str(row.get('source_name') or 'Historical source'))}</h3>"
            f"<span class='pill amber'>{escape(str(row.get('evidence_status') or evidence_status))}</span>"
            f"<p class='workflow'>{escape(str(row.get('venue') or 'venue'))} · {escape(str(row.get('provider') or 'provider'))} · {escape(str(row.get('instrument_scope') or 'scope'))}</p>"
            f"<p class='workflow'>License: {escape(str(row.get('license_label') or 'unknown'))} · Redistribution: {escape(str(row.get('redistribution') or 'internal_only'))} · Execution: {escape(str(row.get('execution_confidence') or 'screen_only_not_executable'))}</p>"
            f"<p class='workflow'>Coverage: {escape(str(row.get('coverage_start') or 'n/a'))} → {escape(str(row.get('coverage_end') or 'n/a'))}</p>"
            f"<ul class='clean evidence-list'><li><span>Raw source</span><code>{escape(str(row.get('raw_path') or 'missing'))}</code></li><li><span>SHA-256</span><code>{escape(str(row.get('sha256') or 'missing'))}</code></li></ul>"
            "</article>"
            for row in sources[:4]
        )
    return f"""
    <section id="source-diagnostics" class="panel"><div class="section-head"><div><h2>Source Diagnostics</h2><p>{escape(str(diagnostics.get('control_note') or 'Historical backtests use hashed point-in-time sources; synthetic fills only; not executable economics.'))}</p></div><span class="pill amber">{escape(evidence_status)}</span></div><div class="grid-3"><div class="kpi"><div class="label">Historical source manifest</div><div class="value">{escape(str(diagnostics.get('source_count') or 0))} sources</div></div><div class="kpi"><div class="label">Required-source coverage</div><div class="value">{escape(str(diagnostics.get('coverage_summary') or 'unknown'))}</div></div><div class="kpi"><div class="label">Manifest hash</div><div class="value"><code>{escape(_hash(diagnostics.get('manifest_sha256')))}</code></div></div></div><div class="compact-proof"><span>Required universe</span><strong>{required_html}</strong><span>Coverage gaps</span><strong>{missing_html}</strong><span>Coverage matrix</span><strong>{coverage_html}</strong></div><details class="disclosure"><summary>Operational source coverage tracker</summary><p class="control-note wide">{escape(str(diagnostics.get('source_tracker_summary') or 'tracker unavailable'))}. This is an operational ledger for source acquisition and replay readiness, not a trading signal.</p><div class="quote-board-grid">{tracker_html}</div></details><details class="disclosure"><summary>Historical source files and replay controls</summary><div class="quote-board-grid">{source_cards}</div><ul class="clean evidence-list"><li><span>Historical source manifest</span><code>{escape(str(diagnostics.get('manifest_path') or 'missing'))}</code></li><li><span>Manifest SHA-256</span><code>{escape(str(diagnostics.get('manifest_sha256') or 'missing'))}</code></li><li><span>Replay controls</span><code>PIT hashed · synthetic fills · not executable economics</code></li></ul></details></section>
    """


def _quote_evidence_ledger_panel(ledger: dict[str, Any]) -> str:
    summary = ledger.get("summary") or {}
    candidates = summary.get("candidates") or []
    records = ledger.get("records") or []
    quote_records = int(summary.get("quote_verified_records") or 0)
    manual_records = int(summary.get("manual_indicative_records") or 0)
    quote_candidates = int(summary.get("quote_verified_candidates") or 0)
    trade_candidates = int(summary.get("trade_verified_candidates") or 0)
    invalid_records = int(summary.get("invalid_records") or ledger.get("invalid_count") or 0)
    rows_html = "".join(
        "<article class='workflow-card'>"
        f"<div class='rank'>{escape(str(row.get('candidate_id') or 'candidate'))}</div>"
        f"<h3>{escape(str(row.get('structure') or 'Quote evidence candidate'))}</h3>"
        f"<div class='stage'>{escape(str(row.get('stage') or 'internal diligence'))}</div>"
        f"<p class='workflow'>Indicative quotes: {escape(str(row.get('indicative_quote_count') or 0))} · Avg mid IV: {escape(_num(row.get('avg_mid_iv')))} vol pts · Publishability: {escape(str(row.get('publishability') or 'not-investor-publishable'))}</p>"
        f"<p class='workflow'>Counterparties: {escape(', '.join(row.get('counterparty_pseudonyms') or row.get('counterparties') or ['none captured']))}</p>"
        f"<span class='pill amber'>{escape(str(row.get('evidence_status') or 'SCREEN-ONLY · NOT EXECUTABLE'))}</span>"
        "</article>"
        for row in candidates[:3]
    )
    if not rows_html:
        rows_html = "<p class='muted'>No manual quote evidence captured yet. Screen-only candidates remain internal review items.</p>"
    controls_html = "".join(
        f"<li><span>Quote control</span><code>{escape(str(control))}</code></li>"
        for control in (summary.get("controls") or [])
    )
    if not controls_html:
        controls_html = "<li><span>Quote control</span><code>Two distinct counterparties required for quote-verified candidate status</code></li>"
    sample_records_html = "".join(
        "<li>"
        f"<span>{escape(str(row.get('rfq_id') or 'rfq'))}</span>"
        f"<code>{escape(str(row.get('counterparty') or 'counterparty'))} · {escape(str(row.get('execution_confidence') or 'screen-only'))} · spread {escape(_num(row.get('spread_vol_pts')))} vol pts · {escape(str(row.get('publishability') or 'not-investor-publishable'))}</code>"
        "</li>"
        for row in records[:3]
    )
    if not sample_records_html:
        sample_records_html = "<li><span>Quote records</span><code>none captured</code></li>"
    return f"""
    <section id="quote-evidence" class="panel"><h2>Quote Evidence Ledger</h2><p>Manual evidence capture for diligence only. Demo/manual indicative rows do not count as real quote-verified evidence. Real quote-verified evidence remains internal diligence only until trade records and the sponsor's counsel-approved business/legal wrapper permit external use.</p><div class="grid-3"><div class="kpi"><div class="label">Demo/manual indicative records</div><div class="value">{manual_records} demo records</div></div><div class="kpi"><div class="label">Real quote-verified records</div><div class="value">{quote_records} real records</div></div><div class="kpi"><div class="label">Trade-verified candidates</div><div class="value">{trade_candidates} trade-verified candidates</div></div></div><p class="control-note wide">Not investor-publishable: indicative quotes are not executable and do not represent a client-facing RFQ, trade, or fund offering. Quote-verified candidates: {quote_candidates}. Invalid rows: {invalid_records}.</p><div class="quote-board-grid">{rows_html}</div><ul class="clean evidence-list">{controls_html}{sample_records_html}</ul></section>
    """


def _readiness_gate_panel(gate: dict[str, Any]) -> str:
    gates = gate.get("gates") or []
    gate_rows = "".join(
        "<article class='workflow-card'>"
        f"<div class='rank'>{'PASS' if row.get('passed') else 'BLOCKED'}</div>"
        f"<h3>{escape(str(row.get('label') or 'Readiness gate'))}</h3>"
        f"<div class='stage'>{escape('passed' if row.get('passed') else 'not ready')}</div>"
        f"<p class='workflow'>Required: {escape(str(row.get('required') or 'missing'))}</p>"
        f"<p class='workflow'>Blocker: {escape(str(row.get('blocker') or 'none'))}</p>"
        "</article>"
        for row in gates
    )
    blockers = gate.get("blockers") or []
    blockers_html = "".join(f"<li><span>Blocker</span><code>{escape(str(blocker))}</code></li>" for blocker in blockers)
    if not blockers_html:
        blockers_html = "<li><span>Blocker</span><code>none captured</code></li>"
    next_actions = gate.get("next_actions") or []
    next_actions_html = "".join(f"<li><span>Next action</span><code>{escape(str(action))}</code></li>" for action in next_actions)
    if not next_actions_html:
        next_actions_html = "<li><span>Next action</span><code>not captured</code></li>"
    return f"""
    <section id="readiness" class="panel"><div class="section-head"><div><h2>Institutional Readiness Gate</h2><p>{escape(str(gate.get('control_note') or 'Readiness gates are diligence controls, not approval to trade, advise, market, or raise capital.'))}</p></div><span class="pill amber">{escape(str(gate.get('label') or 'NOT READY FOR INVESTMENT/CLIENT USE'))}</span></div><div class="grid-3"><div class="kpi"><div class="label">Gate status</div><div class="value">{escape(str(gate.get('summary') or '0/4 readiness gates passed'))}</div></div><div class="kpi"><div class="label">Evidence status</div><div class="value">{escape(str(gate.get('evidence_status') or 'SCREEN-ONLY · NOT EXECUTABLE'))}</div></div><div class="kpi"><div class="label">External use</div><div class="value">BLOCKED</div></div></div><div class="quote-board-grid">{gate_rows}</div><details class="disclosure" open><summary>Blockers and remediation plan</summary><ul class="clean evidence-list">{blockers_html}</ul><h3>Readiness Remediation Plan</h3><ul class="clean evidence-list">{next_actions_html}</ul></details></section>
    """


def _current_source_availability_panel(availability: dict[str, Any]) -> str:
    if not availability:
        return ""
    groups = availability.get("groups") or []
    group_html = "".join(
        "<article class='workflow-card'>"
        f"<h3>{escape(str(row.get('label') or 'Source'))} · {escape(str(row.get('status') or 'unknown'))}</h3>"
        f"<span class='pill {'green' if str(row.get('status')) == 'available' else 'amber'}'>{escape(str(row.get('evidence_status') or availability.get('evidence_status') or 'SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE'))}</span>"
        f"<p class='workflow'>Provider: {escape(str(row.get('provider') or 'missing'))} · License: {escape(str(row.get('license_label') or 'missing'))} · Rows: {escape(_num(row.get('row_count')))}</p>"
        f"<ul class='clean evidence-list'><li><span>SHA-256</span><code>{escape(str(row.get('sha256') or 'missing'))}</code></li><li><span>Status note</span><code>{escape(str(row.get('blocker') or 'internal diligence only'))}</code></li></ul>"
        "</article>"
        for row in groups
    )
    if not group_html:
        group_html = "<p class='muted'>No current source availability rows captured.</p>"
    return f"""
    <section id="current-sources" class="panel"><h2>Current Source Availability</h2><span class="pill amber">{escape(str(availability.get('evidence_status') or 'SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE'))}</span><p>{escape(str(availability.get('control_note') or 'Current captures are internal diligence only and do not clear licensed historical readiness.'))}</p><div class="grid-3"><div class="kpi"><div class="label">Current availability</div><div class="value">{escape(str(availability.get('summary') or '0/6 current source groups available'))}</div></div><div class="kpi"><div class="label">Internal diligence</div><div class="value">{escape('READY' if availability.get('ready_for_internal_diligence') else 'PARTIAL')}</div></div><div class="kpi"><div class="label">Investment/client use</div><div class="value">{escape('READY' if availability.get('ready_for_investment_or_client_use') else 'BLOCKED')}</div></div></div><div class="quote-board-grid">{group_html}</div></section>
    """


def _source_intake_contract_panel(contract: dict[str, Any]) -> str:
    if not contract:
        return ""
    source_rows = "".join(
        "<article class='workflow-card'>"
        f"<div class='rank'>{escape(str(row.get('source_group') or 'Source group'))}</div>"
        f"<h3>{escape(str(row.get('source_group') or 'Source group'))}</h3>"
        f"<p class='workflow'>accepted formats: {escape(', '.join(str(item) for item in (row.get('accepted_formats') or [])))}</p>"
        f"<p class='workflow'>required fields: {escape(', '.join(str(item) for item in (row.get('required_fields') or [])))}</p>"
        f"<p class='workflow'>provider examples: {escape(', '.join(str(item) for item in (row.get('provider_examples') or [])))}</p>"
        "</article>"
        for row in contract.get("required_sources", [])
    )
    gates_html = "".join(
        f"<li><span>Validation gate</span><code>{escape(str(gate))}</code></li>"
        for gate in contract.get("validation_gates", [])
    )
    validation = contract.get("validation_result") or {}
    blockers = validation.get("blockers") or []
    blocker_html = "".join(f"<li><span>Intake blocker</span><code>{escape(str(blocker))}</code></li>" for blocker in blockers[:6])
    if not blocker_html:
        blocker_html = "<li><span>Intake blocker</span><code>none captured</code></li>"
    validation_html = f"""
    <h3>Source Intake Validation</h3><div class="grid-3"><div class="kpi"><div class="label">Validation status</div><div class="value">{escape('READY' if validation.get('ready') else 'NOT READY')}</div></div><div class="kpi"><div class="label">Coverage</div><div class="value">{escape(str(validation.get('covered_source_groups', 0)))}/{escape(str(validation.get('required_source_groups', 6)))} source groups structurally valid</div></div><div class="kpi"><div class="label">Evidence status</div><div class="value">{escape(str(validation.get('evidence_status') or 'SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE'))}</div></div></div><ul class="clean evidence-list">{blocker_html}</ul>
    """
    return f"""
    <section id="source-intake" class="panel"><div class="section-head"><div><h2>Licensed Source Intake Contract</h2><p>{escape(str(contract.get('control_note') or 'Defines required licensed/replay-ready source package.'))}</p></div><span class="pill amber">{escape(str(contract.get('evidence_status') or 'SCREEN-ONLY · INTAKE CONTRACT · NOT EXECUTABLE'))}</span></div>{validation_html}<details class="disclosure"><summary>Required licensed source groups and accepted formats</summary><div class="quote-board-grid">{source_rows}</div></details><details class="disclosure"><summary>Validation gates</summary><ul class="clean evidence-list">{gates_html}</ul></details></section>
    """



def _legal_wrapper_panel(wrapper: dict[str, Any]) -> str:
    matrix = wrapper.get("boundary_matrix") or []
    matrix_html = "".join(
        "<article class='workflow-card'>"
        f"<h3>{escape(str(row.get('activity') or 'Activity'))}</h3>"
        f"<div class='stage'>{escape(str(row.get('current_status') or 'unknown'))}</div>"
        f"<p class='workflow'>Allowed now: {escape(str(row.get('allowed_now')))} · External requires: {escape(str(row.get('external_requires') or 'counsel approval'))}</p>"
        "</article>"
        for row in matrix
    ) or "<p class='muted'>No legal boundary matrix available.</p>"
    blockers = wrapper.get("blockers") or []
    blocker_html = "".join(f"<li><span>Legal blocker</span><code>{escape(str(item))}</code></li>" for item in blockers) or "<li><span>Legal blocker</span><code>none captured</code></li>"
    gates = wrapper.get("hard_gates") or []
    gate_html = "".join(f"<li><span>Hard gate</span><code>{escape(str(item))}</code></li>" for item in gates)
    return f"""
    <section id="legal-wrapper" class="panel"><div class="section-head"><div><h2>Legal / Business Wrapper</h2><p>{escape(str(wrapper.get('control_note') or 'Counsel-approved wrapper required before external use.'))}</p></div><span class="pill amber">{escape(str(wrapper.get('label') or 'DRAFT — NOT APPROVED FOR EXTERNAL USE'))}</span></div><div class="grid-3"><div class="kpi"><div class="label">Counsel approval</div><div class="value">{escape(str(wrapper.get('approved_by_counsel', False)).upper())}</div></div><div class="kpi"><div class="label">Wrapper status</div><div class="value">{escape(str(wrapper.get('status') or 'draft-blocked'))}</div></div><div class="kpi"><div class="label">External use</div><div class="value">BLOCKED</div></div></div><details class="disclosure"><summary>Legal boundary matrix and hard gates</summary><div class="quote-board-grid">{matrix_html}</div><ul class="clean evidence-list">{blocker_html}{gate_html}</ul></details></section>
    """

def render_investor_site_html(data: dict[str, Any]) -> str:
    latest = data.get("latest_run") or {}
    evidence_status = str(data.get("evidence_status") or "SCREEN-ONLY · NOT EXECUTABLE")
    positioning = str(data.get("full_positioning") or data.get("positioning") or "BTC Treasury & Miner Hedging Desk")
    candidates = data.get("top_candidates") or []
    business_steps = data.get("business_model_sequence") or []
    business_html = "".join(f"<li>{escape(str(step))}</li>" for step in business_steps)
    case_studies = data.get("case_studies") or {}
    quote_board = data.get("quote_verification_board") or {}
    quote_evidence = data.get("quote_evidence_ledger") or {}
    backtest_research = data.get("backtest_research") or {}
    source_diagnostics = data.get("historical_source_diagnostics") or {}
    current_source_availability = data.get("current_source_availability") or {}
    readiness_gate = data.get("institutional_readiness_gate") or {}
    source_intake_contract = data.get("licensed_source_intake_contract") or {}
    legal_wrapper = data.get("legal_wrapper_package") or {}
    legal_legend = str(data.get("legal_legend") or "Internal evidence prototype. Public screen/model data only. No RFQ sent. No executable quote.")
    legal_gate = str(data.get("legal_gate") or "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow.")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>BTC Vol Desk — Investor Proof of Concept</title>
  <style>
    :root {{ --bg:#08090a; --panel:#0f1011; --surface:rgba(255,255,255,.035); --surface2:rgba(255,255,255,.055); --border:rgba(255,255,255,.08); --text:#f7f8f8; --muted:#8a8f98; --soft:#d0d6e0; --violet:#7170ff; --gold:#f5b841; --green:#34d399; --red:#fb7185; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; background:radial-gradient(circle at 20% 0%, rgba(113,112,255,.18), transparent 34%), radial-gradient(circle at 85% 12%, rgba(245,184,65,.12), transparent 28%), var(--bg); color:var(--text); font-family:Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; font-feature-settings:'cv01','ss03'; }}
    a {{ color:inherit; text-decoration:none; }}
    .shell {{ max-width:1280px; margin:0 auto; padding:0 28px 64px; }}
    nav {{ position:sticky; top:0; z-index:10; backdrop-filter:blur(18px); background:rgba(8,9,10,.78); border-bottom:1px solid rgba(255,255,255,.06); }}
    .nav-inner {{ max-width:1280px; margin:0 auto; padding:14px 28px; display:flex; align-items:center; justify-content:space-between; gap:20px; }}
    .brand {{ font-weight:600; letter-spacing:-.03em; white-space:nowrap; }}
    .nav-links {{ display:flex; flex-wrap:wrap; justify-content:center; gap:12px 16px; color:var(--muted); font-size:13px; }}
    .badge,.pill {{ display:inline-flex; align-items:center; border-radius:999px; border:1px solid var(--border); padding:7px 10px; font-size:11px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; }}
    .badge {{ color:var(--gold); background:rgba(245,184,65,.1); border-color:rgba(245,184,65,.32); }}
    .pill.green {{ color:var(--green); background:rgba(52,211,153,.1); }} .pill.amber {{ color:var(--gold); background:rgba(245,184,65,.1); }} .pill.violet {{ color:#c7c8ff; background:rgba(113,112,255,.13); }}
    .hero {{ padding:92px 0 58px; display:grid; grid-template-columns:1.15fr .85fr; gap:34px; align-items:center; }}
    h1 {{ margin:18px 0 18px; font-size:64px; line-height:.96; letter-spacing:-1.5px; font-weight:500; }}
    h2 {{ margin:0 0 16px; font-size:30px; line-height:1.08; letter-spacing:-.7px; font-weight:520; }}
    h3 {{ margin:0 0 10px; font-size:18px; letter-spacing:-.24px; }}
    p {{ color:var(--soft); line-height:1.65; }}
    .lead {{ font-size:19px; color:#dce3ee; max-width:760px; }}
    .control-note {{ color:#e7edf7; background:rgba(245,184,65,.08); border:1px solid rgba(245,184,65,.24); border-radius:14px; padding:13px 14px; max-width:760px; }}
    .control-note.small {{ font-size:13px; margin-top:14px; }}
    .control-note.wide {{ grid-column:1 / -1; max-width:none; margin:0 0 -4px; }}
    .panel {{ background:linear-gradient(180deg, rgba(255,255,255,.055), rgba(255,255,255,.025)); border:1px solid var(--border); border-radius:22px; padding:24px; box-shadow:0 24px 90px rgba(0,0,0,.35); }}
    .section-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:16px; }}
    .section-head p {{ margin:4px 0 0; max-width:860px; }}
    .proof-strip {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:0 0 20px; }}
    .proof-card {{ background:rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; }}
    .proof-card strong {{ display:block; margin-top:7px; font-size:17px; color:var(--text); letter-spacing:-.03em; }}
    .compact-proof {{ display:grid; grid-template-columns:180px minmax(0,1fr); gap:10px 14px; margin:16px 0; padding:14px; border-radius:16px; background:rgba(0,0,0,.18); border:1px solid rgba(255,255,255,.06); }}
    .compact-proof span {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
    .compact-proof strong {{ color:var(--soft); font-weight:500; overflow-wrap:anywhere; }}
    .disclosure {{ margin-top:14px; border:1px solid rgba(255,255,255,.08); border-radius:16px; background:rgba(0,0,0,.16); padding:0; overflow:hidden; }}
    .disclosure summary {{ cursor:pointer; padding:14px 16px; color:#e7edf7; font-weight:600; list-style:none; }}
    .disclosure summary::-webkit-details-marker {{ display:none; }}
    .disclosure summary:after {{ content:'+'; float:right; color:var(--gold); }}
    .disclosure[open] summary:after {{ content:'–'; }}
    .disclosure > :not(summary) {{ margin:14px 16px 16px; }}
    .hero-card {{ min-height:360px; display:flex; flex-direction:column; justify-content:space-between; }}
    .kpi-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
    .kpi {{ background:rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:14px; }}
    .label {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
    .value {{ margin-top:8px; font-size:22px; font-weight:600; letter-spacing:-.04em; }}
    section {{ margin:20px 0; }}
    .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
    .candidate-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; }}
    .candidate-card {{ background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:16px; }}
    .rank {{ color:var(--gold); font-family:'JetBrains Mono',monospace; font-size:12px; margin-bottom:8px; }}
    .metric {{ font-size:28px; font-weight:600; letter-spacing:-.05em; margin:10px 0; }}
    .workflow {{ color:var(--muted); font-size:13px; }}
    .case-metrics {{ display:grid; gap:10px; margin-top:18px; }}
    .case-metrics div {{ background:rgba(0,0,0,.22); border:1px solid rgba(255,255,255,.06); border-radius:14px; padding:13px; }}
    .case-metrics strong {{ display:block; color:var(--text); font-size:17px; letter-spacing:-.03em; overflow-wrap:anywhere; }}
    .case-metrics span {{ display:block; color:var(--muted); font-size:12px; margin-top:5px; }}
    .metric-label {{ display:inline-flex; margin-bottom:7px; padding:4px 7px; border-radius:999px; border:1px solid rgba(245,184,65,.22); color:var(--gold); background:rgba(245,184,65,.07); font-size:10px; letter-spacing:.07em; text-transform:uppercase; }}
    .quote-summary {{ display:grid; grid-template-columns:repeat(7, minmax(0,1fr)); gap:10px; margin:16px 0; }}
    .quote-board-grid {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px; margin:16px 0; }}
    .workflow-card {{ background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:16px; }}
    .stage {{ display:inline-flex; font-family:'JetBrains Mono',monospace; color:#c7c8ff; background:rgba(113,112,255,.13); border:1px solid rgba(113,112,255,.24); border-radius:999px; padding:6px 9px; font-size:11px; margin:6px 0; }}
    .timeline {{ display:grid; gap:10px; counter-reset:step; }}
    .timeline div {{ position:relative; padding:14px 14px 14px 46px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.07); border-radius:14px; color:var(--soft); }}
    .timeline div:before {{ counter-increment:step; content:counter(step); position:absolute; left:14px; top:13px; width:22px; height:22px; border-radius:50%; display:grid; place-items:center; background:rgba(113,112,255,.18); color:#c7c8ff; font-size:12px; font-weight:600; }}
    ul.clean {{ list-style:none; margin:0; padding:0; display:grid; gap:10px; }}
    ul.clean li {{ color:var(--soft); padding:10px 0; border-bottom:1px solid rgba(255,255,255,.06); }}
    code {{ font-family:'JetBrains Mono',monospace; color:#cbd5e1; font-size:12px; word-break:break-word; overflow-wrap:anywhere; }}
    .evidence-list li {{ display:grid; grid-template-columns:170px minmax(0,1fr); gap:12px; align-items:start; }}
    .muted {{ color:var(--muted); }}
    .ask {{ border-color:rgba(113,112,255,.28); background:linear-gradient(180deg, rgba(113,112,255,.13), rgba(255,255,255,.025)); }}
    footer {{ padding:24px 0; color:var(--muted); font-size:13px; }}
    @media (max-width:900px) {{ .hero,.grid-2,.grid-3,.candidate-grid,.quote-board-grid,.proof-strip {{ grid-template-columns:1fr; }} .quote-summary {{ grid-template-columns:repeat(2, minmax(0,1fr)); }} h1 {{ font-size:42px; line-height:1.02; }} .nav-links {{ display:none; }} .section-head {{ flex-direction:column; }} .compact-proof {{ grid-template-columns:1fr; }} }}
    @media (max-width:560px) {{ .shell {{ padding:0 14px 42px; }} .nav-inner {{ padding:12px 14px; gap:10px; }} .brand {{ font-size:14px; }} .badge,.pill {{ font-size:10px; padding:6px 8px; }} .hero {{ padding:52px 0 34px; }} h1 {{ font-size:34px; letter-spacing:-.8px; }} h2 {{ font-size:24px; }} .value {{ font-size:19px; }} .quote-summary,.kpi-grid {{ grid-template-columns:1fr; }} .evidence-list li {{ grid-template-columns:1fr; gap:5px; }} }}
  </style>
</head>
<body>
    <nav><div class="nav-inner"><div class="brand">BTC Vol Desk</div><div class="nav-links"><a href="#evidence">Evidence</a><a href="#legal-wrapper">Legal</a><a href="#cases">Case Studies</a><a href="#workflow">Quote Verification</a><a href="#backtest">Backtest</a><a href="#source-diagnostics">Sources</a><a href="#room">Evidence Room</a></div><div class="badge">{escape(evidence_status)}</div></div></nav>
  <main class="shell">
    <section class="hero">
      <div>
        <span class="pill violet">Desk / Platform First · Fund sleeve later</span>
        <h1>{escape(positioning)}</h1>
        <p class="lead">An institutional proof-of-concept for converting fragmented BTC volatility markets into treasury and miner hedging intelligence, then into quote-verified opportunities through a controlled RFQ evidence process.</p>
        <p class="control-note">{escape(legal_legend)} {escape(legal_gate)}</p>
      </div>
      <aside class="panel hero-card">
        <div><span class="badge">Latest evidence run</span><h2>{escape(str(latest.get('run_id','missing')))}</h2><p>{escape(str(latest.get('as_of_cst','missing')))}</p></div>
        <div class="kpi-grid">
          <div class="kpi"><div class="label">BTC reference</div><div class="value">{escape(_money(latest.get('btc_spot')))}</div></div>
          <div class="kpi"><div class="label">Dislocations</div><div class="value">{escape(_num(latest.get('dislocations')))}</div></div>
          <div class="kpi"><div class="label">Configured-source quality</div><div class="value">{escape(str(latest.get('quality_grade','n/a')).upper())}</div></div>
          <div class="kpi"><div class="label">Freshness</div><div class="value">{escape(str(latest.get('freshness_grade','n/a')).upper())}</div></div>
        </div>
        <p class="control-note small">Static screen evidence only; not live market data, not an executable quote, and not client-facing advice.</p>
      </aside>
    </section>

    <section class="proof-strip" aria-label="Executive proof status">
      <article class="proof-card"><div class="label">Internal packet integrity</div><strong>Verifier PASS</strong><p class="workflow">Hashes, controls, credential scan, and CTA scan pass.</p></article>
      <article class="proof-card"><div class="label">External use</div><strong>BLOCKED</strong><p class="workflow">Legal, quote, and licensed-source gates are not cleared.</p></article>
      <article class="proof-card"><div class="label">Current source availability</div><strong>{escape(str(latest.get('coverage_completeness','Current screen/vendor captures partial: CME unavailable')))}</strong><p class="workflow">Current captures support internal review only.</p></article>
      <article class="proof-card"><div class="label">Licensed readiness</div><strong>{escape(str((source_intake_contract.get('validation_result') or {}).get('covered_source_groups', 0)))}/{escape(str((source_intake_contract.get('validation_result') or {}).get('required_source_groups', 6)))} groups ready</strong><p class="workflow">Required before external use.</p></article>
    </section>

    <section class="grid-3">
      <article class="panel"><h3>Problem</h3><p>BTC volatility is fragmented across ETF options, Deribit, CME, and OTC desks. Treasuries and miners need hedging programs, not retail crypto trading screens.</p></article>
      <article class="panel"><h3>Solution</h3><p>A cross-venue evidence engine normalizes exposures, labels source confidence, and identifies review candidates before any quote or execution claim is made.</p></article>
      <article class="panel"><h3>Business</h3><p>Lead with hedge structuring and evidence workflow. Add RFQ partnerships and a fund/risk sleeve only after the chosen wrapper supports it.</p></article>
    </section>

    <section id="evidence" class="panel"><h2>Latest Static Evidence Snapshot</h2><div class="grid-3"><div class="kpi"><div class="label">IBIT BTC/share</div><div class="value">{escape(str(latest.get('btc_per_share','n/a')))}</div></div><div class="kpi"><div class="label">{escape(str(latest.get('coverage_completeness_label','Current screen-source availability')))}</div><div class="value">{escape(str(latest.get('coverage_completeness','Current screen/vendor captures partial: CME unavailable')))}</div></div><div class="kpi"><div class="label">Bundle hash</div><div class="value"><code>{escape(_hash(latest.get('evidence_bundle_sha256')))}</code></div></div></div><p class="muted">Static as of run timestamp; not real-time market data. All displayed economics are public screen marks or model-estimated values. They are evidence inputs, not executable quotes. Overall evidence readiness: {escape(str(latest.get('overall_evidence_readiness','YELLOW until CME/licensed feed and real quote evidence exist')))}.</p><div class="candidate-grid">{_candidate_cards(candidates)}</div></section>

    {_readiness_gate_panel(readiness_gate)}

    {_legal_wrapper_panel(legal_wrapper)}

    {_current_source_availability_panel(current_source_availability)}

    {_source_intake_contract_panel(source_intake_contract)}

    <section id="cases" class="grid-2">
      <p class="control-note wide">Scenario outputs are model-estimated and screen-only; no executable quote, client suitability review, or counterparty commitment is implied.</p>
      {_case_study_panels(case_studies)}
    </section>

    {_quote_verification_board(quote_board)}

    {_backtest_research_panel(backtest_research)}

    {_source_diagnostics_panel(source_diagnostics)}

    {_quote_evidence_ledger_panel(quote_evidence)}

    <section class="grid-2">
      <article class="panel"><h2>Desk / Platform First</h2><ul class="clean">{business_html}</ul><p class="muted">This is the recommended Option B route: prove the desk and platform first; use the fund sleeve as an expansion path, not the opening claim.</p></article>
      <article class="panel ask"><h2>Investor Diligence Ask</h2><p>Use the site to ask for seed/build capital, strategic OTC/broker introductions, legal structuring support, or a pilot treasury/miner mandate.</p><p><strong>Fund sleeve later</strong> once data history, quote evidence, and wrapper are strong enough.</p></article>
    </section>

    <section id="room" class="panel"><h2>Evidence Room</h2><p>Artifacts below are generated by the current evidence monitor and should travel with the investor memo.</p><ul class="clean evidence-list">{_evidence_link('Report', latest.get('report_path'))}{_evidence_link('Internal dashboard', latest.get('dashboard_path'))}{_evidence_link('Evidence bundle', latest.get('evidence_bundle_path'))}{_evidence_link('Evidence manifest', latest.get('evidence_manifest_path'))}{_evidence_link('Candidate ledger', latest.get('candidate_ledger_path'))}{_evidence_link('Quote evidence ledger', latest.get('quote_evidence_ledger_path'))}<li><span>Bundle SHA-256</span><code>{escape(str(latest.get('evidence_bundle_sha256') or 'missing'))}</code></li><li><span>Evidence status</span><code>{escape(evidence_status)}</code></li></ul></section>

    <footer>Control language: screen-only market data, model-estimated IVs, internal proof-of-concept, not executable, not a client portal, not an offer to sell securities or provide advisory/execution services.</footer>
  </main>
</body>
</html>"""


def write_investor_site(data: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_investor_site_html(data), encoding="utf-8")
    return target
