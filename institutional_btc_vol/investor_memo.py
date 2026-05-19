from __future__ import annotations

from pathlib import Path
from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
LEGAL_GATE = "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow."
LEGAL_LEGEND = "Internal evidence prototype. Public screen/model data only. No RFQ sent. No executable quote."


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:,.2f}".rstrip("0").rstrip(".")


def _stage_label(stage: str) -> str:
    labels = {
        "screen_only": "Screen-only",
        "reviewed": "Reviewed",
        "rfq_package_drafted": "Internal RFQ draft",
        "indicative_quote_1": "Indicative quote 1",
        "indicative_quote_2": "Indicative quote 2",
        "quote_verified": "Quote verified",
        "trade_verified": "Post-trade record verified",
    }
    return labels.get(stage, stage.replace("_", " ").title())


def _artifact_line(label: str, value: Any) -> str:
    if not value:
        return f"- {label}: `missing`"
    return f"- {label}: `{value}`"


def _candidate_lines(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "- No screen-only candidates captured in the latest run."
    lines = []
    for row in candidates[:5]:
        spread = row.get("gross_iv_diff_vol_pts")
        spread_text = f"{float(spread):+.2f} vol pts" if spread is not None else "n/a"
        lines.append(
            f"- #{row.get('rank', '—')} {row.get('candidate', 'Candidate')} — {spread_text}; "
            f"priority `{row.get('priority', 'review')}`; {row.get('evidence_status', EVIDENCE_STATUS)}."
        )
    return "\n".join(lines)


def _business_lines(steps: list[Any]) -> str:
    if not steps:
        return "- Research/evidence engine\n- Treasury and miner hedge structuring\n- Partner-led RFQ coordination only after legal wrapper"
    return "\n".join(f"{idx}. {step}" for idx, step in enumerate(steps, start=1))


def _quote_summary_lines(summary: dict[str, Any]) -> str:
    if not summary:
        return "- No quote verification stages populated."
    return "\n".join(f"- {_stage_label(str(stage))}: {count}" for stage, count in summary.items())


def render_investor_memo_markdown(data: dict[str, Any]) -> str:
    latest = data.get("latest_run") or {}
    treasury = (data.get("case_studies") or {}).get("treasury") or {}
    miner = (data.get("case_studies") or {}).get("miner") or {}
    quote_board = data.get("quote_verification_board") or {}
    candidates = data.get("top_candidates") or []
    business_steps = data.get("business_model_sequence") or []
    positioning = data.get("positioning") or "BTC Treasury & Miner Hedging Desk"
    full_positioning = data.get("full_positioning") or f"{positioning} powered by a purpose-built cross-venue volatility evidence engine"
    legal_legend = data.get("legal_legend") or LEGAL_LEGEND
    legal_gate = data.get("legal_gate") or LEGAL_GATE

    return f"""# BTC Treasury & Miner Hedging Desk Investor Memo

**Status:** {data.get('evidence_status', EVIDENCE_STATUS)}
**Positioning:** {full_positioning}

> {legal_legend}

> Not a client portal. Not an execution venue. Not a fund offering. This memo summarizes an internal proof-of-concept and evidence workflow.

## Thesis

BTC volatility markets are fragmented across ETF-listed options, offshore crypto options venues, CME-linked institutional channels, and OTC counterparties. Corporate BTC treasuries and miners need hedge design, evidence integrity, and quote-verification discipline more than another generic crypto trading screen.

The opportunity is a desk/platform: use a purpose-built cross-venue volatility evidence engine to identify screen-only dislocations, convert them into treasury/miner hedge structures, and promote only independently documented opportunities through a manual quote-verification workflow.

## Evidence Snapshot

- Run ID: `{latest.get('run_id', 'missing')}`
- As-of CST: `{latest.get('as_of_cst', 'missing')}`
- BTC reference: `{_money(latest.get('btc_spot'))}`
- IBIT BTC/share: `{latest.get('btc_per_share', 'missing')}`
- Configured-source quality: `{str(latest.get('quality_grade', 'missing')).upper()}` / `{latest.get('quality_score', 'n/a')}`
- {latest.get('coverage_completeness_label', 'Current screen-source availability')}: `{latest.get('coverage_completeness', 'Current screen/vendor captures partial: CME unavailable')}`
- Overall evidence readiness: `{latest.get('overall_evidence_readiness', 'YELLOW until CME/licensed feed and real quote evidence exist')}`
- Freshness: `{str(latest.get('freshness_grade', 'missing')).upper()}`
- Screen-only dislocations: `{latest.get('dislocations', 0)}`
- Quote-review flags: `{latest.get('quote_review_candidates', 0)}`
- Bundle SHA-256: `{latest.get('evidence_bundle_sha256', 'missing')}`

All displayed economics are public screen marks or model-estimated values. They are evidence inputs, not executable quotes.

## Screen-Only Candidate Examples

{_candidate_lines(candidates)}

## Treasury Case Study

- Structure: corporate BTC treasury hedge policy sleeve.
- Hedged BTC: {_num(treasury.get('hedged_btc'))} BTC hedged.
- Caveat: hypothetical scenario only; no suitability, premium, margin, tax, accounting, liquidity, or counterparty commitment is implied.
- Illustrative model output — protected value: {_money(treasury.get('protected_value_at_floor_usd'))} illustrative floor-protected sleeve.
- Illustrative model output — floor / cap: {_money(treasury.get('floor_price'))} / {_money(treasury.get('cap_price'))}.
- Control: {treasury.get('quote_control', 'Premium and executable levels require quote verification.')}

## Miner Case Study

- Structure: runway-protection hedge for conservative monthly production.
- Hedged monthly production: {_num(miner.get('hedged_monthly_btc'))} BTC/month hedged.
- Caveat: hypothetical scenario only; no hedge recommendation, suitability review, margin model, tax/accounting treatment, or executable quote is implied.
- Illustrative model output — monthly floor revenue: {_money(miner.get('monthly_floor_revenue_on_hedged_btc_usd'))}.
- Pre-hedge cash runway: {_num(miner.get('cash_runway_months_before_hedge'))} months.
- Control: {miner.get('quote_control', 'Indicative economics require quote verification before investor/client use.')}

## Quote Verification Workflow

{quote_board.get('control', 'Manual demo workflow only; no RFQ is sent and no executable quote is implied.')}

Current stage counts:

{_quote_summary_lines(quote_board.get('summary') or {})}

Workflow gates:

1. Screen-only dislocation detected by the monitor.
2. Candidate triaged for materiality and structure fit.
3. RFQ package generated for review, not auto-sent.
4. After approval and manual outreach, counterparties may provide indicative quotes.
5. Evidence ledger may mark candidate quote-verified after required indicative quote records are captured.
6. Post-trade verification only occurs after an actual external execution record exists; this memo does not imply execution.

## Business Model Sequence

{_business_lines(business_steps)}

The recommended route remains desk/platform-first. The fund sleeve is an expansion path after evidence, legal wrapper, counterparty access, and investor demand are established.

## Investor Ask

- Seed build capital for data licensing, counterparty connectivity, compliance review, and proof-of-market pilots.
- Introductions to BTC treasury decision-makers, miners, OTC desks, derivatives counsel, and execution partners.
- Approval to harden the evidence engine into a controlled diligence workflow without launching a client portal or executable RFQ system prematurely.

## Evidence Room

{_artifact_line('Report', latest.get('report_path'))}
{_artifact_line('Evidence bundle', latest.get('evidence_bundle_path'))}
{_artifact_line('Evidence manifest', latest.get('evidence_manifest_path'))}
{_artifact_line('Candidate ledger', latest.get('candidate_ledger_path'))}
{_artifact_line('Bundle SHA-256', latest.get('evidence_bundle_sha256'))}

## Limitations / Gating Items

- {EVIDENCE_STATUS}: all current market economics are public screen marks, model-estimated IVs, or internal review outputs.
- CME remains unavailable until licensed/vendor/broker feed is configured.
- Quote-verified economics require two independently captured indicative quote records.
- Trade/post-trade verification requires an actual external execution record; this POC has no trading capability.
- Screen-only dislocation differences are gross IV observations only; they exclude bid/ask width, slippage, financing, borrow, margin, collateral, tax/accounting, settlement, exercise style, venue hours, operational constraints, and counterparty risk.
- Tenor matching between IBIT ETF options and Deribit BTC options is approximate and not an arbitrage claim or trading recommendation.
- {legal_gate}
"""


def write_investor_memo(data: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_investor_memo_markdown(data), encoding="utf-8")
    return target
