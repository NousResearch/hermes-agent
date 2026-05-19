from __future__ import annotations

from pathlib import Path
from typing import Any

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
LEGAL_GATE = "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow."
LEGAL_LEGEND = "Internal evidence prototype. Public screen/model data only. No RFQ sent. No executable quote."
CONTROL_NOTE = "Not a client portal. Not an execution venue. Not a fund offering."


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.2f}"


def _upper(value: Any) -> str:
    return str(value or "n/a").upper()


def _first_candidate(data: dict[str, Any]) -> dict[str, Any]:
    candidates = data.get("top_candidates") or []
    return candidates[0] if candidates else {}


def render_tearsheet_markdown(data: dict[str, Any]) -> str:
    latest = data.get("latest_run") or {}
    quote_summary = ((data.get("quote_evidence_ledger") or {}).get("summary") or {})
    first_candidate = _first_candidate(data)
    positioning = str(data.get("full_positioning") or data.get("positioning") or "BTC Treasury & Miner Hedging Desk")
    evidence_status = str(data.get("evidence_status") or EVIDENCE_STATUS)
    quote_records = int(quote_summary.get("quote_verified_records") or 0)
    manual_records = int(quote_summary.get("manual_indicative_records") or 0)
    quote_candidates = int(quote_summary.get("quote_verified_candidates") or 0)
    trade_candidates = int(quote_summary.get("trade_verified_candidates") or 0)
    top_candidate_name = str(first_candidate.get("candidate") or "No current candidate")
    top_candidate_spread = first_candidate.get("gross_iv_diff_vol_pts")
    top_candidate_line = (
        f"{top_candidate_name} ({float(top_candidate_spread):+.2f} vol pts, {first_candidate.get('evidence_status', evidence_status)})"
        if top_candidate_spread is not None
        else f"{top_candidate_name} ({evidence_status})"
    )
    return f"""# BTC Treasury & Miner Hedging Desk — One-Page Tear Sheet

**Positioning:** {positioning}
**Evidence status:** {evidence_status}
**Control:** {CONTROL_NOTE}
**Legal gate:** {data.get('legal_gate') or LEGAL_GATE}

> {data.get('legal_legend') or LEGAL_LEGEND}

> This is an internal/investor diligence summary. Public screen marks, model-estimated IVs, and manual indicative quote records are evidence inputs only — not executable economics.

## Evidence Snapshot

- Run ID: `{latest.get('run_id', 'missing')}`
- As-of CST: `{latest.get('as_of_cst', 'missing')}`
- BTC Reference: `{_money(latest.get('btc_spot'))}`
- Configured-source quality: `{_upper(latest.get('quality_grade'))}`
- {latest.get('coverage_completeness_label', 'Current screen-source availability')}: `{latest.get('coverage_completeness', 'Current screen/vendor captures partial: CME unavailable')}`
- Overall evidence readiness: `{latest.get('overall_evidence_readiness', 'YELLOW until CME/licensed feed and real quote evidence exist')}`
- Freshness: `{_upper(latest.get('freshness_grade'))}`
- Screen-only dislocations: `{latest.get('dislocations', 0)}`
- Quote-review flags: `{latest.get('quote_review_candidates', 0)}`
- Evidence bundle SHA-256: `{latest.get('evidence_bundle_sha256', 'missing')}`

## Why Now

- Corporate BTC treasuries and miners need risk-management infrastructure, not retail crypto trading screens.
- ETF options, Deribit, CME, and OTC liquidity fragment BTC volatility into inconsistent surfaces.
- A desk/platform can normalize exposures, preserve evidence, and separate screen-only signals from quote-verified diligence.

## Initial ICP

- BTC treasury holders needing board-safe covered-call, collar, or put-spread collar programs.
- Miners needing runway protection without overhedging production or creating collateral stress.
- Strategic investors/OTC partners who can provide broker, legal, or counterparty workflow support.

## Current Evidence

- Top screen-only candidate: {top_candidate_line}
- Quote evidence ledger: {manual_records} demo/manual indicative quote records, {quote_records} real quote-verified records, {quote_candidates} quote-verified candidates, {trade_candidates} trade-verified candidates.
- Real counterparty quote evidence captured: `{quote_records}` records / `{quote_candidates}` candidates.
- Publishability: `not-investor-publishable` until legal wrapper, counterparty permissions, and external records support wider use.

## Evidence Operations

1. Monitor public Deribit and IBIT option screens.
2. Normalize ETF exposure into BTC-equivalent terms.
3. Triage dislocations as `{EVIDENCE_STATUS}`.
4. Capture manual indicative quote evidence in JSONL when available.
5. Require two real counterparty indicative quote records before internal quote-verified status.
6. Require external execution record before post-trade verification.

## Diligence Ask

Use this packet to evaluate seed/build capital, strategic OTC/broker introductions, legal structuring support, or a pilot treasury/miner mandate. The fund/risk sleeve should remain a later-stage extension after evidence history, quote workflow, and the sponsor’s counsel-approved legal wrapper are stronger.

## Hard Boundaries

- {CONTROL_NOTE}
- No RFQ is sent by this packet.
- No executable quote, counterparty commitment, advisory service, or securities offering is implied.
- {data.get('legal_gate') or LEGAL_GATE}
"""


def write_tearsheet(data: dict[str, Any], output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_tearsheet_markdown(data), encoding="utf-8")
    return target
