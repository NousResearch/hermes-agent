from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from institutional_btc_vol.hedge_calculators import (
    build_miner_runway_case_study,
    build_treasury_hedge_case_study,
)
from institutional_btc_vol.quote_evidence import load_quote_evidence_ledger
from institutional_btc_vol.quote_workflow import build_quote_verification_demo_board
from institutional_btc_vol.source_intake import validate_source_intake_manifest
from institutional_btc_vol.legal_wrapper import load_legal_wrapper_package

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
POSITIONING = "BTC Treasury & Miner Hedging Desk"
FULL_POSITIONING = "BTC Treasury & Miner Hedging Desk powered by a purpose-built cross-venue volatility evidence engine"
LEGAL_LEGEND = "Internal evidence prototype. Public screen/model data only. No RFQ sent. No executable quote."
LEGAL_GATE = "Counsel-approved wrapper required before any external client, fund, RFQ, or execution workflow."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_path(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    forbidden = ["token=", "apikey", "api_key", "password", "secret="]
    if any(marker in text.lower() for marker in forbidden):
        return "[REDACTED]"
    return text


def _sha256_file(path_value: Any) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_candidates(path_value: Any, limit: int = 5) -> list[dict[str, Any]]:
    path = Path(str(path_value)) if path_value else None
    if path is None or not path.exists():
        return []
    candidates = []
    for row in _read_jsonl(path)[:limit]:
        candidates.append(
            {
                "rank": row.get("rank"),
                "candidate": row.get("candidate"),
                "priority": row.get("priority", "review"),
                "direction": row.get("direction"),
                "gross_iv_diff_vol_pts": row.get("gross_iv_diff_vol_pts"),
                "recommended_workflow": row.get("recommended_workflow", "review only"),
                "evidence_status": EVIDENCE_STATUS,
            }
        )
    return candidates


def _parse_money(text: str, label: str) -> float | None:
    match = re.search(rf"- {re.escape(label)}:\s*\$?([\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def _parse_int_metric(text: str, label: str) -> int | None:
    match = re.search(rf"- {re.escape(label)}:\s*(\d+)", text)
    return int(match.group(1)) if match else None


def _parse_percent_metric(text: str, label: str) -> float | None:
    match = re.search(rf"- {re.escape(label)}:\s*([\d.]+)%", text)
    return float(match.group(1)) / 100.0 if match else None


def _load_backtest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"scenarios": [], "controls": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"scenarios": [], "controls": []}
    scenarios = []
    for row in payload.get("scenarios") or []:
        if not isinstance(row, dict):
            continue
        scenarios.append(
            {
                "tenor": row.get("tenor", "unknown"),
                "snapshot_count": int(row.get("snapshot_count") or 0),
                "trade_count": int(row.get("trade_count") or 0),
                "gross_pnl": row.get("gross_pnl"),
                "max_drawdown": row.get("max_drawdown"),
                "win_rate": row.get("win_rate"),
                "sample_gate": row.get("sample_gate", "insufficient-history"),
                "cost_per_trade": row.get("cost_per_trade"),
                "slippage_vol_pts": row.get("slippage_vol_pts"),
                "evidence_status": row.get("evidence_status", "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"),
            }
        )
    controls = [str(item) for item in payload.get("controls") or []]
    return {"scenarios": scenarios, "controls": controls, "robustness_metrics": payload.get("robustness_metrics") or {}}


def _latest_backtest_research(backtests_dir: str | Path) -> dict[str, Any]:
    base = Path(backtests_dir)
    files = sorted(base.glob("vol-spread-backtest-*.md"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not files:
        return {
            "available": False,
            "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
            "sample_gate": "insufficient-history",
            "control_note": "No point-in-time backtest artifact is available yet; research economics would be synthetic fills only.",
        }
    path = files[-1]
    text = path.read_text(encoding="utf-8")
    summary_path = path.with_suffix(".json")
    summary = _load_backtest_summary(summary_path)
    tenor_match = re.search(r"- Tenor:\s*`?([A-Za-z0-9_\-]+)`?", text)
    snapshot_count = _parse_int_metric(text, "Snapshot count") or 0
    trade_count = _parse_int_metric(text, "Trade count") or 0
    sample_gate = "pass" if snapshot_count >= 30 and trade_count >= 20 else "insufficient-history"
    return {
        "available": True,
        "path": str(path),
        "summary_json_path": str(summary_path) if summary_path.exists() else None,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "summary_json_sha256": _sha256_file(summary_path),
        "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
        "tenor": tenor_match.group(1) if tenor_match else "unknown",
        "snapshot_count": snapshot_count,
        "trade_count": trade_count,
        "gross_pnl": _parse_money(text, "Gross PnL"),
        "max_drawdown": _parse_money(text, "Max drawdown"),
        "win_rate": _parse_percent_metric(text, "Win rate"),
        "sample_gate": sample_gate,
        "control_note": "Point-in-time research only using synthetic fills; not executable economics and not an investment conclusion.",
        "controls": summary["controls"],
        "scenarios": summary["scenarios"],
        "robustness_metrics": summary.get("robustness_metrics") or {},
    }


HISTORICAL_REQUIRED_SOURCE_LABELS = ["IBIT", "Deribit", "CME", "BTC reference", "IBIT holdings", "Rates/fees"]


def _source_coverage_labels(row: dict[str, Any]) -> list[str]:
    haystack = " ".join(
        str(row.get(key) or "")
        for key in ("source_id", "source_name", "provider", "venue", "instrument_scope", "notes")
    ).lower()
    labels: list[str] = []
    if re.search(r"\bibit\b", haystack) or re.search(r"\bopra\b", haystack):
        labels.append("IBIT")
    if re.search(r"\bderibit\b", haystack):
        labels.append("Deribit")
    if "cme" in haystack or "databento" in haystack:
        labels.append("CME")
    if "btc reference" in haystack or "btc spot" in haystack or "bitcoin reference" in haystack:
        labels.append("BTC reference")
    if "ibit holdings" in haystack or "ishares" in haystack or "btc per share" in haystack:
        labels.append("IBIT holdings")
    if "rate" in haystack or "fee" in haystack:
        labels.append("Rates/fees")
    return [label for label in HISTORICAL_REQUIRED_SOURCE_LABELS if label in labels]


def _source_coverage_matrix(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], str, bool]:
    matrix: list[dict[str, Any]] = []
    not_licensed_covered: list[str] = []
    licensed_covered_count = 0
    fixture_only_count = 0
    missing_count = 0
    for label in HISTORICAL_REQUIRED_SOURCE_LABELS:
        matching = [source for source in sources if label in (source.get("coverage_labels") or [])]
        fixture_sources = [source for source in matching if _source_is_fixture(source)]
        licensed_sources = [source for source in matching if not _source_is_fixture(source)]
        if licensed_sources:
            status = "covered"
            licensed_covered_count += 1
        elif fixture_sources:
            status = "fixture-only"
            fixture_only_count += 1
            not_licensed_covered.append(label)
        else:
            status = "missing"
            missing_count += 1
            not_licensed_covered.append(label)
        matrix.append({"label": label, "status": status, "source_count": len(matching)})
    summary = (
        f"{licensed_covered_count}/{len(HISTORICAL_REQUIRED_SOURCE_LABELS)} licensed source groups covered"
        f" · {fixture_only_count} fixture-only · {missing_count} missing"
    )
    return matrix, not_licensed_covered, summary, not not_licensed_covered


def _source_is_fixture(source: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(source.get(key) or "")
        for key in ("provider", "license_label", "source_id", "source_name", "notes")
    ).lower()
    return "fixture" in haystack


def _source_row_count(source: dict[str, Any]) -> int | None:
    for key in ("row_count", "rows", "record_count", "snapshot_count"):
        value = source.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _tracker_next_action(label: str) -> str:
    actions = {
        "IBIT": "Expand IBIT historical coverage window and retain manifest hashes for replay.",
        "Deribit": "Load Deribit historical options replay source with raw-file hash and coverage dates.",
        "CME": "Configure licensed CME/Databento historical source; keep screen-only until quote/trade evidence exists.",
        "BTC reference": "Add point-in-time BTC reference marks used for every decision frame.",
        "IBIT holdings": "Add point-in-time IBIT holdings/BTC-per-share files for every decision date.",
        "Rates/fees": "Add rates, borrow, fees, and transaction-cost assumptions with source hashes.",
    }
    return actions.get(label, "Capture source coverage, raw hash, and replay eligibility before investment use.")


def _source_coverage_tracker(sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    tracker: list[dict[str, Any]] = []
    covered_count = 0
    fixture_count = 0
    missing_count = 0
    for label in HISTORICAL_REQUIRED_SOURCE_LABELS:
        matching = [source for source in sources if label in (source.get("coverage_labels") or [])]
        if not matching:
            missing_count += 1
            tracker.append(
                {
                    "label": label,
                    "status": "missing",
                    "source_count": 0,
                    "provider": "missing",
                    "coverage_start": None,
                    "coverage_end": None,
                    "row_count": None,
                    "sha256": None,
                    "license_label": "missing",
                    "redistribution": "internal_only",
                    "execution_confidence": "screen_only_not_executable",
                    "blocker": f"No {label} source captured in latest historical manifest.",
                    "next_action": _tracker_next_action(label),
                }
            )
            continue
        primary = matching[0]
        is_fixture = any(_source_is_fixture(source) for source in matching)
        status = "fixture-only" if is_fixture else "covered"
        if is_fixture:
            fixture_count += 1
        else:
            covered_count += 1
        blocker = (
            "Fixture source; replace with licensed historical export before investment use."
            if is_fixture
            else "No blocker captured; verify license scope and expand date coverage before investment use."
        )
        tracker.append(
            {
                "label": label,
                "status": status,
                "source_count": len(matching),
                "provider": primary.get("provider") or "unknown",
                "coverage_start": primary.get("coverage_start"),
                "coverage_end": primary.get("coverage_end"),
                "row_count": _source_row_count(primary),
                "sha256": primary.get("sha256"),
                "license_label": primary.get("license_label") or "unknown",
                "redistribution": primary.get("redistribution") or "internal_only",
                "execution_confidence": primary.get("execution_confidence") or "screen_only_not_executable",
                "blocker": blocker,
                "next_action": _tracker_next_action(label),
            }
        )
    summary = f"{covered_count} covered · {missing_count} missing · {fixture_count} fixture-only"
    return tracker, summary


def _portable_artifact_path(value: Any) -> str | None:
    text = _safe_path(value)
    if not text:
        return text
    marker = "artifacts/institutional/"
    if marker in text:
        return text[text.index(marker) :]
    return text


def _latest_historical_source_diagnostics(historical_dir: str | Path) -> dict[str, Any]:
    base = Path(historical_dir)
    manifests = sorted((base / "manifests").glob("historical-source-manifest*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not manifests:
        empty_tracker, empty_tracker_summary = _source_coverage_tracker([])
        return {
            "available": False,
            "source_count": 0,
            "required_source_labels": HISTORICAL_REQUIRED_SOURCE_LABELS,
            "coverage_summary": f"0/{len(HISTORICAL_REQUIRED_SOURCE_LABELS)} licensed source groups covered · 0 fixture-only · {len(HISTORICAL_REQUIRED_SOURCE_LABELS)} missing",
            "coverage_ready": False,
            "missing_required_source_labels": HISTORICAL_REQUIRED_SOURCE_LABELS,
            "coverage_matrix": [
                {"label": label, "status": "missing", "source_count": 0}
                for label in HISTORICAL_REQUIRED_SOURCE_LABELS
            ],
            "source_coverage_tracker": empty_tracker,
            "source_tracker_summary": empty_tracker_summary,
            "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
            "control_note": "No historical source manifest is available yet. Backtest source provenance must be established before research outputs are treated as reviewable.",
            "sources": [],
        }
    path = manifests[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    sources = []
    for row in payload.get("sources") or []:
        if not isinstance(row, dict):
            continue
        sources.append(
            {
                "source_id": row.get("source_id"),
                "source_name": row.get("source_name", "Historical source"),
                "provider": row.get("provider", "unknown"),
                "venue": row.get("venue", "unknown"),
                "instrument_scope": row.get("instrument_scope", "unknown"),
                "license_label": row.get("license_label", "unknown"),
                "redistribution": row.get("redistribution", "internal_only"),
                "execution_confidence": row.get("execution_confidence", "screen_only_not_executable"),
                "coverage_start": row.get("coverage_start"),
                "coverage_end": row.get("coverage_end"),
                "raw_path": _portable_artifact_path(row.get("raw_path")),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
                "status": row.get("status", "available"),
                "row_count": _source_row_count(row),
                "coverage_labels": _source_coverage_labels(row),
                "evidence_status": payload.get("evidence_status", "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"),
            }
        )
    matrix, missing, summary, ready = _source_coverage_matrix(sources)
    tracker, tracker_summary = _source_coverage_tracker(sources)
    return {
        "available": True,
        "manifest_path": _portable_artifact_path(path),
        "manifest_sha256": _sha256_file(path),
        "source_count": int(payload.get("source_count") or len(sources)),
        "required_source_labels": HISTORICAL_REQUIRED_SOURCE_LABELS,
        "coverage_summary": summary,
        "coverage_ready": ready,
        "missing_required_source_labels": missing,
        "coverage_matrix": matrix,
        "source_coverage_tracker": tracker,
        "source_tracker_summary": tracker_summary,
        "evidence_status": payload.get("evidence_status", "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"),
        "control_note": "Historical backtests use hashed point-in-time sources; synthetic fills only; not executable economics.",
        "sources": sources,
    }


def _institutional_readiness_gate(
    *,
    historical_diagnostics: dict[str, Any],
    backtest_research: dict[str, Any],
    quote_evidence_ledger: dict[str, Any],
    legal_wrapper_approved: bool = False,
) -> dict[str, Any]:
    tracker = historical_diagnostics.get("source_coverage_tracker") or []
    source_ready = bool(historical_diagnostics.get("coverage_ready")) and bool(tracker) and all(
        row.get("status") == "covered" for row in tracker
    )
    robustness = backtest_research.get("robustness_metrics") or {}
    backtest_ready = bool(robustness.get("sample_gate_ready"))
    quote_summary = quote_evidence_ledger.get("summary") or {}
    quote_ready = int(quote_summary.get("quote_verified_candidates") or 0) > 0
    gates = [
        {
            "label": "Licensed historical source coverage",
            "passed": source_ready,
            "blocker": "licensed historical source coverage incomplete",
            "required": "All required source groups covered by non-fixture, replay-ready sources",
        },
        {
            "label": "Backtest sample gate",
            "passed": backtest_ready,
            "blocker": "backtest sample gate insufficient",
            "required": "At least one scenario passes minimum snapshot and synthetic-trade thresholds",
        },
        {
            "label": "Two-counterparty quote evidence",
            "passed": quote_ready,
            "blocker": "two-counterparty quote evidence missing",
            "required": "At least one candidate has quote-verified evidence from two distinct counterparties",
        },
        {
            "label": "Counsel-approved legal wrapper",
            "passed": legal_wrapper_approved,
            "blocker": "counsel-approved wrapper missing",
            "required": "Approved legal/business wrapper before external investor/client use",
        },
    ]
    passed = sum(1 for gate in gates if gate["passed"])
    blockers = [gate["blocker"] for gate in gates if not gate["passed"]]
    return {
        "status": "ready" if passed == len(gates) else "not-ready",
        "summary": f"{passed}/{len(gates)} readiness gates passed",
        "label": "READY" if passed == len(gates) else "NOT READY FOR INVESTMENT/CLIENT USE",
        "gates": gates,
        "blockers": blockers,
        "next_actions": [
            "Replace IBIT/Deribit fixtures and missing groups with licensed/replay-ready source manifests.",
            "Accumulate enough point-in-time replay observations for sample-gate pass thresholds.",
            "Capture two distinct external indicative quote records for at least one candidate.",
            "Obtain counsel-approved business/legal wrapper before external investor/client use.",
        ],
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "control_note": "Readiness gates are diligence controls, not approval to trade, advise, market, or raise capital.",
    }


def _parse_decision_ts(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _source_intake_validation(historical_dir: Path) -> dict[str, Any]:
    manifest_path = historical_dir / "source_intake_manifest.json"
    if not manifest_path.exists():
        return {
            "ready": False,
            "manifest_path": "missing",
            "covered_source_groups": 0,
            "required_source_groups": 6,
            "blockers": ["source intake manifest missing"],
            "source_results": [],
            "evidence_status": "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ready": False,
            "manifest_path": str(manifest_path),
            "covered_source_groups": 0,
            "required_source_groups": 6,
            "blockers": [f"source intake manifest invalid JSON: {exc.msg}"],
            "source_results": [],
            "evidence_status": "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE",
        }
    result = validate_source_intake_manifest(manifest, decision_ts=_parse_decision_ts(manifest.get("decision_ts")))
    return {**result, "manifest_path": str(manifest_path)}


def _count_csv_data_rows(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _availability_row(
    *,
    label: str,
    path: Path | None,
    provider: str,
    license_label: str,
    row_count: int | None = None,
    available: bool | None = None,
    blocker: str | None = None,
) -> dict[str, Any]:
    is_available = (path is not None and path.exists()) if available is None else available
    return {
        "label": label,
        "status": "available" if is_available else "missing",
        "provider": provider if is_available else "missing",
        "license_label": license_label if is_available else "missing",
        "path": _portable_artifact_path(path) if path else None,
        "sha256": _sha256_file(path) if path else None,
        "row_count": row_count,
        "execution_confidence": "screen_only_not_executable",
        "evidence_status": "SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE",
        "blocker": blocker if not is_available else "Available for internal diligence only; does not clear licensed historical readiness.",
    }


def _current_source_availability(data_dir: Path, historical_dir: Path, latest: dict[str, Any]) -> dict[str, Any]:
    run_id = str(latest.get("run_id") or "")
    raw_dir = data_dir / "raw" / run_id if run_id else data_dir / "raw" / "missing"
    rates_path = _first_existing(sorted((historical_dir / "source_audit").glob("fred_rates*.csv"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    ibit_options_path = _first_existing([raw_dir / "nasdaq_ibit_option_chain.json"])
    deribit_path = _first_existing([raw_dir / "deribit_book_summary_btc_options.json"])
    cme_path = _first_existing([raw_dir / "databento_cme_btc_options_bbo_1m.csv", raw_dir / "databento_cme_btc_options_definition.csv"])
    holdings_path = _first_existing([raw_dir / "ishares_ibit_holdings.csv", raw_dir / "ishares_ibit_holdings_fallback_cached.csv"])
    rows = [
        _availability_row(
            label="IBIT options",
            path=ibit_options_path,
            provider="Nasdaq public option-chain JSON",
            license_label="public_screen_source",
            row_count=int(latest.get("ibit_atm_rows") or 0) or None,
            blocker="No current IBIT option-chain capture found for latest run.",
        ),
        _availability_row(
            label="Deribit options",
            path=deribit_path,
            provider="Deribit public API",
            license_label="public_api_screen_data",
            row_count=int(latest.get("deribit_atm_rows") or 0) or None,
            blocker="No current Deribit option capture found for latest run.",
        ),
        _availability_row(
            label="CME Bitcoin options",
            path=cme_path,
            provider="Databento CME",
            license_label="licensed_vendor_api_databento",
            row_count=int(latest.get("cme_rows") or 0) or _count_csv_data_rows(cme_path),
            available=bool(cme_path and int(latest.get("cme_rows") or 0) > 0),
            blocker="No current Databento CME rows found for latest run.",
        ),
        _availability_row(
            label="BTC reference",
            path=None,
            provider="Monitor run manifest",
            license_label="run_derived_reference",
            row_count=1 if latest.get("btc_spot") is not None else None,
            available=latest.get("btc_spot") is not None,
            blocker="No BTC reference mark captured in latest run manifest.",
        ),
        _availability_row(
            label="IBIT holdings",
            path=holdings_path,
            provider="iShares holdings capture/cache",
            license_label="public_fund_holdings_or_valid_cache",
            row_count=_count_csv_data_rows(holdings_path),
            blocker="No current valid IBIT holdings capture/cache found for latest run.",
        ),
        _availability_row(
            label="Rates/fee curves",
            path=rates_path,
            provider="FRED rates CSV",
            license_label="public_reference_rates",
            row_count=_count_csv_data_rows(rates_path),
            blocker="No current rates/fee curve file found in source audit artifacts.",
        ),
    ]
    available_count = sum(1 for row in rows if row["status"] == "available")
    total = len(rows)
    return {
        "summary": f"{available_count}/{total} current source groups available",
        "available_count": available_count,
        "required_count": total,
        "ready_for_internal_diligence": available_count == total,
        "ready_for_investment_or_client_use": False,
        "evidence_status": "SCREEN-ONLY · CURRENT SOURCE AVAILABILITY · NOT EXECUTABLE",
        "control_note": "Current captured sources can support internal diligence, but they do not clear licensed historical readiness, quote verification, legal approval, or executable-economics gates.",
        "groups": rows,
    }


def _licensed_source_intake_contract(validation_result: dict[str, Any] | None = None) -> dict[str, Any]:
    common_formats = ["csv", "jsonl", "parquet"]
    return {
        "status": "not-ready",
        "evidence_status": "SCREEN-ONLY · INTAKE CONTRACT · NOT EXECUTABLE",
        "control_note": "Defines the licensed/replay-ready source package required before historical coverage can pass readiness gates. It does not create live data access or executable economics.",
        "validation_result": validation_result or {
            "ready": False,
            "manifest_path": "missing",
            "covered_source_groups": 0,
            "required_source_groups": 6,
            "blockers": ["source intake manifest missing"],
            "evidence_status": "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE",
        },
        "required_sources": [
            {
                "source_group": "IBIT options history",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest", "source_ref"],
                "provider_examples": ["OPRA/vendor export", "broker historical option chain"],
            },
            {
                "source_group": "Deribit options history",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "instrument_name", "underlying_price", "bid_iv", "ask_iv", "mark_iv", "open_interest", "source_ref"],
                "provider_examples": ["Deribit historical export", "licensed vendor replay"],
            },
            {
                "source_group": "CME Bitcoin options history",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "symbol", "expiration", "strike", "option_type", "bid", "ask", "settlement", "source_ref"],
                "provider_examples": ["Databento licensed history", "CME/broker historical export"],
            },
            {
                "source_group": "BTC reference history",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "btc_usd", "venue_or_index", "source_ref"],
                "provider_examples": ["CF Benchmarks", "vendor BTC index", "broker reference"],
            },
            {
                "source_group": "IBIT holdings history",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "btc_per_share", "shares_outstanding", "fund_assets", "source_ref"],
                "provider_examples": ["iShares daily holdings archive", "fund administrator export"],
            },
            {
                "source_group": "Rates and fee curves",
                "accepted_formats": common_formats,
                "required_fields": ["available_ts", "tenor", "rate", "borrow_or_fee", "source_ref"],
                "provider_examples": ["SOFR/rate vendor", "broker borrow/fee export"],
            },
        ],
        "validation_gates": [
            "No future available_ts relative to decision_ts",
            "Raw file SHA-256 captured before normalization",
            "Provider/license label present for every source group",
            "Fixture/manual_fixture sources cannot satisfy readiness",
            "Rows with crossed/wide/missing markets remain diagnostic, not executable",
        ],
    }


def build_site_data(data_dir: str | Path, *, backtests_dir: str | Path | None = None, historical_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(data_dir)
    resolved_backtests_dir = Path(backtests_dir) if backtests_dir is not None else base.parent / "backtests"
    resolved_historical_dir = Path(historical_dir) if historical_dir is not None else base.parent / "historical"
    runs = _read_jsonl(base / "run_manifest.jsonl")
    latest = runs[-1] if runs else {}
    cme_rows = int(latest.get("cme_rows") or 0)
    coverage_completeness = (
        "Current screen/vendor captures available: Deribit + IBIT + CME Databento"
        if cme_rows > 0
        else "Current screen/vendor captures partial: CME unavailable"
    )
    overall_evidence_readiness = (
        "YELLOW until real quote evidence exists"
        if cme_rows > 0
        else "YELLOW until CME/licensed feed and real quote evidence exist"
    )
    latest_run = {
        "run_id": latest.get("run_id", "missing"),
        "as_of_cst": latest.get("as_of_cst", "missing"),
        "btc_spot": latest.get("btc_spot"),
        "btc_per_share": latest.get("btc_per_share"),
        "quality_score": latest.get("quality_score"),
        "quality_grade": latest.get("quality_grade"),
        "configured_source_quality_label": "Configured-source quality",
        "coverage_completeness_label": "Current screen-source availability",
        "coverage_completeness": coverage_completeness,
        "overall_evidence_readiness": overall_evidence_readiness,
        "cme_rows": cme_rows,
        "freshness_grade": latest.get("freshness_grade"),
        "dislocations": latest.get("dislocations", 0),
        "quote_review_candidates": latest.get("quote_review_candidates", 0),
        "evidence_bundle_sha256": latest.get("evidence_bundle_sha256") or _sha256_file(latest.get("evidence_bundle_path")),
        "report_path": _safe_path(latest.get("report_path")),
        "dashboard_path": _safe_path(latest.get("dashboard_path")),
        "evidence_bundle_path": _safe_path(latest.get("evidence_bundle_path")),
        "evidence_manifest_path": _safe_path(latest.get("evidence_manifest_path")),
        "candidate_ledger_path": _safe_path(latest.get("candidate_ledger_path")),
        "quote_evidence_ledger_path": _safe_path(latest.get("quote_evidence_ledger_path")),
    }
    spot_value = latest.get("btc_spot")
    spot = float(spot_value) if spot_value is not None else None
    candidates = _load_candidates(latest.get("candidate_ledger_path"))
    quote_path = latest.get("quote_evidence_ledger_path")
    quote_evidence_ledger = load_quote_evidence_ledger(quote_path) if quote_path else load_quote_evidence_ledger(base / "__no_quote_evidence_for_latest_run__.jsonl")
    backtest_research = _latest_backtest_research(resolved_backtests_dir)
    historical_diagnostics = _latest_historical_source_diagnostics(resolved_historical_dir)
    current_source_availability = _current_source_availability(base, resolved_historical_dir, latest)
    source_intake_validation = _source_intake_validation(resolved_historical_dir)
    legal_wrapper = load_legal_wrapper_package(base.parent / "legal" / "legal-wrapper-package-v1.json")
    readiness_gate = _institutional_readiness_gate(
        historical_diagnostics=historical_diagnostics,
        backtest_research=backtest_research,
        quote_evidence_ledger=quote_evidence_ledger,
        legal_wrapper_approved=bool(legal_wrapper.get("approved_by_counsel")),
    )
    if spot is None:
        case_studies = {
            "available": False,
            "evidence_status": EVIDENCE_STATUS,
            "control_note": "BTC spot unavailable — case studies suppressed to avoid fallback economics.",
        }
    else:
        case_studies = {
            "available": True,
            "treasury": build_treasury_hedge_case_study(
                btc_held=1000,
                spot=spot,
                hedge_ratio=0.35,
                floor_pct=0.75,
                cap_pct=1.25,
                tenor_days=90,
            ),
            "miner": build_miner_runway_case_study(
                monthly_btc_production=120,
                spot=spot,
                cash_cost_per_btc=42000,
                cash_balance_usd=15_000_000,
                monthly_fixed_cost_usd=4_000_000,
                hedge_ratio=0.5,
                floor_pct=0.7,
                tenor_months=6,
            ),
        }
    return {
        "positioning": POSITIONING,
        "full_positioning": FULL_POSITIONING,
        "evidence_status": EVIDENCE_STATUS,
        "legal_legend": LEGAL_LEGEND,
        "legal_gate": LEGAL_GATE,
        "legal_wrapper_package": legal_wrapper,
        "latest_run": latest_run,
        "top_candidates": candidates,
        "quote_verification_board": build_quote_verification_demo_board(candidates),
        "quote_evidence_ledger": quote_evidence_ledger,
        "backtest_research": backtest_research,
        "historical_source_diagnostics": historical_diagnostics,
        "current_source_availability": current_source_availability,
        "institutional_readiness_gate": readiness_gate,
        "source_intake_validation": source_intake_validation,
        "licensed_source_intake_contract": _licensed_source_intake_contract(source_intake_validation),
        "run_count": len(runs),
        "business_model_sequence": [
            "Research/evidence engine",
            "Treasury and miner hedge structuring",
            "Partner-led RFQ and execution support",
            "Principal risk sleeve only after chosen legal wrapper",
        ],
        "case_studies": case_studies,
    }


def write_site_data(data_dir: str | Path, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build_site_data(data_dir), indent=2), encoding="utf-8")
    return target
