from __future__ import annotations

import csv
import json
import sys
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from institutional_btc_vol.candidate_triage import rank_dislocation_candidates, write_candidate_ledger
from institutional_btc_vol.dashboard import write_dashboard
from institutional_btc_vol.databento_cme import fetch_cme_btc_option_snapshot, load_databento_api_key
from institutional_btc_vol.evidence_bundle import build_evidence_bundle
from institutional_btc_vol.evidence_manifest import build_evidence_manifest, write_evidence_manifest
from institutional_btc_vol.freshness import evaluate_source_freshness
from institutional_btc_vol.history import append_run_manifest, load_recent_runs
from institutional_btc_vol.market_diagnostics import diagnose_deribit_options, diagnose_nasdaq_ibit_chain
from institutional_btc_vol.monitor import (
    compute_btc_per_share,
    estimate_black_scholes_iv,
    generate_monitor_report,
    normalize_deribit_instrument,
    normalize_ibit_option,
    select_atm_by_moneyness,
)
from institutional_btc_vol.quality import compute_quality_score
from institutional_btc_vol.quote_evidence import load_quote_evidence
from institutional_btc_vol.trends import build_trend_summary, extract_iv_benchmarks

DERIBIT_BOOK_SUMMARY = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option"
ISHARES_HOLDINGS_CSV = "https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf/1467271812596.ajax?fileType=csv&fileName=IBIT_holdings&dataType=fund"
NASDAQ_IBIT_OPTIONS = "https://api.nasdaq.com/api/quote/IBIT/option-chain?assetclass=etf&limit=9999"


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_deribit_options() -> list[dict]:
    payload = json.loads(fetch_text(DERIBIT_BOOK_SUMMARY))
    if "result" not in payload:
        raise RuntimeError(f"Unexpected Deribit response keys: {list(payload)}")
    return payload["result"]


def fetch_ibit_option_chain() -> dict:
    payload = json.loads(fetch_text(NASDAQ_IBIT_OPTIONS))
    if not payload.get("data") or "table" not in payload["data"]:
        raise RuntimeError(f"Unexpected Nasdaq IBIT response: {payload.get('status') or list(payload)}")
    return payload["data"]


def parse_nasdaq_last_trade(last_trade: str) -> float | None:
    import re

    match = re.search(r"\$([0-9,.]+)", last_trade or "")
    return float(match.group(1).replace(",", "")) if match else None


def parse_nasdaq_expiry_group(expiry_group: str, as_of: datetime):
    value = str(expiry_group or "").strip()
    formats_with_year = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")
    for fmt in formats_with_year:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    formats_without_year = ("%B %d", "%b %d")
    for fmt in formats_without_year:
        try:
            parsed = datetime.strptime(f"{value} {as_of.year}", f"{fmt} %Y")
            candidate = parsed.date()
            if (candidate - as_of.date()).days < -180:
                candidate = datetime.strptime(f"{value} {as_of.year + 1}", f"{fmt} %Y").date()
            return candidate
        except ValueError:
            pass
    raise ValueError(f"Unsupported Nasdaq expiry group: {expiry_group!r}")


def parse_nasdaq_ibit_rows(chain: dict, as_of: datetime, btc_per_share: float) -> list[dict]:
    rows = chain.get("table", {}).get("rows", [])
    underlying = parse_nasdaq_last_trade(chain.get("lastTrade", ""))
    parsed: list[dict] = []
    current_expiry = None
    for row in rows:
        if row.get("expirygroup"):
            current_expiry = parse_nasdaq_expiry_group(row["expirygroup"], as_of)
            continue
        if current_expiry is None or row.get("strike") in (None, "") or underlying is None:
            continue
        strike = row.get("strike")
        for prefix, option_type in (("c", "call"), ("p", "put")):
            bid = row.get(f"{prefix}_Bid")
            ask = row.get(f"{prefix}_Ask")
            if bid in (None, "--", "") or ask in (None, "--", ""):
                continue
            bid_f = float(str(bid).replace(",", ""))
            ask_f = float(str(ask).replace(",", ""))
            if ask_f <= 0 or bid_f < 0 or ask_f < bid_f:
                continue
            mid = (bid_f + ask_f) / 2.0
            dte = max((current_expiry - as_of.date()).days, 0)
            iv = estimate_black_scholes_iv(
                option_type=option_type,
                spot=underlying,
                strike=float(str(strike).replace(",", "")),
                years_to_expiry=max(dte, 1) / 365,
                price=mid,
                rate=0.045,
            )
            if iv is None:
                continue
            parsed.append(
                {
                    "expiry": current_expiry.isoformat(),
                    "strike": strike,
                    "type": option_type,
                    "bid": bid,
                    "ask": ask,
                    "iv": iv,
                    "volume": row.get(f"{prefix}_Volume"),
                    "open_interest": row.get(f"{prefix}_Openinterest"),
                    "underlying_price": underlying,
                    "timestamp": as_of.isoformat(),
                }
            )
    normalized = []
    for option in parsed:
        normalized.append(asdict(normalize_ibit_option(option, btc_per_share=btc_per_share, as_of=as_of.isoformat())))
    for row in normalized:
        row["expiry"] = row["expiry"].isoformat()
        row["dte"] = max((datetime.fromisoformat(row["expiry"]).date() - as_of.date()).days, 0)
    return normalized


def try_fetch_deribit_options(raw_dir: Path) -> tuple[list[dict], list[dict], list[str], datetime | None]:
    warnings: list[str] = []
    try:
        options = fetch_deribit_options()
        captured_at = datetime.now(ZoneInfo("America/Chicago"))
        diagnostics = [diagnose_deribit_options(options)]
        (raw_dir / "deribit_book_summary_btc_options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
        return options, diagnostics, warnings, captured_at
    except Exception as exc:  # noqa: BLE001 - monitor should emit degraded artifacts, not hard-fail
        warnings.append(f"Deribit fetch or parse failed: {exc}")
        return [], [], warnings, None


def _looks_like_html(text: str) -> bool:
    prefix = text.lstrip()[:500].lower()
    return prefix.startswith("<!doctype html") or prefix.startswith("<html") or "<head>" in prefix


def _find_latest_valid_ishares_cache(base_dir: Path) -> tuple[float, Path] | None:
    raw_root = base_dir / "raw"
    if not raw_root.exists():
        return None
    candidates = sorted(raw_root.glob("*/ishares_ibit_holdings.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
            if _looks_like_html(text):
                continue
            result = compute_btc_per_share(text)
            return result.btc_per_share, candidate
        except Exception:  # noqa: BLE001 - skip bad cache candidates
            continue
    return None


def _derive_market_implied_btc_per_share(ibit_chain: dict | None, btc_spot: float | None) -> tuple[float, list[str]] | None:
    if not ibit_chain or btc_spot is None or btc_spot <= 0:
        return None
    ibit_last = parse_nasdaq_last_trade(str(ibit_chain.get("lastTrade") or ""))
    if ibit_last is None or ibit_last <= 0:
        return None
    btc_per_share = ibit_last / btc_spot
    return btc_per_share, [
        f"IBIT BTC/share market-implied fallback from Nasdaq IBIT last trade / Deribit BTC spot: {btc_per_share:.12f}",
        "IBIT BTC/share market-implied value is not official holdings; use only as lower-confidence continuity evidence",
    ]


def _select_btc_per_share_source(
    *,
    live: tuple[float, list[str]] | None,
    cached: tuple[float, Path | str] | None,
    market_implied: tuple[float, list[str]] | None,
) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    if live is not None:
        value, live_warnings = live
        warnings.extend(live_warnings)
        if market_implied is not None:
            market_value, _ = market_implied
            diff_pct = abs(market_value - value) / value * 100 if value else 0.0
            warnings.append(f"IBIT BTC/share independent market-implied cross-check diff: {diff_pct:.2f}%")
        return value, warnings
    if cached is not None:
        value, cache_path = cached
        warnings.append(f"iShares BTC/share fallback used latest cached official CSV: {cache_path}")
        warnings.append("IBIT holdings source is stale-cache; use for continuity only until current iShares CSV recovers")
        if market_implied is not None:
            market_value, _ = market_implied
            diff_pct = abs(market_value - value) / value * 100 if value else 0.0
            warnings.append(f"IBIT BTC/share independent market-implied cross-check diff: {diff_pct:.2f}%")
        return value, warnings
    if market_implied is not None:
        value, market_warnings = market_implied
        warnings.extend(market_warnings)
        warnings.append("IBIT BTC/share using lower-confidence market-implied fallback because official current and cached holdings were unavailable")
        return value, warnings
    return None, warnings


def try_fetch_btc_per_share(raw_dir: Path, ibit_chain: dict | None = None, btc_spot: float | None = None) -> tuple[float | None, list[str]]:
    warnings: list[str] = []
    raw_path = raw_dir / "ishares_ibit_holdings.csv"
    try:
        text = fetch_text(ISHARES_HOLDINGS_CSV)
        raw_path.write_text(text, encoding="utf-8")
        if _looks_like_html(text):
            raise ValueError("iShares holdings endpoint returned HTML instead of holdings CSV")
        result = compute_btc_per_share(text)
        return _select_btc_per_share_source(
            live=(result.btc_per_share, warnings),
            cached=None,
            market_implied=_derive_market_implied_btc_per_share(ibit_chain, btc_spot),
        )
    except Exception as exc:  # noqa: BLE001 - monitor should degrade gracefully
        warnings.append(f"iShares BTC/share current fetch or parse failed: {exc}")
        cached = _find_latest_valid_ishares_cache(raw_dir.parents[1])
        if cached is not None:
            btc_per_share, cache_path = cached
            cache_text = cache_path.read_text(encoding="utf-8", errors="replace")
            fallback_path = raw_dir / "ishares_ibit_holdings_fallback_cached.csv"
            fallback_path.write_text(cache_text, encoding="utf-8")
            selected_value, selected_warnings = _select_btc_per_share_source(
                live=None,
                cached=(btc_per_share, cache_path),
                market_implied=_derive_market_implied_btc_per_share(ibit_chain, btc_spot),
            )
            warnings.extend(selected_warnings)
            return selected_value, warnings
        selected_value, selected_warnings = _select_btc_per_share_source(
            live=None,
            cached=None,
            market_implied=_derive_market_implied_btc_per_share(ibit_chain, btc_spot),
        )
        warnings.extend(selected_warnings)
        return selected_value, warnings


def build_deribit_atm_rows(options: list[dict], as_of_iso: str) -> tuple[list[dict], float | None, list[str]]:
    warnings: list[str] = []
    rows_by_expiry: dict[str, list[dict]] = {}
    btc_spot_values: list[float] = []
    normalized_count = 0
    for option in options:
        try:
            row = normalize_deribit_instrument(option, as_of=as_of_iso)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Deribit normalize failed for {option.get('instrument_name')}: {exc}")
            continue
        normalized_count += 1
        if row.spot_native > 0:
            btc_spot_values.append(row.spot_native)
        d = asdict(row)
        d["expiry"] = row.expiry.isoformat()
        d["dte"] = max((row.expiry - datetime.fromisoformat(as_of_iso).date()).days, 0)
        rows_by_expiry.setdefault(d["expiry"], []).append(d)

    atm_rows = []
    for expiry, rows in sorted(rows_by_expiry.items()):
        try:
            selected = select_atm_by_moneyness(rows)
        except ValueError:
            continue
        atm_rows.append(selected)
    btc_spot = sum(btc_spot_values) / len(btc_spot_values) if btc_spot_values else None
    warnings.append(f"Deribit normalized rows: {normalized_count}")
    return atm_rows, btc_spot, warnings


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    base_dir = Path(argv[0]) if argv else Path("artifacts/institutional/data")
    now = datetime.now(ZoneInfo("America/Chicago"))
    run_id = f"btcvol-{now:%Y%m%d-%H%M%S}"
    as_of_iso = now.isoformat()
    as_of_cst = now.strftime("%Y-%m-%d %H:%M:%S %Z")

    raw_dir = base_dir / "raw" / run_id
    normalized_dir = base_dir / "normalized" / run_id
    reports_dir = base_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    quality_warnings: list[str] = []
    source_captures: dict[str, datetime | None] = {
        "Deribit": None,
        "Nasdaq IBIT options": None,
        "iShares IBIT holdings": None,
        "Databento CME BTC options": None,
    }

    options, market_diagnostics, deribit_fetch_warnings, deribit_captured_at = try_fetch_deribit_options(raw_dir)
    source_captures["Deribit"] = deribit_captured_at
    quality_warnings.extend(deribit_fetch_warnings)
    deribit_atm_rows, btc_spot, deribit_warnings = build_deribit_atm_rows(options, as_of_iso)
    quality_warnings.extend(deribit_warnings)

    ibit_chain: dict | None = None
    try:
        ibit_chain = fetch_ibit_option_chain()
        source_captures["Nasdaq IBIT options"] = datetime.now(ZoneInfo("America/Chicago"))
        market_diagnostics.append(diagnose_nasdaq_ibit_chain(ibit_chain))
        (raw_dir / "nasdaq_ibit_option_chain.json").write_text(json.dumps(ibit_chain, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        quality_warnings.append(f"IBIT option-chain fetch failed: {exc}")

    btc_per_share, ishares_warnings = try_fetch_btc_per_share(raw_dir, ibit_chain=ibit_chain, btc_spot=btc_spot)
    if btc_per_share is not None:
        source_captures["iShares IBIT holdings"] = datetime.now(ZoneInfo("America/Chicago"))
    quality_warnings.extend(ishares_warnings)

    ibit_atm_rows: list[dict] = []
    ibit_rows: list[dict] = []
    if btc_per_share is not None and ibit_chain is not None:
        try:
            ibit_rows = parse_nasdaq_ibit_rows(ibit_chain, now, btc_per_share)
            ibit_by_expiry: dict[str, list[dict]] = {}
            for row in ibit_rows:
                ibit_by_expiry.setdefault(row["expiry"], []).append(row)
            for _, rows in sorted(ibit_by_expiry.items()):
                try:
                    ibit_atm_rows.append(select_atm_by_moneyness(rows))
                except ValueError:
                    continue
            quality_warnings.append(f"IBIT normalized rows: {len(ibit_rows)}")
            ibit_csv = normalized_dir / "ibit_atm_term_structure.csv"
            with ibit_csv.open("w", newline="", encoding="utf-8") as f:
                fieldnames = ["expiry", "dte", "native_symbol", "option_type", "strike_native", "spot_native", "moneyness_spot", "iv_mark", "open_interest", "volume", "source_confidence", "execution_confidence", "btc_equivalent_per_contract", "strike_btc_equivalent"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(ibit_atm_rows)
        except Exception as exc:  # noqa: BLE001
            quality_warnings.append(f"IBIT option-chain fetch or parse failed: {exc}")

    cme_rows: list[dict] = []
    databento_key = load_databento_api_key()
    if databento_key:
        try:
            cme_snapshot = fetch_cme_btc_option_snapshot(api_key=databento_key, raw_dir=raw_dir, as_of=now)
            cme_rows = list(cme_snapshot.get("normalized_rows") or [])
            source_captures["Databento CME BTC options"] = datetime.now(ZoneInfo("America/Chicago"))
            quality_warnings.append(f"CME Databento normalized rows: {len(cme_rows)}")
            quality_warnings.append(f"CME Databento BBO rows: {cme_snapshot.get('bbo_rows', 0)}")
            cme_csv = normalized_dir / "databento_cme_btc_options_bbo.csv"
            with cme_csv.open("w", newline="", encoding="utf-8") as f:
                fieldnames = ["native_symbol", "instrument_id", "expiry", "dte", "option_type", "strike_native", "underlying", "price_bid", "price_ask", "price_mid", "bid_size", "ask_size", "source_confidence", "execution_confidence", "timestamp_source"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(cme_rows)
        except Exception as exc:  # noqa: BLE001
            quality_warnings.append(f"CME Databento fetch or parse failed: {exc}")
    else:
        quality_warnings.append("CME Databento source missing: DATABENTO_API_KEY not configured")

    dislocations = []
    for ibit in ibit_atm_rows:
        candidates = [row for row in deribit_atm_rows if abs(int(row.get("dte", 0)) - int(ibit.get("dte", 0))) <= 3 and row.get("iv_mark") and ibit.get("iv_mark")]
        if not candidates:
            continue
        deribit = min(candidates, key=lambda row: abs(int(row.get("dte", 0)) - int(ibit.get("dte", 0))))
        dislocations.append(
            {
                "candidate": f"IBIT {ibit['dte']}D ATM vs Deribit {deribit['dte']}D ATM",
                "gross_iv_diff_vol_pts": (float(ibit["iv_mark"]) - float(deribit["iv_mark"])) * 100,
                "confidence": "screen-only",
                "next_action": "quote review" if abs((float(ibit["iv_mark"]) - float(deribit["iv_mark"])) * 100) >= 3 else "watch",
            }
        )
    candidate_triage = rank_dislocation_candidates(dislocations, run_id=run_id, as_of_cst=as_of_cst)
    candidate_ledger_path = normalized_dir / "candidate_triage.jsonl"
    write_candidate_ledger(candidate_ledger_path, candidate_triage)

    atm_csv = normalized_dir / "deribit_atm_term_structure.csv"
    with atm_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["expiry", "dte", "native_symbol", "option_type", "strike_native", "spot_native", "moneyness_spot", "iv_mark", "open_interest", "volume", "source_confidence", "execution_confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deribit_atm_rows)

    report = generate_monitor_report(
        run_id=run_id,
        as_of_cst=as_of_cst,
        btc_spot=btc_spot,
        btc_per_share=btc_per_share,
        deribit_atm_rows=deribit_atm_rows,
        ibit_atm_rows=ibit_atm_rows,
        dislocations=dislocations,
        quality_warnings=quality_warnings,
        cme_rows=cme_rows,
    )
    report_path = reports_dir / f"btc-vol-desk-monitor-{now:%Y-%m-%d-%H%M%S}.md"
    report_path.write_text(report, encoding="utf-8")
    manifest_path = base_dir / "run_manifest.jsonl"
    dashboard_target = base_dir.parent / "dashboard" / "index.html"
    evidence_manifest_json = normalized_dir / "evidence_manifest.json"
    evidence_manifest_md = normalized_dir / "evidence_index.md"
    evidence_bundle_path = normalized_dir / f"{run_id}-evidence-bundle.zip"
    quote_review_candidates = sum(1 for row in dislocations if "quote" in str(row.get("next_action", "")).lower())
    iv_benchmarks = extract_iv_benchmarks(deribit_atm_rows, ibit_atm_rows)
    freshness = evaluate_source_freshness(as_of=datetime.now(ZoneInfo("America/Chicago")), sources=source_captures)
    quality_score = compute_quality_score(
        deribit_rows=len(deribit_atm_rows),
        ibit_rows=len(ibit_rows),
        btc_per_share=btc_per_share,
        cme_available=bool(cme_rows),
        quality_warnings=quality_warnings,
        dislocations=len(dislocations),
        quote_review_candidates=quote_review_candidates,
        freshness=freshness,
    )
    manifest_row = {
        "run_id": run_id,
        "as_of_cst": as_of_cst,
        "btc_spot": btc_spot,
        "btc_per_share": btc_per_share,
        "deribit_atm_rows": len(deribit_atm_rows),
        "ibit_atm_rows": len(ibit_atm_rows),
        "dislocations": len(dislocations),
        "quote_review_candidates": quote_review_candidates,
        "quality_score": quality_score["score"],
        "quality_grade": quality_score["grade"],
        "freshness_grade": freshness["grade"],
        "freshness_stale_sources": freshness["stale_sources"],
        "freshness_missing_sources": freshness["missing_sources"],
        "deribit_market_grade": market_diagnostics[0].get("grade") if market_diagnostics else None,
        "ibit_market_grade": market_diagnostics[1].get("grade") if len(market_diagnostics) > 1 else None,
        "top_candidate_priority": candidate_triage[0].get("priority") if candidate_triage else None,
        "candidate_ledger_path": str(candidate_ledger_path),
        "quote_evidence_ledger_path": str(base_dir / "quote_evidence.jsonl"),
        "evidence_manifest_path": str(evidence_manifest_json),
        "evidence_index_path": str(evidence_manifest_md),
        "evidence_bundle_path": str(evidence_bundle_path),
        "cme_rows": len(cme_rows),
        **iv_benchmarks,
        "report_path": str(report_path),
        "dashboard_path": str(dashboard_target),
    }
    append_run_manifest(manifest_path, manifest_row)
    recent_runs = load_recent_runs(manifest_path, limit=8)
    quote_evidence = load_quote_evidence(base_dir / "quote_evidence.jsonl")
    dashboard_path = write_dashboard(
        dashboard_target,
        run_id=run_id,
        as_of_cst=as_of_cst,
        btc_spot=btc_spot,
        btc_per_share=btc_per_share,
        deribit_atm_rows=deribit_atm_rows,
        ibit_atm_rows=ibit_atm_rows,
        dislocations=dislocations,
        quality_warnings=quality_warnings,
        quality_score=quality_score,
        trend_summary=build_trend_summary(recent_runs),
        quote_evidence=quote_evidence,
        candidate_triage=candidate_triage,
        market_diagnostics=market_diagnostics,
        freshness=freshness,
        recent_runs=recent_runs,
        cme_rows=len(cme_rows),
    )
    evidence_manifest = build_evidence_manifest(
        run_id=run_id,
        as_of_cst=as_of_cst,
        artifacts={
            "report": report_path,
            "dashboard": dashboard_path,
            "deribit_atm_csv": atm_csv,
            "candidate_triage_ledger": candidate_ledger_path,
            "raw_deribit_options": raw_dir / "deribit_book_summary_btc_options.json",
            "raw_ibit_option_chain": raw_dir / "nasdaq_ibit_option_chain.json",
            "raw_ibit_holdings": raw_dir / "ishares_ibit_holdings.csv",
        },
    )
    write_evidence_manifest(evidence_manifest_json, evidence_manifest_md, evidence_manifest)
    bundle_summary = build_evidence_bundle(
        evidence_bundle_path,
        run_id=run_id,
        manifest_json=evidence_manifest_json,
        manifest_markdown=evidence_manifest_md,
    )
    print(json.dumps({"run_id": run_id, "report_path": str(report_path), "dashboard_path": str(dashboard_path), "atm_csv": str(atm_csv), "candidate_ledger_path": str(candidate_ledger_path), "evidence_manifest_path": str(evidence_manifest_json), "evidence_index_path": str(evidence_manifest_md), "evidence_bundle_path": str(evidence_bundle_path),        "evidence_bundle_sha256": bundle_summary["bundle_sha256"], "deribit_atm_rows": len(deribit_atm_rows), "ibit_atm_rows": len(ibit_atm_rows), "cme_rows": len(cme_rows), "dislocations": len(dislocations),"quality_score": quality_score["score"], "quality_grade": quality_score["grade"], "freshness_grade": freshness["grade"], "freshness_stale_sources": freshness["stale_sources"], "freshness_missing_sources": freshness["missing_sources"], "btc_spot": btc_spot, "btc_per_share": btc_per_share}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
