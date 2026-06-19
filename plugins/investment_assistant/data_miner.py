"""Deterministic offline data miner for investment-assistant research inputs."""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .adapters import normalize_market_symbol
from .fmp_provider import FMP_PROVIDER, FmpProvider
from .schemas import Candidate, DiscoveryData, FutuData
from .sec_provider import SecFilingsProvider
from .storage import new_id, utc_now
from .symbol_store import SymbolDataStore


DataLayerState = Literal[
    "fresh",
    "partial",
    "stale",
    "missing",
    "unavailable",
    "not_implemented",
    "skipped",
    "error",
]

DEFAULT_LAYERS = ("sec", "filing_metadata", "filing_text", "filing_sections", "etf")
FMP_LAYER_SPECS: dict[str, tuple[str, str, str]] = {
    "fmp_profile": ("fmp_company_profile.json", "fmp_company_profile", "company_profile"),
    "fmp_etf": ("fmp_etf_exposure.json", "fmp_etf_exposure", "etf_exposure"),
    "fmp_analyst": ("fmp_analyst_expectations.json", "fmp_analyst_expectations", "analyst_expectations"),
    "fmp_transcripts": ("fmp_earnings_transcripts.json", "fmp_earnings_transcripts", "earnings_transcripts"),
    "fmp_metrics": ("fmp_normalized_metrics.json", "fmp_normalized_metrics", "normalized_metrics"),
    "fmp_peers": ("fmp_peer_group.json", "fmp_peer_group", "peer_group"),
    "fmp_ownership": ("fmp_ownership_signal.json", "fmp_ownership_signal", "ownership_signal"),
    "fmp_insider": ("fmp_insider_signal.json", "fmp_insider_signal", "insider_signal"),
}
ETF_SYMBOLS = {"US.QQQ", "US.SOXX", "US.SMH"}
SAFE_SYMBOL_RE = re.compile(r"^[A-Z0-9]+[.][A-Z0-9._-]+$")
SECTION_SPECS = {
    "latest_10k": {
        "form": "10-K",
        "sections": (
            ("part_i_item_1", "PART I", "ITEM 1", "Business"),
            ("part_i_item_1a", "PART I", "ITEM 1A", "Risk Factors"),
            ("part_i_item_1c", "PART I", "ITEM 1C", "Cybersecurity"),
            ("part_ii_item_7", "PART II", "ITEM 7", "Management Discussion and Analysis"),
            ("part_ii_item_7a", "PART II", "ITEM 7A", "Market Risk"),
        ),
    },
    "latest_10q": {
        "form": "10-Q",
        "sections": (
            ("part_i_item_2", "PART I", "ITEM 2", "Management Discussion and Analysis"),
            ("part_ii_item_1a", "PART II", "ITEM 1A", "Risk Factors"),
        ),
    },
    "latest_8k": {
        "form": "8-K",
        "sections": (
            ("item_202", "", "ITEM 2.02", "Results of Operations and Financial Condition"),
            ("item_901", "", "ITEM 9.01", "Financial Statements and Exhibits"),
        ),
    },
    "latest_20f": {
        "form": "20-F",
        "sections": (
            ("item_3d", "", "ITEM 3.D", "Risk Factors"),
            ("item_4", "", "ITEM 4", "Information on the Company"),
            ("item_5", "", "ITEM 5", "Operating and Financial Review and Prospects"),
            ("item_11", "", "ITEM 11", "Market Risk"),
        ),
    },
}


class DataLayerStatus(BaseModel):
    layer: str
    status: DataLayerState
    source: str = ""
    provider: str = ""
    asof: str = ""
    data_asof: str = ""
    updated_at: str = Field(default_factory=utc_now)
    run_id: str = ""
    checksum: str = ""
    path: str = ""
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class SymbolDataManifest(BaseModel):
    artifact_type: str = "symbol_data_manifest"
    symbol: str
    market: str = "US"
    generated_at: str = Field(default_factory=utc_now)
    source_status: DataLayerState = "partial"
    layers: dict[str, DataLayerStatus] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DataMinerRunArtifact(BaseModel):
    artifact_type: str = "data_miner_run"
    run_id: str = Field(default_factory=lambda: new_id("dmr"))
    generated_at: str = Field(default_factory=utc_now)
    output_root: str
    symbols: list[str] = Field(default_factory=list)
    queue: str = "symbols"
    layers: list[str] = Field(default_factory=list)
    symbol_dirs: dict[str, str] = Field(default_factory=dict)
    status_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_data_files_from_triage(
    *,
    triage_state_path: str | Path | None = None,
    output_root: str | Path = "data/investment_assistant",
    queue: Literal["deep", "watch", "all"] = "deep",
    symbols: list[str] | None = None,
    layers: list[str] | None = None,
    max_symbols: int | None = None,
    market: str = "US",
    sec_provider: Any | None = None,
    fmp_provider: Any | None = None,
    skip_existing: bool = False,
    force: bool = False,
) -> DataMinerRunArtifact:
    """Build deterministic symbol data files from a triage artifact or symbols."""

    selected_layers = _normalize_layers(layers)
    selected_symbols = _select_symbols(
        triage_state_path=triage_state_path,
        queue=queue,
        symbols=symbols,
        market=market,
        max_symbols=max_symbols,
    )
    root = Path(output_root)
    run = DataMinerRunArtifact(
        output_root=str(root),
        symbols=selected_symbols,
        queue=queue if triage_state_path else "symbols",
        layers=selected_layers,
    )
    if not selected_symbols:
        run.warnings.append("No symbols selected for data mining.")
        return run

    _load_hermes_env_keys("SEC_EDGAR_IDENTITY", "EDGAR_IDENTITY")
    sec_context = _fetch_sec_context(selected_symbols, sec_provider) if _needs_sec(selected_layers) else {}
    sec_items = sec_context.get("items", {}) if isinstance(sec_context.get("items"), dict) else {}
    run.warnings.extend(str(item) for item in sec_context.get("warnings", []) if item)
    active_fmp_provider = fmp_provider if _needs_fmp(selected_layers) else None
    if active_fmp_provider is None and _needs_fmp(selected_layers):
        active_fmp_provider = FmpProvider()

    for symbol in selected_symbols:
        symbol_dir = _symbol_dir(root, symbol)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        run.symbol_dirs[symbol] = str(symbol_dir)
        manifest = _load_existing_manifest(symbol_dir, symbol)
        if _is_etf_symbol(symbol):
            _write_etf_layers(symbol, symbol_dir, manifest, selected_layers, force=force, skip_existing=skip_existing)
        else:
            item = sec_items.get(symbol)
            _write_sec_layers(symbol, symbol_dir, manifest, item, selected_layers, force=force, skip_existing=skip_existing)
        _write_fmp_layers(
            symbol,
            symbol_dir,
            manifest,
            selected_layers,
            active_fmp_provider,
            force=force,
            skip_existing=skip_existing,
        )
        _finalize_manifest(manifest)
        _write_json(symbol_dir / "manifest.json", manifest.model_dump(mode="json"), force=True, skip_existing=False)

    run.status_counts = _status_counts(root, selected_symbols)
    run_path = root / "data_runs" / run.run_id
    run_path.mkdir(parents=True, exist_ok=True)
    _write_json(run_path / "data_miner_run.json", run.model_dump(mode="json"), force=True, skip_existing=False)
    (run_path / "symbols.txt").write_text("\n".join(selected_symbols) + "\n", encoding="utf-8")
    store = SymbolDataStore(root)
    store.rebuild_index(selected_symbols)
    store.record_data_run(run.model_dump(mode="json"))
    return run


def _select_symbols(
    *,
    triage_state_path: str | Path | None,
    queue: str,
    symbols: list[str] | None,
    market: str,
    max_symbols: int | None,
) -> list[str]:
    if symbols:
        selected = symbols
    elif triage_state_path:
        selected = _symbols_from_triage(_read_json(triage_state_path), queue)
    else:
        selected = []
    normalized = _dedupe([normalize_market_symbol(symbol, market) for symbol in selected])
    if max_symbols is not None:
        normalized = normalized[: max(0, max_symbols)]
    for symbol in normalized:
        _validate_symbol_path(symbol)
    return normalized


def _symbols_from_triage(raw: dict[str, Any], queue: str) -> list[str]:
    triage = raw.get("candidate_triage") if isinstance(raw.get("candidate_triage"), dict) else raw
    if not isinstance(triage, dict):
        return []
    queue_map = {
        "deep": ["deep_enrichment_queue"],
        "watch": ["watchlist"],
        "all": ["deep_enrichment_queue", "watchlist", "deferred", "rejected"],
    }
    result: list[str] = []
    for key in queue_map.get(queue, ["deep_enrichment_queue"]):
        for item in triage.get(key, []) or []:
            if isinstance(item, dict) and item.get("symbol"):
                result.append(str(item["symbol"]))
    return result


def _fetch_sec_context(symbols: list[str], sec_provider: Any | None) -> dict[str, Any]:
    provider = sec_provider or SecFilingsProvider()
    candidates = [_candidate_for_sec(symbol) for symbol in symbols if not _is_etf_symbol(symbol)]
    if not candidates:
        return {"items": {}, "warnings": []}
    return provider.get_sec_context(candidates)


def _candidate_for_sec(symbol: str) -> Candidate:
    return Candidate(
        symbol=symbol,
        name=symbol,
        theme_role="offline SEC data mining target",
        source="data_miner",
        score=0,
        discovery_data=DiscoveryData(
            source="data_miner",
            role="offline SEC data mining target",
            rationale="Synthetic candidate wrapper used only to call SEC provider.",
        ),
        futu_data=FutuData(),
    )


def _write_sec_layers(
    symbol: str,
    symbol_dir: Path,
    manifest: SymbolDataManifest,
    item: dict[str, Any] | None,
    layers: list[str],
    *,
    force: bool,
    skip_existing: bool,
) -> None:
    source_status = str((item or {}).get("source_status") or "missing")
    if "sec" in layers:
        payload = _sec_companyfacts_payload(symbol, item)
        path = symbol_dir / "sec_companyfacts.json"
        _write_json(path, payload, force=force, skip_existing=skip_existing)
        manifest.layers["sec_companyfacts"] = _layer_status(
            "sec_companyfacts",
            _status_from_source(source_status),
            source="sec_companyfacts",
            path=path,
            warnings=payload.get("warnings", []),
            error=str(payload.get("error") or ""),
        )
    if "filing_metadata" in layers:
        payload = _filing_metadata_payload(symbol, item)
        path = symbol_dir / "filing_metadata.json"
        _write_json(path, payload, force=force, skip_existing=skip_existing)
        manifest.layers["filing_metadata"] = _layer_status(
            "filing_metadata",
            _status_from_source(source_status),
            source="sec_filings",
            path=path,
            warnings=payload.get("warnings", []),
            error=str(payload.get("error") or ""),
        )
    if "filing_text" in layers:
        payload = _filing_text_payload(symbol, item, symbol_dir)
        path = symbol_dir / "filing_text.json"
        _write_json(path, payload, force=force, skip_existing=skip_existing)
        manifest.layers["filing_text"] = _layer_status(
            "filing_text",
            str(payload.get("source_status") or "missing"),
            source="sec_filings",
            path=path,
            warnings=payload.get("warnings", []),
        )
    if "filing_sections" in layers:
        payload = _filing_sections_payload(symbol, item, symbol_dir, force=force, skip_existing=skip_existing)
        path = symbol_dir / "filing_sections.json"
        _write_json(path, payload, force=force, skip_existing=skip_existing)
        manifest.layers["filing_sections"] = _layer_status(
            "filing_sections",
            str(payload.get("source_status") or "missing"),
            source="edgartools_sections",
            path=path,
            warnings=payload.get("warnings", []),
            error=str(payload.get("error") or ""),
        )


def _write_etf_layers(
    symbol: str,
    symbol_dir: Path,
    manifest: SymbolDataManifest,
    layers: list[str],
    *,
    force: bool,
    skip_existing: bool,
) -> None:
    if "etf" in layers:
        for filename, layer in (
            ("etf_holdings.json", "etf_holdings"),
            ("overlap_analysis.json", "overlap_analysis"),
        ):
            path = symbol_dir / filename
            payload = {
                "artifact_type": layer,
                "symbol": symbol,
                "generated_at": utc_now(),
                "source_status": "not_implemented",
                "warnings": ["ETF holdings provider not implemented in V1."],
            }
            _write_json(path, payload, force=force, skip_existing=skip_existing)
            manifest.layers[layer] = _layer_status(
                layer,
                "not_implemented",
                source="etf_provider",
                path=path,
                warnings=payload["warnings"],
            )
    for layer in ("sec_companyfacts", "filing_metadata", "filing_text", "filing_sections"):
        if layer == "sec_companyfacts" and "sec" not in layers:
            continue
        if layer in {"filing_metadata", "filing_text", "filing_sections"} and layer not in layers:
            continue
        manifest.layers[layer] = _layer_status(
            layer,
            "skipped",
            source="sec",
            warnings=["ETF symbols are not treated as operating companies in the SEC companyfacts path."],
        )


def _write_fmp_layers(
    symbol: str,
    symbol_dir: Path,
    manifest: SymbolDataManifest,
    layers: list[str],
    provider: Any | None,
    *,
    force: bool,
    skip_existing: bool,
) -> None:
    if provider is None:
        return
    for layer in layers:
        spec = FMP_LAYER_SPECS.get(layer)
        if not spec:
            continue
        filename, manifest_layer, method_name = spec
        path = symbol_dir / filename
        if layer == "fmp_etf" and not _is_etf_symbol(symbol):
            payload = _fmp_skipped_payload(
                manifest_layer,
                symbol,
                "FMP ETF exposure is only collected for supported ETF symbols in this miner path.",
            )
        elif layer != "fmp_etf" and _is_etf_symbol(symbol):
            payload = _fmp_skipped_payload(
                manifest_layer,
                symbol,
                "Operating-company FMP layer skipped for ETF symbol; use fmp_etf for ETF exposure.",
            )
        else:
            try:
                payload = getattr(provider, method_name)(symbol)
            except Exception as exc:
                payload = _fmp_unavailable_payload(manifest_layer, symbol, str(exc))
        _write_json(path, payload, force=force, skip_existing=skip_existing)
        manifest.layers[manifest_layer] = _layer_status(
            manifest_layer,
            _status_from_fmp_source(str(payload.get("source_status") or "missing")),
            source=FMP_PROVIDER,
            path=path,
            warnings=[str(item) for item in payload.get("warnings", []) if item],
            error=str(payload.get("error") or ""),
        )


def _sec_companyfacts_payload(symbol: str, item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return _unavailable_payload("sec_companyfacts", symbol, "No SEC item returned for symbol.")
    return {
        "artifact_type": "sec_companyfacts",
        "symbol": symbol,
        "generated_at": utc_now(),
        "source": "sec_companyfacts",
        "provider": "edgartools",
        "source_status": item.get("source_status", "unknown"),
        "company": {
            "ticker": item.get("ticker", ""),
            "cik": item.get("cik", ""),
            "company_name": item.get("company_name", ""),
            "sic": item.get("sic", ""),
            "industry": item.get("industry", ""),
            "fiscal_year_end": item.get("fiscal_year_end", ""),
            "filer_category": item.get("filer_category", ""),
        },
        "fundamentals": item.get("fundamentals", {}),
        "metric_provenance": item.get("metric_provenance", {}),
        "numeric_evidence": item.get("numeric_evidence", {}),
        "risk_flags": item.get("risk_flags", []),
        "warnings": [],
        "error": item.get("error", ""),
    }


def _filing_metadata_payload(symbol: str, item: dict[str, Any] | None) -> dict[str, Any]:
    if not item:
        return _unavailable_payload("filing_metadata", symbol, "No SEC item returned for symbol.")
    return {
        "artifact_type": "filing_metadata",
        "symbol": symbol,
        "generated_at": utc_now(),
        "source": "sec_filings",
        "provider": "edgartools",
        "source_status": item.get("source_status", "unknown"),
        "filings": item.get("filings", {}),
        "event_context": item.get("event_context", {}),
        "risk_flags": item.get("risk_flags", []),
        "warnings": [],
        "error": item.get("error", ""),
    }


def _filing_text_payload(symbol: str, item: dict[str, Any] | None, symbol_dir: Path) -> dict[str, Any]:
    if not item:
        return _unavailable_payload("filing_text", symbol, "No SEC item returned for symbol.")
    filings = item.get("filings", {}) if isinstance(item.get("filings"), dict) else {}
    raw_dir = symbol_dir / "raw_filings"
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    status_values: list[str] = []
    for key in _filing_keys(filings):
        filing = filings.get(key) if isinstance(filings.get(key), dict) else None
        entry = _capture_filing_text(key, filing, raw_dir)
        entries[key] = entry
        status_values.append(str(entry.get("source_status") or "missing"))
        warnings.extend(str(item) for item in entry.get("warnings", []))
    return {
        "artifact_type": "filing_text",
        "symbol": symbol,
        "generated_at": utc_now(),
        "source": "sec_filings",
        "provider": "edgartools_url",
        "source_status": _combined_status(status_values),
        "filings": entries,
        "warnings": warnings,
    }


def _filing_sections_payload(
    symbol: str,
    item: dict[str, Any] | None,
    symbol_dir: Path,
    *,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    if not item:
        return _unavailable_payload("filing_sections", symbol, "No SEC item returned for symbol.")
    ticker = str(item.get("ticker") or symbol.split(".", 1)[-1]).strip().upper()
    if not ticker:
        return _unavailable_payload("filing_sections", symbol, "No ticker available for section extraction.")
    try:
        edgar = _import_edgar_for_sections()
        edgar.set_identity(_sec_user_agent())
        company = edgar.Company(ticker)
    except Exception as exc:
        return _unavailable_payload("filing_sections", symbol, f"edgartools section provider unavailable: {exc}")

    output_dir = symbol_dir / "filing_sections"
    filings_meta = item.get("filings", {}) if isinstance(item.get("filings"), dict) else {}
    filings: dict[str, Any] = {}
    warnings: list[str] = []
    statuses: list[str] = []
    for filing_key, spec in SECTION_SPECS.items():
        filing_meta = filings_meta.get(filing_key) if isinstance(filings_meta.get(filing_key), dict) else {}
        if not filing_meta:
            continue
        result = _extract_filing_sections(
            company,
            filing_key,
            spec,
            filing_meta,
            output_dir / filing_key,
            force=force,
            skip_existing=skip_existing,
        )
        filings[filing_key] = result
        statuses.append(str(result.get("source_status") or "missing"))
        warnings.extend(str(item) for item in result.get("warnings", []))
    return {
        "artifact_type": "filing_sections",
        "symbol": symbol,
        "generated_at": utc_now(),
        "source": "sec_filings",
        "provider": "edgartools_chunked_document",
        "source_status": _combined_status(statuses),
        "filings": filings,
        "warnings": warnings,
    }


def _extract_filing_sections(
    company: Any,
    filing_key: str,
    spec: dict[str, Any],
    filing_meta: dict[str, Any],
    output_dir: Path,
    *,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    form = str(spec.get("form") or "").strip()
    expected_accession = str(filing_meta.get("accession_number") or "").strip()
    warnings: list[str] = []
    if form == "20-F":
        raw_path = output_dir.parent.parent / "raw_filings" / f"{filing_key}.html"
        if raw_path.exists():
            return _extract_raw_filing_sections(
                raw_path,
                filing_key,
                spec,
                filing_meta,
                output_dir,
                force=force,
                skip_existing=skip_existing,
            )
    try:
        filing = company.get_filings(form=form)[0]
        actual_accession = str(getattr(filing, "accession_number", "") or "")
        if expected_accession and actual_accession and expected_accession != actual_accession:
            warnings.append(
                f"{filing_key} accession mismatch: metadata={expected_accession}, edgartools={actual_accession}."
            )
        obj = filing.obj()
    except Exception as exc:
        return {
            "source_status": "unavailable",
            "form": form,
            "accession_number": expected_accession,
            "sections": {},
            "warnings": [f"{filing_key} section extraction failed: {exc}"],
            "error": str(exc),
        }

    section_entries: dict[str, Any] = {}
    statuses: list[str] = []
    for section_key, part, item, title in spec.get("sections", ()):
        entry = _extract_one_section(
            obj,
            output_dir,
            section_key,
            part,
            item,
            title,
            force=force,
            skip_existing=skip_existing,
        )
        section_entries[section_key] = entry
        statuses.append(str(entry.get("source_status") or "missing"))
        warnings.extend(str(item) for item in entry.get("warnings", []))
    return {
        "source_status": _combined_status(statuses),
        "form": form,
        "accession_number": str(getattr(filing, "accession_number", "") or expected_accession),
        "filing_date": _safe_date_text(getattr(filing, "filing_date", filing_meta.get("filing_date"))),
        "period_of_report": _safe_date_text(getattr(filing, "period_of_report", filing_meta.get("period_of_report"))),
        "primary_document": str(getattr(filing, "primary_document", filing_meta.get("primary_document", "")) or ""),
        "extraction_method": "edgartools_chunked_document.get_item_with_part",
        "detected_sections": _detected_sections(obj),
        "sections": section_entries,
        "warnings": warnings,
    }


def _extract_raw_filing_sections(
    raw_path: Path,
    filing_key: str,
    spec: dict[str, Any],
    filing_meta: dict[str, Any],
    output_dir: Path,
    *,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    text = _html_to_plain_text(raw_path.read_text(encoding="utf-8", errors="replace"))
    section_entries: dict[str, Any] = {}
    statuses: list[str] = []
    warnings: list[str] = []
    for section_key, part, item, title in spec.get("sections", ()):
        entry = _extract_one_raw_section(
            text,
            output_dir,
            section_key,
            part,
            item,
            title,
            force=force,
            skip_existing=skip_existing,
        )
        section_entries[section_key] = entry
        statuses.append(str(entry.get("source_status") or "missing"))
        warnings.extend(str(item) for item in entry.get("warnings", []))
    return {
        "source_status": _combined_status(statuses),
        "form": str(spec.get("form") or ""),
        "accession_number": str(filing_meta.get("accession_number") or ""),
        "filing_date": filing_meta.get("filing_date") or "",
        "period_of_report": filing_meta.get("period_of_report") or "",
        "primary_document": str(filing_meta.get("primary_document") or ""),
        "extraction_method": "raw_html_item_regex",
        "detected_sections": [],
        "sections": section_entries,
        "warnings": warnings,
    }


def _extract_one_raw_section(
    text: str,
    output_dir: Path,
    section_key: str,
    part: str,
    item: str,
    title: str,
    *,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    markdown = _raw_item_markdown(text, item)
    markdown = str(markdown or "").strip()
    if not markdown:
        return {
            "source_status": "missing",
            "section_key": section_key,
            "part": part,
            "item": item,
            "title": title,
            "warnings": [f"{section_key} produced empty markdown."],
        }
    path = output_dir / f"{section_key}.md"
    _write_text(path, f"# {item} {title}\n\n{markdown}\n", force=force, skip_existing=skip_existing)
    return {
        "source_status": "fresh",
        "section_key": section_key,
        "part": part,
        "item": item,
        "title": title,
        "path": str(path),
        "char_count": len(markdown),
        "line_count": markdown.count("\n") + 1,
        "warnings": [],
    }


def _html_to_plain_text(html_text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html_text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li|table|section)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _raw_item_markdown(text: str, item: str) -> str:
    item_match = re.search(r"(\d{1,2})(?:\s*\.?\s*([A-Z]))?", item.upper())
    if not item_match:
        return ""
    target = (item_match.group(1), item_match.group(2) or "")
    headings = list(_raw_item_headings(text))
    target_indices = [idx for idx, heading in enumerate(headings) if heading[:2] == target]
    spans: list[tuple[int, int]] = []
    for idx in target_indices:
        start = headings[idx][2]
        end = len(text)
        for next_heading in headings[idx + 1 :]:
            if next_heading[2] > start:
                end = next_heading[2]
                break
        if end > start:
            spans.append((start, end))
    if not spans:
        return _raw_title_fallback_markdown(text, item)
    start, end = max(spans, key=lambda pair: pair[1] - pair[0])
    section = text[start:end].strip()
    fallback = _raw_title_fallback_markdown(text, item)
    if len(section) < 1_000 and len(fallback) > len(section) * 3:
        section = fallback
    return section[:500_000].strip()


def _raw_item_headings(text: str):
    pattern = re.compile(r"\bITEM\s+(\d{1,2})(?:\s*\.?\s*([A-Z]))?\.?\b", re.IGNORECASE)
    for match in pattern.finditer(text):
        yield (match.group(1), (match.group(2) or "").upper(), match.start())


def _raw_title_fallback_markdown(text: str, item: str) -> str:
    item_text = item.upper()
    if "3" in item_text and "D" in item_text:
        return _slice_by_titles(text, [r"\bRisk factors\b"], [r"\bInformation security\b", r"\bCorporate governance\b"])
    if re.search(r"\b4\b", item_text):
        section = _slice_by_titles(
            text,
            [r"\bAt a glance\s+[–-]\s+\d{4}\s+overview\b"],
            [r"\bFinancial\s+performance\b"],
        )
        if section:
            return section
        return _slice_by_titles(text, [r"\bOur business\b"], [r"\bFinancial\s+performance\b"])
    if re.search(r"\b5\b", item_text):
        return _slice_by_titles(
            text,
            [r"\bFinancial\s+performance\b", r"\bMessage from our CFO\b"],
            [r"\bRisk and security\b", r"\bCorporate governance\b"],
        )
    if re.search(r"\b11\b", item_text):
        return _slice_by_titles(
            text,
            [r"\bFinancial risk management\b", r"\bMarket risk\b"],
            [r"\bCapital risk\b", r"\bCredit risk\b", r"\bLiquidity risk\b"],
            default_chars=80_000,
        )
    return ""


def _slice_by_titles(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    *,
    default_chars: int = 200_000,
) -> str:
    spans: list[tuple[int, int]] = []
    starts = _pattern_positions(text, start_patterns)
    ends = _pattern_positions(text, end_patterns)
    for start in starts:
        end = next((candidate for candidate in ends if candidate > start + 1_000), min(len(text), start + default_chars))
        if end > start:
            spans.append((start, end))
    if not spans:
        return ""
    preferred = [
        pair
        for pair in spans
        if pair[0] > 50_000 and pair[0] < len(text) * 0.75 and pair[1] - pair[0] > 5_000
    ]
    start, end = min(preferred, key=lambda pair: pair[0]) if preferred else max(spans, key=lambda pair: pair[1] - pair[0])
    return text[start:end].strip()


def _pattern_positions(text: str, patterns: list[str]) -> list[int]:
    positions: list[int] = []
    for pattern in patterns:
        positions.extend(match.start() for match in re.finditer(pattern, text, re.IGNORECASE))
    return sorted(set(positions))


def _extract_one_section(
    obj: Any,
    output_dir: Path,
    section_key: str,
    part: str,
    item: str,
    title: str,
    *,
    force: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    try:
        markdown = _section_markdown(obj, part, item)
    except Exception as exc:
        return {
            "source_status": "unavailable",
            "section_key": section_key,
            "part": part,
            "item": item,
            "title": title,
            "warnings": [f"{section_key} extraction failed: {exc}"],
            "error": str(exc),
        }
    markdown = str(markdown or "").strip()
    if not markdown:
        return {
            "source_status": "missing",
            "section_key": section_key,
            "part": part,
            "item": item,
            "title": title,
            "warnings": [f"{section_key} produced empty markdown."],
        }
    path = output_dir / f"{section_key}.md"
    _write_text(path, markdown + "\n", force=force, skip_existing=skip_existing)
    return {
        "source_status": "fresh",
        "section_key": section_key,
        "part": part,
        "item": item,
        "title": title,
        "path": str(path),
        "char_count": len(markdown),
        "line_count": markdown.count("\n") + 1,
        "warnings": [],
    }


def _section_markdown(obj: Any, part: str, item: str) -> str:
    chunked = getattr(obj, "chunked_document", None)
    if chunked is None or not hasattr(chunked, "get_item_with_part"):
        return ""
    for candidate in _item_variants(item):
        try:
            text = chunked.get_item_with_part(part, candidate, markdown=True)
        except Exception:
            continue
        if str(text or "").strip():
            return str(text)
    return ""


def _item_variants(item: str) -> list[str]:
    text = str(item or "").strip()
    if not text:
        return [text]
    variants = [text]
    if text.endswith("."):
        variants.append(text.rstrip("."))
    if "." in text:
        variants.append(text.replace(".", ""))
        variants.append(text.replace(".", ". "))
    return _dedupe([variant.strip() for variant in variants if variant.strip()])


def _detected_sections(obj: Any) -> list[dict[str, Any]]:
    sections = getattr(obj, "sections", None)
    if not hasattr(sections, "items"):
        return []
    result: list[dict[str, Any]] = []
    for key, section in sections.items():
        result.append(
            {
                "section_key": str(key),
                "title": str(getattr(section, "title", "") or ""),
                "part": str(getattr(section, "part", "") or ""),
                "item": str(getattr(section, "item", "") or ""),
                "confidence": _safe_number(getattr(section, "confidence", None)),
                "method": str(getattr(section, "method", "") or ""),
            }
        )
    return result


def _filing_keys(filings: dict[str, Any]) -> list[str]:
    preferred = ["latest_10k", "latest_10q", "latest_8k", "latest_20f", "latest_6k"]
    keys = [key for key in preferred if isinstance(filings.get(key), dict)]
    keys.extend(sorted(key for key, value in filings.items() if key not in keys and isinstance(value, dict)))
    return _dedupe(keys)


def _capture_filing_text(key: str, filing: dict[str, Any] | None, raw_dir: Path) -> dict[str, Any]:
    if not filing or not _filing_document_url(filing):
        return {
            "source_status": "missing",
            "warnings": [f"{key} has no downloadable filing document URL."],
        }
    url = _filing_document_url(filing)
    filename = f"{key}.html"
    path = raw_dir / filename
    try:
        content, truncated, warnings = _fetch_url_bytes(url, _sec_user_agent())
        path.write_bytes(content)
        return {
            "source_status": "partial" if truncated else "fresh",
            "form": filing.get("form", ""),
            "filing_date": filing.get("filing_date"),
            "period_of_report": filing.get("period_of_report"),
            "accession_number": filing.get("accession_number", ""),
            "source_url": url,
            "homepage_url": filing.get("homepage_url") or filing.get("url") or "",
            "text_url": filing.get("text_url") or "",
            "primary_document": filing.get("primary_document") or "",
            "retrieved_at": utc_now(),
            "extraction_method": "sec_filing_url_download",
            "local_file_path": str(path),
            "byte_count": len(content),
            "truncated": truncated,
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "source_status": "unavailable",
            "form": filing.get("form", ""),
            "filing_date": filing.get("filing_date"),
            "period_of_report": filing.get("period_of_report"),
            "accession_number": filing.get("accession_number", ""),
            "source_url": url,
            "homepage_url": filing.get("homepage_url") or filing.get("url") or "",
            "text_url": filing.get("text_url") or "",
            "primary_document": filing.get("primary_document") or "",
            "retrieved_at": utc_now(),
            "extraction_method": "sec_filing_url_download",
            "error": str(exc),
            "warnings": [f"{key} download failed: {exc}"],
        }


def _filing_document_url(filing: dict[str, Any]) -> str:
    for key in ("filing_url", "text_url", "url"):
        value = str(filing.get(key) or "").strip()
        if value:
            return value
    return ""


def _fetch_url_bytes(url: str, user_agent: str) -> tuple[bytes, bool, list[str]]:
    max_bytes = int(os.getenv("IA_MINER_MAX_FILING_BYTES", str(50 * 1024 * 1024)))
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    chunks: list[bytes] = []
    total = 0
    truncated = False
    with urllib.request.urlopen(request, timeout=60) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                allowed = max(0, len(chunk) - (total - max_bytes))
                if allowed:
                    chunks.append(chunk[:allowed])
                truncated = True
                break
            chunks.append(chunk)
    warnings = [f"Filing text truncated at {max_bytes} bytes."] if truncated else []
    return b"".join(chunks), truncated, warnings


def _import_edgar_for_sections() -> Any:
    try:
        from tools.lazy_deps import ensure

        ensure("investment.edgartools", prompt=False)
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    try:
        import edgar
    except Exception as exc:
        raise RuntimeError(f"failed to import edgar module: {exc}") from exc
    return edgar


def _unavailable_payload(artifact_type: str, symbol: str, error: str) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "symbol": symbol,
        "generated_at": utc_now(),
        "source_status": "unavailable",
        "warnings": [error],
        "error": error,
    }


def _load_existing_manifest(symbol_dir: Path, symbol: str) -> SymbolDataManifest:
    path = symbol_dir / "manifest.json"
    if not path.exists():
        return SymbolDataManifest(symbol=symbol, market=symbol.split(".", 1)[0])
    try:
        manifest = SymbolDataManifest.model_validate(_read_json(path))
    except Exception:
        return SymbolDataManifest(symbol=symbol, market=symbol.split(".", 1)[0])
    manifest.symbol = symbol
    manifest.market = symbol.split(".", 1)[0]
    return manifest


def _fmp_skipped_payload(artifact_type: str, symbol: str, warning: str) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "symbol": symbol,
        "provider": FMP_PROVIDER,
        "generated_at": utc_now(),
        "source_status": "skipped",
        "endpoints": [],
        "data_asof": {},
        "warnings": [warning],
        "raw": {},
    }


def _fmp_unavailable_payload(artifact_type: str, symbol: str, error: str) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "symbol": symbol,
        "provider": FMP_PROVIDER,
        "generated_at": utc_now(),
        "source_status": "unavailable",
        "endpoints": [],
        "data_asof": {},
        "warnings": [error],
        "error": error,
        "raw": {},
    }


def _finalize_manifest(manifest: SymbolDataManifest) -> None:
    statuses = [layer.status for layer in manifest.layers.values()]
    if not statuses:
        manifest.source_status = "missing"
    elif all(status == "fresh" for status in statuses):
        manifest.source_status = "fresh"
    elif any(status in {"fresh", "partial"} for status in statuses):
        manifest.source_status = "partial"
    elif any(status == "not_implemented" for status in statuses):
        manifest.source_status = "partial"
    else:
        manifest.source_status = "unavailable"
    for layer in manifest.layers.values():
        manifest.warnings.extend(layer.warnings)
    manifest.warnings = _dedupe(manifest.warnings)


def _layer_status(
    layer: str,
    status: str,
    *,
    source: str = "",
    path: Path | None = None,
    warnings: list[str] | None = None,
    error: str = "",
) -> DataLayerStatus:
    return DataLayerStatus(
        layer=layer,
        status=status if status in DataLayerState.__args__ else "error",
        source=source,
        provider=source,
        asof=utc_now(),
        updated_at=utc_now(),
        path=str(path or ""),
        warnings=warnings or [],
        error=error,
    )


def _status_from_source(source_status: str) -> DataLayerState:
    if source_status == "available":
        return "fresh"
    if source_status == "partial":
        return "partial"
    if source_status in {"missing", "not_configured", "not_available", "unavailable"}:
        return "unavailable"
    return "partial"


def _status_from_fmp_source(source_status: str) -> DataLayerState:
    if source_status == "fresh":
        return "fresh"
    if source_status in {"partial", "rate_limited"}:
        return "partial"
    if source_status == "skipped":
        return "skipped"
    if source_status == "missing":
        return "missing"
    if source_status in {"unavailable", "not_configured", "not_available"}:
        return "unavailable"
    return "partial"


def _combined_status(statuses: list[str]) -> DataLayerState:
    if not statuses:
        return "missing"
    if all(status == "fresh" for status in statuses):
        return "fresh"
    if any(status in {"fresh", "partial"} for status in statuses):
        return "partial"
    if any(status == "unavailable" for status in statuses):
        return "unavailable"
    return "missing"


def _normalize_layers(layers: list[str] | None) -> list[str]:
    values = [str(item).strip().lower() for item in (layers or list(DEFAULT_LAYERS))]
    aliases = {
        "companyfacts": "sec",
        "sec_companyfacts": "sec",
        "filings": "filing_metadata",
        "sections": "filing_sections",
        "fmp": "fmp_all",
        "all_fmp": "fmp_all",
        "fmp_company_profile": "fmp_profile",
        "fmp_etf_exposure": "fmp_etf",
        "fmp_analyst_expectations": "fmp_analyst",
        "fmp_earnings_transcripts": "fmp_transcripts",
        "fmp_normalized_metrics": "fmp_metrics",
        "fmp_peer_group": "fmp_peers",
        "fmp_ownership_signal": "fmp_ownership",
        "fmp_insider_signal": "fmp_insider",
    }
    normalized: list[str] = []
    for item in values:
        if not item:
            continue
        layer = aliases.get(item, item)
        if layer == "fmp_all":
            normalized.extend(FMP_LAYER_SPECS)
            continue
        normalized.append(layer)
    allowed = {"sec", "filing_metadata", "filing_text", "filing_sections", "etf", *FMP_LAYER_SPECS}
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"Unknown data miner layers: {unknown}")
    return _dedupe(normalized)


def _needs_sec(layers: list[str]) -> bool:
    return any(layer in layers for layer in ("sec", "filing_metadata", "filing_text", "filing_sections"))


def _needs_fmp(layers: list[str]) -> bool:
    return any(layer in FMP_LAYER_SPECS for layer in layers)


def _symbol_dir(root: Path, symbol: str) -> Path:
    _validate_symbol_path(symbol)
    return root / "symbols" / symbol


def _validate_symbol_path(symbol: str) -> None:
    if not SAFE_SYMBOL_RE.match(symbol):
        raise ValueError(f"Unsafe or unsupported symbol path: {symbol!r}")


def _is_etf_symbol(symbol: str) -> bool:
    return symbol.upper() in ETF_SYMBOLS


def _write_json(path: Path, payload: Any, *, force: bool, skip_existing: bool) -> None:
    if path.exists() and skip_existing and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str, *, force: bool, skip_existing: bool) -> None:
    if path.exists() and skip_existing and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _status_counts(root: Path, symbols: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in symbols:
        path = _symbol_dir(root, symbol) / "manifest.json"
        if not path.exists():
            counts["missing"] = counts.get("missing", 0) + 1
            continue
        status = str(_read_json(path).get("source_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _sec_user_agent() -> str:
    _load_hermes_env_keys("SEC_EDGAR_IDENTITY", "EDGAR_IDENTITY")
    return (
        os.getenv("SEC_EDGAR_IDENTITY", "").strip()
        or os.getenv("EDGAR_IDENTITY", "").strip()
        or "hermes-agent investment-assistant data-miner"
    )


def _safe_date_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _load_hermes_env_keys(*keys: str) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if not missing:
        return
    try:
        from hermes_constants import get_hermes_env_path

        path = get_hermes_env_path()
    except Exception:
        path = Path.home() / ".hermes" / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    wanted = set(missing)
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in wanted or os.getenv(key):
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
