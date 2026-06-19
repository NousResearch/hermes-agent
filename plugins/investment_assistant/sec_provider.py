"""SEC filings and companyfacts provider backed by edgartools."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from typing import Any

from .schemas import Candidate
from .storage import utc_now


class SecProviderError(RuntimeError):
    """Raised when the SEC provider cannot produce a usable response."""


@dataclass(frozen=True)
class SecProviderConfig:
    identity: str | None
    max_symbols: int
    periodic_stale_after_days: int
    recent_8k_days: int
    enabled: bool

    @classmethod
    def from_env(cls) -> "SecProviderConfig":
        identity = (
            os.getenv("SEC_EDGAR_IDENTITY", "").strip()
            or os.getenv("EDGAR_IDENTITY", "").strip()
            or None
        )
        return cls(
            identity=identity,
            max_symbols=max(1, int(os.getenv("IA_SEC_MAX_SYMBOLS", "8"))),
            periodic_stale_after_days=max(30, int(os.getenv("IA_SEC_PERIODIC_STALE_DAYS", "140"))),
            recent_8k_days=max(1, int(os.getenv("IA_SEC_RECENT_8K_DAYS", "30"))),
            enabled=os.getenv("IA_SEC_ENABLED", "1").strip() != "0",
        )


class SecFilingsProvider:
    """Fetch SEC filings and companyfacts summaries for US equity candidates."""

    def __init__(self, config: SecProviderConfig | None = None):
        self.config = config or SecProviderConfig.from_env()

    def get_sec_context(self, candidates: list[Candidate]) -> dict[str, Any]:
        generated_at = utc_now()
        symbols = _dedupe([candidate.symbol for candidate in candidates if _is_us_symbol(candidate.symbol)])
        if not self.config.enabled:
            return self._not_available(generated_at, "disabled", symbols)
        if not symbols:
            return self._not_available(generated_at, "no_us_equity_symbols", symbols)
        if not self.config.identity:
            return self._not_available(generated_at, "identity_not_configured", symbols)

        try:
            edgar = _import_edgar()
            edgar.set_identity(self.config.identity)
        except Exception as exc:
            return self._not_available(generated_at, f"provider_unavailable: {exc}", symbols)

        warnings: list[str] = []
        items: dict[str, dict[str, Any]] = {}
        requested_symbols = symbols
        fetch_symbols = symbols[: self.config.max_symbols]
        if len(fetch_symbols) < len(requested_symbols):
            warnings.append(
                f"SEC fetch limited to {len(fetch_symbols)} of {len(requested_symbols)} symbols; "
                "raise IA_SEC_MAX_SYMBOLS to fetch more."
            )

        for symbol in fetch_symbols:
            ticker = _ticker_from_symbol(symbol)
            try:
                company = edgar.Company(ticker)
                items[symbol] = self._company_context(symbol, ticker, company)
            except Exception as exc:
                warnings.append(f"SEC fetch failed for {symbol}: {exc}")
                items[symbol] = {
                    "symbol": symbol,
                    "ticker": ticker,
                    "source_status": "unavailable",
                    "error": str(exc),
                    "risk_flags": ["sec_data_unavailable"],
                }

        status = "available" if items and not warnings else "partial"
        if not any(item.get("source_status") == "available" for item in items.values()):
            status = "unavailable"
        return {
            "source": "edgartools",
            "source_status": status,
            "generated_at": generated_at,
            "requested_symbols": requested_symbols,
            "fetched_symbols": fetch_symbols,
            "items": items,
            "warnings": warnings,
        }

    def _company_context(self, symbol: str, ticker: str, company: Any) -> dict[str, Any]:
        facts = company.get_facts()
        filings = {
            "latest_10k": _latest_filing(company, "10-K"),
            "latest_10q": _latest_filing(company, "10-Q"),
            "latest_8k": _latest_filing(company, "8-K"),
            "latest_20f": _latest_filing(company, "20-F"),
            "latest_6k": _latest_filing(company, "6-K"),
        }
        periodic_date = _latest_periodic_date(filings)
        recent_8k = _days_since(_filing_date(filings.get("latest_8k")))
        periodic_age = _days_since(periodic_date)
        stale = periodic_age is None or periodic_age > self.config.periodic_stale_after_days
        fundamentals, metric_provenance = _fundamentals_with_provenance(facts)
        risk_flags = []
        if stale:
            risk_flags.append("sec_periodic_filing_stale")
        if recent_8k is not None and recent_8k <= self.config.recent_8k_days:
            risk_flags.append("recent_8k")
        if fundamentals.get("ttm_net_income") is not None and fundamentals["ttm_net_income"] < 0:
            risk_flags.append("negative_ttm_net_income")

        return {
            "symbol": symbol,
            "ticker": ticker,
            "source_status": "available",
            "cik": _safe_str(getattr(company, "cik", "")),
            "company_name": _safe_str(getattr(company, "name", "")),
            "sic": _safe_str(getattr(company, "sic", "")),
            "industry": _safe_str(getattr(company, "industry", "")),
            "fiscal_year_end": _safe_str(getattr(company, "fiscal_year_end", "")),
            "filer_category": _safe_str(getattr(company, "filer_category", "")),
            "filings": filings,
            "fundamentals": fundamentals,
            "metric_provenance": metric_provenance,
            "numeric_evidence": {
                "source": "sec_companyfacts",
                "provider": "edgartools",
                "llm_generated": False,
                "provenance_included": True,
                "metric_keys": sorted(fundamentals),
            },
            "narrative_evidence": {
                "source_status": "not_implemented",
                "planned_pipeline": "mineru_plus_sub_llm",
                "numeric_extraction_allowed": False,
            },
            "event_context": {
                "latest_periodic_filing_date": _date_iso(periodic_date),
                "periodic_filing_age_days": periodic_age,
                "periodic_filing_stale": stale,
                "latest_8k_age_days": recent_8k,
                "event_risk_level": _event_risk_level(stale, recent_8k, self.config.recent_8k_days),
            },
            "risk_flags": risk_flags,
        }

    def _not_available(self, generated_at: str, reason: str, symbols: list[str]) -> dict[str, Any]:
        status = "not_configured" if reason == "identity_not_configured" else "not_available"
        return {
            "source": "edgartools",
            "source_status": status,
            "generated_at": generated_at,
            "requested_symbols": symbols,
            "fetched_symbols": [],
            "items": {},
            "warnings": [_provider_warning(reason)],
        }


def _import_edgar() -> Any:
    try:
        from tools.lazy_deps import ensure

        ensure("investment.edgartools", prompt=False)
    except Exception as exc:
        raise SecProviderError(str(exc)) from exc

    try:
        import edgar
    except Exception as exc:
        raise SecProviderError(f"failed to import edgar module: {exc}") from exc
    return edgar


def _latest_filing(company: Any, form: str) -> dict[str, Any] | None:
    try:
        filings = company.get_filings(form=form)
        filing = next(
            (item for item in filings if _safe_str(getattr(item, "form", "")).upper() == form.upper()),
            filings[0],
        )
    except Exception:
        return None
    return {
        "form": _safe_str(getattr(filing, "form", form)),
        "filing_date": _date_iso(getattr(filing, "filing_date", None)),
        "period_of_report": _date_iso(getattr(filing, "period_of_report", None)),
        "accession_number": _safe_str(getattr(filing, "accession_number", "")),
        "url": _safe_str(getattr(filing, "url", "")),
        "homepage_url": _safe_str(getattr(filing, "homepage_url", "")),
        "filing_url": _safe_str(getattr(filing, "filing_url", "")),
        "text_url": _safe_str(getattr(filing, "text_url", "")),
        "primary_document": _safe_str(getattr(filing, "primary_document", "")),
        "size": _safe_number(getattr(filing, "size", None)),
    }


def _fundamentals(facts: Any) -> dict[str, Any]:
    metrics, _provenance = _fundamentals_with_provenance(facts)
    return metrics


def _fundamentals_with_provenance(facts: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    provenance: dict[str, Any] = {}

    for key, method in (
        ("ttm_revenue", "get_ttm_revenue"),
        ("ttm_net_income", "get_ttm_net_income"),
    ):
        value = _call_or_none(facts, method)
        metrics[key] = _metric_value(value)
        provenance[key] = _ttm_provenance(value)

    for key, concept, fallback_method in (
        ("annual_revenue", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "get_revenue"),
        ("annual_net_income", "us-gaap:NetIncomeLoss", "get_net_income"),
        ("gross_profit", "us-gaap:GrossProfit", "get_gross_profit"),
        ("operating_income", "us-gaap:OperatingIncomeLoss", "get_operating_income"),
        ("total_assets", "us-gaap:Assets", "get_total_assets"),
        ("total_liabilities", "us-gaap:Liabilities", "get_total_liabilities"),
        ("shareholders_equity", "us-gaap:StockholdersEquity", "get_shareholders_equity"),
    ):
        fact = _get_annual_fact(facts, concept)
        if fact is not None:
            metrics[key] = _metric_value(fact)
            provenance[key] = _fact_provenance(fact)
        else:
            value = _call_or_none(facts, fallback_method)
            metrics[key] = _metric_value(value)
            provenance[key] = _value_provenance(value, concept=concept, method=fallback_method)

    metrics["debt_to_assets"] = _ratio(metrics.get("total_liabilities"), metrics.get("total_assets"))
    metrics["roe"] = _ratio(metrics.get("ttm_net_income"), metrics.get("shareholders_equity"))
    metrics["net_margin"] = _ratio(metrics.get("ttm_net_income"), metrics.get("ttm_revenue"))
    provenance["debt_to_assets"] = _ratio_provenance(
        metrics["debt_to_assets"],
        numerator="total_liabilities",
        denominator="total_assets",
    )
    provenance["roe"] = _ratio_provenance(
        metrics["roe"],
        numerator="ttm_net_income",
        denominator="shareholders_equity",
    )
    provenance["net_margin"] = _ratio_provenance(
        metrics["net_margin"],
        numerator="ttm_net_income",
        denominator="ttm_revenue",
    )
    return metrics, provenance


def _call_or_none(obj: Any, method: str) -> Any:
    try:
        return getattr(obj, method)()
    except Exception:
        return None


def _get_annual_fact(facts: Any, concept: str) -> Any:
    try:
        return facts.get_annual_fact(concept)
    except Exception:
        return None


def _ttm_provenance(value: Any) -> dict[str, Any]:
    base = _value_provenance(value)
    base["derived"] = True
    base["calculation"] = "edgartools_ttm"
    base["concept"] = _safe_str(getattr(value, "concept", base.get("concept", "")))
    base["label"] = _safe_str(getattr(value, "label", ""))
    base["unit"] = _safe_str(getattr(value, "unit", ""))
    base["as_of_date"] = _date_iso(getattr(value, "as_of_date", None))
    base["has_gaps"] = bool(getattr(value, "has_gaps", False))
    base["has_calculated_q4"] = bool(getattr(value, "has_calculated_q4", False))
    warning = _safe_str(getattr(value, "warning", ""))
    if warning:
        base["warning"] = warning
    components = []
    for fact in getattr(value, "period_facts", []) or []:
        components.append(_fact_provenance(fact))
    base["components"] = components
    base["component_count"] = len(components)
    quality_values = [item.get("data_quality") for item in components if item.get("data_quality")]
    base["data_quality"] = "mixed" if len(set(quality_values)) > 1 else (quality_values[0] if quality_values else "")
    return base


def _value_provenance(value: Any, *, concept: str = "", method: str = "") -> dict[str, Any]:
    return {
        "source": "edgartools",
        "concept": concept or _safe_str(getattr(value, "concept", "")),
        "value": _metric_value(value),
        "unit": _safe_str(getattr(value, "unit", getattr(value, "normalized_unit", ""))),
        "derived": False,
        "method": method,
        "provenance_available": value is not None and not isinstance(value, (int, float, str)),
    }


def _fact_provenance(fact: Any) -> dict[str, Any]:
    data = {
        "source": "edgartools",
        "concept": _safe_str(getattr(fact, "concept", "")),
        "label": _safe_str(getattr(fact, "label", "")),
        "value": _metric_value(fact),
        "unit": _safe_str(getattr(fact, "unit", "")),
        "derived": bool(getattr(fact, "calculation_context", None)),
        "calculation_context": _safe_str(getattr(fact, "calculation_context", "")),
        "period_start": _date_iso(getattr(fact, "period_start", None)),
        "period_end": _date_iso(getattr(fact, "period_end", None)),
        "period_type": _safe_str(getattr(fact, "period_type", "")),
        "fiscal_year": _safe_number(getattr(fact, "fiscal_year", None)),
        "fiscal_period": _safe_str(getattr(fact, "fiscal_period", "")),
        "filing_date": _date_iso(getattr(fact, "filing_date", None)),
        "form_type": _safe_str(getattr(fact, "form_type", "")),
        "accession": _safe_str(getattr(fact, "accession", "")),
        "data_quality": _enum_value(getattr(fact, "data_quality", "")),
        "is_audited": bool(getattr(fact, "is_audited", False)),
        "is_restated": bool(getattr(fact, "is_restated", False)),
        "is_estimated": bool(getattr(fact, "is_estimated", False)),
        "confidence_score": _safe_number(getattr(fact, "confidence_score", None)),
        "statement_type": _safe_str(getattr(fact, "statement_type", "")),
        "section": _safe_str(getattr(fact, "section", "")),
    }
    if is_dataclass(fact):
        available_fields = {field.name for field in fields(fact)}
        data["edgartools_fields"] = sorted(available_fields)
    return data


def _ratio_provenance(value: float | None, *, numerator: str, denominator: str) -> dict[str, Any]:
    return {
        "source": "computed_from_sec_companyfacts",
        "value": value,
        "derived": True,
        "formula": f"{numerator} / {denominator}",
        "components": [numerator, denominator],
    }


def _metric_value(value: Any) -> float | None:
    if value is None:
        return None
    value = getattr(value, "value", value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _latest_periodic_date(filings: dict[str, dict[str, Any] | None]) -> date | None:
    dates = [
        _filing_date(filings.get("latest_10q")),
        _filing_date(filings.get("latest_10k")),
        _filing_date(filings.get("latest_20f")),
    ]
    dates = [item for item in dates if item is not None]
    return max(dates) if dates else None


def _filing_date(filing: dict[str, Any] | None) -> date | None:
    if not filing:
        return None
    return _parse_date(filing.get("filing_date"))


def _days_since(value: date | None) -> int | None:
    if value is None:
        return None
    return (datetime.now(timezone.utc).date() - value).days


def _event_risk_level(stale: bool, latest_8k_age: int | None, recent_8k_days: int) -> str:
    if latest_8k_age is not None and latest_8k_age <= recent_8k_days:
        return "medium"
    if stale:
        return "unknown"
    return "low"


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 6)


def _is_us_symbol(symbol: str) -> bool:
    return symbol.upper().startswith("US.") or "." not in symbol


def _ticker_from_symbol(symbol: str) -> str:
    value = symbol.upper().strip()
    if "." in value:
        return value.split(".", 1)[1]
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _date_iso(value: Any) -> str | None:
    parsed = _parse_date(value)
    return parsed.isoformat() if parsed else None


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_number(value: Any) -> int | float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _enum_value(value: Any) -> str:
    if hasattr(value, "value"):
        return _safe_str(value.value)
    return _safe_str(value)


def _provider_warning(reason: str) -> str:
    if reason == "identity_not_configured":
        return "SEC_EDGAR_IDENTITY or EDGAR_IDENTITY is required before fetching SEC filings."
    if reason == "disabled":
        return "SEC filings provider disabled by IA_SEC_ENABLED=0."
    if reason == "no_us_equity_symbols":
        return "SEC filings provider only supports US equity symbols in V1."
    return reason
