"""Financial Modeling Prep supplemental data provider.

FMP is used only for normalized supplemental artifacts. It does not replace
Futu real-time market data or SEC / edgartools filing-derived numbers.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .storage import utc_now


FMP_PROVIDER = "financialmodelingprep"
DEFAULT_BASE_URL = "https://financialmodelingprep.com/stable"


class FmpProviderError(RuntimeError):
    """Raised for unexpected FMP provider failures."""


@dataclass(frozen=True)
class FmpProviderConfig:
    api_key: str | None
    base_url: str
    timeout_seconds: float
    retries: int
    retry_backoff_seconds: float
    rate_limit_delay_seconds: float
    enabled: bool
    limit: int

    @classmethod
    def from_env(cls) -> "FmpProviderConfig":
        _load_hermes_env_keys("FMP_API_KEY")
        settings = _investment_assistant_fmp_settings(_read_hermes_config())
        return cls(
            api_key=os.getenv("FMP_API_KEY", "").strip() or None,
            base_url=_setting_str(settings, "base_url", "IA_FMP_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
            or DEFAULT_BASE_URL,
            timeout_seconds=max(1.0, _setting_float(settings, "timeout_seconds", "IA_FMP_TIMEOUT_SECONDS", 30.0)),
            retries=max(1, _setting_int(settings, "retries", "IA_FMP_RETRIES", 3)),
            retry_backoff_seconds=max(
                0.0,
                _setting_float(settings, "retry_backoff_seconds", "IA_FMP_RETRY_BACKOFF_SECONDS", 0.5),
            ),
            rate_limit_delay_seconds=max(
                0.0,
                _setting_float(settings, "rate_limit_delay_seconds", "IA_FMP_RATE_LIMIT_DELAY_SECONDS", 0.25),
            ),
            enabled=_setting_bool(settings, "enabled", "IA_FMP_ENABLED", True),
            limit=max(1, _setting_int(settings, "limit", "IA_FMP_LIMIT", 25)),
        )


class FmpProvider:
    """Fetch supplemental FMP artifacts for the investment assistant miner."""

    def __init__(
        self,
        config: FmpProviderConfig | None = None,
        *,
        request_json: Callable[[str, dict[str, Any]], Any] | None = None,
    ):
        self.config = config or FmpProviderConfig.from_env()
        self._request_json_override = request_json
        self._last_request_at = 0.0

    def company_profile(self, symbol: str) -> dict[str, Any]:
        return self._artifact(
            artifact_type="fmp_company_profile",
            symbol=symbol,
            endpoint_specs=[("profile", "profile", {"symbol": _ticker(symbol)})],
            transform=lambda raw: {"profile": _first_or_raw(raw.get("profile"))},
        )

    def etf_exposure(self, symbol: str) -> dict[str, Any]:
        endpoint_specs = [
            ("etf_info", "etf/info", {"symbol": _ticker(symbol)}),
            ("holdings", "etf/holdings", {"symbol": _ticker(symbol)}),
            ("sector_weightings", "etf/sector-weightings", {"symbol": _ticker(symbol)}),
            ("country_weightings", "etf/country-weightings", {"symbol": _ticker(symbol)}),
        ]
        return self._artifact(
            artifact_type="fmp_etf_exposure",
            symbol=symbol,
            endpoint_specs=endpoint_specs,
            transform=_transform_etf_exposure,
        )

    def analyst_expectations(self, symbol: str) -> dict[str, Any]:
        ticker = _ticker(symbol)
        endpoint_specs = [
            ("annual_estimates", "analyst-estimates", {"symbol": ticker, "period": "annual", "page": 0, "limit": 10}),
            (
                "quarterly_estimates",
                "analyst-estimates",
                {"symbol": ticker, "period": "quarter", "page": 0, "limit": 10},
            ),
            ("price_target_consensus", "price-target-consensus", {"symbol": ticker}),
            ("price_target_summary", "price-target-summary", {"symbol": ticker}),
            ("ratings_snapshot", "ratings-snapshot", {"symbol": ticker}),
            ("grades_summary", "grades-summary", {"symbol": ticker}),
        ]
        return self._artifact(
            artifact_type="fmp_analyst_expectations",
            symbol=symbol,
            endpoint_specs=endpoint_specs,
            transform=lambda raw: {
                "annual_estimates": raw.get("annual_estimates", []),
                "quarterly_estimates": raw.get("quarterly_estimates", []),
                "price_target_consensus": _first_or_raw(raw.get("price_target_consensus")),
                "price_target_summary": raw.get("price_target_summary", []),
                "ratings_snapshot": _first_or_raw(raw.get("ratings_snapshot")),
                "grades_summary": _first_or_raw(raw.get("grades_summary")),
            },
        )

    def earnings_transcripts(self, symbol: str) -> dict[str, Any]:
        ticker = _ticker(symbol)
        date_result = self._fetch_endpoint("transcript_dates", "earning-call-transcript-dates", {"symbol": ticker})
        raw = {"transcript_dates": date_result.get("raw", [])}
        endpoint_results = [date_result]
        latest = _latest_transcript_date(raw["transcript_dates"])
        if latest:
            transcript_result = self._fetch_endpoint(
                "latest_transcript",
                "earning-call-transcript",
                {
                    "symbol": ticker,
                    "year": latest.get("year"),
                    "quarter": latest.get("quarter"),
                },
            )
            raw["latest_transcript"] = transcript_result.get("raw", [])
            endpoint_results.append(transcript_result)
        payload = self._artifact_from_endpoint_results(
            artifact_type="fmp_earnings_transcripts",
            symbol=symbol,
            endpoint_results=endpoint_results,
            raw=raw,
            structured={
                "transcript_dates": raw.get("transcript_dates", []),
                "latest_transcript_metadata": latest or {},
                "latest_transcript": _first_or_raw(raw.get("latest_transcript")),
            },
        )
        return payload

    def normalized_metrics(self, symbol: str) -> dict[str, Any]:
        ticker = _ticker(symbol)
        endpoint_specs = [
            ("key_metrics_ttm", "key-metrics-ttm", {"symbol": ticker}),
            ("ratios_ttm", "ratios-ttm", {"symbol": ticker}),
            ("financial_scores", "financial-scores", {"symbol": ticker}),
        ]
        return self._artifact(
            artifact_type="fmp_normalized_metrics",
            symbol=symbol,
            endpoint_specs=endpoint_specs,
            transform=lambda raw: {
                "key_metrics_ttm": _first_or_raw(raw.get("key_metrics_ttm")),
                "ratios_ttm": _first_or_raw(raw.get("ratios_ttm")),
                "financial_scores": _first_or_raw(raw.get("financial_scores")),
                "numeric_source_note": "Supplemental normalized FMP metrics; SEC companyfacts remain authoritative.",
            },
        )

    def peer_group(self, symbol: str) -> dict[str, Any]:
        return self._artifact(
            artifact_type="fmp_peer_group",
            symbol=symbol,
            endpoint_specs=[("stock_peers", "stock-peers", {"symbol": _ticker(symbol)})],
            transform=lambda raw: {"peers": _extract_peer_symbols(raw.get("stock_peers"))},
        )

    def ownership_signal(self, symbol: str) -> dict[str, Any]:
        ticker = _ticker(symbol)
        endpoint_specs = [
            (
                "symbol_positions_summary",
                "institutional-ownership/symbol-positions-summary",
                {"symbol": ticker},
            ),
            (
                "holder_analytics",
                "institutional-ownership/extract-analytics/holder",
                {"symbol": ticker, "page": 0, "limit": self.config.limit},
            ),
        ]
        return self._artifact(
            artifact_type="fmp_ownership_signal",
            symbol=symbol,
            endpoint_specs=endpoint_specs,
            transform=lambda raw: {
                "positions_summary": raw.get("symbol_positions_summary", []),
                "holder_analytics": raw.get("holder_analytics", []),
            },
        )

    def insider_signal(self, symbol: str) -> dict[str, Any]:
        ticker = _ticker(symbol)
        endpoint_specs = [
            ("insider_statistics", "insider-trading/statistics", {"symbol": ticker}),
            (
                "insider_transactions",
                "insider-trading/search",
                {"symbol": ticker, "page": 0, "limit": self.config.limit},
            ),
        ]
        return self._artifact(
            artifact_type="fmp_insider_signal",
            symbol=symbol,
            endpoint_specs=endpoint_specs,
            transform=lambda raw: {
                "statistics": raw.get("insider_statistics", []),
                "recent_transactions": raw.get("insider_transactions", []),
                "deterministic_summary": _summarize_insider_transactions(raw.get("insider_transactions", [])),
            },
        )

    def _artifact(
        self,
        *,
        artifact_type: str,
        symbol: str,
        endpoint_specs: list[tuple[str, str, dict[str, Any]]],
        transform: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        endpoint_results = [self._fetch_endpoint(name, path, params) for name, path, params in endpoint_specs]
        raw = {result["name"]: result.get("raw") for result in endpoint_results}
        return self._artifact_from_endpoint_results(
            artifact_type=artifact_type,
            symbol=symbol,
            endpoint_results=endpoint_results,
            raw=raw,
            structured=transform(raw),
        )

    def _artifact_from_endpoint_results(
        self,
        *,
        artifact_type: str,
        symbol: str,
        endpoint_results: list[dict[str, Any]],
        raw: dict[str, Any],
        structured: dict[str, Any],
    ) -> dict[str, Any]:
        statuses = [str(result.get("source_status") or "missing") for result in endpoint_results]
        warnings = [
            warning
            for result in endpoint_results
            for warning in result.get("warnings", [])
            if warning
        ]
        return {
            "artifact_type": artifact_type,
            "symbol": _normalize_symbol(symbol),
            "provider": FMP_PROVIDER,
            "generated_at": utc_now(),
            "source_status": _combined_status(statuses),
            "endpoints": [_endpoint_metadata(result) for result in endpoint_results],
            "data_asof": _data_asof(endpoint_results),
            "warnings": warnings,
            "raw": raw,
            **structured,
        }

    def _fetch_endpoint(self, name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enabled:
            return _skipped_endpoint(name, path, "FMP provider disabled by investment_assistant.fmp.enabled=false.")
        if not self.config.api_key:
            return _skipped_endpoint(name, path, "FMP_API_KEY is required for FMP supplemental data.")
        try:
            raw, url, attempts = self._request_json(path, params)
        except urllib.error.HTTPError as exc:
            status = "rate_limited" if exc.code == 429 else "unavailable"
            return _failed_endpoint(name, path, params, status, f"FMP HTTP {exc.code}: {exc.reason}")
        except Exception as exc:
            return _failed_endpoint(name, path, params, "unavailable", str(exc))
        return {
            "name": name,
            "path": path,
            "params": _safe_params(params),
            "source_status": "fresh",
            "retrieved_at": utc_now(),
            "url": _redact_apikey(url),
            "attempts": attempts,
            "raw": raw,
            "warnings": [],
        }

    def _request_json(self, path: str, params: dict[str, Any]) -> tuple[Any, str, list[dict[str, Any]]]:
        query = {key: value for key, value in params.items() if value is not None}
        if self._request_json_override:
            raw = self._request_json_override(path, query)
            return raw, self._url(path, query), [{"attempt": 1, "status": "success", "override": True}]
        last_exc: Exception | None = None
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.config.retries + 1):
            url = self._url(path, query)
            try:
                self._wait_for_rate_limit()
                request = urllib.request.Request(url, headers={"User-Agent": "hermes-agent investment-assistant"})
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read()
                self._last_request_at = time.monotonic()
                attempts.append({"attempt": attempt, "status": "success", "byte_count": len(body)})
                return json.loads(body.decode("utf-8")), url, attempts
            except urllib.error.HTTPError as exc:
                self._last_request_at = time.monotonic()
                attempts.append({"attempt": attempt, "status": "failed", "http_status": exc.code})
                if exc.code == 429 and attempt < self.config.retries and self.config.retry_backoff_seconds:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                attempts.append({"attempt": attempt, "status": "failed", "error": str(exc)})
                if attempt < self.config.retries and self.config.retry_backoff_seconds:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
        assert last_exc is not None
        raise last_exc

    def _url(self, path: str, params: dict[str, Any]) -> str:
        query = dict(params)
        query["apikey"] = self.config.api_key
        return f"{self.config.base_url}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"

    def _wait_for_rate_limit(self) -> None:
        delay = self.config.rate_limit_delay_seconds
        if delay <= 0 or self._last_request_at <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _transform_etf_exposure(raw: dict[str, Any]) -> dict[str, Any]:
    holdings = raw.get("holdings", []) or []
    return {
        "profile": _first_or_raw(raw.get("etf_info")),
        "holdings": holdings,
        "sector_exposure": raw.get("sector_weightings", []),
        "country_exposure": raw.get("country_weightings", []),
        "concentration_summary": _concentration_summary(holdings),
        "overlap_helper": {
            "holding_symbols": [
                str(item.get("symbol") or item.get("asset") or "").upper()
                for item in holdings
                if isinstance(item, dict) and (item.get("symbol") or item.get("asset"))
            ],
        },
    }


def _concentration_summary(holdings: Any) -> dict[str, Any]:
    if not isinstance(holdings, list):
        return {"holding_count": 0, "top_10_weight": None, "top_5": []}
    weighted: list[tuple[float, dict[str, Any]]] = []
    for item in holdings:
        if not isinstance(item, dict):
            continue
        weight = _safe_float(
            item.get("weightPercentage")
            or item.get("weight")
            or item.get("percentage")
            or item.get("marketValuePercentage")
        )
        if weight is None:
            continue
        weighted.append((weight, item))
    weighted.sort(key=lambda pair: pair[0], reverse=True)
    return {
        "holding_count": len(holdings),
        "top_10_weight": round(sum(weight for weight, _item in weighted[:10]), 6) if weighted else None,
        "top_5": [
            {
                "symbol": item.get("symbol") or item.get("asset") or "",
                "name": item.get("name") or item.get("assetName") or "",
                "weight": weight,
            }
            for weight, item in weighted[:5]
        ],
    }


def _summarize_insider_transactions(transactions: Any) -> dict[str, Any]:
    if not isinstance(transactions, list):
        return {"transaction_count": 0}
    acquired = disposed = 0.0
    buy_count = sell_count = 0
    for item in transactions:
        if not isinstance(item, dict):
            continue
        transaction_type = str(item.get("transactionType") or item.get("typeOfOwner") or "").lower()
        securities = _safe_float(item.get("securitiesTransacted") or item.get("securitiesOwned") or 0) or 0.0
        if "sale" in transaction_type or "sell" in transaction_type or item.get("acquistionOrDisposition") == "D":
            sell_count += 1
            disposed += securities
        elif "purchase" in transaction_type or item.get("acquistionOrDisposition") == "A":
            buy_count += 1
            acquired += securities
    return {
        "transaction_count": len(transactions),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "shares_acquired": acquired,
        "shares_disposed": disposed,
    }


def _latest_transcript_date(items: Any) -> dict[str, Any] | None:
    if not isinstance(items, list) or not items:
        return None
    candidates = [item for item in items if isinstance(item, dict)]
    if not candidates:
        return None
    return candidates[0]


def _extract_peer_symbols(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        peers = raw.get("peersList") or raw.get("peers") or raw.get("symbols")
        return [str(item).upper() for item in peers or [] if item]
    if isinstance(raw, list):
        result: list[str] = []
        for item in raw:
            if isinstance(item, str):
                result.append(item.upper())
            elif isinstance(item, dict):
                result.extend(str(value).upper() for value in item.get("peersList") or item.get("peers") or [] if value)
                if item.get("symbol"):
                    result.append(str(item["symbol"]).upper())
        return _dedupe(result)
    return []


def _first_or_raw(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _endpoint_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": result.get("name"),
        "path": result.get("path"),
        "params": result.get("params", {}),
        "source_status": result.get("source_status"),
        "retrieved_at": result.get("retrieved_at", ""),
        "url": result.get("url", ""),
        "warnings": result.get("warnings", []),
        "error": result.get("error", ""),
        "attempts": result.get("attempts", []),
    }


def _data_asof(endpoint_results: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(result.get("name")): str(result.get("retrieved_at") or "")
        for result in endpoint_results
        if result.get("retrieved_at")
    }


def _combined_status(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    if all(status == "fresh" for status in statuses):
        return "fresh"
    if any(status == "fresh" for status in statuses):
        return "partial"
    if any(status == "rate_limited" for status in statuses):
        return "partial"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    return "unavailable"


def _skipped_endpoint(name: str, path: str, warning: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "params": {},
        "source_status": "skipped",
        "retrieved_at": utc_now(),
        "url": "",
        "raw": [],
        "warnings": [warning],
    }


def _failed_endpoint(name: str, path: str, params: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "params": _safe_params(params),
        "source_status": status,
        "retrieved_at": utc_now(),
        "url": "",
        "raw": [],
        "warnings": [error],
        "error": error,
    }


def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key != "apikey"}


def _redact_apikey(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted = [(key, "***" if key == "apikey" else value) for key, value in pairs]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(redacted)))


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ticker(symbol: str) -> str:
    normalized = _normalize_symbol(symbol)
    return normalized.split(".", 1)[1] if "." in normalized else normalized


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if "." not in value:
        return f"US.{value}"
    return value


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


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


def _read_hermes_config() -> dict[str, Any]:
    try:
        from hermes_constants import get_hermes_config_path

        path = get_hermes_config_path()
    except Exception:
        path = Path.home() / ".hermes" / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _investment_assistant_fmp_settings(config: dict[str, Any]) -> dict[str, Any]:
    plugin_config = config.get("investment_assistant", {}) if isinstance(config, dict) else {}
    if not isinstance(plugin_config, dict):
        return {}
    settings = plugin_config.get("fmp", {})
    return settings if isinstance(settings, dict) else {}


def _setting_str(settings: dict[str, Any], key: str, env_name: str, default: str) -> str:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.strip() or default
    value = settings.get(key, default)
    return str(value or default).strip() or default


def _setting_int(settings: dict[str, Any], key: str, env_name: str, default: int) -> int:
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else settings.get(key, default)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _setting_float(settings: dict[str, Any], key: str, env_name: str, default: float) -> float:
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else settings.get(key, default)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _setting_bool(settings: dict[str, Any], key: str, env_name: str, default: bool) -> bool:
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else settings.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
