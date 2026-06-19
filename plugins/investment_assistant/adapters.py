"""Data-provider adapters for investment assistant workflows."""

from __future__ import annotations

import math
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import pstdev
from typing import Any

from .schemas import (
    Candidate,
    CandidateDataQuality,
    CurrentHolding,
    CurrentPortfolio,
    DiscoveryData,
    FutuData,
    ThemeCoverageRequirement,
    ThemeDiscoveryPlan,
    ResearchSource,
)
from .storage import utc_now
from .theme_discovery import build_theme_discovery_plan, normalize_futu_symbol


class FutuAdapterError(RuntimeError):
    """Raised when Futu OpenD or futu-api cannot provide required data."""


SUPPORTED_THEME_KEYS = ("ai", "storage", "semiconductor", "power")
_THEME_ALIASES = {
    "ai": {"ai", "人工智能", "artificial intelligence", "artificial_intelligence"},
    "storage": {"storage", "memory", "存储"},
    "semiconductor": {"semiconductor", "semi", "chip", "半导体"},
    "power": {"power", "electricity", "utility", "电力"},
}


def canonical_theme_key(value: str) -> str:
    """Normalize an explicit theme key.

    This intentionally rejects free-form descriptions. The agent must pass a
    small structured key such as ``ai`` and put longer prose in
    ``theme_description``.
    """
    normalized = " ".join(str(value or "").strip().lower().replace("-", " ").split())
    for canonical, aliases in _THEME_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    raise FutuAdapterError(
        "Unknown theme_key. Use one of: "
        + ", ".join(SUPPORTED_THEME_KEYS)
        + ". Do not pass free-form prose as theme_key; use theme_description for that."
    )


def normalize_us_symbol(symbol: str) -> str:
    return normalize_futu_symbol(symbol, "US")


def normalize_market_symbol(symbol: str, market: str) -> str:
    return normalize_futu_symbol(symbol, market)


def _quote_unavailable_candidate(
    *,
    code: str,
    discovery: DiscoveryData,
    reason: str,
    source_tags: set[str],
    plate_memberships: list[dict[str, str]],
) -> Candidate:
    """Preserve a discovered candidate even when Futu cannot quote it."""

    futu_data = FutuData.from_parts(
        last_price=None,
        quote={},
        technical={},
        liquidity={},
        options={"has_option_data": False, "status": "quote_unavailable"},
        data_asof={},
        score_breakdown={},
    )
    futu_data.warnings.append(reason)
    evidence = []
    if discovery.rationale:
        evidence.append(f"Discovery rationale: {discovery.rationale}")
    evidence.append(f"Futu quote unavailable: {reason}")
    return Candidate(
        symbol=code,
        name=code,
        theme_role=discovery.role,
        source="futu_opend",
        source_tags=sorted(set(source_tags) | {"futu_quote_unavailable"}),
        score=0,
        candidate_status="quote_unavailable",
        eligible_for_portfolio=False,
        exclusion_reasons=[reason],
        data_quality=CandidateDataQuality(
            freshness="unavailable",
            missing_fields=["futu_market_snapshot"],
            warnings=[reason],
        ),
        discovery_data=discovery,
        futu_data=futu_data,
        plate_memberships=plate_memberships,
        evidence=evidence,
        risk_tags=["quote_unavailable"],
    )


@dataclass(frozen=True)
class FutuOpenDConfig:
    host: str
    port: int
    trd_env: str
    market: str
    acc_id: int | None
    currency: str | None
    kline_count: int
    max_candidates: int | None
    quote_rate_limit_calls: int
    quote_rate_limit_window: float
    quote_rate_limit_retries: int
    screener_rate_limit_calls: int
    screener_rate_limit_window: float
    screener_rate_limit_retries: int
    plate_stock_limit: int
    max_theme_plates: int
    fetch_options: bool
    option_dte_min: int
    option_dte_max: int
    option_snapshot_limit: int

    @classmethod
    def from_env(cls) -> "FutuOpenDConfig":
        acc_id_raw = os.getenv("FUTU_ACC_ID", "").strip()
        max_candidates_raw = os.getenv("IA_FUTU_MAX_CANDIDATES", "").strip()
        return cls(
            host=os.getenv("FUTU_OPEND_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=int(os.getenv("FUTU_OPEND_PORT", "11111")),
            trd_env=os.getenv("FUTU_TRD_ENV", "SIMULATE").strip().upper() or "SIMULATE",
            market=os.getenv("FUTU_DEFAULT_MARKET", "US").strip().upper() or "US",
            acc_id=int(acc_id_raw) if acc_id_raw else None,
            currency=os.getenv("FUTU_CURRENCY", "").strip().upper() or None,
            kline_count=max(30, int(os.getenv("IA_FUTU_KLINE_COUNT", "120"))),
            max_candidates=_optional_positive_int(max_candidates_raw, minimum=5),
            quote_rate_limit_calls=max(0, int(os.getenv("IA_FUTU_QUOTE_RATE_LIMIT_CALLS", "55"))),
            quote_rate_limit_window=max(0.0, float(os.getenv("IA_FUTU_QUOTE_RATE_LIMIT_WINDOW", "30"))),
            quote_rate_limit_retries=max(0, int(os.getenv("IA_FUTU_QUOTE_RATE_LIMIT_RETRIES", "1"))),
            screener_rate_limit_calls=max(0, int(os.getenv("IA_FUTU_SCREENER_RATE_LIMIT_CALLS", "8"))),
            screener_rate_limit_window=max(0.0, float(os.getenv("IA_FUTU_SCREENER_RATE_LIMIT_WINDOW", "30"))),
            screener_rate_limit_retries=max(0, int(os.getenv("IA_FUTU_SCREENER_RATE_LIMIT_RETRIES", "2"))),
            plate_stock_limit=max(5, int(os.getenv("IA_FUTU_PLATE_STOCK_LIMIT", "30"))),
            max_theme_plates=max(1, int(os.getenv("IA_FUTU_MAX_THEME_PLATES", "4"))),
            fetch_options=os.getenv("IA_FUTU_FETCH_OPTIONS", "1").strip() != "0",
            option_dte_min=max(1, int(os.getenv("IA_FUTU_OPTION_DTE_MIN", "14"))),
            option_dte_max=max(1, int(os.getenv("IA_FUTU_OPTION_DTE_MAX", "45"))),
            option_snapshot_limit=max(10, int(os.getenv("IA_FUTU_OPTION_SNAPSHOT_LIMIT", "60"))),
        )


class _FutuQuoteRateLimiter:
    """Process-local sliding-window limiter for Futu quote API calls."""

    def __init__(self, max_calls: int, window_seconds: float, *, serialize_calls: bool = False):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.serialize_calls = serialize_calls
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()
        self._call_lock = threading.Lock()

    def acquire(self) -> None:
        if self.max_calls <= 0 or self.window_seconds <= 0:
            return

        while True:
            sleep_for = 0.0
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return
                sleep_for = max(0.0, self._calls[0] + self.window_seconds - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def run(self, func: Callable[..., Any], *args, **kwargs):
        if not self.serialize_calls:
            self.acquire()
            return func(*args, **kwargs)
        with self._call_lock:
            self.acquire()
            return func(*args, **kwargs)


_QUOTE_RATE_LIMITERS: dict[tuple[str, int, int, float], _FutuQuoteRateLimiter] = {}
_QUOTE_RATE_LIMITERS_LOCK = threading.Lock()
_SCREENER_RATE_LIMITERS: dict[tuple[str, int, int, float], _FutuQuoteRateLimiter] = {}
_SCREENER_RATE_LIMITERS_LOCK = threading.Lock()
_FUTU_SCREENER_APIS = {
    "get_stock_filter",
    "get_plate_list",
    "get_plate_stock",
    "get_owner_plate",
}


def _quote_rate_limiter_for(config: FutuOpenDConfig) -> _FutuQuoteRateLimiter:
    key = (
        config.host,
        config.port,
        config.quote_rate_limit_calls,
        config.quote_rate_limit_window,
    )
    with _QUOTE_RATE_LIMITERS_LOCK:
        limiter = _QUOTE_RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _FutuQuoteRateLimiter(config.quote_rate_limit_calls, config.quote_rate_limit_window)
            _QUOTE_RATE_LIMITERS[key] = limiter
        return limiter


def _screener_rate_limiter_for(config: FutuOpenDConfig) -> _FutuQuoteRateLimiter:
    key = (
        config.host,
        config.port,
        config.screener_rate_limit_calls,
        config.screener_rate_limit_window,
    )
    with _SCREENER_RATE_LIMITERS_LOCK:
        limiter = _SCREENER_RATE_LIMITERS.get(key)
        if limiter is None:
            limiter = _FutuQuoteRateLimiter(
                config.screener_rate_limit_calls,
                config.screener_rate_limit_window,
                serialize_calls=True,
            )
            _SCREENER_RATE_LIMITERS[key] = limiter
        return limiter


@dataclass(frozen=True)
class ThemeUniverse:
    canonical_theme: str
    source_tags: list[str]
    candidates: list[Candidate]
    warnings: list[str]
    discovery_thesis: str = ""
    coverage_requirements: list[ThemeCoverageRequirement] = field(default_factory=list)
    research_trace: list[ResearchSource] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    data_asof: dict[str, str] = field(default_factory=dict)


class MarketDataAdapter:
    """Futu-backed market/candidate data adapter.

    V1 intentionally does not read current holdings here. Candidate generation
    must stay independent from the user's portfolio to avoid anchoring bias.
    """

    _THEME_BENCHMARKS = {
        "storage": ["US.SMH", "US.SOXX"],
        "ai": ["US.QQQ", "US.SMH"],
        "semiconductor": ["US.SMH", "US.SOXX"],
        "power": ["US.XLU"],
    }
    _REGIME_PROXIES = [
        "US.SPY",
        "US.QQQ",
        "US.IWM",
        "US.TLT",
        "US.IEF",
        "US.SHY",
        "US.UUP",
        "US.GLD",
        "US.VIXY",
        "US.VXX",
    ]

    def __init__(self, config: FutuOpenDConfig | None = None):
        self.config = config or FutuOpenDConfig.from_env()
        self._quote_rate_limiter = _quote_rate_limiter_for(self.config)
        self._screener_rate_limiter = _screener_rate_limiter_for(self.config)

    def _quote_call(self, func: Callable[..., Any], *args, **kwargs):
        limiter, retries, retry_window = self._quote_policy_for(func)
        attempts = 0
        while True:
            result = _run_with_rate_limiter(limiter, func, *args, **kwargs)
            if not _quote_result_is_rate_limited(result):
                return result
            attempts += 1
            if attempts > retries:
                return result
            time.sleep(max(1.0, retry_window))

    def _quote_policy_for(self, func: Callable[..., Any]) -> tuple[_FutuQuoteRateLimiter, int, float]:
        api_name = _futu_api_name(func)
        if api_name in _FUTU_SCREENER_APIS:
            return (
                self._screener_rate_limiter,
                self.config.screener_rate_limit_retries,
                self.config.screener_rate_limit_window,
            )
        return (
            self._quote_rate_limiter,
            self.config.quote_rate_limit_retries,
            self.config.quote_rate_limit_window,
        )

    def get_theme_universe(
        self,
        theme: str,
        required_symbols: list[str] | None = None,
        theme_description: str = "",
    ) -> ThemeUniverse:
        canonical = self._canonical_theme(theme)
        discovery_plan = build_theme_discovery_plan(
            canonical,
            market=self.config.market,
            theme_description=theme_description,
            required_symbols=required_symbols or [],
        )
        seeds = _seeds_from_discovery_plan(discovery_plan, required_symbols or [], self.config.market)
        candidates, source_tags, warnings = self._load_futu_candidates(
            canonical,
            seeds,
            discovery_plan.plate_keywords,
        )
        if not candidates:
            raise FutuAdapterError(
                "Futu returned no usable candidate data. Check quote permissions and symbol coverage."
            )
        warnings = [*discovery_plan.warnings, *warnings]
        warnings.append("News evidence is not wired into this V1 adapter yet.")
        return ThemeUniverse(
            canonical_theme=canonical,
            source_tags=source_tags,
            candidates=candidates,
            warnings=warnings,
            discovery_thesis=discovery_plan.initial_thesis,
            coverage_requirements=discovery_plan.coverage_requirements,
            research_trace=discovery_plan.research_trace,
            search_queries=discovery_plan.search_queries,
            data_asof=discovery_plan.data_asof,
        )

    def get_market_regime(self, theme: str) -> dict[str, Any]:
        canonical = self._canonical_theme(theme)
        self._check_opend()
        futu = _import_futu()
        theme_benchmarks = self._THEME_BENCHMARKS.get(canonical, [])
        codes = _dedupe(self._REGIME_PROXIES + theme_benchmarks)
        quote_ctx = futu.OpenQuoteContext(host=self.config.host, port=self.config.port)
        warnings: list[str] = []
        try:
            ret, snapshot = self._quote_call(quote_ctx.get_market_snapshot, codes)
            _check_ret(futu, ret, snapshot, "get_market_snapshot(regime proxies)")
            snapshot_by_code = {
                _safe_str(_row_get(row, "code")): row
                for _, row in _iter_rows(snapshot)
                if _safe_str(_row_get(row, "code"))
            }
            metrics = {}
            for code in codes:
                row = snapshot_by_code.get(code)
                if row is None:
                    warnings.append(f"Missing regime proxy snapshot: {code}")
                    continue
                try:
                    kline_rows = self._get_daily_kline(quote_ctx, futu, code)
                except FutuAdapterError as exc:
                    warnings.append(str(exc))
                    kline_rows = []
                metrics[code] = _benchmark_metrics(row, kline_rows)
            return _build_macro_context(canonical, theme_benchmarks, metrics, warnings)
        finally:
            quote_ctx.close()

    def _canonical_theme(self, theme: str) -> str:
        return canonical_theme_key(theme)

    def _load_futu_candidates(
        self,
        canonical_theme: str,
        seeds: list[tuple[str, DiscoveryData]],
        plate_keywords: list[str],
    ) -> tuple[list[Candidate], list[str], list[str]]:
        self._check_opend()
        futu = _import_futu()
        quote_ctx = futu.OpenQuoteContext(host=self.config.host, port=self.config.port)
        try:
            resolved = self._resolve_theme_symbols(quote_ctx, futu, canonical_theme, seeds, plate_keywords)
            codes = resolved["codes"]
            discovery_by_code = resolved["discovery_by_code"]
            source_tags_by_code = resolved["source_tags_by_code"]
            memberships_by_code = resolved["memberships_by_code"]
            source_tags = resolved["source_tags"]
            warnings = resolved["warnings"]

            snapshot_by_code, quote_errors = self._get_market_snapshot_result(quote_ctx, futu, codes, warnings)

            candidates: list[Candidate] = []
            for code in codes:
                discovery = discovery_by_code.get(
                    code,
                    DiscoveryData(role="theme exposure"),
                )
                plate_memberships = memberships_by_code.get(code, [])
                row = snapshot_by_code.get(code)
                if row is None:
                    candidates.append(
                        _quote_unavailable_candidate(
                            code=code,
                            discovery=discovery,
                            reason=quote_errors.get(code) or "Futu snapshot missing for candidate.",
                            source_tags=source_tags_by_code.get(code, set()),
                            plate_memberships=plate_memberships,
                        )
                    )
                    continue
                last_price = _safe_float(_row_get(row, "last_price"))
                if last_price <= 0:
                    candidates.append(
                        _quote_unavailable_candidate(
                            code=code,
                            discovery=discovery,
                            reason="Futu snapshot returned last_price <= 0.",
                            source_tags=source_tags_by_code.get(code, set()),
                            plate_memberships=plate_memberships,
                        )
                    )
                    continue
                kline_warning = None
                try:
                    kline_rows = self._get_daily_kline(quote_ctx, futu, code)
                except FutuAdapterError as exc:
                    kline_warning = str(exc)
                    warnings.append(kline_warning)
                    kline_rows = []
                rs, volatility = _market_factors(kline_rows)
                turnover = _safe_float(_row_get(row, "turnover"))
                volume = _safe_float(_row_get(row, "volume"))
                change_rate = _safe_float(_row_get(row, "change_rate"))
                bid_price = _safe_float(_row_get(row, "bid_price"))
                ask_price = _safe_float(_row_get(row, "ask_price"))
                score_breakdown = _candidate_score_breakdown(rs, volatility, turnover, change_rate)
                score = score_breakdown["total"]
                name = _safe_str(_row_get(row, "name")) or code
                update_time = _safe_str(_row_get(row, "update_time"))
                spread_bps = _spread_bps(bid_price, ask_price)
                evidence = [
                    f"Futu snapshot last_price={last_price}, turnover={turnover:.0f}.",
                    f"Futu daily kline rows={len(kline_rows)}, relative_strength={rs:.1f}, volatility={volatility:.4f}.",
                ]
                if discovery.rationale:
                    evidence.append(f"Discovery rationale: {discovery.rationale}")
                if plate_memberships:
                    evidence.append(
                        "Futu plate memberships="
                        + ", ".join(item.get("plate_name", item.get("plate_code", "")) for item in plate_memberships[:3])
                        + "."
                    )
                if update_time:
                    evidence.append(f"Futu snapshot update_time={update_time}.")
                options_surface, option_warning = (
                    self._get_options_surface(quote_ctx, futu, code, last_price)
                    if self.config.fetch_options
                    else ({"has_option_data": False, "status": "disabled"}, None)
                )
                if option_warning:
                    warnings.append(option_warning)
                quote_payload = {
                    "update_time": update_time,
                    "open_price": _safe_float(_row_get(row, "open_price")),
                    "high_price": _safe_float(_row_get(row, "high_price")),
                    "low_price": _safe_float(_row_get(row, "low_price")),
                    "prev_close_price": _safe_float(_row_get(row, "prev_close_price")),
                    "turnover": turnover,
                    "volume": volume,
                    "change_rate": change_rate,
                    "turnover_rate": _safe_float(_row_get(row, "turnover_rate")),
                    "amplitude": _safe_float(_row_get(row, "amplitude")),
                    "volume_ratio": _safe_float(_row_get(row, "volume_ratio")),
                    "bid_ask_ratio": _safe_float(_row_get(row, "bid_ask_ratio")),
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "total_market_val": _safe_float(_row_get(row, "total_market_val")),
                    "circular_market_val": _safe_float(_row_get(row, "circular_market_val")),
                    "pe_ratio": _safe_float(_row_get(row, "pe_ratio")),
                    "pe_ttm_ratio": _safe_float(_row_get(row, "pe_ttm_ratio")),
                    "pb_ratio": _safe_float(_row_get(row, "pb_ratio")),
                    "ey_ratio": _safe_float(_row_get(row, "ey_ratio")),
                    "net_asset": _safe_float(_row_get(row, "net_asset")),
                    "net_profit": _safe_float(_row_get(row, "net_profit")),
                    "earning_per_share": _safe_float(_row_get(row, "earning_per_share")),
                    "net_asset_per_share": _safe_float(_row_get(row, "net_asset_per_share")),
                    "outstanding_shares": _safe_float(_row_get(row, "outstanding_shares")),
                    "highest52weeks_price": _safe_float(_row_get(row, "highest52weeks_price")),
                    "lowest52weeks_price": _safe_float(_row_get(row, "lowest52weeks_price")),
                    "dividend_ttm": _safe_float(_row_get(row, "dividend_ttm")),
                    "dividend_ratio_ttm": _safe_float(_row_get(row, "dividend_ratio_ttm")),
                    "enable_margin": _safe_str(_row_get(row, "enable_margin")),
                    "enable_short_sell": _safe_str(_row_get(row, "enable_short_sell")),
                    "short_available_volume": _safe_float(_row_get(row, "short_available_volume")),
                    "short_sell_rate": _safe_float(_row_get(row, "short_sell_rate")),
                }
                technical_payload = {
                    "trend": _trend_label(rs),
                    "relative_strength_60d": rs,
                    "realized_volatility": volatility,
                    "return_20d": _return_over(kline_rows, 20),
                    "return_60d": _return_over(kline_rows, 60),
                    "daily_returns_60d": _daily_returns(kline_rows, 60),
                    "daily_kline_rows": len(kline_rows),
                }
                liquidity_payload = {
                    "turnover": turnover,
                    "volume": volume,
                    "volume_ratio": _safe_float(_row_get(row, "volume_ratio")),
                    "turnover_rate": _safe_float(_row_get(row, "turnover_rate")),
                    "bid_ask_ratio": _safe_float(_row_get(row, "bid_ask_ratio")),
                    "spread_bps": spread_bps,
                    "liquidity_score": score_breakdown["liquidity"],
                }
                data_asof = {
                    "quote": update_time,
                    "kline": utc_now(),
                    "options": str(options_surface.get("data_asof", "")),
                }
                missing_fields = []
                if not kline_rows:
                    missing_fields.append("futu_history_kline")
                if not update_time:
                    missing_fields.append("quote_update_time")
                data_warnings = [kline_warning] if kline_warning else []
                candidates.append(
                    Candidate(
                        symbol=code,
                        name=name,
                        theme_role=discovery.role,
                        source="futu_opend",
                        source_tags=sorted(source_tags_by_code.get(code, set())),
                        score=score,
                        candidate_status="futu_enriched",
                        eligible_for_portfolio=True,
                        data_quality=CandidateDataQuality(
                            freshness="partial" if missing_fields else "fresh",
                            quote_asof=update_time or None,
                            kline_asof=data_asof["kline"],
                            options_asof=data_asof["options"] or None,
                            missing_fields=missing_fields,
                            warnings=data_warnings,
                        ),
                        discovery_data=discovery,
                        futu_data=FutuData.from_parts(
                            last_price=last_price,
                            quote=quote_payload,
                            technical=technical_payload,
                            liquidity=liquidity_payload,
                            options=options_surface,
                            data_asof=data_asof,
                            score_breakdown=score_breakdown,
                        ),
                        plate_memberships=plate_memberships,
                        evidence=evidence,
                        risk_tags=[],
                    )
                )
            return candidates, source_tags, warnings
        finally:
            quote_ctx.close()

    def _get_market_snapshot_by_code(
        self,
        quote_ctx,
        futu,
        codes: list[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        rows, _errors = self._get_market_snapshot_result(quote_ctx, futu, codes, warnings)
        return rows

    def _get_market_snapshot_result(
        self,
        quote_ctx,
        futu,
        codes: list[str],
        warnings: list[str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        try:
            ret, snapshot = self._quote_call(quote_ctx.get_market_snapshot, codes)
            _check_ret(futu, ret, snapshot, "get_market_snapshot")
            rows = {
                _safe_str(_row_get(row, "code")): row
                for _, row in _iter_rows(snapshot)
                if _safe_str(_row_get(row, "code"))
            }
            errors = {
                code: "Futu batch snapshot response omitted this symbol."
                for code in codes
                if code not in rows
            }
            return rows, errors
        except FutuAdapterError as exc:
            warnings.append(f"Futu batch snapshot failed; retrying symbols individually: {exc}")

        rows_by_code: dict[str, Any] = {}
        errors_by_code: dict[str, str] = {}
        for code in codes:
            try:
                ret, snapshot = self._quote_call(quote_ctx.get_market_snapshot, [code])
                _check_ret(futu, ret, snapshot, f"get_market_snapshot({code})")
            except FutuAdapterError as exc:
                warnings.append(f"Skipped {code}: {exc}")
                errors_by_code[code] = str(exc)
                continue
            for _, row in _iter_rows(snapshot):
                row_code = _safe_str(_row_get(row, "code"))
                if row_code:
                    rows_by_code[row_code] = row
            if code not in rows_by_code and code not in errors_by_code:
                errors_by_code[code] = "Futu individual snapshot response omitted this symbol."
        return rows_by_code, errors_by_code

    def _resolve_theme_symbols(
        self,
        quote_ctx,
        futu,
        canonical_theme: str,
        seeds: list[tuple[str, DiscoveryData]],
        plate_keywords: list[str],
    ) -> dict[str, object]:
        ordered_codes: list[str] = []
        roles: dict[str, str] = {}
        discovery_by_code: dict[str, DiscoveryData] = {}
        source_tags_by_code: dict[str, set[str]] = {}
        memberships_by_code: dict[str, list[dict[str, str]]] = {}
        warnings: list[str] = []
        source_tags = {
            "pydantic_ai_theme_discovery",
            "futu_market_snapshot",
            "futu_history_kline",
        }
        if self.config.fetch_options:
            source_tags.update(
                {
                    "futu_option_expiration_date",
                    "futu_option_chain",
                    "futu_option_snapshot",
                }
            )

        for code, discovery in seeds:
            tags = {"pydantic_ai_theme_discovery"}
            if discovery.role == "required base holding":
                tags.add("required_symbol")
                source_tags.add("required_symbol")
            _add_symbol(
                ordered_codes,
                roles,
                source_tags_by_code,
                code,
                discovery.role,
                tags,
                self.config.max_candidates,
            )
            discovery_by_code[code] = discovery

        plates = []
        try:
            plates = self._find_theme_plates(quote_ctx, futu, plate_keywords)
            if plates:
                source_tags.add("futu_plate_list")
        except FutuAdapterError as exc:
            warnings.append(str(exc))

        for plate in plates[: self.config.max_theme_plates]:
            try:
                ret, data = self._quote_call(quote_ctx.get_plate_stock, plate["plate_code"])
                _check_ret(futu, ret, data, f"get_plate_stock({plate['plate_code']})")
                source_tags.add("futu_plate_stock")
                for _, row in list(_iter_rows(data))[: self.config.plate_stock_limit]:
                    code = _safe_str(_row_get(row, "code"))
                    name = _safe_str(_row_get(row, "stock_name")) or code
                    if not code or not code.startswith(f"{self.config.market}."):
                        continue
                    membership = {
                        "plate_code": plate["plate_code"],
                        "plate_name": plate["plate_name"],
                        "plate_type": plate.get("plate_type", ""),
                    }
                    memberships_by_code.setdefault(code, []).append(membership)
                    discovery = DiscoveryData(
                        source="futu_plate_stock",
                        role=f"{plate['plate_name']} constituent",
                        rationale=f"Discovered from Futu plate {plate['plate_name']}.",
                        subthemes=[plate["plate_name"]],
                        exposure_type="plate_constituent",
                    )
                    _add_symbol(
                        ordered_codes,
                        roles,
                        source_tags_by_code,
                        code,
                        f"{plate['plate_name']} constituent",
                        {"futu_plate_stock", f"futu_plate:{plate['plate_code']}"},
                        self.config.max_candidates,
                        name=name,
                    )
                    discovery_by_code.setdefault(code, discovery)
            except FutuAdapterError as exc:
                warnings.append(str(exc))

        try:
            owner_memberships = self._get_owner_plates(quote_ctx, futu, ordered_codes)
            if owner_memberships:
                source_tags.add("futu_owner_plate")
            for code, memberships in owner_memberships.items():
                memberships_by_code.setdefault(code, [])
                known = {item["plate_code"] for item in memberships_by_code[code]}
                for membership in memberships:
                    if membership["plate_code"] not in known:
                        memberships_by_code[code].append(membership)
                source_tags_by_code.setdefault(code, set()).add("futu_owner_plate")
        except FutuAdapterError as exc:
            warnings.append(str(exc))

        return {
            "codes": ordered_codes,
            "roles": roles,
            "discovery_by_code": discovery_by_code,
            "source_tags_by_code": source_tags_by_code,
            "memberships_by_code": memberships_by_code,
            "source_tags": sorted(source_tags),
            "warnings": warnings,
        }

    def _find_theme_plates(self, quote_ctx, futu, keywords: list[str]) -> list[dict[str, str]]:
        if not keywords:
            return []
        ret, data = self._quote_call(
            quote_ctx.get_plate_list,
            _parse_quote_market(futu, self.config.market),
            futu.Plate.ALL,
        )
        _check_ret(futu, ret, data, "get_plate_list")
        matches: list[dict[str, str]] = []
        seen: set[str] = set()
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for _, row in _iter_rows(data):
                plate_code = _safe_str(_row_get(row, "code"))
                plate_name = _safe_str(_row_get(row, "plate_name", _row_get(row, "stock_name")))
                if not plate_code or plate_code in seen:
                    continue
                if keyword_lower in plate_name.lower():
                    matches.append(
                        {
                            "plate_code": plate_code,
                            "plate_name": plate_name,
                            "plate_type": "futu_plate",
                        }
                    )
                    seen.add(plate_code)
                    break
        return matches

    def _get_owner_plates(self, quote_ctx, futu, codes: list[str]) -> dict[str, list[dict[str, str]]]:
        if not codes:
            return {}
        basic_info = self._get_stock_basicinfo_by_code(quote_ctx, futu, codes)
        supported_codes = _owner_plate_supported_codes(codes, basic_info)
        if not supported_codes:
            return {}
        memberships: dict[str, list[dict[str, str]]] = {}
        for chunk in _chunks(supported_codes, 100):
            try:
                ret, data = self._quote_call(quote_ctx.get_owner_plate, chunk)
                _check_ret(futu, ret, data, "get_owner_plate")
                _merge_owner_plate_rows(memberships, data)
            except FutuAdapterError:
                for code in chunk:
                    try:
                        ret, data = self._quote_call(quote_ctx.get_owner_plate, [code])
                        _check_ret(futu, ret, data, f"get_owner_plate({code})")
                        _merge_owner_plate_rows(memberships, data)
                    except FutuAdapterError:
                        continue
        return memberships

    def _get_stock_basicinfo_by_code(self, quote_ctx, futu, codes: list[str]) -> dict[str, dict[str, str]]:
        if not codes:
            return {}
        market = _parse_quote_market(futu, self.config.market)
        info_by_code: dict[str, dict[str, str]] = {}
        stock_type = getattr(futu.SecurityType, "STOCK", None)
        if stock_type is None:
            return info_by_code
        for chunk in _chunks(codes, 200):
            try:
                ret, data = self._quote_call(
                    quote_ctx.get_stock_basicinfo,
                    market,
                    stock_type,
                    code_list=chunk,
                )
                _check_ret(futu, ret, data, "get_stock_basicinfo")
            except FutuAdapterError:
                continue
            for _, row in _iter_rows(data):
                code = _safe_str(_row_get(row, "code"))
                if not code:
                    continue
                info_by_code[code] = {
                    "name": _safe_str(_row_get(row, "name")),
                    "stock_type": _safe_str(_row_get(row, "stock_type")),
                    "stock_child_type": _safe_str(_row_get(row, "stock_child_type")),
                    "stock_owner": _safe_str(_row_get(row, "stock_owner")),
                    "exchange_type": _safe_str(_row_get(row, "exchange_type")),
                }
        return info_by_code

    def _get_options_surface(self, quote_ctx, futu, code: str, last_price: float) -> tuple[dict[str, object], str | None]:
        try:
            ret, expirations = self._quote_call(quote_ctx.get_option_expiration_date, code)
            _check_ret(futu, ret, expirations, f"get_option_expiration_date({code})")
            expiration = _select_expiration(expirations, self.config.option_dte_min, self.config.option_dte_max)
            if not expiration:
                return ({"has_option_data": False, "status": "no_expiration"}, None)

            ret, chain = self._quote_call(
                quote_ctx.get_option_chain,
                code,
                start=expiration["strike_time"],
                end=expiration["strike_time"],
                option_type=futu.OptionType.ALL,
            )
            _check_ret(futu, ret, chain, f"get_option_chain({code})")
            option_codes = _select_option_codes(chain, last_price, self.config.option_snapshot_limit)
            if not option_codes:
                return (
                    {
                        "has_option_data": False,
                        "status": "empty_chain",
                        "expiration": expiration,
                    },
                    None,
                )

            ret, snapshot = self._quote_call(quote_ctx.get_market_snapshot, option_codes)
            _check_ret(futu, ret, snapshot, f"get_market_snapshot(options:{code})")
            summary = _summarize_option_snapshot(snapshot, expiration)
            summary["data_asof"] = utc_now()
            return summary, None
        except FutuAdapterError as exc:
            return (
                {
                    "has_option_data": False,
                    "status": "error",
                    "error": str(exc),
                },
                f"Futu option surface unavailable for {code}: {exc}",
            )

    def _get_daily_kline(self, quote_ctx, futu, code: str) -> list[dict[str, float]]:
        ret, data, _page_key = self._quote_call(
            quote_ctx.request_history_kline,
            code,
            ktype=futu.KLType.K_DAY,
            autype=futu.AuType.QFQ,
            max_count=self.config.kline_count,
        )
        _check_ret(futu, ret, data, f"request_history_kline({code})")
        rows: list[dict[str, float]] = []
        for _, row in _iter_rows(data):
            rows.append(
                {
                    "close": _safe_float(_row_get(row, "close")),
                    "volume": _safe_float(_row_get(row, "volume")),
                    "turnover": _safe_float(_row_get(row, "turnover")),
                }
            )
        return rows

    def _check_opend(self) -> None:
        try:
            with socket.create_connection((self.config.host, self.config.port), timeout=2):
                return
        except OSError as exc:
            raise FutuAdapterError(
                f"Cannot connect to Futu OpenD at {self.config.host}:{self.config.port}. Start OpenD before planning."
            ) from exc


class PortfolioAdapter:
    """Futu-backed portfolio adapter."""

    def __init__(self, config: FutuOpenDConfig | None = None):
        self.config = config or FutuOpenDConfig.from_env()

    def get_current_portfolio(self) -> CurrentPortfolio:
        self._check_opend()
        futu = _import_futu()
        trd_ctx = futu.OpenSecTradeContext(
            host=self.config.host,
            port=self.config.port,
            filter_trdmarket=_parse_market(futu, self.config.market),
        )
        try:
            query_kwargs = {
                "trd_env": _parse_trd_env(futu, self.config.trd_env),
            }
            if self.config.acc_id is not None:
                query_kwargs["acc_id"] = self.config.acc_id
            if self.config.currency:
                query_kwargs["currency"] = self.config.currency
            ret, acc_data = trd_ctx.accinfo_query(**query_kwargs)
            _check_ret(futu, ret, acc_data, "accinfo_query")

            funds = _first_row(acc_data)
            cash = _safe_float(_row_get(funds, "cash"))
            market_value = _safe_float(_row_get(funds, "market_val"))
            total_assets = _safe_float(_row_get(funds, "total_assets"))
            if total_assets <= 0:
                total_assets = cash + market_value
            if total_assets <= 0:
                raise FutuAdapterError("Futu account returned total_assets <= 0.")

            position_kwargs = {
                "trd_env": _parse_trd_env(futu, self.config.trd_env),
            }
            if self.config.acc_id is not None:
                position_kwargs["acc_id"] = self.config.acc_id
            ret, pos_data = trd_ctx.position_list_query(**position_kwargs)
            _check_ret(futu, ret, pos_data, "position_list_query")

            holdings: list[CurrentHolding] = []
            for _, row in _iter_rows(pos_data):
                code = _safe_str(_row_get(row, "code"))
                if not code:
                    continue
                holdings.append(
                    CurrentHolding(
                        symbol=code,
                        quantity=max(0, int(_safe_float(_row_get(row, "qty")))),
                        market_value=max(0.0, _safe_float(_row_get(row, "market_val"))),
                        cost_basis=_safe_float(_row_get(row, "cost_price")),
                        can_sell_qty=max(0, int(_safe_float(_row_get(row, "can_sell_qty")))),
                    )
                )

            return CurrentPortfolio(
                total_assets=total_assets,
                cash=max(0.0, cash),
                holdings=holdings,
                data_asof=utc_now(),
                source="futu_opend",
                warnings=[],
            )
        finally:
            trd_ctx.close()

    def _check_opend(self) -> None:
        try:
            with socket.create_connection((self.config.host, self.config.port), timeout=2):
                return
        except OSError as exc:
            raise FutuAdapterError(
                f"Cannot connect to Futu OpenD at {self.config.host}:{self.config.port}. Start OpenD before reading portfolio."
            ) from exc


def holding_index(portfolio: CurrentPortfolio) -> dict[str, CurrentHolding]:
    return {holding.symbol: holding for holding in portfolio.holdings}


def _import_futu():
    try:
        import futu
    except ImportError as exc:
        raise FutuAdapterError(
            "futu-api is not installed. Install it in the Hermes environment, e.g. `uv pip install futu-api --python .venv/bin/python`."
        ) from exc
    return futu


def _check_ret(futu, ret, data, action: str) -> None:
    if ret != futu.RET_OK:
        raise FutuAdapterError(f"Futu {action} failed: {data}")


def _quote_result_is_rate_limited(result) -> bool:
    if not isinstance(result, tuple) or not result:
        return False
    ret = result[0]
    if ret == 0:
        return False
    data = result[1] if len(result) > 1 else ""
    text = str(data).lower()
    return any(
        marker in text
        for marker in (
            "频率过高",
            "频率太高",
            "请求太频繁",
            "too frequent",
            "rate limit",
            "too many requests",
        )
    )


def _futu_api_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__name__", "")
    if name:
        return str(name)
    wrapped = getattr(func, "func", None)
    if wrapped is not None:
        return str(getattr(wrapped, "__name__", ""))
    return ""


def _run_with_rate_limiter(limiter: Any, func: Callable[..., Any], *args, **kwargs):
    run = getattr(limiter, "run", None)
    if callable(run):
        return run(func, *args, **kwargs)
    limiter.acquire()
    return func(*args, **kwargs)


def _parse_trd_env(futu, value: str):
    return futu.TrdEnv.REAL if str(value).upper() == "REAL" else futu.TrdEnv.SIMULATE


def _parse_market(futu, value: str):
    mapping = {
        "US": futu.TrdMarket.US,
        "HK": futu.TrdMarket.HK,
        "HKCC": futu.TrdMarket.HKCC,
        "CN": futu.TrdMarket.CN,
        "NONE": futu.TrdMarket.NONE,
    }
    return mapping.get(str(value).upper(), futu.TrdMarket.US)


def _parse_quote_market(futu, value: str):
    mapping = {
        "US": futu.Market.US,
        "HK": futu.Market.HK,
        "SH": futu.Market.SH,
        "SZ": futu.Market.SZ,
    }
    return mapping.get(str(value).upper(), futu.Market.US)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _benchmark_metrics(row, kline_rows: list[dict[str, float]]) -> dict[str, object]:
    last_price = _safe_float(_row_get(row, "last_price"))
    prev_close = _safe_float(_row_get(row, "prev_close_price"))
    change_rate = ((last_price / prev_close - 1) * 100) if last_price > 0 and prev_close > 0 else 0.0
    closes = [item["close"] for item in kline_rows if item.get("close", 0) > 0]
    ma50 = sum(closes[-50:]) / min(50, len(closes)) if closes else None
    return {
        "code": _safe_str(_row_get(row, "code")),
        "name": _safe_str(_row_get(row, "name")),
        "last_price": last_price,
        "change_rate": round(change_rate, 4),
        "turnover": _safe_float(_row_get(row, "turnover")),
        "volume": _safe_float(_row_get(row, "volume")),
        "return_20d": _return_over(kline_rows, 20),
        "return_60d": _return_over(kline_rows, 60),
        "realized_volatility": _realized_volatility(kline_rows, 60),
        "above_ma50": bool(last_price > ma50) if ma50 else None,
        "data_asof": _safe_str(_row_get(row, "update_time")),
    }


def _build_macro_context(
    theme: str,
    theme_benchmarks: list[str],
    metrics: dict[str, dict[str, object]],
    warnings: list[str],
) -> dict[str, Any]:
    generated_at = utc_now()
    spy = metrics.get("US.SPY", {})
    qqq = metrics.get("US.QQQ", {})
    iwm = metrics.get("US.IWM", {})
    tlt = metrics.get("US.TLT", {})
    ief = metrics.get("US.IEF", {})
    shy = metrics.get("US.SHY", {})
    uup = metrics.get("US.UUP", {})
    vixy = metrics.get("US.VIXY", {})
    vxx = metrics.get("US.VXX", {})
    theme_items = {code: metrics.get(code, {}) for code in theme_benchmarks if code in metrics}
    theme_return_60d_values = [
        float(item["return_60d"])
        for item in theme_items.values()
        if isinstance(item.get("return_60d"), int | float)
    ]
    theme_return_60d = (
        sum(theme_return_60d_values) / len(theme_return_60d_values)
        if theme_return_60d_values
        else None
    )
    spy_return_60d = spy.get("return_60d")
    qqq_return_60d = qqq.get("return_60d")
    signals = _macro_signals(tlt, uup, vixy, vxx, spy, qqq, iwm)
    benchmark_context = {
        "artifact_type": "benchmark_context",
        "generated_at": generated_at,
        "source": "futu_market_snapshot_and_history_kline",
        "theme": theme,
        "broad_market": {
            "SPY": spy,
            "QQQ": qqq,
            "IWM": iwm,
        },
        "theme_benchmarks": theme_items,
        "theme_relative_strength": {
            "theme_return_60d": round(theme_return_60d, 6)
            if theme_return_60d is not None
            else None,
            "theme_vs_spy_60d": round(theme_return_60d - spy_return_60d, 6)
            if isinstance(theme_return_60d, int | float) and isinstance(spy_return_60d, int | float)
            else None,
            "theme_vs_qqq_60d": round(theme_return_60d - qqq_return_60d, 6)
            if isinstance(theme_return_60d, int | float) and isinstance(qqq_return_60d, int | float)
            else None,
        },
        "warnings": warnings,
    }
    market_regime = {
        "artifact_type": "market_regime",
        "generated_at": generated_at,
        "source": "futu_etf_proxy",
        "items": {
            "rates_proxy": {
                "TLT": tlt,
                "IEF": ief,
                "SHY": shy,
                "signal": _rates_signal(tlt),
            },
            "volatility_proxy": {
                "VIXY": vixy,
                "VXX": vxx,
                "signal": _volatility_signal(vixy, vxx),
            },
            "usd_proxy": {
                "UUP": uup,
                "signal": _usd_signal(uup),
            },
            "market_breadth_proxy": {
                "IWM_vs_SPY_20d": _spread_return(iwm, spy, "return_20d"),
                "SPY_above_ma50": spy.get("above_ma50"),
                "QQQ_above_ma50": qqq.get("above_ma50"),
                "signal": _breadth_signal(iwm, spy),
            },
        },
        "signals": signals,
        "warnings": warnings,
    }
    return {
        "benchmark_context": benchmark_context,
        "market_regime": market_regime,
        "warnings": warnings,
    }


def _macro_signals(*items: dict[str, object]) -> list[str]:
    signals = []
    labels = [
        _rates_signal(items[0]),
        _usd_signal(items[1]),
        _volatility_signal(items[2], items[3]),
        _breadth_signal(items[6], items[4]),
    ]
    for label in labels:
        if label and label != "neutral":
            signals.append(label)
    return signals


def _rates_signal(tlt: dict[str, object]) -> str:
    ret = tlt.get("return_20d")
    if not isinstance(ret, int | float):
        return "unknown_rates_proxy"
    if ret <= -0.03:
        return "rising_rate_pressure"
    if ret >= 0.03:
        return "falling_rate_tailwind"
    return "neutral"


def _usd_signal(uup: dict[str, object]) -> str:
    ret = uup.get("return_20d")
    if not isinstance(ret, int | float):
        return "unknown_usd_proxy"
    if ret >= 0.02:
        return "strong_usd"
    if ret <= -0.02:
        return "weak_usd"
    return "neutral"


def _volatility_signal(vixy: dict[str, object], vxx: dict[str, object]) -> str:
    values = [
        item.get("return_20d")
        for item in (vixy, vxx)
        if isinstance(item.get("return_20d"), int | float)
    ]
    if not values:
        return "unknown_volatility_proxy"
    avg = sum(values) / len(values)
    if avg >= 0.10:
        return "rising_volatility"
    if avg <= -0.10:
        return "falling_volatility"
    return "neutral"


def _breadth_signal(iwm: dict[str, object], spy: dict[str, object]) -> str:
    spread = _spread_return(iwm, spy, "return_20d")
    if not isinstance(spread, int | float):
        return "unknown_breadth_proxy"
    if spread <= -0.03:
        return "narrow_large_cap_led_market"
    if spread >= 0.03:
        return "broadening_risk_appetite"
    return "neutral"


def _spread_return(left: dict[str, object], right: dict[str, object], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if not isinstance(left_value, int | float) or not isinstance(right_value, int | float):
        return None
    return round(left_value - right_value, 6)


def _add_symbol(
    ordered_codes: list[str],
    roles: dict[str, str],
    source_tags_by_code: dict[str, set[str]],
    code: str,
    role: str,
    source_tags: set[str],
    max_candidates: int | None,
    name: str = "",
) -> None:
    if not code:
        return
    if code not in roles:
        if max_candidates is not None and len(ordered_codes) >= max_candidates:
            return
        ordered_codes.append(code)
        roles[code] = role or name or "theme exposure"
    source_tags_by_code.setdefault(code, set()).update(source_tags)


def _chunks(items: list[str], size: int):
    size = max(1, size)
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _owner_plate_supported_codes(codes: list[str], basic_info: dict[str, dict[str, str]]) -> list[str]:
    supported: list[str] = []
    for code in codes:
        stock_type = _safe_str((basic_info.get(code) or {}).get("stock_type")).upper()
        if stock_type and stock_type != "STOCK":
            continue
        supported.append(code)
    return supported


def _merge_owner_plate_rows(memberships: dict[str, list[dict[str, str]]], data) -> None:
    for _, row in _iter_rows(data):
        code = _safe_str(_row_get(row, "code"))
        plate_code = _safe_str(_row_get(row, "plate_code"))
        if not code or not plate_code:
            continue
        membership = {
            "plate_code": plate_code,
            "plate_name": _safe_str(_row_get(row, "plate_name")),
            "plate_type": _safe_str(_row_get(row, "plate_type")),
        }
        existing = {item["plate_code"] for item in memberships.setdefault(code, [])}
        if plate_code not in existing:
            memberships[code].append(membership)


def _seeds_from_discovery_plan(
    plan: ThemeDiscoveryPlan,
    required_symbols: list[str],
    market: str,
) -> list[tuple[str, DiscoveryData]]:
    required_codes = [normalize_market_symbol(symbol, market) for symbol in required_symbols]
    required_codes = [symbol for symbol in required_codes if symbol]
    roles_by_code = {
        normalize_market_symbol(seed.symbol, market): seed.role
        for seed in plan.seed_symbols
        if normalize_market_symbol(seed.symbol, market)
    }
    discovery_by_code = {
        normalize_market_symbol(seed.symbol, market): _discovery_from_seed(seed)
        for seed in plan.seed_symbols
        if normalize_market_symbol(seed.symbol, market)
    }

    seeds: list[tuple[str, DiscoveryData]] = []
    seen: set[str] = set()
    for code in required_codes:
        if code in seen:
            continue
        discovery = discovery_by_code.get(code) or DiscoveryData(
            source="user_required_symbol",
            role=roles_by_code.get(code) or "required base holding",
            rationale="User-required base holding.",
            exposure_type="required_symbol",
            source_ids=[],
            confidence="high",
            freshness="unknown",
        )
        seeds.append((code, discovery))
        seen.add(code)

    for seed in plan.seed_symbols:
        code = normalize_market_symbol(seed.symbol, market)
        if not code or code in seen:
            continue
        seeds.append((code, _discovery_from_seed(seed)))
        seen.add(code)
    return seeds


def _discovery_from_seed(seed) -> DiscoveryData:
    return DiscoveryData(
        role=seed.role,
        rationale=seed.rationale,
        subthemes=seed.subthemes,
        value_chain_stage=seed.value_chain_stage,
        exposure_type=seed.exposure_type,
        exposure_purity=seed.exposure_purity,
        source_ids=seed.source_ids,
        confidence=seed.confidence,
        freshness=seed.freshness,
    )


def _optional_positive_int(value: str, minimum: int) -> int | None:
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return max(minimum, parsed)


def _iter_rows(data):
    if data is None:
        return []
    if hasattr(data, "iterrows"):
        return data.iterrows()
    return enumerate(data or [])


def _first_row(data):
    if data is None:
        return {}
    if hasattr(data, "empty") and data.empty:
        return {}
    if hasattr(data, "iloc"):
        return data.iloc[0]
    if isinstance(data, list) and data:
        return data[0]
    return {}


def _row_get(row, key: str, default=None):
    try:
        value = row.get(key, default)
    except AttributeError:
        value = row[key] if key in row else default
    return default if _is_nan(value) else value


def _is_nan(value) -> bool:
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value, default: float = 0.0) -> float:
    if value is None or _is_nan(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value) -> str:
    if value is None or _is_nan(value):
        return ""
    return str(value).strip()


def _return_over(kline_rows: list[dict[str, float]], lookback: int) -> float | None:
    closes = [row["close"] for row in kline_rows if row.get("close", 0) > 0]
    if len(closes) <= lookback:
        return None
    start = closes[-lookback - 1]
    end = closes[-1]
    if start <= 0:
        return None
    return round(end / start - 1, 6)


def _daily_returns(kline_rows: list[dict[str, float]], lookback: int) -> list[float]:
    closes = [row["close"] for row in kline_rows if row.get("close", 0) > 0]
    returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    return [round(value, 6) for value in returns[-lookback:]]


def _realized_volatility(kline_rows: list[dict[str, float]], lookback: int) -> float | None:
    returns = _daily_returns(kline_rows, lookback)
    if len(returns) < 2:
        return None
    return round(pstdev(returns) * math.sqrt(252), 6)


def _select_expiration(data, dte_min: int, dte_max: int) -> dict[str, object] | None:
    rows = []
    for _, row in _iter_rows(data):
        strike_time = _safe_str(_row_get(row, "strike_time"))
        if not strike_time:
            continue
        dte = int(_safe_float(_row_get(row, "option_expiry_date_distance"), default=-1))
        if dte < 0:
            dte = _days_to(strike_time)
        if dte < 0:
            continue
        rows.append(
            {
                "strike_time": strike_time,
                "dte": dte,
                "expiration_cycle": _safe_str(_row_get(row, "expiration_cycle")),
            }
        )
    if not rows:
        return None
    midpoint = (dte_min + dte_max) / 2
    in_range = [row for row in rows if dte_min <= int(row["dte"]) <= dte_max]
    choices = in_range or [row for row in rows if int(row["dte"]) > 0]
    if not choices:
        return None
    return min(choices, key=lambda item: abs(int(item["dte"]) - midpoint))


def _days_to(date_text: str) -> int:
    try:
        target = datetime.fromisoformat(date_text[:10]).date()
    except ValueError:
        return -1
    return (target - datetime.now(timezone.utc).date()).days


def _select_option_codes(chain, last_price: float, limit: int) -> list[str]:
    rows = []
    for _, row in _iter_rows(chain):
        code = _safe_str(_row_get(row, "code"))
        strike = _safe_float(_row_get(row, "strike_price"))
        option_type = _safe_str(_row_get(row, "option_type"))
        if not code or strike <= 0 or not option_type:
            continue
        distance = abs(strike / last_price - 1) if last_price > 0 else 0.0
        if distance > 0.30:
            continue
        rows.append((distance, option_type, code))
    rows.sort(key=lambda item: (item[0], item[1]))
    calls = [code for _, option_type, code in rows if "CALL" in option_type.upper()]
    puts = [code for _, option_type, code in rows if "PUT" in option_type.upper()]
    per_side = max(1, limit // 2)
    selected = calls[:per_side] + puts[:per_side]
    if len(selected) < limit:
        selected_set = set(selected)
        selected.extend(code for _, _, code in rows if code not in selected_set)
    return selected[:limit]


def _summarize_option_snapshot(snapshot, expiration: dict[str, object]) -> dict[str, object]:
    rows = []
    iv_values = []
    spreads = []
    for _, row in _iter_rows(snapshot):
        if _safe_str(_row_get(row, "option_valid")).lower() == "false":
            continue
        bid = _safe_float(_row_get(row, "bid_price"))
        ask = _safe_float(_row_get(row, "ask_price"))
        spread = _spread_bps(bid, ask)
        if spread is not None:
            spreads.append(spread)
        iv = _safe_float(_row_get(row, "option_implied_volatility"))
        if iv > 0:
            iv_values.append(iv)
        rows.append(
            {
                "code": _safe_str(_row_get(row, "code")),
                "option_type": _safe_str(_row_get(row, "option_type")),
                "strike": _safe_float(_row_get(row, "option_strike_price")),
                "last_price": _safe_float(_row_get(row, "last_price")),
                "bid": bid,
                "ask": ask,
                "spread_bps": spread,
                "volume": int(_safe_float(_row_get(row, "volume"))),
                "open_interest": int(_safe_float(_row_get(row, "option_open_interest"))),
                "implied_volatility": iv,
                "delta": _safe_float(_row_get(row, "option_delta")),
                "gamma": _safe_float(_row_get(row, "option_gamma")),
                "vega": _safe_float(_row_get(row, "option_vega")),
                "theta": _safe_float(_row_get(row, "option_theta")),
                "rho": _safe_float(_row_get(row, "option_rho")),
            }
        )

    call_rows = [row for row in rows if "CALL" in str(row["option_type"]).upper()]
    put_rows = [row for row in rows if "PUT" in str(row["option_type"]).upper()]
    avg_iv = sum(iv_values) / len(iv_values) if iv_values else None
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    return {
        "has_option_data": bool(rows),
        "status": "ok" if rows else "empty_snapshot",
        "expiration": expiration,
        "contracts_sampled": len(rows),
        "call_count": len(call_rows),
        "put_count": len(put_rows),
        "avg_implied_volatility": round(avg_iv, 4) if avg_iv is not None else None,
        "iv_rank_proxy": _iv_rank_proxy(avg_iv),
        "avg_spread_bps": round(avg_spread, 2) if avg_spread is not None else None,
        "call_candidates": _top_option_rows(call_rows),
        "put_candidates": _top_option_rows(put_rows),
    }


def _top_option_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    liquid = sorted(
        rows,
        key=lambda item: (
            int(item.get("open_interest") or 0),
            int(item.get("volume") or 0),
            float(item.get("bid") or 0),
        ),
        reverse=True,
    )
    return liquid[:5]


def _iv_rank_proxy(avg_iv: float | None) -> float | None:
    if avg_iv is None or avg_iv <= 0:
        return None
    value = avg_iv * 100 if avg_iv <= 2 else avg_iv
    return round(max(0.0, min(100.0, value)), 2)


def _market_factors(kline_rows: list[dict[str, float]]) -> tuple[float, float]:
    closes = [row["close"] for row in kline_rows if row.get("close", 0) > 0]
    if len(closes) < 2:
        return 50.0, 0.0
    lookback = min(60, len(closes) - 1)
    start = closes[-lookback - 1]
    end = closes[-1]
    total_return = (end / start - 1) if start > 0 else 0.0
    returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]
    volatility = pstdev(returns[-lookback:]) * math.sqrt(252) if returns else 0.0
    relative_strength = max(0.0, min(100.0, 50.0 + total_return * 100.0))
    return relative_strength, max(0.0, volatility)


def _candidate_score(
    relative_strength: float,
    volatility: float,
    turnover: float,
    change_rate: float,
) -> float:
    return _candidate_score_breakdown(relative_strength, volatility, turnover, change_rate)["total"]


def _candidate_score_breakdown(
    relative_strength: float,
    volatility: float,
    turnover: float,
    change_rate: float,
) -> dict[str, float]:
    liquidity_score = min(20.0, math.log10(max(turnover, 1.0)) * 2.0)
    momentum_score = relative_strength * 0.55
    change_score = max(-10.0, min(10.0, change_rate)) * 1.2
    volatility_penalty = min(20.0, volatility * 18.0)
    score = 35.0 + momentum_score + liquidity_score + change_score - volatility_penalty
    return {
        "total": round(max(0.0, min(100.0, score)), 2),
        "momentum": round(momentum_score, 2),
        "liquidity": round(liquidity_score, 2),
        "change": round(change_score, 2),
        "volatility_penalty": round(-volatility_penalty, 2),
    }


def _spread_bps(bid_price: float, ask_price: float) -> float | None:
    midpoint = (bid_price + ask_price) / 2
    if bid_price <= 0 or ask_price <= 0 or midpoint <= 0:
        return None
    return round((ask_price - bid_price) / midpoint * 10_000, 2)


def _trend_label(relative_strength: float) -> str:
    if relative_strength >= 65:
        return "uptrend"
    if relative_strength <= 40:
        return "downtrend"
    return "neutral"
