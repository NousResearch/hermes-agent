from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from plugins.investment_assistant.adapters import FutuAdapterError, FutuOpenDConfig, MarketDataAdapter, ThemeUniverse
from plugins.investment_assistant import adapters as ia_adapters
from plugins.investment_assistant import agents as ia_agents
from plugins.investment_assistant import market_artifacts as ia_market_artifacts
from plugins.investment_assistant import pydantic_runtime
from plugins.investment_assistant import theme_discovery
from plugins.investment_assistant import futu_theme_discovery
from plugins.investment_assistant import discovery_v1
from plugins.investment_assistant import filter_calibration
from plugins.investment_assistant import fundamental_tools
from plugins.investment_assistant import candidate_triage
from plugins.investment_assistant import data_miner
from plugins.investment_assistant import deep_research
from plugins.investment_assistant import exposure_drift
from plugins.investment_assistant import exposure_ledger
from plugins.investment_assistant import fmp_provider
from plugins.investment_assistant import filing_summary
from plugins.investment_assistant import lightweight_enrichment
from plugins.investment_assistant import option_market_data
from plugins.investment_assistant import pydantic_hitl_cli
from plugins.investment_assistant import pydantic_resume
from plugins.investment_assistant import portfolio_monitor
from plugins.investment_assistant import portfolio_architect
from plugins.investment_assistant import portfolio_revision
from plugins.investment_assistant import portfolio_weight_formula
from plugins.investment_assistant import sec_provider
from plugins.investment_assistant import skill_runtime
from plugins.investment_assistant import symbol_store
from plugins.investment_assistant import candidate_pool as ia_candidate_pool
from plugins.investment_assistant import workflow as ia_workflow
from plugins.investment_assistant import websearch_discovery
from plugins.investment_assistant.market_artifacts import build_market_artifacts, reflect_candidate_pool
from plugins.investment_assistant import output_guard
from plugins.investment_assistant.schemas import (
    Candidate,
    CandidatePool,
    CandidateThesisAssessment,
    CalibratedFilter,
    CalibrationInputProbe,
    CalibrationTrial,
    CalibrationTrialResult,
    CurrentHolding,
    CurrentPortfolio,
    DiscoveryData,
    DiscoveryFilterDecision,
    DiscoveryFilterAuditItem,
    DiscoveryFilterPlan,
    DiscoveryLayerFilterAudit,
    DiscoveryOmission,
    ExecutedDiscoveryProbe,
    FilterCalibrationArtifact,
    FutuData,
    InvestmentPolicy,
    OmittedCandidate,
    PortfolioHolding,
    PortfolioMap,
    PortfolioMaps,
    PortfolioSleeve,
    ResearchSource,
    ThemeCoverageRequirement,
    ThemeDomain,
    ThemeDomainCandidate,
    ThemeDiscoveryPlan,
    ThemeDiscoverySeed,
    ThemeSubdomain,
    ThesisSynthesis,
    WorkflowState,
)
from plugins.investment_assistant.storage import InvestmentAssistantStore
from plugins.investment_assistant.workflow import InvestmentAssistantWorkflow
from plugins.investment_assistant.tools import (
    IA_PORTFOLIO_WORKFLOW_SCHEMA,
    current_hermes_tenant,
    handle_ia_portfolio_workflow,
    transform_llm_output,
)

def _use_default_research_settings(monkeypatch):
    monkeypatch.setattr(pydantic_runtime, "_read_hermes_config", lambda: {})
    for name in [
        "IA_RESEARCH_WEB_ENABLED",
        "IA_RESEARCH_MAX_SEARCHES",
        "IA_RESEARCH_MAX_FETCHES",
        "IA_RESEARCH_REQUIRE_SOURCES_FOR_MUST_CONSIDER",
        "IA_RESEARCH_THINKING_EFFORT",
        "IA_RESEARCH_WEB_SEARCH_MODE",
        "IA_RESEARCH_WEB_FETCH_MODE",
        "IA_RESEARCH_WEB_FETCH_TIMEOUT",
        "IA_RESEARCH_WEB_FETCH_RETRIES",
        "IA_RESEARCH_WEB_FETCH_MAX_CONTENT_LENGTH",
        "IA_RESEARCH_REQUEST_TIMEOUT",
        "IA_PYDANTIC_TRACE_PART_DELTAS",
    ]:
        monkeypatch.delenv(name, raising=False)


def _fake_futu_data(
    *,
    last_price: float,
    relative_strength: float,
    volatility: float,
    market_snapshot: dict | None = None,
    technical_summary: dict | None = None,
    liquidity_context: dict | None = None,
    options_surface: dict | None = None,
) -> FutuData:
    technical = {
        "relative_strength_60d": relative_strength,
        "realized_volatility": volatility,
        **(technical_summary or {}),
    }
    return FutuData.from_parts(
        last_price=last_price,
        quote=market_snapshot or {},
        technical=technical,
        liquidity=liquidity_context or {},
        options=options_surface or {"has_option_data": False, "status": "not_wired"},
        data_asof={},
    )


def _fake_discovery_data(role: str, rationale: str = "") -> DiscoveryData:
    return DiscoveryData(
        role=role,
        rationale=rationale or f"Fake discovery rationale for {role}.",
        subthemes=[role],
        value_chain_stage=role,
        exposure_type="direct",
        exposure_purity="medium",
    )


def _ai_coverage_requirements() -> list[ThemeCoverageRequirement]:
    return [
        ThemeCoverageRequirement(
            key="gpu_accelerator",
            name="GPU/accelerator compute",
            thesis="Accelerator platforms are a required compute exposure for this test plan.",
            candidate_symbols=["US.NVDA"],
            must_consider_symbols=[],
        ),
        ThemeCoverageRequirement(
            key="hbm_dram",
            name="HBM/DRAM memory",
            thesis="Memory bandwidth is a required exposure for this test plan.",
            candidate_symbols=["US.MU"],
            must_consider_symbols=[],
        ),
        ThemeCoverageRequirement(
            key="essd_nand_storage",
            name="eSSD/NAND/storage",
            thesis="Storage throughput is a required exposure for this test plan.",
            candidate_symbols=["US.SNDK"],
            must_consider_symbols=["US.SNDK"],
        ),
        ThemeCoverageRequirement(
            key="optical_networking_components",
            name="Optical networking/components",
            thesis="Optical connectivity is a required exposure for this test plan.",
            candidate_symbols=["US.COHR"],
            must_consider_symbols=["US.COHR"],
        ),
        ThemeCoverageRequirement(
            key="custom_silicon_connectivity",
            name="Custom silicon/connectivity",
            thesis="Custom silicon and connectivity are required exposures for this test plan.",
            candidate_symbols=["US.MRVL"],
            must_consider_symbols=["US.MRVL"],
        ),
        ThemeCoverageRequirement(
            key="foundry_equipment",
            name="Foundry/equipment",
            thesis="Foundry and equipment are required exposures for this test plan.",
            candidate_symbols=["US.TSM"],
            must_consider_symbols=[],
        ),
        ThemeCoverageRequirement(
            key="power_cooling",
            name="Power/cooling",
            thesis="Power and cooling are required exposures for this test plan.",
            candidate_symbols=["US.VRT"],
            must_consider_symbols=[],
        ),
        ThemeCoverageRequirement(
            key="cloud_platform",
            name="Cloud/platform",
            thesis="Cloud platforms are required exposures for this test plan.",
            candidate_symbols=["US.MSFT"],
            must_consider_symbols=[],
        ),
    ]


def _ai_domain_tree() -> list[ThemeDomain]:
    return [
        ThemeDomain(
            key="compute",
            name="Compute infrastructure",
            thesis="Accelerator and custom compute are core discovery domains.",
            importance="core",
            subdomains=[
                ThemeSubdomain(
                    key="gpu_accelerator",
                    name="GPU/accelerator compute",
                    thesis="Accelerator platforms are core compute exposure.",
                    importance="high",
                    candidate_limit_reason="Keep the required leader and let downstream data validate concentration.",
                    candidates=[
                        ThemeDomainCandidate(
                            symbol="NVDA",
                            role="required compute leader",
                            rationale="Required user symbol and core accelerator exposure.",
                            priority="must_consider",
                        )
                    ],
                ),
                ThemeSubdomain(
                    key="custom_silicon_connectivity",
                    name="Custom silicon/connectivity",
                    thesis="Custom silicon and connectivity are adjacent compute bottlenecks.",
                    importance="high",
                    candidate_limit_reason="Keep one representative connectivity/custom silicon candidate for this test.",
                    candidates=[
                        ThemeDomainCandidate(
                            symbol="MRVL",
                            role="custom silicon connectivity",
                            rationale="Represents custom silicon and data-center connectivity.",
                            priority="must_consider",
                        )
                    ],
                ),
            ],
        ),
        ThemeDomain(
            key="memory_storage_network",
            name="Memory, storage, and optical network",
            thesis="Bandwidth and persistence bottlenecks should be discovered separately.",
            importance="important",
            subdomains=[
                ThemeSubdomain(
                    key="hbm_dram",
                    name="HBM/DRAM memory",
                    thesis="Memory bandwidth is a required exposure for this test plan.",
                    importance="high",
                    candidate_limit_reason="Use the memory pure-play representative.",
                    candidates=[
                        ThemeDomainCandidate(symbol="MU", role="HBM/DRAM memory", rationale="Memory bandwidth candidate.")
                    ],
                ),
                ThemeSubdomain(
                    key="essd_nand_storage",
                    name="eSSD/NAND/storage",
                    thesis="Storage throughput is a required exposure for this test plan.",
                    importance="high",
                    candidate_limit_reason="Use the storage pure-play representative.",
                    candidates=[
                        ThemeDomainCandidate(
                            symbol="SNDK",
                            role="AI eSSD and NAND storage",
                            rationale="Storage throughput candidate.",
                            priority="must_consider",
                        )
                    ],
                ),
                ThemeSubdomain(
                    key="optical_networking_components",
                    name="Optical networking/components",
                    thesis="Optical connectivity is a required exposure for this test plan.",
                    importance="high",
                    candidate_limit_reason="Use the optical-components representative.",
                    candidates=[
                        ThemeDomainCandidate(
                            symbol="COHR",
                            role="optical components",
                            rationale="Optical components candidate.",
                            priority="must_consider",
                        )
                    ],
                ),
            ],
        ),
        ThemeDomain(
            key="manufacturing_infrastructure_platforms",
            name="Manufacturing, infrastructure, and platforms",
            thesis="Upstream manufacturing, physical infrastructure, and platforms complete the map.",
            importance="important",
            subdomains=[
                ThemeSubdomain(
                    key="foundry_equipment",
                    name="Foundry/equipment",
                    thesis="Foundry and equipment are required exposures for this test plan.",
                    importance="medium",
                    candidate_limit_reason="Use one foundry representative for this test.",
                    candidates=[ThemeDomainCandidate(symbol="TSM", role="foundry", rationale="Foundry candidate.")],
                ),
                ThemeSubdomain(
                    key="power_cooling",
                    name="Power/cooling",
                    thesis="Power and cooling are required exposures for this test plan.",
                    importance="medium",
                    candidate_limit_reason="Use one power/cooling representative for this test.",
                    candidates=[
                        ThemeDomainCandidate(
                            symbol="VRT",
                            role="data center infrastructure",
                            rationale="Power and cooling candidate.",
                        )
                    ],
                ),
                ThemeSubdomain(
                    key="cloud_platform",
                    name="Cloud/platform",
                    thesis="Cloud platforms are required exposures for this test plan.",
                    importance="medium",
                    candidate_limit_reason="Use one cloud-platform representative for this test.",
                    candidates=[
                        ThemeDomainCandidate(symbol="MSFT", role="cloud platform", rationale="Cloud platform candidate.")
                    ],
                ),
            ],
        ),
    ]


def _ai_research_trace() -> list[ResearchSource]:
    return [
        ResearchSource(
            source_id="src_ai_compute",
            title="AI compute and infrastructure research note",
            url="https://example.com/ai-compute",
            publisher="Example Research",
            retrieved_at="2026-05-20T00:00:00+00:00",
            source_type="web",
            summary="Test source covering compute, storage, optical connectivity, custom silicon, and cloud.",
            symbols=["US.NVDA", "US.SNDK", "US.COHR", "US.MRVL", "US.MU", "US.TSM", "US.VRT", "US.MSFT"],
            coverage_keys=[
                "gpu_accelerator",
                "hbm_dram",
                "essd_nand_storage",
                "optical_networking_components",
                "custom_silicon_connectivity",
                "foundry_equipment",
                "power_cooling",
                "cloud_platform",
            ],
        )
    ]


def _candidate(
    symbol: str,
    role: str,
    *,
    score: float = 80,
    eligible: bool = True,
) -> Candidate:
    return Candidate(
        symbol=symbol,
        name=symbol,
        theme_role=role,
        source="fake_test_data",
        source_tags=["pydantic_ai_theme_discovery", "fake_market_snapshot", "fake_history_kline"],
        score=score,
        candidate_status="futu_enriched" if eligible else "quote_unavailable",
        eligible_for_portfolio=eligible,
        discovery_data=_fake_discovery_data(role),
        futu_data=_fake_futu_data(
            last_price=100,
            relative_strength=75,
            volatility=0.25,
            market_snapshot={"turnover": 1_000_000_000, "volume": 1_000_000},
            liquidity_context={"liquidity_score": 0.9},
        ),
    )


class FakeMarketDataAdapter:
    def get_theme_universe(
        self,
        theme: str,
        required_symbols: list[str] | None = None,
        theme_description: str = "",
    ) -> ThemeUniverse:
        candidates = [
            Candidate(
                symbol="US.MU",
                name="Micron Technology",
                theme_role="memory cycle leader",
                source="fake_test_data",
                source_tags=["pydantic_ai_theme_discovery", "fake_market_snapshot", "fake_history_kline"],
                score=91,
                discovery_data=_fake_discovery_data("memory cycle leader"),
                futu_data=_fake_futu_data(
                    last_price=132.20,
                    relative_strength=84,
                    volatility=0.39,
                    market_snapshot={
                        "turnover": 1_200_000_000,
                        "volume": 10_000_000,
                        "total_market_val": 140_000_000_000,
                        "pe_ttm_ratio": 18,
                        "pb_ratio": 2.1,
                        "net_profit": 8_000_000_000,
                        "earning_per_share": 5.2,
                        "highest52weeks_price": 150,
                        "lowest52weeks_price": 80,
                        "volume_ratio": 1.4,
                    },
                    technical_summary={
                        "trend": "uptrend",
                        "daily_returns_60d": [0.01, -0.004, 0.006, 0.003] * 15,
                    },
                    liquidity_context={"liquidity_score": 0.93},
                    options_surface={"has_option_data": False, "status": "not_wired"},
                ),
                plate_memberships=[{"plate_code": "US.LIST23925", "plate_name": "存储概念股"}],
            ),
            Candidate(
                symbol="US.WDC",
                name="Western Digital",
                theme_role="HDD and flash exposure",
                source="fake_test_data",
                source_tags=["pydantic_ai_theme_discovery", "fake_market_snapshot", "fake_history_kline"],
                score=78,
                discovery_data=_fake_discovery_data("HDD and flash exposure"),
                futu_data=_fake_futu_data(
                    last_price=74.10,
                    relative_strength=70,
                    volatility=0.36,
                    market_snapshot={
                        "turnover": 700_000_000,
                        "volume": 8_000_000,
                        "total_market_val": 70_000_000_000,
                        "pe_ttm_ratio": 22,
                        "pb_ratio": 2.4,
                        "net_profit": 3_500_000_000,
                        "earning_per_share": 3.8,
                        "highest52weeks_price": 90,
                        "lowest52weeks_price": 45,
                        "volume_ratio": 1.2,
                    },
                    technical_summary={
                        "trend": "uptrend",
                        "daily_returns_60d": [0.008, -0.003, 0.004, 0.002] * 15,
                    },
                    liquidity_context={"liquidity_score": 0.88},
                    options_surface={"has_option_data": False, "status": "not_wired"},
                ),
                plate_memberships=[{"plate_code": "US.LIST23925", "plate_name": "存储概念股"}],
            ),
            Candidate(
                symbol="US.STX",
                name="Seagate Technology",
                theme_role="HDD infrastructure exposure",
                source="fake_test_data",
                source_tags=["pydantic_ai_theme_discovery", "fake_market_snapshot", "fake_history_kline"],
                score=74,
                discovery_data=_fake_discovery_data("HDD infrastructure exposure"),
                futu_data=_fake_futu_data(
                    last_price=91.30,
                    relative_strength=65,
                    volatility=0.31,
                    market_snapshot={
                        "turnover": 650_000_000,
                        "volume": 7_500_000,
                        "total_market_val": 35_000_000_000,
                        "pe_ttm_ratio": 20,
                        "pb_ratio": 2.0,
                        "net_profit": 2_900_000_000,
                        "earning_per_share": 6.1,
                        "highest52weeks_price": 110,
                        "lowest52weeks_price": 55,
                        "volume_ratio": 1.1,
                    },
                    technical_summary={
                        "trend": "uptrend",
                        "daily_returns_60d": [0.007, -0.002, 0.003, 0.001] * 15,
                    },
                    liquidity_context={"liquidity_score": 0.85},
                    options_surface={"has_option_data": False, "status": "not_wired"},
                ),
                plate_memberships=[{"plate_code": "US.LIST23925", "plate_name": "存储概念股"}],
            ),
        ]
        for symbol in reversed(required_symbols or []):
            candidates.insert(
                0,
                Candidate(
                    symbol=symbol,
                    name=symbol,
                    theme_role="required base holding",
                    source="fake_test_data",
                    source_tags=["required_symbol", "fake_market_snapshot", "fake_history_kline"],
                    score=100,
                    discovery_data=_fake_discovery_data("required base holding"),
                    futu_data=_fake_futu_data(
                        last_price=100,
                        relative_strength=100,
                        volatility=0.2,
                        market_snapshot={
                            "turnover": 2_000_000_000,
                            "volume": 20_000_000,
                            "total_market_val": 500_000_000_000,
                            "pe_ttm_ratio": 30,
                            "pb_ratio": 6,
                        },
                        technical_summary={
                            "trend": "uptrend",
                            "daily_returns_60d": [0.005, -0.002, 0.004, 0.001] * 15,
                        },
                        liquidity_context={"liquidity_score": 0.95},
                        options_surface={"has_option_data": False, "status": "not_wired"},
                    ),
                ),
            )
        return ThemeUniverse(
            canonical_theme=theme.lower(),
            source_tags=["fake_test_data"],
            candidates=candidates,
            warnings=[],
        )


def test_real_market_data_adapter_rejects_free_form_theme():
    adapter = MarketDataAdapter()

    assert adapter._canonical_theme("AI") == "ai"
    assert adapter._canonical_theme("semiconductor") == "semiconductor"
    with pytest.raises(FutuAdapterError):
        adapter._canonical_theme("AI持仓版图规划，包含半导体、存储、网络和电力")


def test_futu_candidate_limit_is_unbounded_unless_configured(monkeypatch):
    monkeypatch.delenv("IA_FUTU_MAX_CANDIDATES", raising=False)

    assert FutuOpenDConfig.from_env().max_candidates is None

    monkeypatch.setenv("IA_FUTU_MAX_CANDIDATES", "12")
    assert FutuOpenDConfig.from_env().max_candidates == 12

    monkeypatch.setenv("IA_FUTU_MAX_CANDIDATES", "0")
    assert FutuOpenDConfig.from_env().max_candidates is None


def test_futu_quote_rate_limit_defaults_below_opend_limit(monkeypatch):
    monkeypatch.delenv("IA_FUTU_QUOTE_RATE_LIMIT_CALLS", raising=False)
    monkeypatch.delenv("IA_FUTU_QUOTE_RATE_LIMIT_WINDOW", raising=False)
    monkeypatch.delenv("IA_FUTU_SCREENER_RATE_LIMIT_CALLS", raising=False)
    monkeypatch.delenv("IA_FUTU_SCREENER_RATE_LIMIT_WINDOW", raising=False)
    monkeypatch.delenv("IA_FUTU_SCREENER_RATE_LIMIT_RETRIES", raising=False)

    config = FutuOpenDConfig.from_env()

    assert config.quote_rate_limit_calls == 55
    assert config.quote_rate_limit_window == 30
    assert config.quote_rate_limit_retries == 1
    assert config.screener_rate_limit_calls == 8
    assert config.screener_rate_limit_window == 30
    assert config.screener_rate_limit_retries == 2


def test_futu_quote_call_invokes_rate_limiter_before_api_call():
    class RecordingLimiter:
        def __init__(self):
            self.events = []

        def acquire(self):
            self.events.append("acquire")

    calls = []
    adapter = MarketDataAdapter()
    limiter = RecordingLimiter()
    adapter._quote_rate_limiter = limiter

    def fake_quote_method(value):
        calls.append(value)
        limiter.events.append("api")
        return "ok"

    result = adapter._quote_call(fake_quote_method, "US.NVDA")

    assert result == "ok"
    assert calls == ["US.NVDA"]
    assert limiter.events == ["acquire", "api"]


def test_futu_stock_filter_call_uses_screener_rate_limiter():
    class RecordingLimiter:
        def __init__(self):
            self.events = []

        def acquire(self):
            self.events.append("acquire")

    quote_limiter = RecordingLimiter()
    screener_limiter = RecordingLimiter()
    adapter = MarketDataAdapter()
    adapter._quote_rate_limiter = quote_limiter
    adapter._screener_rate_limiter = screener_limiter

    def get_stock_filter():
        screener_limiter.events.append("api")
        return "ok"

    result = adapter._quote_call(get_stock_filter)

    assert result == "ok"
    assert quote_limiter.events == []
    assert screener_limiter.events == ["acquire", "api"]


def test_futu_quote_call_retries_once_after_rate_limit_response(monkeypatch):
    monkeypatch.setenv("IA_FUTU_QUOTE_RATE_LIMIT_CALLS", "0")
    monkeypatch.setenv("IA_FUTU_QUOTE_RATE_LIMIT_WINDOW", "30")
    monkeypatch.setenv("IA_FUTU_QUOTE_RATE_LIMIT_RETRIES", "1")
    sleeps = []
    monkeypatch.setattr(ia_adapters.time, "sleep", lambda seconds: sleeps.append(seconds))

    adapter = MarketDataAdapter()
    responses = [
        (1, "行情快照请求频率过高：每 30 秒最多 60 次"),
        (0, [{"code": "US.NVDA"}]),
    ]

    result = adapter._quote_call(lambda: responses.pop(0))

    assert result == (0, [{"code": "US.NVDA"}])
    assert sleeps == [30]


def test_futu_stock_filter_call_retries_with_screener_window(monkeypatch):
    monkeypatch.setenv("IA_FUTU_SCREENER_RATE_LIMIT_CALLS", "0")
    monkeypatch.setenv("IA_FUTU_SCREENER_RATE_LIMIT_WINDOW", "12")
    monkeypatch.setenv("IA_FUTU_SCREENER_RATE_LIMIT_RETRIES", "1")
    sleeps = []
    monkeypatch.setattr(ia_adapters.time, "sleep", lambda seconds: sleeps.append(seconds))

    adapter = MarketDataAdapter()
    responses = [
        (1, "条件选股频率太高，请求失败，每30秒最多10次。"),
        (0, "ok"),
    ]

    def get_stock_filter():
        return responses.pop(0)

    result = adapter._quote_call(get_stock_filter)

    assert result == (0, "ok")
    assert sleeps == [12]


def test_futu_quote_rate_limiter_is_shared_per_opend_config(monkeypatch):
    monkeypatch.setenv("IA_FUTU_QUOTE_RATE_LIMIT_CALLS", "55")
    monkeypatch.setenv("IA_FUTU_QUOTE_RATE_LIMIT_WINDOW", "30")
    monkeypatch.setenv("IA_FUTU_SCREENER_RATE_LIMIT_CALLS", "8")
    monkeypatch.setenv("IA_FUTU_SCREENER_RATE_LIMIT_WINDOW", "30")
    config = FutuOpenDConfig.from_env()

    first = MarketDataAdapter(config=config)
    second = MarketDataAdapter(config=config)

    assert first._quote_rate_limiter is second._quote_rate_limiter
    assert first._screener_rate_limiter is second._screener_rate_limiter


def test_candidate_stores_futu_fields_only_in_sub_schema():
    candidate = Candidate(
        symbol="US.SNDK",
        name="Sandisk",
        theme_role="AI eSSD and NAND storage bottleneck",
        source="futu_opend",
        score=88,
        discovery_data=_fake_discovery_data(
            "AI eSSD and NAND storage bottleneck",
            "SNDK is included for AI inference storage throughput exposure.",
        ),
        futu_data=FutuData.from_parts(
            last_price=1200,
            quote={
                "update_time": "2026-05-20 09:30:00",
                "turnover": 1_500_000_000,
                "volume": 1_200_000,
                "pe_ttm_ratio": 38,
                "total_market_val": 180_000_000_000,
            },
            technical={
                "trend": "uptrend",
                "relative_strength_60d": 92,
                "return_60d": 0.74,
                "daily_kline_rows": 120,
            },
            liquidity={
                "turnover": 1_500_000_000,
                "spread_bps": 4.2,
                "liquidity_score": 19.0,
            },
            options={
                "has_option_data": True,
                "status": "ok",
                "iv_rank_proxy": 83,
                "contracts_sampled": 40,
                "put_candidates": [{"code": "US.SNDK260619P01000000"}],
            },
            data_asof={"quote": "2026-05-20 09:30:00", "kline": "2026-05-20T01:30:00Z"},
        ),
    )

    assert candidate.futu_data.quote is not None
    assert candidate.futu_data.quote.last_price == 1200
    assert candidate.futu_data.quote.pe_ttm_ratio == 38
    assert candidate.futu_data.technical is not None
    assert candidate.futu_data.technical.return_60d == 0.74
    assert candidate.futu_data.liquidity is not None
    assert candidate.futu_data.liquidity.spread_bps == 4.2
    assert candidate.futu_data.options is not None
    assert candidate.futu_data.options.iv_rank_proxy == 83
    assert candidate.futu_data.options.put_candidates[0]["code"] == "US.SNDK260619P01000000"
    dumped = candidate.model_dump(mode="json")
    assert "futu_data" in dumped
    assert "market_snapshot" not in dumped
    assert "technical_summary" not in dumped
    assert "liquidity_context" not in dumped
    assert "options_surface" not in dumped
    assert "last_price" not in dumped


def test_real_market_data_adapter_has_no_static_theme_seed_table():
    assert not hasattr(MarketDataAdapter, "_THEME_SEEDS")
    assert not hasattr(ia_market_artifacts, "_EXPECTED_EXPOSURES")


def test_theme_discovery_prompt_is_provider_and_theme_agnostic():
    instructions = theme_discovery._DISCOVERY_INSTRUCTIONS

    assert "Futu" not in instructions
    assert "SEC" not in instructions
    assert "US AI-specific" not in instructions
    assert "US.SNDK" not in instructions
    assert "US.COHR" not in instructions
    assert "US.MRVL" not in instructions
    assert not hasattr(theme_discovery, "_AI_US_COVERAGE_REQUIREMENTS")


def test_discovery_v1_prompt_keeps_distinct_bottleneck_branches_separate():
    instructions = discovery_v1._DISCOVERY_V1_INSTRUCTIONS

    assert "Keep materially different bottleneck branches separate" in instructions
    assert "memory/storage" in instructions
    assert "power/cooling" in instructions
    assert "optical/networking" in instructions


def test_theme_discovery_plan_is_ai_authored_and_keeps_required_symbols(monkeypatch):
    def fake_theme_agent(theme, market, theme_description, required_symbols):
        return ThemeDiscoveryPlan(
            theme=theme,
            market=market,
            theme_description=theme_description,
            initial_thesis="AI data-center bottlenecks include accelerators, memory, storage, optics, power, and cloud.",
            domain_tree=_ai_domain_tree(),
            coverage_requirements=_ai_coverage_requirements(),
            seed_symbols=[
                ThemeDiscoverySeed(symbol="NVDA", market=market, role="required compute leader", source_ids=["src_ai_compute"]),
            ],
            plate_keywords=["人工智能", "数据中心"],
            research_trace=_ai_research_trace(),
            search_queries=["AI data center bottleneck candidates"],
            data_asof={"research": "2026-05-20T00:00:00+00:00"},
            pydantic_ai={"mode": "pydantic_ai_theme_discovery", "mock": True},
        )

    monkeypatch.setattr(theme_discovery, "_run_pydantic_theme_agent", fake_theme_agent)

    plan = theme_discovery.build_theme_discovery_plan(
        "ai",
        market="US",
        theme_description="AI target map",
        required_symbols=["US.NVDA"],
    )

    assert plan.market == "US"
    assert plan.pydantic_ai["mode"] == "pydantic_ai_theme_discovery"
    assert [seed.symbol for seed in plan.seed_symbols] == [
        "US.NVDA",
        "US.MRVL",
        "US.MU",
        "US.SNDK",
        "US.COHR",
        "US.TSM",
        "US.VRT",
        "US.MSFT",
    ]
    assert plan.domain_tree[1].subdomains[1].candidates[0].symbol == "US.SNDK"
    assert {requirement.key for requirement in plan.coverage_requirements} >= {
        "essd_nand_storage",
        "optical_networking_components",
        "custom_silicon_connectivity",
    }
    assert plan.plate_keywords == ["人工智能", "数据中心"]
    assert plan.research_trace[0].source_id == "src_ai_compute"
    assert plan.seed_symbols[0].source_ids == ["src_ai_compute"]


def test_theme_discovery_agent_uses_research_capabilities(monkeypatch):
    calls = []

    class FakeAgent:
        def run_sync(self, prompt, **kwargs):
            class Result:
                output = ThemeDiscoveryPlan(
                    theme="storage",
                    market="US",
                    initial_thesis="Storage discovery thesis.",
                    coverage_requirements=[
                        ThemeCoverageRequirement(
                            key="storage",
                            name="Storage",
                            thesis="Storage is the required exposure.",
                            candidate_symbols=["SNDK"],
                        )
                    ],
                    seed_symbols=[
                        ThemeDiscoverySeed(symbol="SNDK", market="US", role="storage"),
                    ],
                    warnings=["model assumption pending downstream validation"],
                )

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(), {"model": "test-model", "base_url": "https://example.com"}, {
            "available": True,
            "mode": "pydantic_ai_theme_discovery_research_agent",
            "capabilities": {"enabled": True},
        }

    monkeypatch.setattr(theme_discovery, "create_pydantic_agent", fake_create_agent)

    plan = theme_discovery._run_pydantic_theme_agent(
        theme="storage",
        market="US",
        theme_description="",
        required_symbols=[],
    )

    assert plan.theme == "storage"
    assert calls[0]["enable_web_search"] is True
    assert calls[0]["enable_web_fetch"] is True
    assert calls[0]["agent_kind"] == "theme_discovery_research_agent"
    assert calls[0]["agent_skill_names"] == ["theme-discovery"]


def test_futu_assisted_theme_discovery_registers_futu_tool(monkeypatch):
    calls = []
    tool_calls = []

    class FakeExplorer:
        def explore(self, **kwargs):
            tool_calls.append(kwargs)
            return {
                "theme": kwargs["theme"],
                "market": kwargs["market"],
                "generated_at": "2026-05-20T00:00:00+00:00",
                "tool": "futu_explore_theme_candidates",
                "plate_keywords": kwargs["plate_keywords"],
                "plate_matches": [
                    {
                        "keyword": "robotics",
                        "plate_code": "US.LIST2653",
                        "plate_name": "Robotics",
                        "sample_symbols": ["US.ISRG", "US.ROK"],
                    }
                ],
                "candidates": [
                    {
                        "symbol": "US.ISRG",
                        "name": "Intuitive Surgical",
                        "sources": ["futu_plate:US.LIST2653"],
                        "snapshot": {"last_price": 1.0},
                    }
                ],
                "warnings": [],
            }

    class FakeAgent:
        def __init__(self):
            self.tools = {}

        def tool_plain(self, **tool_kwargs):
            calls.append({"tool_plain": tool_kwargs})

            def decorator(func):
                self.tools[tool_kwargs["name"]] = func
                return func

            return decorator

        def run_sync(self, prompt, **kwargs):
            payload = json.loads(prompt)
            assert "futu_stock_filter_catalog" not in payload
            assert payload["tool_contract"]["must_call_sequence"] == [
                "futu_search_screener_catalog",
                "futu_explore_theme_candidates",
            ]
            assert payload["tool_contract"]["must_search_catalog_before_futu_explore"] is True
            search_result = self.tools["futu_search_screener_catalog"](
                query="robotics",
                market=payload["market"],
                limit=5,
            )
            assert search_result["market"] == "US"
            evidence = self.tools["futu_explore_theme_candidates"](
                theme=payload["theme"],
                market=payload["market"],
                plate_keywords=["robotics"],
                must_check_symbols=["ISRG"],
                max_candidates=20,
            )
            assert evidence["plate_matches"][0]["plate_code"] == "US.LIST2653"

            class Result:
                output = ThemeDiscoveryPlan(
                    theme="robotics",
                    market="US",
                    initial_thesis="Robotics needs surgical robots, industrial automation, and sensing.",
                    domain_tree=[
                        ThemeDomain(
                            key="robotics_systems",
                            name="Robotics systems",
                            thesis="Robot OEMs and deployed systems express the theme directly.",
                            importance="core",
                            subdomains=[
                                ThemeSubdomain(
                                    key="surgical_robotics",
                                    name="Surgical robotics",
                                    thesis="Surgical robots are direct monetized robotics exposure.",
                                    importance="high",
                                    candidates=[
                                        ThemeDomainCandidate(
                                            symbol="ISRG",
                                            role="surgical robotics leader",
                                            rationale="Found in Futu robotics plate evidence.",
                                            priority="must_consider",
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                    coverage_requirements=[
                        ThemeCoverageRequirement(
                            key="surgical_robotics",
                            name="Surgical robotics",
                            thesis="Surgical robots should be evaluated as direct robotics exposure.",
                            candidate_symbols=["ISRG"],
                            must_consider_symbols=["ISRG"],
                        )
                    ],
                    seed_symbols=[
                        ThemeDiscoverySeed(
                            symbol="ISRG",
                            market="US",
                            role="surgical robotics leader",
                            rationale="Futu robotics plate evidence supports direct robotics exposure.",
                            subthemes=["Robotics systems", "Surgical robotics"],
                            value_chain_stage="Surgical robotics",
                            exposure_type="direct operating company",
                            exposure_purity="high",
                            source_ids=["futu_plate_US_LIST2653"],
                            confidence="medium",
                            freshness="fresh",
                        )
                    ],
                    plate_keywords=["robotics"],
                    research_trace=[
                        ResearchSource(
                            source_id="futu_plate_US_LIST2653",
                            title="Futu plate Robotics",
                            source_type="other",
                            summary="Futu robotics plate includes US.ISRG.",
                            symbols=["ISRG"],
                            coverage_keys=["surgical_robotics"],
                        )
                    ],
                    warnings=["Futu plate evidence still requires downstream validation."],
                )

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(), {"model": "test-model", "base_url": "https://example.com"}, {
            "available": True,
            "mode": "pydantic_ai_futu_assisted_theme_discovery_agent",
            "capabilities": {"enabled": True},
        }

    monkeypatch.setattr(futu_theme_discovery, "create_pydantic_agent", fake_create_agent)

    plan = futu_theme_discovery.build_futu_assisted_theme_discovery_plan(
        "robotics",
        market="US",
        theme_description="Build a robotics portfolio map.",
        required_symbols=[],
        explorer=FakeExplorer(),
    )

    assert plan.theme == "robotics"
    assert plan.seed_symbols[0].symbol == "US.ISRG"
    assert tool_calls[0]["plate_keywords"] == ["robotics"]
    assert calls[0]["agent_kind"] == "futu_assisted_theme_discovery_agent"
    assert calls[0]["enable_web_search"] is False
    assert calls[0]["enable_web_fetch"] is False
    assert calls[0]["agent_skill_names"] == ["theme-discovery"]
    tool_names = [call["tool_plain"]["name"] for call in calls[1:3]]
    assert tool_names == ["futu_search_screener_catalog", "futu_explore_theme_candidates"]
    assert "local cached Futu screener options" in calls[1]["tool_plain"]["description"]
    assert "App-style screener catalog" in calls[2]["tool_plain"]["description"]
    assert "所属行业/概念/板块" in calls[2]["tool_plain"]["description"]
    assert "MARKET_VAL" in calls[2]["tool_plain"]["description"]
    assert plan.pydantic_ai["futu_tool_calls"][0]["candidate_count"] == 1
    assert plan.pydantic_ai["futu_catalog_search_calls"][0]["query"] == "robotics"
    assert [item["tool"] for item in plan.pydantic_ai["futu_tool_sequence"]] == [
        "futu_search_screener_catalog",
        "futu_explore_theme_candidates",
    ]
    assert plan.pydantic_ai["discovery_loop"] == "futu_tool_audit_revise"


def test_ai_discovery_v1_registers_file_tools_and_converts_to_theme_plan(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self):
            self.tools = {}

        def tool_plain(self, **tool_kwargs):
            calls.append({"tool_plain": tool_kwargs})

            def decorator(func):
                self.tools[tool_kwargs["name"]] = func
                return func

            return decorator

        def run_sync(self, prompt, **kwargs):
            payload = json.loads(prompt)
            assert payload["output_contract"] == "Return DiscoveryPackage only. No portfolio weights, orders, or trade plan."
            assert payload["required_symbols"] == ["US.QQQ", "US.NVDA"]
            listed = self.tools["list_futu_screener_catalog"](
                path="",
                market=payload["market"],
                max_depth=1,
                limit=20,
            )
            assert listed["status"] == "ok"
            assert any(item["path"] == "markets/US/index.md" for item in listed["entries"])
            read = self.tools["read_futu_screener_catalog"](
                path="markets/US/filters/market_quote.md",
                market=payload["market"],
                max_chars=3000,
            )
            assert "TURNOVER" in read["content"]
            probe = self.tools["run_futu_stock_filter"](
                stock_filter_specs=[
                    {"type": "simple", "stock_field": "MARKET_VAL", "filter_min": 5_000_000_000},
                    {"type": "accumulate", "stock_field": "TURNOVER", "days": 20, "is_no_filter": True},
                ],
                market=payload["market"],
                plate_code="US.LIST23925",
                limit=30,
            )
            assert probe["trace_id"] == "tool_003"

            class Result:
                output = discovery_v1.DiscoveryPackage(
                    theme="ai",
                    market="US",
                    initial_thesis="AI discovery should cover compute, storage, networking, and power bottlenecks.",
                    layers=[
                        discovery_v1.DiscoveryLayer(
                            key="memory_storage",
                            name="Memory and storage",
                            thesis="AI inference and training need memory bandwidth and enterprise storage throughput.",
                            importance="core",
                            candidate_symbols=["US.SNDK"],
                        )
                    ],
                    filter_plans_by_layer=[
                        DiscoveryFilterPlan(
                            layer_key="memory_storage",
                            layer_name="Memory and storage",
                            target_candidate_profile="Liquid memory, NAND, eSSD, and storage beneficiaries.",
                            plate_search_terms=["storage", "memory"],
                            plate_codes_to_probe=["US.LIST23925"],
                            filter_decisions=[
                                DiscoveryFilterDecision(
                                    category="plate",
                                    decision="use_now",
                                    planned_fields=["US.LIST23925"],
                                    rationale="Use the storage concept plate to avoid missing eSSD/NAND names.",
                                ),
                                DiscoveryFilterDecision(
                                    category="financial",
                                    decision="defer_to_later_enrichment",
                                    planned_fields=["SUM_OF_BUSINESS_GROWTH"],
                                    rationale="Exact financial numbers are later enrichment, not final discovery evidence.",
                                ),
                            ],
                        )
                    ],
                    executed_filter_probes=[
                        ExecutedDiscoveryProbe(
                            layer_key="memory_storage",
                            probe_type="subdomain_plate",
                            plate_code="US.LIST23925",
                            stock_filter_specs=[
                                {"type": "simple", "stock_field": "MARKET_VAL", "filter_min": 5_000_000_000},
                                {"type": "accumulate", "stock_field": "TURNOVER", "days": 20, "is_no_filter": True},
                            ],
                            trace_id=probe["trace_id"],
                            result_status=probe["status"],
                            result_count=probe["returned_count"],
                            candidate_symbols=probe["sample_symbols"],
                        )
                    ],
                    layer_filter_audits=[
                        DiscoveryLayerFilterAudit(
                            layer_key="memory_storage",
                            layer_name="Memory and storage",
                            hypothesis="Storage bottlenecks should be probed through storage plates, size, and liquidity.",
                            plate_codes_considered=["US.LIST23925"],
                            plate_codes_used=["US.LIST23925"],
                            candidate_symbols_from_probes=probe["sample_symbols"],
                            result_summary="Storage plate plus market-cap/liquidity filters surfaced SNDK and WDC.",
                        )
                    ],
                    candidates=[
                        discovery_v1.DiscoveryCandidate(
                            symbol="US.SNDK",
                            layer_key="memory_storage",
                            subdomain="eSSD/NAND storage",
                            role="storage bottleneck candidate",
                            rationale="Found by the storage plate probe and should be validated with filings later.",
                            priority="must_consider",
                            source_trace_ids=[probe["trace_id"]],
                        ),
                        discovery_v1.DiscoveryCandidate(
                            symbol="US.WDC",
                            layer_key="memory_storage",
                            subdomain="storage",
                            role="storage candidate",
                            rationale="Found by the storage plate probe and should be validated with filings later.",
                            priority="watchlist",
                            source_trace_ids=[probe["trace_id"]],
                        )
                    ],
                    omissions_to_investigate=[
                        DiscoveryOmission(
                            symbol="US.COHR",
                            layer_key="optical_networking",
                            source_trace_ids=[],
                            exclusion_reason="clear_theme_mismatch",
                            explanation="Synthetic test omission for a non-storage symbol.",
                        )
                    ],
                    next_enrichment_needed=["SEC filings", "Futu market data"],
                    warnings=["Unit-test fake package."],
                )

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(), {"model": "test-model", "base_url": "https://example.com"}, {
            "available": True,
            "mode": "pydantic_ai_theme_discovery_v1_filter_planning_agent",
            "capabilities": {"enabled": True},
        }

    monkeypatch.setattr(discovery_v1, "create_pydantic_agent", fake_create_agent)
    monkeypatch.setattr(
        discovery_v1,
        "_run_futu_filter",
        lambda **kwargs: {
            "status": "ok",
            "returned_count": 2,
            "all_count": 2,
            "sample_symbols": ["US.SNDK", "US.WDC"],
            "sample_candidates": [{"symbol": "US.SNDK"}, {"symbol": "US.WDC"}],
            "diagnosis": "usable",
        },
    )

    plan = discovery_v1.build_ai_discovery_v1_plan(
        "AI target map",
        market="US",
        theme_description="AI版图持仓建设",
        required_symbols=["QQQ", "NVDA"],
    )

    assert plan.theme == "AI target map"
    assert plan.seed_symbols[0].symbol == "US.QQQ"
    assert plan.seed_symbols[1].symbol == "US.NVDA"
    assert any(seed.symbol == "US.SNDK" for seed in plan.seed_symbols)
    assert plan.filter_plans_by_layer[0].layer_key == "memory_storage"
    assert plan.executed_filter_probes[0].candidate_symbols == ["US.SNDK", "US.WDC"]
    assert plan.layer_filter_audits[0].result_summary
    assert plan.omissions_to_investigate[0].symbol == "US.COHR"
    assert plan.pydantic_ai["tool_calls"][2]["tool"] == "run_futu_stock_filter"
    assert calls[0]["agent_kind"] == "theme_discovery_v1_filter_planning_agent"
    assert calls[0]["enable_web_search"] is True
    assert calls[0]["enable_web_fetch"] is True
    assert calls[0]["agent_skill_names"] == ["theme-discovery"]
    assert [call["tool_plain"]["name"] for call in calls[1:4]] == [
        "list_futu_screener_catalog",
        "read_futu_screener_catalog",
        "run_futu_stock_filter",
    ]


def test_websearch_discovery_uses_budgeted_tool_and_converts_to_theme_plan(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self):
            self.tools = {}

        def tool_plain(self, **tool_kwargs):
            calls.append({"tool_plain": tool_kwargs})

            def decorator(func):
                self.tools[tool_kwargs["name"]] = func
                return func

            return decorator

        def run_sync(self, prompt, **kwargs):
            payload = json.loads(prompt)
            assert payload["search_budget"]["hard_limit"] is True
            first = self.tools["web_search_budgeted"](
                query="AI storage eSSD SanDisk spin off public ticker",
                layer_key="memory_storage",
                rationale="Find AI storage bottleneck candidates.",
                max_results=5,
            )
            second = self.tools["web_search_budgeted"](
                query="AI optical transceivers public companies",
                layer_key="optical_networking",
                rationale="This should hit the hard budget.",
                max_results=5,
            )
            assert first["status"] == "ok"
            assert second["status"] == "budget_exhausted"

            class Result:
                output = websearch_discovery.WebSearchDiscoveryArtifact(
                    theme="AI target map",
                    market="US",
                    initial_thesis="AI discovery should separate compute, storage, networking, and power bottlenecks.",
                    layers=[
                        websearch_discovery.WebSearchDiscoveryLayer(
                            key="memory_storage",
                            name="Memory and storage",
                            thesis="AI inference needs HBM, DRAM, NAND, and enterprise SSD throughput.",
                            importance="core",
                            candidate_symbols=["US.SNDK", "US.MU"],
                        )
                    ],
                    candidates=[
                        websearch_discovery.WebSearchDiscoveryCandidate(
                            symbol="US.SNDK",
                            layer_key="memory_storage",
                            subdomain="eSSD/NAND",
                            role="must-consider storage pure play",
                            rationale="Search surfaced SanDisk as a direct AI storage candidate.",
                            priority="must_consider",
                            source_trace_ids=[first["trace_id"]],
                            confidence="high",
                        ),
                        websearch_discovery.WebSearchDiscoveryCandidate(
                            symbol="US.MU",
                            layer_key="memory_storage",
                            subdomain="HBM/DRAM",
                            role="memory bandwidth leader",
                            rationale="HBM/DRAM bottleneck exposure.",
                            priority="strong_candidate",
                            source_trace_ids=[first["trace_id"]],
                        ),
                    ],
                    search_queries_used=["AI storage eSSD SanDisk spin off public ticker"],
                    next_enrichment_needed=["Futu lightweight market data", "SEC filings"],
                )

                usage = None

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(), {"model": "test-model", "base_url": "https://example.com"}, {
            "available": True,
            "mode": "pydantic_ai_websearch_theme_discovery_agent",
            "capabilities": {"enabled": True},
        }

    monkeypatch.setattr(websearch_discovery, "create_pydantic_agent", fake_create_agent)
    monkeypatch.setattr(
        websearch_discovery,
        "_run_duckduckgo_search",
        lambda query, max_results: [
            {
                "title": "SanDisk AI storage coverage",
                "url": "https://example.com/sndk",
                "snippet": "SanDisk storage candidate.",
            }
        ],
    )

    plan = websearch_discovery.build_websearch_discovery_plan(
        "AI target map",
        market="US",
        theme_description="AI版图持仓建设",
        required_symbols=["QQQ", "NVDA"],
        max_searches=1,
        max_results=5,
    )

    assert calls[0]["agent_kind"] == "websearch_theme_discovery_agent"
    assert calls[0]["enable_web_search"] is False
    assert calls[0]["enable_web_fetch"] is False
    assert calls[0]["agent_skill_names"] is None
    assert [call["tool_plain"]["name"] for call in calls[1:]] == ["web_search_budgeted"]
    assert plan.seed_symbols[0].symbol == "US.QQQ"
    assert plan.seed_symbols[1].symbol == "US.NVDA"
    assert any(seed.symbol == "US.SNDK" for seed in plan.seed_symbols)
    assert plan.filter_plans_by_layer == []
    assert plan.executed_filter_probes == []
    assert plan.pydantic_ai["mode"] == "pydantic_ai_budgeted_websearch_discovery"
    assert plan.pydantic_ai["search_budget"]["successful_searches"] == 1
    assert [call["result"]["status"] for call in plan.pydantic_ai["tool_calls"]] == [
        "ok",
        "budget_exhausted",
    ]
    assert plan.research_trace[0].source_id == "web_001"


def test_two_stage_websearch_discovery_executes_planned_searches(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, output_type):
            self.output_type = output_type

        def tool_plain(self, **_tool_kwargs):
            pytest.fail("Two-stage websearch discovery should not register agent tools.")

        def run_sync(self, prompt, **kwargs):
            payload = json.loads(prompt)
            if self.output_type is websearch_discovery.WebSearchDiscoverySearchPlan:
                assert payload["task"] == "two_stage_websearch_theme_discovery_search_plan"

                class Result:
                    output = websearch_discovery.WebSearchDiscoverySearchPlan(
                        theme="AI target map",
                        market="US",
                        initial_thesis="AI infrastructure discovery needs planned branch coverage.",
                        layers=[
                            websearch_discovery.WebSearchDiscoveryLayer(
                                key="memory_storage",
                                name="Memory and storage",
                                importance="core",
                            ),
                            websearch_discovery.WebSearchDiscoveryLayer(
                                key="optical_networking",
                                name="Optical networking",
                                importance="important",
                            ),
                            websearch_discovery.WebSearchDiscoveryLayer(
                                key="power_cooling",
                                name="Power and cooling",
                                importance="important",
                            ),
                        ],
                        search_tasks=[
                            websearch_discovery.WebSearchDiscoverySearchTask(
                                task_id="broad_map",
                                layer_key="memory_storage",
                                layer_name="Memory and storage",
                                query="AI infrastructure public company map",
                                priority="optional",
                                omission_risk="low",
                            ),
                            websearch_discovery.WebSearchDiscoverySearchTask(
                                task_id="storage_bottleneck",
                                layer_key="memory_storage",
                                layer_name="Memory and storage",
                                branch="eSSD NAND HBM DRAM",
                                query="AI data center eSSD NAND HBM DRAM public companies SanDisk WDC MU",
                                priority="important",
                                omission_risk="high",
                            ),
                            websearch_discovery.WebSearchDiscoverySearchTask(
                                task_id="optical_bottleneck",
                                layer_key="optical_networking",
                                layer_name="Optical networking",
                                branch="optical transceivers interconnect",
                                query="AI data center optical transceiver public companies LITE COHR CRDO MRVL",
                                priority="required",
                                omission_risk="high",
                            ),
                            websearch_discovery.WebSearchDiscoverySearchTask(
                                task_id="power_bottleneck",
                                layer_key="power_cooling",
                                layer_name="Power and cooling",
                                query="AI data center power cooling public companies VRT CEG GEV",
                                priority="important",
                                omission_risk="medium",
                            ),
                        ],
                    )

                    usage = None

                return Result()

            assert self.output_type is websearch_discovery.WebSearchDiscoveryArtifact
            assert payload["task"] == "two_stage_websearch_theme_discovery_synthesis"
            executed = payload["executed_searches"]
            assert [item["args"]["task_id"] for item in executed] == [
                "optical_bottleneck",
                "storage_bottleneck",
            ]
            assert payload["skipped_search_tasks"][0]["task_id"] == "power_bottleneck"

            class Result:
                output = websearch_discovery.WebSearchDiscoveryArtifact(
                    theme="AI target map",
                    market="US",
                    initial_thesis="Planned search separates storage, optical networking, and power.",
                    layers=[
                        websearch_discovery.WebSearchDiscoveryLayer(
                            key="memory_storage",
                            name="Memory and storage",
                            thesis="AI inference stresses HBM, DRAM, NAND, and enterprise SSD throughput.",
                            importance="core",
                            candidate_symbols=["US.SNDK", "US.MU"],
                        ),
                        websearch_discovery.WebSearchDiscoveryLayer(
                            key="optical_networking",
                            name="Optical networking",
                            thesis="AI clusters need optical interconnect and transceiver capacity.",
                            importance="important",
                            candidate_symbols=["US.LITE", "US.COHR"],
                        ),
                    ],
                    candidates=[
                        websearch_discovery.WebSearchDiscoveryCandidate(
                            symbol="SNDK",
                            layer_key="memory_storage",
                            subdomain="persistent storage",
                            role="AI storage candidate",
                            rationale="Planned storage search surfaced SanDisk.",
                            priority="must_consider",
                            source_trace_ids=["web_002"],
                        ),
                        websearch_discovery.WebSearchDiscoveryCandidate(
                            symbol="LITE",
                            layer_key="optical_networking",
                            subdomain="optical components",
                            role="optical networking candidate",
                            rationale="Planned optical search surfaced Lumentum.",
                            priority="strong_candidate",
                            source_trace_ids=["web_001"],
                        ),
                    ],
                    search_queries_used=[
                        "AI data center eSSD NAND HBM DRAM public companies SanDisk WDC MU",
                        "AI data center optical transceiver public companies LITE COHR CRDO MRVL",
                    ],
                )

                usage = None

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(kwargs["output_type"]), {"model": "test-model"}, {
            "available": True,
            "mode": f"pydantic_ai_{kwargs['agent_kind']}",
        }

    def fake_search(query, max_results):
        return [
            {
                "title": f"result for {query}",
                "url": "https://example.com/search",
                "snippet": "search result",
            }
        ][:max_results]

    monkeypatch.setattr(websearch_discovery, "create_pydantic_agent", fake_create_agent)
    monkeypatch.setattr(websearch_discovery, "_run_duckduckgo_search", fake_search)

    plan = websearch_discovery.build_two_stage_websearch_discovery_plan(
        "AI target map",
        market="US",
        theme_description="AI版图持仓建设",
        required_symbols=["QQQ", "SOXX", "NVDA"],
        max_searches=2,
        max_results=3,
    )

    assert [call["agent_kind"] for call in calls] == [
        "websearch_theme_discovery_search_planner",
        "websearch_theme_discovery_synthesizer",
    ]
    assert all(call["enable_web_search"] is False for call in calls)
    assert all(call["enable_web_fetch"] is False for call in calls)
    assert plan.pydantic_ai["mode"] == "pydantic_ai_two_stage_websearch_discovery"
    assert plan.pydantic_ai["search_budget"]["planned_task_count"] == 4
    assert plan.pydantic_ai["search_budget"]["successful_searches"] == 2
    assert plan.pydantic_ai["search_budget"]["skipped_task_count"] == 2
    assert [call["args"]["task_id"] for call in plan.pydantic_ai["tool_calls"]] == [
        "optical_bottleneck",
        "storage_bottleneck",
    ]
    assert any(seed.symbol == "US.SNDK" for seed in plan.seed_symbols)
    assert any(seed.symbol == "US.LITE" for seed in plan.seed_symbols)


def test_native_websearch_discovery_uses_native_capability_overrides(monkeypatch):
    calls = []

    class FakeAgent:
        def run_sync(self, prompt, **kwargs):
            payload = json.loads(prompt)
            assert payload["native_web_search_budget"]["max_uses"] == 3
            assert "event_stream_handler" in kwargs

            class Result:
                output = websearch_discovery.WebSearchDiscoveryArtifact(
                    theme="AI target map",
                    market="US",
                    initial_thesis="Native websearch discovery should keep output compact.",
                    layers=[
                        websearch_discovery.WebSearchDiscoveryLayer(
                            key="memory_storage",
                            name="Memory and storage",
                            thesis="AI inference needs HBM, DRAM, NAND, and enterprise SSD throughput.",
                            importance="core",
                            candidate_symbols=["US.SNDK", "US.MU"],
                        )
                    ],
                    candidates=[
                        websearch_discovery.WebSearchDiscoveryCandidate(
                            symbol="SNDK",
                            layer_key="memory_storage",
                            subdomain="eSSD/NAND",
                            role="native-search storage candidate",
                            rationale="Native search surfaced SanDisk as AI storage candidate.",
                            priority="must_consider",
                            source_trace_ids=["native_websearch"],
                        )
                    ],
                    search_queries_used=["SanDisk AI enterprise SSD public ticker SNDK"],
                )

                usage = None

            return Result()

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return FakeAgent(), {"model": "test-model"}, {
            "available": True,
            "mode": f"pydantic_ai_{kwargs['agent_kind']}",
        }

    monkeypatch.setattr(websearch_discovery, "create_pydantic_agent", fake_create_agent)

    plan = websearch_discovery.build_native_websearch_discovery_plan(
        "AI target map",
        market="US",
        theme_description="AI版图持仓建设",
        required_symbols=["QQQ", "NVDA"],
        max_searches=3,
    )

    assert calls[0]["agent_kind"] == "native_websearch_theme_discovery_agent"
    assert calls[0]["enable_web_search"] is True
    assert calls[0]["enable_web_fetch"] is False
    assert calls[0]["research_overrides"]["web_search_mode"] == "native"
    assert calls[0]["research_overrides"]["max_searches"] == 3
    assert calls[0]["research_overrides"]["web_search_context_size"] == "low"
    assert plan.pydantic_ai["mode"] == "pydantic_ai_native_websearch_discovery"
    assert any(seed.symbol == "US.SNDK" for seed in plan.seed_symbols)
    assert plan.pydantic_ai["tool_calls"][0]["tool"] == "web_search"
    assert plan.research_trace[0].source_id == "native_web_001"


def test_websearch_discovery_rejects_non_ascii_layer_symbols():
    with pytest.raises(ValueError, match="Invalid layer candidate symbol"):
        websearch_discovery.WebSearchDiscoveryArtifact(
            theme="ai",
            market="US",
            initial_thesis="AI discovery thesis.",
            layers=[
                websearch_discovery.WebSearchDiscoveryLayer(
                    key="recent_listings",
                    name="Recent listings",
                    candidate_symbols=["US.ALĐAB"],
                )
            ],
            candidates=[
                websearch_discovery.WebSearchDiscoveryCandidate(
                    symbol="US.ALAB",
                    layer_key="recent_listings",
                    role="AI infrastructure listing",
                )
            ],
        )


def test_native_websearch_trace_prefers_builtin_tool_calls():
    events = [
        {
            "event": "part_delta",
            "time": "2026-01-01T00:00:00+00:00",
            "delta_preview": '{"query":"AI storage","queries":["AI storage","AI optical"]}',
        },
        {
            "event": "builtin_tool_call",
            "time": "2026-01-01T00:00:01+00:00",
            "tool": "web_search",
            "args": '{"query":"AI storage","queries":["AI storage","AI optical"]}',
        },
    ]

    calls = websearch_discovery._native_search_calls_from_events(events)

    assert len(calls) == 1
    assert calls[0]["args"]["source_event"] == "builtin_tool_call"
    assert calls[0]["args"]["queries"] == ["AI storage", "AI optical"]


def test_websearch_discovery_prompts_warn_against_unicode_tickers():
    prompts = [
        websearch_discovery._TWO_STAGE_SYNTHESIS_INSTRUCTIONS,
        websearch_discovery._NATIVE_WEBSEARCH_DISCOVERY_INSTRUCTIONS,
    ]

    for prompt in prompts:
        assert "ASCII-only" in prompt
        assert "Unicode" in prompt
        assert "US.SMCI" in prompt
        assert "US.SMCİ" in prompt


def test_native_websearch_prompt_requires_public_market_change_sweep():
    prompt = websearch_discovery._NATIVE_WEBSEARCH_DISCOVERY_INSTRUCTIONS

    assert "public-market-change omission sweep" in prompt
    assert "spin-offs" in prompt
    assert "ticker changes" in prompt
    assert "newly public" in prompt
    assert "Do not assume a parent company" in prompt
    assert "Do not satisfy the public-market-change sweep with one global query only" in prompt
    assert "first generate a compact layer vocabulary" in prompt
    assert "core technologies" in prompt
    assert "product categories" in prompt
    assert "supply-chain bottlenecks" in prompt
    assert "customer or\n  use-case terms" in prompt
    assert "hardcoding theme-specific keywords or tickers" in prompt
    assert "NAND, eSSD, SSD, HDD" not in prompt
    assert "do not prematurely compress the candidate universe" in prompt
    assert "Favor recall over precision" in prompt
    assert "Do not compress away plausible public candidates" in prompt
    assert "8-20 candidates per important layer" in prompt


def test_discovery_v1_preview_preserves_required_symbols_as_candidate_source():
    output = discovery_v1.DiscoveryPreviewPackage(
        theme="AI target map",
        market="US",
        initial_thesis="Preview test.",
        layers=[
            discovery_v1.DiscoveryPreviewLayer(
                key="core_beta",
                name="Core beta",
                importance="core",
                economic_mechanism="User-required core exposure.",
                exposure_types=["ETF", "GPU"],
                filters_used=["required_symbols input"],
            )
        ],
        candidates=[
            discovery_v1.DiscoveryPreviewCandidate(
                symbol="QQQ",
                layer_key="core_beta",
                role="user-required broad tech ETF",
                priority="strong_candidate",
                rationale="Model included this required symbol.",
            ),
            discovery_v1.DiscoveryPreviewCandidate(
                symbol="SOXX",
                layer_key="core_beta",
                role="user-required semiconductor ETF",
                priority="strong_candidate",
                rationale="Model included this required symbol.",
            ),
        ],
    )

    preview = discovery_v1._preview_payload_with_constraints(
        output,
        market="US",
        required_symbols=["US.QQQ", "US.SOXX", "US.NVDA"],
    )

    candidates_by_symbol = {item["symbol"]: item for item in preview["candidates"]}
    assert set(candidates_by_symbol) >= {"US.QQQ", "US.SOXX", "US.NVDA"}
    assert candidates_by_symbol["US.QQQ"]["priority"] == "must_consider"
    assert candidates_by_symbol["US.SOXX"]["priority"] == "must_consider"
    assert candidates_by_symbol["US.NVDA"]["priority"] == "must_consider"
    assert "candidate_symbols" not in preview["layers"][0]
    assert "US.NVDA" in {
        symbol
        for symbols in preview["candidates_by_layer"].values()
        for symbol in symbols
    }
    assert any("required_symbols" in warning for warning in preview["warnings"])


def test_discovery_v1_preview_rejects_malformed_candidate_symbol():
    with pytest.raises(ValueError, match="Invalid candidate symbol"):
        discovery_v1.DiscoveryPreviewCandidate(
            symbol="US.AL A B",
            layer_key="optical",
            role="malformed symbol",
            rationale="Should be rejected before artifact persistence.",
        )


def test_discovery_v1_persists_research_trace_and_web_search_queries():
    package = discovery_v1.DiscoveryPackage(
        theme="ai",
        market="US",
        initial_thesis="AI discovery should use web research and Futu probes.",
        layers=[
            discovery_v1.DiscoveryLayer(
                key="research_layer",
                name="Research layer",
                thesis="Optional layer for research trace test.",
                importance="optional",
                candidate_symbols=["US.NVDA"],
            )
        ],
        research_trace=[
            ResearchSource(
                source_id="model_src",
                title="Model-authored research source",
                summary="Source returned directly by the discovery package.",
            )
        ],
        search_queries=["model supplied query"],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="cross_layer",
                probe_type="web",
                trace_id="web_search: turn0search0",
                rationale="Web probe changed the theme map toward data-center bottlenecks.",
                result_status="usable",
            )
        ],
        candidates=[
            discovery_v1.DiscoveryCandidate(
                symbol="US.NVDA",
                layer_key="research_layer",
                role="AI accelerator candidate",
                rationale="Required for test.",
                priority="must_consider",
            )
        ],
    )

    plan = discovery_v1._theme_discovery_plan_from_package(
        package,
        theme="AI target map",
        market="US",
        theme_description="AI版图持仓建设",
        required_symbols=[],
        pydantic_ai={
            "events": [
                {
                    "time": "2026-06-03T16:43:45+00:00",
                    "event": "builtin_tool_call",
                    "tool": "web_search",
                    "call_id": "ws_test",
                    "args": json.dumps(
                        {
                            "query": "AI data center optical networking 2026",
                            "queries": [
                                "AI data center optical networking 2026",
                                "AI eSSD storage demand 2026",
                            ],
                        }
                    ),
                },
                {
                    "time": "2026-06-03T16:43:46+00:00",
                    "event": "builtin_tool_result",
                    "tool": "web_search",
                    "call_id": "ws_test",
                    "content_preview": json.dumps(
                        {
                            "results": [
                                {
                                    "title": "AI networking bottlenecks",
                                    "url": "https://example.com/ai-networking",
                                    "snippet": "Optical networking and AI data centers.",
                                    "publisher": "Example Research",
                                }
                            ]
                        }
                    ),
                },
            ]
        },
    )

    assert plan.search_queries == [
        "model supplied query",
        "AI data center optical networking 2026",
        "AI eSSD storage demand 2026",
    ]
    source_ids = {source.source_id for source in plan.research_trace}
    assert "model_src" in source_ids
    assert "ws_test:result_1" in source_ids
    assert "web_search: turn0search0" in source_ids
    web_source = next(source for source in plan.research_trace if source.source_id == "ws_test:result_1")
    assert web_source.url == "https://example.com/ai-networking"
    assert web_source.summary == "Optical networking and AI data centers."


def test_discovery_v1_requires_layer_plate_hits_to_be_candidates():
    with pytest.raises(ValueError, match="US.SNDK"):
        discovery_v1.DiscoveryPackage(
            theme="ai",
            market="US",
            initial_thesis="AI storage bottlenecks should be discovered.",
            layers=[
                discovery_v1.DiscoveryLayer(
                    key="memory_storage",
                    name="Memory and storage",
                    thesis="AI inference needs memory bandwidth and storage throughput.",
                    importance="important",
                    candidate_symbols=["US.SNDK", "US.MU"],
                )
            ],
            filter_plans_by_layer=[
                DiscoveryFilterPlan(
                    layer_key="memory_storage",
                    layer_name="Memory and storage",
                    target_candidate_profile="Liquid storage and memory candidates.",
                    plate_codes_to_probe=["US.LIST23925"],
                    filter_decisions=[
                        DiscoveryFilterDecision(
                            category="plate",
                            decision="use_now",
                            planned_fields=["US.LIST23925"],
                            rationale="Use the exact storage concept plate.",
                        )
                    ],
                )
            ],
            executed_filter_probes=[
                ExecutedDiscoveryProbe(
                    layer_key="memory_storage",
                    probe_type="subdomain_plate",
                    plate_code="US.LIST23925",
                    trace_id="tool_025",
                    result_status="usable",
                    result_count=2,
                    candidate_symbols=["US.SNDK", "US.MU"],
                )
            ],
            candidates=[
                discovery_v1.DiscoveryCandidate(
                    symbol="US.MU",
                    layer_key="memory_storage",
                    subdomain="HBM/DRAM",
                    role="memory candidate",
                    rationale="Found by storage probe.",
                    priority="strong_candidate",
                    source_trace_ids=["tool_025"],
                )
            ],
            omissions_to_investigate=[],
        )


def test_discovery_v1_requires_probe_hits_even_when_layer_symbols_omit_them():
    with pytest.raises(ValueError, match="US.SNDK"):
        discovery_v1.DiscoveryPackage(
            theme="ai",
            market="US",
            initial_thesis="AI storage bottlenecks should be discovered.",
            layers=[
                discovery_v1.DiscoveryLayer(
                    key="memory_storage",
                    name="Memory and storage",
                    thesis="AI inference needs memory bandwidth and storage throughput.",
                    importance="important",
                    candidate_symbols=["US.MU"],
                )
            ],
            filter_plans_by_layer=[
                DiscoveryFilterPlan(
                    layer_key="memory_storage",
                    layer_name="Memory and storage",
                    target_candidate_profile="Liquid storage and memory candidates.",
                    plate_codes_to_probe=["US.LIST23925"],
                    filter_decisions=[
                        DiscoveryFilterDecision(
                            category="plate",
                            decision="use_now",
                            planned_fields=["US.LIST23925"],
                            rationale="Use the exact storage concept plate.",
                        )
                    ],
                )
            ],
            executed_filter_probes=[
                ExecutedDiscoveryProbe(
                    layer_key="memory_storage",
                    probe_type="subdomain_plate",
                    plate_code="US.LIST23925",
                    trace_id="tool_025",
                    result_status="usable",
                    result_count=2,
                    candidate_symbols=["US.SNDK", "US.MU"],
                )
            ],
            candidates=[
                discovery_v1.DiscoveryCandidate(
                    symbol="US.MU",
                    layer_key="memory_storage",
                    subdomain="HBM/DRAM",
                    role="memory candidate",
                    rationale="Found by storage probe.",
                    priority="strong_candidate",
                    source_trace_ids=["tool_025"],
                )
            ],
            omissions_to_investigate=[],
        )


def test_discovery_v1_allows_probe_hit_with_structured_hard_omission():
    package = discovery_v1.DiscoveryPackage(
        theme="ai",
        market="US",
        initial_thesis="AI storage bottlenecks should be discovered.",
        layers=[
            discovery_v1.DiscoveryLayer(
                key="memory_storage",
                name="Memory and storage",
                thesis="AI inference needs memory bandwidth and storage throughput.",
                importance="important",
                candidate_symbols=["US.MU"],
            )
        ],
        filter_plans_by_layer=[
            DiscoveryFilterPlan(
                layer_key="memory_storage",
                layer_name="Memory and storage",
                target_candidate_profile="Liquid storage and memory candidates.",
                plate_codes_to_probe=["US.LIST23925"],
                filter_decisions=[
                    DiscoveryFilterDecision(
                        category="plate",
                        decision="use_now",
                        planned_fields=["US.LIST23925"],
                        rationale="Use the exact storage concept plate.",
                    )
                ],
            )
        ],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="memory_storage",
                probe_type="subdomain_plate",
                plate_code="US.LIST23925",
                trace_id="tool_025",
                result_status="usable",
                result_count=2,
                candidate_symbols=["US.SNDK", "US.MU"],
            )
        ],
        candidates=[
            discovery_v1.DiscoveryCandidate(
                symbol="US.MU",
                layer_key="memory_storage",
                subdomain="HBM/DRAM",
                role="memory candidate",
                rationale="Found by storage probe.",
                priority="strong_candidate",
                source_trace_ids=["tool_025"],
            )
        ],
        omissions_to_investigate=[
            DiscoveryOmission(
                symbol="US.SNDK",
                layer_key="memory_storage",
                source_trace_ids=["tool_025"],
                exclusion_reason="unsupported_security_type",
                explanation="Example hard exclusion for test coverage.",
            )
        ],
    )

    assert package.omissions_to_investigate[0].symbol == "US.SNDK"


def test_discovery_v1_does_not_require_every_large_probe_hit():
    package = discovery_v1.DiscoveryPackage(
        theme="ai",
        market="US",
        initial_thesis="AI software probes can return a broad universe.",
        layers=[
            discovery_v1.DiscoveryLayer(
                key="ai_software",
                name="AI software",
                thesis="Broad AI software application layer.",
                importance="important",
                candidate_symbols=["US.PLTR"],
            )
        ],
        filter_plans_by_layer=[
            DiscoveryFilterPlan(
                layer_key="ai_software",
                layer_name="AI software",
                target_candidate_profile="Liquid AI software candidates.",
                plate_codes_to_probe=["US.LIST23492"],
                filter_decisions=[
                    DiscoveryFilterDecision(
                        category="plate",
                        decision="use_now",
                        planned_fields=["US.LIST23492"],
                        rationale="Use AI application software plate.",
                    )
                ],
            )
        ],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="ai_software",
                probe_type="subdomain_plate",
                plate_code="US.LIST23492",
                trace_id="tool_025",
                result_status="usable",
                result_count=20,
                candidate_symbols=[
                    "US.PLTR",
                    "US.SNOW",
                    "US.DDOG",
                    "US.NET",
                    "US.RDDT",
                    "US.DOCU",
                    "US.ZM",
                ],
            )
        ],
        candidates=[
            discovery_v1.DiscoveryCandidate(
                symbol="US.PLTR",
                layer_key="ai_software",
                subdomain="AI application software",
                role="AI platform candidate",
                rationale="Found by AI software probe.",
                priority="strong_candidate",
                source_trace_ids=["tool_025"],
            )
        ],
        omissions_to_investigate=[],
    )

    assert package.candidates[0].symbol == "US.PLTR"


def test_discovery_filter_audit_category_is_prompt_guided_not_schema_limited():
    audit = DiscoveryLayerFilterAudit(
        layer_key="ai_software",
        used_filters=[
            DiscoveryFilterAuditItem(
                category="momentum_and_liquidity",
                decision="used",
                stock_fields=["TURNOVER", "MA_ALIGNMENT_LONG"],
                filter_summary="Use liquidity and trend filters as a discovery aid.",
            )
        ],
    )

    assert audit.used_filters[0].category == "momentum_and_liquidity"


def test_discovery_warnings_flag_broad_plate_without_refinement():
    package = discovery_v1.DiscoveryPackage(
        theme="ai",
        market="US",
        initial_thesis="AI software probes can return a broad universe.",
        layers=[
            discovery_v1.DiscoveryLayer(
                key="ai_software",
                name="AI software",
                thesis="Broad AI software application layer.",
                importance="important",
                candidate_symbols=["US.PLTR"],
            )
        ],
        filter_plans_by_layer=[
            DiscoveryFilterPlan(
                layer_key="ai_software",
                layer_name="AI software",
                target_candidate_profile="Liquid AI software candidates.",
                plate_codes_to_probe=["US.LIST23492"],
                filter_decisions=[
                    DiscoveryFilterDecision(
                        category="plate",
                        decision="use_now",
                        planned_fields=["US.LIST23492"],
                        rationale="Use AI application software plate.",
                    )
                ],
            )
        ],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="ai_software",
                probe_type="subdomain_plate",
                plate_code="US.LIST23492",
                trace_id="tool_025",
                result_status="usable",
                result_count=27,
                candidate_symbols=["US.PLTR", "US.SNOW", "US.DDOG"],
            )
        ],
        candidates=[
            discovery_v1.DiscoveryCandidate(
                symbol="US.PLTR",
                layer_key="ai_software",
                subdomain="AI application software",
                role="AI platform candidate",
                rationale="Found by AI software probe.",
                priority="strong_candidate",
                source_trace_ids=["tool_025"],
            )
        ],
    )

    warnings = discovery_v1._discovery_warnings(package)

    assert any("same-plate refinement" in warning for warning in warnings)


def test_discovery_warnings_accept_broad_plate_with_refinement():
    package = discovery_v1.DiscoveryPackage(
        theme="ai",
        market="US",
        initial_thesis="AI software probes can return a broad universe.",
        layers=[
            discovery_v1.DiscoveryLayer(
                key="ai_software",
                name="AI software",
                thesis="Broad AI software application layer.",
                importance="important",
                candidate_symbols=["US.PLTR"],
            )
        ],
        filter_plans_by_layer=[
            DiscoveryFilterPlan(
                layer_key="ai_software",
                layer_name="AI software",
                target_candidate_profile="Liquid AI software candidates.",
                plate_codes_to_probe=["US.LIST23492"],
                filter_decisions=[
                    DiscoveryFilterDecision(
                        category="plate",
                        decision="use_now",
                        planned_fields=["US.LIST23492"],
                        rationale="Use AI application software plate.",
                    )
                ],
            )
        ],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="ai_software",
                probe_type="subdomain_plate",
                plate_code="US.LIST23492",
                trace_id="tool_025",
                result_status="usable",
                result_count=27,
                candidate_symbols=["US.PLTR", "US.SNOW", "US.DDOG"],
            ),
            ExecutedDiscoveryProbe(
                layer_key="ai_software",
                probe_type="subdomain_plate_refinement",
                plate_code="US.LIST23492",
                trace_id="tool_026",
                result_status="usable",
                result_count=8,
                stock_filter_specs=[
                    {"type": "simple", "stock_field": "MARKET_VAL", "filter_min": 10_000_000_000},
                    {"type": "accumulate", "stock_field": "TURNOVER", "days": 20, "filter_min": 100_000_000},
                ],
                candidate_symbols=["US.PLTR", "US.SNOW"],
            ),
        ],
        candidates=[
            discovery_v1.DiscoveryCandidate(
                symbol="US.PLTR",
                layer_key="ai_software",
                subdomain="AI application software",
                role="AI platform candidate",
                rationale="Found by AI software refinement probe.",
                priority="strong_candidate",
                source_trace_ids=["tool_026"],
            )
        ],
    )

    warnings = discovery_v1._discovery_warnings(package)

    assert not any("same-plate refinement" in warning for warning in warnings)


def test_discovery_v1_progress_log_is_opt_in(capsys):
    silent_recorder = discovery_v1.DiscoveryRecorder(
        max_catalog_reads=0,
        max_filter_runs=0,
        progress=False,
    )
    silent_recorder.progress_log("tool_start", tool="run_futu_stock_filter")
    assert capsys.readouterr().err == ""

    noisy_recorder = discovery_v1.DiscoveryRecorder(
        max_catalog_reads=0,
        max_filter_runs=0,
        progress=True,
    )
    noisy_recorder.progress_log("tool_start", tool="run_futu_stock_filter")

    err = capsys.readouterr().err
    assert "IA_DISCOVERY_V1_PROGRESS" in err
    assert "tool_start" in err
    assert "run_futu_stock_filter" in err


def test_futu_stock_filter_catalog_reflects_sdk_enums():
    catalog = futu_theme_discovery.futu_stock_filter_catalog("US")

    assert catalog["source"] == "futu_sdk_reflection"
    assert catalog["api"] == "OpenQuoteContext.get_stock_filter"
    assert "MARKET_VAL" in catalog["filter_types"]["simple"]["fields"]
    assert "TURNOVER" in catalog["filter_types"]["accumulate"]["fields"]
    assert "RETURN_ON_EQUITY_RATE" in catalog["filter_types"]["financial"]["fields"]
    assert "MA_ALIGNMENT_LONG" in catalog["filter_types"]["pattern"]["fields"]
    assert "MA20" in catalog["filter_types"]["custom_indicator"]["fields"]
    assert catalog["sort_dir"] == ["ASCEND", "DESCEND"]
    assert "K_DAY" in catalog["supported_pattern_ktype"]
    assert catalog["raw_sdk_filter_catalog"]["filter_types"]["simple"]["fields"] == catalog["filter_types"]["simple"]["fields"]


def test_futu_stock_filter_catalog_exposes_app_screener_choices():
    catalog = futu_theme_discovery.futu_stock_filter_catalog("US")
    categories = {item["key"]: item for item in catalog["app_screener_catalog"]}

    assert set(categories) == {
        "market_quote",
        "valuation",
        "dividend",
        "technical",
        "financial",
        "analysis",
        "options",
    }
    quote_choices = {item["label"]: item for item in categories["market_quote"]["choices"]}
    assert quote_choices["交易所/市场"]["capability"] == "market_selector"
    assert quote_choices["所属行业/概念/板块"]["capability"] == "plate_selector"
    assert quote_choices["涨跌幅/振幅/成交量/成交额/换手率"]["type"] == "accumulate"
    assert categories["dividend"]["choices"][0]["capability"] == "non_stock_filter"
    assert "dividend_ratio_ttm" in categories["dividend"]["choices"][0]["alternate_source"]
    assert categories["analysis"]["choices"][0]["capability"] == "external_or_future_adapter"
    assert categories["options"]["choices"][0]["capability"] == "option_chain_enrichment"


def test_futu_tool_description_includes_offline_catalog_documents():
    catalog = futu_theme_discovery.futu_stock_filter_catalog("US")
    offline_catalog = {
        "source": "manual_futu_screener_catalog_export",
        "path": "/tmp/futu_screener_catalog.json",
        "generated_at": "2026-05-21T00:00:00+00:00",
        "market": "US",
        "warnings": [],
        "markdown_tree": {
            "root": "/tmp/futu_screener_catalog",
            "index": "index.md",
            "documents": [
                {
                    "path": "markets/US/plates/industry.md",
                    "title": "US Futu Plate INDUSTRY",
                    "kind": "plate_list",
                    "market": "US",
                    "category": "INDUSTRY",
                    "count": 1,
                }
            ],
        },
        "markets": {
            "US": {
                "plates": {
                    "INDUSTRY": [
                        {
                            "code": "US.LIST2015",
                            "name": "半导体",
                            "plate_type": "INDUSTRY",
                            "plate_id": "2015",
                        }
                    ]
                }
            }
        },
    }

    description = futu_theme_discovery._futu_explore_tool_description(catalog, offline_catalog)

    assert "offline_documents" in description
    assert "offline_futu_plate_counts" in description
    assert "plate_codes" in description
    assert "markets/US/plates/industry.md" in description
    assert "INDUSTRY" in description


def test_search_futu_screener_catalog_finds_plates_and_fields(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    stock_filter_catalog = futu_theme_discovery.futu_stock_filter_catalog("US")
    catalog_path.write_text(
        json.dumps(
            {
                "source": "manual_futu_screener_catalog_export",
                "generated_at": "2026-05-21T00:00:00+00:00",
                "markets": {
                    "US": {
                        "stock_filter_catalog": stock_filter_catalog,
                        "plates": {
                            "CONCEPT": [
                                {
                                    "code": "US.LIST2136",
                                    "name": "人工智能",
                                    "plate_type": "CONCEPT",
                                    "plate_id": "2136",
                                }
                            ]
                        },
                    }
                },
                "markdown_tree": {
                    "root": str(tmp_path / "docs"),
                    "index": "index.md",
                    "documents": [
                        {
                            "path": "markets/US/filters/valuation.md",
                            "title": "US 估值 Screener Options",
                            "kind": "app_category",
                            "market": "US",
                            "category": "valuation",
                        }
                    ],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plate_result = futu_theme_discovery.search_futu_screener_catalog(
        query="人工智能",
        market="US",
        path=catalog_path,
    )
    valuation_result = futu_theme_discovery.search_futu_screener_catalog(
        query="市盈率",
        market="US",
        path=catalog_path,
    )

    assert plate_result["plates"][0]["code"] == "US.LIST2136"
    assert any(item["field"] == "PE_TTM" for item in valuation_result["stock_fields"])


def test_load_futu_screener_catalog_snapshot_missing_is_explicit(tmp_path):
    missing = tmp_path / "missing.json"

    catalog = futu_theme_discovery.load_futu_screener_catalog_snapshot(
        market="US",
        path=missing,
    )

    assert catalog["source"] == "offline_snapshot_missing"
    assert catalog["path"] == str(missing)
    assert "Run plugins/investment_assistant/scripts/export_futu_screener_catalog.py" in catalog["warnings"][0]


def test_futu_stock_filter_specs_build_sdk_filters():
    futu = ia_adapters._import_futu()

    filters = futu_theme_discovery._build_futu_stock_filters(
        futu,
        [
            {"type": "simple", "stock_field": "MARKET_VAL", "sort": "DESCEND"},
            {
                "type": "accumulate",
                "stock_field": "TURNOVER",
                "days": 20,
                "filter_min": 100_000_000,
            },
        ],
    )

    assert type(filters[0]).__name__ == "SimpleFilter"
    assert filters[0].stock_field == futu.StockField.MARKET_VAL
    assert filters[0].sort == futu.SortDir.DESCEND
    assert filters[0].is_no_filter is True
    assert type(filters[1]).__name__ == "AccumulateFilter"
    assert filters[1].stock_field == futu.StockField.TURNOVER
    assert filters[1].days == 20
    assert filters[1].filter_min == 100_000_000
    assert filters[1].is_no_filter is False


def test_filter_calibration_validates_stock_filter_specs_against_catalog():
    errors = filter_calibration.validate_stock_filter_specs_against_catalog(
        [{"type": "simple", "stock_field": "NOT_A_REAL_FIELD", "sort": "DESCEND"}],
        market="US",
    )

    assert "stock_field is unsupported" in errors[0]


def test_filter_calibration_builds_auditable_artifact_from_trial_results():
    probe = CalibrationInputProbe(
        name="memory_momentum",
        rationale="AI storage bottleneck candidates should be tested with ranking and thresholds.",
        signal_type="technical",
        source_categories=["technical"],
        stock_filter_specs=[
            {
                "type": "accumulate",
                "stock_field": "CHANGE_RATE",
                "days": 60,
                "sort": "DESCEND",
            }
        ],
        focus_symbols=["US.SNDK", "US.WDC"],
    )
    trial = CalibrationTrial(
        trial_id="trial_memory_rank",
        probe_name="memory_momentum",
        mode="baseline",
        stock_filter_specs=[
            {
                "type": "accumulate",
                "stock_field": "CHANGE_RATE",
                "days": 60,
                "sort": "DESCEND",
            }
        ],
        result_limit=80,
    )
    result = CalibrationTrialResult(
        trial_id="trial_memory_rank",
        probe_name="memory_momentum",
        diagnosis="usable",
        all_count=42,
        returned_count=2,
        sample_symbols=["US.SNDK", "US.MU"],
        focus_symbols_included=["US.SNDK"],
        focus_symbols_missing=["US.WDC"],
    )
    selected = CalibratedFilter(
        probe_name="memory_momentum",
        selected_mode="rank_then_score",
        selected_trial_id="trial_memory_rank",
        selection_reason="硬过滤会遗漏 WDC，因此保留为排序证据。",
    )

    artifact = filter_calibration.build_filter_calibration_artifact(
        theme="AI",
        market="US",
        probes=[probe],
        trials=[trial],
        trial_results=[result],
        calibrated_filters=[selected],
    )

    assert isinstance(artifact, FilterCalibrationArtifact)
    assert artifact.input_probe_count == 1
    assert artifact.calibrated_filters[0].selected_filters == trial.stock_filter_specs
    assert artifact.focus_symbol_audit["included_any"] == ["US.SNDK"]
    assert artifact.focus_symbol_audit["missing_all"] == ["US.WDC"]


def test_filter_calibration_rejects_selection_unknown_trial():
    probe = CalibrationInputProbe(
        name="quality",
        stock_filter_specs=[{"type": "simple", "stock_field": "MARKET_VAL", "sort": "DESCEND"}],
    )
    trial = CalibrationTrial(
        trial_id="trial_quality_rank",
        probe_name="quality",
        stock_filter_specs=[{"type": "simple", "stock_field": "MARKET_VAL", "sort": "DESCEND"}],
    )
    selected = CalibratedFilter(
        probe_name="quality",
        selected_mode="calibrated_filter",
        selected_trial_id="missing_trial",
        selection_reason="bad",
    )

    with pytest.raises(ValueError, match="unknown selected_trial_id"):
        filter_calibration.build_filter_calibration_artifact(
            theme="AI",
            market="US",
            probes=[probe],
            trials=[trial],
            trial_results=[],
            calibrated_filters=[selected],
        )


def test_theme_domain_candidate_role_is_filled_by_normalizer():
    plan = ThemeDiscoveryPlan(
        theme="storage",
        market="US",
        initial_thesis="Storage map.",
        domain_tree=[
            ThemeDomain(
                key="storage",
                name="Storage",
                importance="core",
                subdomains=[
                    ThemeSubdomain(
                        key="nand",
                        name="NAND and eSSD",
                        thesis="NAND and eSSD should be evaluated as a storage bottleneck.",
                        importance="high",
                        candidates=[ThemeDomainCandidate(symbol="SNDK")],
                    )
                ],
            )
        ],
        seed_symbols=[],
        warnings=["model hypothesis pending downstream validation"],
    )

    observations = theme_discovery._audit_theme_discovery_plan("storage", "US", [], plan)

    assert observations == []
    assert plan.domain_tree[0].subdomains[0].candidates[0].role == "NAND and eSSD"
    assert plan.seed_symbols[0].symbol == "US.SNDK"
    assert plan.seed_symbols[0].role == "NAND and eSSD"


def test_pydantic_trace_skips_part_delta_by_default(monkeypatch):
    class Delta:
        part_delta_kind = "tool_call"
        args_delta = '{"large":"streaming final json"}'
        tool_name_delta = ""
        tool_call_id = "call_1"
        tool_call_id_delta = ""

    class Event:
        event_kind = "part_delta"
        delta = Delta()
        index = 1

    monkeypatch.delenv("IA_PYDANTIC_TRACE_PART_DELTAS", raising=False)

    assert pydantic_runtime._summarize_pydantic_event(Event(), 120) is None

    monkeypatch.setenv("IA_PYDANTIC_TRACE_PART_DELTAS", "1")

    assert "args_delta_len" in pydantic_runtime._summarize_pydantic_event(Event(), 120)


def test_futu_assisted_discovery_is_separate_from_original_theme_discovery():
    assert futu_theme_discovery.build_futu_assisted_theme_discovery_plan is not theme_discovery.build_theme_discovery_plan
    assert "futu_explore_theme_candidates" not in theme_discovery._DISCOVERY_INSTRUCTIONS
    assert "futu_explore_theme_candidates" in futu_theme_discovery._FUTU_ASSISTED_DISCOVERY_INSTRUCTIONS


def test_candidate_pool_defaults_to_futu_assisted_discovery(monkeypatch):
    calls = []

    class FutuCapableAdapter:
        config = types.SimpleNamespace(market="US")

        def _load_futu_candidates(self, canonical_theme, seeds, plate_keywords):
            raise AssertionError("patched futu-assisted builder should own enrichment")

    def fake_futu_assisted_builder(theme, policy, *, market_data=None, explorer=None):
        calls.append(
            {
                "theme": theme,
                "required_symbols": policy.required_symbols,
                "adapter": market_data,
                "explorer": explorer,
            }
        )
        return CandidatePool(
            theme=theme,
            generated_from=["pydantic_ai_futu_assisted_theme_discovery"],
            candidates=[
                _candidate("US.ROK", "industrial automation", score=72),
                _candidate("US.ISRG", "surgical robotics", score=91),
            ],
            discovery_thesis="Futu-assisted robotics thesis.",
        )

    monkeypatch.delenv("IA_CANDIDATE_DISCOVERY_MODE", raising=False)
    monkeypatch.setattr(
        futu_theme_discovery,
        "build_futu_assisted_candidate_pool",
        fake_futu_assisted_builder,
    )

    policy = InvestmentPolicy(theme="robotics", required_symbols=["US.ROK"])
    pool = ia_candidate_pool.build_candidate_pool(
        "robotics",
        policy,
        market_data=FutuCapableAdapter(),
    )

    assert calls and calls[0]["theme"] == "robotics"
    assert pool.generated_from == ["pydantic_ai_futu_assisted_theme_discovery"]
    assert [candidate.symbol for candidate in pool.candidates] == ["US.ROK", "US.ISRG"]


def test_candidate_pool_legacy_mode_keeps_original_theme_discovery_path(monkeypatch):
    monkeypatch.setenv("IA_CANDIDATE_DISCOVERY_MODE", "legacy")

    policy = InvestmentPolicy(theme="storage")
    pool = ia_candidate_pool.build_candidate_pool(
        "storage",
        policy,
        market_data=FakeMarketDataAdapter(),
    )

    assert pool.generated_from == ["fake_test_data"]
    assert [candidate.symbol for candidate in pool.candidates[:3]] == [
        "US.MU",
        "US.WDC",
        "US.STX",
    ]


def test_research_capabilities_install_missing_optional_groups(monkeypatch):
    _use_default_research_settings(monkeypatch)
    installed = []
    search_attempts = {"count": 0}
    fetch_kwargs = []

    class FakeThinking:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWebSearch:
        def __init__(self, **kwargs):
            search_attempts["count"] += 1
            if search_attempts["count"] == 1:
                raise RuntimeError(
                    "WebSearch(local='duckduckgo') requires the duckduckgo optional group — "
                    'pip install "pydantic-ai-slim[duckduckgo]".'
                )
            self.kwargs = kwargs

    class FakeWebFetch:
        def __init__(self, **kwargs):
            fetch_kwargs.append(kwargs)
            if kwargs.get("native") is not False:
                raise AssertionError("web fetch must use local fallback")
            if "max_uses" in kwargs:
                raise AssertionError("max_uses is native-only for WebFetch")
            self.kwargs = kwargs

    fake_capabilities = types.ModuleType("pydantic_ai.capabilities")
    fake_capabilities.Thinking = FakeThinking
    fake_capabilities.WebSearch = FakeWebSearch
    fake_capabilities.WebFetch = FakeWebFetch
    monkeypatch.setitem(sys.modules, "pydantic_ai", types.ModuleType("pydantic_ai"))
    monkeypatch.setitem(sys.modules, "pydantic_ai.capabilities", fake_capabilities)
    monkeypatch.setattr(pydantic_runtime, "_local_web_fetch_tool", lambda settings: "pydantic_web_fetch_tool")

    def fake_ensure(feature, *, prompt=True, force=False):
        installed.append((feature, prompt, force))

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)

    capabilities, status = pydantic_runtime._research_capabilities(
        enable_web_search=True,
        enable_web_fetch=True,
    )

    assert installed == [("investment.pydantic_ai", False, True)]
    assert search_attempts["count"] == 2
    assert len(capabilities) == 3
    assert status["web_search"] is True
    assert status["web_fetch"] is True
    assert fetch_kwargs == [{"native": False, "local": "pydantic_web_fetch_tool"}]


def test_research_openai_chat_model_string_is_rewritten_to_responses():
    assert (
        pydantic_runtime._model_name_for_agent(
            "openai:gpt-5.5",
            prefer_openai_responses=True,
        )
        == "openai-responses:gpt-5.5"
    )
    assert (
        pydantic_runtime._model_name_for_agent(
            "openai-chat:gpt-5.5",
            prefer_openai_responses=True,
        )
        == "openai-responses:gpt-5.5"
    )
    assert (
        pydantic_runtime._model_name_for_agent(
            "openai:gpt-5.5",
            prefer_openai_responses=False,
        )
        == "openai:gpt-5.5"
    )


def test_pydantic_agent_uses_responses_when_thinking_is_enabled(monkeypatch):
    monkeypatch.setattr(pydantic_runtime, "ensure_pydantic_ai_available", lambda: "test-version")
    monkeypatch.setattr(
        pydantic_runtime,
        "load_model_config",
        lambda: {
            "model": "gpt-5.5",
            "base_url": "https://api.openai.com/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(
        pydantic_runtime,
        "research_settings",
        lambda: pydantic_runtime.ResearchSettings(
            web_enabled=False,
            thinking_effort="high",
            request_timeout=45,
        ),
    )

    captured = {}

    class FakeAgent:
        def __init__(self, model, **kwargs):
            captured["agent_model"] = model
            captured["agent_kwargs"] = kwargs

    class FakeThinking:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWebSearch:
        pass

    class FakeWebFetch:
        pass

    def fake_create_model(model_config, *, prefer_openai_responses=False):
        captured["prefer_openai_responses"] = prefer_openai_responses
        return "fake-responses-model", {
            "model": "openai-responses:gpt-5.5",
            "api_mode": "openai_responses",
        }

    fake_pydantic_ai = types.ModuleType("pydantic_ai")
    fake_pydantic_ai.Agent = FakeAgent
    fake_capabilities = types.ModuleType("pydantic_ai.capabilities")
    fake_capabilities.Thinking = FakeThinking
    fake_capabilities.WebSearch = FakeWebSearch
    fake_capabilities.WebFetch = FakeWebFetch
    monkeypatch.setitem(sys.modules, "pydantic_ai", fake_pydantic_ai)
    monkeypatch.setitem(sys.modules, "pydantic_ai.capabilities", fake_capabilities)
    monkeypatch.setattr(pydantic_runtime, "_create_model", fake_create_model)

    _agent, _model_config, runtime = pydantic_runtime.create_pydantic_agent(
        output_type=ThemeDiscoveryPlan,
        instructions="test",
        agent_kind="theme_discovery",
        enable_web_search=False,
        enable_web_fetch=False,
    )

    assert captured["prefer_openai_responses"] is True
    assert runtime["api_mode"] == "openai_responses"
    assert runtime["model"] == "openai-responses:gpt-5.5"
    assert captured["agent_kwargs"]["model_settings"] == {"timeout": 45}
    assert len(captured["agent_kwargs"]["capabilities"]) == 1
    assert captured["agent_kwargs"]["capabilities"][0].kwargs == {"effort": "high"}


def test_research_capabilities_use_local_tools_when_model_cannot_support_native(monkeypatch):
    _use_default_research_settings(monkeypatch)
    web_kwargs = []

    class FakeThinking:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeWebSearch:
        def __init__(self, **kwargs):
            web_kwargs.append(("search", kwargs))
            if kwargs.get("native") is not False:
                raise AssertionError("web search should use local fallback")

    class FakeWebFetch:
        def __init__(self, **kwargs):
            web_kwargs.append(("fetch", kwargs))
            if kwargs.get("native") is not False:
                raise AssertionError("web fetch should use local fallback")
            if "max_uses" in kwargs:
                raise AssertionError("max_uses is native-only for WebFetch")

    fake_capabilities = types.ModuleType("pydantic_ai.capabilities")
    fake_capabilities.Thinking = FakeThinking
    fake_capabilities.WebSearch = FakeWebSearch
    fake_capabilities.WebFetch = FakeWebFetch
    monkeypatch.setitem(sys.modules, "pydantic_ai", types.ModuleType("pydantic_ai"))
    monkeypatch.setitem(sys.modules, "pydantic_ai.capabilities", fake_capabilities)
    monkeypatch.setattr(pydantic_runtime, "_local_web_fetch_tool", lambda settings: "pydantic_web_fetch_tool")

    capabilities, status = pydantic_runtime._research_capabilities(
        enable_web_search=True,
        enable_web_fetch=True,
        enable_provider_web_tools=False,
    )

    assert len(capabilities) == 3
    assert status["web_search"] is True
    assert status["web_fetch"] is True
    assert web_kwargs == [
        ("search", {"local": "duckduckgo", "native": False}),
        ("fetch", {"native": False, "local": "pydantic_web_fetch_tool"}),
    ]


def test_research_model_settings_control_web_modes(monkeypatch):
    monkeypatch.setattr(
        pydantic_runtime,
        "_read_hermes_config",
        lambda: {
            "investment_assistant": {
                "model_settings": {
                    "web_search_mode": "local",
                    "web_fetch_mode": "native",
                    "max_searches": 3,
                    "max_fetches": 2,
                    "web_fetch_timeout": 12,
                    "web_fetch_retries": 2,
                    "web_fetch_max_content_length": 12000,
                    "request_timeout": 45,
                }
            }
        },
    )
    monkeypatch.delenv("IA_RESEARCH_WEB_SEARCH_MODE", raising=False)
    monkeypatch.delenv("IA_RESEARCH_WEB_FETCH_MODE", raising=False)
    monkeypatch.delenv("IA_RESEARCH_MAX_SEARCHES", raising=False)
    monkeypatch.delenv("IA_RESEARCH_MAX_FETCHES", raising=False)
    monkeypatch.delenv("IA_RESEARCH_WEB_FETCH_TIMEOUT", raising=False)
    monkeypatch.delenv("IA_RESEARCH_WEB_FETCH_RETRIES", raising=False)
    monkeypatch.delenv("IA_RESEARCH_WEB_FETCH_MAX_CONTENT_LENGTH", raising=False)
    monkeypatch.delenv("IA_RESEARCH_REQUEST_TIMEOUT", raising=False)

    settings = pydantic_runtime.research_settings()

    assert settings.web_search_mode == "local"
    assert settings.web_fetch_mode == "native"
    assert settings.max_searches == 3
    assert settings.max_fetches == 2
    assert settings.web_fetch_timeout == 12
    assert settings.web_fetch_retries == 2
    assert settings.web_fetch_max_content_length == 12000
    assert settings.request_timeout == 45
    assert pydantic_runtime._agent_model_settings(settings) == {"timeout": 45}
    assert pydantic_runtime._web_search_kwargs(settings, enable_provider_web_tools=True) == {
        "native": False,
        "local": "duckduckgo",
    }
    assert pydantic_runtime._web_fetch_kwargs(settings) == {"native": True, "local": False}


def test_research_web_fetch_defaults_to_local(monkeypatch):
    _use_default_research_settings(monkeypatch)

    settings = pydantic_runtime.research_settings()
    fetch_kwargs = pydantic_runtime._web_fetch_kwargs(settings)

    assert settings.web_fetch_mode == "local"
    assert fetch_kwargs["native"] is False
    assert fetch_kwargs["local"].name == "web_fetch"
    assert fetch_kwargs["local"].max_retries == settings.web_fetch_retries
    assert fetch_kwargs["local"].timeout == settings.web_fetch_timeout + 5


def test_research_request_timeout_default_allows_slow_structured_discovery(monkeypatch):
    _use_default_research_settings(monkeypatch)

    settings = pydantic_runtime.research_settings()

    assert settings.request_timeout == 420.0
    assert pydantic_runtime._agent_model_settings(settings) == {"timeout": 420.0}


def test_theme_discovery_requires_domain_tree():
    plan = ThemeDiscoveryPlan(
        theme="storage",
        market="US",
        initial_thesis="Generic storage map.",
        coverage_requirements=[],
        seed_symbols=[
            ThemeDiscoverySeed(symbol="SNDK", market="US", role="storage candidate"),
        ],
    )

    observations = theme_discovery._audit_theme_discovery_plan("storage", "US", [], plan)

    assert observations
    assert "domain_tree" in observations[0]["message"]


def test_theme_discovery_requires_source_or_warning_for_must_consider():
    plan = ThemeDiscoveryPlan(
        theme="storage",
        market="US",
        initial_thesis="Storage map.",
        domain_tree=[
            ThemeDomain(
                key="storage",
                name="Storage",
                thesis="Storage is a required exposure.",
                importance="core",
                subdomains=[
                    ThemeSubdomain(
                        key="storage",
                        name="Storage",
                        thesis="Storage is a required exposure.",
                        importance="high",
                        candidate_limit_reason="Use the representative storage candidate.",
                        candidates=[
                            ThemeDomainCandidate(
                                symbol="SNDK",
                                role="storage",
                                priority="must_consider",
                            )
                        ],
                    )
                ],
            )
        ],
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="storage",
                name="Storage",
                thesis="Storage is a required exposure.",
                candidate_symbols=["SNDK"],
                must_consider_symbols=["SNDK"],
            )
        ],
        seed_symbols=[
            ThemeDiscoverySeed(symbol="SNDK", market="US", role="storage"),
        ],
    )

    observations = theme_discovery._audit_theme_discovery_plan("storage", "US", [], plan)

    assert observations
    assert "must-consider" in observations[0]["message"]


def test_theme_discovery_adds_low_confidence_seeds_for_coverage_references():
    plan = ThemeDiscoveryPlan(
        theme="ai",
        market="US",
        initial_thesis="AI app and data platform map.",
        domain_tree=[
            ThemeDomain(
                key="platform",
                name="Platform",
                thesis="Platform candidates should be evaluated.",
                subdomains=[
                    ThemeSubdomain(
                        key="platform",
                        name="Platform candidates",
                        thesis="Platform candidates should be evaluated.",
                        candidate_limit_reason="Use one platform representative.",
                        candidates=[ThemeDomainCandidate(symbol="MSFT", role="cloud platform")],
                    )
                ],
            )
        ],
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="enterprise_ai_apps_and_data_platforms",
                name="Enterprise AI apps and data platforms",
                thesis="Application and data platforms should be evaluated.",
                candidate_symbols=["ADBE"],
                must_consider_symbols=["ADBE"],
            )
        ],
        seed_symbols=[
            ThemeDiscoverySeed(symbol="MSFT", market="US", role="cloud platform"),
        ],
    )

    observations = theme_discovery._audit_theme_discovery_plan("ai", "US", [], plan)

    assert observations == []
    added = next(seed for seed in plan.seed_symbols if seed.symbol == "US.ADBE")
    assert added.confidence == "low"
    assert added.exposure_type == "coverage_requirement_reference"
    assert "US.ADBE" in " ".join(plan.warnings)


def test_theme_discovery_drops_unknown_source_ids_with_warning():
    plan = ThemeDiscoveryPlan(
        theme="ai",
        market="US",
        initial_thesis="AI compute map.",
        domain_tree=[
            ThemeDomain(
                key="compute",
                name="Compute",
                thesis="Compute accelerators should be evaluated.",
                importance="core",
                subdomains=[
                    ThemeSubdomain(
                        key="compute",
                        name="Compute",
                        thesis="Compute accelerators should be evaluated.",
                        importance="high",
                        candidate_limit_reason="Use one compute representative.",
                        candidates=[
                            ThemeDomainCandidate(
                                symbol="AMD",
                                role="compute candidate",
                                priority="must_consider",
                            )
                        ],
                    )
                ],
            )
        ],
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="compute",
                name="Compute",
                thesis="Compute accelerators should be evaluated.",
                candidate_symbols=["AMD"],
                must_consider_symbols=["AMD"],
            )
        ],
        seed_symbols=[
            ThemeDiscoverySeed(
                symbol="AMD",
                market="US",
                role="compute candidate",
                source_ids=["missing_source"],
                confidence="high",
                freshness="fresh",
            ),
        ],
        research_trace=[],
    )

    observations = theme_discovery._audit_theme_discovery_plan("ai", "US", [], plan)

    assert observations == []
    assert plan.seed_symbols[0].source_ids == []
    assert plan.seed_symbols[0].confidence == "low"
    assert plan.seed_symbols[0].freshness == "unknown"
    assert "missing_source" in " ".join(plan.warnings)


def test_theme_discovery_rejects_non_ascii_ticker():
    plan = ThemeDiscoveryPlan(
        theme="storage",
        market="US",
        seed_symbols=[
            ThemeDiscoverySeed(symbol="DDОG", market="US", role="non-ascii ticker"),
        ],
    )

    observations = theme_discovery._audit_theme_discovery_plan("storage", "US", [], plan)

    assert observations
    assert "non-ASCII" in observations[0]["message"]


def test_theme_discovery_normalizes_zero_width_symbol_characters():
    assert theme_discovery.normalize_futu_symbol("US.Q\u200dQQ", "US") == "US.QQQ"


def test_theme_discovery_normalizes_symbols_by_requested_market(monkeypatch):
    def fake_theme_agent(theme, market, theme_description, required_symbols):
        return ThemeDiscoveryPlan(
            theme=theme,
            market=market,
            initial_thesis="HK platform discovery thesis.",
            domain_tree=[
                ThemeDomain(
                    key="platform",
                    name="Platform candidates",
                    thesis="Platform businesses are the required exposure for this test.",
                    importance="core",
                    subdomains=[
                        ThemeSubdomain(
                            key="platform",
                            name="Platform candidates",
                            thesis="Platform businesses are the required exposure for this test.",
                            importance="high",
                            candidate_limit_reason="Keep two HK platform candidates for validation.",
                            candidates=[
                                ThemeDomainCandidate(
                                    symbol="00700",
                                    role="required platform",
                                    priority="must_consider",
                                ),
                                ThemeDomainCandidate(symbol="09988", role="cloud platform"),
                            ],
                        )
                    ],
                )
            ],
            coverage_requirements=[
                ThemeCoverageRequirement(
                    key="platform",
                    name="Platform candidates",
                    thesis="Platform businesses are the required exposure for this test.",
                    candidate_symbols=["00700", "09988"],
                    must_consider_symbols=["00700"],
                )
            ],
            seed_symbols=[
                ThemeDiscoverySeed(symbol="00700", market=market, role="required platform", source_ids=["src_hk_platform"]),
                ThemeDiscoverySeed(symbol="09988", market=market, role="cloud platform", source_ids=["src_hk_platform"]),
            ],
            research_trace=[
                ResearchSource(
                    source_id="src_hk_platform",
                    title="HK platform source",
                    url="https://example.com/hk-platform",
                    summary="HK platform candidates.",
                    symbols=["HK.00700", "HK.09988"],
                    coverage_keys=["platform"],
                )
            ],
            pydantic_ai={"mode": "pydantic_ai_theme_discovery", "mock": True},
        )

    monkeypatch.setattr(theme_discovery, "_run_pydantic_theme_agent", fake_theme_agent)

    plan = theme_discovery.build_theme_discovery_plan(
        "ai",
        market="HK",
        required_symbols=["HK.00700"],
    )

    assert plan.market == "HK"
    assert [seed.symbol for seed in plan.seed_symbols] == ["HK.00700", "HK.09988"]


def test_resolve_theme_symbols_uses_discovery_plate_keywords():
    class FakeQuoteContext:
        pass

    class FakeFutu:
        pass

    class RecordingMarketDataAdapter(MarketDataAdapter):
        def __init__(self):
            super().__init__()
            self.seen_plate_keywords = None

        def _find_theme_plates(self, quote_ctx, futu, keywords):
            self.seen_plate_keywords = keywords
            return []

        def _get_owner_plates(self, quote_ctx, futu, codes):
            return {}

    adapter = RecordingMarketDataAdapter()

    result = adapter._resolve_theme_symbols(
        FakeQuoteContext(),
        FakeFutu(),
        "ai",
        [("US.NVDA", _fake_discovery_data("compute leader", "GPU compute leader."))],
        ["人工智能", "数据中心"],
    )

    assert adapter.seen_plate_keywords == ["人工智能", "数据中心"]
    assert result["codes"] == ["US.NVDA"]
    assert result["discovery_by_code"]["US.NVDA"].rationale == "GPU compute leader."


def test_owner_plate_supported_codes_skip_known_non_stock_types():
    codes = ["US.QQQ", "US.NVDA", "US.UNKNOWN", "US.SMH", "US.SNDK"]
    basic_info = {
        "US.QQQ": {"stock_type": "ETF"},
        "US.NVDA": {"stock_type": "STOCK"},
        "US.SMH": {"stock_type": "ETF"},
        "US.SNDK": {"stock_type": "stock"},
    }

    assert ia_adapters._owner_plate_supported_codes(codes, basic_info) == [
        "US.NVDA",
        "US.UNKNOWN",
        "US.SNDK",
    ]


def test_triage_high_salience_must_review_symbols_focuses_bottlenecks():
    artifact = lightweight_enrichment.LightweightEnrichmentArtifact(
        theme="ai",
        plan=lightweight_enrichment.LightweightEnrichmentPlan(theme="ai"),
        candidates=[
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.SNDK",
                layers=["memory_storage_hbm"],
                quote_status="ok",
                kline_status="ok",
                return_60d=2.9,
                turnover=20_000_000_000,
                plate_memberships=[{"plate_name": "存储概念股"}],
            ),
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.LITE",
                layers=["networking_optical_power"],
                quote_status="ok",
                kline_status="ok",
                return_60d=0.8,
                turnover=5_000_000_000,
                plate_memberships=[{"plate_name": "光通信"}],
            ),
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.BE",
                layers=["power_cooling_electrical"],
                quote_status="ok",
                kline_status="ok",
                return_60d=0.64,
                turnover=2_000_000_000,
                plate_memberships=[{"plate_name": "电气设备及零件"}],
            ),
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.MEGA",
                layers=["cloud_ai_factories"],
                quote_status="ok",
                kline_status="ok",
                return_60d=0.9,
                turnover=50_000_000_000,
            ),
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.THIN",
                layers=["memory_storage_hbm"],
                quote_status="ok",
                kline_status="ok",
                return_60d=0.9,
                turnover=100_000,
            ),
        ],
    )

    assert candidate_triage._high_salience_must_review_symbols(artifact) == [
        "US.SNDK",
        "US.LITE",
        "US.BE",
    ]


def test_candidate_triage_plan_requires_user_input_and_layer_budgets():
    discovery = ThemeDiscoveryPlan(
        theme="ai",
        domain_tree=[
            ThemeDomain(
                key="compute",
                name="Compute",
                importance="core",
                subdomains=[
                    ThemeSubdomain(
                        key="compute",
                        name="Compute",
                        candidates=[ThemeDomainCandidate(symbol="US.NVDA", role="required compute leader")],
                    )
                ],
            ),
            ThemeDomain(
                key="memory_storage",
                name="Memory/storage",
                importance="important",
                subdomains=[
                    ThemeSubdomain(
                        key="memory_storage",
                        name="Memory/storage",
                        candidates=[ThemeDomainCandidate(symbol="US.SNDK", role="storage bottleneck")],
                    )
                ],
            ),
        ],
    )
    artifact = candidate_triage.CandidateTriagePlanArtifact(
        theme="ai",
        prompt_to_user="请选择一个候选粗筛策略。",
        recommended_option_id="coverage_balanced",
        strategy_options=[
            candidate_triage.TriageStrategyOption(
                option_id="coverage_balanced",
                name="Coverage balanced",
                selection_rules=["Preserve each important layer."],
                deep_research_total=3,
                expected_watchlist_count=2,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(layer_key="compute", deep_research_count=1),
                    candidate_triage.LayerResearchBudget(layer_key="memory_storage", deep_research_count=2),
                ],
                best_for="完整覆盖主要分支。",
                tradeoffs=["研究数量较多。"],
            ),
            candidate_triage.TriageStrategyOption(
                option_id="bottleneck_momentum",
                name="Bottleneck momentum",
                selection_rules=["Favor bottlenecks with strong recent Futu momentum."],
                deep_research_total=2,
                expected_watchlist_count=3,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(layer_key="compute", deep_research_count=1),
                    candidate_triage.LayerResearchBudget(layer_key="memory_storage", deep_research_count=1),
                ],
                best_for="优先研究瓶颈和动量。",
                tradeoffs=["可能弱化全产业链覆盖。"],
            ),
        ],
    )

    candidate_triage._validate_triage_plan(artifact, discovery, ["US.NVDA", "US.SNDK"])


def test_candidate_triage_plan_rejects_missing_core_layer_budget():
    discovery = ThemeDiscoveryPlan(
        theme="ai",
        domain_tree=[
            ThemeDomain(key="compute", name="Compute", importance="core"),
            ThemeDomain(key="memory_storage", name="Memory/storage", importance="important"),
        ],
    )
    artifact = candidate_triage.CandidateTriagePlanArtifact(
        theme="ai",
        prompt_to_user="请选择一个候选粗筛策略。",
        recommended_option_id="coverage_balanced",
        strategy_options=[
            candidate_triage.TriageStrategyOption(
                option_id="coverage_balanced",
                name="Coverage balanced",
                selection_rules=["Preserve compute only."],
                deep_research_total=1,
                expected_watchlist_count=1,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(layer_key="compute", deep_research_count=1),
                ],
                best_for="测试缺失预算。",
                tradeoffs=["Missing an important layer."],
            ),
            candidate_triage.TriageStrategyOption(
                option_id="quality",
                name="Quality",
                selection_rules=["Favor quality leaders."],
                deep_research_total=1,
                expected_watchlist_count=1,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(layer_key="compute", deep_research_count=1),
                ],
                best_for="测试缺失预算。",
                tradeoffs=["Missing an important layer."],
            ),
        ],
    )

    with pytest.raises(ValueError, match="omitted core/important layer budgets"):
        candidate_triage._validate_triage_plan(artifact, discovery, ["US.NVDA", "US.SNDK"])


def test_candidate_triage_requires_research_budget_allocations(monkeypatch):
    monkeypatch.setenv("IA_TRIAGE_DEEP_MIN", "1")
    monkeypatch.setenv("IA_TRIAGE_DEEP_MAX", "2")
    artifact = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        triage_summary="Triage completed.",
        research_budget_summary="One symbol should receive deep research.",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["compute"],
                evidence_summary=["Valid Futu quote and core compute role."],
                rationale="Compute anchor needs deep validation.",
            )
        ],
        layer_audits=[
            candidate_triage.TriageLayerAudit(
                layer_key="compute",
                coverage_status="covered",
                selected_symbols=["US.NVDA"],
                rationale="Compute covered by NVDA.",
            )
        ],
    )

    with pytest.raises(ValueError, match="research_budget_allocations"):
        candidate_triage._validate_triage(artifact, ["US.NVDA"], [], [])


def test_candidate_triage_research_budget_allocations_reconcile(monkeypatch):
    monkeypatch.setenv("IA_TRIAGE_DEEP_MIN", "1")
    monkeypatch.setenv("IA_TRIAGE_DEEP_MAX", "2")
    artifact = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        triage_summary="Triage completed.",
        research_budget_summary="Spend one deep-research slot on compute and retain a peer.",
        research_budget_allocations=[
            candidate_triage.ResearchBudgetAllocation(
                layer_key="compute",
                layer_name="Compute",
                allocation_goal="compare_peers",
                deep_research_budget=1,
                deep_research_symbols=["US.NVDA"],
                watchlist_symbols=["US.AMD"],
                rationale="Validate the anchor now and retain the same-layer peer for later comparison.",
            )
        ],
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["compute"],
                evidence_summary=["Valid Futu quote and core compute role."],
                rationale="Compute anchor needs deep validation.",
            )
        ],
        watchlist=[
            candidate_triage.CompactTriageDecision(
                symbol="US.AMD",
                bucket="watchlist",
                priority="medium",
                layer_keys=["compute"],
                rationale="Same-layer compute peer retained without immediate deep spend.",
            )
        ],
        layer_audits=[
            candidate_triage.TriageLayerAudit(
                layer_key="compute",
                coverage_status="covered",
                selected_symbols=["US.NVDA"],
                watchlist_symbols=["US.AMD"],
                rationale="Compute has one deep candidate and one retained peer.",
            )
        ],
    )

    candidate_triage._validate_triage(artifact, ["US.NVDA", "US.AMD"], [], [])


def test_candidate_triage_rejects_mismatched_research_budget_count(monkeypatch):
    monkeypatch.setenv("IA_TRIAGE_DEEP_MIN", "1")
    monkeypatch.setenv("IA_TRIAGE_DEEP_MAX", "2")
    artifact = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        triage_summary="Triage completed.",
        research_budget_summary="Budget count intentionally mismatches symbols.",
        research_budget_allocations=[
            candidate_triage.ResearchBudgetAllocation(
                layer_key="compute",
                allocation_goal="compare_peers",
                deep_research_budget=2,
                deep_research_symbols=["US.NVDA"],
                rationale="Invalid budget for test.",
            )
        ],
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["compute"],
                evidence_summary=["Valid Futu quote and core compute role."],
                rationale="Compute anchor needs deep validation.",
            )
        ],
        layer_audits=[
            candidate_triage.TriageLayerAudit(
                layer_key="compute",
                coverage_status="covered",
                selected_symbols=["US.NVDA"],
                rationale="Compute covered by NVDA.",
            )
        ],
    )

    with pytest.raises(ValueError, match="does not equal deep_research_symbols count"):
        candidate_triage._validate_triage(artifact, ["US.NVDA"], [], [])


def test_real_market_data_adapter_skips_unsupported_snapshot_symbols():
    class FakeFutu:
        RET_OK = 0

    class FakeQuoteContext:
        def get_market_snapshot(self, codes):
            if len(codes) > 1:
                return 1, "暂不提供美股 OTC 市场行情 ABLZF"
            if codes[0] == "US.NVDA":
                return 0, [{"code": "US.NVDA", "last_price": 100}]
            return 1, "暂不提供美股 OTC 市场行情 ABLZF"

    warnings: list[str] = []
    adapter = MarketDataAdapter()
    limiter_calls = 0

    def acquire():
        nonlocal limiter_calls
        limiter_calls += 1

    adapter._quote_rate_limiter.acquire = acquire

    rows = adapter._get_market_snapshot_by_code(
        FakeQuoteContext(),
        FakeFutu(),
        ["US.NVDA", "US.ABLZF"],
        warnings,
    )

    assert list(rows) == ["US.NVDA"]
    assert limiter_calls == 3
    assert any("retrying symbols individually" in warning for warning in warnings)
    assert any("Skipped US.ABLZF" in warning for warning in warnings)


class FakePortfolioAdapter:
    def get_current_portfolio(self):
        raise AssertionError("V1 workflow must not read current portfolio")


class FailingMarketDataAdapter:
    def get_theme_universe(
        self,
        theme: str,
        required_symbols: list[str] | None = None,
        theme_description: str = "",
    ) -> ThemeUniverse:
        raise FutuAdapterError("Futu get_market_snapshot failed: 暂不提供美股 OTC 市场行情 ABLZF")


class FakeSecProvider:
    def get_sec_context(self, candidates: list[Candidate]) -> dict:
        items = {}
        for candidate in candidates:
            items[candidate.symbol] = {
                "symbol": candidate.symbol,
                "ticker": candidate.symbol.split(".", 1)[-1],
                "source_status": "available",
                "cik": "0000000000",
                "company_name": candidate.name,
                "industry": "Semiconductors",
                "filings": {
                    "latest_10k": {
                        "form": "10-K",
                        "filing_date": "2026-02-01",
                        "period_of_report": "2025-12-31",
                        "accession_number": "fake-10k",
                        "url": "https://www.sec.gov/fake",
                    },
                    "latest_10q": {
                        "form": "10-Q",
                        "filing_date": "2026-05-01",
                        "period_of_report": "2026-03-31",
                        "accession_number": "fake-10q",
                        "url": "https://www.sec.gov/fake",
                    },
                    "latest_8k": {
                        "form": "8-K",
                        "filing_date": "2026-05-05",
                        "period_of_report": "2026-05-05",
                        "accession_number": "fake-8k",
                        "url": "https://www.sec.gov/fake",
                    },
                },
                "fundamentals": {
                    "ttm_revenue": 12_000_000_000,
                    "ttm_net_income": 2_000_000_000,
                    "gross_profit": 5_000_000_000,
                    "operating_income": 2_400_000_000,
                    "total_assets": 20_000_000_000,
                    "total_liabilities": 8_000_000_000,
                    "shareholders_equity": 12_000_000_000,
                    "debt_to_assets": 0.4,
                    "roe": 0.166667,
                    "net_margin": 0.166667,
                },
                "numeric_evidence": {
                    "source": "sec_companyfacts",
                    "provider": "edgartools",
                    "llm_generated": False,
                },
                "narrative_evidence": {
                    "source_status": "not_implemented",
                    "planned_pipeline": "mineru_plus_sub_llm",
                    "numeric_extraction_allowed": False,
                },
                "event_context": {
                    "latest_periodic_filing_date": "2026-05-01",
                    "periodic_filing_age_days": 18,
                    "periodic_filing_stale": False,
                    "latest_8k_age_days": 14,
                    "event_risk_level": "medium",
                },
                "risk_flags": ["recent_8k"],
            }
        return {
            "source": "edgartools",
            "source_status": "available",
            "generated_at": "2026-05-19T00:00:00+00:00",
            "requested_symbols": [candidate.symbol for candidate in candidates],
            "fetched_symbols": [candidate.symbol for candidate in candidates],
            "items": items,
            "warnings": [],
        }


def _fake_pydantic_thesis(policy, candidate_pool, reflection, market_artifacts=None):
    assessments: list[CandidateThesisAssessment] = []
    seen: set[str] = set()

    def add_assessment(candidate: Candidate, action: str = "watch"):
        symbol = candidate.symbol.upper()
        if symbol in seen:
            return
        seen.add(symbol)
        assessments.append(
            CandidateThesisAssessment(
                symbol=candidate.symbol,
                role=candidate.theme_role,
                thesis_fit="high" if action == "include" else "medium",
                recommended_action=action,
                evidence_summary=[
                    f"{candidate.symbol} assessed from discovery_data, futu_data, and available SEC artifacts."
                ],
                metrics_considered=[
                    "research_trace",
                    "discovery_data",
                    "futu_quote",
                    "technical_summary",
                    "liquidity_context",
                    "fundamental_data",
                    "event_data",
                ],
                concerns=[],
            )
        )

    for candidate in candidate_pool.candidates[:8]:
        add_assessment(candidate, "include" if len(assessments) < 5 else "watch")

    by_symbol = {candidate.symbol.upper(): candidate for candidate in candidate_pool.candidates}
    for requirement in candidate_pool.coverage_requirements:
        for symbol in requirement.must_consider_symbols:
            candidate = by_symbol.get(symbol.upper())
            if candidate and candidate.eligible_for_portfolio:
                add_assessment(candidate, "watch")

    return ThesisSynthesis(
        theme=candidate_pool.theme,
        primary_thesis="测试桩综合判断：先从候选池的主题发现出发，再结合 Futu、SEC、技术面、流动性和风险资料形成版图输入。",
        thesis_points=["候选池不是最终答案，后续组合阶段当前未启用。"],
        key_bottlenecks=["测试瓶颈：算力、存储、网络、电力"],
        metrics_considered=[
            "discovery_thesis_and_coverage_requirements",
            "discovery_research_trace_and_source_ids",
            "candidate_discovery_role_and_exposure_purity",
            "futu_quote_price_market_cap_turnover_valuation_profitability",
            "technical_trend_relative_strength_returns_realized_volatility",
            "liquidity_turnover_volume_spread",
            "sec_companyfacts_revenue_income_margin_roe_debt",
            "filing_freshness_recent_8k_event_risk",
            "market_regime_benchmark_context_macro_proxies",
            "correlation_diversification_and_risk_flags",
            "data_quality_missing_fields_and_warnings",
        ],
        candidate_assessments=assessments,
        portfolio_implications=["组合权重阶段当前未启用。"],
        data_gaps=[],
        warnings=[],
        pydantic_ai={"available": True, "mode": "pydantic_ai_thesis_synthesis_agent", "mock": True},
    )


def _fake_pydantic_architect(
    policy,
    candidate_pool,
    reflection,
    market_artifacts=None,
    thesis_synthesis=None,
):
    selected = []
    required = {symbol.upper() for symbol in policy.required_symbols}
    for candidate in candidate_pool.candidates:
        if candidate.symbol.upper() in required:
            selected.append(candidate)
    for candidate in candidate_pool.candidates:
        if candidate.symbol.upper() not in required:
            selected.append(candidate)
        if len(selected) >= 5:
            break

    sleeve = min(policy.target_portfolio_weight, 1 - policy.cash_reserve)
    per_holding = round(sleeve / len(selected), 4)
    holdings = [
        PortfolioHolding(
            symbol=candidate.symbol,
            target_weight=per_holding,
            role=candidate.theme_role,
            rationale=f"{candidate.name} is selected by the fake PydanticAI architect.",
            evidence_refs=[f"candidate_pool:{candidate.symbol}"],
        )
        for candidate in selected
    ]
    holding_symbols = {holding.symbol.upper() for holding in holdings}
    candidates_by_symbol = {candidate.symbol.upper(): candidate for candidate in candidate_pool.candidates}
    omitted = []
    for requirement in candidate_pool.coverage_requirements:
        if requirement.priority == "optional":
            continue
        for symbol in requirement.must_consider_symbols:
            candidate = candidates_by_symbol.get(symbol.upper())
            if not candidate or not candidate.eligible_for_portfolio or symbol.upper() in holding_symbols:
                continue
            omitted.append(
                OmittedCandidate(
                    symbol=candidate.symbol,
                    role=candidate.theme_role,
                    reason="测试桩未选择该 must-consider 候选，作为 omission audit 记录。",
                    reason_category="weight_budget",
                    substitute_symbols=[holdings[0].symbol] if holdings else [],
                    importance="high",
                )
            )
    sleeve_weight = round(sum(item.target_weight for item in holdings), 4)
    first_symbols = [item.symbol for item in holdings[: max(1, len(holdings) // 2)]]
    second_symbols = [item.symbol for item in holdings if item.symbol not in first_symbols]
    first_weight = round(sum(item.target_weight for item in holdings if item.symbol in first_symbols), 4)
    second_weight = round(sum(item.target_weight for item in holdings if item.symbol in second_symbols), 4)
    return PortfolioMaps(
        theme=candidate_pool.theme,
        maps=[
            PortfolioMap(
                map_id="ai_architect_core",
                name="AI architect core map",
                objective=policy.objective,
                sleeve_weight=sleeve_weight,
                positioning="Mocked AI-authored target map with explicit sleeves.",
                best_for="Tests that need a typed PydanticAI map without calling a live model.",
                allocation_logic=["Required symbols stay first.", "Remaining candidates are diversified by role."],
                sleeves=[
                    PortfolioSleeve(
                        name="Core anchor",
                        role="Required or highest-confidence base holdings",
                        target_weight=first_weight,
                        holding_symbols=first_symbols,
                        rationale="Keeps the user-specified base exposure in every map.",
                    ),
                    PortfolioSleeve(
                        name="Theme satellite",
                        role="Additional theme breadth",
                        target_weight=second_weight,
                        holding_symbols=second_symbols,
                        rationale="Adds non-required candidates from the unbiased pool.",
                    ),
                ],
                holdings=holdings,
                cash_weight=policy.cash_reserve,
                thesis="Typed portfolio map generated by the mocked PydanticAI architect.",
                risks=["Mock artifact for tests."],
                missing_exposure=reflection.missing_exposure if reflection else [],
                reflection_notes=["Generated without reading current holdings."],
                omitted_candidates=omitted,
            )
        ],
        pydantic_ai={"available": True, "mode": "pydantic_ai_agent", "mock": True},
        warnings=[],
    )


@pytest.fixture(autouse=True)
def mock_pydantic_architect(monkeypatch):
    monkeypatch.setattr(
        "plugins.investment_assistant.agents._run_pydantic_thesis_agent",
        _fake_pydantic_thesis,
    )
    monkeypatch.setattr(
        "plugins.investment_assistant.agents._run_pydantic_ai_agent",
        _fake_pydantic_architect,
    )


def _workflow(store: InvestmentAssistantStore) -> InvestmentAssistantWorkflow:
    return InvestmentAssistantWorkflow(
        store=store,
        market_data=FakeMarketDataAdapter(),
        portfolio=FakePortfolioAdapter(),
        sec_filings=FakeSecProvider(),
    )


def _fake_theme_discovery_plan(
    theme: str,
    *,
    market: str = "US",
    theme_description: str = "",
    required_symbols: list[str] | None = None,
    seed_symbol: str | None = None,
    initial_thesis: str = "Discovery thesis before downstream validation.",
) -> ThemeDiscoveryPlan:
    required_symbols = required_symbols or []
    symbol = seed_symbol or (required_symbols[0] if required_symbols else "US.ISRG")
    raw_symbol = symbol.split(".", 1)[-1]
    return ThemeDiscoveryPlan(
        theme=theme,
        market=market,
        theme_description=theme_description,
        initial_thesis=initial_thesis,
        domain_tree=[
            ThemeDomain(
                key="primary_domain",
                name="Primary domain",
                thesis="The discovery agent found a primary investable domain.",
                importance="core",
                subdomains=[
                    ThemeSubdomain(
                        key="primary_subdomain",
                        name="Primary subdomain",
                        thesis="The subdomain should be validated downstream.",
                        importance="high",
                        candidates=[
                            ThemeDomainCandidate(
                                symbol=raw_symbol,
                                role="primary candidate",
                                rationale="Futu-assisted discovery candidate.",
                                priority="must_consider",
                            )
                        ],
                    )
                ],
            )
        ],
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="primary_subdomain",
                name="Primary subdomain",
                thesis="The downstream flow should validate this subdomain.",
                candidate_symbols=[symbol],
                must_consider_symbols=[symbol],
            )
        ],
        seed_symbols=[
            ThemeDiscoverySeed(
                symbol=symbol,
                market=market,
                role="primary candidate",
                rationale="Futu-assisted discovery candidate.",
                subthemes=["Primary domain", "Primary subdomain"],
                value_chain_stage="Primary subdomain",
                exposure_type="direct operating company",
                exposure_purity="high",
            )
            for symbol in ([*required_symbols] if required_symbols else [symbol])
        ],
        plate_keywords=["主题"],
        filter_plans_by_layer=[
            DiscoveryFilterPlan(
                layer_key="primary_domain",
                layer_name="Primary domain",
                target_candidate_profile="Primary liquid candidates for the requested theme.",
                plate_search_terms=["主题"],
                plate_codes_to_probe=["US.LIST0001"],
                filter_decisions=[
                    DiscoveryFilterDecision(
                        category="plate",
                        decision="use_now",
                        planned_fields=["US.LIST0001"],
                        rationale="Use a theme-relevant Futu plate for initial discovery.",
                    ),
                    DiscoveryFilterDecision(
                        category="liquidity",
                        decision="use_now",
                        planned_fields=["TURNOVER"],
                        planned_thresholds_or_ranking="20-day turnover ranked descending",
                        rationale="Prefer names liquid enough for later portfolio construction review.",
                    ),
                ],
            )
        ],
        executed_filter_probes=[
            ExecutedDiscoveryProbe(
                layer_key="primary_domain",
                probe_type="subdomain_plate",
                plate_code="US.LIST0001",
                stock_filter_specs=[
                    {"type": "simple", "stock_field": "MARKET_VAL", "is_no_filter": True, "sort": "DESCEND"},
                    {"type": "accumulate", "stock_field": "TURNOVER", "days": 20, "is_no_filter": True},
                ],
                trace_id="tool_003",
                result_status="ok",
                result_count=12,
                candidate_symbols=[symbol],
            )
        ],
        layer_filter_audits=[
            DiscoveryLayerFilterAudit(
                layer_key="primary_domain",
                layer_name="Primary domain",
                hypothesis="Primary theme exposure should be tested through plate and liquidity probes.",
                plate_codes_considered=["US.LIST0001"],
                plate_codes_used=["US.LIST0001"],
                candidate_symbols_from_probes=[symbol],
                result_summary="Plate and liquidity probes produced the primary seed candidate.",
            )
        ],
        omissions_to_investigate=[],
        next_enrichment_needed=["SEC filings and market-data enrichment have not run."],
        pydantic_ai={
            "mode": "pydantic_ai_theme_discovery_v1_filter_planning_agent",
            "tool_calls": [],
        },
        warnings=["Discovery v1 still requires downstream validation."],
    )


def _fake_lightweight_artifact(discovery: ThemeDiscoveryPlan) -> lightweight_enrichment.LightweightEnrichmentArtifact:
    symbols = [seed.symbol for seed in discovery.seed_symbols] or ["US.NVDA"]
    return lightweight_enrichment.LightweightEnrichmentArtifact(
        theme=discovery.theme,
        market=discovery.market,
        plan=lightweight_enrichment.LightweightEnrichmentPlan(
            theme=discovery.theme,
            market=discovery.market,
            planning_summary="Fake lightweight Futu plan for workflow tests.",
        ),
        candidates=[
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol=symbol,
                layers=["primary_domain"],
                role="workflow test candidate",
                quote_status="ok",
                kline_status="ok",
                owner_plate_status="ok",
                last_price=100.0,
                turnover=10_000_000,
                return_60d=0.2,
                relative_strength_60d=0.7,
                data_quality="fresh",
            )
            for symbol in symbols
        ],
        check_summary={"quote_ok": len(symbols), "kline_ok": len(symbols)},
        warnings=["Fake lightweight warning."],
        pydantic_ai={"mock": True, "mode": "fake_lightweight"},
    )


def _fake_triage_plan(
    discovery: ThemeDiscoveryPlan,
    lightweight: lightweight_enrichment.LightweightEnrichmentArtifact,
) -> candidate_triage.CandidateTriagePlanArtifact:
    return candidate_triage.CandidateTriagePlanArtifact(
        theme=discovery.theme,
        market=discovery.market,
        planning_summary="Fake triage strategy plan.",
        candidate_count=len(lightweight.candidates),
        layer_count=len(discovery.domain_tree),
        recommended_option_id="coverage_balanced",
        prompt_to_user="请选择一个候选粗筛策略。",
        strategy_options=[
            candidate_triage.TriageStrategyOption(
                option_id="coverage_balanced",
                name="Coverage balanced",
                description="Preserve core coverage before deep research.",
                selection_rules=["Preserve each important layer."],
                deep_research_total=1,
                expected_watchlist_count=0,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(
                        layer_key="primary_domain",
                        layer_name="Primary domain",
                        deep_research_count=1,
                    )
                ],
                best_for="Balanced workflow tests.",
                tradeoffs=["More validation later."],
            ),
            candidate_triage.TriageStrategyOption(
                option_id="quality",
                name="Quality leaders",
                description="Favor liquid leaders.",
                selection_rules=["Favor liquidity and quality."],
                deep_research_total=1,
                expected_watchlist_count=0,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(
                        layer_key="primary_domain",
                        layer_name="Primary domain",
                        deep_research_count=1,
                    )
                ],
                best_for="Quality workflow tests.",
                tradeoffs=["May miss satellites."],
            ),
        ],
        warnings=["Fake triage plan warning."],
        pydantic_ai={"mock": True, "mode": "fake_triage_plan"},
    )


def _fake_candidate_triage_artifact(
    discovery: ThemeDiscoveryPlan,
    lightweight: lightweight_enrichment.LightweightEnrichmentArtifact,
    triage_strategy=None,
) -> candidate_triage.CandidateTriageArtifact:
    symbol = lightweight.candidates[0].symbol if lightweight.candidates else "US.NVDA"
    return candidate_triage.CandidateTriageArtifact(
        theme=discovery.theme,
        market=discovery.market,
        triage_summary="Fake candidate triage completed.",
        research_budget_summary="Fake triage spends one deep-research slot on the primary domain.",
        research_budget_allocations=[
            candidate_triage.ResearchBudgetAllocation(
                layer_key="primary_domain",
                layer_name="Primary domain",
                allocation_goal="preserve_core_coverage",
                deep_research_budget=1,
                deep_research_symbols=[symbol],
                rationale="Fake workflow test budget allocation.",
            )
        ],
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol=symbol,
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["primary_domain"],
                role="workflow test deep research candidate",
                evidence_summary=["Fake lightweight evidence supports deep research."],
                research_needs=["sec_fundamentals"],
                rationale="Selected by fake triage agent.",
            )
        ],
        watchlist=[],
        layer_audits=[
            candidate_triage.TriageLayerAudit(
                layer_key="primary_domain",
                layer_name="Primary domain",
                coverage_status="covered",
                selected_symbols=[symbol],
                rationale="Covered by fake triage.",
            )
        ],
        triage_criteria_used=["Confirmed user triage strategy."],
        pydantic_ai={"mock": True, "mode": "fake_candidate_triage", "selected": triage_strategy or {}},
    )


def _mock_workflow_triage_dependencies(monkeypatch, fake_discovery):
    monkeypatch.setattr(ia_workflow, "build_ai_discovery_v1_plan", fake_discovery)
    monkeypatch.setattr(ia_workflow, "build_lightweight_enrichment_artifact", _fake_lightweight_artifact)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_plan", _fake_triage_plan)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_artifact", _fake_candidate_triage_artifact)


def test_pydantic_triage_strategy_selection_parses_user_answer():
    discovery = _fake_theme_discovery_plan("ai", seed_symbol="US.NVDA")
    lightweight = _fake_lightweight_artifact(discovery)
    plan = _fake_triage_plan(discovery, lightweight)

    selection = candidate_triage.select_triage_strategy(
        plan,
        answer="选 2，但 SNDK/COHR/LITE 必须 deep research",
        modifications="把存储和光通信预算调高。",
        must_include_symbols=["sndk", "US.COHR", "LITE"],
        exclude_symbols=["US.TEST"],
    )

    assert selection.selected_option_id == "quality"
    assert selection.selected_option["option_id"] == "quality"
    assert selection.must_include_symbols == ["US.SNDK", "US.COHR", "US.LITE"]
    assert selection.exclude_symbols == ["US.TEST"]
    assert selection.unmatched_answer_needs_agent_interpretation is False


def test_pydantic_resume_candidate_triage_from_saved_artifacts(tmp_path, monkeypatch):
    discovery = _fake_theme_discovery_plan("ai", seed_symbol="US.NVDA")
    lightweight = _fake_lightweight_artifact(discovery)
    plan = _fake_triage_plan(discovery, lightweight)
    discovery_path = tmp_path / "discovery.json"
    lightweight_path = tmp_path / "lightweight.json"
    plan_path = tmp_path / "triage_plan.json"
    discovery_path.write_text(json.dumps(discovery.model_dump(mode="json")), encoding="utf-8")
    lightweight_path.write_text(json.dumps(lightweight.model_dump(mode="json")), encoding="utf-8")
    plan_path.write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_build_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=None):
        seen["discovery_theme"] = discovery_arg.theme
        seen["lightweight_count"] = len(lightweight_arg.candidates)
        seen["triage_strategy"] = triage_strategy
        return _fake_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=triage_strategy)

    monkeypatch.setattr(candidate_triage, "build_candidate_triage_artifact", fake_build_candidate_triage_artifact)

    artifact, selection = pydantic_resume.resume_candidate_triage_from_files(
        discovery_path=discovery_path,
        lightweight_path=lightweight_path,
        plan_path=plan_path,
        answer="选 1",
        must_include_symbols=["SNDK"],
    )

    assert artifact.artifact_type == "candidate_triage"
    assert selection.selected_option_id == "coverage_balanced"
    assert selection.must_include_symbols == ["US.SNDK"]
    assert seen["discovery_theme"] == "ai"
    assert seen["lightweight_count"] == 1
    assert isinstance(seen["triage_strategy"], candidate_triage.TriageStrategySelection)


def test_pydantic_hitl_accepts_discovery_v1_preview_wrapper(tmp_path):
    discovery_path = tmp_path / "discovery_preview.json"
    lightweight_path = tmp_path / "lightweight.json"
    plan_path = tmp_path / "triage_plan.json"
    state_path = tmp_path / "hitl_state.json"
    discovery_path.write_text(
        json.dumps(
            {
                "preview": {
                    "theme": "AI preview",
                    "market": "US",
                    "initial_thesis": "Preview discovery thesis.",
                    "layers": [
                        {
                            "key": "memory_storage",
                            "name": "Memory and storage",
                            "importance": "important",
                            "economic_mechanism": "AI workloads need memory bandwidth and storage throughput.",
                        }
                    ],
                    "candidates": [
                        {
                            "symbol": "US.SNDK",
                            "layer_key": "memory_storage",
                            "role": "enterprise SSD candidate",
                            "priority": "must_consider",
                            "rationale": "Storage bottleneck exposure from preview discovery.",
                        }
                    ],
                    "warnings": ["Preview-only discovery output."],
                    "next_enrichment_needed": ["Run lightweight Futu enrichment."],
                },
                "pydantic_ai": {"mode": "discovery_v1_preview"},
            }
        ),
        encoding="utf-8",
    )
    lightweight = lightweight_enrichment.LightweightEnrichmentArtifact(
        theme="AI preview",
        market="US",
        plan=lightweight_enrichment.LightweightEnrichmentPlan(
            theme="AI preview",
            market="US",
            planning_summary="Preview lightweight artifact.",
        ),
        candidates=[
            lightweight_enrichment.LightweightCandidateEvidence(
                symbol="US.SNDK",
                layers=["memory_storage"],
                role="enterprise SSD candidate",
                quote_status="ok",
                kline_status="ok",
                last_price=100.0,
                turnover=10_000_000,
                data_quality="fresh",
            )
        ],
    )
    plan = candidate_triage.CandidateTriagePlanArtifact(
        theme="AI preview",
        market="US",
        planning_summary="Preview triage strategy plan.",
        candidate_count=1,
        layer_count=1,
        recommended_option_id="coverage_balanced",
        prompt_to_user="请选择候选粗筛策略。",
        strategy_options=[
            candidate_triage.TriageStrategyOption(
                option_id="coverage_balanced",
                name="Coverage balanced",
                deep_research_total=1,
                layer_budgets=[
                    candidate_triage.LayerResearchBudget(
                        layer_key="memory_storage",
                        layer_name="Memory and storage",
                        deep_research_count=1,
                    )
                ],
            )
        ],
    )
    lightweight_path.write_text(json.dumps(lightweight.model_dump(mode="json")), encoding="utf-8")
    plan_path.write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")

    state = pydantic_resume.create_candidate_triage_hitl_state_from_files(
        discovery_path=discovery_path,
        lightweight_path=lightweight_path,
        plan_path=plan_path,
        output_path=state_path,
    )

    assert state_path.exists()
    assert state.discovery.theme == "AI preview"
    assert state.discovery.seed_symbols[0].symbol == "US.SNDK"
    assert state.discovery.domain_tree[0].key == "memory_storage"
    assert state.discovery.coverage_requirements[0].must_consider_symbols == ["US.SNDK"]
    assert any("converted to ThemeDiscoveryPlan" in warning for warning in state.warnings)


def test_pydantic_candidate_triage_hitl_state_saves_and_resumes(tmp_path, monkeypatch):
    discovery = _fake_theme_discovery_plan("ai", seed_symbol="US.NVDA")
    lightweight = _fake_lightweight_artifact(discovery)
    plan = _fake_triage_plan(discovery, lightweight)
    state_path = tmp_path / "candidate_triage_hitl.json"
    completed_path = tmp_path / "candidate_triage_completed.json"
    seen: dict[str, object] = {}

    def fake_build_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=None):
        seen["triage_strategy"] = triage_strategy
        return _fake_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=triage_strategy)

    monkeypatch.setattr(candidate_triage, "build_candidate_triage_artifact", fake_build_candidate_triage_artifact)

    waiting = pydantic_resume.create_candidate_triage_hitl_state(discovery, lightweight, plan)
    pydantic_resume.save_candidate_triage_hitl_state(waiting, state_path)
    loaded = pydantic_resume.load_candidate_triage_hitl_state(state_path)

    assert loaded.status == "waiting_for_human"
    assert loaded.state == "candidate_triage_plan"
    assert loaded.candidate_triage is None
    assert loaded.prompt_to_user == "请选择一个候选粗筛策略。"
    assert loaded.allowed_actions == ["answer", "resume", "cancel"]

    completed = pydantic_resume.resume_candidate_triage_hitl_from_file(
        state_path=state_path,
        output_path=completed_path,
        answer="选 2，但 SNDK/COHR 必须 deep research",
        must_include_symbols=["SNDK", "COHR"],
    )
    reloaded_completed = pydantic_resume.load_candidate_triage_hitl_state(completed_path)

    assert completed.status == "completed"
    assert completed.state == "candidate_triage_complete"
    assert completed.selection
    assert completed.selection.selected_option_id == "quality"
    assert completed.selection.must_include_symbols == ["US.SNDK", "US.COHR"]
    assert completed.candidate_triage
    assert reloaded_completed.candidate_triage
    assert isinstance(seen["triage_strategy"], candidate_triage.TriageStrategySelection)


def test_pydantic_hitl_cli_create_show_and_resume(tmp_path, monkeypatch, capsys):
    discovery = _fake_theme_discovery_plan("ai", seed_symbol="US.NVDA")
    lightweight = _fake_lightweight_artifact(discovery)
    plan = _fake_triage_plan(discovery, lightweight)
    discovery_path = tmp_path / "discovery.json"
    lightweight_path = tmp_path / "lightweight.json"
    plan_path = tmp_path / "triage_plan.json"
    state_path = tmp_path / "hitl_state.json"
    completed_path = tmp_path / "hitl_completed.json"
    discovery_path.write_text(json.dumps(discovery.model_dump(mode="json")), encoding="utf-8")
    lightweight_path.write_text(json.dumps(lightweight.model_dump(mode="json")), encoding="utf-8")
    plan_path.write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")

    def fake_build_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=None):
        return _fake_candidate_triage_artifact(discovery_arg, lightweight_arg, triage_strategy=triage_strategy)

    monkeypatch.setattr(candidate_triage, "build_candidate_triage_artifact", fake_build_candidate_triage_artifact)

    assert pydantic_hitl_cli.main(
        [
            "create-triage-state",
            "--discovery",
            str(discovery_path),
            "--lightweight",
            str(lightweight_path),
            "--plan",
            str(plan_path),
            "--output",
            str(state_path),
        ]
    ) == 0
    created_out = capsys.readouterr().out
    assert "status: waiting_for_human" in created_out
    assert state_path.exists()

    assert pydantic_hitl_cli.main(["show", "--state", str(state_path)]) == 0
    show_out = capsys.readouterr().out
    assert "options:" in show_out
    assert "coverage_balanced" in show_out

    assert pydantic_hitl_cli.main(
        [
            "resume-triage",
            "--state",
            str(state_path),
            "--output",
            str(completed_path),
            "--answer",
            "选 2",
            "--must-include",
            "SNDK",
            "COHR",
        ]
    ) == 0
    resumed_out = capsys.readouterr().out
    completed = pydantic_resume.load_candidate_triage_hitl_state(completed_path)
    assert "status: completed" in resumed_out
    assert completed.selection
    assert completed.selection.selected_option_id == "quality"
    assert completed.selection.must_include_symbols == ["US.SNDK", "US.COHR"]
    assert completed.candidate_triage


def test_workflow_start_runs_to_candidate_triage_plan_and_waits(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_discovery(theme, *, market="US", theme_description="", required_symbols=None):
        calls.append(
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
            }
        )
        return _fake_theme_discovery_plan(
            theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
            seed_symbol="US.WDC",
            initial_thesis="Storage discovery thesis.",
        )

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)

    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={"theme": "storage"},
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value
    assert result["status"] == "waiting_for_human"
    assert result["human_action"]["kind"] == "select_candidate_triage_strategy"
    assert result["allowed_actions"] == ["answer_human_input", "select_option", "continue", "status", "cancel"]
    assert "answer_human_input" in IA_PORTFOLIO_WORKFLOW_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "select_option" in IA_PORTFOLIO_WORKFLOW_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "build_portfolio_maps" in IA_PORTFOLIO_WORKFLOW_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "候选粗筛策略计划" in result["display_response"]
    assert "目标仓位" not in result["display_response"]
    artifact_types = [item["type"] for item in store.list_artifacts(result["session_id"])]
    assert "theme_discovery" in artifact_types
    assert "futu_lightweight_enrichment" in artifact_types
    assert "candidate_triage_plan" in artifact_types
    assert "portfolio_maps" not in artifact_types
    assert calls[0]["theme"] == "storage"


def test_workflow_start_accepts_free_form_theme_for_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_discovery(theme, *, market="US", theme_description="", required_symbols=None):
        calls.append(theme)
        return _fake_theme_discovery_plan(
            theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
            seed_symbol="US.NVDA",
            initial_thesis="AI free-form discovery thesis.",
        )

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    workflow = _workflow(InvestmentAssistantStore())

    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={"theme": "AI持仓版图规划，包含半导体、存储、网络和电力"},
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value
    assert calls == ["AI持仓版图规划，包含半导体、存储、网络和电力"]


def test_workflow_discover_only_runs_theme_discovery_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_discovery(theme, *, market="US", theme_description="", required_symbols=None):
        calls.append(
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
            }
        )
        return ThemeDiscoveryPlan(
            theme=theme,
            market=market,
            theme_description=theme_description,
            initial_thesis="Robotics discovery thesis before downstream validation.",
            domain_tree=[
                ThemeDomain(
                    key="robotics_systems",
                    name="Robotics systems",
                    thesis="Robot OEMs and deployed systems are direct robotics exposures.",
                    importance="core",
                    subdomains=[
                        ThemeSubdomain(
                            key="surgical_robotics",
                            name="Surgical robotics",
                            thesis="Surgical robots are monetized direct robotics exposure.",
                            importance="high",
                            candidates=[
                                ThemeDomainCandidate(
                                    symbol="ISRG",
                                    role="surgical robotics leader",
                                    rationale="Futu robotics plate candidate.",
                                    priority="must_consider",
                                )
                            ],
                        )
                    ],
                )
            ],
            coverage_requirements=[
                ThemeCoverageRequirement(
                    key="surgical_robotics",
                    name="Surgical robotics",
                    thesis="Evaluate surgical robotics as direct exposure.",
                    candidate_symbols=["US.ISRG"],
                    must_consider_symbols=["US.ISRG"],
                )
            ],
            seed_symbols=[
                ThemeDiscoverySeed(
                    symbol="US.ISRG",
                    market=market,
                    role="surgical robotics leader",
                    rationale="Direct robotics exposure.",
                    subthemes=["Robotics systems", "Surgical robotics"],
                    value_chain_stage="Surgical robotics",
                    exposure_type="direct operating company",
                    exposure_purity="high",
                )
            ],
            plate_keywords=["机器人"],
            pydantic_ai={
                "mode": "pydantic_ai_futu_assisted_theme_discovery_agent",
                "futu_tool_calls": [
                    {
                        "plate_keywords": ["机器人"],
                        "must_check_symbols": ["US.ISRG"],
                        "candidate_count": 12,
                        "plate_match_count": 1,
                    }
                ],
            },
            warnings=["Futu-assisted discovery still requires downstream validation."],
        )

    monkeypatch.setattr(ia_workflow, "build_ai_discovery_v1_plan", fake_discovery)
    store = InvestmentAssistantStore()
    workflow = InvestmentAssistantWorkflow(store=store)

    result = workflow.run(
        tenant="cli:test",
        action="discover",
        payload={
            "theme": "robotics",
            "theme_description": "机器人/具身智能版图",
            "required_symbols": ["ISRG"],
        },
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.THEME_DISCOVERY_COMPLETE.value
    assert result["status"] == "completed"
    assert calls == [
        {
            "theme": "robotics",
            "market": "US",
            "theme_description": "机器人/具身智能版图",
            "required_symbols": ["US.ISRG"],
        }
    ]
    assert result["data"]["theme_discovery"]["initial_thesis"]
    assert result["data"]["theme_discovery_artifact_id"]
    assert "后续步骤已暂停" in result["display_response"]
    assert "US.ISRG" in result["display_response"]
    assert "portfolio_maps" not in result["data"]
    assert "candidate_pool" not in [item["type"] for item in store.list_artifacts(result["session_id"])]
    assert "portfolio_maps" not in [item["type"] for item in store.list_artifacts(result["session_id"])]
    assert result["answer_contract"]["allowed_symbols"] == ["US.ISRG"]
    state_runs = store.list_state_runs(result["session_id"])
    state_run_states = [item["state"] for item in state_runs]
    assert WorkflowState.EXPANDING_THEME.value in state_run_states
    assert WorkflowState.THEME_DISCOVERY_COMPLETE.value in state_run_states


def test_workflow_discover_only_can_use_websearch_discovery_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_websearch_discovery(
        theme,
        *,
        market="US",
        theme_description="",
        required_symbols=None,
        max_searches=None,
        max_results=None,
    ):
        calls.append(
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
                "max_searches": max_searches,
                "max_results": max_results,
            }
        )
        return _fake_theme_discovery_plan(
            theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
            seed_symbol="US.SNDK",
            initial_thesis="Budgeted websearch AI discovery thesis.",
        )

    monkeypatch.setattr(ia_workflow, "build_websearch_discovery_plan", fake_websearch_discovery)
    monkeypatch.setattr(
        ia_workflow,
        "build_ai_discovery_v1_plan",
        lambda *args, **kwargs: pytest.fail("Futu classifier discovery path should not be called"),
    )
    workflow = InvestmentAssistantWorkflow(store=InvestmentAssistantStore())

    result = workflow.run(
        tenant="cli:test",
        action="discover",
        payload={
            "theme": "ai",
            "theme_description": "AI版图持仓建设",
            "required_symbols": ["QQQ", "NVDA"],
            "discovery_mode": "websearch",
            "max_searches": 3,
            "max_results": 4,
        },
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.THEME_DISCOVERY_COMPLETE.value
    assert calls == [
        {
            "theme": "ai",
            "market": "US",
            "theme_description": "AI版图持仓建设",
            "required_symbols": ["US.QQQ", "US.NVDA"],
            "max_searches": 3,
            "max_results": 4,
        }
    ]


def test_workflow_start_with_discovery_only_uses_discover_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_discovery(theme, *, market="US", theme_description="", required_symbols=None):
        return ThemeDiscoveryPlan(
            theme=theme,
            market=market,
            theme_description=theme_description,
            initial_thesis="Space economy discovery thesis.",
            domain_tree=[
                ThemeDomain(
                    key="launch",
                    name="Launch",
                    thesis="Launch providers are direct space exposure.",
                    subdomains=[
                        ThemeSubdomain(
                            key="launch_services",
                            name="Launch services",
                            thesis="Launch services provide direct orbital access exposure.",
                            candidates=[
                                ThemeDomainCandidate(symbol="RKLB", role="launch provider"),
                            ],
                        )
                    ],
                )
            ],
            coverage_requirements=[
                ThemeCoverageRequirement(
                    key="launch_services",
                    name="Launch services",
                    thesis="Launch services should be checked.",
                    candidate_symbols=["US.RKLB"],
                )
            ],
            seed_symbols=[ThemeDiscoverySeed(symbol="US.RKLB", market=market, role="launch provider")],
            warnings=["Discovery only."],
        )

    monkeypatch.setattr(ia_workflow, "build_ai_discovery_v1_plan", fake_discovery)
    workflow = InvestmentAssistantWorkflow(store=InvestmentAssistantStore())

    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={
            "theme": "太空领域",
            "discovery_only": True,
        },
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.THEME_DISCOVERY_COMPLETE.value
    assert result["data"]["theme_discovery"]["seed_symbols"][0]["symbol"] == "US.RKLB"


def test_workflow_preserves_required_symbols_as_discovery_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []

    def fake_discovery(theme, *, market="US", theme_description="", required_symbols=None):
        calls.append(
            {
                "theme": theme,
                "market": market,
                "theme_description": theme_description,
                "required_symbols": required_symbols,
            }
        )
        return _fake_theme_discovery_plan(
            theme,
            market=market,
            theme_description=theme_description,
            required_symbols=required_symbols,
            initial_thesis="AI discovery with required base holdings.",
        )

    monkeypatch.setattr(ia_workflow, "build_ai_discovery_v1_plan", fake_discovery)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={
            "theme_key": "ai",
            "theme_description": "AI target map with storage and power as sub-exposures.",
            "required_symbols": "QQQ, SOXX(SMH), NVDA",
        },
    )

    assert result["success"] is True
    assert calls[0]["required_symbols"] == ["US.QQQ", "US.SOXX", "US.SMH", "US.NVDA"]
    policy = store.latest_artifact(result["session_id"], "policy")
    assert policy["payload"]["theme"] == "ai"
    assert policy["payload"]["required_symbols"] == ["US.QQQ", "US.SOXX", "US.SMH", "US.NVDA"]
    discovery = store.latest_artifact(result["session_id"], "theme_discovery")
    discovery_symbols = [item["symbol"] for item in discovery["payload"]["seed_symbols"]]
    assert discovery_symbols[:4] == ["US.QQQ", "US.SOXX", "US.SMH", "US.NVDA"]


def test_workflow_public_actions_stop_before_candidate_pool_and_maps(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.WDC")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={"theme": "storage"},
    )

    assert result["success"] is True
    assert result["state"] == WorkflowState.NEEDS_CANDIDATE_TRIAGE_STRATEGY.value
    assert result["human_action"]["kind"] == "select_candidate_triage_strategy"
    assert result["allowed_actions"] == ["answer_human_input", "select_option", "continue", "status", "cancel"]
    assert result["answer_contract"]["mode"] == "artifact_only"
    assert result["answer_contract"]["agent_may_rephrase"] is True
    assert result["answer_contract"]["fallback_response_on_validation_failure"] is True
    assert result["display_response"]
    assert "theme_key" not in result["display_response"]
    assert result["fallback_response"] == result["display_response"]
    assert result["agent_brief"]
    assert "Use agent_brief" in result["next_instruction_for_agent"]
    artifact_types = [item["type"] for item in store.list_artifacts(result["session_id"])]
    assert "initial_request" in artifact_types
    assert "policy" in artifact_types
    assert "theme_discovery" in artifact_types
    assert "futu_lightweight_enrichment" in artifact_types
    assert "candidate_triage_plan" in artifact_types
    assert "candidate_pool_initial" not in artifact_types
    assert "candidate_pool" not in artifact_types
    assert "market_context" not in artifact_types
    assert "thesis_synthesis" not in artifact_types
    assert "portfolio_maps" not in artifact_types
    assert "current_portfolio" not in artifact_types
    assert "construction_plan" not in artifact_types
    state_runs = store.list_state_runs(result["session_id"])
    state_run_states = [item["state"] for item in state_runs]
    assert WorkflowState.NEW.value in state_run_states
    assert WorkflowState.EXPANDING_THEME.value in state_run_states
    assert WorkflowState.BUILDING_MARKET_ARTIFACTS.value in state_run_states
    assert WorkflowState.BUILDING_UNBIASED_CANDIDATE_POOL.value in state_run_states
    assert WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value not in state_run_states
    assert all(item["status"] == "completed" for item in state_runs)
    assert result["data"]["state_runs"]


def test_discovery_failure_is_workflow_status_not_tool_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        ia_workflow,
        "build_ai_discovery_v1_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("futu discovery failed")),
    )
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={"theme": "ai"},
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert "这次组合版图生成没有完成" in result["display_response"]
    assert "futu discovery failed" in result["display_response"]
    assert "BUILDING_UNBIASED_CANDIDATE_POOL" not in result["display_response"]
    assert "theme_key" not in result["display_response"]
    assert "workflow_error" in [item["type"] for item in store.list_artifacts(result["session_id"])]
    failed_runs = [item for item in store.list_state_runs(result["session_id"]) if item["status"] == "failed"]
    assert failed_runs
    assert failed_runs[-1]["state"] == WorkflowState.EXPANDING_THEME.value
    assert "futu discovery failed" in failed_runs[-1]["error"]["message"]
    assert result["data"]["state_runs"][-1]["status"] == "failed"


def test_workflow_does_not_invoke_architect_from_public_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.WDC")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    result = workflow.run(
        tenant="cli:test",
        action="start",
        payload={"theme": "storage"},
    )

    artifact_types = [item["type"] for item in store.list_artifacts(result["session_id"])]
    assert result["success"] is True
    assert result["status"] == "waiting_for_human"
    assert "portfolio_maps" not in artifact_types
    assert "workflow_error" not in artifact_types


def test_portfolio_map_architect_is_disabled_in_current_mvp():
    policy = InvestmentPolicy(
        theme="ai",
        target_portfolio_weight=0.5,
        cash_reserve=0.1,
        single_name_limit=0.2,
    )
    pool = CandidatePool(
        theme="ai",
        generated_from=["fake_test_data"],
        candidates=[
            _candidate("US.NVDA", "AI accelerator", score=95),
            _candidate("US.COHR", "optical components and lasers", score=92),
            _candidate("US.MRVL", "custom AI silicon and connectivity", score=88),
        ],
        discovery_thesis="AI bottleneck thesis with optical and connectivity candidates.",
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="optical_networking_components",
                name="Optical networking/components",
                candidate_symbols=["US.COHR"],
                must_consider_symbols=["US.COHR"],
            ),
            ThemeCoverageRequirement(
                key="custom_silicon_connectivity",
                name="Custom silicon/connectivity",
                candidate_symbols=["US.MRVL"],
                must_consider_symbols=["US.MRVL"],
            ),
        ],
    )
    reflection = reflect_candidate_pool(policy, pool, {})

    with pytest.raises(NotImplementedError, match="Portfolio-map architecture is not part"):
        ia_agents.build_portfolio_maps(policy, pool, reflection)


def _write_symbol_architect_material(root, symbol, filing_text, fundamentals):
    symbol_dir = root / "symbols" / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "filing_summary.md").write_text(filing_text, encoding="utf-8")
    (symbol_dir / "filing_summary.meta.json").write_text(
        json.dumps({"artifact_type": "filing_summary", "status": "fresh"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (symbol_dir / "sec_companyfacts.json").write_text(
        json.dumps(
            {
                "artifact_type": "sec_companyfacts",
                "company": {"ticker": symbol.split(".")[-1]},
                "fundamentals": fundamentals,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (symbol_dir / "filing_metadata.json").write_text(
        json.dumps({"artifact_type": "filing_metadata", "symbol": symbol}, ensure_ascii=False),
        encoding="utf-8",
    )
    (symbol_dir / "manifest.json").write_text(
        json.dumps({"artifact_type": "symbol_data_manifest", "layers": {}}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_symbol_deep_research_material(root, symbol, filing_text, fundamentals=None):
    fundamentals = fundamentals or {"ttm_revenue": 1_000_000.0, "ttm_net_income": 100_000.0}
    _write_symbol_architect_material(root, symbol, filing_text, fundamentals)
    symbol_dir = root / "symbols" / symbol
    (symbol_dir / "fmp_company_profile.json").write_text(
        json.dumps(
            {
                "artifact_type": "fmp_company_profile",
                "source_status": "fresh",
                "profile": {
                    "symbol": symbol.split(".")[-1],
                    "companyName": f"{symbol} Inc.",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                    "marketCap": 1_000_000_000,
                    "beta": 1.2,
                    "description": f"{symbol} profile description.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (symbol_dir / "filing_metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "filing_metadata",
                "symbol": symbol,
                "source_status": "fresh",
                "filings": {
                    "latest_10q": {
                        "form": "10-Q",
                        "filing_date": "2026-05-01",
                        "period_of_report": "2026-03-31",
                        "accession_number": f"{symbol}-10q",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (symbol_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "symbol_data_manifest",
                "source_status": "fresh",
                "layers": {
                    "filing_summary": {
                        "status": "fresh",
                        "path": f"symbols/{symbol}/filing_summary.md",
                        "meta_path": f"symbols/{symbol}/filing_summary.meta.json",
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_portfolio_architect_context_uses_triage_not_discovery(tmp_path):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_architect_material(root, "US.NVDA", "NVIDIA filing summary.", {"ttm_revenue": 1_000_000.0})
    _write_symbol_architect_material(root, "US.SNDK", "Sandisk filing summary.", {"ttm_revenue": 2_000_000.0})
    _write_symbol_architect_material(root, "US.WDC", "Western Digital filing summary.", {"ttm_revenue": 3_000_000.0})

    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                evidence_summary=["Futu quote ok."],
                rationale="Required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.SNDK",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["memory_storage"],
                role="NAND and enterprise SSD",
                evidence_summary=["Futu quote ok."],
                rationale="Storage bottleneck candidate.",
            ),
        ],
        watchlist=[
            candidate_triage.CompactTriageDecision(
                symbol="US.WDC",
                bucket="watchlist",
                priority="medium",
                layer_keys=["memory_storage"],
                rationale="Watchlist peer.",
            )
        ],
    )
    policy = InvestmentPolicy(
        theme="ai",
        required_symbols=["US.NVDA"],
        target_portfolio_weight=0.95,
        cash_reserve=0.05,
        single_name_limit=0.2,
    )
    lightweight = {
        "candidates": [
            {"symbol": "US.NVDA", "last_price": 100.0, "trend": "uptrend"},
            {"symbol": "US.SNDK", "last_price": 50.0, "trend": "uptrend"},
        ]
    }

    context, warnings = portfolio_architect.build_architect_context(
        policy=policy,
        triage=triage,
        root=root,
        lightweight=lightweight,
    )

    assert context["excluded_upstream_discovery"] is True
    assert context["input_boundary"]["uses_theme_discovery_directly"] is False
    assert "theme_discovery" not in context
    assert context["eligible_symbols"] == ["US.NVDA", "US.SNDK"]
    assert context["watchlist_symbols"] == ["US.WDC"]
    assert context["symbol_materials"]["US.SNDK"]["filing_summary"] == "Sandisk filing summary."
    assert context["symbol_materials"]["US.SNDK"]["sec_companyfacts"]["fundamentals"]["ttm_revenue"] == 2_000_000.0
    assert context["symbol_materials"]["US.SNDK"]["futu_enrichment"]["trend"] == "uptrend"
    assert warnings == []


def test_portfolio_architect_context_uses_deep_research_as_primary_input(tmp_path):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_architect_material(root, "US.NVDA", "NVIDIA raw filing summary.", {"ttm_revenue": 1_000_000.0})
    _write_symbol_architect_material(root, "US.CRDO", "Credo raw filing summary.", {"ttm_revenue": 2_000_000.0})
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="Required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.CRDO",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["optical_networking"],
                role="Optical networking satellite.",
                rationale="Optical candidate.",
            ),
        ],
    )
    deep_report = deep_research.DeepResearchReport(
        theme="ai",
        research_summary="AI compute and optical candidates were researched.",
        candidate_cards=[
            deep_research.CandidateResearchCard(
                symbol="US.NVDA",
                layer_keys=["compute"],
                exposure_summary="Direct compute platform.",
                filing_takeaways=["Datacenter demand is central."],
                candidate_decision="core_candidate",
                evidence_refs=["filing_summary:US.NVDA"],
            ),
            deep_research.CandidateResearchCard(
                symbol="US.CRDO",
                layer_keys=["optical_networking"],
                exposure_summary="Optical networking exposure but lower confidence.",
                filing_takeaways=["High growth but evidence is less complete."],
                candidate_decision="watchlist",
                evidence_refs=["filing_summary:US.CRDO"],
            ),
        ],
        layer_conclusions=[
            deep_research.LayerResearchConclusion(
                layer_key="optical_networking",
                watchlist_symbols=["US.CRDO"],
                peer_tradeoff_summary="CRDO remains watchlist until evidence improves.",
            )
        ],
    )
    policy = InvestmentPolicy(theme="ai", required_symbols=["US.NVDA"])

    context, warnings = portfolio_architect.build_architect_context(
        policy=policy,
        triage=triage,
        root=root,
        deep_research=deep_report.model_dump(mode="json"),
        deep_research_path=tmp_path / "deep_research_report.json",
    )

    assert context["input_boundary"]["uses_deep_research_report"] is True
    assert context["input_boundary"]["uses_filing_summaries"] is False
    assert context["eligible_symbols"] == ["US.NVDA", "US.CRDO"]
    assert context["deep_research_report"]["candidate_cards"][0]["symbol"] == "US.NVDA"
    assert "filing_summary" not in context["symbol_materials"]["US.NVDA"]
    assert context["symbol_materials"]["US.NVDA"]["deep_research_card"]["candidate_decision"] == "core_candidate"
    assert any("US.CRDO: deep_research candidate_decision is watchlist" in warning for warning in warnings)


def test_portfolio_architect_context_keeps_unresearched_deep_research_candidates(tmp_path):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_architect_material(root, "US.NVDA", "NVIDIA raw filing summary.", {"ttm_revenue": 1_000_000.0})
    _write_symbol_architect_material(root, "US.MSFT", "Microsoft raw filing summary.", {"ttm_revenue": 2_000_000.0})
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="Required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.MSFT",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["cloud_ai_platform", "ai_application_software"],
                role="Cloud and software platform.",
                rationale="Platform peer should remain visible.",
            ),
        ],
    )
    deep_report = deep_research.DeepResearchReport(
        theme="ai",
        research_summary="NVDA was deeply researched; MSFT remained optional.",
        candidate_cards=[
            deep_research.CandidateResearchCard(
                symbol="US.NVDA",
                layer_keys=["compute"],
                exposure_summary="Direct compute platform.",
                filing_takeaways=["Datacenter demand is central."],
                candidate_decision="core_candidate",
                evidence_refs=["filing_summary:US.NVDA"],
            )
        ],
        unresearched_candidates=[
            deep_research.UnresearchedCandidateCard(
                symbol="US.MSFT",
                layer_keys=["cloud_ai_platform", "ai_application_software"],
                intake_action="optional_read_filing_analysis",
                original_priority="high",
                reason="Mega-cap platform remains visible but was not deeply read.",
                evidence_refs=["intake:US.MSFT"],
                data_gaps=["US.MSFT: not deeply researched in this pass."],
            )
        ],
        layer_conclusions=[
            deep_research.LayerResearchConclusion(
                layer_key="compute",
                selected_symbols=["US.NVDA"],
            )
        ],
    )
    policy = InvestmentPolicy(theme="ai", required_symbols=["US.NVDA"])

    context, warnings = portfolio_architect.build_architect_context(
        policy=policy,
        triage=triage,
        root=root,
        deep_research=deep_report.model_dump(mode="json"),
        deep_research_path=tmp_path / "deep_research_report.json",
    )

    assert context["eligible_symbols"] == ["US.NVDA", "US.MSFT"]
    assert context["researched_symbols"] == ["US.NVDA"]
    assert context["unresearched_symbols"] == ["US.MSFT"]
    assert context["deep_research_report"]["unresearched_candidates"][0]["symbol"] == "US.MSFT"
    assert context["symbol_materials"]["US.MSFT"]["research_status"] == "unresearched_lightweight"
    assert "filing_summary" not in context["symbol_materials"]["US.MSFT"]
    assert any("unresearched lightweight candidates" in warning for warning in warnings)


def test_portfolio_architect_deep_research_watchlist_is_not_forced_important():
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="Required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.CRDO",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["optical_networking"],
                role="Optical networking satellite.",
                rationale="Optical candidate.",
            ),
        ],
    )
    selection = portfolio_architect.PostEnrichmentSelection(
        selected_for_portfolio=[
            portfolio_architect.SelectedPortfolioCandidate(
                symbol="US.NVDA",
                conviction="core",
                layer_keys=["compute"],
                role="Compute anchor",
                why_selected=["Deep research marked it as core."],
                evidence_refs=["filing_summary:US.NVDA"],
            )
        ],
        selection_summary="Select the researched core candidate and leave watchlist names out.",
    )

    selected = portfolio_architect._validate_post_enrichment_selection(
        triage,
        selection,
        {"US.NVDA": "core_candidate", "US.CRDO": "watchlist"},
    )

    assert selected == {"US.NVDA"}
    assert not any("US.CRDO" in warning for warning in selection.warnings)


def test_deep_research_context_reads_selected_filing_summaries(tmp_path):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_deep_research_material(
        root,
        "US.NVDA",
        "# US.NVDA Filing Summary\n\n## AI / Data Center Relevance\nNVIDIA datacenter demand remains strong.",
    )
    _write_symbol_deep_research_material(
        root,
        "US.SNDK",
        "# US.SNDK Filing Summary\n\n## Demand Signals\nEnterprise SSD demand improved.",
    )
    intake = {
        "artifact_type": "research_intake_triage",
        "intake_summary": "Read NVDA and SNDK.",
        "must_read_filing_analysis": [
            {
                "symbol": "US.NVDA",
                "original_triage_bucket": "deep_enrichment_queue",
                "layer_keys": ["compute"],
                "priority": "critical",
                "read_action": "must_read_filing_analysis",
                "reason": "Compute anchor.",
            },
            {
                "symbol": "US.SNDK",
                "original_triage_bucket": "deep_enrichment_queue",
                "layer_keys": ["memory_storage"],
                "priority": "high",
                "read_action": "must_read_filing_analysis",
                "reason": "Storage bottleneck.",
            },
        ],
        "optional_read_filing_analysis": [
            {
                "symbol": "US.MSFT",
                "original_triage_bucket": "deep_enrichment_queue",
                "layer_keys": ["cloud_ai_platform", "ai_application_software"],
                "priority": "high",
                "read_action": "optional_read_filing_analysis",
                "reason": "Mega-cap platform remains visible but is not deeply read in this pass.",
                "available_light_materials": ["profile and companyfacts available"],
                "missing_or_stale_materials": ["filing_summary partial"],
            }
        ],
    }
    triage = {
        "theme": "ai",
        "deep_enrichment_queue": [
            {"symbol": "US.NVDA", "priority": "critical", "layer_keys": ["compute"]},
            {"symbol": "US.SNDK", "priority": "high", "layer_keys": ["memory_storage"]},
        ],
    }

    context, warnings = deep_research.build_deep_research_context(
        intake=intake,
        triage=triage,
        root=root,
    )

    assert context["input_boundary"]["uses_filing_summary_markdown"] is True
    assert context["input_boundary"]["forbids_portfolio_weights"] is True
    assert context["researched_symbols"] == ["US.NVDA", "US.SNDK"]
    assert context["unresearched_candidates"][0]["symbol"] == "US.MSFT"
    assert context["unresearched_candidates"][0]["layer_keys"] == [
        "cloud_ai_platform",
        "ai_application_software",
    ]
    assert "datacenter demand" in context["symbol_materials"]["US.NVDA"]["filing_summary"].lower()
    assert context["symbol_materials"]["US.SNDK"]["evidence_refs"]["filing_summary"] == "filing_summary:US.SNDK"
    assert warnings == []


def test_deep_research_builds_and_persists_report(tmp_path, monkeypatch):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_deep_research_material(root, "US.NVDA", "NVIDIA filing summary.", {"ttm_revenue": 1_000_000.0})
    _write_symbol_deep_research_material(root, "US.SNDK", "Sandisk filing summary.", {"ttm_revenue": 2_000_000.0})
    intake_path = tmp_path / "intake.json"
    intake_path.write_text(
        json.dumps(
            {
                "artifact_type": "research_intake_triage",
                "must_read_filing_analysis": [
                    {
                        "symbol": "US.NVDA",
                        "original_triage_bucket": "deep_enrichment_queue",
                        "layer_keys": ["compute"],
                        "priority": "critical",
                        "read_action": "must_read_filing_analysis",
                    },
                    {
                        "symbol": "US.SNDK",
                        "original_triage_bucket": "deep_enrichment_queue",
                        "layer_keys": ["memory_storage"],
                        "priority": "high",
                        "read_action": "must_read_filing_analysis",
                    },
                ],
                "optional_read_filing_analysis": [
                    {
                        "symbol": "US.MSFT",
                        "layer_keys": ["cloud_ai_platform"],
                        "priority": "high",
                        "read_action": "optional_read_filing_analysis",
                        "reason": "Keep platform candidate visible without deep read.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    seen = {}

    def fake_run_deep_research_agent(context):
        seen["context"] = context
        return (
            deep_research.DeepResearchReport(
                theme="ai",
                research_summary="Compute and storage both need validation.",
                candidate_cards=[
                    deep_research.CandidateResearchCard(
                        symbol="US.NVDA",
                        layer_keys=["compute"],
                        intake_action="must_read_filing_analysis",
                        original_priority="critical",
                        theme_exposure="direct",
                        exposure_summary="Direct datacenter compute exposure.",
                        business_quality="excellent",
                        filing_takeaways=["Datacenter demand is central."],
                        key_risks=["High expectations."],
                        candidate_decision="core_candidate",
                        confidence="high",
                        evidence_refs=["filing_summary:US.NVDA", "sec_companyfacts:US.NVDA"],
                    ),
                    deep_research.CandidateResearchCard(
                        symbol="US.SNDK",
                        layer_keys=["memory_storage"],
                        intake_action="must_read_filing_analysis",
                        original_priority="high",
                        theme_exposure="strong",
                        exposure_summary="Enterprise SSD storage exposure.",
                        business_quality="mixed",
                        filing_takeaways=["Storage cycle is improving."],
                        key_risks=["Cyclical pricing."],
                        candidate_decision="satellite_candidate",
                        confidence="medium",
                        evidence_refs=["filing_summary:US.SNDK", "sec_companyfacts:US.SNDK"],
                    ),
                ],
                layer_conclusions=[
                    deep_research.LayerResearchConclusion(
                        layer_key="compute",
                        selected_symbols=["US.NVDA"],
                        peer_tradeoff_summary="NVDA is the compute anchor.",
                    ),
                    deep_research.LayerResearchConclusion(
                        layer_key="memory_storage",
                        selected_symbols=["US.SNDK"],
                        peer_tradeoff_summary="SNDK is the storage expression.",
                    ),
                ],
            ),
            {"model": "test-model", "api_mode": "test"},
            {"requests": 1},
        )

    monkeypatch.setattr(deep_research, "_run_deep_research_agent", fake_run_deep_research_agent)

    report, run = deep_research.build_deep_research_report_from_files(
        intake_path=intake_path,
        root=root,
        output_dir=tmp_path / "run",
    )

    assert seen["context"]["researched_symbols"] == ["US.NVDA", "US.SNDK"]
    assert report.candidate_cards[0].candidate_decision == "core_candidate"
    assert Path(run.report_path).exists()
    persisted = json.loads(Path(run.report_path).read_text(encoding="utf-8"))
    assert persisted["candidate_cards"][1]["symbol"] == "US.SNDK"
    assert persisted["unresearched_candidates"][0]["symbol"] == "US.MSFT"
    assert persisted["unresearched_candidates"][0]["intake_action"] == "optional_read_filing_analysis"
    assert run.status == "fresh"


def test_deep_research_batches_large_symbol_sets(tmp_path, monkeypatch):
    root = tmp_path / "data" / "investment_assistant"
    for symbol in ("US.NVDA", "US.SNDK", "US.MRVL"):
        _write_symbol_deep_research_material(root, symbol, f"{symbol} filing summary.")
    intake_path = tmp_path / "intake.json"
    intake_path.write_text(
        json.dumps(
            {
                "artifact_type": "research_intake_triage",
                "must_read_filing_analysis": [
                    {
                        "symbol": symbol,
                        "layer_keys": ["compute" if symbol == "US.NVDA" else "memory_storage"],
                        "priority": "high",
                        "read_action": "must_read_filing_analysis",
                    }
                    for symbol in ("US.NVDA", "US.SNDK", "US.MRVL")
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run_deep_research_agent(context):
        calls.append(context["researched_symbols"])
        cards = [
            deep_research.CandidateResearchCard(
                symbol=symbol,
                layer_keys=context["symbol_materials"][symbol]["intake_decision"]["layer_keys"],
                exposure_summary=f"{symbol} exposure.",
                filing_takeaways=[f"{symbol} filing takeaway."],
                candidate_decision="satellite_candidate",
                evidence_refs=[f"filing_summary:{symbol}"],
            )
            for symbol in context["researched_symbols"]
        ]
        return (
            deep_research.DeepResearchReport(
                theme="ai",
                research_summary="Batch summary.",
                candidate_cards=cards,
                layer_conclusions=[
                    deep_research.LayerResearchConclusion(
                        layer_key="compute",
                        selected_symbols=[card.symbol for card in cards if "compute" in card.layer_keys],
                    ),
                    deep_research.LayerResearchConclusion(
                        layer_key="memory_storage",
                        selected_symbols=[card.symbol for card in cards if "memory_storage" in card.layer_keys],
                    ),
                ],
            ),
            {"model": "test-model"},
            {"requests": 1, "input_tokens": 10},
        )

    monkeypatch.setattr(deep_research, "_run_deep_research_agent", fake_run_deep_research_agent)

    report, run = deep_research.build_deep_research_report_from_files(
        intake_path=intake_path,
        root=root,
        output_dir=tmp_path / "run",
        batch_size=1,
    )

    assert calls == [["US.NVDA"], ["US.SNDK"], ["US.MRVL"]]
    assert [card.symbol for card in report.candidate_cards] == ["US.NVDA", "US.SNDK", "US.MRVL"]
    assert run.batch_count == 3
    assert len(run.batch_paths) == 3
    assert run.usage["requests"] == 3
    assert run.usage["input_tokens"] == 30


def test_deep_research_validation_rejects_unknown_symbols():
    context = {"researched_symbols": ["US.NVDA"]}
    report = deep_research.DeepResearchReport(
        theme="ai",
        candidate_cards=[
            deep_research.CandidateResearchCard(
                symbol="US.UNKNOWN",
                layer_keys=["compute"],
                exposure_summary="Unknown.",
                filing_takeaways=["Bad symbol."],
                evidence_refs=["filing_summary:US.UNKNOWN"],
            )
        ],
        layer_conclusions=[
            deep_research.LayerResearchConclusion(layer_key="compute", selected_symbols=["US.UNKNOWN"])
        ],
    )

    with pytest.raises(ValueError, match="unknown candidate card symbol"):
        deep_research._validate_deep_research_report(context, report)


def test_portfolio_architect_builds_and_persists_maps_from_triage_wrapper(tmp_path, monkeypatch):
    root = tmp_path / "data" / "investment_assistant"
    _write_symbol_architect_material(root, "US.NVDA", "NVIDIA filing summary.", {"ttm_revenue": 1_000_000.0})
    _write_symbol_architect_material(root, "US.SNDK", "Sandisk filing summary.", {"ttm_revenue": 2_000_000.0})

    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="Required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.SNDK",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["memory_storage"],
                role="NAND and enterprise SSD",
                rationale="Storage bottleneck candidate.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.WDC",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["memory_storage"],
                role="HDD and storage peer",
                rationale="Storage peer for tradeoff audit.",
            ),
        ],
    )
    triage_path = tmp_path / "triage_wrapper.json"
    triage_path.write_text(
        json.dumps(
            {
                "candidate_triage": triage.model_dump(mode="json"),
                "discovery": {"must_not_be_used": True},
                "lightweight": {
                    "candidates": [
                        {"symbol": "US.NVDA", "last_price": 100.0},
                        {"symbol": "US.SNDK", "last_price": 50.0},
                        {"symbol": "US.WDC", "last_price": 60.0},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    policy = InvestmentPolicy(
        theme="ai",
        required_symbols=["US.NVDA"],
        target_portfolio_weight=0.95,
        cash_reserve=0.05,
        single_name_limit=0.5,
    )
    seen = {}

    def fake_run_architect_agent(context):
        seen["context"] = context
        return (
            portfolio_architect.PortfolioArchitectResult(
                theme="ai",
                selection=portfolio_architect.PostEnrichmentSelection(
                    selected_for_portfolio=[
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.NVDA",
                            conviction="core",
                            layer_keys=["compute"],
                            role="GPU accelerator",
                            why_selected=["Required compute leader."],
                            evidence_refs=["filing_summary:US.NVDA"],
                        ),
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.SNDK",
                            conviction="high",
                            layer_keys=["memory_storage"],
                            role="NAND and enterprise SSD",
                            why_selected=["Storage bottleneck candidate."],
                            evidence_refs=["filing_summary:US.SNDK"],
                        ),
                    ],
                    selection_summary="Selected compute and storage representatives before weights.",
                    watchlist_after_enrichment=[
                        portfolio_architect.PostEnrichmentSelectionDecision(
                            symbol="US.WDC",
                            decision="watchlist",
                            priority="high",
                            layer_keys=["memory_storage"],
                            reason="SNDK is selected as the sharper NAND/SSD representative in this test.",
                            substitute_symbols=["US.SNDK"],
                        )
                    ],
                    peer_tradeoffs=[
                        portfolio_architect.PeerTradeoff(
                            layer_key="memory_storage",
                            comparable_symbols=["US.SNDK", "US.WDC"],
                            selected_symbols=["US.SNDK"],
                            non_selected_symbols=["US.WDC"],
                            rationale="Select one storage representative before weights.",
                        )
                    ],
                ),
                portfolio_maps=PortfolioMaps(
                    theme="ai",
                    maps=[
                        PortfolioMap(
                            map_id="ai_test_map",
                            name="AI test map",
                            objective="balanced",
                            sleeve_weight=0.95,
                            positioning="Test positioning.",
                            best_for="Testing.",
                            allocation_logic=["Use triage only."],
                            sleeves=[
                                PortfolioSleeve(
                                    name="Compute",
                                    role="GPU exposure",
                                    target_weight=0.5,
                                    holding_symbols=["US.NVDA"],
                                    rationale="Required compute sleeve.",
                                ),
                                PortfolioSleeve(
                                    name="Storage",
                                    role="Storage bottleneck",
                                    target_weight=0.45,
                                    holding_symbols=["US.SNDK"],
                                    rationale="Storage sleeve.",
                                ),
                            ],
                            holdings=[
                                PortfolioHolding(
                                    symbol="US.NVDA",
                                    target_weight=0.5,
                                    role="GPU accelerator",
                                    rationale="Selected from candidate triage.",
                                    evidence_refs=["triage:US.NVDA", "filing_summary:US.NVDA"],
                                ),
                                PortfolioHolding(
                                    symbol="US.SNDK",
                                    target_weight=0.45,
                                    role="NAND and enterprise SSD",
                                    rationale="Selected from candidate triage.",
                                    evidence_refs=["triage:US.SNDK", "filing_summary:US.SNDK"],
                                ),
                            ],
                            cash_weight=0.05,
                            thesis="Test map.",
                        )
                    ],
                ),
                map_weight_rationales=[
                    portfolio_architect.PortfolioMapWeightRationale(
                        map_id="ai_test_map",
                        holding_count_rationale="Two holdings are enough for this narrow test fixture.",
                        sleeve_weight_rationale=[
                            "Compute receives the larger sleeve because NVDA is the required core.",
                            "Storage receives the remaining sleeve after cash reserve.",
                        ],
                        high_beta_position_sizing=["SNDK is sized below the compute core."],
                        selected_but_unheld_explanations=[],
                        risk_budget_notes=["Single-name limit is respected."],
                    )
                ],
            ),
            {"available": True, "mode": "fake"},
            {"requests": 1},
        )

    monkeypatch.setattr(portfolio_architect, "_run_architect_agent", fake_run_architect_agent)

    architect_result, run = portfolio_architect.build_portfolio_maps_from_files(
        triage_path=triage_path,
        root=root,
        policy=policy,
    )

    assert architect_result.selection.selected_for_portfolio[1].symbol == "US.SNDK"
    assert architect_result.portfolio_maps.maps[0].holdings[1].symbol == "US.SNDK"
    assert seen["context"]["excluded_upstream_discovery"] is True
    assert "discovery" not in seen["context"]
    assert (Path(run.portfolio_maps_path)).exists()
    assert (Path(run.context_path)).exists()
    saved_context = json.loads(Path(run.context_path).read_text(encoding="utf-8"))
    assert saved_context["eligible_symbols"] == ["US.NVDA", "US.SNDK", "US.WDC"]


def test_portfolio_architect_rejects_symbols_outside_triage_deep_queue():
    policy = InvestmentPolicy(
        theme="ai",
        required_symbols=[],
        target_portfolio_weight=0.95,
        cash_reserve=0.05,
        single_name_limit=0.5,
    )
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                role="GPU accelerator",
                rationale="Required compute leader.",
            )
        ],
    )
    maps = PortfolioMaps(
        theme="ai",
        maps=[
            PortfolioMap(
                map_id="bad_map",
                name="Bad map",
                objective="balanced",
                sleeve_weight=0.95,
                positioning="Bad positioning.",
                best_for="No one.",
                sleeves=[
                    PortfolioSleeve(
                        name="Bad sleeve",
                        role="Bad exposure",
                        target_weight=0.5,
                        holding_symbols=["US.MSFT"],
                        rationale="Invalid symbol.",
                    ),
                    PortfolioSleeve(
                        name="Another bad sleeve",
                        role="Bad exposure",
                        target_weight=0.45,
                        holding_symbols=["US.MSFT"],
                        rationale="Invalid symbol.",
                    )
                ],
                holdings=[
                    PortfolioHolding(
                        symbol="US.MSFT",
                        target_weight=0.95,
                        role="Not in deep queue",
                        rationale="Invalid symbol.",
                    )
                ],
                cash_weight=0.05,
                thesis="Invalid map.",
            )
        ],
    )

    with pytest.raises(ValueError, match="not in selection.selected_for_portfolio"):
        portfolio_architect._validate_portfolio_maps(policy, triage, maps, {"US.NVDA"})


def test_portfolio_architect_selection_warns_when_peer_tradeoffs_are_empty():
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                role="GPU accelerator",
                rationale="Required compute leader.",
            )
        ],
    )
    selection = portfolio_architect.PostEnrichmentSelection(
        selected_for_portfolio=[
            portfolio_architect.SelectedPortfolioCandidate(
                symbol="US.NVDA",
                conviction="core",
                role="GPU accelerator",
                why_selected=["Required compute leader."],
                evidence_refs=["triage:US.NVDA"],
            )
        ],
        selection_summary="Selected the only eligible candidate.",
    )

    selected = portfolio_architect._validate_post_enrichment_selection(triage, selection)

    assert selected == {"US.NVDA"}
    assert any("peer_tradeoffs" in warning for warning in selection.warnings)


def test_portfolio_architect_requires_user_required_symbols_in_selection():
    policy = InvestmentPolicy(theme="ai", required_symbols=["US.NVDA"])
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="User-required compute leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.MSFT",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["cloud"],
                role="cloud AI platform",
                rationale="Cloud platform candidate.",
            ),
        ],
    )

    with pytest.raises(ValueError, match="user-required"):
        portfolio_architect._validate_required_symbols_selected(policy, triage, {"US.MSFT"})


def test_portfolio_architect_warns_when_layer_peer_tradeoff_is_incomplete():
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.MSFT",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["L5_cloud_platforms"],
                role="cloud AI platform",
                rationale="Cloud platform leader.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.GOOGL",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["L5_cloud_platforms"],
                role="cloud AI platform peer",
                rationale="Cloud platform peer.",
            ),
        ],
    )
    selection = portfolio_architect.PostEnrichmentSelection(
        selected_for_portfolio=[
            portfolio_architect.SelectedPortfolioCandidate(
                symbol="US.MSFT",
                conviction="core",
                layer_keys=["L5_cloud_platforms"],
                role="cloud AI platform",
                why_selected=["Selected as the representative cloud platform candidate."],
                evidence_refs=["triage:US.MSFT"],
            )
        ],
        watchlist_after_enrichment=[
            portfolio_architect.PostEnrichmentSelectionDecision(
                symbol="US.GOOGL",
                decision="watchlist",
                priority="high",
                layer_keys=["L5_cloud_platforms"],
                reason="Watchlisted as a credible cloud peer but not selected in this map.",
                substitute_symbols=["US.MSFT"],
            )
        ],
        peer_tradeoffs=[
            portfolio_architect.PeerTradeoff(
                layer_key="L1_core",
                comparable_symbols=["US.MSFT"],
                selected_symbols=["US.MSFT"],
                non_selected_symbols=[],
                rationale="Placeholder tradeoff for a different layer.",
            )
        ],
        selection_summary="Selected one cloud representative and watchlisted the other.",
    )

    selected = portfolio_architect._validate_post_enrichment_selection(triage, selection)

    assert selected == {"US.MSFT"}
    assert any("L5_cloud_platforms" in warning for warning in selection.warnings)


def test_portfolio_architect_warns_when_map_omits_high_priority_selected_candidate():
    policy = InvestmentPolicy(
        theme="ai",
        target_portfolio_weight=0.9,
        cash_reserve=0.1,
        single_name_limit=0.6,
    )
    triage = candidate_triage.CandidateTriageArtifact(
        theme="ai",
        market="US",
        deep_enrichment_queue=[
            candidate_triage.TriageCandidateDecision(
                symbol="US.NVDA",
                bucket="deep_enrichment_queue",
                priority="critical",
                layer_keys=["compute"],
                role="GPU accelerator",
                rationale="Core compute candidate.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.CRDO",
                bucket="deep_enrichment_queue",
                priority="high",
                layer_keys=["optical_networking"],
                role="optical DSP and connectivity",
                rationale="Important optical networking candidate.",
            ),
            candidate_triage.TriageCandidateDecision(
                symbol="US.MSFT",
                bucket="deep_enrichment_queue",
                priority="medium",
                layer_keys=["cloud"],
                role="cloud AI platform",
                rationale="Cloud platform candidate.",
            ),
        ],
    )
    maps = PortfolioMaps(
        theme="ai",
        maps=[
            PortfolioMap(
                map_id="ai_without_crdo",
                name="AI without CRDO",
                objective="balanced",
                sleeve_weight=0.9,
                positioning="Test map that omits one high-priority selected candidate.",
                best_for="Validator warning test.",
                allocation_logic=["Use selected candidates but not all in every map."],
                sleeves=[
                    PortfolioSleeve(
                        name="Compute",
                        role="Core compute",
                        target_weight=0.5,
                        holding_symbols=["US.NVDA"],
                        rationale="Core compute sleeve.",
                    ),
                    PortfolioSleeve(
                        name="Cloud",
                        role="Platform exposure",
                        target_weight=0.4,
                        holding_symbols=["US.MSFT"],
                        rationale="Cloud sleeve.",
                    ),
                ],
                holdings=[
                    PortfolioHolding(
                        symbol="US.NVDA",
                        target_weight=0.5,
                        role="GPU accelerator",
                        rationale="Core compute candidate.",
                        evidence_refs=["triage:US.NVDA"],
                    ),
                    PortfolioHolding(
                        symbol="US.MSFT",
                        target_weight=0.4,
                        role="cloud AI platform",
                        rationale="Cloud platform candidate.",
                        evidence_refs=["triage:US.MSFT"],
                    ),
                ],
                cash_weight=0.1,
                thesis="Test thesis.",
            )
        ],
    )

    portfolio_architect._validate_portfolio_maps(
        policy,
        triage,
        maps,
        {"US.NVDA", "US.CRDO", "US.MSFT"},
    )

    assert any("US.CRDO" in warning for warning in maps.warnings)


def test_thesis_synthesis_is_disabled_in_current_mvp():
    policy = InvestmentPolicy(
        theme="ai",
        target_portfolio_weight=0.5,
        cash_reserve=0.1,
        single_name_limit=0.2,
    )
    pool = CandidatePool(
        theme="ai",
        generated_from=["fake_test_data"],
        candidates=[
            _candidate("US.NVDA", "AI accelerator", score=95),
            _candidate("US.AMD", "AI accelerator challenger", score=91),
            _candidate("US.MU", "HBM/DRAM memory", score=90),
            _candidate("US.TSM", "advanced foundry", score=89),
            _candidate("US.VRT", "power and cooling", score=88),
            _candidate("US.COHR", "optical components and lasers", score=92),
        ],
        discovery_thesis="AI bottleneck thesis with optical components.",
        coverage_requirements=[
            ThemeCoverageRequirement(
                key="optical_networking_components",
                name="Optical networking/components",
                candidate_symbols=["US.COHR"],
                must_consider_symbols=["US.COHR"],
            )
        ],
    )

    with pytest.raises(NotImplementedError, match="Thesis synthesis is not part"):
        ia_agents.synthesize_portfolio_thesis(policy, pool, reflect_candidate_pool(policy, pool, {}))


def test_answer_human_input_runs_candidate_triage_but_not_portfolio_maps(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    started = workflow.run("cli:test", "start", payload={"theme": "AI"})

    answer_result = workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=started["session_id"],
        payload={"answer": "选 1"},
    )

    assert started["status"] == "waiting_for_human"
    assert answer_result["success"] is True
    assert answer_result["status"] == "completed"
    assert answer_result["state"] == WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value
    assert answer_result["data"]["selected_triage_strategy"]["selected_option_id"] == "coverage_balanced"
    artifact_types = [item["type"] for item in store.list_artifacts(started["session_id"])]
    assert "triage_strategy_selection" in artifact_types
    assert "candidate_triage" in artifact_types
    assert "selected_map" not in artifact_types
    assert "portfolio_maps" not in artifact_types
    assert "current_portfolio" not in artifact_types
    assert "construction_plan" not in artifact_types


def test_workflow_builds_portfolio_maps_after_candidate_triage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    seen: dict[str, object] = {}

    def fake_build_portfolio_maps_from_triage(*, policy, triage, root=None, lightweight=None, **kwargs):
        seen["policy"] = policy
        seen["triage"] = triage
        seen["lightweight"] = lightweight
        seen["deep_research"] = kwargs.get("deep_research")
        return (
            portfolio_architect.PortfolioArchitectResult(
                theme=policy.theme,
                selection=portfolio_architect.PostEnrichmentSelection(
                    selected_for_portfolio=[
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.NVDA",
                            conviction="core",
                            layer_keys=["primary_domain"],
                            role="workflow test portfolio core",
                            why_selected=["Selected from candidate triage."],
                            evidence_refs=["candidate_triage:US.NVDA"],
                        )
                    ],
                    selection_summary="Selected one workflow-test candidate before weights.",
                    peer_tradeoffs=[
                        portfolio_architect.PeerTradeoff(
                            layer_key="primary_domain",
                            comparable_symbols=["US.NVDA"],
                            selected_symbols=["US.NVDA"],
                            non_selected_symbols=[],
                            rationale="Only deep-research candidate in the test fixture.",
                        )
                    ],
                ),
                portfolio_maps=PortfolioMaps(
                    theme=policy.theme,
                    maps=[
                        PortfolioMap(
                            map_id="ai_workflow_test_map",
                            name="AI workflow test map",
                            objective=policy.objective,
                            sleeve_weight=policy.target_portfolio_weight,
                            positioning="Workflow integration test map.",
                            best_for="Testing Hermes workflow wiring.",
                            allocation_logic=["Use only saved candidate triage."],
                            sleeves=[
                                PortfolioSleeve(
                                    name="Primary domain",
                                    role="Workflow test sleeve",
                                    target_weight=policy.target_portfolio_weight,
                                    holding_symbols=["US.NVDA"],
                                    rationale="Single test sleeve.",
                                )
                            ],
                            holdings=[
                                PortfolioHolding(
                                    symbol="US.NVDA",
                                    target_weight=policy.target_portfolio_weight,
                                    role="workflow test portfolio core",
                                    rationale="Selected from candidate triage artifact.",
                                    evidence_refs=["candidate_triage:US.NVDA"],
                                )
                            ],
                            cash_weight=policy.cash_reserve,
                            thesis="Workflow test thesis.",
                        )
                    ],
                ),
                map_weight_rationales=[
                    portfolio_architect.PortfolioMapWeightRationale(
                        map_id="ai_workflow_test_map",
                        holding_count_rationale="The test fixture has one eligible symbol.",
                        sleeve_weight_rationale=["The single sleeve receives the full target weight."],
                    )
                ],
            ),
            portfolio_architect.PortfolioArchitectRunArtifact(
                root=str(tmp_path / "data" / "investment_assistant"),
                context_path=str(tmp_path / "context.json"),
                portfolio_maps_path=str(tmp_path / "portfolio_maps.json"),
                eligible_symbols=["US.NVDA"],
                pydantic_ai={"mock": True},
            ),
        )

    monkeypatch.setattr(ia_workflow, "build_portfolio_maps_from_triage", fake_build_portfolio_maps_from_triage)

    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    started = workflow.run(
        "cli:test",
        "start",
        payload={
            "theme": "AI",
            "target_portfolio_weight": 0.95,
            "cash_reserve": 0.05,
            "single_name_limit": 0.5,
        },
    )
    triaged = workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=started["session_id"],
        payload={"answer": "选 1"},
    )
    result = workflow.run(
        "cli:test",
        "build_portfolio_maps",
        session_id=started["session_id"],
        payload={},
    )

    assert triaged["state"] == WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value
    assert result["success"] is True
    assert result["status"] == "waiting_for_human"
    assert result["state"] == WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value
    assert result["human_action"]["kind"] == "select_portfolio_map"
    assert "select_option" in result["allowed_actions"]
    assert result["data"]["map_ids"] == ["ai_workflow_test_map"]
    assert result["answer_contract"]["allowed_map_ids"] == ["ai_workflow_test_map"]
    assert "目标组合版图草案" in result["display_response"]
    assert seen["policy"].target_portfolio_weight == 0.95
    assert seen["triage"].theme == "ai"
    assert isinstance(seen["lightweight"], dict)
    assert seen["deep_research"] == {}
    artifact_types = [item["type"] for item in store.list_artifacts(started["session_id"])]
    assert "portfolio_architect_result" in artifact_types
    assert "portfolio_architect_run" in artifact_types
    assert "portfolio_maps" not in artifact_types

    selected = workflow.run(
        "cli:test",
        "select_option",
        session_id=started["session_id"],
        payload={"answer": "选 ai_workflow_test_map"},
    )

    assert selected["success"] is True
    assert selected["status"] == "completed"
    assert selected["state"] == WorkflowState.TARGET_PORTFOLIO_MAP_SELECTED.value
    assert selected["data"]["selected_map_id"] == "ai_workflow_test_map"
    assert "已记录你的目标组合版图选择" in selected["display_response"]
    artifact_types = [item["type"] for item in store.list_artifacts(started["session_id"])]
    assert "selected_portfolio_map" in artifact_types


def _install_revision_workflow_fakes(monkeypatch, tmp_path):
    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)

    def fake_build_portfolio_maps_from_triage(*, policy, triage, root=None, lightweight=None, **kwargs):
        return (
            portfolio_architect.PortfolioArchitectResult(
                theme=policy.theme,
                selection=portfolio_architect.PostEnrichmentSelection(
                    selected_for_portfolio=[
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.NVDA",
                            conviction="core",
                            layer_keys=["compute"],
                            role="compute anchor",
                            why_selected=["Core AI compute exposure."],
                            evidence_refs=["deep:US.NVDA"],
                        ),
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.MU",
                            conviction="high",
                            layer_keys=["memory"],
                            role="HBM memory",
                            why_selected=["Memory bandwidth exposure."],
                            evidence_refs=["deep:US.MU"],
                        ),
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.SNDK",
                            conviction="satellite",
                            layer_keys=["storage"],
                            role="enterprise storage",
                            why_selected=["Storage throughput exposure."],
                            evidence_refs=["deep:US.SNDK"],
                        ),
                    ],
                    selection_summary="Selected compute, memory, and storage candidates before weights.",
                    peer_tradeoffs=[
                        portfolio_architect.PeerTradeoff(
                            layer_key="memory_storage",
                            comparable_symbols=["US.MU", "US.SNDK"],
                            selected_symbols=["US.MU", "US.SNDK"],
                            rationale="Both express different bottleneck exposures.",
                        )
                    ],
                ),
                portfolio_maps=PortfolioMaps(
                    theme=policy.theme,
                    maps=[
                        PortfolioMap(
                            map_id="ai_revision_map",
                            name="AI revision base map",
                            objective=policy.objective,
                            sleeve_weight=0.95,
                            positioning="Revision test base map.",
                            best_for="Testing map revision.",
                            allocation_logic=["Use three researched holdings."],
                            sleeves=[
                                PortfolioSleeve(
                                    name="Compute",
                                    role="Core compute",
                                    target_weight=0.50,
                                    holding_symbols=["US.NVDA"],
                                    rationale="Compute anchor.",
                                ),
                                PortfolioSleeve(
                                    name="Memory and storage",
                                    role="Bottleneck exposure",
                                    target_weight=0.45,
                                    holding_symbols=["US.MU", "US.SNDK"],
                                    rationale="Memory and storage bottleneck sleeve.",
                                ),
                            ],
                            holdings=[
                                PortfolioHolding(
                                    symbol="US.NVDA",
                                    target_weight=0.50,
                                    role="compute anchor",
                                    rationale="Core compute evidence.",
                                    evidence_refs=["deep:US.NVDA"],
                                ),
                                PortfolioHolding(
                                    symbol="US.MU",
                                    target_weight=0.25,
                                    role="HBM memory",
                                    rationale="Memory evidence.",
                                    evidence_refs=["deep:US.MU"],
                                ),
                                PortfolioHolding(
                                    symbol="US.SNDK",
                                    target_weight=0.20,
                                    role="enterprise storage",
                                    rationale="Storage evidence.",
                                    evidence_refs=["deep:US.SNDK"],
                                ),
                            ],
                            cash_weight=0.05,
                            thesis="Revision workflow base thesis.",
                        )
                    ],
                ),
                map_weight_rationales=[
                    portfolio_architect.PortfolioMapWeightRationale(
                        map_id="ai_revision_map",
                        holding_count_rationale="Three holdings keep the fixture compact.",
                        sleeve_weight_rationale=["Compute receives 50%, memory/storage receives 45%."],
                        risk_budget_notes=["Single-name test limit is relaxed by payload."],
                    )
                ],
            ),
            portfolio_architect.PortfolioArchitectRunArtifact(
                root=str(tmp_path / "data" / "investment_assistant"),
                context_path=str(tmp_path / "context.json"),
                portfolio_maps_path=str(tmp_path / "portfolio_maps.json"),
                eligible_symbols=["US.NVDA", "US.MU", "US.SNDK"],
                pydantic_ai={"mock": True},
            ),
        )

    monkeypatch.setattr(ia_workflow, "build_portfolio_maps_from_triage", fake_build_portfolio_maps_from_triage)


def _prepare_selected_revision_workflow(tmp_path, monkeypatch) -> tuple[InvestmentAssistantWorkflow, InvestmentAssistantStore, str]:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _install_revision_workflow_fakes(monkeypatch, tmp_path)
    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    started = workflow.run(
        "cli:test",
        "start",
        payload={
            "theme": "AI",
            "target_portfolio_weight": 0.95,
            "cash_reserve": 0.05,
            "single_name_limit": 0.60,
        },
    )
    workflow.run("cli:test", "answer_human_input", session_id=started["session_id"], payload={"answer": "选 1"})
    workflow.run("cli:test", "build_portfolio_maps", session_id=started["session_id"], payload={})
    workflow.run(
        "cli:test",
        "select_option",
        session_id=started["session_id"],
        payload={"answer": "选 ai_revision_map"},
    )
    return workflow, store, started["session_id"]


def test_workflow_revises_selected_portfolio_map_and_confirms(tmp_path, monkeypatch):
    workflow, store, session_id = _prepare_selected_revision_workflow(tmp_path, monkeypatch)

    def fake_build_revision(*, user_request, base_map, architect_result, policy, deep_research=None):
        assert "MU" in user_request
        assert base_map.map_id == "ai_revision_map"
        revised_map = PortfolioMap.model_validate(
            {
                **base_map.model_dump(mode="json"),
                "holdings": [
                    {
                        "symbol": "US.NVDA",
                        "target_weight": 0.40,
                        "role": "compute anchor",
                        "rationale": "Reduced to fund higher memory/storage exposure.",
                        "evidence_refs": ["deep:US.NVDA"],
                    },
                    {
                        "symbol": "US.MU",
                        "target_weight": 0.30,
                        "role": "HBM memory",
                        "rationale": "Increased per user request.",
                        "evidence_refs": ["deep:US.MU"],
                    },
                    {
                        "symbol": "US.SNDK",
                        "target_weight": 0.25,
                        "role": "enterprise storage",
                        "rationale": "Increased per user request.",
                        "evidence_refs": ["deep:US.SNDK"],
                    },
                ],
                "sleeves": [
                    {
                        "name": "Compute",
                        "role": "Core compute",
                        "target_weight": 0.40,
                        "holding_symbols": ["US.NVDA"],
                        "rationale": "Still the compute anchor.",
                    },
                    {
                        "name": "Memory and storage",
                        "role": "Bottleneck exposure",
                        "target_weight": 0.55,
                        "holding_symbols": ["US.MU", "US.SNDK"],
                        "rationale": "Raised memory/storage sleeve.",
                    },
                ],
                "allocation_logic": ["Increase memory/storage by reducing compute concentration."],
            }
        )
        patch = portfolio_revision.PortfolioRevisionPatch(
            base_map_id=base_map.map_id,
            user_request=user_request,
            revision_intent="Raise MU and SNDK weights.",
            edits=[
                portfolio_revision.PortfolioRevisionEdit(
                    edit_type="adjust_weight",
                    target=portfolio_revision.RevisionTarget(kind="symbol", id="US.MU", label="MU"),
                    direction="increase",
                    magnitude=portfolio_revision.RevisionMagnitude(kind="ai_decide"),
                ),
                portfolio_revision.PortfolioRevisionEdit(
                    edit_type="adjust_weight",
                    target=portfolio_revision.RevisionTarget(kind="symbol", id="US.SNDK", label="SNDK"),
                    direction="increase",
                    magnitude=portfolio_revision.RevisionMagnitude(kind="ai_decide"),
                ),
            ],
        )
        revision = portfolio_revision.PortfolioMapRevision(
            base_map_id=base_map.map_id,
            patch_id=patch.patch_id,
            revised_map=revised_map,
            change_summary=["Raised MU and SNDK while keeping total sleeve and cash unchanged."],
            weight_changes=[
                portfolio_revision.RevisionWeightChange(
                    symbol="US.MU",
                    old_weight=0.25,
                    new_weight=0.30,
                    direction="increase",
                    reason="User requested more memory exposure.",
                ),
                portfolio_revision.RevisionWeightChange(
                    symbol="US.SNDK",
                    old_weight=0.20,
                    new_weight=0.25,
                    direction="increase",
                    reason="User requested more storage exposure.",
                ),
                portfolio_revision.RevisionWeightChange(
                    symbol="US.NVDA",
                    old_weight=0.50,
                    new_weight=0.40,
                    direction="decrease",
                    reason="Funding source.",
                ),
            ],
            funding_sources=["Reduced US.NVDA by 10 percentage points."],
            tradeoff_explanation=["Higher memory/storage exposure increases bottleneck cyclicality."],
            risk_delta=["More exposure to memory and storage cycle."],
            reduced_or_removed=["US.NVDA"],
        )
        run = portfolio_revision.PortfolioRevisionRunArtifact(
            base_map_id=base_map.map_id,
            allowed_symbols=["US.NVDA", "US.MU", "US.SNDK"],
        )
        return patch, revision, run

    monkeypatch.setattr(ia_workflow, "build_portfolio_revision_from_artifacts", fake_build_revision)

    revised = workflow.run(
        "cli:test",
        "revise_portfolio_map",
        session_id=session_id,
        payload={"request": "我想把 MU，SNDK 的比例提高一些"},
    )

    assert revised["success"] is True
    assert revised["status"] == "waiting_for_human"
    assert revised["state"] == WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value
    assert revised["human_action"]["kind"] == "confirm_portfolio_revision"
    assert "portfolio_map_revision" in revised["data"]
    assert "确认" in revised["display_response"]

    confirmed = workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=session_id,
        payload={"answer": "确认"},
    )

    assert confirmed["success"] is True
    assert confirmed["state"] == WorkflowState.TARGET_PORTFOLIO_MAP_REVISION_SELECTED.value
    assert confirmed["data"]["selected_portfolio_map_revision"]["selected_map"]["holdings"][1]["symbol"] == "US.MU"
    artifact_types = [item["type"] for item in store.list_artifacts(session_id)]
    assert "portfolio_revision_patch" in artifact_types
    assert "portfolio_map_revision" in artifact_types
    assert "selected_portfolio_map_revision" in artifact_types


def test_workflow_revision_can_clarify_then_generate_review(tmp_path, monkeypatch):
    workflow, _store, session_id = _prepare_selected_revision_workflow(tmp_path, monkeypatch)
    calls = {"count": 0}

    def fake_build_revision(*, user_request, base_map, architect_result, policy, deep_research=None):
        calls["count"] += 1
        patch = portfolio_revision.PortfolioRevisionPatch(
            base_map_id=base_map.map_id,
            user_request=user_request,
            revision_intent="Clarify whether storage increase should be small or aggressive.",
            edits=[
                portfolio_revision.PortfolioRevisionEdit(
                    edit_type="adjust_weight",
                    target=portfolio_revision.RevisionTarget(kind="symbol", id="US.SNDK", label="SNDK"),
                    direction="increase",
                    magnitude=portfolio_revision.RevisionMagnitude(kind="unspecified"),
                )
            ],
            needs_clarification=calls["count"] == 1,
            clarification_question="SNDK 要小幅提高，还是由 AI 在风险预算内决定？",
            clarification_options=["小幅提高", "让 AI 决定"],
        )
        if calls["count"] == 1:
            return patch, None, portfolio_revision.PortfolioRevisionRunArtifact(
                status="needs_clarification",
                base_map_id=base_map.map_id,
                allowed_symbols=["US.NVDA", "US.MU", "US.SNDK"],
            )
        revised_map = PortfolioMap.model_validate(
            {
                **base_map.model_dump(mode="json"),
                "holdings": [
                    {
                        "symbol": "US.NVDA",
                        "target_weight": 0.45,
                        "role": "compute anchor",
                        "rationale": "Reduced to fund SNDK.",
                        "evidence_refs": ["deep:US.NVDA"],
                    },
                    {
                        "symbol": "US.MU",
                        "target_weight": 0.25,
                        "role": "HBM memory",
                        "rationale": "Kept stable.",
                        "evidence_refs": ["deep:US.MU"],
                    },
                    {
                        "symbol": "US.SNDK",
                        "target_weight": 0.25,
                        "role": "enterprise storage",
                        "rationale": "Raised after clarification.",
                        "evidence_refs": ["deep:US.SNDK"],
                    },
                ],
                "sleeves": [
                    {
                        "name": "Compute",
                        "role": "Core compute",
                        "target_weight": 0.45,
                        "holding_symbols": ["US.NVDA"],
                        "rationale": "Still core compute.",
                    },
                    {
                        "name": "Memory and storage",
                        "role": "Bottleneck exposure",
                        "target_weight": 0.50,
                        "holding_symbols": ["US.MU", "US.SNDK"],
                        "rationale": "Raised storage exposure.",
                    },
                ],
            }
        )
        revision = portfolio_revision.PortfolioMapRevision(
            base_map_id=base_map.map_id,
            patch_id=patch.patch_id,
            revised_map=revised_map,
            change_summary=["Raised SNDK after user clarification."],
            funding_sources=["Reduced US.NVDA."],
            tradeoff_explanation=["More storage exposure, less compute concentration."],
            risk_delta=["Slightly higher storage cyclicality."],
            reduced_or_removed=["US.NVDA"],
        )
        return patch, revision, portfolio_revision.PortfolioRevisionRunArtifact(
            base_map_id=base_map.map_id,
            allowed_symbols=["US.NVDA", "US.MU", "US.SNDK"],
        )

    monkeypatch.setattr(ia_workflow, "build_portfolio_revision_from_artifacts", fake_build_revision)

    clarification = workflow.run(
        "cli:test",
        "revise_portfolio_map",
        session_id=session_id,
        payload={"request": "我想把 SNDK 提高"},
    )
    assert clarification["state"] == WorkflowState.NEEDS_PORTFOLIO_REVISION_CLARIFICATION.value
    assert clarification["human_action"]["kind"] == "clarify_portfolio_revision"
    assert "需要你确认" in clarification["display_response"]

    reviewed = workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=session_id,
        payload={"answer": "让 AI 在风险预算内决定"},
    )
    assert reviewed["success"] is True
    assert reviewed["state"] == WorkflowState.NEEDS_PORTFOLIO_REVISION_REVIEW.value
    assert reviewed["human_action"]["kind"] == "confirm_portfolio_revision"
    assert calls["count"] == 2


def test_portfolio_weight_formula_allocates_from_ai_scores_deterministically():
    portfolio_map = PortfolioMap(
        map_id="formula_test_map",
        name="Formula test map",
        objective="balanced",
        sleeve_weight=0.12,
        positioning="Formula test.",
        best_for="Formula unit test.",
        sleeves=[
            PortfolioSleeve(
                name="Memory Storage",
                role="AI memory/storage bottleneck",
                target_weight=0.12,
                holding_symbols=["US.MU", "US.SNDK"],
                rationale="Test sleeve.",
            )
        ],
        holdings=[
            PortfolioHolding(
                symbol="US.MU",
                target_weight=0.09,
                role="HBM/DRAM anchor",
                rationale="Test MU rationale.",
                evidence_refs=["test:US.MU"],
            ),
            PortfolioHolding(
                symbol="US.SNDK",
                target_weight=0.03,
                role="NAND/SSD satellite",
                rationale="Test SNDK rationale.",
                evidence_refs=["test:US.SNDK"],
            ),
        ],
        cash_weight=0.88,
        thesis="Formula test thesis.",
    )
    context = portfolio_weight_formula.build_formula_context(
        portfolio_map=portfolio_map,
        deep_research={},
        user_intent="test 3:1",
        single_name_limit=0.15,
        portfolio_style="balanced",
    )
    assert context["policy"]["portfolio_style"] == "balanced"
    assert {
        option["style"]
        for option in context["policy"]["available_portfolio_styles"]
    } >= {"balanced", "conviction", "bottleneck_barbell", "concentrated_growth"}
    scoring = portfolio_weight_formula.PortfolioWeightScoring(
        theme="ai",
        scoring_intent="test deterministic allocation",
        sleeve_scores=[
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="memory_storage",
                sleeve_name="Memory Storage",
                holding_symbols=["US.MU", "US.SNDK"],
                importance_score=1,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Test sleeve score.",
                why_not_higher="Already high.",
                why_not_lower="Core bottleneck.",
                evidence_refs=["test:sleeve"],
            )
        ],
        candidate_scores=[
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.MU",
                sleeve_key="memory_storage",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                market_signal=1,
                valuation_adjustment=1,
                liquidity_score=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Test MU score.",
                why_not_higher="Risk cap.",
                why_not_lower="Core memory score.",
                evidence_refs=["test:US.MU"],
            ),
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.SNDK",
                sleeve_key="memory_storage",
                role_importance=1 / 3,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                market_signal=1,
                valuation_adjustment=1,
                liquidity_score=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Test SNDK score.",
                why_not_higher="Satellite score.",
                why_not_lower="Storage relevance.",
                evidence_refs=["test:US.SNDK"],
            ),
        ],
    )

    report = portfolio_weight_formula.allocate_from_scoring(
        context=context,
        scoring=scoring,
        single_name_limit=0.15,
    )

    weights = {item.symbol: item.target_weight for item in report.candidate_allocations}
    assert weights == {"US.MU": 0.09, "US.SNDK": 0.03}
    assert report.formula["candidate_raw_score"].startswith("role_importance")
    assert report.formula["sleeve_normalization"]
    assert report.formula["candidate_normalization"]
    assert report.calculation_steps["allocation_pipeline"] == [
        "score",
        "style_adjust",
        "normalize_with_floors_and_caps",
        "round_largest_remainder",
        "validate",
    ]
    assert report.calculation_steps["policy_inputs"] == {
        "sleeve_weight": 0.12,
        "cash_weight": 0.88,
        "single_name_limit": 0.15,
        "portfolio_style": "balanced",
        "sleeve_score_exponent": 1.0,
        "candidate_score_exponent": 1.0,
        "precision": 0.001,
    }
    assert report.reference_comparison[0].delta == 0
    assert report.reference_comparison[1].delta == 0


def test_portfolio_weight_formula_initial_context_allocates_without_reference_map():
    deep_research = {
        "theme": "AI",
        "cross_layer_thesis": ["Memory and networking are key AI bottlenecks."],
        "layer_conclusions": [
            {
                "layer_key": "memory_storage",
                "layer_name": "Memory and Storage",
                "peer_tradeoff_summary": "Memory bottleneck layer.",
                "selected_symbols": ["US.MU", "US.SNDK"],
            }
        ],
        "candidate_cards": [
            {
                "symbol": "US.MU",
                "candidate_decision": "high_conviction_candidate",
                "layer_keys": ["memory_storage"],
                "theme_exposure": "HBM/DRAM anchor",
                "business_quality": "strong",
                "exposure_summary": "Direct memory bottleneck.",
                "evidence_refs": ["filing_summary:US.MU"],
            },
            {
                "symbol": "US.SNDK",
                "candidate_decision": "satellite_candidate",
                "layer_keys": ["memory_storage"],
                "theme_exposure": "NAND/SSD satellite",
                "business_quality": "medium",
                "exposure_summary": "Enterprise SSD exposure.",
                "evidence_refs": ["filing_summary:US.SNDK"],
            },
            {
                "symbol": "US.WDC",
                "candidate_decision": "watchlist",
                "layer_keys": ["memory_storage"],
                "theme_exposure": "HDD storage watchlist",
                "business_quality": "medium",
                "exposure_summary": "Mass capacity storage exposure.",
                "evidence_refs": ["filing_summary:US.WDC"],
            },
        ],
        "unresearched_candidates": [
            {
                "symbol": "US.MSFT",
                "layer_keys": ["memory_storage"],
                "intake_action": "optional_read_filing_analysis",
                "original_priority": "high",
                "reason": "Platform candidate not deeply researched in this pass.",
                "evidence_refs": ["intake:US.MSFT"],
                "data_gaps": ["US.MSFT: not deeply researched in this pass."],
            }
        ],
    }
    with pytest.raises(portfolio_weight_formula.ResearchCompletenessError) as exc_info:
        portfolio_weight_formula.build_initial_formula_context(
            deep_research=deep_research,
            user_intent="initial formula map",
            sleeve_weight=0.12,
            cash_weight=0.88,
        )
    assert [issue["symbol"] for issue in exc_info.value.issues] == ["US.MSFT"]

    context = portfolio_weight_formula.build_initial_formula_context(
        deep_research=deep_research,
        user_intent="initial formula map",
        sleeve_weight=0.12,
        cash_weight=0.88,
        portfolio_style="conviction",
        allow_incomplete_research=True,
    )

    assert context["context_mode"] == "initial_map_weight_generation"
    assert context["policy"]["portfolio_style"] == "conviction"
    assert context["policy"]["portfolio_style_profile"]["candidate_score_exponent"] > 1
    assert "reference_weights_for_code_only" not in context
    expected_symbols = {
        item["symbol"]
        for item in context["portfolio_structure_without_target_weights"]["holdings"]
    }
    assert expected_symbols == {"US.MU", "US.SNDK", "US.MSFT"}
    assert context["research_cards"]["US.MSFT"]["research_status"] == "unresearched_lightweight"

    scoring = portfolio_weight_formula.PortfolioWeightScoring(
        theme="AI",
        scoring_intent="initial formula map",
        sleeve_scores=[
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="memory_storage",
                sleeve_name="Memory and Storage",
                holding_symbols=["US.MU", "US.SNDK", "US.MSFT"],
                importance_score=1,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Memory is the only test sleeve.",
                why_not_higher="No other test budget.",
                why_not_lower="Only sleeve in context.",
                evidence_refs=["test:sleeve"],
            )
        ],
        candidate_scores=[
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.MU",
                sleeve_key="memory_storage",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                market_signal=1,
                valuation_adjustment=1,
                liquidity_score=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Core memory score.",
                why_not_higher="Single-name cap.",
                why_not_lower="Core bottleneck.",
                evidence_refs=["filing_summary:US.MU"],
            ),
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.SNDK",
                sleeve_key="memory_storage",
                role_importance=1 / 3,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                market_signal=1,
                valuation_adjustment=1,
                liquidity_score=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Satellite storage score.",
                why_not_higher="Lower role importance.",
                why_not_lower="Relevant storage candidate.",
                evidence_refs=["filing_summary:US.SNDK"],
            ),
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.MSFT",
                sleeve_key="memory_storage",
                role_importance=1 / 6,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                market_signal=1,
                valuation_adjustment=1,
                liquidity_score=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Lightweight candidate score.",
                why_not_higher="Not deeply researched.",
                why_not_lower="Still visible to downstream scoring.",
                evidence_refs=["intake:US.MSFT"],
            ),
        ],
    )
    report = portfolio_weight_formula.allocate_from_scoring(
        context=context,
        scoring=scoring,
        single_name_limit=0.15,
    )

    weights = {item.symbol: item.target_weight for item in report.candidate_allocations}
    assert weights == {"US.MU": 0.094, "US.SNDK": 0.019, "US.MSFT": 0.007}
    assert report.reference_comparison == []


def _monitor_test_map(holdings):
    return {
        "map_id": "monitor_test_map",
        "cash_weight": 0.05,
        "holdings": [
            {
                "symbol": symbol,
                "target_weight": weight,
                "sleeve_key": sleeve_key,
                "role": role,
                "rationale": f"{symbol} target rationale.",
                "evidence_refs": [f"test:{symbol}"],
            }
            for symbol, weight, sleeve_key, role in holdings
        ],
    }


def _monitor_market_data(prices, *, extended=False):
    market = {}
    for symbol, price in prices.items():
        bars = []
        for index in range(80):
            if extended:
                close = price * (0.40 + 0.60 * index / 79)
            else:
                close = price * (1 + (((index % 6) - 3) * 0.002))
            bars.append(
                portfolio_monitor.PriceBar(
                    date=f"2026-03-{index + 1:02d}",
                    open=close * 0.99,
                    high=close * 1.01,
                    low=close * 0.99,
                    close=close,
                    volume=1_000_000,
                )
            )
        bars[-1].close = price
        bars[-1].high = price * 1.01
        bars[-1].low = price * 0.99
        market[symbol] = portfolio_monitor.MarketPriceData(
            symbol=symbol,
            last_price=price,
            update_time="2026-06-17T00:00:00+00:00",
            kline=bars,
        )
    return market


def test_portfolio_monitor_generates_add_plan_for_underweight_position():
    portfolio_map = _monitor_test_map(
        [
            ("US.NVDA", 0.10, "core_ai", "core compute"),
            ("US.MU", 0.08, "memory_storage", "high conviction memory"),
        ]
    )
    portfolio = CurrentPortfolio(
        total_assets=100_000,
        cash=15_000,
        holdings=[
            CurrentHolding(symbol="US.NVDA", quantity=80, market_value=12_000, can_sell_qty=80),
        ],
        data_asof="2026-06-17T00:00:00+00:00",
        source="test",
    )
    market = _monitor_market_data({"US.NVDA": 150, "US.MU": 100})

    result = portfolio_monitor.monitor_portfolio(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        market_data=market,
    )

    actions = {action.symbol: action for action in result.rebalance_plan.actions}
    assert actions["US.MU"].action == "add"
    assert actions["US.MU"].reason_code == "underweight"
    assert actions["US.MU"].quantity == 80
    assert actions["US.MU"].target_trade_value == 8000
    assert result.rebalance_plan.cash_required == 8000
    assert result.rebalance_plan.post_trade_cash == 7000
    assert result.rebalance_plan.simulated_orders[0].code == "US.MU"
    assert result.rebalance_plan.simulated_orders[0].side == "BUY"


def test_portfolio_monitor_trims_overweight_position_back_to_band_when_trend_is_strong():
    portfolio_map = _monitor_test_map([("US.NVDA", 0.10, "core_ai", "core compute")])
    portfolio = CurrentPortfolio(
        total_assets=100_000,
        cash=10_000,
        holdings=[
            CurrentHolding(symbol="US.NVDA", quantity=160, market_value=16_000, can_sell_qty=160),
        ],
        data_asof="2026-06-17T00:00:00+00:00",
        source="test",
    )
    market = _monitor_market_data({"US.NVDA": 100}, extended=True)

    result = portfolio_monitor.monitor_portfolio(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        market_data=market,
    )

    action = next(item for item in result.rebalance_plan.actions if item.symbol == "US.NVDA")
    assert action.action == "trim"
    assert action.reason_code == "overweight_take_profit_to_band"
    assert action.quantity == 40
    assert action.target_trade_value == 4000
    assert result.rebalance_plan.cash_released == 4000
    assert action.simulated_order is not None
    assert action.simulated_order.side == "SELL"


def test_portfolio_monitor_watches_within_band_extended_position_without_order():
    portfolio_map = _monitor_test_map([("US.NVDA", 0.10, "core_ai", "core compute")])
    portfolio = CurrentPortfolio(
        total_assets=100_000,
        cash=10_000,
        holdings=[
            CurrentHolding(symbol="US.NVDA", quantity=110, market_value=11_000, can_sell_qty=110),
        ],
        data_asof="2026-06-17T00:00:00+00:00",
        source="test",
    )
    market = _monitor_market_data({"US.NVDA": 100}, extended=True)

    result = portfolio_monitor.monitor_portfolio(
        portfolio_map=portfolio_map,
        portfolio=portfolio,
        market_data=market,
    )

    action = next(item for item in result.rebalance_plan.actions if item.symbol == "US.NVDA")
    assert action.action == "watch"
    assert action.reason_code == "within_band_extended_watch"
    assert action.quantity == 0
    assert action.simulated_order is None
    assert result.rebalance_plan.simulated_orders == []


def test_exposure_ledger_groups_sndk_put_spread_before_intent_classification():
    futu_portfolio = {
        "funds": {"total_assets": 196_324.05, "cash": -11_192.54},
        "positions": [
            {
                "code": "US.SNDK260717P1550000",
                "name": "SNDK 260717 1550.00P",
                "qty": -1,
                "can_sell_qty": -1,
                "cost_price": 83.05,
                "market_val": -7109.48,
                "pl_val": 1195.52,
            },
            {
                "code": "US.SNDK260717P1450000",
                "name": "SNDK 260717 1450.00P",
                "qty": 1,
                "can_sell_qty": 1,
                "cost_price": 60.15,
                "market_val": 5078.38,
                "pl_val": -936.62,
            },
        ],
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(futu_portfolio)

    assert len(ledger.option_strategies) == 1
    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "bull_put_spread"
    assert strategy.underlying == "US.SNDK"
    assert strategy.expiry == "2026-07-17"
    assert strategy.defined_risk is True
    assert strategy.net_opening_credit == 2290
    assert strategy.spread_width == 10000
    assert strategy.max_profit == 2290
    assert strategy.max_loss == 7710
    assert strategy.short_assignment_notional == 155000
    assert strategy.intent_guess == "defined_risk_premium_income"
    assert "Do not treat the short leg as a naked assignment obligation." in strategy.risk_notes
    assert any(question.topic == "option_strategy_intent" for question in ledger.clarification_questions)


def test_option_market_data_extracts_held_option_codes_and_underlyings():
    futu_portfolio = {
        "positions": [
            {"code": "US.MSFT270115C350000", "qty": 1},
            {"code": "US.NVDA260717P210000", "qty": -1},
            {"code": "US.NVDA", "qty": 100},
        ]
    }

    option_codes, underlyings = option_market_data._option_codes_from_portfolio(futu_portfolio)

    assert option_codes == ["US.MSFT270115C350000", "US.NVDA260717P210000"]
    assert underlyings == ["US.MSFT", "US.NVDA"]


def test_exposure_ledger_normalizes_leveraged_etf_and_maps_to_underlying_sleeve():
    portfolio_map = _monitor_test_map([("US.SNDK", 0.03, "memory_storage", "storage")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {
                "code": "US.SNXX",
                "name": "2倍做多SNDK ETF-Tradr",
                "qty": 216,
                "cost_price": 19.286,
                "market_val": 8186.75,
            }
        ],
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
    )

    position = ledger.positions[0]
    assert position.instrument_type == "leveraged_etf"
    assert position.raw_code == "US.SNXX"
    assert position.underlying == "US.SNDK"
    assert position.theme_sleeve == "memory_storage"
    assert position.leverage_factor == 2
    assert position.effective_long_exposure == 16373.5
    assert position.needs_human_clarification is True
    question = next(item for item in ledger.clarification_questions if item.topic == "leveraged_etf_intent")
    assert question.symbols == ["US.SNXX", "US.SNDK"]


def test_exposure_ledger_marks_long_call_as_capital_efficiency_or_speculation_question():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 20_000},
        "positions": [
            {
                "code": "US.QQQ271217C400000",
                "name": "QQQ 271217 400.00C",
                "qty": 1,
                "cost_price": 221.81,
                "market_val": 36_254,
            }
        ],
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(futu_portfolio)

    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "long_call"
    assert strategy.intent_guess == "capital_efficient_long_exposure_or_speculation"
    assert strategy.defined_risk is True
    assert strategy.max_loss == 22181
    position = ledger.positions[0]
    assert position.instrument_type == "option_leg"
    assert position.effective_long_exposure == 0
    assert position.needs_human_clarification is True
    question = ledger.clarification_questions[0]
    assert question.topic == "option_strategy_intent"
    assert "资金效率工具" in question.question


def test_exposure_ledger_does_not_count_long_call_delta_without_confirmed_intent():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 20_000},
        "positions": [
            {
                "code": "US.MSFT270115C350000",
                "name": "MSFT 270115 350.00C",
                "qty": 1,
                "cost_price": 120,
                "market_val": 12_000,
            }
        ],
    }
    option_market_data = {
        "data": [
            {
                "code": "US.MSFT270115C350000",
                "option_delta": 0.6,
                "underlying_last_price": 500,
            }
        ]
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        option_market_data=option_market_data,
    )

    strategy = ledger.option_strategies[0]
    assert strategy.effective_long_exposure == 30_000
    assert strategy.needs_human_clarification is True
    position = ledger.positions[0]
    assert position.effective_long_exposure == 0
    assert position.portfolio_role == "option_overlay"


def test_exposure_ledger_counts_confirmed_long_call_delta_adjusted_exposure():
    portfolio_map = _monitor_test_map([("US.MSFT", 0.25, "cloud_platform", "cloud core")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 20_000},
        "positions": [
            {
                "code": "US.MSFT270115C350000",
                "name": "MSFT 270115 350.00C",
                "qty": 1,
                "cost_price": 120,
                "market_val": 12_000,
            }
        ],
    }
    option_market_data = {
        "data": [
                {
                    "code": "US.MSFT270115C350000",
                    "option_delta": 0.7,
                }
        ],
        "underlying_quotes": {"US.MSFT": {"last_price": 500}},
    }
    policy = exposure_ledger.ExposureIntentPolicy(long_call_intent="long_term_capital_efficiency")

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=policy,
        option_market_data=option_market_data,
    )
    report = exposure_drift.compute_exposure_drift(portfolio_map=portfolio_map, ledger=ledger)

    strategy = ledger.option_strategies[0]
    assert strategy.intent_guess == "long_term_capital_efficiency"
    assert strategy.effective_long_exposure == 35_000
    assert any("delta-adjusted exposure is mapped" in note for note in strategy.risk_notes)
    position = ledger.positions[0]
    assert position.portfolio_role == "delta_adjusted_option_exposure"
    assert position.effective_long_exposure == 35_000
    msft = next(item for item in report.positions if item.symbol == "US.MSFT")
    assert msft.status == "overweight"
    assert msft.current_effective_exposure == 35_000
    assert msft.current_effective_weight == 0.35
    assert msft.overlays[0].counted_in_target_exposure is True
    assert "delta-adjusted effective exposure" in msft.overlays[0].target_exposure_reason


def test_exposure_ledger_maps_direct_stock_to_target_map_sleeve_without_question():
    portfolio_map = _monitor_test_map([("US.NVDA", 0.13, "compute_accelerator", "core compute")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 20_000},
        "positions": [
            {
                "code": "US.NVDA",
                "name": "NVIDIA",
                "qty": 128,
                "cost_price": 193.296,
                "market_val": 26_624,
            }
        ],
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
    )

    position = ledger.positions[0]
    assert position.instrument_type == "stock"
    assert position.underlying == "US.NVDA"
    assert position.theme_sleeve == "compute_accelerator"
    assert position.effective_long_exposure == 26624
    assert position.needs_human_clarification is False
    assert ledger.clarification_questions == []


def test_exposure_ledger_applies_user_confirmed_intent_policy():
    portfolio_map = _monitor_test_map(
        [
            ("US.SNDK", 0.03, "memory_storage", "storage"),
            ("US.LITE", 0.02, "optical_interconnect_networking", "optical"),
            ("US.QQQ", 0.10, "core_beta", "core beta"),
        ]
    )
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": -5_000},
        "positions": [
            {
                "code": "US.SNDK260717P1550000",
                "name": "SNDK 260717 1550.00P",
                "qty": -1,
                "cost_price": 83.05,
                "market_val": -7109.48,
            },
            {
                "code": "US.SNDK260717P1450000",
                "name": "SNDK 260717 1450.00P",
                "qty": 1,
                "cost_price": 60.15,
                "market_val": 5078.38,
            },
            {
                "code": "US.QQQ271217C400000",
                "name": "QQQ 271217 400.00C",
                "qty": 1,
                "cost_price": 221.81,
                "market_val": 36_254,
            },
            {
                "code": "US.SNXX",
                "name": "2倍做多SNDK ETF-Tradr",
                "qty": 216,
                "market_val": 8_000,
            },
            {
                "code": "HK.07709",
                "name": "南方两倍做多海力士",
                "qty": 100,
                "market_val": 14_480,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(
        defined_risk_credit_spread_intent="independent_high_iv_premium",
        long_call_intent="long_term_capital_efficiency",
        leveraged_etf_intent="long_term_leveraged_exposure_limited_rebalance",
        negative_cash_policy="temporary_allowed_strict_risk",
        ignored_markets={"HK"},
    )

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=policy,
    )

    spread = next(item for item in ledger.option_strategies if item.strategy_type == "bull_put_spread")
    assert spread.intent_guess == "independent_high_iv_premium_income"
    assert spread.needs_human_clarification is False
    assert any("independent premium-income overlay" in note for note in spread.risk_notes)

    long_call = next(item for item in ledger.option_strategies if item.strategy_type == "long_call")
    assert long_call.intent_guess == "long_term_capital_efficiency"
    assert long_call.needs_human_clarification is False
    assert any("delta/Greeks are required" in note for note in long_call.risk_notes)

    leveraged = next(item for item in ledger.positions if item.raw_code == "US.SNXX")
    assert leveraged.portfolio_role == "strategic_leveraged_exposure_limited_rebalance"
    assert leveraged.intent_guess == "long_term_leveraged_exposure_limited_rebalance"
    assert leveraged.effective_long_exposure == 16000
    assert leveraged.needs_human_clarification is False

    hk_position = next(item for item in ledger.positions if item.raw_code == "HK.07709")
    assert hk_position.portfolio_role == "ignored_market"
    assert hk_position.theme_sleeve == "ignored"
    assert hk_position.effective_long_exposure == 0
    assert hk_position.needs_human_clarification is False

    assert any("Negative cash is allowed by user policy" in warning for warning in ledger.warnings)
    assert not any(question.topic == "margin_policy" for question in ledger.clarification_questions)
    assert not any(question.topic == "leveraged_etf_intent" for question in ledger.clarification_questions)
    assert not any(
        question.topic == "option_strategy_intent" and "US.SNDK" in question.symbols
        for question in ledger.clarification_questions
    )
    assert not any(
        question.topic == "option_strategy_intent" and "US.QQQ" in question.symbols
        for question in ledger.clarification_questions
    )


def test_exposure_ledger_cli_applies_intent_policy_path(tmp_path):
    portfolio_path = tmp_path / "portfolio.json"
    policy_path = tmp_path / "policy.json"
    output_dir = tmp_path / "out"
    portfolio_path.write_text(
        json.dumps(
            {
                "funds": {"total_assets": 100_000, "cash": -5_000},
                "positions": [
                    {
                        "code": "US.QQQ271217C400000",
                        "name": "QQQ 271217 400.00C",
                        "qty": 1,
                        "cost_price": 221.81,
                        "market_val": 36_254,
                    },
                    {
                        "code": "US.SNXX",
                        "name": "2倍做多SNDK ETF-Tradr",
                        "qty": 216,
                        "market_val": 8_000,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    policy_path.write_text(
        json.dumps(
            {
                "long_call_intent": "long_term_capital_efficiency",
                "leveraged_etf_intent": "long_term_leveraged_exposure_limited_rebalance",
                "negative_cash_policy": "temporary_allowed_strict_risk",
            }
        ),
        encoding="utf-8",
    )

    exit_code = exposure_ledger.main(
        [
            "build",
            "--futu-portfolio-path",
            str(portfolio_path),
            "--intent-policy-path",
            str(policy_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    ledger = json.loads((output_dir / "normalized_exposure_ledger.json").read_text(encoding="utf-8"))
    assert ledger["option_strategies"][0]["intent_guess"] == "long_term_capital_efficiency"
    leveraged = next(item for item in ledger["positions"] if item["raw_code"] == "US.SNXX")
    assert leveraged["portfolio_role"] == "strategic_leveraged_exposure_limited_rebalance"
    assert not ledger["clarification_questions"]


def test_exposure_ledger_classifies_short_call_as_covered_when_stock_is_available():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {"code": "US.NVDA", "qty": 128, "market_val": 26_000},
            {
                "code": "US.NVDA260717C240000",
                "name": "NVDA 260717 240.00C",
                "qty": -1,
                "cost_price": 2.49,
                "market_val": -109,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(short_call_intent="covered_call_income")

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        intent_policy=policy,
    )

    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "covered_call"
    assert strategy.intent_guess == "covered_call_income"
    assert strategy.defined_risk is True
    assert strategy.coverage_status == "covered"
    assert strategy.underlying_share_quantity == 128
    assert strategy.required_underlying_share_quantity == 100
    assert strategy.needs_human_clarification is False
    assert strategy.warnings == []
    assert ledger.clarification_questions == []


def test_exposure_ledger_uses_signed_quantity_for_short_call_delta():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {"code": "US.NVDA", "qty": 128, "market_val": 26_000},
            {
                "code": "US.NVDA260717C240000",
                "name": "NVDA 260717 240.00C",
                "qty": -1,
                "cost_price": 2.49,
                "market_val": -109,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(short_call_intent="covered_call_income")
    option_market_data = {
        "data": [
            {
                "code": "US.NVDA260717C240000",
                "option_delta": 0.4,
                "underlying_last_price": 200,
            }
        ]
    }

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        intent_policy=policy,
        option_market_data=option_market_data,
    )

    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "covered_call"
    assert strategy.effective_long_exposure == 0
    assert strategy.effective_short_exposure == 8_000
    option_position = next(item for item in ledger.positions if item.instrument_type == "option_leg")
    assert option_position.effective_long_exposure == 0
    assert option_position.effective_short_exposure == 0
    assert option_position.portfolio_role == "option_overlay"


def test_exposure_ledger_preserves_covered_call_intent_when_not_fully_covered():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {"code": "US.NVDA", "qty": 50, "market_val": 10_000},
            {
                "code": "US.NVDA260717C240000",
                "name": "NVDA 260717 240.00C",
                "qty": -1,
                "cost_price": 2.49,
                "market_val": -109,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(short_call_intent="covered_call_income")

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        intent_policy=policy,
    )

    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "covered_call"
    assert strategy.intent_guess == "covered_call_income"
    assert strategy.coverage_status == "partially_covered"
    assert strategy.underlying_share_quantity == 50
    assert strategy.required_underlying_share_quantity == 100
    assert strategy.defined_risk is False
    assert strategy.needs_human_clarification is False
    assert "not fully covered" in strategy.warnings[0]
    assert ledger.clarification_questions == []


def test_exposure_ledger_classifies_short_put_as_premium_income_with_assignment_ok():
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {
                "code": "US.NVDA260717P210000",
                "name": "NVDA 260717 210.00P",
                "qty": -1,
                "cost_price": 10.23,
                "market_val": -957.86,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(short_put_intent="premium_income_willing_assignment")

    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        intent_policy=policy,
    )

    strategy = ledger.option_strategies[0]
    assert strategy.strategy_type == "short_put"
    assert strategy.intent_guess == "premium_income_willing_assignment"
    assert strategy.short_assignment_notional == 21000
    assert strategy.needs_human_clarification is False
    assert strategy.warnings == []
    assert ledger.clarification_questions == []


def test_exposure_drift_counts_leveraged_etf_and_tracks_independent_put_spread_separately():
    portfolio_map = _monitor_test_map([("US.SNDK", 0.10, "memory_storage", "storage core")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": -5_000},
        "positions": [
            {
                "code": "US.SNDK260717P1550000",
                "name": "SNDK 260717 1550.00P",
                "qty": -1,
                "cost_price": 83.05,
                "market_val": -7109.48,
            },
            {
                "code": "US.SNDK260717P1450000",
                "name": "SNDK 260717 1450.00P",
                "qty": 1,
                "cost_price": 60.15,
                "market_val": 5078.38,
            },
            {
                "code": "US.SNXX",
                "name": "2倍做多SNDK ETF-Tradr",
                "qty": 216,
                "market_val": 8_000,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(
        defined_risk_credit_spread_intent="independent_high_iv_premium",
        leveraged_etf_intent="long_term_leveraged_exposure_limited_rebalance",
        negative_cash_policy="temporary_allowed_strict_risk",
    )
    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=policy,
    )

    report = exposure_drift.compute_exposure_drift(portfolio_map=portfolio_map, ledger=ledger)

    sndk = next(item for item in report.positions if item.symbol == "US.SNDK")
    assert sndk.status == "overweight"
    assert sndk.current_effective_exposure == 16_000
    assert sndk.current_effective_weight == 0.16
    assert sndk.drift_value == 6_000
    assert sndk.leveraged_effective_exposure == 16_000
    assert sndk.direct_market_value == 0
    assert sndk.option_max_loss == 7_710
    assert sndk.option_short_assignment_notional == 155_000
    assert sndk.overlays[0].strategy_type == "bull_put_spread"
    assert sndk.overlays[0].counted_in_target_exposure is False
    assert "Independent premium overlay" in sndk.overlays[0].target_exposure_reason
    assert any("Negative cash" in warning for warning in report.warnings)


def test_exposure_drift_counts_stock_and_preserves_covered_call_overlay():
    portfolio_map = _monitor_test_map([("US.NVDA", 0.20, "compute_accelerator", "core compute")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {"code": "US.NVDA", "qty": 128, "market_val": 26_000},
            {
                "code": "US.NVDA260717C240000",
                "name": "NVDA 260717 240.00C",
                "qty": -1,
                "cost_price": 2.49,
                "market_val": -109,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(short_call_intent="covered_call_income")
    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=policy,
    )

    report = exposure_drift.compute_exposure_drift(portfolio_map=portfolio_map, ledger=ledger)

    nvda = next(item for item in report.positions if item.symbol == "US.NVDA")
    assert nvda.status == "overweight"
    assert nvda.current_effective_exposure == 26_000
    assert nvda.direct_market_value == 26_000
    assert nvda.option_market_value == -109
    assert nvda.overlays[0].strategy_type == "covered_call"
    assert nvda.overlays[0].coverage_status == "covered"
    assert nvda.overlays[0].counted_in_target_exposure is False
    assert "underlying shares remain" in nvda.overlays[0].target_exposure_reason


def test_exposure_drift_cli_writes_report(tmp_path):
    portfolio_map = _monitor_test_map([("US.SNDK", 0.10, "memory_storage", "storage core")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {
                "code": "US.SNXX",
                "name": "2倍做多SNDK ETF-Tradr",
                "qty": 216,
                "market_val": 8_000,
            },
        ],
    }
    policy = exposure_ledger.ExposureIntentPolicy(
        leveraged_etf_intent="long_term_leveraged_exposure_limited_rebalance"
    )
    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
        intent_policy=policy,
    )
    map_path = tmp_path / "portfolio_map.json"
    ledger_path = tmp_path / "normalized_exposure_ledger.json"
    output_dir = tmp_path / "out"
    map_path.write_text(json.dumps(portfolio_map), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger.model_dump(mode="json")), encoding="utf-8")

    exit_code = exposure_drift.main(
        [
            "build",
            "--portfolio-map-path",
            str(map_path),
            "--ledger-path",
            str(ledger_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    report = json.loads((output_dir / "exposure_drift_report.json").read_text(encoding="utf-8"))
    sndk = next(item for item in report["positions"] if item["symbol"] == "US.SNDK")
    assert sndk["current_effective_exposure"] == 16_000


def test_exposure_drift_extra_position_uses_theme_sleeve_not_portfolio_role():
    portfolio_map = _monitor_test_map([("US.NVDA", 0.10, "compute_accelerator", "compute")])
    futu_portfolio = {
        "funds": {"total_assets": 100_000, "cash": 10_000},
        "positions": [
            {
                "code": "US.WDC",
                "name": "Western Digital",
                "qty": 10,
                "market_val": 2_000,
            }
        ],
    }
    ledger = exposure_ledger.build_exposure_ledger_from_futu_portfolio(
        futu_portfolio,
        portfolio_map=portfolio_map,
    )

    report = exposure_drift.compute_exposure_drift(portfolio_map=portfolio_map, ledger=ledger)

    wdc = next(item for item in report.positions if item.symbol == "US.WDC")
    assert wdc.status == "extra"
    assert wdc.sleeve_key == "memory_storage"
    assert wdc.contributions[0].portfolio_role == "target_or_extra_position"


def test_portfolio_weight_formula_caps_sleeve_by_candidate_capacity():
    context = {
        "artifact_type": "portfolio_weight_formula_context",
        "map_id": "initial_formula_map",
        "theme": "AI",
        "policy": {
            "sleeve_weight": 0.2,
            "cash_weight": 0.8,
            "single_name_limit": 0.1,
            "precision": 0.001,
        },
        "portfolio_structure_without_target_weights": {
            "sleeves": [
                {"sleeve_key": "single_name_layer", "holding_symbols": ["US.NVDA"]},
                {"sleeve_key": "other_layer", "holding_symbols": ["US.MU"]},
            ],
            "holdings": [
                {"symbol": "US.NVDA"},
                {"symbol": "US.MU"},
            ],
        },
        "formula_contract": {
            "sleeve_raw_score": "test",
            "candidate_raw_score": "test",
        },
    }
    scoring = portfolio_weight_formula.PortfolioWeightScoring(
        theme="AI",
        scoring_intent="capacity cap test",
        sleeve_scores=[
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="single_name_layer",
                importance_score=1,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="High-scoring one-name sleeve.",
                why_not_higher="Candidate capacity cap.",
                why_not_lower="High raw score.",
            ),
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="other_layer",
                importance_score=0.1,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Lower-scoring sleeve.",
                why_not_higher="Lower raw score.",
                why_not_lower="Receives overflow from capacity cap.",
            ),
        ],
        candidate_scores=[
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.NVDA",
                sleeve_key="single_name_layer",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="One-name candidate.",
                why_not_higher="Single-name cap.",
                why_not_lower="High raw score.",
                evidence_refs=["test:US.NVDA"],
            ),
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.MU",
                sleeve_key="other_layer",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Other candidate.",
                why_not_higher="Single-name cap.",
                why_not_lower="Overflow sleeve.",
                evidence_refs=["test:US.MU"],
            ),
        ],
    )

    report = portfolio_weight_formula.allocate_from_scoring(
        context=context,
        scoring=scoring,
        single_name_limit=0.1,
    )

    sleeve_weights = {item.sleeve_key: item.target_weight for item in report.sleeve_allocations}
    candidate_weights = {item.symbol: item.target_weight for item in report.candidate_allocations}
    assert sleeve_weights["single_name_layer"] == 0.1
    assert candidate_weights == {"US.NVDA": 0.1, "US.MU": 0.1}


def test_portfolio_weight_formula_style_exponent_concentrates_weights():
    base_context = {
        "artifact_type": "portfolio_weight_formula_context",
        "map_id": "style_test",
        "theme": "AI",
        "policy": {
            "sleeve_weight": 0.2,
            "cash_weight": 0.8,
            "single_name_limit": 0.2,
            "portfolio_style": "balanced",
            "precision": 0.001,
        },
        "portfolio_structure_without_target_weights": {
            "sleeves": [
                {"sleeve_key": "primary_layer", "holding_symbols": ["US.NVDA"]},
                {"sleeve_key": "secondary_layer", "holding_symbols": ["US.MU"]},
            ],
            "holdings": [
                {"symbol": "US.NVDA"},
                {"symbol": "US.MU"},
            ],
        },
        "formula_contract": {
            "sleeve_raw_score": "test",
            "candidate_raw_score": "test",
        },
    }
    scoring = portfolio_weight_formula.PortfolioWeightScoring(
        theme="AI",
        scoring_intent="style concentration test",
        sleeve_scores=[
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="primary_layer",
                importance_score=1,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Primary layer.",
                why_not_higher="Already highest.",
                why_not_lower="Best evidence.",
            ),
            portfolio_weight_formula.SleeveFormulaScore(
                sleeve_key="secondary_layer",
                importance_score=0.25,
                opportunity_score=1,
                evidence_strength=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Secondary layer.",
                why_not_higher="Lower priority.",
                why_not_lower="Still relevant.",
            ),
        ],
        candidate_scores=[
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.NVDA",
                sleeve_key="primary_layer",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Primary candidate.",
                why_not_higher="Already highest.",
                why_not_lower="Best evidence.",
                evidence_refs=["test:US.NVDA"],
            ),
            portfolio_weight_formula.CandidateFormulaScore(
                symbol="US.MU",
                sleeve_key="secondary_layer",
                role_importance=1,
                theme_fit=1,
                evidence_strength=1,
                business_quality=1,
                growth_quality=1,
                risk_penalty=0,
                overlap_penalty=0,
                rationale="Secondary candidate.",
                why_not_higher="Secondary sleeve.",
                why_not_lower="Still relevant.",
                evidence_refs=["test:US.MU"],
            ),
        ],
    )

    balanced = portfolio_weight_formula.allocate_from_scoring(
        context=base_context,
        scoring=scoring,
        single_name_limit=0.2,
    )
    concentrated_context = json.loads(json.dumps(base_context))
    concentrated_context["policy"]["portfolio_style"] = "concentrated_growth"
    concentrated = portfolio_weight_formula.allocate_from_scoring(
        context=concentrated_context,
        scoring=scoring,
        single_name_limit=0.2,
    )

    balanced_weights = {item.symbol: item.target_weight for item in balanced.candidate_allocations}
    concentrated_weights = {item.symbol: item.target_weight for item in concentrated.candidate_allocations}
    assert concentrated_weights["US.NVDA"] > balanced_weights["US.NVDA"]
    assert concentrated_weights["US.MU"] < balanced_weights["US.MU"]


def test_workflow_uses_deep_research_report_when_building_portfolio_maps(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    seen: dict[str, object] = {}

    def fake_build_portfolio_maps_from_triage(*, policy, triage, root=None, lightweight=None, **kwargs):
        seen["deep_research"] = kwargs.get("deep_research")
        return (
            portfolio_architect.PortfolioArchitectResult(
                theme=policy.theme,
                selection=portfolio_architect.PostEnrichmentSelection(
                    selected_for_portfolio=[
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.NVDA",
                            conviction="core",
                            layer_keys=["primary_domain"],
                            role="deep research portfolio core",
                            why_selected=["Selected from deep research."],
                            evidence_refs=["filing_summary:US.NVDA"],
                        )
                    ],
                    selection_summary="Selected the deep-research core candidate before weights.",
                    peer_tradeoffs=[
                        portfolio_architect.PeerTradeoff(
                            layer_key="primary_domain",
                            comparable_symbols=["US.NVDA"],
                            selected_symbols=["US.NVDA"],
                            rationale="Deep research has one construction candidate in this fixture.",
                        )
                    ],
                ),
                portfolio_maps=PortfolioMaps(
                    theme=policy.theme,
                    maps=[
                        PortfolioMap(
                            map_id="ai_deep_research_map",
                            name="AI deep research map",
                            objective=policy.objective,
                            sleeve_weight=policy.target_portfolio_weight,
                            positioning="Deep research integration test map.",
                            best_for="Testing deep-research workflow wiring.",
                            allocation_logic=["Use saved deep-research report."],
                            sleeves=[
                                PortfolioSleeve(
                                    name="Primary domain",
                                    role="Workflow test sleeve",
                                    target_weight=policy.target_portfolio_weight,
                                    holding_symbols=["US.NVDA"],
                                    rationale="Single test sleeve.",
                                )
                            ],
                            holdings=[
                                PortfolioHolding(
                                    symbol="US.NVDA",
                                    target_weight=policy.target_portfolio_weight,
                                    role="deep research portfolio core",
                                    rationale="Selected from deep-research report.",
                                    evidence_refs=["filing_summary:US.NVDA"],
                                )
                            ],
                            cash_weight=policy.cash_reserve,
                            thesis="Deep research workflow test thesis.",
                        )
                    ],
                ),
                map_weight_rationales=[
                    portfolio_architect.PortfolioMapWeightRationale(
                        map_id="ai_deep_research_map",
                        holding_count_rationale="The fixture has one researched symbol.",
                        sleeve_weight_rationale=["The single sleeve receives the full target weight."],
                    )
                ],
            ),
            portfolio_architect.PortfolioArchitectRunArtifact(
                root=str(tmp_path / "data" / "investment_assistant"),
                context_path=str(tmp_path / "context.json"),
                portfolio_maps_path=str(tmp_path / "portfolio_maps.json"),
                eligible_symbols=["US.NVDA"],
                researched_symbols=["US.NVDA"],
                pydantic_ai={"mock": True},
            ),
        )

    monkeypatch.setattr(ia_workflow, "build_portfolio_maps_from_triage", fake_build_portfolio_maps_from_triage)

    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    started = workflow.run("cli:test", "start", payload={"theme": "AI"})
    triaged = workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=started["session_id"],
        payload={"answer": "选 1"},
    )
    deep_artifact = store.add_artifact(
        started["session_id"],
        "deep_research_report",
        deep_research.DeepResearchReport(
            theme="ai",
            candidate_cards=[
                deep_research.CandidateResearchCard(
                    symbol="US.NVDA",
                    layer_keys=["primary_domain"],
                    exposure_summary="Direct compute platform.",
                    filing_takeaways=["Datacenter demand is central."],
                    candidate_decision="core_candidate",
                    evidence_refs=["filing_summary:US.NVDA"],
                )
            ],
        ),
    )
    result = workflow.run(
        "cli:test",
        "build_portfolio_maps",
        session_id=started["session_id"],
        payload={},
    )

    assert triaged["state"] == WorkflowState.CANDIDATE_TRIAGE_COMPLETE.value
    assert result["success"] is True
    assert result["data"]["deep_research_report_artifact_id"] == deep_artifact["artifact_id"]
    assert seen["deep_research"]["candidate_cards"][0]["symbol"] == "US.NVDA"


def test_workflow_retries_failed_portfolio_architect_with_continue(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)
    calls = {"count": 0}

    def fake_build_portfolio_maps_from_triage(*, policy, triage, root=None, lightweight=None, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("PydanticAI peer_tradeoffs must explain selected vs non-selected high-priority candidates for layers: L5_cloud_platforms")
        return (
            portfolio_architect.PortfolioArchitectResult(
                theme=policy.theme,
                selection=portfolio_architect.PostEnrichmentSelection(
                    selected_for_portfolio=[
                        portfolio_architect.SelectedPortfolioCandidate(
                            symbol="US.NVDA",
                            conviction="core",
                            layer_keys=["primary_domain"],
                            role="workflow test portfolio core",
                            why_selected=["Selected from candidate triage."],
                            evidence_refs=["candidate_triage:US.NVDA"],
                        )
                    ],
                    selection_summary="Retry selected the saved deep-research candidate.",
                    peer_tradeoffs=[
                        portfolio_architect.PeerTradeoff(
                            layer_key="primary_domain",
                            comparable_symbols=["US.NVDA"],
                            selected_symbols=["US.NVDA"],
                            non_selected_symbols=[],
                            rationale="Only eligible candidate in retry test.",
                        )
                    ],
                ),
                portfolio_maps=PortfolioMaps(
                    theme=policy.theme,
                    maps=[
                        PortfolioMap(
                            map_id="ai_retry_map",
                            name="AI retry map",
                            objective=policy.objective,
                            sleeve_weight=policy.target_portfolio_weight,
                            positioning="Retry integration test map.",
                            best_for="Testing recoverable architect failure.",
                            allocation_logic=["Reuse saved candidate triage."],
                            sleeves=[
                                PortfolioSleeve(
                                    name="Primary domain",
                                    role="Workflow test sleeve",
                                    target_weight=policy.target_portfolio_weight,
                                    holding_symbols=["US.NVDA"],
                                    rationale="Single retry sleeve.",
                                )
                            ],
                            holdings=[
                                PortfolioHolding(
                                    symbol="US.NVDA",
                                    target_weight=policy.target_portfolio_weight,
                                    role="workflow test portfolio core",
                                    rationale="Selected from saved candidate triage artifact.",
                                    evidence_refs=["candidate_triage:US.NVDA"],
                                )
                            ],
                            cash_weight=policy.cash_reserve,
                            thesis="Retry test thesis.",
                        )
                    ],
                ),
                map_weight_rationales=[
                    portfolio_architect.PortfolioMapWeightRationale(
                        map_id="ai_retry_map",
                        holding_count_rationale="The retry fixture has one eligible symbol.",
                        sleeve_weight_rationale=["The single sleeve receives the target weight."],
                    )
                ],
            ),
            portfolio_architect.PortfolioArchitectRunArtifact(
                root=str(tmp_path / "data" / "investment_assistant"),
                context_path=str(tmp_path / "context.json"),
                portfolio_maps_path=str(tmp_path / "portfolio_maps.json"),
                eligible_symbols=["US.NVDA"],
                pydantic_ai={"mock": True, "retry": True},
            ),
        )

    monkeypatch.setattr(ia_workflow, "build_portfolio_maps_from_triage", fake_build_portfolio_maps_from_triage)

    store = InvestmentAssistantStore()
    workflow = _workflow(store)
    started = workflow.run("cli:test", "start", payload={"theme": "AI"})
    workflow.run(
        "cli:test",
        "answer_human_input",
        session_id=started["session_id"],
        payload={"answer": "选 1"},
    )
    failed = workflow.run(
        "cli:test",
        "build_portfolio_maps",
        session_id=started["session_id"],
        payload={},
    )
    retried = workflow.run(
        "cli:test",
        "continue",
        session_id=started["session_id"],
        payload={},
    )

    assert failed["success"] is True
    assert failed["status"] == "failed"
    assert failed["state"] == WorkflowState.DRAFTING_TARGET_PORTFOLIO_MAPS.value
    assert failed["allowed_actions"] == ["build_portfolio_maps", "continue", "status", "cancel"]
    assert "继续生成组合版图" in failed["display_response"]
    assert retried["success"] is True
    assert retried["status"] == "waiting_for_human"
    assert retried["state"] == WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value
    assert retried["data"]["map_ids"] == ["ai_retry_map"]
    assert calls["count"] == 2


def test_sec_provider_unavailable_is_explicit_and_non_blocking():
    universe = FakeMarketDataAdapter().get_theme_universe("storage")
    pool = CandidatePool(
        theme=universe.canonical_theme,
        generated_from=universe.source_tags,
        candidates=universe.candidates,
        warnings=universe.warnings,
    )
    policy = InvestmentPolicy(theme="storage")
    artifacts = build_market_artifacts(
        policy,
        pool,
        sec_context={
            "source": "edgartools",
            "source_status": "not_configured",
            "items": {},
            "warnings": ["SEC_EDGAR_IDENTITY or EDGAR_IDENTITY is required."],
        },
    )

    assert artifacts["sec_filings_context"]["source_status"] == "not_configured"
    assert artifacts["fundamental_quality"]["sec_source_status"] == "not_configured"
    reflection = reflect_candidate_pool(policy, pool, artifacts)
    assert "sec_filings_context" in reflection.stale_data


def test_sec_stale_periodic_filing_is_reflected():
    universe = FakeMarketDataAdapter().get_theme_universe("storage")
    pool = CandidatePool(
        theme=universe.canonical_theme,
        generated_from=universe.source_tags,
        candidates=universe.candidates,
        warnings=universe.warnings,
    )
    policy = InvestmentPolicy(theme="storage")
    first = universe.candidates[0]
    artifacts = build_market_artifacts(
        policy,
        pool,
        sec_context={
            "source": "edgartools",
            "source_status": "available",
            "items": {
                first.symbol: {
                    "symbol": first.symbol,
                    "source_status": "available",
                    "filings": {},
                    "fundamentals": {},
                    "event_context": {
                        "latest_periodic_filing_date": "2025-01-01",
                        "periodic_filing_age_days": 503,
                        "periodic_filing_stale": True,
                        "event_risk_level": "unknown",
                    },
                    "risk_flags": ["sec_periodic_filing_stale"],
                }
            },
            "warnings": [],
        },
    )

    assert artifacts["sec_filings_context"]["items"][0]["event_context"]["periodic_filing_stale"] is True
    assert "sec_periodic_filing_stale" in artifacts["risk_flags"]["items"][0]["risk_tags"]
    reflection = reflect_candidate_pool(policy, pool, artifacts)
    assert "sec_periodic_filings" in reflection.stale_data


def test_tool_handler_uses_latest_session_when_session_id_is_omitted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "test-session")

    def fake_discovery(theme, **kwargs):
        return _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.VST")

    _mock_workflow_triage_dependencies(monkeypatch, fake_discovery)

    raw = handle_ia_portfolio_workflow(
        {"action": "start", "payload": {"theme": "power"}},
    )
    started = json.loads(raw)
    raw_status = handle_ia_portfolio_workflow({"action": "status"})
    status = json.loads(raw_status)

    assert started["success"] is True
    assert status["success"] is True
    assert status["session_id"] == started["session_id"]


def test_cli_tenant_is_stable_and_can_resume_legacy_cli_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "new-cli-process")
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_PLATFORM", raising=False)

    assert current_hermes_tenant() == "cli:local"

    store = InvestmentAssistantStore()
    legacy = store.create_session(
        tenant_id="cli:old-cli-process",
        theme="ai",
        state=WorkflowState.NEEDS_PORTFOLIO_MAP_REVIEW.value,
        status="completed",
    )
    workflow = _workflow(store)

    explicit = workflow.run("cli:local", "status", session_id=legacy["session_id"])
    implicit = workflow.run("cli:local", "status")

    assert explicit["success"] is True
    assert explicit["session_id"] == legacy["session_id"]
    assert implicit["success"] is True
    assert implicit["session_id"] == legacy["session_id"]


def test_tool_handler_allows_safe_rephrase_and_falls_back_on_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "artifact-session")
    monkeypatch.setattr(
        ia_workflow,
        "build_ai_discovery_v1_plan",
        lambda theme, **kwargs: _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA"),
    )
    monkeypatch.setattr(ia_workflow, "build_lightweight_enrichment_artifact", _fake_lightweight_artifact)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_plan", _fake_triage_plan)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_artifact", _fake_candidate_triage_artifact)
    monkeypatch.setattr(
        output_guard,
        "_llm_judge_agent_response",
        lambda text, guard: (
            "US.AAPL" not in text and "40%" not in text,
            [] if "US.AAPL" not in text and "40%" not in text else ["invented_fact"],
            "llm",
        ),
    )

    raw = handle_ia_portfolio_workflow(
        {
            "action": "start",
            "payload": {"theme_key": "ai", "required_symbols": ["QQQ", "NVDA"]},
        },
    )
    result = json.loads(raw)

    assert result["success"] is True
    assert result["answer_contract"]["mode"] == "artifact_only"
    assert result["display_response"]
    assert transform_llm_output(
        response_text=(
            "我会先按 AI 主题准备版图。默认目标仓位是 15.0%，现金底线是 10.0%。"
            "你可以直接确认，也可以告诉我改得更激进。"
        ),
        session_id="artifact-session",
    ) is None

    raw = handle_ia_portfolio_workflow(
        {
            "action": "start",
            "payload": {"theme_key": "ai", "required_symbols": ["QQQ", "NVDA"]},
        },
    )
    result = json.loads(raw)

    assert transform_llm_output(
        response_text="我建议直接买入 US.AAPL，并把目标仓位提高到 40%。",
        session_id="artifact-session",
    ) == result["fallback_response"]


def test_output_guard_falls_back_when_llm_judge_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "judge-down-session")
    monkeypatch.setattr(
        ia_workflow,
        "build_ai_discovery_v1_plan",
        lambda theme, **kwargs: _fake_theme_discovery_plan(theme, **kwargs, seed_symbol="US.NVDA"),
    )
    monkeypatch.setattr(ia_workflow, "build_lightweight_enrichment_artifact", _fake_lightweight_artifact)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_plan", _fake_triage_plan)
    monkeypatch.setattr(ia_workflow, "build_candidate_triage_artifact", _fake_candidate_triage_artifact)

    def raise_judge_error(text, guard):
        raise RuntimeError("judge down")

    monkeypatch.setattr(output_guard, "_llm_judge_agent_response", raise_judge_error)

    raw = handle_ia_portfolio_workflow(
        {"action": "start", "payload": {"theme_key": "ai"}},
    )
    result = json.loads(raw)

    assert transform_llm_output(
        response_text="我会用自然语言解释这份 workflow。",
        session_id="judge-down-session",
    ) == result["fallback_response"]


def test_fundamental_data_catalog_exposes_real_sec_layers():
    catalog = fundamental_tools.read_data_layer_catalog()

    layers = {item["layer"]: item for item in catalog["layers"]}
    assert layers["sec_companyfacts"]["status"] == "available"
    assert layers["sec_companyfacts"]["source"] == "sec.gov via edgartools"
    assert layers["sec_companyfacts"]["numeric_llm_generated"] is False
    assert layers["filing_narrative_summary"]["status"] == "planned"


def test_fundamental_freshness_missing_context_requests_refresh(tmp_path):
    store = InvestmentAssistantStore(tmp_path / "ia.sqlite")
    session = store.create_session(
        tenant_id="tenant",
        theme="ai",
        state=WorkflowState.THEME_DISCOVERY_COMPLETE.value,
        status="completed",
    )
    store.add_artifact(session["session_id"], "initial_request", {"market": "US"})
    store.add_artifact(
        session["session_id"],
        "theme_discovery",
        {
            "seed_symbols": [{"symbol": "NVDA", "role": "compute leader"}],
            "coverage_requirements": [
                {
                    "candidate_symbols": ["MU"],
                    "must_consider_symbols": ["SNDK"],
                }
            ],
        },
    )

    report = fundamental_tools.inspect_fundamental_freshness(
        session["session_id"],
        store=store,
    )

    assert report["symbols"] == ["US.NVDA", "US.MU", "US.SNDK"]
    assert report["should_refresh"] is True
    assert report["per_symbol"]["US.NVDA"]["freshness"] == "missing"
    assert report["per_symbol"]["US.NVDA"]["refresh_reasons"] == ["missing_context"]


def test_build_fundamental_context_uses_real_provider_boundary_and_persists(tmp_path):
    class FakeSecProvider:
        def __init__(self):
            self.seen_symbols = []

        def get_sec_context(self, candidates):
            self.seen_symbols = [candidate.symbol for candidate in candidates]
            items = {
                symbol: {
                    "symbol": symbol,
                    "ticker": symbol.split(".", 1)[1],
                    "source_status": "available",
                    "filings": {"latest_10q": {"filing_date": "2026-05-01"}},
                    "fundamentals": {
                        "ttm_revenue": 100.0,
                        "ttm_net_income": 20.0,
                        "total_assets": 300.0,
                        "total_liabilities": 90.0,
                    },
                    "numeric_evidence": {
                        "source": "sec_companyfacts",
                        "provider": "edgartools",
                        "llm_generated": False,
                    },
                    "event_context": {
                        "latest_periodic_filing_date": "2026-05-01",
                        "periodic_filing_age_days": 10,
                        "periodic_filing_stale": False,
                        "event_risk_level": "low",
                    },
                    "risk_flags": [],
                }
                for symbol in self.seen_symbols
            }
            return {
                "source": "edgartools",
                "source_status": "available",
                "generated_at": fundamental_tools.utc_now(),
                "requested_symbols": self.seen_symbols,
                "fetched_symbols": self.seen_symbols,
                "items": items,
                "warnings": [],
            }

    store = InvestmentAssistantStore(tmp_path / "ia.sqlite")
    session = store.create_session(
        tenant_id="tenant",
        theme="ai",
        state=WorkflowState.THEME_DISCOVERY_COMPLETE.value,
        status="completed",
    )
    store.add_artifact(session["session_id"], "initial_request", {"market": "US"})
    store.add_artifact(session["session_id"], "policy", {"required_symbols": ["QQQ"]})
    store.add_artifact(
        session["session_id"],
        "theme_discovery",
        {
            "seed_symbols": [
                {"symbol": "NVDA", "role": "compute leader"},
                {"symbol": "MU", "role": "memory"},
            ],
        },
    )
    provider = FakeSecProvider()

    result = fundamental_tools.build_fundamental_context(
        session["session_id"],
        store=store,
        sec_provider=provider,
        trigger="test",
        reason="unit test",
    )
    context = fundamental_tools.read_fundamental_context(
        session["session_id"],
        symbols=["NVDA"],
        store=store,
    )

    assert provider.seen_symbols == ["US.QQQ", "US.NVDA", "US.MU"]
    assert result["source_status"] == "available"
    assert result["freshness"]["per_symbol"]["US.NVDA"]["freshness"] == "fresh"
    assert context["context_artifact_id"] == result["fundamental_context_artifact_id"]
    assert list(context["items"]) == ["US.NVDA"]
    assert context["items"]["US.NVDA"]["numeric_evidence"]["llm_generated"] is False


def test_sec_provider_preserves_edgartools_metric_provenance():
    from dataclasses import dataclass
    from datetime import date

    @dataclass
    class FakeFact:
        concept: str
        label: str
        value: float
        numeric_value: float
        unit: str
        period_start: date
        period_end: date
        period_type: str
        fiscal_year: int
        fiscal_period: str
        filing_date: date
        form_type: str
        accession: str
        data_quality: str
        is_audited: bool = False
        is_restated: bool = False
        is_estimated: bool = False
        confidence_score: float = 0.9
        calculation_context: str | None = None
        statement_type: str = "IncomeStatement"
        section: str = "Revenue"

    class FakeTTM:
        concept = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        label = "Revenue"
        value = 300.0
        unit = "USD"
        as_of_date = date(2026, 5, 2)
        has_gaps = False
        has_calculated_q4 = True
        warning = "derived q4"

        def __init__(self):
            self.period_facts = [
                FakeFact(
                    concept=self.concept,
                    label="Revenue",
                    value=100.0,
                    numeric_value=100.0,
                    unit="USD",
                    period_start=date(2026, 2, 1),
                    period_end=date(2026, 5, 2),
                    period_type="duration",
                    fiscal_year=2027,
                    fiscal_period="Q1",
                    filing_date=date(2026, 5, 28),
                    form_type="10-Q",
                    accession="000-test-q",
                    data_quality="high",
                ),
                FakeFact(
                    concept=self.concept,
                    label="Revenue",
                    value=200.0,
                    numeric_value=200.0,
                    unit="USD",
                    period_start=date(2025, 11, 2),
                    period_end=date(2026, 1, 31),
                    period_type="duration",
                    fiscal_year=2026,
                    fiscal_period="Q4",
                    filing_date=date(2026, 3, 11),
                    form_type="10-K",
                    accession="000-test-k",
                    data_quality="medium",
                    calculation_context="derived_q4_fy_minus_ytd9",
                ),
            ]

    class FakeFacts:
        def get_ttm_revenue(self):
            return FakeTTM()

        def get_ttm_net_income(self):
            return None

        def get_annual_fact(self, concept):
            if concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax":
                return FakeFact(
                    concept=concept,
                    label="Revenue",
                    value=250.0,
                    numeric_value=250.0,
                    unit="USD",
                    period_start=date(2025, 2, 2),
                    period_end=date(2026, 1, 31),
                    period_type="duration",
                    fiscal_year=2026,
                    fiscal_period="FY",
                    filing_date=date(2026, 3, 11),
                    form_type="10-K",
                    accession="000-test-k",
                    data_quality="high",
                )
            return None

        def get_net_income(self):
            return None

        def get_gross_profit(self):
            return None

        def get_operating_income(self):
            return None

        def get_total_assets(self):
            return None

        def get_total_liabilities(self):
            return None

        def get_shareholders_equity(self):
            return None

    fundamentals, provenance = sec_provider._fundamentals_with_provenance(FakeFacts())

    assert fundamentals["ttm_revenue"] == 300.0
    assert fundamentals["annual_revenue"] == 250.0
    assert provenance["ttm_revenue"]["component_count"] == 2
    assert provenance["ttm_revenue"]["has_calculated_q4"] is True
    assert provenance["ttm_revenue"]["data_quality"] == "mixed"
    assert provenance["ttm_revenue"]["components"][1]["calculation_context"] == "derived_q4_fy_minus_ytd9"
    assert provenance["annual_revenue"]["accession"] == "000-test-k"
    assert provenance["annual_revenue"]["form_type"] == "10-K"


def test_data_miner_writes_sec_files_with_raw_filing_text(tmp_path, monkeypatch):
    class FakeSecProvider:
        def __init__(self):
            self.seen_symbols = []

        def get_sec_context(self, candidates):
            self.seen_symbols = [candidate.symbol for candidate in candidates]
            return {
                "source": "edgartools",
                "source_status": "available",
                "generated_at": data_miner.utc_now(),
                "requested_symbols": self.seen_symbols,
                "fetched_symbols": self.seen_symbols,
                "items": {
                    "US.MRVL": {
                        "symbol": "US.MRVL",
                        "ticker": "MRVL",
                        "source_status": "available",
                        "cik": "1835632",
                        "company_name": "Marvell Technology, Inc.",
                        "industry": "Semiconductors",
                        "filings": {
                            "latest_10q": {
                                "form": "10-Q",
                                "filing_date": "2026-06-01",
                                "period_of_report": "2026-05-02",
                                "accession_number": "0000000000-26-000001",
                                "url": "https://www.sec.gov/Archives/test/mrvl-10q.htm",
                            }
                        },
                        "fundamentals": {
                            "ttm_revenue": 100.0,
                            "ttm_net_income": 20.0,
                        },
                        "numeric_evidence": {
                            "source": "sec_companyfacts",
                            "provider": "edgartools",
                            "llm_generated": False,
                            "provenance_included": True,
                        },
                        "metric_provenance": {
                            "ttm_revenue": {
                                "source": "edgartools",
                                "value": 100.0,
                                "components": [],
                            }
                        },
                        "event_context": {
                            "latest_periodic_filing_date": "2026-06-01",
                            "periodic_filing_stale": False,
                        },
                        "risk_flags": [],
                    }
                },
                "warnings": [],
            }

    monkeypatch.setattr(
        data_miner,
        "_fetch_url_bytes",
        lambda url, user_agent: (b"<html>mrvl filing</html>", False, []),
    )

    class FakeSection:
        def __init__(self, title, part, item):
            self.title = title
            self.part = part
            self.item = item
            self.confidence = 0.95
            self.method = "fake"

    class FakeChunkedDocument:
        def get_item_with_part(self, part, item, markdown=False):
            assert markdown is True
            return f"{part} {item} MRVL section body"

    class FakeFilingObj:
        chunked_document = FakeChunkedDocument()
        sections = {
            "part_i_item_1a": FakeSection("Risk Factors", "PART I", "ITEM 1A"),
            "part_ii_item_7": FakeSection("MD&A", "PART II", "ITEM 7"),
        }

    class FakeFiling:
        def __init__(self, form):
            self.form = form
            self.accession_number = {
                "10-K": "0000000000-26-000001",
                "10-Q": "0000000000-26-000002",
                "8-K": "0000000000-26-000003",
            }[form]
            self.filing_date = "2026-06-01"
            self.period_of_report = "2026-05-02"
            self.primary_document = f"mrvl-{form.lower()}.htm"

        def obj(self):
            return FakeFilingObj()

    class FakeCompany:
        def __init__(self, ticker):
            self.ticker = ticker

        def get_filings(self, form):
            return [FakeFiling(form)]

    class FakeEdgar:
        def set_identity(self, identity):
            self.identity = identity

        def Company(self, ticker):
            return FakeCompany(ticker)

    monkeypatch.setattr(data_miner, "_import_edgar_for_sections", lambda: FakeEdgar())
    provider = FakeSecProvider()

    run = data_miner.build_data_files_from_triage(
        symbols=["MRVL"],
        output_root=tmp_path / "ia_data",
        layers=["sec", "filing_metadata", "filing_text", "filing_sections"],
        sec_provider=provider,
        force=True,
    )

    symbol_dir = tmp_path / "ia_data" / "symbols" / "US.MRVL"
    assert provider.seen_symbols == ["US.MRVL"]
    assert run.symbols == ["US.MRVL"]
    assert run.status_counts == {"fresh": 1}
    assert (symbol_dir / "manifest.json").exists()
    assert (symbol_dir / "sec_companyfacts.json").exists()
    assert (symbol_dir / "filing_metadata.json").exists()
    assert (symbol_dir / "filing_text.json").exists()
    assert (symbol_dir / "filing_sections.json").exists()
    assert (symbol_dir / "raw_filings" / "latest_10q.html").read_bytes() == b"<html>mrvl filing</html>"

    companyfacts = json.loads((symbol_dir / "sec_companyfacts.json").read_text(encoding="utf-8"))
    filing_text = json.loads((symbol_dir / "filing_text.json").read_text(encoding="utf-8"))
    filing_sections = json.loads((symbol_dir / "filing_sections.json").read_text(encoding="utf-8"))
    manifest = json.loads((symbol_dir / "manifest.json").read_text(encoding="utf-8"))

    assert companyfacts["fundamentals"]["ttm_revenue"] == 100.0
    assert companyfacts["numeric_evidence"]["llm_generated"] is False
    assert companyfacts["numeric_evidence"]["provenance_included"] is True
    assert companyfacts["metric_provenance"]["ttm_revenue"]["source"] == "edgartools"
    assert filing_text["filings"]["latest_10q"]["source_status"] == "fresh"
    assert filing_sections["filings"]["latest_10q"]["sections"]["part_i_item_2"]["source_status"] == "fresh"
    section_path = symbol_dir / "filing_sections" / "latest_10q" / "part_i_item_2.md"
    assert "MRVL section body" in section_path.read_text(encoding="utf-8")
    assert manifest["layers"]["filing_sections"]["status"] == "fresh"
    assert manifest["layers"]["filing_text"]["status"] == "fresh"


def test_data_miner_marks_etf_layers_without_sec_provider(tmp_path):
    class FailingSecProvider:
        def get_sec_context(self, candidates):
            raise AssertionError("ETF symbols should not call the operating-company SEC provider")

    run = data_miner.build_data_files_from_triage(
        symbols=["US.QQQ"],
        output_root=tmp_path / "ia_data",
        layers=["sec", "filing_metadata", "filing_text", "etf"],
        sec_provider=FailingSecProvider(),
        force=True,
    )

    symbol_dir = tmp_path / "ia_data" / "symbols" / "US.QQQ"
    manifest = json.loads((symbol_dir / "manifest.json").read_text(encoding="utf-8"))
    assert run.symbols == ["US.QQQ"]
    assert run.status_counts == {"partial": 1}
    assert manifest["layers"]["etf_holdings"]["status"] == "not_implemented"
    assert manifest["layers"]["sec_companyfacts"]["status"] == "skipped"
    assert manifest["layers"]["filing_metadata"]["status"] == "skipped"
    assert manifest["layers"]["filing_text"]["status"] == "skipped"


def test_fmp_provider_missing_key_creates_skipped_artifact():
    calls = []

    def fake_request(path, params):
        calls.append((path, params))
        return []

    provider = fmp_provider.FmpProvider(
        fmp_provider.FmpProviderConfig(
            api_key=None,
            base_url="https://example.test/stable",
            timeout_seconds=1,
            retries=1,
            retry_backoff_seconds=0,
            rate_limit_delay_seconds=0,
            enabled=True,
            limit=5,
        ),
        request_json=fake_request,
    )

    artifact = provider.etf_exposure("US.QQQ")

    assert calls == []
    assert artifact["artifact_type"] == "fmp_etf_exposure"
    assert artifact["provider"] == "financialmodelingprep"
    assert artifact["source_status"] == "skipped"
    assert all(endpoint["source_status"] == "skipped" for endpoint in artifact["endpoints"])
    assert "FMP_API_KEY" in artifact["warnings"][0]


def test_fmp_provider_etf_exposure_populates_holdings_and_metadata():
    def fake_request(path, params):
        assert params["symbol"] == "QQQ"
        if path == "etf/info":
            return [{"symbol": "QQQ", "name": "Invesco QQQ Trust"}]
        if path == "etf/holdings":
            return [
                {"symbol": "US.NVDA", "name": "NVIDIA", "weightPercentage": 8.5},
                {"symbol": "US.MSFT", "name": "Microsoft", "weightPercentage": 6.1},
            ]
        if path == "etf/sector-weightings":
            return [{"sector": "Technology", "weightPercentage": 50.0}]
        if path == "etf/country-weightings":
            return [{"country": "United States", "weightPercentage": 95.0}]
        raise AssertionError(path)

    provider = fmp_provider.FmpProvider(
        fmp_provider.FmpProviderConfig(
            api_key="secret-key",
            base_url="https://example.test/stable",
            timeout_seconds=1,
            retries=1,
            retry_backoff_seconds=0,
            rate_limit_delay_seconds=0,
            enabled=True,
            limit=5,
        ),
        request_json=fake_request,
    )

    artifact = provider.etf_exposure("QQQ")

    assert artifact["symbol"] == "US.QQQ"
    assert artifact["source_status"] == "fresh"
    assert artifact["profile"]["name"] == "Invesco QQQ Trust"
    assert artifact["concentration_summary"]["top_10_weight"] == 14.6
    assert artifact["overlap_helper"]["holding_symbols"] == ["US.NVDA", "US.MSFT"]
    assert artifact["endpoints"][0]["url"].endswith("apikey=%2A%2A%2A")
    assert artifact["raw"]["holdings"][0]["symbol"] == "US.NVDA"


def test_fmp_provider_partial_failure_preserves_endpoint_warning():
    def fake_request(path, params):
        if path == "price-target-summary":
            raise RuntimeError("temporary quota failure")
        return [{"symbol": params["symbol"], "path": path}]

    provider = fmp_provider.FmpProvider(
        fmp_provider.FmpProviderConfig(
            api_key="secret-key",
            base_url="https://example.test/stable",
            timeout_seconds=1,
            retries=1,
            retry_backoff_seconds=0,
            rate_limit_delay_seconds=0,
            enabled=True,
            limit=5,
        ),
        request_json=fake_request,
    )

    artifact = provider.analyst_expectations("US.NVDA")

    assert artifact["source_status"] == "partial"
    failed = [endpoint for endpoint in artifact["endpoints"] if endpoint["name"] == "price_target_summary"]
    assert failed[0]["source_status"] == "unavailable"
    assert "temporary quota failure" in artifact["warnings"]
    assert artifact["annual_estimates"][0]["path"] == "analyst-estimates"


def test_data_miner_writes_optional_fmp_layers(tmp_path):
    class FailingSecProvider:
        def get_sec_context(self, candidates):
            raise AssertionError("FMP-only run should not call SEC provider")

    class FakeFmpProvider:
        def __init__(self):
            self.calls = []

        def etf_exposure(self, symbol):
            self.calls.append(("etf_exposure", symbol))
            return {
                "artifact_type": "fmp_etf_exposure",
                "symbol": symbol,
                "provider": "financialmodelingprep",
                "generated_at": data_miner.utc_now(),
                "source_status": "fresh",
                "endpoints": [],
                "data_asof": {},
                "warnings": [],
                "holdings": [{"symbol": "US.NVDA", "weightPercentage": 8.5}],
                "raw": {},
            }

        def analyst_expectations(self, symbol):
            self.calls.append(("analyst_expectations", symbol))
            return {
                "artifact_type": "fmp_analyst_expectations",
                "symbol": symbol,
                "provider": "financialmodelingprep",
                "generated_at": data_miner.utc_now(),
                "source_status": "partial",
                "endpoints": [],
                "data_asof": {},
                "warnings": ["ratings endpoint unavailable"],
                "annual_estimates": [],
                "raw": {},
            }

    fmp = FakeFmpProvider()

    run = data_miner.build_data_files_from_triage(
        symbols=["US.QQQ", "US.NVDA"],
        output_root=tmp_path / "ia_data",
        layers=["fmp_etf", "fmp_analyst"],
        sec_provider=FailingSecProvider(),
        fmp_provider=fmp,
        force=True,
    )

    qqq_dir = tmp_path / "ia_data" / "symbols" / "US.QQQ"
    nvda_dir = tmp_path / "ia_data" / "symbols" / "US.NVDA"
    qqq_manifest = json.loads((qqq_dir / "manifest.json").read_text(encoding="utf-8"))
    nvda_manifest = json.loads((nvda_dir / "manifest.json").read_text(encoding="utf-8"))
    qqq_etf = json.loads((qqq_dir / "fmp_etf_exposure.json").read_text(encoding="utf-8"))
    nvda_analyst = json.loads((nvda_dir / "fmp_analyst_expectations.json").read_text(encoding="utf-8"))

    assert run.status_counts == {"partial": 2}
    assert fmp.calls == [("etf_exposure", "US.QQQ"), ("analyst_expectations", "US.NVDA")]
    assert qqq_etf["holdings"][0]["symbol"] == "US.NVDA"
    assert nvda_analyst["warnings"] == ["ratings endpoint unavailable"]
    assert qqq_manifest["layers"]["fmp_etf_exposure"]["status"] == "fresh"
    assert qqq_manifest["layers"]["fmp_analyst_expectations"]["status"] == "skipped"
    assert nvda_manifest["layers"]["fmp_etf_exposure"]["status"] == "skipped"
    assert nvda_manifest["layers"]["fmp_analyst_expectations"]["status"] == "partial"


def test_symbol_data_store_supports_symbol_and_layer_crud(tmp_path):
    store = symbol_store.SymbolDataStore(tmp_path / "ia_store")

    created = store.create_symbol("NVDA", name="NVIDIA", tags=["ai"])
    layer_entry = store.put_layer(
        "US.NVDA",
        "sec_companyfacts",
        {
            "artifact_type": "sec_companyfacts",
            "symbol": "US.NVDA",
            "provider": "edgartools",
            "source_status": "fresh",
            "warnings": [],
            "fundamentals": {"ttm_revenue": 100.0},
        },
        provider="edgartools",
        status="fresh",
        run_id="dmr_test",
    )

    assert created["symbol"] == "US.NVDA"
    assert layer_entry["checksum"]
    assert store.get_layer("NVDA", "sec_companyfacts")["fundamentals"]["ttm_revenue"] == 100.0
    assert store.list_layers("US.NVDA")["sec_companyfacts"]["status"] == "fresh"
    assert store.list_symbols(layer="sec_companyfacts", status="fresh")[0]["symbol"] == "US.NVDA"

    stale = store.mark_layer_stale("US.NVDA", "sec_companyfacts", reason="test stale")
    assert stale["status"] == "stale"
    assert "test stale" in stale["warnings"]

    store.delete_layer("US.NVDA", "sec_companyfacts")
    assert "sec_companyfacts" not in store.list_layers("US.NVDA")

    store.delete_symbol("US.NVDA")
    assert store.get_symbol("US.NVDA")["deleted"] is True
    assert store.list_symbols() == []
    assert store.list_symbols(include_deleted=True)[0]["symbol"] == "US.NVDA"


def test_symbol_data_store_ingests_batch_without_losing_existing_layers(tmp_path):
    target = tmp_path / "target"
    batch = tmp_path / "batch"
    source_symbol_dir = batch / "symbols" / "US.NVDA"
    source_symbol_dir.mkdir(parents=True)
    (source_symbol_dir / "sec_companyfacts.json").write_text(
        json.dumps(
            {
                "artifact_type": "sec_companyfacts",
                "symbol": "US.NVDA",
                "provider": "edgartools",
                "source_status": "available",
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (source_symbol_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "symbol_data_manifest",
                "symbol": "US.NVDA",
                "market": "US",
                "source_status": "fresh",
                "layers": {
                    "sec_companyfacts": {
                        "layer": "sec_companyfacts",
                        "status": "fresh",
                        "source": "sec_companyfacts",
                        "path": str(source_symbol_dir / "sec_companyfacts.json"),
                        "warnings": [],
                    }
                },
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    store = symbol_store.SymbolDataStore(target)
    store.put_layer(
        "US.NVDA",
        "fmp_analyst_expectations",
        {
            "artifact_type": "fmp_analyst_expectations",
            "symbol": "US.NVDA",
            "provider": "financialmodelingprep",
            "source_status": "partial",
            "warnings": ["quota"],
        },
        provider="financialmodelingprep",
        status="partial",
        run_id="dmr_fmp",
    )

    manifest = store.ingest_symbol_dir(source_symbol_dir, run_id="dmr_sec")

    assert set(manifest["layers"]) == {"fmp_analyst_expectations", "sec_companyfacts"}
    assert manifest["layers"]["sec_companyfacts"]["path"] == "symbols/US.NVDA/sec_companyfacts.json"
    assert manifest["layers"]["sec_companyfacts"]["run_id"] == "dmr_sec"
    assert manifest["layers"]["fmp_analyst_expectations"]["status"] == "partial"
    assert (target / "symbols" / "US.NVDA" / "sec_companyfacts.json").exists()
    assert store.get_layer("US.NVDA", "sec_companyfacts")["provider"] == "edgartools"
    assert {row["symbol"] for row in store.list_symbols()} == {"US.NVDA"}


def test_filing_summary_writes_markdown_meta_and_manifest(monkeypatch, tmp_path):
    root = tmp_path / "ia"
    symbol_dir = root / "symbols" / "US.MRVL"
    section_dir = symbol_dir / "filing_sections" / "latest_10q"
    section_dir.mkdir(parents=True)
    (section_dir / "part_i_item_2.md").write_text(
        "Management says data center demand improved and margins were affected by product mix.",
        encoding="utf-8",
    )
    (symbol_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "symbol_data_manifest",
                "symbol": "US.MRVL",
                "market": "US",
                "source_status": "fresh",
                "layers": {},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    (symbol_dir / "filing_metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "filing_metadata",
                "symbol": "US.MRVL",
                "source_status": "available",
                "filings": {
                    "latest_10q": {
                        "form": "10-Q",
                        "filing_date": "2026-05-28",
                        "period_of_report": "2026-05-02",
                        "accession_number": "abc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (symbol_dir / "filing_sections.json").write_text(
        json.dumps(
            {
                "artifact_type": "filing_sections",
                "symbol": "US.MRVL",
                "source_status": "fresh",
                "filings": {
                    "latest_10q": {
                        "form": "10-Q",
                        "filing_date": "2026-05-28",
                        "sections": {
                            "part_i_item_2": {
                                "source_status": "fresh",
                                "item": "Item 2",
                                "title": "Management Discussion and Analysis",
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_run_summary_agent(payload):
        assert payload["symbol"] == "US.MRVL"
        assert payload["source_sections"][0]["source_label"] == "latest_10q / Item 2"
        return (
            filing_summary.FilingSummaryOutput(
                markdown=(
                    "# US.MRVL Filing Summary\n\n"
                    "## Source Files\n"
                    "- latest_10q / 10-Q / 2026-05-28 / Item 2\n\n"
                    "## Business Overview\n"
                    "Data center demand improved. [latest_10q / Item 2]\n\n"
                    "## Recent Operating Discussion\n\n"
                    "## Demand Signals\n\n"
                    "## Margin / Cost Signals\n\n"
                    "## AI / Data Center Relevance\n\n"
                    "## Key Risks\n\n"
                    "## Changes vs Prior Filing\n\n"
                    "## Open Questions\n\n"
                    "## Data Quality Notes\n"
                ),
                warnings=[],
            ),
            {"model": "test-model", "api_mode": "test"},
            {"requests": 1},
        )

    monkeypatch.setattr(filing_summary, "_run_summary_agent", fake_run_summary_agent)

    result = filing_summary.summarize_symbol_filings(symbol_dir, root=root, run_id="fsr_test")

    summary_path = symbol_dir / "filing_summary.md"
    meta_path = symbol_dir / "filing_summary.meta.json"
    manifest = json.loads((symbol_dir / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert result["status"] == "fresh"
    assert "Data center demand improved" in summary_path.read_text(encoding="utf-8")
    assert meta["source_files"][0]["source_label"] == "latest_10q / Item 2"
    assert meta["summary_path"] == "filing_summary.md"
    assert manifest["layers"]["filing_summary"]["status"] == "fresh"
    assert manifest["layers"]["filing_summary"]["path"] == "symbols/US.MRVL/filing_summary.md"
    assert manifest["layers"]["filing_summary"]["meta_path"] == "symbols/US.MRVL/filing_summary.meta.json"


def test_current_mvp_agent_skills_are_discoverable():
    pytest.importorskip("pydantic_ai_skills")

    skills = skill_runtime.discover_local_agent_skills()

    assert {
        "candidate-triage",
        "deep-research",
        "filing-narrative-summary",
        "fundamental-analysis",
        "portfolio-architect",
        "portfolio-revision",
        "portfolio-weight-formula",
        "research-intake-triage",
        "theme-discovery",
    }.issubset(set(skills))
    assert "candidate-enrichment" not in skills
    assert "thesis-synthesis" not in skills

    capability = skill_runtime.create_agent_skills_capability(["candidate-triage"])
    assert sorted(capability.toolset.skills) == ["candidate-triage"]
    skill = capability.toolset.get_skill("candidate-triage")
    assert "Enrich and triage candidates before deep research." in skill.description

    intake_capability = skill_runtime.create_agent_skills_capability(["research-intake-triage"])
    assert sorted(intake_capability.toolset.skills) == ["research-intake-triage"]
    intake_skill = intake_capability.toolset.get_skill("research-intake-triage")
    assert "Select filing analyses to read before deep research." in intake_skill.description

    deep_capability = skill_runtime.create_agent_skills_capability(["deep-research"])
    assert sorted(deep_capability.toolset.skills) == ["deep-research"]
    deep_skill = deep_capability.toolset.get_skill("deep-research")
    assert "Read filing summaries and rank candidates." in deep_skill.description

    revision_capability = skill_runtime.create_agent_skills_capability(["portfolio-revision"])
    assert sorted(revision_capability.toolset.skills) == ["portfolio-revision"]
    revision_skill = revision_capability.toolset.get_skill("portfolio-revision")
    assert "Revise an existing portfolio map." in revision_skill.description

    formula_capability = skill_runtime.create_agent_skills_capability(["portfolio-weight-formula"])
    assert sorted(formula_capability.toolset.skills) == ["portfolio-weight-formula"]
    formula_skill = formula_capability.toolset.get_skill("portfolio-weight-formula")
    assert "Score portfolio weights for formula allocation." in formula_skill.description


def test_retired_downstream_agent_skills_are_not_loadable():
    pytest.importorskip("pydantic_ai_skills")

    with pytest.raises(ValueError, match="Unknown investment assistant agent skill: candidate-enrichment"):
        skill_runtime.create_agent_skills_capability(["candidate-enrichment"])
    with pytest.raises(ValueError, match="Unknown investment assistant agent skill: thesis-synthesis"):
        skill_runtime.create_agent_skills_capability(["thesis-synthesis"])
