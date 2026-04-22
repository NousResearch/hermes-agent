from __future__ import annotations

from market_monitor.parsers import (
    CaamNevProdSalesParser,
    CadaNevReportMetaParser,
    CpcaMonthlyMarketParser,
    DongchediModelRankParser,
    EvcipaInfraParser,
)


def build_parser_registry() -> dict[str, type]:
    return {
        "cpca_monthly_market": CpcaMonthlyMarketParser,
        "caam_nev_prod_sales": CaamNevProdSalesParser,
        "cada_nev_report_meta": CadaNevReportMetaParser,
        "dongchedi_model_rank": DongchediModelRankParser,
        "evcipa_monthly_infra": EvcipaInfraParser,
    }
