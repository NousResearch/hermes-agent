from __future__ import annotations

import os

import pytest

from market_monitor.fetchers.base import HttpFetcher
from market_monitor.registry import build_parser_registry

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("MARKET_MONITOR_LIVE") != "1", reason="set MARKET_MONITOR_LIVE=1 to run live parser smoke tests")
def test_live_smoke_fetch_and_parse_cpca() -> None:
    fetcher = HttpFetcher(timeout=20)
    fetch_result = fetcher.fetch(
        source_id="cpca",
        dataset_id="cpca_monthly_market",
        url="https://www.cpcaauto.com/",
    )
    parser = build_parser_registry()["cpca_monthly_market"]()
    output = parser.parse(fetch_result, dataset_id="cpca_monthly_market")
    assert output.snapshot.source_id == "cpca"
