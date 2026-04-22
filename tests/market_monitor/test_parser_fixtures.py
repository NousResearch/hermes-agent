from __future__ import annotations

from pathlib import Path

from market_monitor.fetchers.base import FetchResult
from market_monitor.parsers.cpca import CpcaMonthlyMarketParser
from market_monitor.parsers.dongchedi import DongchediModelRankParser

FIXTURE_DIR = Path("tests/fixtures/market_monitor")


def _fetch_result(source_id: str, dataset_id: str, fixture_name: str, *, period_hint: str | None = None) -> FetchResult:
    text = (FIXTURE_DIR / fixture_name).read_text(encoding="utf-8")
    return FetchResult(
        source_id=source_id,
        dataset_id=dataset_id,
        fetch_time="2026-04-22T08:00:00Z",
        source_url=f"https://example.com/{fixture_name}",
        content_type="html",
        content_bytes=text.encode("utf-8"),
        text=text,
        local_path=f"/tmp/{fixture_name}",
        status_code=200,
        headers={},
        period_hint=period_hint,
    )


def test_dongchedi_parser_accepts_window_initial_state_fixture() -> None:
    parser = DongchediModelRankParser()
    result = parser.parse(_fetch_result("dongchedi", "dongchedi_model_rank", "dongchedi_initial_state.html"))

    assert [obs.ranking for obs in result.observations] == [1, 2]
    assert {entity.name_norm for entity in result.entities} >= {"比亚迪", "海鸥"}


def test_cpca_parser_normalizes_period_hint_when_page_only_has_compact_month() -> None:
    parser = CpcaMonthlyMarketParser()
    result = parser.parse(
        _fetch_result(
            "cpca",
            "cpca_monthly_market",
            "cpca_period_hint.html",
            period_hint="2026年3月",
        )
    )

    scopes = {obs.metric_scope: obs.value_numeric for obs in result.observations}
    assert result.observations[0].period_label == "2026-03"
    assert scopes["retail"] == 820000
    assert scopes["wholesale"] == 950000
    assert scopes["export"] == 120000
