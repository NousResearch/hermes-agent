from __future__ import annotations

from market_monitor.parsers.caam import CaamNevProdSalesParser
from market_monitor.parsers.dongchedi import DongchediModelRankParser
from market_monitor.parsers.evcipa import EvcipaInfraParser
from market_monitor.fetchers.base import FetchResult


CAAM_HTML = """
<html><body>
<h1>2026年3月新能源汽车产销情况简析</h1>
<p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>
</body></html>
"""

EVCIPA_HTML = """
<html><body>
<h1>2026年2月全国电动汽车充换电基础设施运行情况</h1>
<p>截至2026年2月，全国充电基础设施累计数量为1450.0万台，同比上升49.1%。其中，公共充电桩390.0万台，随车配建私人充电桩1060.0万台。</p>
<p>2026年2月比2026年1月增加34.2万台，桩车增量比为1:2.7。</p>
</body></html>
"""

DONGCHEDI_HTML = """
<html><body>
<script id="__DATA__" type="application/json">
{
  "period":"2026-03",
  "items":[
    {"rank":1,"brand":"比亚迪","model":"海鸥","sales":30123},
    {"rank":2,"brand":"特斯拉","model":"Model Y","sales":28999}
  ]
}
</script>
</body></html>
"""


def _fetch_result(source_id: str, dataset_id: str, text: str) -> FetchResult:
    return FetchResult(
        source_id=source_id,
        dataset_id=dataset_id,
        fetch_time="2026-04-22T08:00:00Z",
        source_url=f"https://example.com/{dataset_id}",
        content_type="html",
        content_bytes=text.encode("utf-8"),
        text=text,
        local_path=f"/tmp/{dataset_id}.html",
        status_code=200,
        headers={},
        period_hint=None,
    )


def test_caam_parser_extracts_market_totals_and_growth_metrics() -> None:
    parser = CaamNevProdSalesParser()
    result = parser.parse(_fetch_result("caam", "caam_nev_prod_sales", CAAM_HTML))

    values = {(obs.metric_name, obs.metric_type): obs.value_numeric for obs in result.observations}

    assert values[("production_volume", "absolute")] == 1234000
    assert values[("sales_volume", "absolute")] == 1188000
    assert values[("production_volume", "yoy")] == 45.6
    assert values[("sales_volume", "yoy")] == 41.2
    assert values[("market_share", "share")] == 43.1


def test_evcipa_parser_extracts_totals_components_increment_and_ratio() -> None:
    parser = EvcipaInfraParser()
    result = parser.parse(_fetch_result("evcipa", "evcipa_monthly_infra", EVCIPA_HTML))

    values = {(obs.metric_name, obs.metric_type): obs.value_numeric for obs in result.observations}

    assert values[("charging_piles_total", "absolute")] == 14500000
    assert values[("public_charging_piles", "absolute")] == 3900000
    assert values[("private_charging_piles", "absolute")] == 10600000
    assert values[("monthly_increment", "absolute")] == 342000
    assert values[("pile_vehicle_ratio", "absolute")] == 2.7


def test_dongchedi_parser_extracts_rankings_brands_and_models() -> None:
    parser = DongchediModelRankParser()
    result = parser.parse(_fetch_result("dongchedi", "dongchedi_model_rank", DONGCHEDI_HTML))

    ranked = sorted(
        (obs.ranking, obs.value_numeric) for obs in result.observations if obs.metric_type == "ranking"
    )
    entity_names = {entity.name_norm for entity in result.entities}

    assert ranked == [(1, 30123), (2, 28999)]
    assert {"比亚迪", "特斯拉", "海鸥", "Model Y"}.issubset(entity_names)


def test_evcipa_parser_rejects_negative_numbers_from_malformed_content() -> None:
    parser = EvcipaInfraParser()
    bad_html = EVCIPA_HTML.replace("34.2万台", "-34.2万台")

    try:
        parser.parse(_fetch_result("evcipa", "evcipa_monthly_infra", bad_html))
    except ValueError as exc:
        assert "negative" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for malformed negative infrastructure increment")
