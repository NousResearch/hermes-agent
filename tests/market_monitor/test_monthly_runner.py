from __future__ import annotations

from pathlib import Path

from market_monitor.db import Database, initialize_database
from market_monitor.models import FetchResult
from market_monitor.runners import run_monthly


def test_run_monthly_executes_plan_and_writes_observations(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)

    def fake_fetch(item):
        if item.dataset_id == "caam_nev_prod_sales":
            text = "<h1>2026年3月新能源汽车产销情况简析</h1><p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>"
        elif item.dataset_id == "evcipa_monthly_infra":
            text = "<h1>2026年2月全国电动汽车充换电基础设施运行情况</h1><p>截至2026年2月，全国充电基础设施累计数量为1450.0万台。其中，公共充电桩390.0万台，随车配建私人充电桩1060.0万台。</p><p>2026年2月比2026年1月增加34.2万台，桩车增量比为1:2.7。</p>"
        elif item.dataset_id == "dongchedi_model_rank":
            text = '<script id="__DATA__" type="application/json">{"period":"2026-03","items":[{"rank":1,"brand":"比亚迪","model":"海鸥","sales":30123}]}</script>'
        elif item.dataset_id == "cada_nev_report_meta":
            text = '<h1>2026年2月份全国新能源乘用车市场深度分析报告</h1>'
        else:
            text = '<h1>2026年1月份全国新能源市场深度分析报告</h1><p>零售82.0万辆，批发95.0万辆，出口12.0万辆</p>'
        return FetchResult(
            source_id=item.source_id,
            dataset_id=item.dataset_id,
            fetch_time="2026-04-22T08:00:00Z",
            source_url=item.url,
            content_type="html",
            content_bytes=text.encode("utf-8"),
            text=text,
            local_path="",
            status_code=200,
            headers={},
        )

    results = run_monthly(db=db, raw_root=raw_root, fetch_fn=fake_fetch)

    assert len(results) == 5
    assert db.scalar("SELECT COUNT(*) FROM observations") > 0
