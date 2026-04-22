from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from market_monitor.backfill import ingest_backfill_fetch_results
from market_monitor.dashboard import build_dashboard_bundle, create_dashboard_payload, start_dashboard_server
from market_monitor.db import Database, initialize_database
from market_monitor.models import FetchResult


def test_backfill_ingests_multiple_historical_periods(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)

    results = [
        FetchResult(
            source_id="caam",
            dataset_id="caam_nev_prod_sales",
            fetch_time="2026-03-10T00:00:00Z",
            source_url="http://example.com/2026-02",
            content_type="html",
            content_bytes="<h1>2026年2月新能源汽车产销情况简析</h1><p>2026年2月，新能源汽车产量为100.0万辆，同比增长30.0%；销量为95.0万辆，同比增长28.0%，新能源汽车新车销量达到汽车新车总销量的40.0%。</p>".encode("utf-8"),
            text="<h1>2026年2月新能源汽车产销情况简析</h1><p>2026年2月，新能源汽车产量为100.0万辆，同比增长30.0%；销量为95.0万辆，同比增长28.0%，新能源汽车新车销量达到汽车新车总销量的40.0%。</p>",
            local_path="",
        ),
        FetchResult(
            source_id="caam",
            dataset_id="caam_nev_prod_sales",
            fetch_time="2026-04-10T00:00:00Z",
            source_url="http://example.com/2026-03",
            content_type="html",
            content_bytes="<h1>2026年3月新能源汽车产销情况简析</h1><p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>".encode("utf-8"),
            text="<h1>2026年3月新能源汽车产销情况简析</h1><p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>",
            local_path="",
        ),
    ]

    summary = ingest_backfill_fetch_results(db=db, raw_root=raw_root, fetch_results=results)

    assert summary["ingested_fetch_results"] == 2
    assert summary["observations_written"] >= 10
    periods = [row["period_label"] for row in db.query("SELECT DISTINCT period_label FROM observations ORDER BY period_label")]
    assert periods == ["2026-02", "2026-03"]


def test_build_dashboard_bundle_writes_index_and_data_json(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-1", "obs-1", "caam_nev_prod_sales", "caam", "2026-03", "month", "sales_volume", "production", "absolute", 1188000, "vehicles", "2026-04-10"),
    )

    bundle_dir = tmp_path / "dashboard"
    paths = build_dashboard_bundle(db=db, out_dir=bundle_dir)

    assert paths["index_html"].exists()
    assert paths["data_json"].exists()
    payload = json.loads(paths["data_json"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2"
    assert payload["periods"] == ["2026-03"]
    assert payload["results"]["2026-03"]["datasets"][0]["dataset_id"] == "caam_nev_prod_sales"
    html = paths["index_html"].read_text(encoding="utf-8")
    assert "China EV Market Dashboard" in html
    assert "innerHTML" not in html
    assert "api" in html


def test_dashboard_server_serves_index_and_json(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-1", "obs-1", "evcipa_monthly_infra", "evcipa", "2026-02", "month", "charging_piles_total", "charging_infrastructure", "absolute", 14500000, "piles", "2026-03-23"),
    )
    bundle_dir = tmp_path / "dashboard"
    build_dashboard_bundle(db=db, out_dir=bundle_dir)

    server = start_dashboard_server(bundle_dir=bundle_dir, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        index_html = urllib.request.urlopen(f"{base}/index.html").read().decode("utf-8")
        data_json = json.loads(urllib.request.urlopen(f"{base}/data.json").read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "China EV Market Dashboard" in index_html
    assert data_json["schema_version"] == "2"
    assert data_json["periods"] == ["2026-02"]


def test_create_dashboard_payload_orders_periods_descending(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-a", "obs-a", "caam_nev_prod_sales", "caam", "2026-02", "month", "sales_volume", "production", "absolute", 950000, "vehicles", "2026-03-10"),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-b", "obs-b", "caam_nev_prod_sales", "caam", "2026-03", "month", "sales_volume", "production", "absolute", 1188000, "vehicles", "2026-04-10"),
    )

    payload = create_dashboard_payload(db)

    assert payload["schema_version"] == "2"
    assert payload["periods"] == ["2026-03", "2026-02"]
