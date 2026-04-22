from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from market_monitor.api import start_api_server
from market_monitor.cli import build_dashboard, start_local_servers
from market_monitor.db import Database, initialize_database
from market_monitor.models import FetchResult
from market_monitor.runners import run_monthly, summarize_run_status
from market_monitor.validation import validate_observations_against_history


def _insert_market_observation(db: Database, *, obs_id: str, period_label: str, value_numeric: float, published_at: str = "2026-04-08") -> None:
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            obs_id,
            "logical:sales:retail:nev_total",
            "cpca_monthly_market",
            "cpca",
            period_label,
            "month",
            "sales_volume",
            "retail",
            "absolute",
            "nev_total",
            value_numeric,
            "vehicles",
            published_at,
        ),
    )


def test_validate_observations_against_history_flags_suspicious_jumps(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-old", period_label="2026-02", value_numeric=100000)

    findings = validate_observations_against_history(
        db,
        [
            {
                "observation_key": "logical:sales:retail:nev_total",
                "metric_name": "sales_volume",
                "metric_scope": "retail",
                "value_numeric": 250000,
                "period_label": "2026-03",
            }
        ],
    )

    assert findings.warnings
    assert findings.warnings[0].code == "suspicious_jump"


def test_validate_observations_against_history_ignores_future_periods(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-newer", period_label="2026-03", value_numeric=300000)

    findings = validate_observations_against_history(
        db,
        [
            {
                "dataset_id": "cpca_monthly_market",
                "source_id": "cpca",
                "energy_type": "nev_total",
                "metric_name": "sales_volume",
                "metric_scope": "retail",
                "metric_type": "absolute",
                "value_numeric": 120000,
                "period_label": "2026-02",
            }
        ],
    )

    assert findings.errors == ()
    assert findings.warnings == ()


def test_validate_observations_against_history_uses_latest_prior_revision_only(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at, is_latest, revision_no) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-r1", "logical:sales:retail:nev_total", "cpca_monthly_market", "cpca", "2026-02", "month", "sales_volume", "retail", "absolute", "nev_total", 100000, "vehicles", "2026-03-08", 0, 1),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at, is_latest, revision_no) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("obs-r2", "logical:sales:retail:nev_total", "cpca_monthly_market", "cpca", "2026-02", "month", "sales_volume", "retail", "absolute", "nev_total", 200000, "vehicles", "2026-03-09", 1, 2),
    )

    findings = validate_observations_against_history(
        db,
        [
            {
                "dataset_id": "cpca_monthly_market",
                "source_id": "cpca",
                "energy_type": "nev_total",
                "metric_name": "sales_volume",
                "metric_scope": "retail",
                "metric_type": "absolute",
                "value_numeric": 220000,
                "period_label": "2026-03",
            }
        ],
    )

    assert findings.errors == ()
    assert findings.warnings == ()


def test_run_monthly_records_fetch_and_validation_failures_and_returns_partial_success(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-old", period_label="2026-02", value_numeric=100000)

    def fake_fetch(item):
        if item.dataset_id == "caam_nev_prod_sales":
            return FetchResult(
                source_id=item.source_id,
                dataset_id=item.dataset_id,
                fetch_time="2026-04-22T08:00:00Z",
                source_url=item.url,
                content_type="html",
                content_bytes=(
                    "<h1>2026年3月新能源汽车产销情况简析</h1>"
                    "<p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>"
                ).encode("utf-8"),
                text="<h1>2026年3月新能源汽车产销情况简析</h1><p>2026年3月，新能源汽车产量为123.4万辆，同比增长45.6%；销量为118.8万辆，同比增长41.2%，新能源汽车新车销量达到汽车新车总销量的43.1%。</p>",
                local_path="",
                status_code=200,
                headers={},
            )
        if item.dataset_id == "cpca_monthly_market":
            return FetchResult(
                source_id=item.source_id,
                dataset_id=item.dataset_id,
                fetch_time="2026-04-22T08:00:00Z",
                source_url=item.url,
                content_type="html",
                content_bytes="<h1>2026年3月全国新能源市场深度分析报告</h1><p>零售40.0万辆，批发95.0万辆，出口12.0万辆</p>".encode("utf-8"),
                text="<h1>2026年3月全国新能源市场深度分析报告</h1><p>零售40.0万辆，批发95.0万辆，出口12.0万辆</p>",
                local_path="",
                status_code=200,
                headers={},
            )
        raise RuntimeError(f"boom:{item.dataset_id}")

    results = run_monthly(db=db, raw_root=raw_root, fetch_fn=fake_fetch)

    statuses = {item["dataset_id"]: item["status"] for item in results}
    assert statuses["caam_nev_prod_sales"] == "success"
    assert statuses["cpca_monthly_market"] == "validation_failed"
    assert statuses["cada_nev_report_meta"] == "fetch_failed"
    assert summarize_run_status(results) == "partial_success"
    assert db.scalar("SELECT COUNT(*) FROM ingestion_job_log WHERE status = 'fetch_failed'") >= 1
    assert db.scalar("SELECT COUNT(*) FROM ingestion_job_log WHERE status = 'validation_failed'") == 1


def test_dashboard_bundle_contains_charting_and_freshness_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-1", period_label="2026-02", value_numeric=100000, published_at="2026-03-08")
    _insert_market_observation(db, obs_id="obs-2", period_label="2026-03", value_numeric=120000, published_at="2026-04-08")

    bundle = build_dashboard(db_path=db_path, out_dir=tmp_path / "dashboard")

    payload = json.loads(bundle["data_json"].read_text(encoding="utf-8"))
    html = bundle["index_html"].read_text(encoding="utf-8")
    dataset = payload["results"]["2026-03"]["datasets"][0]
    assert dataset["freshness"]["latest_published_at"] == "2026-04-08"
    assert dataset["freshness"]["metric_count"] >= 1
    assert "series-chart" in html
    assert "svg" in html
    assert "http://127.0.0.1:* http://localhost:*" in html


def test_cli_start_local_servers_exposes_api_and_dashboard(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-1", period_label="2026-03", value_numeric=120000, published_at="2026-04-08")

    servers = start_local_servers(db_path=db_path, out_dir=tmp_path / "dashboard", api_port=0, dashboard_port=0)
    api_server = servers["api_server"]
    dashboard_server = servers["dashboard_server"]
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    dashboard_thread = threading.Thread(target=dashboard_server.serve_forever, daemon=True)
    api_thread.start()
    dashboard_thread.start()
    try:
        periods = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{api_server.server_port}/periods").read().decode("utf-8"))
        index_html = urllib.request.urlopen(f"http://127.0.0.1:{dashboard_server.server_port}/index.html").read().decode("utf-8")
    finally:
        api_server.shutdown()
        dashboard_server.shutdown()
        api_server.server_close()
        dashboard_server.server_close()
        api_thread.join(timeout=2)
        dashboard_thread.join(timeout=2)

    assert periods["periods"] == ["2026-03"]
    assert "China EV Market Dashboard" in index_html


def test_api_health_reports_period_count(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-1", period_label="2026-03", value_numeric=120000)

    server = start_api_server(db=db, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health")
        payload = json.loads(response.read().decode("utf-8"))
        cors_header = response.headers.get("Access-Control-Allow-Origin")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["status"] == "ok"
    assert payload["period_count"] == 1
    assert cors_header == "*"


def test_api_returns_400_for_invalid_top_n(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _insert_market_observation(db, obs_id="obs-1", period_label="2026-03", value_numeric=120000)

    server = start_api_server(db=db, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/rankings/brands?period_label=2026-03&top_n=abc")
        body = json.loads(exc_info.value.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exc_info.value.code == 400
    assert body["error"] == "invalid_top_n"


def test_start_local_servers_rejects_non_loopback_without_allow_remote(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    with pytest.raises(ValueError):
        start_local_servers(db_path=db_path, out_dir=tmp_path / "dashboard", api_host="0.0.0.0")


def test_summarize_run_status_preserves_specific_terminal_failure() -> None:
    assert summarize_run_status([{"status": "validation_failed"}]) == "validation_failed"
    assert summarize_run_status([{"status": "parse_failed"}]) == "parse_failed"
