from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request
from pathlib import Path

from market_monitor.api import build_api_payloads, start_api_server
from market_monitor.db import Database, initialize_database


def _seed_db(db: Database) -> None:
    db.execute(
        "INSERT INTO entities (entity_id, entity_type, name_norm) VALUES (?, ?, ?)",
        ("brand:比亚迪", "brand", "比亚迪"),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("market-1", "market-1", "cpca_monthly_market", "cpca", "2026-03", "month", "sales_volume", "retail", "absolute", "nev_total", 820000, "vehicles", "2026-04-08"),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, ranking, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("brand-rank-1", "brand-rank-1", "cpca_brand_rank", "cpca", "2026-03", "month", "sales_volume", "wholesale", "ranking", 210000, 1, "vehicles", "2026-04-08"),
    )
    db.execute(
        "INSERT INTO observation_entities (obs_id, entity_id, entity_role) VALUES (?, ?, ?)",
        ("brand-rank-1", "brand:比亚迪", "brand"),
    )


def test_build_api_payloads_exposes_periods_results_series_and_rankings(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _seed_db(db)

    payloads = build_api_payloads(db)

    assert payloads["periods"]["periods"] == ["2026-03"]
    assert payloads["results"]["2026-03"]["period_label"] == "2026-03"
    assert payloads["series"][0]["value_numeric"] == 820000
    assert payloads["brand_rankings"][0]["brand_name"] == "比亚迪"
    assert payloads["health"]["status"] == "ok"


def test_api_server_serves_periods_results_series_rankings_and_health(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    _seed_db(db)

    server = start_api_server(db=db, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        periods = json.loads(urllib.request.urlopen(f"{base}/periods").read().decode("utf-8"))
        results = json.loads(urllib.request.urlopen(f"{base}/results/2026-03").read().decode("utf-8"))
        query = urllib.parse.urlencode({"metric_name": "sales_volume", "metric_scope": "retail", "energy_type": "nev_total"})
        series = json.loads(urllib.request.urlopen(f"{base}/series?{query}").read().decode("utf-8"))
        rankings = json.loads(urllib.request.urlopen(f"{base}/rankings/brands?period_label=2026-03&top_n=5").read().decode("utf-8"))
        health = json.loads(urllib.request.urlopen(f"{base}/health").read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert periods["periods"] == ["2026-03"]
    assert results["period_label"] == "2026-03"
    assert series[0]["value_numeric"] == 820000
    assert rankings[0]["brand_name"] == "比亚迪"
    assert health["status"] == "ok"
