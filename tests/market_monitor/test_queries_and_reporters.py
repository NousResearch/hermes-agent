from __future__ import annotations

from pathlib import Path

from market_monitor.db import Database, initialize_database
from market_monitor.queries import get_brand_ranking, get_latest_market_snapshot, get_metric_series
from market_monitor.reporters import render_monthly_summary


def test_queries_return_latest_market_snapshot_and_series(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)

    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "obs-1",
            "obs-sales-2026-02",
            "caam_nev_prod_sales",
            "caam",
            "2026-02",
            "month",
            "sales_volume",
            "production",
            "absolute",
            "nev_total",
            1000000,
            "vehicles",
            "2026-03-10",
        ),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "obs-2",
            "obs-sales-2026-03",
            "caam_nev_prod_sales",
            "caam",
            "2026-03",
            "month",
            "sales_volume",
            "production",
            "absolute",
            "nev_total",
            1188000,
            "vehicles",
            "2026-04-10",
        ),
    )

    latest = get_latest_market_snapshot(db, metric_name="sales_volume", metric_scope="production")
    series = get_metric_series(db, metric_name="sales_volume", metric_scope="production")

    assert latest["period_label"] == "2026-03"
    assert latest["value_numeric"] == 1188000
    assert [row["period_label"] for row in series] == ["2026-02", "2026-03"]


def test_brand_ranking_and_monthly_summary_rendering(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)

    db.execute(
        "INSERT INTO entities (entity_id, entity_type, name_norm) VALUES (?, ?, ?)",
        ("brand:比亚迪", "brand", "比亚迪"),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, ranking, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "brand-rank-1",
            "brand-rank-1",
            "cpca_brand_rank",
            "cpca",
            "2026-03",
            "month",
            "sales_volume",
            "wholesale",
            "ranking",
            210000,
            1,
            "vehicles",
            "2026-04-08",
        ),
    )
    db.execute(
        "INSERT INTO observation_entities (obs_id, entity_id, entity_role) VALUES (?, ?, ?)",
        ("brand-rank-1", "brand:比亚迪", "brand"),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, energy_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "market-1",
            "market-1",
            "cpca_monthly_market",
            "cpca",
            "2026-03",
            "month",
            "sales_volume",
            "retail",
            "absolute",
            "nev_total",
            820000,
            "vehicles",
            "2026-04-08",
        ),
    )

    ranking = get_brand_ranking(db, period_label="2026-03", top_n=5)
    summary = render_monthly_summary(db, period_label="2026-03")

    assert ranking[0]["brand_name"] == "比亚迪"
    assert ranking[0]["ranking"] == 1
    assert "2026-03" in summary
    assert "820000" in summary
    assert "比亚迪" in summary
