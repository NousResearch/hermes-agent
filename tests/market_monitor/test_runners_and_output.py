from __future__ import annotations

from pathlib import Path

from market_monitor.settings import load_source_catalog
from market_monitor.runners import build_run_plan, render_structured_results
from market_monitor.db import Database, initialize_database


def test_load_source_catalog_reads_all_sources_and_datasets() -> None:
    catalog = load_source_catalog(Path("configs/china_ev_market_sources.yaml"))

    assert len(catalog.sources) == 5
    assert {source.source_id for source in catalog.sources} == {"cpca", "caam", "cada", "dongchedi", "evcipa"}
    assert catalog.datasets_by_id["evcipa_monthly_infra"].source_id == "evcipa"


def test_build_run_plan_filters_by_frequency() -> None:
    catalog = load_source_catalog(Path("configs/china_ev_market_sources.yaml"))

    monthly_plan = build_run_plan(catalog, frequency="monthly")
    weekly_plan = build_run_plan(catalog, frequency="weekly")

    assert len(monthly_plan) == 5
    assert weekly_plan == []


def test_render_structured_results_groups_observations_by_dataset_and_period(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    initialize_database(db_path)
    db = Database(db_path)
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "evcipa-total",
            "evcipa-total",
            "evcipa_monthly_infra",
            "evcipa",
            "2026-02",
            "month",
            "charging_piles_total",
            "charging_infrastructure",
            "absolute",
            14500000,
            "piles",
            "2026-03-23",
        ),
    )
    db.execute(
        "INSERT INTO observations (obs_id, observation_key, dataset_id, source_id, period_label, period_type, metric_name, metric_scope, metric_type, value_numeric, unit, published_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "evcipa-public",
            "evcipa-public",
            "evcipa_monthly_infra",
            "evcipa",
            "2026-02",
            "month",
            "public_charging_piles",
            "charging_infrastructure",
            "absolute",
            3900000,
            "piles",
            "2026-03-23",
        ),
    )

    payload = render_structured_results(db, period_label="2026-02")

    assert payload["period_label"] == "2026-02"
    assert payload["datasets"][0]["dataset_id"] == "evcipa_monthly_infra"
    assert payload["datasets"][0]["metrics"][0]["metric_name"] == "charging_piles_total"
