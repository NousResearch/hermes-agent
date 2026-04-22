from __future__ import annotations

import sqlite3
from pathlib import Path

from market_monitor.db import Database, initialize_database
from market_monitor.fetchers.base import FetchResult
from market_monitor.models import ParseOutput, RawSnapshot
from market_monitor.pipelines.ingest import IngestionPipeline
from market_monitor.registry import build_parser_registry


class DummyParser:
    source_id = "dummy"
    parser_version = "test-1"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        return ParseOutput(
            snapshot=RawSnapshot(
                snapshot_id="dummy:snapshot:1",
                source_id="dummy",
                dataset_id=dataset_id,
                fetch_time=fetch_result.fetch_time,
                source_url=fetch_result.source_url,
                period_hint="2026-03",
                content_type=fetch_result.content_type,
                content_hash="",
                local_path=fetch_result.local_path,
                parse_status="parsed",
                parser_version=self.parser_version,
            )
        )


def test_initialize_database_creates_expected_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"

    initialize_database(db_path)

    conn = sqlite3.connect(db_path)
    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    assert {
        "sources",
        "datasets",
        "raw_snapshots",
        "entities",
        "alias_mappings",
        "observations",
        "observation_entities",
        "parse_warnings",
        "ingestion_job_log",
    }.issubset(table_names)


def test_ingestion_pipeline_deduplicates_identical_raw_snapshot_content(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry={"dummy_dataset": DummyParser})

    fetch_result = FetchResult(
        source_id="dummy",
        dataset_id="dummy_dataset",
        fetch_time="2026-04-22T08:00:00Z",
        source_url="https://example.com/report",
        content_type="html",
        content_bytes=b"<html>same</html>",
        text="<html>same</html>",
        local_path="",
        status_code=200,
        headers={},
        period_hint="2026-03",
    )

    first = pipeline.ingest_fetch_result(fetch_result)
    second = pipeline.ingest_fetch_result(fetch_result)

    assert first.snapshot.content_hash == second.snapshot.content_hash
    assert db.scalar("SELECT COUNT(*) FROM raw_snapshots") == 1


def test_build_parser_registry_includes_all_mvp_datasets() -> None:
    registry = build_parser_registry()

    assert {
        "cpca_monthly_market",
        "caam_nev_prod_sales",
        "cada_nev_report_meta",
        "dongchedi_model_rank",
        "evcipa_monthly_infra",
    }.issubset(registry.keys())
