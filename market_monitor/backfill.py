from __future__ import annotations

from pathlib import Path

from market_monitor.db import Database
from market_monitor.models import FetchResult
from market_monitor.pipelines.ingest import IngestionPipeline
from market_monitor.registry import build_parser_registry


def ingest_backfill_fetch_results(*, db: Database, raw_root: Path, fetch_results: list[FetchResult]) -> dict:
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry=build_parser_registry())
    ingested = 0
    observations_written = 0
    for fetch_result in fetch_results:
        parsed = pipeline.ingest_fetch_result(fetch_result)
        ingested += 1
        observations_written += len(parsed.observations)
    return {
        "ingested_fetch_results": ingested,
        "observations_written": observations_written,
    }
