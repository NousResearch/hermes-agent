from __future__ import annotations

from pathlib import Path
import sqlite3

from market_monitor.db import Database, initialize_database
from market_monitor.fetchers.base import FetchResult
from market_monitor.models import ObservationEntityLink, ObservationRecord, ParseOutput, RawSnapshot
from market_monitor.pipelines.ingest import IngestionPipeline


class RevisionParser:
    source_id = "dummy"
    parser_version = "rev-1"

    def __init__(self, value: float = 100.0):
        self.value = value

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        obs = ObservationRecord(
            obs_id="logical:sales",
            dataset_id=dataset_id or "dummy_dataset",
            source_id="dummy",
            period_label="2026-03",
            period_type="month",
            metric_name="sales_volume",
            metric_scope="retail",
            metric_type="absolute",
            value_numeric=self.value,
            unit="vehicles",
            published_at=fetch_result.fetch_time[:10],
            source_url=fetch_result.source_url,
        )
        return ParseOutput(
            snapshot=RawSnapshot(
                snapshot_id=f"dummy:{fetch_result.fetch_time}",
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
            ),
            observations=[obs],
            warnings=["source layout changed"],
        )


class BrokenLinkParser:
    source_id = "dummy"
    parser_version = "broken-1"

    def parse(self, fetch_result: FetchResult, dataset_id: str | None = None) -> ParseOutput:
        obs = ObservationRecord(
            obs_id="logical:broken",
            dataset_id=dataset_id or "dummy_dataset",
            source_id="dummy",
            period_label="2026-03",
            period_type="month",
            metric_name="sales_volume",
            metric_scope="retail",
            metric_type="absolute",
            value_numeric=10,
            unit="vehicles",
            published_at=fetch_result.fetch_time[:10],
            source_url=fetch_result.source_url,
        )
        return ParseOutput(
            snapshot=RawSnapshot(
                snapshot_id=f"dummy:{fetch_result.fetch_time}",
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
            ),
            observations=[obs],
            observation_entities=[ObservationEntityLink(obs_id=obs.obs_id, entity_id="brand:missing", entity_role="brand")],
        )


def _fetch(fetch_time: str, body: str = "<html>x</html>") -> FetchResult:
    return FetchResult(
        source_id="dummy",
        dataset_id="dummy_dataset",
        fetch_time=fetch_time,
        source_url="https://example.com/report",
        content_type="html",
        content_bytes=body.encode("utf-8"),
        text=body,
        local_path="",
        status_code=200,
        headers={},
        period_hint="2026-03",
    )


def test_ingestion_is_atomic_and_logs_persist_failed_when_link_insert_breaks(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry={"dummy_dataset": BrokenLinkParser})

    try:
        pipeline.ingest_fetch_result(_fetch("2026-04-22T08:00:00Z"))
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("Expected sqlite3.IntegrityError from broken entity link")

    assert db.scalar("SELECT COUNT(*) FROM raw_snapshots") == 0
    assert db.scalar("SELECT COUNT(*) FROM observations") == 0
    assert db.scalar("SELECT COUNT(*) FROM observation_entities") == 0
    assert db.scalar("SELECT COUNT(*) FROM ingestion_job_log WHERE status = 'persist_failed'") == 1


def test_repeated_identical_observation_does_not_create_new_revision(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry={"dummy_dataset": RevisionParser})

    pipeline.ingest_fetch_result(_fetch("2026-04-22T08:00:00Z", "<html>a</html>"))
    pipeline.ingest_fetch_result(_fetch("2026-04-23T08:00:00Z", "<html>b</html>"))

    rows = db.query("SELECT obs_id, observation_key, revision_no, is_latest, value_numeric FROM observations")
    assert len(rows) == 1
    assert rows[0]["revision_no"] == 1
    assert rows[0]["is_latest"] == 1


def test_changed_value_creates_new_revision_and_marks_old_not_latest(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    parser_registry = {
        "dummy_dataset": lambda: RevisionParser(100.0),
    }
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry=parser_registry)
    pipeline.ingest_fetch_result(_fetch("2026-04-22T08:00:00Z", "<html>a</html>"))

    pipeline2 = IngestionPipeline(db=db, raw_root=raw_root, parser_registry={"dummy_dataset": lambda: RevisionParser(200.0)})
    pipeline2.ingest_fetch_result(_fetch("2026-04-24T08:00:00Z", "<html>changed</html>"))

    rows = db.query("SELECT obs_id, observation_key, revision_no, is_latest, value_numeric FROM observations ORDER BY revision_no")
    assert len(rows) == 2
    assert rows[0]["revision_no"] == 1 and rows[0]["is_latest"] == 0
    assert rows[1]["revision_no"] == 2 and rows[1]["is_latest"] == 1
    assert rows[0]["observation_key"] == rows[1]["observation_key"]


def test_parse_warnings_are_persisted(tmp_path: Path) -> None:
    db_path = tmp_path / "monitor.sqlite"
    raw_root = tmp_path / "raw"
    initialize_database(db_path)
    db = Database(db_path)
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry={"dummy_dataset": RevisionParser})

    pipeline.ingest_fetch_result(_fetch("2026-04-22T08:00:00Z"))

    warnings = db.query("SELECT warning_text FROM parse_warnings")
    assert [row["warning_text"] for row in warnings] == ["source layout changed"]
