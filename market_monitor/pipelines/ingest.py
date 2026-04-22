from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import uuid

from market_monitor.db import Database
from market_monitor.fetchers.base import content_hash, store_raw_fetch_result
from market_monitor.models import FetchResult, ParseOutput
from market_monitor.validation import validate_observations_against_history, validate_parse_output


class IngestionPipeline:
    def __init__(self, *, db: Database, raw_root: Path, parser_registry: dict[str, type]):
        self.db = db
        self.raw_root = Path(raw_root)
        self.parser_registry = parser_registry

    def ingest_fetch_result(self, fetch_result: FetchResult) -> ParseOutput:
        stored = store_raw_fetch_result(fetch_result, self.raw_root)
        digest = content_hash(stored.content_bytes)
        existing_snapshot = self.db.get_existing_snapshot(stored.source_id, digest)

        try:
            parser_cls = self.parser_registry[stored.dataset_id]
            parser = parser_cls()
            output = parser.parse(stored, dataset_id=stored.dataset_id)
        except Exception as exc:
            self.db.insert_job_log(
                job_id=str(uuid.uuid4()),
                source_id=stored.source_id,
                dataset_id=stored.dataset_id,
                run_at=_utc_now(),
                status="parse_failed",
                rows_extracted=0,
                snapshots_created=0,
                error_message=str(exc),
            )
            raise

        if existing_snapshot is not None:
            output.snapshot.snapshot_id = existing_snapshot["snapshot_id"]
            output.snapshot.local_path = existing_snapshot["local_path"]
        else:
            output.snapshot.local_path = stored.local_path
        output.snapshot.content_hash = digest
        for obs in output.observations:
            obs.snapshot_id = output.snapshot.snapshot_id
            if obs.observation_key is None:
                obs.observation_key = obs.obs_id

        validation = validate_parse_output(output.observations)
        history_validation = validate_observations_against_history(self.db, output.observations)
        output.warnings.extend(finding.message for finding in history_validation.warnings)
        validation_errors = list(validation.errors) + list(history_validation.errors)
        if validation_errors:
            error_message = "; ".join(finding.message for finding in validation_errors)
            self.db.insert_job_log(
                job_id=str(uuid.uuid4()),
                source_id=stored.source_id,
                dataset_id=stored.dataset_id,
                run_at=_utc_now(),
                status="validation_failed",
                rows_extracted=0,
                snapshots_created=0,
                error_message=error_message,
            )
            raise ValueError(error_message)

        try:
            persist_summary = self.db.persist_ingestion(
                snapshot=output.snapshot,
                entities=output.entities,
                observations=output.observations,
                observation_entities=output.observation_entities,
                warnings=output.warnings,
                snapshots_created=0 if existing_snapshot is not None else 1,
                job_id=str(uuid.uuid4()),
                run_at=_utc_now(),
            )
        except Exception as exc:
            self.db.insert_job_log(
                job_id=str(uuid.uuid4()),
                source_id=stored.source_id,
                dataset_id=stored.dataset_id,
                run_at=_utc_now(),
                status="persist_failed",
                rows_extracted=0,
                snapshots_created=0,
                error_message=str(exc),
            )
            raise
        output.snapshot.parse_status = str(persist_summary["status"])
        return output


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
