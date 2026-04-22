from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from market_monitor.db import Database
from market_monitor.fetchers.base import HttpFetcher
from market_monitor.models import FetchResult
from market_monitor.pipelines.ingest import IngestionPipeline
from market_monitor.registry import build_parser_registry
from market_monitor.settings import SourceCatalog, load_source_catalog


DEFAULT_CONFIG_PATH = Path("configs/china_ev_market_sources.yaml")
TERMINAL_RUN_STATUSES = {
    "success",
    "persisted_with_warnings",
    "fetch_failed",
    "parse_failed",
    "validation_failed",
    "persist_failed",
    "skipped",
}


@dataclass(frozen=True)
class RunItem:
    source_id: str
    dataset_id: str
    dataset_name: str
    frequency: str
    url: str


def build_run_plan(catalog: SourceCatalog, *, frequency: str) -> list[RunItem]:
    plan: list[RunItem] = []
    for source in catalog.sources:
        if not source.active or source.update_frequency != frequency:
            continue
        for dataset in source.datasets:
            plan.append(
                RunItem(
                    source_id=source.source_id,
                    dataset_id=dataset.dataset_id,
                    dataset_name=dataset.dataset_name,
                    frequency=frequency,
                    url=dataset.data_url,
                )
            )
    return plan


def render_structured_results(db: Database, *, period_label: str) -> dict:
    rows = db.query(
        """
        SELECT dataset_id, source_id, obs_id, observation_key, metric_name, metric_scope, metric_type, energy_type,
               value_numeric, value_text, unit, ranking, published_at, source_url, is_latest, revision_no, snapshot_id
        FROM observations
        WHERE period_label = ? AND is_latest = 1
        ORDER BY dataset_id, ranking IS NULL, ranking, metric_name, revision_no
        """,
        (period_label,),
    )
    warning_rows = db.query(
        """
        SELECT DISTINCT rs.dataset_id, pw.warning_text
        FROM parse_warnings pw
        JOIN raw_snapshots rs ON rs.snapshot_id = pw.snapshot_id
        WHERE rs.period_hint = ? OR EXISTS (
            SELECT 1 FROM observations o WHERE o.snapshot_id = rs.snapshot_id AND o.period_label = ?
        )
        ORDER BY rs.dataset_id, pw.warning_text
        """,
        (period_label, period_label),
    )
    warnings_by_dataset: dict[str, list[str]] = {}
    for row in warning_rows:
        warnings_by_dataset.setdefault(row["dataset_id"], []).append(row["warning_text"])

    grouped: dict[str, dict] = {}
    for row in rows:
        item = grouped.setdefault(
            row["dataset_id"],
            {
                "dataset_id": row["dataset_id"],
                "source_id": row["source_id"],
                "warnings": warnings_by_dataset.get(row["dataset_id"], []),
                "freshness": {"latest_published_at": None, "metric_count": 0},
                "metrics": [],
            },
        )
        item["metrics"].append(
            {
                "obs_id": row["obs_id"],
                "observation_key": row["observation_key"],
                "metric_name": row["metric_name"],
                "metric_scope": row["metric_scope"],
                "metric_type": row["metric_type"],
                "energy_type": row["energy_type"],
                "value_numeric": row["value_numeric"],
                "value_text": row["value_text"],
                "unit": row["unit"],
                "ranking": row["ranking"],
                "published_at": row["published_at"],
                "source_url": row["source_url"],
                "snapshot_id": row["snapshot_id"],
                "is_latest": bool(row["is_latest"]),
                "revision_no": row["revision_no"],
            }
        )
        item["freshness"]["metric_count"] += 1
        item["freshness"]["latest_published_at"] = _max_iso_date(
            item["freshness"]["latest_published_at"],
            row["published_at"],
        )
    return {
        "period_label": period_label,
        "datasets": list(grouped.values()),
    }


def run_frequency(
    *,
    db: Database,
    raw_root: Path,
    frequency: str,
    fetch_fn: Callable[[RunItem], FetchResult] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[dict]:
    catalog = load_source_catalog(config_path)
    plan = build_run_plan(catalog, frequency=frequency)
    pipeline = IngestionPipeline(db=db, raw_root=raw_root, parser_registry=build_parser_registry())
    fetcher = HttpFetcher()
    results: list[dict] = []
    for item in plan:
        try:
            fetched = fetch_fn(item) if fetch_fn is not None else fetcher.fetch(source_id=item.source_id, dataset_id=item.dataset_id, url=item.url)
        except Exception as exc:
            db.insert_job_log(
                job_id=f"fetch-failed:{item.dataset_id}:{_utc_now()}",
                source_id=item.source_id,
                dataset_id=item.dataset_id,
                run_at=_utc_now(),
                status="fetch_failed",
                rows_extracted=0,
                snapshots_created=0,
                error_message=str(exc),
            )
            results.append(
                {
                    "source_id": item.source_id,
                    "dataset_id": item.dataset_id,
                    "observation_count": 0,
                    "warning_count": 0,
                    "snapshot_id": None,
                    "status": "fetch_failed",
                    "error_message": str(exc),
                }
            )
            continue

        try:
            parsed = pipeline.ingest_fetch_result(fetched)
            results.append(
                {
                    "source_id": item.source_id,
                    "dataset_id": item.dataset_id,
                    "observation_count": len(parsed.observations),
                    "warning_count": len(parsed.warnings),
                    "snapshot_id": parsed.snapshot.snapshot_id,
                    "status": parsed.snapshot.parse_status,
                    "error_message": None,
                }
            )
        except Exception as exc:
            status = _latest_job_status(db, item.source_id, item.dataset_id) or "persist_failed"
            results.append(
                {
                    "source_id": item.source_id,
                    "dataset_id": item.dataset_id,
                    "observation_count": 0,
                    "warning_count": 0,
                    "snapshot_id": None,
                    "status": status,
                    "error_message": str(exc),
                }
            )
    return results


def summarize_run_status(results: list[dict]) -> str:
    if not results:
        return "skipped"
    statuses = {result.get("status") for result in results}
    if statuses <= {"success", "persisted_with_warnings", "skipped"}:
        return "success" if statuses != {"skipped"} else "skipped"
    if statuses.isdisjoint({"success", "persisted_with_warnings", "skipped"}):
        for preferred in ("fetch_failed", "parse_failed", "validation_failed", "persist_failed"):
            if preferred in statuses:
                return preferred
        return next(iter(statuses)) or "persist_failed"
    return "partial_success"


def run_monthly(*, db: Database, raw_root: Path, fetch_fn: Callable[[RunItem], FetchResult] | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict]:
    return run_frequency(db=db, raw_root=raw_root, frequency="monthly", fetch_fn=fetch_fn, config_path=config_path)


def run_weekly(*, db: Database, raw_root: Path, fetch_fn: Callable[[RunItem], FetchResult] | None = None, config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict]:
    return run_frequency(db=db, raw_root=raw_root, frequency="weekly", fetch_fn=fetch_fn, config_path=config_path)


def _max_iso_date(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def _latest_job_status(db: Database, source_id: str, dataset_id: str) -> str | None:
    rows = db.query(
        """
        SELECT status FROM ingestion_job_log
        WHERE source_id = ? AND dataset_id = ?
        ORDER BY run_at DESC
        LIMIT 1
        """,
        (source_id, dataset_id),
    )
    return rows[0]["status"] if rows else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
