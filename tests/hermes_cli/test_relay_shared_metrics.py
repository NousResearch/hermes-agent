"""Focused tests for the Hermes shared-metrics durable store."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hermes_cli.observability import shared_metrics as shared_metrics_module
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_contract import (
    COUNT_BUCKETS,
    DURATION_BUCKETS,
    EXECUTION_SURFACES,
    MODEL_FAMILIES,
    MODEL_LOCALITIES,
    MODEL_OUTCOMES,
    PRIMARY_MODEL_CALL_ROLE,
    PROVIDER_FAMILIES,
    TASK_END_REASONS,
    TASK_ENTRYPOINTS,
    TASK_OUTCOMES,
    TASK_TERMINATIONS,
    count_bucket,
    duration_bucket,
    execution_surface,
    model_call_outcome,
    model_call_dimensions,
    model_family,
    model_locality,
    provider_family,
    task_counter,
    task_start_fields,
    task_terminal_fields,
    task_terminal_state,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "hermes_cli"
    / "observability"
    / "schemas"
    / "hermes.shared_metrics.v1.schema.json"
)


def _schema_validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )


def _package_dimension_schema() -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"]["model_call_counter"]["properties"]["dimensions"]


def _task_dimension_schema(kind: str) -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"][kind]["properties"]["dimensions"]


def _dimensions() -> dict[str, str]:
    return {
        "call_role": PRIMARY_MODEL_CALL_ROLE,
        "locality": "remote",
        "model_family": "claude",
        "outcome": "success",
        "provider_family": "direct",
    }


def _record_model_calls_in_process(
    database_path: str,
    outbox_directory: str,
    count: int,
    start_barrier: Any | None = None,
) -> None:
    if start_barrier is not None:
        start_barrier.wait()
    store = SharedMetricsStore(Path(database_path), Path(outbox_directory))
    for _ in range(count):
        store.record_model_call(_dimensions(), "test-version")


def test_model_call_counter_survives_restart_and_exports_only_new_deltas(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    store.record_model_call(_dimensions(), "test-version")

    first_paths = store.create_and_export_package()

    assert len(first_paths) == 1
    first_package = json.loads(first_paths[0].read_text(encoding="utf-8"))
    _schema_validator().validate(first_package)
    uuid.UUID(first_package["package_id"])
    uuid.UUID(first_package["install_id"])
    assert first_package["schema_version"] == "hermes.shared_metrics.v1"
    assert first_package["resource"] == {"hermes_version": "test-version"}
    assert first_package["metrics"] == [
        {
            "name": "hermes.model_call.count",
            "type": "counter",
            "dimensions": _dimensions(),
            "value": 2,
        }
    ]

    restarted = SharedMetricsStore(database_path, outbox_directory)
    assert restarted.counter_snapshot()[0]["value"] == 2
    assert restarted.counter_snapshot()[0]["packaged_value"] == 2
    assert restarted.create_and_export_package() == []
    assert len(list(outbox_directory.glob("*.json"))) == 1

    restarted.record_model_call(_dimensions(), "test-version")
    second_paths = restarted.create_and_export_package()

    assert len(second_paths) == 1
    second_package = json.loads(second_paths[0].read_text(encoding="utf-8"))
    assert second_package["package_id"] != first_package["package_id"]
    assert second_package["install_id"] == first_package["install_id"]
    assert second_package["metrics"][0]["value"] == 1
    assert restarted.counter_snapshot()[0]["value"] == 3
    assert restarted.counter_snapshot()[0]["packaged_value"] == 3


def test_due_export_runs_once_per_utc_day_and_catches_up_pending_deltas(
    tmp_path, monkeypatch
):
    current_time = datetime(2026, 7, 28, 9, tzinfo=timezone.utc)
    monkeypatch.setattr(shared_metrics_module, "_utc_now", lambda: current_time)
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    store.record_model_call(_dimensions(), "test-version")
    assert len(store.create_and_export_package_if_due()) == 1

    current_time = datetime(2026, 7, 28, 18, tzinfo=timezone.utc)
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_model_call(_dimensions(), "test-version")
    assert store.create_and_export_package_if_due() == []
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 1
    assert store.counter_snapshot()[0] == {
        "period_start": "2026-07-28",
        "metric_name": "hermes.model_call.count",
        "hermes_version": "test-version",
        "dimensions": _dimensions(),
        "value": 2,
        "packaged_value": 1,
    }

    current_time = datetime(2026, 7, 29, 9, tzinfo=timezone.utc)
    store.record_model_call(_dimensions(), "test-version")
    assert len(store.create_and_export_package_if_due()) == 2
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 3
    assert all(
        row["value"] == row["packaged_value"] for row in store.counter_snapshot()
    )

    store.record_model_call(_dimensions(), "test-version")
    assert store.create_and_export_package_if_due() == []
    assert len(list((tmp_path / "outbox").glob("*.json"))) == 3


def test_package_schema_matches_the_model_call_contract():
    properties = _package_dimension_schema()["properties"]

    assert properties["call_role"] == {"const": PRIMARY_MODEL_CALL_ROLE}
    assert set(properties["locality"]["enum"]) == MODEL_LOCALITIES
    assert set(properties["model_family"]["enum"]) == MODEL_FAMILIES
    assert set(properties["outcome"]["enum"]) == MODEL_OUTCOMES
    assert set(properties["provider_family"]["enum"]) == PROVIDER_FAMILIES
































def test_retention_prunes_only_expired_exported_history(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)

    store.record_model_call(_dimensions(), "expired-version")
    [expired_path] = store.create_and_export_package()
    store.record_model_call(_dimensions(), "current-version")
    [current_path] = store.create_and_export_package()
    store.record_model_call(_dimensions(), "pending-version")
    pending_package = store._create_package()
    assert pending_package is not None

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE counter_aggregates
            SET period_start = '2026-05-01'
            WHERE hermes_version = 'expired-version'
            """
        )
        connection.execute(
            """
            UPDATE package_outbox
            SET period_start = '2026-05-01T00:00:00Z',
                period_end = '2026-05-02T00:00:00Z',
                exported_at = '2026-05-02T00:00:00Z'
            WHERE package_id = ?
            """,
            (expired_path.stem,),
        )

    store._prune_expired_history(
        now=datetime(2026, 7, 23, tzinfo=timezone.utc)
    )

    assert not expired_path.exists()
    assert current_path.exists()
    assert not (outbox_directory / f"{pending_package['package_id']}.json").exists()
    with sqlite3.connect(database_path) as connection:
        outbox_rows = connection.execute(
            "SELECT package_id, exported_at FROM package_outbox ORDER BY package_id"
        ).fetchall()
        aggregate_versions = {
            row[0]
            for row in connection.execute(
                "SELECT hermes_version FROM counter_aggregates"
            ).fetchall()
        }
    assert {row[0] for row in outbox_rows} == {
        current_path.stem,
        pending_package["package_id"],
    }
    assert next(
        row[1] for row in outbox_rows if row[0] == pending_package["package_id"]
    ) is None
    assert aggregate_versions == {"current-version", "pending-version"}








def test_concurrent_package_builders_commit_one_delta(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    ready = threading.Barrier(2)

    def export() -> list[Path]:
        worker_store = SharedMetricsStore(database_path, outbox_directory)
        ready.wait(timeout=5)
        return worker_store.create_and_export_package()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(export) for _ in range(2)]
        for future in futures:
            future.result()

    with sqlite3.connect(database_path) as connection:
        [outbox_count] = connection.execute(
            "SELECT COUNT(*) FROM package_outbox"
        ).fetchone()
    [package_path] = list(outbox_directory.glob("*.json"))
    package = json.loads(package_path.read_text(encoding="utf-8"))

    assert outbox_count == 1
    assert package["metrics"][0]["value"] == 1
    assert store.counter_snapshot()[0]["packaged_value"] == 1






def test_cross_process_model_call_updates_are_transactional(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    context = mp.get_context("spawn")
    start_barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_record_model_calls_in_process,
            args=(str(database_path), str(outbox_directory), 10, start_barrier),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    restarted = SharedMetricsStore(database_path, outbox_directory)
    assert restarted.counter_snapshot()[0]["value"] == 20




@pytest.mark.skipif(os.name == "nt", reason="POSIX permission modes are unavailable")
def test_store_and_export_are_owner_only(tmp_path):
    database_path = tmp_path / "private-store" / "metrics.sqlite3"
    outbox_directory = tmp_path / "private-outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    [package_path] = store.create_and_export_package()

    assert stat.S_IMODE(database_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(outbox_directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(database_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(package_path.stat().st_mode) == 0o600
