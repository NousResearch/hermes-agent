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
from agent import relay_runtime
from hermes_cli.observability import shared_metrics as shared_metrics_module
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_contract import (
    COUNT_BUCKETS,
    DURATION_BUCKETS,
    EXECUTION_SURFACES,
    LEGACY_MODEL_CALL_METRIC,
    MODEL_CALL_PROFILE_MODEL,
    MODEL_IDENTIFIER_MAX_LENGTH,
    MODEL_ROUTE_METRIC,
    PROVIDER_IDENTIFIER_MAX_LENGTH,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    TASK_END_REASONS,
    TASK_ENTRYPOINTS,
    TASK_OUTCOMES,
    TASK_TERMINATIONS,
    TOOL_APPROVAL_ATTRIBUTIONS,
    TOOL_APPROVAL_OUTCOMES,
    TOOL_CATEGORIES,
    TOOL_LATENCY_BUCKETS,
    TOOL_OUTCOMES,
    TOOL_RETRY_BUCKETS,
    count_bucket,
    duration_bucket,
    execution_surface,
    model_call_dimensions,
    model_call_fields,
    task_counter,
    task_start_fields,
    task_terminal_fields,
    task_terminal_state,
    tool_approval_counter,
    tool_approval_outcome,
    tool_call_dimensions,
    tool_category,
    tool_latency_bucket,
    tool_outcome,
    tool_retry_bucket,
    tool_terminal_fields,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "hermes_cli"
    / "observability"
    / "schemas"
    / "hermes.shared_metrics.v2.schema.json"
)
LEGACY_SCHEMA_PATH = SCHEMA_PATH.with_name("hermes.shared_metrics.v1.schema.json")


def _schema_validator(path: Path = SCHEMA_PATH):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.FormatChecker(),
    )


def _package_dimension_schema() -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"]["model_route_counter"]["properties"]["dimensions"]


def _task_dimension_schema(kind: str) -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"][kind]["properties"]["dimensions"]


def _tool_dimension_schema(kind: str) -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"][kind]["properties"]["dimensions"]


def _dimensions() -> dict[str, str]:
    return {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
    }


def _legacy_dimensions() -> dict[str, str]:
    return {
        "call_role": "primary",
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
    assert first_package["schema_version"] == "hermes.shared_metrics.v2"
    assert first_package["resource"] == {"hermes_version": "test-version"}
    assert first_package["metrics"] == [
        {
            "name": MODEL_ROUTE_METRIC,
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


def test_v2_package_preserves_pending_v1_model_counters(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    period_start = shared_metrics_module._utc_now().date().isoformat()
    legacy_dimensions_json = json.dumps(
        _legacy_dimensions(),
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO counter_aggregates(
                period_start,
                metric_name,
                hermes_version,
                dimensions_json,
                value,
                packaged_value
            ) VALUES (?, ?, ?, ?, 3, 0)
            """,
            (
                period_start,
                LEGACY_MODEL_CALL_METRIC,
                "test-version",
                legacy_dimensions_json,
            ),
        )
    store.record_model_call(_dimensions(), "test-version")

    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    _schema_validator().validate(package)

    assert package["schema_version"] == "hermes.shared_metrics.v2"
    assert package["metrics"] == [
        {
            "name": LEGACY_MODEL_CALL_METRIC,
            "type": "counter",
            "dimensions": _legacy_dimensions(),
            "value": 3,
        },
        {
            "name": MODEL_ROUTE_METRIC,
            "type": "counter",
            "dimensions": _dimensions(),
            "value": 1,
        },
    ]
    assert all(
        row["value"] == row["packaged_value"] for row in store.counter_snapshot()
    )


def test_v1_outbox_package_exports_unchanged_after_upgrade(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    package_id = str(uuid.uuid4())
    payload = {
        "schema_version": "hermes.shared_metrics.v1",
        "package_id": package_id,
        "install_id": str(uuid.uuid4()),
        "period_start": "2026-07-28T00:00:00Z",
        "period_end": "2026-07-29T00:00:00Z",
        "generated_at": "2026-07-29T01:00:00Z",
        "resource": {"hermes_version": "legacy-version"},
        "metrics": [
            {
                "name": LEGACY_MODEL_CALL_METRIC,
                "type": "counter",
                "dimensions": _legacy_dimensions(),
                "value": 2,
            }
        ],
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    _schema_validator(LEGACY_SCHEMA_PATH).validate(payload)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO package_outbox(
                package_id,
                period_start,
                period_end,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                package_id,
                payload["period_start"],
                payload["period_end"],
                payload_json,
                payload["generated_at"],
            ),
        )

    assert store.create_and_export_package() == [
        outbox_directory / f"{package_id}.json"
    ]
    exported = json.loads(
        (outbox_directory / f"{package_id}.json").read_text(encoding="utf-8")
    )
    assert exported == payload
    with sqlite3.connect(database_path) as connection:
        [persisted_payload_json] = connection.execute(
            "SELECT payload_json FROM package_outbox WHERE package_id = ?",
            (package_id,),
        ).fetchone()
    assert persisted_payload_json == payload_json


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
        "metric_name": MODEL_ROUTE_METRIC,
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
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = _package_dimension_schema()["properties"]

    assert schema["properties"]["schema_version"]["const"] == "hermes.shared_metrics.v2"
    assert set(properties) == {"model", "provider"}
    assert properties["model"]["maxLength"] == MODEL_IDENTIFIER_MAX_LENGTH
    assert properties["provider"]["maxLength"] == PROVIDER_IDENTIFIER_MAX_LENGTH
    assert "enum" not in properties["model"]
    assert "enum" not in properties["provider"]


def test_v1_package_schema_retains_the_legacy_model_contract():
    schema = json.loads(LEGACY_SCHEMA_PATH.read_text(encoding="utf-8"))
    model_counter = schema["$defs"]["model_call_counter"]

    assert schema["properties"]["schema_version"]["const"] == "hermes.shared_metrics.v1"
    assert model_counter["properties"]["name"]["const"] == LEGACY_MODEL_CALL_METRIC
    assert set(model_counter["properties"]["dimensions"]["properties"]) == {
        "call_role",
        "locality",
        "model_family",
        "outcome",
        "provider_family",
    }


def test_package_schema_matches_the_tool_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    tool = _tool_dimension_schema("tool_call_counter")["properties"]
    approval = _tool_dimension_schema("tool_approval_counter")["properties"]

    assert set(tool["tool_category"]["enum"]) == TOOL_CATEGORIES
    assert set(tool["outcome"]["enum"]) == TOOL_OUTCOMES
    assert set(tool["approval_outcome"]["enum"]) == TOOL_APPROVAL_OUTCOMES
    assert tool["latency_bucket"] == {"$ref": "#/$defs/tool_latency_bucket"}
    assert tool["retry_count_bucket"] == {"$ref": "#/$defs/tool_retry_bucket"}
    assert set(schema["$defs"]["tool_latency_bucket"]["enum"]) == (
        TOOL_LATENCY_BUCKETS
    )
    assert set(schema["$defs"]["tool_retry_bucket"]["enum"]) == TOOL_RETRY_BUCKETS
    assert set(approval["attribution"]["enum"]) == TOOL_APPROVAL_ATTRIBUTIONS
    assert set(approval["outcome"]["enum"]) == (
        TOOL_APPROVAL_OUTCOMES - {"not_required"}
    )


@pytest.mark.parametrize(
    ("toolset", "expected"),
    [
        ("", "unknown"),
        ("file", "file"),
        ("terminal", "terminal"),
        ("code_execution", "code_execution"),
        ("delegation", "delegation"),
        ("skills", "skill"),
        ("browser-cdp", "browser"),
        ("image_gen", "media"),
        ("homeassistant", "home_automation"),
        ("kanban", "planning"),
        ("project", "project"),
        ("discord", "communication"),
        ("feishu_doc", "communication"),
        ("mcp-github", "mcp"),
        ("private_plugin", "other"),
    ],
)
def test_tool_category_uses_bounded_runtime_toolsets(toolset, expected):
    assert tool_category({"toolset": toolset}) == expected


def test_tool_category_does_not_classify_raw_tool_names():
    assert tool_category({"tool_name": "read_file"}) == "unknown"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("ok", "success"),
        ("error", "failed"),
        ("blocked", "blocked"),
        ("cancelled", "cancelled"),
        ("timeout", "timed_out"),
        ("private", "unknown"),
        (None, "unknown"),
    ],
)
def test_tool_outcome_is_bounded(status, expected):
    assert tool_outcome({"status": status}) == expected


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("once", "approved"),
        ("session", "approved"),
        ("always", "approved"),
        ("smart_approve", "approved"),
        ("deny", "denied"),
        ("smart_deny", "denied"),
        ("timeout", "timed_out"),
        (None, "unknown"),
    ],
)
def test_tool_approval_outcome_is_bounded(choice, expected):
    assert tool_approval_outcome({"choice": choice}) == expected


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (0, "lt_100ms"),
        (100, "100ms_to_250ms"),
        (250, "250ms_to_500ms"),
        (500, "500ms_to_1s"),
        (1_000, "1s_to_2s"),
        (2_000, "2s_to_5s"),
        (5_000, "5s_to_10s"),
        (10_000, "10s_to_30s"),
        (30_000, "gte_30s"),
        (-1, "unknown"),
        (True, "unknown"),
        ("100", "unknown"),
    ],
)
def test_tool_latency_bucket_is_bounded(duration_ms, expected):
    assert tool_latency_bucket(duration_ms) == expected


@pytest.mark.parametrize(
    ("retry_count", "expected"),
    [
        (0, "0"),
        (1, "1"),
        (2, "2"),
        (3, "3_to_5"),
        (6, "6_to_10"),
        (11, "gte_11"),
        (None, "unknown"),
        (-1, "unknown"),
        (True, "unknown"),
    ],
)
def test_tool_retry_bucket_requires_an_explicit_non_negative_count(
    retry_count,
    expected,
):
    assert tool_retry_bucket(retry_count) == expected


def test_model_call_fields_report_terminal_model_and_provider_without_a_catalog():
    assert model_call_fields({
        "model": "fallback/model",
        "response_model": "NVIDIA/Nemotron-3-Ultra",
        "provider": "OpenRouter",
        "base_url": "https://private-endpoint.example/v1",
    }) == {
        "model": "nvidia/nemotron-3-ultra",
        "provider": "openrouter",
    }
    assert model_call_fields({
        "model": "ZAI/GLM-5.2",
        "provider": "Brev",
    }) == {
        "model": "zai/glm-5.2",
        "provider": "brev",
    }


def test_auxiliary_logical_scope_projects_one_normalized_terminal_route():
    event = SimpleNamespace(
        kind="scope",
        category="function",
        name=relay_runtime.LOGICAL_LLM_SCOPE,
        scope_category="end",
        category_profile=None,
        data={
            "model": "Accepted/Model",
            "outcome": "success",
            "provider": "OpenRouter",
        },
        metadata={
            relay_runtime.RUNTIME_SCHEMA_KEY: relay_runtime.RUNTIME_SCHEMA_VERSION,
            relay_runtime.RUNTIME_INSTANCE_KEY: "runtime-1",
            "hermes.call_role": "auxiliary:compression",
        },
    )

    assert model_call_dimensions(event) == {
        "model": "accepted/model",
        "provider": "openrouter",
    }

    event.data.update({
        "model": "configured/model",
        "response_model": "malformed response model",
    })
    assert model_call_dimensions(event) == {
        "model": "configured/model",
        "provider": "openrouter",
    }

    event.metadata["hermes.call_role"] = "primary"
    assert model_call_dimensions(event) is None


@pytest.mark.parametrize(
    "response_model",
    [
        "contains a space",
        "x" * (MODEL_IDENTIFIER_MAX_LENGTH + 1),
    ],
)
def test_model_call_fields_fall_back_when_response_model_is_invalid(response_model):
    assert model_call_fields({
        "model": "nvidia/nemotron-3-ultra",
        "response_model": response_model,
        "provider": "openrouter",
    }) == {
        "model": "nvidia/nemotron-3-ultra",
        "provider": "openrouter",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", ""),
        ("model", "contains a space"),
        ("model", "contains\ncontrol"),
        ("model", "_" + "private"),
        ("model", "x" * (MODEL_IDENTIFIER_MAX_LENGTH + 1)),
        ("model", object()),
        ("provider", ""),
        ("provider", "private provider"),
        ("provider", "x" * (PROVIDER_IDENTIFIER_MAX_LENGTH + 1)),
        ("provider", object()),
    ],
)
def test_model_call_fields_collapse_malformed_identifiers(field, value):
    event = {"model": "nvidia/nemotron-3-ultra", "provider": "openrouter"}
    event[field] = value

    assert model_call_fields(event)[field] == "unknown"


def test_tool_subscriber_contract_accepts_only_bounded_events():
    terminal = SimpleNamespace(
        kind="scope",
        category="tool",
        category_profile={},
        name="hermes.tool_call",
        scope_category="end",
        metadata={"hermes.metrics.schema_version": "hermes.metrics.event.v2"},
        data={
            "approval_outcome": "approved",
            "latency_bucket": "250ms_to_500ms",
            "outcome": "success",
            "retry_count_bucket": "0",
            "tool_category": "terminal",
        },
    )
    assert tool_call_dimensions(terminal) == terminal.data

    terminal.data["result"] = "must-not-pass"
    assert tool_call_dimensions(terminal) is None
    terminal.data.pop("result")
    terminal.data["tool_category"] = "private-tool-name"
    assert tool_call_dimensions(terminal) is None
    terminal.data["tool_category"] = "terminal"
    terminal.category_profile["tool_name"] = "must-not-pass"
    assert tool_call_dimensions(terminal) is None

    approval = SimpleNamespace(
        kind="mark",
        category=None,
        category_profile=None,
        name="hermes.tool_approval",
        scope_category=None,
        metadata={"hermes.metrics.schema_version": "hermes.metrics.event.v2"},
        data={"attribution": "unattributed", "outcome": "denied"},
    )
    assert tool_approval_counter(approval) == (
        "hermes.tool_approval.count",
        approval.data,
    )
    approval.data["command"] = "must-not-pass"
    assert tool_approval_counter(approval) is None


def test_store_does_not_record_the_retired_model_metric(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    with pytest.raises(ValueError, match="Unsupported shared metric"):
        store.record_counter(
            LEGACY_MODEL_CALL_METRIC,
            _legacy_dimensions(),
            "test-version",
        )

    assert store.counter_snapshot() == []
































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
