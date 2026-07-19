"""Focused tests for the Hermes shared-metrics durable store."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sqlite3
import stat
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
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


def test_package_schema_matches_the_model_call_contract():
    properties = _package_dimension_schema()["properties"]

    assert properties["call_role"] == {"const": PRIMARY_MODEL_CALL_ROLE}
    assert set(properties["locality"]["enum"]) == MODEL_LOCALITIES
    assert set(properties["model_family"]["enum"]) == MODEL_FAMILIES
    assert set(properties["outcome"]["enum"]) == MODEL_OUTCOMES
    assert set(properties["provider_family"]["enum"]) == PROVIDER_FAMILIES


def test_package_schema_matches_the_task_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    start = _task_dimension_schema("task_started_counter")["properties"]
    terminal = _task_dimension_schema("task_finished_counter")["properties"]

    assert set(schema["$defs"]["execution_surface"]["enum"]) == EXECUTION_SURFACES
    assert set(schema["$defs"]["task_entrypoint"]["enum"]) == TASK_ENTRYPOINTS
    assert set(schema["$defs"]["duration_bucket"]["enum"]) == DURATION_BUCKETS
    assert set(schema["$defs"]["count_bucket"]["enum"]) == COUNT_BUCKETS
    assert start["entrypoint"] == {"$ref": "#/$defs/task_entrypoint"}
    assert set(terminal["end_reason"]["enum"]) == TASK_END_REASONS
    assert set(terminal["outcome"]["enum"]) == TASK_OUTCOMES
    assert set(terminal["termination"]["enum"]) == TASK_TERMINATIONS


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("", "unknown"),
        ("not-a-hermes-provider", "unknown"),
        ("custom", "custom"),
        ("custom-local", "custom"),
        ("custom:private-endpoint", "custom"),
        ("lmstudio", "local"),
        ("lm_studio", "local"),
        ("ollama", "local"),
        ("nous", "aggregator"),
        ("openrouter", "aggregator"),
        ("kilo", "aggregator"),
        ("copilot-acp", "aggregator"),
        ("huggingface", "aggregator"),
        ("novita", "aggregator"),
        ("anthropic", "direct"),
        ("google", "direct"),
        ("openai-api", "direct"),
    ],
)
def test_provider_family_uses_bounded_product_categories(provider, expected):
    assert provider_family({"provider": provider}) == expected


def test_provider_family_does_not_resolve_live_provider_metadata(monkeypatch):
    def fail_live_lookup(_provider):
        raise AssertionError("telemetry must not refresh provider metadata")

    monkeypatch.setattr("hermes_cli.providers.get_provider", fail_live_lookup)
    assert provider_family({"provider": "anthropic"}) == "direct"


def test_locality_uses_the_endpoint_only_for_local_classification():
    kwargs = {
        "provider": "custom",
        "base_url": "http://127.0.0.1:11434/v1",
    }

    assert provider_family(kwargs) == "custom"
    assert model_locality(kwargs) == "local"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google/gemma-3", "gemma"),
        ("x-ai/grok-4", "grok"),
        ("minimax/minimax-m2.5", "minimax"),
        ("xiaomi/mimo-v2", "mimo"),
        ("amazon/nova-pro", "nova"),
        ("stepfun/step-3.5", "step"),
        ("arcee-ai/trinity-large", "trinity"),
    ],
)
def test_model_family_covers_families_evidenced_by_the_hermes_catalog(model, expected):
    assert model_family({"model": model}) == expected


@pytest.mark.parametrize(
    "model",
    [
        "private-gptish-model",
        "innovation-private",
        "mimosa-private",
        "stepstone-private",
        "supernova-private",
    ],
)
def test_model_family_requires_identifier_boundaries(model):
    assert model_family({"model": model}) == "unknown"


def test_model_family_accepts_only_allowlisted_declared_metadata():
    assert model_family({"model": "private", "model_family": "qwen"}) == "qwen"
    assert model_family({"model": "private", "model_family": "private"}) == "unknown"


def test_model_family_prefers_the_provider_reported_terminal_model():
    assert (
        model_family({"model": "gpt-5", "response_model": "claude-sonnet"}) == "claude"
    )


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("", "unknown"),
        ("cli", "cli"),
        ("api_server", "api"),
        ("cron", "scheduled_task"),
        ("whatsapp_cloud", "gateway"),
        ("private-surface", "other"),
    ],
)
def test_execution_surface_uses_the_hermes_platform_registry(platform, expected):
    assert execution_surface({"platform": platform}) == expected


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("cli", "interactive"),
        ("tui", "interactive"),
        ("whatsapp_cloud", "gateway_message"),
        ("cron", "scheduled_task"),
        ("api_server", "api"),
        ("private-surface", "other"),
    ],
)
def test_task_start_fields_use_bounded_surface_and_entrypoint(platform, expected):
    fields = task_start_fields({"platform": platform})

    assert fields["entrypoint"] == expected
    assert fields["execution_surface"] in EXECUTION_SURFACES


def test_task_start_fields_identify_delegated_work_without_exporting_parent_id():
    fields = task_start_fields({
        "platform": "cli",
        "parent_session_id": "private-parent-session",
    })

    assert fields == {
        "entrypoint": "delegated",
        "execution_surface": "cli",
    }
    assert "private-parent-session" not in json.dumps(fields)


@pytest.mark.parametrize(
    ("duration_ms", "expected"),
    [
        (0, "lt_1s"),
        (999, "lt_1s"),
        (1_000, "1s_to_5s"),
        (5_000, "5s_to_30s"),
        (30_000, "30s_to_2m"),
        (120_000, "2m_to_10m"),
        (600_000, "gte_10m"),
    ],
)
def test_duration_bucket_boundaries(duration_ms, expected):
    assert duration_bucket(duration_ms) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (1, "1"),
        (2, "2"),
        (3, "3_to_5"),
        (6, "6_to_10"),
        (11, "gte_11"),
    ],
)
def test_count_bucket_boundaries(count, expected):
    assert count_bucket(count) == expected


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            {"completed": True, "turn_exit_reason": "text_response(stop)"},
            ("success", "completed", "none"),
        ),
        (
            {"failed": True, "turn_exit_reason": "all_retries_exhausted_no_response"},
            ("failed", "failed", "none"),
        ),
        (
            {"interrupted": True, "turn_exit_reason": "interrupted_by_user"},
            ("cancelled", "user_cancelled", "user_cancelled"),
        ),
        (
            {"turn_exit_reason": "budget_exhausted"},
            ("failed", "iteration_limit", "system_aborted"),
        ),
        (
            {"turn_exit_reason": "guardrail_halt"},
            ("failed", "guardrail_blocked", "system_aborted"),
        ),
        (
            {"failed": True, "turn_exit_reason": "provider_timeout"},
            ("timed_out", "timed_out", "timed_out"),
        ),
        (
            {"failed": True, "turn_exit_reason": "approval_denied"},
            ("failed", "approval_denied", "none"),
        ),
    ],
)
def test_task_terminal_state_is_bounded(event, expected):
    assert task_terminal_state(event) == expected


def test_model_outcome_fails_closed_to_a_bounded_value():
    assert model_call_outcome({"outcome": "private"}) == "failed"


def test_unlisted_model_collapses_to_a_bounded_value():
    assert model_family({"model": "private-model-name"}) == "unknown"


def test_subscriber_contract_rejects_unknown_fields_and_dimension_values():
    event = SimpleNamespace(
        kind="scope",
        category="llm",
        category_profile={"model_name": "gpt"},
        name="hermes.model_call",
        scope_category="end",
        metadata={"hermes.metrics.schema_version": "hermes.metrics.event.v1"},
        data={
            "call_role": "primary",
            "locality": "remote",
            "model_family": "gpt",
            "outcome": "success",
            "provider_family": "direct",
        },
    )

    assert model_call_dimensions(event) == {
        "call_role": "primary",
        "locality": "remote",
        "model_family": "gpt",
        "outcome": "success",
        "provider_family": "direct",
    }
    event.category_profile["model_name"] = "private-model-name"
    assert model_call_dimensions(event) is None
    event.category_profile["model_name"] = "gpt"
    event.data["prompt"] = "must-not-pass"
    assert model_call_dimensions(event) is None
    event.data.pop("prompt")
    event.metadata["prompt"] = "must-not-pass"
    assert model_call_dimensions(event) is None
    event.metadata.pop("prompt")
    event.category_profile["private"] = "must-not-pass"
    assert model_call_dimensions(event) is None
    event.category_profile.pop("private")
    event.category = "function"
    assert model_call_dimensions(event) is None


def test_task_subscriber_contract_accepts_only_bounded_scope_events():
    start = SimpleNamespace(
        kind="scope",
        category="function",
        category_profile=None,
        name="hermes.task_run",
        scope_category="start",
        metadata={"hermes.metrics.schema_version": "hermes.metrics.event.v1"},
        data={"entrypoint": "interactive", "execution_surface": "cli"},
    )
    assert task_counter(start) == (
        "hermes.task_run.started",
        {"entrypoint": "interactive", "execution_surface": "cli"},
    )

    terminal_fields = task_terminal_fields(
        {
            "platform": "cli",
            "completed": True,
            "turn_exit_reason": "text_response(stop)",
        },
        duration_ms=6_000,
        model_call_count=2,
        tool_call_count=3,
        retry_count=1,
    )
    end = SimpleNamespace(**{
        **start.__dict__,
        "scope_category": "end",
        "data": terminal_fields,
    })
    assert task_counter(end) == (
        "hermes.task_run.finished",
        terminal_fields,
    )

    end.data["task_id"] = "must-not-pass"
    assert task_counter(end) is None
    end.data.pop("task_id")
    end.data["outcome"] = "private"
    assert task_counter(end) is None
    end.data["outcome"] = "success"
    end.metadata["prompt"] = "must-not-pass"
    assert task_counter(end) is None


def test_store_rejects_an_unsupported_schema_version(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE telemetry_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO telemetry_state(key, value) VALUES ('schema_version', '999')"
        )

    with pytest.raises(RuntimeError, match="Unsupported shared-metrics store schema"):
        SharedMetricsStore(database_path, tmp_path / "outbox")

    with sqlite3.connect(database_path) as connection:
        [schema_version] = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = 'schema_version'"
        ).fetchone()
    assert schema_version == "999"


def test_pending_metrics_keep_the_version_recorded_at_event_time(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_model_call(_dimensions(), "version-a")
    store.record_model_call(_dimensions(), "version-b")

    packages = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in store.create_and_export_package()
    ]

    assert {package["resource"]["hermes_version"] for package in packages} == {
        "version-a",
        "version-b",
    }
    assert all(package["metrics"][0]["value"] == 1 for package in packages)


def test_store_exports_task_started_and_terminal_counters(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_counter(
        "hermes.task_run.started",
        {"entrypoint": "interactive", "execution_surface": "cli"},
        "test-version",
    )
    terminal = task_terminal_fields(
        {
            "platform": "cli",
            "completed": True,
            "turn_exit_reason": "text_response(stop)",
        },
        duration_ms=2_000,
        model_call_count=1,
        tool_call_count=2,
        retry_count=0,
    )
    store.record_counter("hermes.task_run.finished", terminal, "test-version")

    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    _schema_validator().validate(package)

    assert {metric["name"] for metric in package["metrics"]} == {
        "hermes.task_run.finished",
        "hermes.task_run.started",
    }


def test_package_schema_rejects_unknown_fields(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    store.record_model_call(_dimensions(), "test-version")
    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    invalid_package = deepcopy(package)
    invalid_package["prompt"] = "must-not-be-accepted"

    jsonschema = pytest.importorskip("jsonschema")
    with pytest.raises(jsonschema.ValidationError):
        _schema_validator().validate(invalid_package)


def test_store_rejects_dimensions_outside_the_metric_contract(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        store.record_counter(
            "hermes.model_call.count",
            {"prompt": "must-not-be-persisted"},
            "test-version",
        )

    assert store.counter_snapshot() == []


def test_package_builder_rejects_tampered_dimensions(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE counter_aggregates SET dimensions_json = ?",
            (json.dumps({"prompt": "must-not-be-exported"}),),
        )

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        store.create_and_export_package()

    assert list(outbox_directory.glob("*.json")) == []


def test_pending_package_retry_reuses_the_same_package_and_file(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    [package_path] = store.create_and_export_package()
    original_payload = package_path.read_bytes()

    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE package_outbox SET exported_at = NULL")

    restarted = SharedMetricsStore(database_path, outbox_directory)
    assert restarted.create_and_export_package() == [package_path]
    assert package_path.read_bytes() == original_payload
    assert list(outbox_directory.glob("*.json")) == [package_path]


def test_file_export_failure_retries_committed_outbox_without_duplicate_delta(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated atomic export failure")

    module_globals = SharedMetricsStore._export_pending_packages.__globals__
    original_write = module_globals["atomic_json_write"]
    monkeypatch.setitem(module_globals, "atomic_json_write", fail_write)
    with pytest.raises(OSError, match="simulated atomic export failure"):
        store.create_and_export_package()

    with sqlite3.connect(database_path) as connection:
        package_id, exported_at = connection.execute(
            "SELECT package_id, exported_at FROM package_outbox"
        ).fetchone()
    assert exported_at is None
    assert store.counter_snapshot()[0]["packaged_value"] == 1
    assert list(outbox_directory.glob("*.json")) == []

    monkeypatch.setitem(module_globals, "atomic_json_write", original_write)
    assert store.create_and_export_package() == [
        outbox_directory / f"{package_id}.json"
    ]
    assert len(list(outbox_directory.glob("*.json"))) == 1
    assert store.create_and_export_package() == []


def test_package_export_does_not_chase_concurrent_updates(tmp_path, monkeypatch):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    store.record_model_call(_dimensions(), "test-version")
    original_create = store._create_package
    create_calls = 0

    def create_and_record_another():
        nonlocal create_calls
        create_calls += 1
        package = original_create()
        if create_calls == 1:
            store.record_model_call(_dimensions(), "test-version")
        return package

    monkeypatch.setattr(store, "_create_package", create_and_record_another)
    first_paths = store.create_and_export_package()

    assert create_calls == 1
    assert len(first_paths) == 1
    [counter] = store.counter_snapshot()
    assert counter["metric_name"] == "hermes.model_call.count"
    assert counter["dimensions"] == _dimensions()
    assert counter["value"] == 2
    assert counter["packaged_value"] == 1

    second_paths = store.create_and_export_package()
    assert len(second_paths) == 1
    assert store.counter_snapshot()[0]["packaged_value"] == 2


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


def test_concurrent_model_call_updates_are_transactional(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    SharedMetricsStore(database_path, outbox_directory)

    def record_calls(count: int) -> None:
        store = SharedMetricsStore(database_path, outbox_directory)
        for _ in range(count):
            store.record_model_call(_dimensions(), "test-version")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(record_calls, 10) for _ in range(2)]
        for future in futures:
            future.result()

    restarted = SharedMetricsStore(database_path, outbox_directory)
    assert restarted.counter_snapshot()[0]["value"] == 20


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
