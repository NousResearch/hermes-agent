"""Focused tests for the Rabbit shared-metrics durable store."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import stat
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from agent import relay_runtime
from rabbit_cli.observability import shared_metrics as shared_metrics_module
from rabbit_cli.observability.shared_metrics import SharedMetricsStore
from rabbit_cli.observability.shared_metrics_contract import (
    CLIENT_ACTIVE_METRIC,
    CLIENT_ARCHITECTURES,
    CLIENT_INSTALL_METHODS,
    CLIENT_OS_FAMILIES,
    COUNT_BUCKETS,
    DURATION_BUCKETS,
    EXECUTION_SURFACES,
    LEGACY_MODEL_CALL_METRIC,
    MODEL_IDENTIFIER_MAX_LENGTH,
    MODEL_ROUTE_METRIC,
    PROVIDER_IDENTIFIER_MAX_LENGTH,
    SKILL_LIFECYCLE_ACTIONS,
    SKILL_POST_PATCH_STATES,
    SKILL_PROVENANCES,
    SKILL_REUSE_STATES,
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
    client_active_counter,
    client_architecture,
    client_install_method,
    client_os_family,
    client_resource,
    model_call_dimensions,
    model_call_fields,
    skill_counter,
    skill_lifecycle_fields,
    skill_load_fields,
    task_counter,
    task_duration_counter,
    task_terminal_fields,
    tool_approval_counter,
    tool_approval_outcome,
    tool_call_dimensions,
    tool_latency_dimensions,
    tool_usage_dimensions,
    tool_category,
    tool_latency_bucket,
    tool_outcome,
    tool_retry_bucket,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "rabbit_cli"
    / "observability"
    / "schemas"
    / "rabbit.shared_metrics.v3.schema.json"
)
LEGACY_SCHEMA_PATH = SCHEMA_PATH.with_name("rabbit.shared_metrics.v1.schema.json")


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
    dimensions = schema["$defs"][kind]["properties"]["dimensions"]
    # The terminal counter lists its current v3 shape first, then the v2 shape it still drains.
    return dimensions["oneOf"][0] if "oneOf" in dimensions else dimensions


def _tool_dimension_schema(kind: str) -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return schema["$defs"][kind]["properties"]["dimensions"]


def _dimensions() -> dict[str, str]:
    return {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
    }


def _resource(
    rabbit_version: str = "test-version",
    *,
    os_family: str = "linux",
    architecture: str = "x86_64",
    install_method: str = "git",
) -> dict[str, str]:
    return {
        "architecture": architecture,
        "rabbit_version": rabbit_version,
        "install_method": install_method,
        "os_family": os_family,
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
        store.record_model_call(_dimensions(), _resource())


def _record_client_active_in_process(
    database_path: str,
    outbox_directory: str,
    start_barrier: Any,
) -> None:
    store = SharedMetricsStore(Path(database_path), Path(outbox_directory))
    start_barrier.wait()
    store.record_client_active(_resource())










def test_client_active_uses_a_transactional_rolling_24_hour_latch(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    store = SharedMetricsStore(database_path, outbox_directory)
    now = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(shared_metrics_module, "_utc_now", lambda: now)

    assert store.record_client_active(_resource())
    assert not store.record_client_active(_resource())

    now += timedelta(hours=23, minutes=59, seconds=59)
    assert not store.record_client_active(_resource())

    now += timedelta(seconds=1)
    assert store.record_client_active(_resource())

    active = [
        counter
        for counter in store.counter_snapshot()
        if counter["metric_name"] == CLIENT_ACTIVE_METRIC
    ]
    assert [counter["dimensions"] for counter in active] == [{}, {}]
    assert [counter["period_start"] for counter in active] == [
        "2026-07-22",
        "2026-07-23",
    ]
    assert [counter["value"] for counter in active] == [1, 1]


def test_client_active_recovers_from_an_invalid_latch_and_creates_identity(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "metrics.sqlite3"
    store = SharedMetricsStore(database_path, tmp_path / "outbox")
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO telemetry_state(key, value) VALUES (?, ?)",
            ("client_active_recorded_at", "invalid-timestamp"),
        )
    now = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(shared_metrics_module, "_utc_now", lambda: now)

    assert store.record_client_active(_resource())

    with sqlite3.connect(database_path) as connection:
        state = dict(
            connection.execute(
                "SELECT key, value FROM telemetry_state WHERE key != 'schema_version'"
            ).fetchall()
        )
    uuid.UUID(state["install_id"])
    assert state["client_active_recorded_at"] == "2026-07-22T10:00:00Z"


def test_client_active_rebases_a_future_latch_without_double_counting(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "metrics.sqlite3"
    store = SharedMetricsStore(database_path, tmp_path / "outbox")
    now = datetime(2026, 7, 22, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(shared_metrics_module, "_utc_now", lambda: now)

    assert store.record_client_active(_resource())
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE telemetry_state SET value = ? WHERE key = ?",
            ("2026-07-24T10:00:00Z", "client_active_recorded_at"),
        )

    assert not store.record_client_active(_resource())
    with sqlite3.connect(database_path) as connection:
        latch = connection.execute(
            "SELECT value FROM telemetry_state WHERE key = ?",
            ("client_active_recorded_at",),
        ).fetchone()[0]

    assert latch == "2026-07-22T10:00:00Z"
    [counter] = store.counter_snapshot()
    assert counter["metric_name"] == CLIENT_ACTIVE_METRIC
    assert counter["value"] == 1






def test_package_schema_matches_the_model_call_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = _package_dimension_schema()["properties"]

    assert schema["properties"]["schema_version"]["const"] == "rabbit.shared_metrics.v3"
    assert set(properties) == {"call_role", "error_class", "model", "outcome", "provider", "ttft_bucket"}
    # Rows counted before the v3 upgrade carry only model/provider and must still drain.
    assert set(_package_dimension_schema()["required"]) == {"model", "provider"}
    assert properties["model"]["maxLength"] == MODEL_IDENTIFIER_MAX_LENGTH
    assert properties["provider"]["maxLength"] == PROVIDER_IDENTIFIER_MAX_LENGTH
    assert "enum" not in properties["model"]
    assert "enum" not in properties["provider"]


def test_client_resource_classification_is_bounded():
    assert client_os_family("Darwin") == "macos"
    assert client_os_family("Windows") == "windows"
    assert client_architecture("AMD64") == "x86_64"
    assert client_architecture("aarch64") == "arm64"
    assert client_architecture("armv7l") == "arm"
    assert client_install_method("Homebrew") == "homebrew"
    assert client_install_method("nix") == "nixos"
    assert client_install_method("apt") == "apt"

    assert client_resource(
        "",
        os_name="privacy-os-canary",
        architecture="privacy-arch-canary",
        install_method="privacy-install-canary",
    ) == _resource(
        "unknown",
        os_family="unknown",
        architecture="unknown",
        install_method="unknown",
    )

def test_package_schema_matches_the_client_resource_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    resource = schema["properties"]["resource"]

    # Every v2 package records the complete bounded client resource.
    assert set(resource["required"]) == {
        "architecture",
        "rabbit_version",
        "install_method",
        "os_family",
    }
    assert set(resource["properties"]) == {
        "architecture",
        "rabbit_version",
        "install_method",
        "os_family",
    }
    assert set(resource["properties"]["os_family"]["enum"]) == CLIENT_OS_FAMILIES
    assert set(resource["properties"]["architecture"]["enum"]) == (CLIENT_ARCHITECTURES)
    assert set(resource["properties"]["install_method"]["enum"]) == (
        CLIENT_INSTALL_METHODS
    )


def test_client_active_mark_accepts_only_an_empty_allowlisted_payload():
    event = SimpleNamespace(
        kind="mark",
        category=None,
        category_profile=None,
        name="rabbit.client.active",
        scope_category=None,
        metadata={
            "rabbit.metrics.schema_version": "rabbit.metrics.event.v3",
        },
        data={},
    )

    assert client_active_counter(event) == (CLIENT_ACTIVE_METRIC, {})

    with_payload = deepcopy(event)
    with_payload.data = {"session_id": "privacy-canary"}
    assert client_active_counter(with_payload) is None

    wrong_schema = deepcopy(event)
    wrong_schema.metadata["rabbit.metrics.schema_version"] = "unknown"
    assert client_active_counter(wrong_schema) is None


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

def test_v1_package_schema_retains_the_legacy_model_contract():
    schema = json.loads(LEGACY_SCHEMA_PATH.read_text(encoding="utf-8"))
    model_counter = schema["$defs"]["model_call_counter"]

    assert schema["properties"]["schema_version"]["const"] == "rabbit.shared_metrics.v1"
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


def test_package_schema_matches_the_skill_contract():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    lifecycle = _tool_dimension_schema("skill_lifecycle_counter")["properties"]
    load = _tool_dimension_schema("skill_load_counter")["properties"]

    assert set(lifecycle["action"]["enum"]) == SKILL_LIFECYCLE_ACTIONS
    assert set(schema["$defs"]["skill_provenance"]["enum"]) == SKILL_PROVENANCES
    assert set(load["reuse_state"]["enum"]) == SKILL_REUSE_STATES
    assert set(load["post_patch_state"]["enum"]) == SKILL_POST_PATCH_STATES
    assert load["use_count_bucket"] == {"$ref": "#/$defs/count_bucket"}


@pytest.mark.parametrize(
    ("toolset", "expected"),
    [
        ("", "unknown"),
        ("file", "file"),
        ("browser-cdp", "browser"),
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
        ("smart_approve", "approved"),
        ("smart_deny", "denied"),
        ("timeout", "timed_out"),
        ("cancelled", "cancelled"),
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


def test_model_call_fields_report_terminal_model_and_shipped_provider():
    assert model_call_fields({
        "model": "fallback/model",
        "response_model": "NVIDIA/Nemotron-3-Ultra",
        "provider": "OpenRouter",
        "base_url": "https://private-endpoint.example/v1",
    }) == {
        "model": "nvidia/nemotron-3-ultra",
        "provider": "openrouter",
    }
    # A provider Rabbit does not ship is user-named (a custom endpoint key): neither it nor
    # the model id it serves leaves the machine.
    assert model_call_fields({
        "model": "ZAI/GLM-5.2",
        "provider": "Brev",
    }) == {
        "model": "custom",
        "provider": "custom",
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
            "rabbit.call_role": "auxiliary:compression",
        },
    )

    assert model_call_dimensions(event) == {
        "call_role": "auxiliary",
        "error_class": "none",
        "model": "accepted/model",
        "outcome": "success",
        "provider": "openrouter",
        "ttft_bucket": "unknown",
    }

    event.data.update({
        "model": "configured/model",
        "response_model": "malformed response model",
    })
    assert model_call_dimensions(event) == {
        "call_role": "auxiliary",
        "error_class": "none",
        "model": "configured/model",
        "outcome": "success",
        "provider": "openrouter",
        "ttft_bucket": "unknown",
    }

    event.metadata["rabbit.call_role"] = "primary"
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
        name="rabbit.tool_call",
        scope_category="end",
        metadata={"rabbit.metrics.schema_version": "rabbit.metrics.event.v3"},
        data={
            "approval_outcome": "approved",
            "error_class": "none",
            "latency_bucket": "250ms_to_500ms",
            "outcome": "success",
            "retry_count_bucket": "0",
            "tool_category": "terminal",
            "tool_name": "terminal",
        },
    )
    assert tool_call_dimensions(terminal) == {
        "approval_outcome": "approved", "outcome": "success", "tool_category": "terminal",
    }
    assert tool_latency_dimensions(terminal) == {
        "latency_bucket": "250ms_to_500ms", "retry_count_bucket": "0", "tool_category": "terminal",
    }
    assert tool_usage_dimensions(terminal) == {
        "error_class": "none", "outcome": "success", "tool_name": "terminal",
    }
    terminal.data["tool_name"] = "private-plugin-tool"
    assert tool_usage_dimensions(terminal) is None
    assert tool_call_dimensions(terminal) is None
    terminal.data["tool_name"] = "terminal"

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
        name="rabbit.tool_approval",
        scope_category=None,
        metadata={"rabbit.metrics.schema_version": "rabbit.metrics.event.v3"},
        data={"attribution": "unattributed", "outcome": "denied"},
    )
    assert tool_approval_counter(approval) == (
        "rabbit.tool_approval.count",
        approval.data,
    )
    approval.data["command"] = "must-not-pass"
    assert tool_approval_counter(approval) is None


def test_skill_subscriber_contract_accepts_only_bounded_marks():
    metadata = {"rabbit.metrics.schema_version": "rabbit.metrics.event.v3"}
    lifecycle = SimpleNamespace(
        kind="mark",
        category=None,
        category_profile=None,
        name="rabbit.skill.lifecycle",
        scope_category=None,
        metadata=metadata,
        data={"action": "patched", "provenance": "agent_created"},
    )
    assert skill_counter(lifecycle) == (
        "rabbit.skill.lifecycle.count",
        lifecycle.data,
    )

    load = SimpleNamespace(**{
        **lifecycle.__dict__,
        "name": "rabbit.skill.load",
        "data": {
            "post_patch_state": "reused_after_patch",
            "provenance": "agent_created",
            "reuse_state": "reused",
            "skill_name": "custom",
            "use_count_bucket": "3_to_5",
        },
    })
    assert skill_counter(load) == ("rabbit.skill.load.count", load.data)

    # Only bundled/optional skill names (public) may appear; a local name is refused.
    load.data["skill_name"] = "privacy-canary"
    assert skill_counter(load) is None
    load.data["skill_name"] = "custom"
    load.data["provenance"] = "private-repository"
    assert skill_counter(load) is None
    lifecycle.metadata["skill_name"] = "privacy-canary"
    assert skill_counter(lifecycle) is None

def test_skill_event_fields_are_bounded_and_reject_malformed_usage():
    assert skill_lifecycle_fields({
        "action": "patched",
        "provenance": "agent_created",
        "skill_name": "privacy-canary",
    }) == {"action": "patched", "provenance": "agent_created"}
    assert skill_lifecycle_fields({"action": "deleted"}) is None
    assert skill_load_fields({
        "provenance": "private-repository",
        "use_count": 2,
        "reused": True,
        "reuse_after_patch": False,
        "skill_name": "privacy-canary",
    }) == {
        "post_patch_state": "no_new_patch",
        "provenance": "unknown",
        "reuse_state": "reused",
        "skill_name": "custom",
        "use_count_bucket": "2",
    }
    assert skill_load_fields({
        "use_count": 1,
        "reused": False,
        "reuse_after_patch": False,
        "skill_name": "codex",
    }) == {
        "post_patch_state": "not_applicable",
        "provenance": "unknown",
        "reuse_state": "first_use",
        "skill_name": "codex",
        "use_count_bucket": "1",
    }
    assert (
        skill_load_fields({
            "use_count": 1,
            "reused": False,
            "reuse_after_patch": True,
        })
        is None
    )

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









def test_store_does_not_record_the_retired_model_metric(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    with pytest.raises(ValueError, match="Unsupported shared metric"):
        store.record_counter(
            LEGACY_MODEL_CALL_METRIC,
            _legacy_dimensions(),
            _resource(),
        )

    assert store.counter_snapshot() == []


def test_store_rejects_dimensions_outside_the_metric_contract(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        store.record_counter(
            MODEL_ROUTE_METRIC,
            {"prompt": "must-not-be-persisted"},
            _resource(),
        )

    assert store.counter_snapshot() == []

def test_store_rejects_client_resources_outside_the_contract(tmp_path):
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")

    with pytest.raises(ValueError, match="Unsupported shared-metrics client resource"):
        store.record_model_call(
            _dimensions(),
            {
                **_resource(),
                "architecture": "privacy-architecture-canary",
            },
        )

    assert store.counter_snapshot() == []

















def test_concurrent_model_call_updates_are_transactional(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    SharedMetricsStore(database_path, outbox_directory)

    def record_calls(count: int) -> int:
        store = SharedMetricsStore(database_path, outbox_directory)
        busy_calls = 0
        for _ in range(count):
            try:
                store.record_model_call(_dimensions(), _resource())
            except sqlite3.OperationalError as exc:
                assert exc.sqlite_errorcode == sqlite3.SQLITE_BUSY
                busy_calls += 1
        return busy_calls

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(record_calls, 10) for _ in range(2)]

    restarted = SharedMetricsStore(database_path, outbox_directory)
    # Check lossless increments without requiring contended writes to block.
    for _ in range(sum(future.result() for future in futures)):
        restarted.record_model_call(_dimensions(), _resource())
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


def test_a_write_the_busy_store_cannot_take_is_deferred_not_lost(tmp_path):
    """Another writer holding the store past the short busy timeout: the caller returns at once and the
    increment lands with the next write (or the exit drain), never raised or dropped."""
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    blocker = sqlite3.connect(tmp_path / "metrics.sqlite3")
    blocker.execute("BEGIN IMMEDIATE")
    try:
        store.record_model_call(_dimensions(), _resource())
    finally:
        blocker.rollback()
        blocker.close()
    assert store.counter_snapshot() == []
    store.record_model_call(_dimensions(), _resource())
    assert store.counter_snapshot()[0]["value"] == 2


def test_cross_process_client_active_attempts_record_one_install(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    context = mp.get_context("spawn")
    start_barrier = context.Barrier(2)
    processes = [
        context.Process(
            target=_record_client_active_in_process,
            args=(str(database_path), str(outbox_directory), start_barrier),
        )
        for _ in range(2)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    store = SharedMetricsStore(database_path, outbox_directory)
    [active] = store.counter_snapshot()
    assert active["metric_name"] == CLIENT_ACTIVE_METRIC
    assert active["dimensions"] == {}
    assert active["value"] == 1


def test_schema_initialization_waits_for_an_existing_writer(tmp_path):
    database_path = tmp_path / "metrics.sqlite3"
    outbox_directory = tmp_path / "outbox"
    database_path.touch()
    blocker = sqlite3.connect(database_path)
    blocker.execute("BEGIN IMMEDIATE")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            SharedMetricsStore,
            database_path,
            outbox_directory,
        )
        try:
            time.sleep(0.4)
            assert not future.done()
        finally:
            blocker.rollback()
            blocker.close()
        store = future.result(timeout=2)

    assert store.counter_snapshot() == []


