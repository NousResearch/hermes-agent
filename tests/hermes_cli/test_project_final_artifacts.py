from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_usage_ledger as ledger
from hermes_cli import project_final_artifacts as artifacts
from hermes_cli.project_finalization_contract import (
    create_project_finalization,
    ensure_project_finalization_schema,
    record_final_artifacts,
    register_project_member,
)
from hermes_cli.project_final_artifacts import (
    ProjectFinalizationSnapshot,
    aggregate_snapshot_usage,
    build_final_report,
    publish_project_final_artifacts,
    write_project_final_artifacts,
)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE run_usage (
            board TEXT NOT NULL, task_id TEXT NOT NULL, run_id INTEGER NOT NULL,
            call_kind TEXT NOT NULL, api_call_index INTEGER NOT NULL,
            provider TEXT NOT NULL, model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0, cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            reasoning_tokens INTEGER NOT NULL DEFAULT 0, elapsed_ms INTEGER NOT NULL DEFAULT 0,
            aux_input_tokens INTEGER, aux_output_tokens INTEGER,
            aux_cache_read_tokens INTEGER, aux_cache_write_tokens INTEGER,
            parent_task_id TEXT, profile TEXT, token_source TEXT NOT NULL,
            cost_usd REAL, cost_status TEXT, checker_result TEXT,
            repair_cycle INTEGER NOT NULL DEFAULT 0, accepted_result_tokens INTEGER,
            api_calls INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT '' ,
            PRIMARY KEY (board, task_id, run_id, call_kind, api_call_index)
        );
        CREATE TABLE run_usage_parents (
            board TEXT NOT NULL, task_id TEXT NOT NULL, run_id INTEGER NOT NULL,
            call_kind TEXT NOT NULL, api_call_index INTEGER NOT NULL, parent_task_id TEXT NOT NULL,
            PRIMARY KEY (board, task_id, run_id, call_kind, api_call_index, parent_task_id)
        );
        """
    )
    return conn


def _snapshot(tmp_path: Path) -> ProjectFinalizationSnapshot:
    return ProjectFinalizationSnapshot(
        board_id="board/one",
        root_task_id="t_root",
        generation=1,
        goal="Ship the project",
        title="Project finish",
        terminal_outcome="COMPLETE",
        checker_task_id="t_checker",
        checker_verdict="PASS",
        required_tasks=[{"task_id": "t_root", "membership_kind": "required"}],
        support_tasks=[{"task_id": "t_support", "membership_kind": "support"}],
        repair_tasks=[],
        runs=[{"task_id": "t_root", "run_id": 1, "status": "completed"}],
        commits=[{"sha": "abc", "subject": "finish"}],
        tests=[{"command": "scripts/run_tests.sh", "result": "passed"}],
        evidence=[{"path": "proof.txt", "sha256": "a" * 64}],
        what_done=["Built artifacts"],
        what_verified=["Focused tests"],
        blockers=[],
        next_step="Review the artifacts",
        delivery={"status": "not_sent"},
        cleanup={"status": "not_scheduled"},
        limitations=["No runtime activation"],
        usage={"usage_status": "unknown", "unknown": ["no usage events"]},
        created_at="2026-07-16T00:00:00Z",
    )


def test_publish_is_deterministic_and_uses_profile_safe_root(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = _db()
    ensure_project_finalization_schema(conn)
    create_project_finalization(
        conn,
        board_id="board/one",
        root_task_id="t_root",
        final_checker_task_id="t_checker",

    )
    snapshot = _snapshot(tmp_path)

    first = publish_project_final_artifacts(conn, snapshot)
    second = publish_project_final_artifacts(conn, snapshot)

    assert first.report_path == second.report_path
    assert first.report_sha256 == second.report_sha256
    assert Path(first.report_path).parent == home / "reports/project-finalization/board_one/t_root/generation-1"
    assert Path(first.report_path).read_bytes() == Path(second.report_path).read_bytes()
    assert hashlib.sha256(Path(first.report_path).read_bytes()).hexdigest() == first.report_sha256
    assert json.loads(Path(first.manifest_path).read_text()) ["manifest_schema_version"] == 1
    assert Path(first.report_path).read_text().count("# Goal") == 1


def test_usage_project_aggregation_deduplicates_multi_parent_events():
    conn = _db()
    ledger.record_run_usage(
        conn,
        board="b", task_id="t_root", run_id=1, call_kind="primary", api_call_index=0,
        provider="p", model="m", input_tokens=10, output_tokens=4,
        token_source="provider_authoritative", profile="builder",
    )
    ledger.record_parent(conn, board="b", task_id="t_root", run_id=1, call_kind="primary", api_call_index=0, parent_task_id="t_parent1")
    ledger.record_parent(conn, board="b", task_id="t_root", run_id=1, call_kind="primary", api_call_index=0, parent_task_id="t_parent2")

    result = ledger.aggregate_project_usage(conn, board="b", task_ids={"t_root", "t_root"})

    assert result["total_api_calls"] == 1
    assert result["total_input_tokens"] == 10
    group = result["groups"][0]
    assert {key: group[key] for key in (
        "role", "profile", "provider", "model", "call_kind", "record_count",
        "api_calls", "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_write_tokens", "reasoning_tokens",
    )} == {
        "role": "unknown", "profile": "builder", "provider": "p", "model": "m",
        "call_kind": "primary", "record_count": 1, "api_calls": 1,
        "input_tokens": 10, "output_tokens": 4, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "reasoning_tokens": 0,
    }


def test_missing_evidence_and_usage_conflict_are_rejected(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = _db()
    ensure_project_finalization_schema(conn)
    create_project_finalization(conn, board_id="b", root_task_id="r", final_checker_task_id="c")
    bad = ProjectFinalizationSnapshot(
        board_id="b", root_task_id="r", generation=1, terminal_outcome="BLOCKED",
        checker_task_id="c", checker_verdict="FAIL_TERMINAL", evidence=[], created_at="x",
    )
    with pytest.raises(ValueError, match="evidence"):
        publish_project_final_artifacts(conn, bad)

    snapshot = _snapshot(tmp_path)
    snapshot = ProjectFinalizationSnapshot(**{**snapshot.__dict__, "board_id": "b", "root_task_id": "r", "checker_task_id": "c"})
    first = publish_project_final_artifacts(conn, snapshot)
    conn.execute(
        "UPDATE project_finalizations SET usage_summary_json=? WHERE board_id='b' AND root_task_id='r' AND generation=1",
        ('{"usage_status":"different"}',),
    )
    with pytest.raises(ValueError, match="usage"):
        publish_project_final_artifacts(conn, snapshot)


def test_report_uses_the_exact_top_level_contract_headings(tmp_path):
    report = build_final_report(_snapshot(tmp_path)).decode("utf-8")

    expected = [
        "# Goal", "# Terminal outcome", "# What was done", "# What was verified",
        "# What failed", "# Current exact state", "# Remaining blockers",
        "# Next actionable step", "# Tasks and runs", "# Commits", "# Tests",
        "# Evidence", "# Usage", "# Telegram delivery", "# Cleanup schedule",
        "# Limitations",
    ]

    assert [line for line in report.splitlines() if line.startswith("# ")] == expected


def test_query_usage_orders_call_kind_as_part_of_event_identity():
    conn = _db()
    for call_kind in ("primary", "auxiliary"):
        ledger.record_run_usage(
            conn,
            board="b", task_id="t", run_id=1, call_kind=call_kind, api_call_index=0,
            provider="p", model="m", input_tokens=1, output_tokens=1,
            token_source="provider_authoritative",
        )

    assert [row["call_kind"] for row in ledger.query_usage(conn, board="b")] == [
        "auxiliary", "primary",
    ]


def test_snapshot_usage_includes_member_roles_and_drops_private_payloads(tmp_path):
    conn = _db()
    snapshot = ProjectFinalizationSnapshot(
        **{
            **_snapshot(tmp_path).__dict__,
            "repair_tasks": [{"task_id": "t_repair", "membership_kind": "repair"}],
            "usage": {
                "usage_status": "known",
                "prompt": "do not persist this",
                "response": "do not persist this either",
                "authorization": "Bearer do-not-persist",
                "provider_payload": {"secret": "do not persist"},
            },
        }
    )
    for task_id in ("t_root", "t_support", "t_repair", "t_checker"):
        ledger.record_run_usage(
            conn,
            board=snapshot.board_id, task_id=task_id, run_id=1,
            call_kind="primary", api_call_index=0, provider="p", model="m",
            input_tokens=1, output_tokens=1, token_source="provider_authoritative",
        )

    usage = aggregate_snapshot_usage(conn, snapshot)
    assert [(group["role"], group["record_count"]) for group in usage["groups"]] == [
        ("builder", 2), ("checker", 1), ("repair", 1),
    ]

    published = write_project_final_artifacts(snapshot, hermes_home=tmp_path / "home")
    rendered = Path(published.usage_summary_path).read_text(encoding="utf-8")
    for prohibited in ("prompt", "response", "authorization", "provider_payload", "do not persist"):
        assert prohibited not in rendered


def test_failed_atomic_publication_leaves_no_partial_final_set(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    root = tmp_path / "home" / "reports" / "project-finalization" / "board_one" / "t_root" / "generation-1"
    real_atomic_write = artifacts._atomic_write

    def fail_manifest(path, data):
        if path.name == "manifest.json":
            raise OSError("injected publication failure")
        real_atomic_write(path, data)

    monkeypatch.setattr(artifacts, "_atomic_write", fail_manifest)

    with pytest.raises(OSError, match="injected"):
        write_project_final_artifacts(snapshot, hermes_home=tmp_path / "home")

    assert not any((root / filename).exists() for filename in artifacts.ARTIFACT_FILENAMES)


@pytest.mark.parametrize("published_count", (1, 2))
def test_hard_crash_partial_matching_set_is_completed_on_restart(
    tmp_path, published_count
):
    snapshot = _snapshot(tmp_path)
    home = tmp_path / "home"
    report, manifest, usage, root = artifacts.build_project_artifacts(
        snapshot,
        hermes_home=home,
    )
    root.mkdir(parents=True)
    expected = dict(zip(artifacts.ARTIFACT_FILENAMES, (report, manifest, usage)))
    for filename in artifacts.ARTIFACT_FILENAMES[:published_count]:
        (root / filename).write_bytes(expected[filename])

    recovered = write_project_final_artifacts(snapshot, hermes_home=home)

    assert all(
        (root / filename).read_bytes() == data
        for filename, data in expected.items()
    )
    assert recovered.report_sha256 == hashlib.sha256(report).hexdigest()


def test_hard_crash_partial_conflicting_set_stays_fail_closed(tmp_path):
    snapshot = _snapshot(tmp_path)
    home = tmp_path / "home"
    _, _, _, root = artifacts.build_project_artifacts(snapshot, hermes_home=home)
    root.mkdir(parents=True)
    (root / "final-report.md").write_bytes(b"not the authoritative report")

    with pytest.raises(ValueError, match="conflicting immutable"):
        write_project_final_artifacts(snapshot, hermes_home=home)

    assert not (root / "manifest.json").exists()
    assert not (root / "usage-summary.json").exists()
