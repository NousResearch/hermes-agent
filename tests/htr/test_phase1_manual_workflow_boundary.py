"""Tests for Task 17 — Phase 1 Boundary / End-to-End Manual Workflow Freeze."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import get_args

import pytest

import htr
from htr import contracts, events, io, paths, schemas
from htr.state import RUN_COMPLETED


def _load_task16_test_helpers():
    helper_path = Path(__file__).with_name("test_run_final_closure.py")
    spec = importlib.util.spec_from_file_location("task16_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TASK16 = _load_task16_test_helpers()

PHASE1_RECORD_JSON_FILES: tuple[str, ...] = tuple(
    f"{record_type}.json" for record_type in contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN
)

PHASE1_RECORD_PATH_BY_TYPE: dict[str, callable] = {
    "run_completion_record": contracts.run_completion_record_json_path,
    "run_review_record": contracts.run_review_record_json_path,
    "run_followup_plan_record": contracts.run_followup_plan_record_json_path,
    "run_execution_request_record": contracts.run_execution_request_record_json_path,
    "run_execution_result_record": contracts.run_execution_result_record_json_path,
    "run_execution_verification_record": contracts.run_execution_verification_record_json_path,
    "run_post_verification_followup_plan_record": (
        contracts.run_post_verification_followup_plan_record_json_path
    ),
    "run_post_verification_execution_request_record": (
        contracts.run_post_verification_execution_request_record_json_path
    ),
    "run_post_verification_execution_result_record": (
        contracts.run_post_verification_execution_result_record_json_path
    ),
    "run_post_verification_execution_verification_record": (
        contracts.run_post_verification_execution_verification_record_json_path
    ),
    "run_final_closure_record": contracts.run_final_closure_record_json_path,
}

PHASE1_RUN_LEVEL_WORKFLOW_EVENT_TYPES: tuple[str, ...] = (
    events.EVENT_TYPE_MANUAL_RUN_COMPLETED,
    events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
    events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED,
    events.EVENT_TYPE_RUN_EXECUTION_REQUESTED,
    events.EVENT_TYPE_RUN_EXECUTION_COMPLETED,
    events.EVENT_TYPE_RUN_EXECUTION_REJECTED,
    events.EVENT_TYPE_RUN_POST_VERIFICATION_FOLLOWUP_PLANNED,
    events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_REQUESTED,
    events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_RESULT_RECORDED,
    events.EVENT_TYPE_RUN_POST_VERIFICATION_EXECUTION_VERIFICATION_RECORDED,
    events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED,
)

FORBIDDEN_LIFECYCLE_IMPORTS = frozenset(
    {
        "runtime",
        "delegate_task",
        "scheduler",
        "queue",
        "database",
        "sqlite",
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "webbrowser",
        "os",
        "pytest",
        "playwright",
        "selenium",
        "heal",
        "deco",
    }
)

FOLLOWUP_OR_EXECUTION_APIS = frozenset(
    {
        "plan_run_followup",
        "plan_post_verification_followup",
        "request_run_execution",
        "request_post_verification_execution",
        "execute_run_execution_request",
        "verify_run_execution_result",
        "record_post_verification_execution_result",
        "record_post_verification_execution_verification",
        "make_run_followup_plan_record",
        "make_run_execution_request_record",
        "make_run_execution_result_record",
        "make_run_execution_verification_record",
        "make_run_post_verification_followup_plan_record",
        "make_run_post_verification_execution_request_record",
        "make_run_post_verification_execution_result_record",
        "make_run_post_verification_execution_verification_record",
    }
)


def _run_full_phase1_manual_workflow(tmp_path):
    chain = TASK16._run_with_post_verification_execution_verification(tmp_path)
    (
        run_id,
        task_ids,
        attempt_ids,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    ) = chain
    before = TASK16._snapshot(tmp_path, run_id, task_ids, attempt_ids)
    closure = TASK16._run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    )
    written = events.record_run_final_closure(
        tmp_path, run_id, closure, actor="human"
    )
    after = TASK16._snapshot(tmp_path, run_id, task_ids, attempt_ids)
    return {
        "run_id": run_id,
        "task_ids": task_ids,
        "attempt_ids": attempt_ids,
        "records": {
            "completion": completion,
            "review": review,
            "plan": plan,
            "request": request,
            "result": result,
            "verification": verification,
            "pvfp": pvfp,
            "pver": pver,
            "pve_result": pve_result,
            "pve_verification": pve_verification,
            "closure": written,
        },
        "before_closure": before,
        "after_closure": after,
    }


def _run_level_workflow_event_types(all_events):
    return [
        event["event_type"]
        for event in all_events
        if event.get("task_id") is None and event.get("attempt_id") is None
    ]


def _called_names_in_function(module_path: Path, function_name: str) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Name):
                        names.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        names.add(func.attr)
            break
    return names


# --- Phase 1 boundary constants ---


def test_phase1_record_chain_constant_matches_frozen_chain():
    assert contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN == (
        "run_completion_record",
        "run_review_record",
        "run_followup_plan_record",
        "run_execution_request_record",
        "run_execution_result_record",
        "run_execution_verification_record",
        "run_post_verification_followup_plan_record",
        "run_post_verification_execution_request_record",
        "run_post_verification_execution_result_record",
        "run_post_verification_execution_verification_record",
        "run_final_closure_record",
    )
    assert len(contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN) == 11


def test_phase1_terminal_record_and_event_constants():
    assert contracts.PHASE1_TERMINAL_RECORD_TYPE == "run_final_closure_record"
    assert contracts.PHASE1_TERMINAL_EVENT_TYPE == "run_final_closure_recorded"
    assert contracts.PHASE1_TERMINAL_EVENT_TYPE == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED


def test_phase1_boundary_status_is_constant_only():
    assert contracts.PHASE1_BOUNDARY_STATUS == "phase1_manual_workflow_frozen"
    assert contracts.PHASE1_BOUNDARY_STATUS not in events.EVENT_TYPES
    assert not hasattr(events, "EVENT_TYPE_PHASE1_BOUNDARY")


def test_phase1_constants_exported_from_package():
    assert htr.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN == (
        contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN
    )
    assert htr.PHASE1_TERMINAL_RECORD_TYPE == contracts.PHASE1_TERMINAL_RECORD_TYPE
    assert htr.PHASE1_TERMINAL_EVENT_TYPE == contracts.PHASE1_TERMINAL_EVENT_TYPE
    assert htr.PHASE1_BOUNDARY_STATUS == contracts.PHASE1_BOUNDARY_STATUS


# --- End-to-end manual workflow ---


def test_phase1_end_to_end_manual_workflow_writes_records_in_order(tmp_path):
    workflow = _run_full_phase1_manual_workflow(tmp_path)
    run_id = workflow["run_id"]
    run_root = paths.run_root(run_id, tmp_path)

    seen_mtime: list[float] = []
    for record_type in contracts.PHASE1_MANUAL_WORKFLOW_RECORD_CHAIN:
        record_path = PHASE1_RECORD_PATH_BY_TYPE[record_type](run_id, tmp_path)
        assert record_path.exists(), f"missing {record_type}"
        mtime = record_path.stat().st_mtime
        assert mtime >= (seen_mtime[-1] if seen_mtime else 0)
        seen_mtime.append(mtime)

    json_names = {path.name for path in run_root.glob("run_*_record.json")}
    assert json_names == set(PHASE1_RECORD_JSON_FILES)


def test_phase1_end_to_end_manual_workflow_appends_run_level_events_in_order(tmp_path):
    workflow = _run_full_phase1_manual_workflow(tmp_path)
    run_id = workflow["run_id"]
    all_events = events.read_task_events(run_id, base_dir=tmp_path)
    run_level_types = _run_level_workflow_event_types(all_events)

    expected_prefix = [
        events.EVENT_TYPE_MANUAL_RUN_COMPLETED,
        events.EVENT_TYPE_MANUAL_RUN_REVIEWED,
        events.EVENT_TYPE_MANUAL_RUN_FOLLOWUP_PLANNED,
        events.EVENT_TYPE_RUN_EXECUTION_REQUESTED,
        events.EVENT_TYPE_RUN_EXECUTION_COMPLETED,
    ]
    assert run_level_types[: len(expected_prefix)] == expected_prefix
    assert run_level_types[-1] == events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED
    for event_type in run_level_types:
        assert event_type in PHASE1_RUN_LEVEL_WORKFLOW_EVENT_TYPES


def test_phase1_end_to_end_closure_fingerprints_match_source_of_truth_records(tmp_path):
    workflow = _run_full_phase1_manual_workflow(tmp_path)
    records = workflow["records"]
    closure = records["closure"]

    assert closure["source_run_completion_fingerprint"] == contracts.run_completion_fingerprint(
        records["completion"]
    )
    assert closure["source_run_review_fingerprint"] == contracts.run_review_fingerprint(
        records["review"]
    )
    assert closure["source_run_followup_plan_fingerprint"] == (
        contracts.run_followup_plan_fingerprint(records["plan"])
    )
    assert closure["source_run_execution_request_fingerprint"] == (
        contracts.run_execution_request_fingerprint(records["request"])
    )
    assert closure["source_run_execution_result_fingerprint"] == (
        contracts.run_execution_result_fingerprint(records["result"])
    )
    assert closure["source_run_execution_verification_fingerprint"] == (
        contracts.run_execution_verification_fingerprint(records["verification"])
    )
    assert closure["source_post_verification_followup_plan_fingerprint"] == (
        contracts.run_post_verification_followup_plan_fingerprint(records["pvfp"])
    )
    assert closure["source_post_verification_execution_request_fingerprint"] == (
        contracts.run_post_verification_execution_request_fingerprint(records["pver"])
    )
    assert closure["source_post_verification_execution_result_fingerprint"] == (
        contracts.run_post_verification_execution_result_fingerprint(records["pve_result"])
    )
    assert closure["source_post_verification_execution_verification_fingerprint"] == (
        contracts.run_post_verification_execution_verification_fingerprint(
            records["pve_verification"]
        )
    )


def test_phase1_final_closure_preserves_prior_run_task_attempt_snapshots(tmp_path):
    """Closure itself must not mutate prior snapshots.

    This does not claim a global hard lock on later task/attempt APIs.
    Phase 1 treats run_final_closure_record.json as the manual-chain boundary.
    """
    workflow = _run_full_phase1_manual_workflow(tmp_path)
    before = workflow["before_closure"]
    after = workflow["after_closure"]

    assert after["final_closure_exists"] is True
    assert after["manifest"] == before["manifest"]
    assert after["manifest"]["status"] == RUN_COMPLETED
    assert after["completion"] == before["completion"]
    assert after["review"] == before["review"]
    assert after["followup"] == before["followup"]
    assert after["request"] == before["request"]
    assert after["result"] == before["result"]
    assert after["verification"] == before["verification"]
    assert after["post_verification_plan"] == before["post_verification_plan"]
    assert after["post_verification_execution_request"] == before[
        "post_verification_execution_request"
    ]
    assert after["post_verification_execution_result"] == before[
        "post_verification_execution_result"
    ]
    assert after["post_verification_execution_verification"] == before[
        "post_verification_execution_verification"
    ]
    assert after["task_statuses"] == before["task_statuses"]
    assert after["attempt_statuses"] == before["attempt_statuses"]


def test_phase1_workflow_does_not_create_boundary_record_file(tmp_path):
    workflow = _run_full_phase1_manual_workflow(tmp_path)
    run_id = workflow["run_id"]
    run_root = paths.run_root(run_id, tmp_path)
    assert not (run_root / "phase1_boundary_record.json").exists()
    assert all(
        path.name != "phase1_boundary_record.json" for path in run_root.rglob("*.json")
    )


# --- Boundary regression ---


def test_no_event_constant_equals_phase1_boundary_status():
    event_values = {
        value
        for name, value in vars(events).items()
        if name.startswith("EVENT_TYPE_") and isinstance(value, str)
    }
    assert contracts.PHASE1_BOUNDARY_STATUS not in event_values
    assert contracts.PHASE1_BOUNDARY_STATUS not in events.EVENT_TYPES


def test_no_phase1_boundary_schema_exists():
    schema_names = set(get_args(schemas.SchemaName))
    assert "phase1_boundary_record" not in schema_names


def test_no_phase1_boundary_public_apis_exist():
    assert not hasattr(htr, "record_phase1_boundary")
    assert not hasattr(htr, "make_phase1_boundary_record")
    assert not hasattr(contracts, "record_phase1_boundary")
    assert not hasattr(contracts, "make_phase1_boundary_record")


def test_event_types_frozen_at_final_closure_with_no_task17_additions():
    assert events.EVENT_TYPE_RUN_FINAL_CLOSURE_RECORDED in events.EVENT_TYPES
    assert "run_final_closure_recorded" in events.EVENT_TYPES
    task17_event_names = [
        name
        for name in dir(events)
        if name.startswith("EVENT_TYPE_") and "PHASE1" in name
    ]
    assert task17_event_names == []


def test_record_run_final_closure_does_not_call_followup_or_execution_apis():
    repo_root = Path(__file__).resolve().parents[2]
    called = _called_names_in_function(
        repo_root / "htr" / "events.py", "record_run_final_closure"
    )
    assert FOLLOWUP_OR_EXECUTION_APIS.isdisjoint(called)


def test_record_run_final_closure_only_writes_closure_record_and_event(tmp_path):
    workflow = TASK16._run_with_post_verification_execution_verification(tmp_path)
    (
        run_id,
        _task_ids,
        _attempt_ids,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    ) = workflow
    before_files = {
        path.name
        for path in paths.run_root(run_id, tmp_path).glob("*")
        if path.is_file()
    }
    before_events = len(events.read_task_events(run_id, base_dir=tmp_path))
    closure = TASK16._run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    after_files = {
        path.name
        for path in paths.run_root(run_id, tmp_path).glob("*")
        if path.is_file()
    }
    assert after_files - before_files == {"run_final_closure_record.json"}
    assert len(events.read_task_events(run_id, base_dir=tmp_path)) == before_events + 1


def test_record_run_final_closure_does_not_mutate_run_manifest_status(tmp_path):
    workflow = TASK16._run_with_post_verification_execution_verification(tmp_path)
    (
        run_id,
        _task_ids,
        _attempt_ids,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    ) = workflow
    manifest_path = paths.run_manifest_path(run_id, tmp_path)
    original_manifest = io.read_json(manifest_path)
    closure = TASK16._run_final_closure_record(
        run_id,
        completion,
        review,
        plan,
        request,
        result,
        verification,
        pvfp,
        pver,
        pve_result,
        pve_verification,
    )
    events.record_run_final_closure(tmp_path, run_id, closure, actor="human")
    assert io.read_json(manifest_path) == original_manifest
    assert io.read_json(manifest_path)["status"] == RUN_COMPLETED


# --- Import / call boundary ---


@pytest.mark.parametrize(
    "relative_path",
    [
        "htr/contracts.py",
        "htr/events.py",
        "htr/schemas.py",
        "htr/__init__.py",
    ],
)
def test_phase1_lifecycle_modules_do_not_import_forbidden_modules(relative_path):
    repo_root = Path(__file__).resolve().parents[2]
    tree = ast.parse((repo_root / relative_path).read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert FORBIDDEN_LIFECYCLE_IMPORTS.isdisjoint(imported), relative_path


def test_record_run_final_closure_docstring_does_not_schedule_task17():
    doc = inspect.getdoc(events.record_run_final_closure) or ""
    lowered = doc.lower()
    assert "task 17" not in lowered
    assert "followup loop" in lowered or "follow-up" in lowered or "followup" in lowered
