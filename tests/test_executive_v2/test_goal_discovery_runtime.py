"""Canary tests for Executive Runtime Goal Discovery Runtime.

The runtime is intentionally read-only: it searches existing local artifacts and
emits a serializable goal discovery report before new work is created. It never
creates goals, knowledge, strategies, contracts, Kanban tasks, workers, provider
calls, or writes to GBrain/Obsidian.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "agent" / "executive" / "goal_discovery_runtime.py"
SCHEMA_PATH = REPO_ROOT / "agent" / "executive" / "schemas" / "goal_discovery_report_schema.json"


def test_goal_discovery_report_has_required_minimum_fields(tmp_path):
    from agent.executive.capability_discovery_runtime import CapabilityDiscoveryIndex, ObjectiveContext
    from agent.executive.goal_discovery_runtime import GoalDiscoveryIndex, discover_goal_related_work

    goal_dir = tmp_path / "reports" / "HERMES_EXECUTIVE_RUNTIME_GOAL_DISCOVERY_CANARY_20260703T000000Z"
    goal_dir.mkdir(parents=True)
    (goal_dir / "goal_discovery_report.json").write_text(
        json.dumps(
            {
                "matched_goals": [],
                "objective": "Goal Discovery Runtime canary",
                "result": "PASS",
            }
        ),
        encoding="utf-8",
    )
    (goal_dir / "canary_validation.md").write_text(
        "# Goal Discovery Runtime canary validation\nPASS compile tests schema hashes rollback\n",
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "reports" / "HERMES_EXECUTIVE_RUNTIME_CAPABILITY_DISCOVERY_CHECKPOINT_20260703T185206Z"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "CAPABILITY_DISCOVERY_CHECKPOINT.md").write_text(
        "# Capability Discovery checkpoint\ncheckpoint_pass official frozen source checkpoint for goal discovery\n",
        encoding="utf-8",
    )
    contract_dir = tmp_path / "contracts"
    contract_dir.mkdir()
    (contract_dir / "execution_contract_v1.json").write_text(
        '{"contract_id":"c1","hard_constraints":["search-only"],"rollback_strategy":"delete additive files"}',
        encoding="utf-8",
    )
    kanban_dir = tmp_path / "kanban_refs"
    kanban_dir.mkdir()
    (kanban_dir / "kanban_mapping.md").write_text(
        "Kanban reference only: TaskSpec board worker dispatch claim. Do not create tasks.",
        encoding="utf-8",
    )
    capability_dir = tmp_path / "capabilities"
    capability_dir.mkdir()
    (capability_dir / "capability_discovery_runtime.py").write_text(
        "# capability discovery runtime canary search only report checkpoint schema",
        encoding="utf-8",
    )

    context = ObjectiveContext(
        objective_text="""
        Implementar canary Goal Discovery Runtime. Debe descubrir objetivos,
        reportes, checkpoints, contratos, referencias Kanban y capacidades
        previas antes de crear trabajo nuevo. Solo buscar. No ejecutar.
        """,
        user_id="canary-user",
        constraints=("search-only", "no-create", "no-execute"),
        source_checkpoint=str(checkpoint_dir),
    )
    index = GoalDiscoveryIndex(
        capability_index=CapabilityDiscoveryIndex(
            workflow_roots=(tmp_path,),
            report_roots=(tmp_path / "reports",),
            checkpoint_roots=(tmp_path / "reports",),
            capability_roots=(capability_dir,),
        ),
        goal_roots=(tmp_path / "reports",),
        report_roots=(tmp_path / "reports",),
        checkpoint_roots=(tmp_path / "reports",),
        contract_roots=(contract_dir,),
        kanban_ref_roots=(kanban_dir,),
        capability_roots=(capability_dir,),
    )

    data = discover_goal_related_work(context, index=index).to_dict()

    assert set(data) == {
        "matched_goals",
        "matched_reports",
        "matched_checkpoints",
        "matched_contracts",
        "matched_kanban_refs",
        "matched_capabilities",
        "possible_duplicates",
        "related_work",
        "confidence",
        "reusable_prior_work",
        "missing_goal_context",
    }
    assert data["matched_goals"], "expected prior goal/objective artifact match"
    assert data["matched_reports"], "expected report match"
    assert data["matched_checkpoints"], "expected checkpoint match"
    assert data["matched_contracts"], "expected contract reference match"
    assert data["matched_kanban_refs"], "expected Kanban reference match"
    assert data["matched_capabilities"], "expected capability match"
    assert data["related_work"], "expected related work summary"
    assert data["reusable_prior_work"], "expected reusable prior work"
    assert 0.0 <= data["confidence"] <= 1.0


def test_goal_discovery_marks_missing_context_without_creating(tmp_path):
    from agent.executive.capability_discovery_runtime import CapabilityDiscoveryIndex, ObjectiveContext
    from agent.executive.goal_discovery_runtime import GoalDiscoveryIndex, discover_goal_related_work

    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    context = ObjectiveContext(
        objective_text="discover a nonexistent purple goal canary reference",
        user_id="canary-user",
    )
    empty_index = GoalDiscoveryIndex(
        capability_index=CapabilityDiscoveryIndex(
            workflow_roots=(tmp_path / "workflows",),
            report_roots=(tmp_path / "reports",),
            checkpoint_roots=(tmp_path / "checkpoints",),
            capability_roots=(tmp_path / "capabilities",),
        ),
        goal_roots=(tmp_path / "goals",),
        report_roots=(tmp_path / "reports",),
        checkpoint_roots=(tmp_path / "checkpoints",),
        contract_roots=(tmp_path / "contracts",),
        kanban_ref_roots=(tmp_path / "kanban_refs",),
        capability_roots=(tmp_path / "capabilities",),
    )

    data = discover_goal_related_work(context, index=empty_index).to_dict()

    after = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    assert after == before
    assert data["matched_goals"] == []
    assert data["matched_capabilities"] == []
    assert data["confidence"] == 0.0
    assert "goals" in data["missing_goal_context"]
    assert "reports" in data["missing_goal_context"]
    assert "checkpoints" in data["missing_goal_context"]


def test_goal_discovery_report_schema_validates_runtime_output(tmp_path):
    from agent.executive.capability_discovery_runtime import CapabilityDiscoveryIndex, ObjectiveContext
    from agent.executive.goal_discovery_runtime import GoalDiscoveryIndex, discover_goal_related_work

    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema is not installed")

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "goal.md").write_text(
        "Goal Discovery Runtime report checkpoint contract Kanban capability search-only canary",
        encoding="utf-8",
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    report = discover_goal_related_work(
        ObjectiveContext("goal discovery runtime report checkpoint contract Kanban capability", user_id="u"),
        index=GoalDiscoveryIndex(
            capability_index=CapabilityDiscoveryIndex(
                workflow_roots=(tmp_path / "reports",),
                report_roots=(tmp_path / "reports",),
                checkpoint_roots=(tmp_path / "reports",),
                capability_roots=(tmp_path / "reports",),
            ),
            goal_roots=(tmp_path / "reports",),
            report_roots=(tmp_path / "reports",),
            checkpoint_roots=(tmp_path / "reports",),
            contract_roots=(tmp_path / "reports",),
            kanban_ref_roots=(tmp_path / "reports",),
            capability_roots=(tmp_path / "reports",),
        ),
    ).to_dict()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(report, schema)


def test_goal_discovery_runtime_source_has_no_prohibited_runtime_hooks():
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                calls.append(func.attr)
            elif isinstance(func, ast.Name):
                calls.append(func.id)

    banned_import_fragments = [
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "openai",
        "anthropic",
        "gbrain",
        "obsidian",
        "notebooklm",
        "goalmanager_bridge",
        "worker_dispatch",
    ]
    found_imports = [m for m in imported_modules if any(b in m.lower() for b in banned_import_fragments)]
    assert not found_imports

    banned_call_names = {
        "write_text",
        "write_bytes",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "open",
        "system",
        "run",
        "Popen",
    }
    assert not (set(calls) & banned_call_names)


def test_goal_discovery_runtime_does_not_define_strategy_contract_apply_or_worker_apis():
    source = MODULE_PATH.read_text(encoding="utf-8").lower()
    prohibited_phrases = [
        "def build_strategy",
        "def create_execution_contract",
        "def apply_goal",
        "def create_kanban",
        "def spawn_worker",
        "def run_worker",
        "gbrain import",
        "obsidian",
        "notebooklm",
    ]
    found = [phrase for phrase in prohibited_phrases if phrase in source]
    assert not found
