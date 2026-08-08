"""Canary tests for Executive Runtime Capability Discovery Runtime.

The runtime is intentionally read-only: it searches existing local indexes and
artifacts, emits a serializable capability report, and never creates goals,
knowledge, strategy, execution contracts, Kanban tasks, workers, provider calls,
or writes to GBrain/Obsidian.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "agent" / "executive" / "capability_discovery_runtime.py"
SCHEMA_PATH = REPO_ROOT / "agent" / "executive" / "schemas" / "capability_report_schema.json"


def test_capability_discovery_report_has_required_minimum_fields(tmp_path):
    from agent.executive.capability_discovery_runtime import (
        CapabilityDiscoveryIndex,
        ObjectiveContext,
        discover_capabilities,
    )

    skills = tmp_path / "skills" / "software-development" / "test-driven-development"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        """---
name: test-driven-development
description: TDD workflow for tests before code
---
# Test-driven development
Use tests, compile checks, schema validation, hashes, and rollback evidence.
""",
        encoding="utf-8",
    )
    reports = tmp_path / "reports" / "HERMES_EXECUTIVE_RUNTIME_OBJECTIVE_COMPLETENESS_CHECKPOINT_20260703T183126Z"
    reports.mkdir(parents=True)
    (reports / "OBJECTIVE_COMPLETENESS_CHECKPOINT.md").write_text(
        """# Objective Completeness Analyzer checkpoint
Result: HERMES_EXECUTIVE_RUNTIME_OBJECTIVE_COMPLETENESS_CHECKPOINT_PASS
Contains capability discovery, reports, manifest, hashes, rollback references.
""",
        encoding="utf-8",
    )
    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "capability_report_schema.json").write_text(
        '{"title":"Capability Report Schema"}',
        encoding="utf-8",
    )

    context = ObjectiveContext(
        objective_text="""
        Implementar canary Capability Discovery Runtime. Debe descubrir
        capacidades existentes, skills, workflows, reports, checkpoints,
        policies, templates, schemas, tests, hashes y rollback antes de crear
        trabajo nuevo. No Goal Discovery. No Knowledge Discovery. No Strategy
        Builder. No Execution Contract. No Kanban. No Workers.
        """,
        user_id="canary-user",
        constraints=("search-only", "no-create", "no-execute"),
        source_checkpoint=str(reports),
    )
    index = CapabilityDiscoveryIndex(
        skill_roots=(tmp_path / "skills",),
        workflow_roots=(tmp_path,),
        policy_roots=(tmp_path,),
        template_roots=(tmp_path,),
        report_roots=(tmp_path / "reports",),
        checkpoint_roots=(tmp_path / "reports",),
        capability_roots=(tmp_path,),
    )

    data = discover_capabilities(context, index=index).to_dict()

    assert set(data) == {
        "matched_skills",
        "matched_workflows",
        "matched_roles",
        "matched_policies",
        "matched_templates",
        "matched_reports",
        "matched_checkpoints",
        "matched_capabilities",
        "confidence",
        "reusable_assets",
        "missing_capabilities",
    }
    assert data["matched_skills"], "expected existing skill match"
    assert data["matched_reports"], "expected report match"
    assert data["matched_checkpoints"], "expected checkpoint match"
    assert data["matched_templates"], "expected schema/template match"
    assert data["matched_capabilities"], "expected combined capability matches"
    assert 0.0 <= data["confidence"] <= 1.0
    assert any(asset["path"].endswith("SKILL.md") for asset in data["reusable_assets"])


def test_capability_discovery_marks_missing_categories_without_creating(tmp_path):
    from agent.executive.capability_discovery_runtime import (
        CapabilityDiscoveryIndex,
        ObjectiveContext,
        discover_capabilities,
    )

    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    context = ObjectiveContext(
        objective_text="discover a nonexistent quantum banana runtime capability",
        user_id="canary-user",
    )
    empty_index = CapabilityDiscoveryIndex(
        skill_roots=(tmp_path / "skills",),
        workflow_roots=(tmp_path / "workflows",),
        role_roots=(tmp_path / "roles",),
        policy_roots=(tmp_path / "policies",),
        template_roots=(tmp_path / "templates",),
        report_roots=(tmp_path / "reports",),
        checkpoint_roots=(tmp_path / "checkpoints",),
        capability_roots=(tmp_path / "capabilities",),
    )

    data = discover_capabilities(context, index=empty_index).to_dict()

    after = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
    assert after == before
    assert data["matched_capabilities"] == []
    assert data["confidence"] == 0.0
    assert "skills" in data["missing_capabilities"]
    assert "workflows" in data["missing_capabilities"]


def test_capability_report_schema_validates_runtime_output(tmp_path):
    from agent.executive.capability_discovery_runtime import (
        CapabilityDiscoveryIndex,
        ObjectiveContext,
        discover_capabilities,
    )

    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        pytest.skip("jsonschema is not installed")

    (tmp_path / "skills" / "x").mkdir(parents=True)
    (tmp_path / "skills" / "x" / "SKILL.md").write_text(
        "name: x\ndescription: capability discovery search only canary\n",
        encoding="utf-8",
    )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    report = discover_capabilities(
        ObjectiveContext("capability discovery search only canary", user_id="u"),
        index=CapabilityDiscoveryIndex(skill_roots=(tmp_path / "skills",)),
    ).to_dict()

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(report, schema)


def test_capability_discovery_runtime_source_has_no_prohibited_runtime_hooks():
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
        "kanban",
        "worker_dispatch",
        "goalmanager",
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


def test_capability_discovery_runtime_does_not_define_goal_strategy_contract_or_worker_apis():
    source = MODULE_PATH.read_text(encoding="utf-8").lower()
    prohibited_phrases = [
        "def discover_goals",
        "def build_strategy",
        "def create_execution_contract",
        "def create_kanban",
        "def spawn_worker",
        "def run_worker",
        "gbrain import",
        "obsidian",
        "notebooklm",
    ]
    found = [phrase for phrase in prohibited_phrases if phrase in source]
    assert not found
