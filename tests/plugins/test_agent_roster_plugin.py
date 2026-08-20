from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "agent-roster"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_core():
    return _load_module("agent_roster_core_test", PLUGIN_DIR / "roster_core.py")


def _load_hooks():
    return _load_module("agent_roster_hooks_test", PLUGIN_DIR / "__init__.py")


def _load_api():
    return _load_module("agent_roster_plugin_api_test", PLUGIN_DIR / "dashboard" / "plugin_api.py")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _role(role_id: str, **extra) -> dict:
    role = {
        "id": role_id,
        "display_name": role_id.title(),
        "status": "active",
        "category": "test",
        "mission": f"Mission for {role_id}",
        "allowed_task_types": [role_id.upper(), "RESEARCH"],
        "allowed_boards": ["default", "geld-ideen-pipeline"],
        "forbidden": [],
        "output_contract": {"required_sections": ["Summary", "Evidence"]},
    }
    role.update(extra)
    return role


def _seed_profiles(root: Path) -> None:
    _write_yaml(
        root / "config.yaml",
        {
            "plugins": {"enabled": ["agent-roster"]},
            "dashboard": {
                "agent_roster": {
                    "enabled": True,
                    "strict_mode": "block",
                    "role_config_key": "profile_role",
                }
            },
            "profile_role": _role(
                "default",
                display_name="Clank",
                allowed_task_types=["IDEA", "BUILD", "LAUNCH", "QA"],
                forbidden=[],
            ),
        },
    )
    _write_yaml(
        root / "profiles" / "researcher" / "config.yaml",
        {"profile_role": _role("researcher", forbidden=["code_changes", "deployment", "git_push"])},
    )
    _write_yaml(
        root / "profiles" / "reviewer" / "config.yaml",
        {"profile_role": _role("reviewer", allowed_task_types=["QA", "REVIEW"], forbidden=["deployment"])},
    )
    _write_yaml(root / "profiles" / "no-role" / "config.yaml", {"model": {"provider": "test"}})


def test_roster_reads_profile_roles_and_surfaces_missing_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _seed_profiles(tmp_path)

    core = _load_core()
    data = core.build_roster()

    names = {p["name"] for p in data["profiles"]}
    assert {"default", "researcher", "reviewer", "no-role"}.issubset(names)
    researcher = next(p for p in data["profiles"] if p["name"] == "researcher")
    assert researcher["role"]["id"] == "researcher"
    assert "code_changes" in researcher["role"]["forbidden"]
    assert data["summary"]["profiles_missing_role_metadata"] == 1
    assert any(v["code"] == "profile_missing_role_metadata" and v["profile"] == "no-role" for v in data["violations"])


def test_roster_flags_kanban_assignment_and_quality_gate_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _seed_profiles(tmp_path)

    conn = kb.connect(board="default")
    try:
        unknown_id = kb.create_task(conn, title="[BUILD] Ship production feature", body="Build it fast", assignee="ghost")
        no_prd_id = kb.create_task(conn, title="[BUILD] Add checkout", body="Goal: sell faster", assignee="default")
        review_id = kb.create_task(conn, title="[QA] Review checkout", body="Review build output", assignee="reviewer", parents=[no_prd_id])
    finally:
        conn.close()

    core = _load_core()
    data = core.build_roster()
    by_code = {(v["code"], v.get("task_id")) for v in data["violations"]}

    assert ("task_assignee_profile_missing", unknown_id) in by_code
    assert ("prd_lite_missing", no_prd_id) in by_code
    assert ("reviewer_gate_missing", no_prd_id) not in by_code
    assert any(stage["key"] == "build" and stage["task_count"] >= 2 for stage in data["pipeline"]["stages"])
    assert any(stage["key"] == "qa" and stage["task_count"] >= 1 for stage in data["pipeline"]["stages"])
    assert review_id


def test_reviewer_assigned_qa_task_does_not_require_second_reviewer_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _seed_profiles(tmp_path)

    conn = kb.connect(board="default")
    try:
        qa_id = kb.create_task(conn, title="[QA] Review final output", body="Evidence: check the build", assignee="reviewer")
    finally:
        conn.close()

    core = _load_core()
    data = core.build_roster()
    assert ("reviewer_gate_missing", qa_id) not in {(v["code"], v.get("task_id")) for v in data["violations"]}


def test_pre_llm_call_injects_current_role_and_task_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "researcher"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "researcher")
    _seed_profiles(tmp_path)
    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="[RESEARCH] Compare tools", body="Find sources", assignee="researcher")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    hooks = _load_hooks()
    result = hooks.on_pre_llm_call(session_id="s1", user_message="work")

    assert isinstance(result, dict)
    ctx = result["context"]
    assert "Agent Roster Role Context" in ctx
    assert "Profile: researcher" in ctx
    assert "Forbidden actions: code_changes, deployment, git_push" in ctx
    assert f"Task: {task_id} — [RESEARCH] Compare tools" in ctx


def test_pre_tool_call_blocks_forbidden_actions_in_strict_mode_and_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "researcher"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "researcher")
    _seed_profiles(tmp_path)

    hooks = _load_hooks()
    result = hooks.on_pre_tool_call(
        tool_name="write_file",
        args={"path": "app.py", "content": "print('x')"},
        session_id="s1",
        tool_call_id="tc1",
    )

    assert result["action"] == "block"
    assert "code_changes" in result["message"]
    audit_path = tmp_path / "logs" / "agent-roster-audit.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "tool_policy_block"
    assert rows[-1]["profile"] == "researcher"
    assert rows[-1]["tool_name"] == "write_file"


def test_freeform_german_forbidden_rules_map_to_canonical_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "researcher"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "researcher")
    _seed_profiles(tmp_path)
    _write_yaml(
        tmp_path / "profiles" / "researcher" / "config.yaml",
        {"profile_role": _role("researcher", forbidden=["Code ändern", "Konfiguration ändern", "Git push"])},
    )

    hooks = _load_hooks()
    result = hooks.on_pre_tool_call(tool_name="write_file", args={"path": "app.py"}, session_id="s1")

    assert result["action"] == "block"
    assert "code_changes" in result["message"]


def test_kanban_complete_blocks_build_task_without_prd_or_reviewer_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    _seed_profiles(tmp_path)
    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="[BUILD] Ship checkout", body="Goal: ship", assignee="default")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    hooks = _load_hooks()
    result = hooks.on_pre_tool_call(tool_name="kanban_complete", args={"task_id": task_id}, session_id="s1")

    assert result["action"] == "block"
    assert "PRD-lite fields" in result["message"]


def test_kanban_complete_allows_build_task_with_prd_and_reviewer_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    _seed_profiles(tmp_path)
    complete_body = """
    Goal: ship checkout
    Non-goal: subscriptions
    Target user: buyer
    Acceptance criteria: payment succeeds
    Success metric: conversion
    Risks: payment failure
    Dependencies: provider
    Definition of done: reviewer approved
    """
    conn = kb.connect(board="default")
    try:
        task_id = kb.create_task(conn, title="[BUILD] Ship checkout", body=complete_body, assignee="default")
        kb.create_task(conn, title="[QA] Review checkout", body="Review evidence", assignee="reviewer", parents=[task_id])
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)

    hooks = _load_hooks()
    result = hooks.on_pre_tool_call(tool_name="kanban_complete", args={"task_id": task_id}, session_id="s1")

    assert result is None


def test_warn_strict_mode_alias_normalizes_to_audit_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "researcher"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "researcher")
    _seed_profiles(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["dashboard"]["agent_roster"]["strict_mode"] = "warn"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    core = _load_core()
    hooks = _load_hooks()
    result = hooks.on_pre_tool_call(tool_name="write_file", args={"path": "app.py"}, session_id="s1")

    assert core.roster_config()["strict_mode"] == "audit"
    assert result is None
    audit_path = tmp_path / "logs" / "agent-roster-audit.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "tool_policy_audit"
    assert rows[-1]["mode"] == "audit"


def test_patch_mode_config_files_are_classified_as_config_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _seed_profiles(tmp_path)

    core = _load_core()
    actions = core.classify_tool_actions(
        "patch",
        {
            "mode": "patch",
            "patch": "*** Begin Patch\n*** Update File: AGENTS.md\n@@\n-old\n+new\n*** End Patch",
        },
    )

    assert "code_changes" in actions
    assert "config_changes" in actions


def test_post_tool_audit_does_not_persist_tool_result_excerpts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    _seed_profiles(tmp_path)

    hooks = _load_hooks()
    sensitive_result = "private credential value: super-sensitive-value"
    hooks.on_post_tool_call(tool_name="read_file", args={"path": ".env"}, result=sensitive_result, session_id="s1")

    audit_path = tmp_path / "logs" / "agent-roster-audit.jsonl"
    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["event"] == "tool_call"
    assert row["result_redacted"] is True
    assert "result_excerpt" not in row
    assert "super-sensitive-value" not in json.dumps(row)


def test_dashboard_api_routes_return_roster_envelopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    _seed_profiles(tmp_path)

    api = _load_api()
    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/agent-roster")
    client = TestClient(app)

    roster = client.get("/api/plugins/agent-roster/roster").json()
    violations = client.get("/api/plugins/agent-roster/violations").json()
    pipeline = client.get("/api/plugins/agent-roster/pipeline").json()

    assert roster["summary"]["profile_count"] >= 4
    assert "profiles" in roster and "violations" in roster
    assert violations["count"] == len(violations["violations"])
    assert pipeline["count"] == len(pipeline["stages"])


def test_dashboard_assets_register_agent_roster_plugin():
    manifest = json.loads((PLUGIN_DIR / "dashboard" / "manifest.json").read_text(encoding="utf-8"))
    js = (PLUGIN_DIR / "dashboard" / "dist" / "index.js").read_text(encoding="utf-8")

    assert manifest["name"] == "agent-roster"
    assert manifest["tab"]["path"] == "/agent-roster"
    assert manifest["api"] == "plugin_api.py"
    assert "SDK.fetchJSON" in js
    assert "__HERMES_PLUGINS__.register(\"agent-roster\"" in js
