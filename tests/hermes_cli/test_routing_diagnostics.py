"""The operator can inspect a prospective decision without publishing state."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_cli_validate_and_explain_are_read_only(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    policy = {
        "schema_version": 1, "policy_id": "inspection", "revision": 1, "approval_ref": "test-only",
        "routes": [{
            "route_id": rid, "route_revision": 1, "provider": "custom", "model": rid,
            "endpoint": "http://127.0.0.1:1/v1", "maker": "test", "model_family": "fixture",
            "status": "approved", "allowed_roles": ["builder"], "capabilities": [],
            "verified_input_budget": capacity, "allowed_reasoning": ["high"],
            "qualifications": ["deep"], "assessment": "fixture", "evidence": ["fixture:only"],
        } for rid, capacity in [("small", 100), ("large", 200000)]],
        "rankings": {"builder": {"deep": ["small", "large"]}},
    }
    req = dict(schema_version=1, role="builder", execution_kind="diagnostic", execution_id="inspect",
               attempt_id="1", task_class="cross-component", required_capabilities=[],
               input_tokens=1000, reserve_tokens=8192, reasoning="high",
               provenance=dict(frozen_sha="", verified_by="test", complete=True, contributors=[]))
    policy_file, req_file = tmp_path / "policy.json", tmp_path / "requirements.json"
    policy_file.write_text(json.dumps(policy))
    req_file.write_text(json.dumps(req))
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(tmp_path))
    root = Path(__file__).resolve().parents[2]

    def run(action, *extra):
        proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", "routing", action,
                               str(policy_file), *extra, "--json"], cwd=root, env=env,
                              capture_output=True, text=True, timeout=30)
        return proc, json.loads(proc.stdout) if proc.returncode in (0, 1) else {}

    proc, result = run("validate")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert result["valid"] is True
    proc, decision = run("explain", str(req_file))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert decision["selected"]["route_id"] == "large"
    assert decision["rejections"]["small"] == ["input_too_large"]
    req["input_tokens"] = 300000
    req_file.write_text(json.dumps(req))
    proc, blocked = run("explain", str(req_file))
    assert proc.returncode == 1
    assert blocked["reason"] == "no_qualified_route"
    assert set(blocked["rejections"]) == {"small", "large"}
    assert not (home / "model_routing.db").exists()
    assert not list(home.rglob("kanban.db"))


def test_cli_reconcile_records_an_auditable_replay_authorization(tmp_path):
    from agent.managed_route_runtime import resolve_route
    from agent.model_selection_store import activate_policy, list_outcomes, publish_policy

    home = tmp_path / "home"
    home.mkdir()
    policy = {
        "schema_version": 1,
        "policy_id": "kanban-default",
        "revision": 1,
        "approval_ref": "test-only",
        "routes": [{
            "route_id": "builder-route", "route_revision": 1, "provider": "custom",
            "model": "fixture", "endpoint": "http://127.0.0.1:1/v1", "maker": "test",
            "model_family": "fixture", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["high"], "qualifications": ["deep"],
            "assessment": "fixture", "evidence": ["fixture:only"],
        }],
        "rankings": {"builder": {"deep": ["builder-route"]}},
    }
    record = publish_policy(home, policy, approval_ref="test-only")
    activate_policy(home, "kanban-default", record["revision"])
    resolution = resolve_route(home, "kanban-default", {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "reconcile-test", "attempt_id": "1",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "high",
        "provenance": {"frozen_sha": "", "verified_by": "test", "complete": True,
                       "contributors": []},
    }, now=1)
    env = dict(os.environ, HERMES_HOME=str(home), HOME=str(tmp_path))
    root = Path(__file__).resolve().parents[2]

    proc = subprocess.run([
        sys.executable, "-m", "hermes_cli.main", "kanban", "routing", "reconcile",
        resolution["receipt_id"], "--reason", "verified no external effect",
        "--approval-ref", "operator:test", "--json",
    ], cwd=root, env=env, capture_output=True, text=True, timeout=30)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout) == {
        "approval_ref": "operator:test",
        "reason": "verified no external effect",
        "receipt_id": resolution["receipt_id"],
    }
    outcome = list_outcomes(home, resolution["receipt_id"])[-1]
    assert outcome["kind"] == "routing_replay_authorized"
    assert outcome["payload"] == {
        "approval_ref": "operator:test",
        "reason": "verified no external effect",
    }
