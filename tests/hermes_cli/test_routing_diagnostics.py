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


def test_receipt_cli_distinguishes_selection_from_observed_outcomes(tmp_path, monkeypatch):
    from agent.managed_route_runtime import resolve_route
    from agent.model_selection_store import activate_policy, append_outcome, get_receipt, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    policy = {
        "schema_version": 1, "policy_id": "inspection", "revision": 1,
        "approval_ref": "test-only",
        "routes": [{
            "route_id": "selected", "route_revision": 1, "provider": "custom",
            "model": "selected-model", "endpoint": "http://127.0.0.1:1/v1",
            "maker": "fixture", "model_family": "fixture", "status": "approved",
            "allowed_roles": ["builder"], "capabilities": [],
            "verified_input_budget": 200000, "allowed_reasoning": ["high"],
            "qualifications": ["deep"], "assessment": "fixture", "evidence": ["fixture:only"],
        }],
        "rankings": {"builder": {"deep": ["selected"]}},
    }
    publish_policy(home, policy, approval_ref="test-only")
    activate_policy(home, "inspection", 1)
    kb.init_db()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="inspect receipt", assignee="builder")
    resolution = resolve_route(home, "inspection", {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": task_id, "attempt_id": "1", "task_class": "cross-component",
        "required_capabilities": [], "input_tokens": 1000, "reserve_tokens": 8192,
        "reasoning": "high",
        "provenance": {"frozen_sha": "", "verified_by": "test", "complete": True,
                       "contributors": []},
    }, now=1)
    receipt_id = resolution["receipt_id"]
    original = get_receipt(home, receipt_id)
    with kbc.connect_closing() as conn:
        conn.execute("UPDATE tasks SET routing_receipt_id=? WHERE id=?", (receipt_id, task_id))
        conn.commit()
    env = dict(os.environ, HOME=str(tmp_path))

    def run(action, identity, as_json):
        proc = subprocess.run([
            sys.executable, "-m", "hermes_cli.main", "kanban", "routing", action, identity,
            *(["--json"] if as_json else []),
        ], env=env, capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc.stdout

    for action, identity in [("receipt", receipt_id), ("receipt-for-task", task_id)]:
        result = json.loads(run(action, identity, True))
        assert all(
            event["kind"] == "routing_selected"
            for event in result["outcomes"]
        )
        assert "observed: unknown" in run(action, identity, False)

    append_outcome(home, receipt_id, "routing_started", {
        "actual_provider": "custom", "actual_model": "observed-model",
    })
    for action, identity in [("receipt", receipt_id), ("receipt-for-task", task_id)]:
        result = json.loads(run(action, identity, True))
        assert result["outcomes"][-1]["payload"]["actual_model"] == "observed-model"
        text = run(action, identity, False)
        assert "selected-model" in text
        assert "observed: custom/observed-model" in text
    assert get_receipt(home, receipt_id) == original


def test_classification_cli_records_distinct_operator_authority(tmp_path):
    from agent.model_selection_classification import get_classification

    home = tmp_path / "home"
    home.mkdir()
    req = dict(schema_version=1, role="builder", execution_kind="delegation",
               execution_id="child", attempt_id="1", task_class="established-pattern",
               required_capabilities=[], input_tokens=1000, reserve_tokens=8192, reasoning="high",
               provenance=dict(frozen_sha="", verified_by="parent", complete=True, contributors=[]))
    source = tmp_path / "scope.json"
    source.write_text(json.dumps({"requirements": req, "complete": True,
                                 "risk_flags": [], "evidence": ["scope:reviewed"]}))
    env = dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(home))
    command = [sys.executable, "-m", "hermes_cli.main", "kanban", "routing", "attest",
               str(source), "--expected-version", "0", "--approval-ref", "operator:reviewed", "--json"]
    proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    record = json.loads(proc.stdout)
    assert record == get_classification(home, req)
    assert record["authority"] == "operator"
    assert record["attester"] == {"approval_ref": "operator:reviewed"}
    stale = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
    assert stale.returncode != 0
    assert get_classification(home, req) == record
