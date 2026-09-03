"""P0 profile/task capability contract checks for Kanban routing."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def contract_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "worker").mkdir(parents=True)
    (home / "profiles" / "reviewer").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    for name in list(sys.modules):
        if name == "hermes_constants" or name.startswith("hermes_cli"):
            del sys.modules[name]
    from hermes_cli import kanban_db

    yield home, kanban_db


def _write_profile_contract(home: Path, profile: str, body: str) -> None:
    profile_dir = home / "profiles" / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(body, encoding="utf-8")


def _spawn(*args, **kwargs) -> int:
    return 12345


def test_task_contract_normalizes_explicit_p0_fields_and_bau_plane_default(contract_home):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  enforce_kanban_route_compatibility: true
  capabilities: [provider.read, github.write]
  allowed_actions: [provider.read, github.write]
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(
            conn,
            title="contracted",
            assignee="worker",
            task_contract={
                "required_capabilities": ["provider.read", "provider.read"],
                "allowed_actions": ["provider.read"],
                "forbidden_actions": ["email.send"],
                "required_evidence": ["exact_external_readback_or_raw_source"],
                "safety_acceptance": ["no customer-visible mutation"],
                "outcome_acceptance": ["readback reconciled"],
                "recoverable_conditions": ["transient provider failure"],
                "hard_stop_conditions": ["ambiguous write target"],
                "evidence_hierarchy": ["exact_external_readback_or_raw_source"],
                "status_fields": {
                    "prepared": "route selected",
                    "implemented": "code changed",
                    "verified": "tests pass",
                    "reviewed": None,
                    "ci_pr_merge_observed": None,
                    "production_executed": None,
                    "business_outcome": "SAFE_BUT_NO_OUTCOME",
                    "gaps": ["review pending"],
                },
            },
        )
        task = kb.get_task(conn, tid)
    from hermes_cli.config import DEFAULT_CONFIG

    assert task is not None
    assert DEFAULT_CONFIG["profile_contract"]["plane_for_bau_process_tracking"] is False
    contract = task.task_contract
    assert contract["required_capabilities"] == ["provider.read"]
    assert contract["allowed_actions"] == ["provider.read"]
    assert contract["forbidden_actions"] == ["email.send"]
    assert contract["required_evidence"] == ["exact_external_readback_or_raw_source"]
    assert contract["safety_acceptance"] == ["no customer-visible mutation"]
    assert contract["outcome_acceptance"] == ["readback reconciled"]
    assert contract["recoverable_conditions"] == ["transient provider failure"]
    assert contract["hard_stop_conditions"] == ["ambiguous write target"]
    assert contract["status_fields"]["prepared"] == "route selected"
    assert contract["status_fields"]["business_outcome"] == "SAFE_BUT_NO_OUTCOME"
    assert contract["status_fields"]["gaps"] == ["review pending"]
    assert contract["plane_for_bau_process_tracking"] is False
    assert task.project_id is None


def test_dispatch_spawns_when_profile_satisfies_task_contract(contract_home):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  enforce_kanban_route_compatibility: true
  capabilities: [provider.read]
  allowed_actions: [provider.read]
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(
            conn,
            title="compatible",
            assignee="worker",
            task_contract={
                "required_capabilities": ["provider.read"],
                "allowed_actions": ["provider.read"],
                "required_evidence": ["exact_external_readback_or_raw_source"],
            },
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn, dry_run=False, max_spawn=1)
        task = kb.get_task(conn, tid)

    assert result.spawned and result.spawned[0][0] == tid
    assert result.skipped_incompatible == []
    assert task.status == "running"


def test_dispatch_blocks_missing_capability_before_claim(contract_home):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  enforce_kanban_route_compatibility: true
  capabilities: [provider.read]
  allowed_actions: [provider.read]
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(
            conn,
            title="needs send",
            assignee="worker",
            task_contract={
                "required_capabilities": ["email.send"],
                "allowed_actions": ["email.send"],
                "required_evidence": ["exact_external_readback_or_raw_source"],
            },
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn, dry_run=False, max_spawn=1)
        task = kb.get_task(conn, tid)

    assert result.spawned == []
    assert result.skipped_incompatible[0][0] == tid
    assert result.skipped_incompatible[0][2]["missing_capabilities"] == ["email.send"]
    assert result.skipped_incompatible[0][2]["missing_allowed_actions"] == ["email.send"]
    assert task.status == "blocked"
    assert task.block_kind == "capability"
    assert task.claim_lock is None


def test_sanitized_display_alone_is_not_authoritative_evidence(contract_home):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  enforce_kanban_route_compatibility: true
  capabilities: [provider.read]
  allowed_actions: [provider.read]
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(
            conn,
            title="sanitized only",
            assignee="worker",
            task_contract={
                "required_capabilities": ["provider.read"],
                "allowed_actions": ["provider.read"],
                "required_evidence": ["sanitized_display"],
            },
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn, dry_run=False, max_spawn=1)
        task = kb.get_task(conn, tid)

    assert result.spawned == []
    assert result.skipped_incompatible[0][0] == tid
    assert result.skipped_incompatible[0][2]["missing_capabilities"] == [
        "authoritative_non_sanitized_evidence"
    ]
    assert task.status == "blocked"
    assert task.claim_lock is None


def test_legacy_profiles_ignore_task_contract_until_enabled(contract_home):
    home, kb = contract_home
    _write_profile_contract(home, "worker", "profile_contract: {}\n")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(
            conn,
            title="legacy",
            assignee="worker",
            task_contract={"required_capabilities": ["unavailable.capability"]},
        )
        result = kb.dispatch_once(conn, spawn_fn=_spawn, dry_run=False, max_spawn=1)
        task = kb.get_task(conn, tid)

    assert result.spawned and result.spawned[0][0] == tid
    assert result.skipped_incompatible == []
    assert task.status == "running"


def test_route_required_refuses_task_tool_without_selected_route(contract_home, monkeypatch):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  route_required_for_task_tools: true
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(conn, title="route gated", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker-test") is not None
        run = kb.latest_run(conn, tid)
        assert run is not None
        run_id = str(run.id)
    db_path = kb.kanban_db_path()

    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "worker"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.delenv("HERMES_ROUTE_SELECTED", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_TASK", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_SESSION_ID", raising=False)
    for name in ["tools.kanban_tools", "hermes_cli.config"]:
        sys.modules.pop(name, None)
    kt = importlib.import_module("tools.kanban_tools")

    blocked = json.loads(kt._handle_block({"reason": "should not write"}))
    assert blocked["error"]
    assert "route_required_for_task_tools" in blocked["error"]

    monkeypatch.setenv("HERMES_ROUTE_TASK", tid)
    assert kt._reject_missing_route("kanban_block") is None

    allowed = json.loads(kt._handle_block({"reason": "selected route bound"}))
    assert allowed["ok"] is True
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"


def test_route_required_accepts_route_governor_runtime_state(contract_home, monkeypatch):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  route_required_for_task_tools: true
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(conn, title="route gated via db", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker-test") is not None
        run = kb.latest_run(conn, tid)
        assert run is not None
        run_id = str(run.id)
    db_path = kb.kanban_db_path()

    runtime_db = home / "unified-control" / "runtime.db"
    runtime_db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    with sqlite3.connect(runtime_db) as conn:
        conn.execute(
            """CREATE TABLE route_state(
                profile TEXT NOT NULL,
                task_id TEXT NOT NULL,
                session_id TEXT,
                selected_at REAL NOT NULL,
                PRIMARY KEY(profile, task_id)
            )"""
        )
        conn.execute(
            "INSERT INTO route_state(profile, task_id, session_id, selected_at) VALUES(?,?,?,?)",
            ("worker", tid, "session-1", 1.0),
        )

    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "worker"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.setenv("HERMES_SESSION_ID", "session-1")
    monkeypatch.delenv("HERMES_ROUTE_SELECTED", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_TASK", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_SESSION_ID", raising=False)
    for name in ["tools.kanban_tools", "hermes_cli.config"]:
        sys.modules.pop(name, None)
    kt = importlib.import_module("tools.kanban_tools")

    assert kt._reject_missing_route("kanban_block") is None
    allowed = json.loads(kt._handle_block({"reason": "selected route in runtime db"}))
    assert allowed["ok"] is True
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"


def test_route_required_accepts_recent_route_when_session_env_unavailable(contract_home, monkeypatch):
    home, kb = contract_home
    _write_profile_contract(
        home,
        "worker",
        """
profile_contract:
  route_required_for_task_tools: true
""",
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Default")
        tid = kb.create_task(conn, title="route gated via recent db", assignee="worker")
        assert kb.claim_task(conn, tid, claimer="worker-test") is not None
        run = kb.latest_run(conn, tid)
        assert run is not None
        run_id = str(run.id)
    db_path = kb.kanban_db_path()

    runtime_db = home / "unified-control" / "runtime.db"
    runtime_db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    import time

    with sqlite3.connect(runtime_db) as conn:
        conn.execute(
            """CREATE TABLE route_state(
                profile TEXT NOT NULL,
                task_id TEXT NOT NULL,
                session_id TEXT,
                selected_at REAL NOT NULL,
                PRIMARY KEY(profile, task_id)
            )"""
        )
        conn.execute(
            "INSERT INTO route_state(profile, task_id, session_id, selected_at) VALUES(?,?,?,?)",
            ("worker", "session-key-only", "session-key-only", time.time()),
        )

    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "worker"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", run_id)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_SELECTED", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_TASK", raising=False)
    monkeypatch.delenv("HERMES_ROUTE_SESSION_ID", raising=False)
    for name in ["tools.kanban_tools", "hermes_cli.config"]:
        sys.modules.pop(name, None)
    kt = importlib.import_module("tools.kanban_tools")

    assert kt._reject_missing_route("kanban_block") is None
    allowed = json.loads(kt._handle_block({"reason": "recent selected route in runtime db"}))
    assert allowed["ok"] is True
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"
