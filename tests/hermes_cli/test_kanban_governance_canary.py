"""Level 5B-P0 Governance Primitive Canary Test Suite.

Proves:
1. Machine-enforced budget guardrail at the persistence boundary (PEP).
2. Atomic mutation denial (0 partial task rows, 0 partial dependency rows).
3. Durable DenialEvent persistence in SQLite ledger with policy hash.
4. Ingress consistency across Direct DB API, Tool handler, and CLI.
5. Immutability of policy and transaction-isolation of denial records.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_governance import (
    GovernanceBudgetExceeded,
    load_governance_policy,
)
from tools import kanban_tools


POLICY_YAML = """policy_id: level5b-p0-v1
immutable_for_run: true

budget:
  max_total_tasks: 2

counting:
  scope: current_board
  includes:
    - planning_root
    - specialist_tasks
    - p0_checkpoints
  excludes:
    - archived_external_history
    - production_board_tasks
  count_basis: persisted_task_rows

enforcement:
  mode: hard_denial
  reason_code: BUDGET_EXHAUSTED
  terminal_mode: GOVERNANCE_LOCKED
  permitted_next_actions:
    - attach_evidence
    - record_unresolved_risk
    - block_budget_exhausted
    - request_reauthorization
"""


def test_level5b_p0_governance_primitive_canary(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()

    board_slug = "level5b-canary"
    kb.create_board(board_slug, name="Level 5B Canary")
    b_dir = kb.board_dir(board_slug)

    # Write policy into the board dir
    policy_path = b_dir / "governance_policy.yaml"
    policy_path.write_text(POLICY_YAML, encoding="utf-8")

    # Step 0: Verify policy loading and baseline hash
    loaded = load_governance_policy(board_slug=board_slug)
    assert loaded is not None, f"Governance policy missing for board {board_slug}"
    policy, policy_hash_before, resolved_path = loaded
    assert policy.get("policy_id") == "level5b-p0-v1"
    assert policy.get("budget", {}).get("max_total_tasks") == 2
    raw_bytes = policy_path.read_bytes()
    expected_hash = hashlib.sha256(raw_bytes).hexdigest()
    assert policy_hash_before == expected_hash

    # Step 1: Precondition — fill budget (2 tasks)
    with kbc.connect_closing(board=board_slug) as conn:
        t1 = kb.create_task(conn, title="Canary Task 1 — Planning Root", assignee="p0-orchestrator", board=board_slug)
        t2 = kb.create_task(conn, title="Canary Task 2 — Specialist", assignee="builder", board=board_slug)
        tasks = conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall()
        assert len(tasks) == 2, f"Expected 2 tasks, found {len(tasks)}"

    task_count_before = 2

    # Step 2: Denied Request 1 — Direct DB API
    with kbc.connect_closing(board=board_slug) as conn:
        with pytest.raises(GovernanceBudgetExceeded) as exc_info:
            kb.create_task(conn, title="Illegal Task 3 — Direct DB", assignee="builder", board=board_slug)
        exc = exc_info.value
        assert exc.code == "BUDGET_EXHAUSTED"
        assert exc.policy_id == "level5b-p0-v1"
        assert exc.policy_hash == policy_hash_before
        assert exc.observed_task_count == 2
        assert exc.configured_limit == 2
        assert exc.ingress == "db_api"

        # Assert zero graph residue
        tasks = conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall()
        assert len(tasks) == 2

    # Step 3: Denied Request 2 — Tool Ingress (kanban_create handler)
    tool_resp_str = kanban_tools._handle_create({
        "title": "Illegal Task 4 — Agent Tool",
        "assignee": "verifier",
        "board": board_slug,
    })
    tool_resp = json.loads(tool_resp_str)
    assert "error" in tool_resp, f"Tool response missing error: {tool_resp}"
    assert tool_resp.get("code") == "BUDGET_EXHAUSTED"
    assert tool_resp.get("reason_code") == "BUDGET_EXHAUSTED"
    assert tool_resp.get("policy_id") == "level5b-p0-v1"
    assert tool_resp.get("policy_hash") == policy_hash_before
    assert tool_resp.get("terminal_mode") == "GOVERNANCE_LOCKED"
    assert "permitted_next_actions" in tool_resp
    assert tool_resp.get("observed_task_count") == 2
    assert tool_resp.get("configured_limit") == 2
    assert tool_resp.get("ingress") == "tool"

    with kbc.connect_closing(board=board_slug) as conn:
        tasks = conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall()
        assert len(tasks) == 2

    # Step 4: Denied Request 3 — CLI Ingress (hermes kanban create)
    cmd = [
        sys.executable,
        "-m", "hermes_cli.main",
        "kanban",
        "--board", board_slug,
        "create",
        "Illegal Task 5 — CLI Shell",
        "--assignee", "verifier",
        "--json",
    ]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HERMES_KANBAN_BOARD"] = board_slug
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode != 0, f"CLI create unexpectedly succeeded: {res.stdout}"

    cli_payload = {}
    try:
        cli_payload = json.loads(res.stdout.strip())
    except Exception:
        for line in res.stdout.splitlines():
            try:
                parsed = json.loads(line.strip())
                if isinstance(parsed, dict):
                    cli_payload = parsed
                    break
            except Exception:
                pass
    assert cli_payload.get("code") == "BUDGET_EXHAUSTED" or "BUDGET_EXHAUSTED" in res.stderr
    assert cli_payload.get("policy_id") == "level5b-p0-v1" or "level5b-p0-v1" in res.stderr

    with kbc.connect_closing(board=board_slug) as conn:
        tasks = conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall()
        assert len(tasks) == 2

    # Step 5: Required Results Verification
    with kbc.connect_closing(board=board_slug) as conn:
        tasks = conn.execute("SELECT id FROM tasks WHERE status != 'archived'").fetchall()
        assert len(tasks) == task_count_before == 2

        # partial_task_rows: 0
        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert total_tasks == 2

        # partial_dependency_rows: 0
        total_links = conn.execute("SELECT COUNT(*) FROM task_links").fetchone()[0]
        assert total_links == 0

        # governance_events: 3 durable denials
        denial_rows = conn.execute(
            "SELECT * FROM governance_events WHERE decision = 'DENY' ORDER BY id ASC"
        ).fetchall()
        assert len(denial_rows) == 3, f"Expected 3 denial records, found {len(denial_rows)}"

        required_fields = [
            "event_id", "policy_id", "policy_hash", "actor", "ingress",
            "requested_operation", "decision", "reason_code",
            "observed_task_count", "configured_limit", "timestamp"
        ]
        ingresses = set()
        for r in denial_rows:
            for field in required_fields:
                assert r[field] is not None, f"Field {field} is None in row {dict(r)}"
            assert r["policy_id"] == "level5b-p0-v1"
            assert r["policy_hash"] == policy_hash_before
            assert r["reason_code"] == "BUDGET_EXHAUSTED"
            assert r["decision"] == "DENY"
            assert r["observed_task_count"] == 2
            assert r["configured_limit"] == 2
            ingresses.add(r["ingress"])

        assert ingresses == {"db_api", "tool", "cli"}

    # Step 6: Policy Hash Immutability
    raw_bytes_after = policy_path.read_bytes()
    policy_hash_after = hashlib.sha256(raw_bytes_after).hexdigest()
    assert policy_hash_before == policy_hash_after
