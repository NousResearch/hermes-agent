"""Deterministic integration regressions for repair-to-canary completion proof."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real isolated board, never the developer's live HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _requirements(*kinds: str) -> list[dict[str, str]]:
    return [
        {"id": f"proof-{index}", "kind": kind, "description": f"{kind} proof"}
        for index, kind in enumerate(kinds, start=1)
    ]


def _observed(identifier: str, status: str, payload: dict | None = None) -> dict:
    item = {
        "id": identifier,
        "status": status,
        "observed_at": int(time.time()),
        "source": "repair-to-canary-regression",
    }
    if payload is not None:
        item["payload"] = payload
    return item


def _local_check_payload() -> dict[str, object]:
    command = [sys.executable, "-c", "raise SystemExit(0)"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0
    return {"command": " ".join(command), "exit_code": result.returncode}


def _ready_task(conn, **kwargs: Any) -> str:
    task_id = kb.create_task(conn, title="repair-to-canary", assignee="worker", **kwargs)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    return task_id


def test_blocked_dependency_is_repaired_when_its_prerequisite_completes(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parent = _ready_task(conn)
        child = _ready_task(conn)
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # The real claim guard re-gates the linked child until repair completes.
        assert kb.claim_task(conn, child, claimer="worker") is None
        assert kb.get_task(conn, child).status == "todo"

        assert kb.claim_task(conn, parent, claimer="worker") is not None
        assert kb.complete_task(conn, parent, summary="merged prerequisite")

        repaired = kb.get_task(conn, child)
        assert repaired is not None and repaired.status == "ready"
        events = kb.list_events(conn, child)
        assert any(event.kind == "claim_rejected" for event in events)


def test_local_test_evidence_without_canary_keeps_card_in_flight(kanban_home: Path) -> None:
    contract = {"version": 1, "required": _requirements("check", "canary"), "observed": []}
    with kbc.connect_closing() as conn:
        task_id = _ready_task(conn, acceptance_evidence=contract)
        evidence = {
            **contract,
            "observed": [_observed("proof-1", "passed", _local_check_payload())],
        }

        assert not kb.complete_task(conn, task_id, acceptance_evidence=evidence)
        task = kb.get_task(conn, task_id)
        rejection = [event for event in kb.list_events(conn, task_id) if event.kind == "acceptance_evidence_rejected"][-1]

    assert task is not None and task.status == "ready"
    assert rejection.payload == {"classification": "unsatisfied", "required_ids": ["proof-2"]}


def test_dirty_pr_receipt_keeps_card_in_flight_with_readback(tmp_path: Path, kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real gh subprocess returns a failing required check; no network is used."""
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    sha = "a" * 40
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "endpoint = sys.argv[2]\n"
        f"sha = {sha!r}\n"
        "if endpoint == 'graphql': value = {'data': {'repository': {'pullRequest': {'headRefOid': sha, 'baseRefName': 'main', 'state': 'OPEN', 'baseRef': {'branchProtectionRule': {'requiredStatusChecks': [{'context': 'required', 'app': {'databaseId': 1}}]}}}}}}\n"
        "elif '/rules/branches/' in endpoint: value = [[]]\n"
        "elif '/check-runs' in endpoint: value = [{'total_count': 1, 'check_runs': [{'id': 7, 'name': 'required', 'head_sha': sha, 'app': {'id': 1}, 'status': 'completed', 'conclusion': 'failure', 'html_url': 'https://example.invalid/check/7'}]}]\n"
        "elif '/statuses' in endpoint: value = [[]]\n"
        "elif '/pulls/' in endpoint: value = {'head': {'sha': sha}, 'base': {'ref': 'main'}, 'state': 'open'}\n"
        "else: raise SystemExit(2)\n"
        "print(json.dumps(value))\n"
    )
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])

    with kbc.connect_closing() as conn:
        task_id = _ready_task(conn, completion_contract="acme/repo")
        assert not kb.complete_task(conn, task_id, metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        task = kb.get_task(conn, task_id)
        receipt = [event for event in kb.list_events(conn, task_id) if event.kind == "pr_acceptance"][-1].payload

    assert task is not None and task.status == "ready"
    assert receipt["classification"] == "failure"
    assert receipt["checks"] == [{"name": "required", "id": 7, "url": "https://example.invalid/check/7", "head_sha": sha, "classification": "failure", "conclusion": "failure"}]
    assert "retry completion" in receipt["recovery"]


def test_passed_canary_without_a_live_worker_is_rejected(kanban_home: Path) -> None:
    contract = {"version": 1, "required": _requirements("canary"), "observed": []}
    with kbc.connect_closing() as conn:
        task_id = _ready_task(conn, acceptance_evidence=contract)
        evidence = {
            **contract,
            "observed": [_observed("proof-1", "passed", {"pid": 999999, "run_id": 1, "spawned_at": 10, "heartbeat_at": 11})],
        }
        assert not kb.complete_task(conn, task_id, acceptance_evidence=evidence)
        task = kb.get_task(conn, task_id)
        rejection = [event for event in kb.list_events(conn, task_id) if event.kind == "acceptance_evidence_rejected"][-1]

    assert task is not None and task.status == "ready"
    assert rejection.payload == {"classification": "canary_readback_mismatch", "required_ids": ["proof-1"]}


def test_repair_canary_pid_fresh_heartbeat_and_evidence_readback_complete(kanban_home: Path) -> None:
    contract = {"version": 1, "required": _requirements("check", "canary"), "observed": []}
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with kbc.connect_closing() as conn:
            task_id = _ready_task(conn, acceptance_evidence=contract)
            claimed = kb.claim_task(conn, task_id, claimer="worker")
            assert claimed is not None
            run_id = claimed.current_run_id
            assert run_id is not None
            kbd._set_worker_pid(conn, task_id, worker.pid)
            # The contract requires a heartbeat later than the durable spawn timestamp.
            time.sleep(1.05)
            assert kbd.heartbeat_worker(conn, task_id, note="fresh proof", expected_run_id=run_id)
            before = kb.get_task(conn, task_id)
            run = kb.get_run(conn, run_id)
            assert before is not None and run is not None
            assert before.worker_pid == worker.pid == run.worker_pid
            assert before.last_heartbeat_at == run.last_heartbeat_at
            evidence = {
                **contract,
                "observed": [
                    _observed("proof-1", "passed", _local_check_payload()),
                    _observed("proof-2", "passed", {
                        "pid": worker.pid,
                        "run_id": run_id,
                        "spawned_at": run.started_at,
                        "heartbeat_at": before.last_heartbeat_at,
                    }),
                ],
            }
            assert kb.complete_task(conn, task_id, expected_run_id=run_id, acceptance_evidence=evidence, summary="canary proven")
            completed = kb.get_task(conn, task_id)
            closed_run = kb.get_run(conn, run_id)
            receipt = [event for event in kb.list_events(conn, task_id) if event.kind == "acceptance_evidence"][-1]

        assert completed is not None and completed.status == "done"
        assert closed_run is not None and closed_run.outcome == "completed"
        # Completion clears live PID ownership; the closed run metadata retains
        # the evidence snapshot that bound the live PID before terminalization.
        assert closed_run.worker_pid is None
        assert closed_run.last_heartbeat_at == before.last_heartbeat_at
        assert isinstance(closed_run.metadata, dict)
        assert closed_run.metadata["acceptance_evidence"]["observed"][1]["payload"]["pid"] == worker.pid
        assert receipt.payload == {"required_ids": ["proof-1", "proof-2"]}
    finally:
        worker.terminate()
        worker.wait(timeout=5)
