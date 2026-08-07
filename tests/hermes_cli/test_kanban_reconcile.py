"""Atomic external-projection reconciliation tests for Hermes Kanban."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import io
import json
import threading

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


LINEAGE = "sha256:" + "1" * 64


def key(char: str) -> str:
    return "sha256:" + char * 64


def request(
    operation: str = "create-if-absent",
    *,
    revision: str = "2026-07-27T18:00:00Z",
    request_key: str = key("2"),
    expected: dict | None = None,
    title: str = "Redacted task",
    parents: list[str] | None = None,
) -> dict:
    value = {
        "schema_version": 1,
        "operation": operation,
        "canonical_ref": "nuri-com/nuri-infra#37",
        "idempotency_key": request_key,
        "idempotency_lineage": LINEAGE,
        "source_revision": revision,
    }
    if operation in {"create-if-absent", "replace-unclaimed"}:
        value["task"] = {
            "title": title,
            "body": "Redacted bounded scope",
            "assignee": "nuriforge",
            "project": "nuri-infra",
            "parents": parents or [],
            "canonical_ref": value["canonical_ref"],
            "source_status": "Ready",
            "source_revision": revision,
            "idempotency_lineage": LINEAGE,
            "idempotency_key": request_key,
        }
    if expected is not None:
        value["expected"] = expected
    return value


def expected(
    task_id: str,
    revision: str = "2026-07-27T18:00:00Z",
    request_key: str = key("2"),
) -> dict:
    return {
        "task_id": task_id,
        "status": "ready",
        "source_revision": revision,
        "idempotency_lineage": LINEAGE,
        "idempotency_key": request_key,
        "claim_lock": None,
        "run_id": None,
    }


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb.init_db()
    return tmp_path


def test_create_is_atomic_across_independent_connections_and_replays_exactly(kanban_home):
    payload = request()
    barrier = threading.Barrier(2)

    def create() -> dict:
        with kb.connect_closing() as conn:
            barrier.wait(timeout=5)
            return kb.reconcile_task(conn, payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _: create(), range(2)))

    assert first == second
    assert first["outcome"] == "created"
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks WHERE status != 'archived'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM kanban_reconcile_requests").fetchone()[0] == 1
        stored = " ".join(
            str(row[0] or "")
            for table, column in (
                ("kanban_reconcile_requests", "result_json"),
                ("task_events", "payload"),
            )
            for row in conn.execute(f"SELECT {column} FROM {table}").fetchall()
        )
        assert "Redacted bounded scope" not in stored


def test_reused_request_key_with_changed_body_conflicts_without_mutation(kanban_home):
    original = request()
    changed = copy.deepcopy(original)
    changed["task"]["body"] = "different body"
    with kb.connect_closing() as conn:
        created = kb.reconcile_task(conn, original)
        conflict = kb.reconcile_task(conn, changed)
        assert conflict["outcome"] == "conflict"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        assert kb.get_task(conn, created["task_id"]).body == "Redacted bounded scope"


def test_replace_is_atomic_and_replay_is_stable(kanban_home):
    with kb.connect_closing() as conn:
        created = kb.reconcile_task(conn, request())
        replacement = request(
            "replace-unclaimed",
            revision="2026-07-27T18:01:00Z",
            request_key=key("3"),
            expected=expected(created["task_id"]),
            title="Revised task",
        )
        result = kb.reconcile_task(conn, replacement)
        assert result["outcome"] == "replaced"
        assert result["replaced_task_id"] == created["task_id"]
        assert kb.get_task(conn, created["task_id"]).status == "archived"
        assert kb.get_task(conn, result["task_id"]).status == "ready"
        assert kb.reconcile_task(conn, replacement) == result


def test_replace_rolls_back_archival_state_and_ledger_on_create_failure(kanban_home):
    with kb.connect_closing() as conn:
        created = kb.reconcile_task(conn, request())
        broken = request(
            "replace-unclaimed",
            revision="2026-07-27T18:01:00Z",
            request_key=key("4"),
            expected=expected(created["task_id"]),
            parents=["missing-parent"],
        )
        with pytest.raises(ValueError, match="unknown parent"):
            kb.reconcile_task(conn, broken)
        assert kb.get_task(conn, created["task_id"]).status == "ready"
        state = conn.execute(
            "SELECT task_id, source_revision, active FROM kanban_reconcile_state WHERE lineage=?",
            (LINEAGE,),
        ).fetchone()
        assert tuple(state) == (created["task_id"], "2026-07-27T18:00:00Z", 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM kanban_reconcile_requests WHERE request_key=?",
            (key("4"),),
        ).fetchone()[0] == 0


def test_cancel_unclaimed_and_stale_revision(kanban_home):
    with kb.connect_closing() as conn:
        created = kb.reconcile_task(conn, request())
        cancelled = kb.reconcile_task(
            conn,
            request(
                "cancel-unclaimed",
                revision="2026-07-27T18:02:00Z",
                request_key=key("5"),
                expected=expected(created["task_id"]),
            ),
        )
        assert cancelled["outcome"] == "cancelled"
        assert kb.get_task(conn, created["task_id"]).status == "archived"
        stale = kb.reconcile_task(
            conn,
            request(
                revision="2026-07-27T18:01:00Z",
                request_key=key("6"),
            ),
        )
        assert stale["outcome"] == "stale-source-revision"
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_dispatch_claim_and_replace_race_never_overwrites_claimed_work(kanban_home):
    for index in range(8):
        lineage = "sha256:" + f"{index + 10:064x}"
        create_payload = request(request_key="sha256:" + f"{index + 100:064x}")
        create_payload["idempotency_lineage"] = lineage
        create_payload["task"]["idempotency_lineage"] = lineage
        with kb.connect_closing() as conn:
            created = kb.reconcile_task(conn, create_payload)
        replace_payload = request(
            "replace-unclaimed",
            revision="2026-07-27T18:01:00Z",
            request_key="sha256:" + f"{index + 200:064x}",
            expected={
                **expected(
                    created["task_id"],
                    request_key=create_payload["idempotency_key"],
                ),
                "idempotency_lineage": lineage,
            },
        )
        replace_payload["idempotency_lineage"] = lineage
        replace_payload["task"]["idempotency_lineage"] = lineage
        barrier = threading.Barrier(2)

        def claim():
            with kb.connect_closing() as conn:
                barrier.wait(timeout=5)
                return kb.claim_task(conn, created["task_id"], claimer="race-test")

        def replace():
            with kb.connect_closing() as conn:
                barrier.wait(timeout=5)
                return kb.reconcile_task(conn, replace_payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            claimed_future = pool.submit(claim)
            replaced_future = pool.submit(replace)
            claimed = claimed_future.result(timeout=10)
            replaced = replaced_future.result(timeout=10)

        with kb.connect_closing() as conn:
            old = kb.get_task(conn, created["task_id"])
            if claimed is not None:
                assert replaced["outcome"] == "claimed"
                assert old.status == "running"
                assert replaced["task_id"] == old.id
            else:
                assert replaced["outcome"] == "replaced"
                assert old.status == "archived"
                assert kb.get_task(conn, replaced["task_id"]).status == "ready"


def test_cli_requires_canonical_json_and_emits_canonical_result(kanban_home, monkeypatch, capsys):
    payload = request()
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(canonical))
    assert kc._cmd_reconcile(type("Args", (), {"input": "-"})()) == 0
    output = capsys.readouterr().out
    decoded = json.loads(output)
    assert decoded["outcome"] == "created"
    assert output == json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload, indent=2)))
    assert kc._cmd_reconcile(type("Args", (), {"input": "-"})()) == 2
    assert capsys.readouterr().out == '{"outcome":"invalid-input","schema_version":1}\n'


def test_delegated_child_guard_emits_canonical_reconcile_result(monkeypatch, capsys):
    assert "reconcile" in kc._DELEGATED_CHILD_DENIED_ACTIONS
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = root.parse_args(["kanban", "reconcile", "--input", "-"])

    assert kc.kanban_command(args) == 1
    captured = capsys.readouterr()
    assert captured.out == '{"outcome":"permission-denied","schema_version":1}\n'
    assert captured.err == ""
