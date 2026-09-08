"""An external dependency wait must gate native claims, not just first admission."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.create_board("policy")
    with kbc.connect_closing(board="policy") as conn:
        yield home, conn


def configure(home, code, *, timeout=2):
    script = home / "policy.py"
    script.write_text("import json, sys, time\n" + code + "\n", encoding="utf-8")
    path = kb.board_metadata_path("policy")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["pre_claim"] = {
        "command": [sys.executable, str(script)], "timeout_seconds": timeout,
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")


def response(allow, reason="github_dependencies_incomplete"):
    return "print(" + repr(json.dumps({"allow": allow, "reason": reason})) + ")"


def task(conn, status="ready", **kwargs):
    task_id = kb.create_task(conn, title="original", assignee="default", **kwargs)
    if status != kb.get_task(conn, task_id).status:
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    return task_id


@pytest.mark.parametrize("status", ["ready", "review"])
@pytest.mark.parametrize("code", [
    response(False), "sys.exit(3)", "print('not JSON')", "print('x' * 4097)",
    "print('{\"allow\":1,\"reason\":\"bad\"}')", "time.sleep(5)",
])
def test_denied_claim_never_opens_run_and_wait_is_stable(board, status, code):
    home, conn = board
    original = task(conn, status)
    configure(home, code)
    claim = kb.claim_review_task if status == "review" else kb.claim_task
    assert claim(conn, original) is None
    waiting = kb.get_task(conn, original)
    assert waiting.status == "todo"
    assert waiting.current_run_id is None and waiting.consecutive_failures == 0
    assert kb.list_runs(conn, original) == []
    events = kb.list_events(conn, original)
    assert events[-1].kind == "dependency_wait"
    assert events[-1].payload["source"] == "pre_claim"
    assert events[-1].payload["source_status"] == status
    assert kb.recompute_ready(conn) == 0
    assert kb.list_events(conn, original) == events


def test_parent_wait_and_review_resume_require_fresh_policy(board):
    home, conn = board
    parent = task(conn)
    original = task(conn, "todo", parents=[parent], body="DO NOT SEND BODY")
    configure(home, response(False))
    assert kb.recompute_ready(conn) == 0
    assert kb.list_events(conn, original)[-1].payload["reason"] == "github_dependencies_incomplete"
    configure(home, response(True))
    assert kb.recompute_ready(conn) == 0  # External allow cannot override native parents.
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    assert kb.recompute_ready(conn) == 1
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (original,))
    configure(home, response(False))
    assert kb.claim_review_task(conn, original) is None
    configure(home, response(True))
    assert kb.recompute_ready(conn) == 1
    assert kb.get_task(conn, original).status == "review"
    configure(home, response(False, "fresh_claim_denied"))
    assert kb.claim_review_task(conn, original) is None
    assert kb.list_events(conn, original)[-1].payload["reason"] == "fresh_claim_denied"
    assert kb.list_runs(conn, original) == []


def test_dispatch_joins_lock_and_outer_transaction_cannot_invoke_policy(board):
    home, conn = board
    original = task(conn)
    configure(home, response(False))
    assert conn.isolation_level is None
    conn.execute("SELECT 1").fetchone()
    assert not conn.in_transaction
    with kbc.write_txn(conn):
        assert kb.claim_task(conn, original) is None
        assert conn.in_transaction
        assert kb.get_task(conn, original).status == "ready"
    with kbc._dispatch_tick_lock(kb.kanban_db_path("policy")) as held:
        assert held
        assert kbd.dispatch_once(conn, board="policy").skipped_locked
        # The same actual FD may be joined by a direct claim, not a second tick.
        assert kb.claim_task(conn, original) is None
        assert kb.get_task(conn, original).status == "todo"
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (original,))
    spawned = []
    result = kbd.dispatch_once(
        conn, board="policy", spawn_fn=lambda *a, **kw: spawned.append(a),
        reconcile_orphans=False,
    )
    assert not result.skipped_locked
    assert spawned == [] and kb.get_task(conn, original).status == "todo"
    assert kb.list_runs(conn, original) == []


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        capture_output=True, text=True, encoding="utf-8", timeout=20, check=False,
    )


def test_native_read_only_inventory_and_setup_round_trip(board):
    home, conn = board
    identity = "external:example/original#17"
    original = task(conn, "todo", idempotency_key=identity)
    kb.add_comment(conn, original, "tester", "event ordering")
    event_ids = sorted(event.id for event in kb.list_events(conn, original))
    conn.execute("UPDATE task_events SET created_at=-id WHERE task_id=?", (original,))
    # A normal fresh connection would backfill this legacy running row.
    legacy = task(conn, "running")
    before = conn.execute("PRAGMA data_version").fetchone()[0]
    with kbc._dispatch_tick_lock(kb.kanban_db_path("policy")) as held:
        assert held
        listed = cli("--board", "policy", "list", "--no-promote", "--json")
        assert listed.returncode == 0, listed.stderr
        shown = cli("--board", "policy", "show", original, "--read-only", "--json")
        assert shown.returncode == 0, shown.stderr
    detail = json.loads(shown.stdout)
    assert detail["task"]["idempotency_key"] == identity
    assert next(row for row in json.loads(listed.stdout) if row["id"] == original)["idempotency_key"] == identity
    assert all(type(event["id"]) is int for event in detail["events"])
    assert [event["id"] for event in detail["events"]] == list(reversed(event_ids))
    assert kb.get_task(conn, original).status == "todo"
    assert kb.get_task(conn, legacy).current_run_id is None
    assert conn.execute("PRAGMA data_version").fetchone()[0] == before
    with kbc.connect_closing(board="policy", read_only=True) as observer:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            observer.execute("UPDATE tasks SET status='ready' WHERE id=?", (original,))
    configured = cli(
        "boards", "set-pre-claim", "--timeout", "2", "--json", "policy", "--",
        sys.executable, "-c", response(False),
    )
    assert configured.returncode == 0, configured.stderr
    assert json.loads(configured.stdout)["pre_claim"]["timeout_seconds"] == 2
    assert kb.claim_task(conn, original) is None
    cleared = cli("boards", "set-pre-claim", "--clear", "--json", "policy")
    assert cleared.returncode == 0, cleared.stderr
    assert "pre_claim" not in json.loads(cleared.stdout)
    assert kb.recompute_ready(conn) == 1
    assert kb.claim_task(conn, original) is not None


def test_policy_child_can_read_board_without_recursion_and_dry_run_is_observational(board):
    home, conn = board
    original = task(conn, body="PRIVATE TASK PROSE")
    capture = home / "request.json"
    configure(home, "\n".join([
        "import subprocess",
        "from pathlib import Path",
        "request = json.load(sys.stdin)",
        f"Path({str(capture)!r}).write_text(json.dumps(request), encoding='utf-8')",
        "base = [sys.executable, '-m', 'hermes_cli.main', 'kanban', '--board', request['board']]",
        "for args in [['list', '--no-promote', '--json'], ['show', request['task']['id'], '--read-only', '--json']]:",
        "    subprocess.run(base + args, stdout=subprocess.DEVNULL, check=True, timeout=10)",
        response(False),
    ]), timeout=20)
    before = kb.list_events(conn, original)
    kbd.dispatch_once(conn, board="policy", dry_run=True, reconcile_orphans=False)
    assert not capture.exists()
    assert kb.list_events(conn, original) == before
    assert kb.claim_task(conn, original) is None
    packet = json.loads(capture.read_text(encoding="utf-8"))
    assert set(packet["task"]) == {"id", "status", "assignee", "project_id", "idempotency_key"}
    assert packet["board"] == "policy" and packet["phase"] == "claim"
    assert kb.get_task(conn, original).status == "todo"
    assert kb.list_runs(conn, original) == []


@pytest.mark.parametrize("child_kind", ["fresh", "fork"])
def test_only_actual_process_owner_can_reenter_canonical_lock(board, child_kind):
    if child_kind == "fork" and not hasattr(os, "fork"):
        pytest.skip("fork inheritance is unavailable on this platform")
    home, _ = board
    path = kb.kanban_db_path("policy")
    alias = home / "lock-alias.db"
    alias.symlink_to(path)
    script = "\n".join([
        "import os, subprocess, sys",
        "from pathlib import Path",
        "from hermes_cli import kanban_db_connect as kbc",
        "path, alias = map(Path, sys.argv[1:3])",
        "with kbc._dispatch_tick_lock(path, required=True) as held:",
        "    assert held",
        "    with kbc._dispatch_tick_lock(alias, reentrant=True, required=True) as joined:",
        "        assert joined",
        "    if sys.argv[3] == 'fork':",
        "        pid = os.fork()",
        "        if pid == 0:",
        "            with kbc._dispatch_tick_lock(alias, reentrant=True, required=True) as inherited:",
        "                os._exit(1 if inherited else 0)",
        "        assert os.waitpid(pid, 0)[1] == 0",
        "    else:",
        "        child = 'from pathlib import Path; from hermes_cli import kanban_db_connect as kbc; import sys\\nwith kbc._dispatch_tick_lock(Path(sys.argv[1]), reentrant=True, required=True) as held: assert not held'",
        "        subprocess.run([sys.executable, '-c', child, str(alias)], check=True, timeout=10)",
        "    with kbc._dispatch_tick_lock(path, required=True) as contender:",
        "        assert not contender  # Neither child nor alias join released the owner.",
        "with kbc._dispatch_tick_lock(alias, required=True) as released:",
        "    assert released",
    ])
    subprocess.run(
        [sys.executable, "-c", script, str(path), str(alias), child_kind],
        check=True, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("raw", ["{", "[]", '{"pre_claim":null}', '{"pre_claim":{}}',
                                     '{"pre_claim":{"command":["relative"],"timeout_seconds":2}}'])
def test_bad_existing_metadata_cannot_disable_policy(board, raw):
    _, conn = board
    original = task(conn)
    kb.board_metadata_path("policy").write_text(raw, encoding="utf-8")
    assert kb.claim_task(conn, original) is None
    assert kb.get_task(conn, original).status == "todo"
    assert kb.list_events(conn, original)[-1].payload["reason"] == "pre_claim_invalid_configuration"
    assert kb.list_runs(conn, original) == []


@pytest.mark.parametrize("identity", ["wrong-current", "alias", "default"])
def test_actual_connection_identity_selects_the_policy(board, monkeypatch, identity):
    home, existing = board
    configure(home, response(False))
    path = kb.kanban_db_path("policy")
    if identity == "default":
        policy = json.loads(kb.board_metadata_path("policy").read_text(encoding="utf-8"))["pre_claim"]
        kb.create_board("default")
        kb.write_board_metadata("default", pre_claim=policy)
        path = kb.kanban_db_path("default")
    if identity == "alias":
        alias = home / "alias.db"
        alias.symlink_to(path)
        monkeypatch.setenv("HERMES_KANBAN_DB", str(alias))
        path = alias
    kb.set_current_board("default" if identity != "default" else "policy")
    with kbc.connect_closing(path) as conn:
        original = task(conn)
        assert kb.claim_task(conn, original) is None
        assert kb.get_task(conn, original).status == "todo"
        assert kb.list_runs(conn, original) == []
    if identity == "alias":
        assert existing.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0


def test_required_lock_failure_cannot_fall_through_to_legacy_promotion(board, monkeypatch):
    home, conn = board
    original = task(conn, "todo")
    configure(home, response(True))
    real_open = Path.open

    def deny_lock(path, *args, **kwargs):
        if path.name.endswith(".dispatch.lock"):
            raise PermissionError("fixture lock unavailable")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_lock)
    assert kb.recompute_ready(conn) == 0
    assert kb.get_task(conn, original).status == "todo"
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (original,))
    assert kb.claim_task(conn, original) is None
    assert kb.list_runs(conn, original) == []
    assert kbd.dispatch_once(conn, board="policy").skipped_locked


@pytest.mark.parametrize("change", ["body", "event", "parent", "policy", "competing-claim"])
def test_external_check_holds_no_database_transaction_and_stale_allow_cannot_claim(board, change):
    home, conn = board
    parent = task(conn)
    conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
    original = task(conn, parents=[parent])
    started, release = home / "started", home / "release"
    configure(home, "\n".join([
        "from pathlib import Path",
        f"Path({str(started)!r}).touch()",
        f"while not Path({str(release)!r}).exists(): time.sleep(0.01)",
        response(True),
    ]), timeout=15)

    def first_claim():
        with kbc.connect_closing(board="policy") as other:
            return kb.claim_task(other, original)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first_claim)
        try:
            deadline = time.monotonic() + 10
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert started.exists(), "policy did not start"
            # This real write succeeds while the external command is still waiting.
            mutations = {
                "body": lambda: conn.execute("UPDATE tasks SET body='changed' WHERE id=?", (original,)),
                "event": lambda: kb.add_comment(conn, original, author="tester", body="changed"),
                "parent": lambda: conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,)),
                "policy": lambda: kb.write_board_metadata("policy", clear_pre_claim=True),
                "competing-claim": lambda: kb.claim_task(conn, original),
            }
            result = mutations[change]()
            if change == "competing-claim":
                assert result is None
        finally:
            release.touch()
        claimed = future.result(timeout=10)
    if change == "competing-claim":
        assert claimed is not None and len(kb.list_runs(conn, original)) == 1
    else:
        assert claimed is None and kb.list_runs(conn, original) == []
        assert kb.get_task(conn, original).status == "ready"


def test_promotion_commits_each_slot_and_keeps_sticky_and_breaker_holds(board):
    home, conn = board
    originals = [task(conn, "todo", priority=p) for p in (1, 3, 2)]
    recoverable = task(conn, "blocked", priority=4, max_retries=2)
    conn.execute("UPDATE tasks SET consecutive_failures=1 WHERE id=?", (recoverable,))
    sticky = task(conn)
    assert kb.block_task(conn, sticky, reason="human decision")
    breaker = task(conn, "blocked", max_retries=1)
    conn.execute("UPDATE tasks SET consecutive_failures=1 WHERE id=?", (breaker,))
    trace = home / "promotion-trace.jsonl"
    configure(home, "\n".join([
        "import subprocess",
        "from pathlib import Path",
        "request = json.load(sys.stdin)",
        "out = subprocess.check_output([sys.executable, '-m', 'hermes_cli.main', 'kanban', '--board', request['board'], 'list', '--no-promote', '--json'])",
        "ready = [row['id'] for row in json.loads(out) if row['status'] == 'ready']",
        f"with Path({str(trace)!r}).open('a', encoding='utf-8') as stream: stream.write(json.dumps([request['task']['id'], ready]) + '\\n')",
        "print(json.dumps({'allow':len(ready) < 1,'reason':'capacity'}))",
    ]), timeout=20)
    assert kb.recompute_ready(conn) == 1
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert [row[0] for row in rows] == [recoverable, originals[1], originals[2], originals[0]]
    assert rows[0][1] == [] and rows[1][1] == [recoverable]
    assert kb.get_task(conn, recoverable).status == "ready"
    assert kb.get_task(conn, sticky).status == "blocked"
    assert kb.get_task(conn, breaker).status == "blocked"


@pytest.mark.parametrize("status", ["ready", "review"])
def test_policy_latency_does_not_consume_the_new_claim_lease(board, status):
    home, conn = board
    original = task(conn, status)
    checked_at = home / "checked-at"
    configure(home, "\n".join([
        "from pathlib import Path", "time.sleep(2)",
        f"Path({str(checked_at)!r}).write_text(str(int(time.time())), encoding='utf-8')",
        response(True),
    ]), timeout=10)
    claim = kb.claim_review_task if status == "review" else kb.claim_task
    claimed = claim(conn, original, ttl_seconds=30)
    assert claimed is not None
    assert claimed.claim_expires >= int(checked_at.read_text(encoding="utf-8")) + 30


def test_delegated_claim_is_refused_before_external_policy_runs(board):
    from agent.delegation_context import delegated_child_context

    home, conn = board
    original = task(conn)
    invoked = home / "must-not-invoke"
    configure(home, f"from pathlib import Path\nPath({str(invoked)!r}).touch()\n" + response(True))
    with delegated_child_context():
        with pytest.raises(PermissionError):
            kb.claim_task(conn, original)
        with pytest.raises(PermissionError):
            kb.write_board_metadata("policy", clear_pre_claim=True)
    assert not invoked.exists()
    assert kb.get_task(conn, original).status == "ready"
    assert kb.list_runs(conn, original) == []


def test_setup_rejects_custom_database_and_read_only_requires_existing_schema(board, monkeypatch):
    home, conn = board
    path = kb.board_metadata_path("policy")
    before = path.read_text(encoding="utf-8")
    custom = home / "custom.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(custom))
    result = cli("boards", "set-pre-claim", "--json", "policy", "--", sys.executable, "-c", response(True))
    assert result.returncode != 0 and "canonical" in result.stderr
    assert path.read_text(encoding="utf-8") == before
    with pytest.raises(sqlite3.OperationalError):
        with kbc.connect_closing(custom, read_only=True):
            pytest.fail("missing board was opened")
    assert not custom.exists()
