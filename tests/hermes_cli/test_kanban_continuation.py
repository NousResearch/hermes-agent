"""Explicit continuation is an operator action, never a PR/progress-comment heuristic."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def prior_attempt(conn):
    tid = kb.create_task(conn, title="Amend the existing implementation", assignee="builder")
    first = kb.claim_task(conn, tid)
    kb.add_comment(conn, tid, author="builder", body="Existing work: https://github.com/example/project/pull/12")
    kb.block_task(conn, tid, reason="Awaiting the exact behavior decision", expected_run_id=first.current_run_id)
    comment = kb.add_comment(conn, tid, author="operator", body="Approved: continue this task and amend the existing PR")
    kb.unblock_task(conn, tid)
    return tid, first.current_run_id, comment


def test_continuation_retains_gated_assignee_compare_and_swap(board):
    from hermes_cli.kanban_continuation import resolve_respawn_admission

    tid, source_run, authority = prior_attempt(board)
    grant = authorize(board, tid, source_run, authority)
    assert kb.claim_task(board, tid, continuation_id=grant, expect_assignee="other") is None
    assert kb.get_task(board, tid).current_run_id is None
    assert resolve_respawn_admission(board, tid, "ready", "active_pr") == (None, grant)
    claimed = kb.claim_task(board, tid, continuation_id=grant, expect_assignee="builder")
    assert claimed is not None
    assert claimed.assignee == "builder"
    assert claimed.current_run_id != source_run
    assert kb.claim_task(board, tid, continuation_id=grant, expect_assignee="builder") is None


def test_explicit_same_owner_continuation_enters_normal_dispatch(board):
    tid, source_run, authority = prior_attempt(board)
    started = []
    before = kbd.dispatch_once(board, spawn_fn=lambda t, w: started.append(t.id), max_in_progress=8)
    assert (tid, "active_pr") in before.respawn_guarded
    authorize = getattr(kb, "authorize_continuation", None)
    assert callable(authorize), "A native explicit same-owner continuation operation is missing"
    grant = authorize(board, tid, profile="builder", expected_run_id=source_run,
                      authority_comment_id=authority, actor="operator", reason="Approved amendment of existing work")
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: started.append(t.id), max_in_progress=8)
    assert [entry[0] for entry in result.spawned] == [tid]
    assert started == [tid]
    task = kb.get_task(board, tid)
    assert task.assignee == "builder"
    assert task.current_run_id != source_run
    consumed = board.execute("SELECT payload,run_id FROM task_events WHERE task_id=? AND kind='continuation_consumed'", (tid,)).fetchone()
    assert consumed is not None
    assert consumed["run_id"] == task.current_run_id
    import json
    assert json.loads(consumed["payload"])["authorization_id"] == grant


def test_operator_surface_records_authority_without_claiming(board):
    from hermes_cli.kanban import run_slash
    tid, source_run, authority = prior_attempt(board)
    output = run_slash(f'continue {tid} --owner-profile builder --from-run {source_run} --authority-comment {authority} --reason "Approved amendment"')
    assert "Continuation authorized" in output
    assert kb.get_task(board, tid).status == "ready"
    assert kb.get_task(board, tid).current_run_id is None


def run_operator_cli(board, tmp_path, *args, **env_overrides):
    import os
    import subprocess
    import sys

    database = board.execute("PRAGMA database_list").fetchone()[2]
    assert Path(database).is_relative_to(tmp_path)
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_KANBAN_DB=database,
               HERMES_KANBAN_HOME=os.environ["HERMES_HOME"], **env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=Path(__file__).parents[2], env=env, capture_output=True, text=True,
        timeout=30, check=False,
    )


@pytest.mark.parametrize("prefix,suffix,actor", [
    ([], [], "default"),
    (["--profile", "operator"], [], "operator"),
    ([], ["--profile", "operator"], "operator"),
    (["--profile=operator"], [], "operator"),
])
def test_top_level_cli_authorizes_owner_without_selecting_it_as_global_profile(
    board, tmp_path, prefix, suffix, actor,
):
    import json
    import os

    operator_home = Path(os.environ["HERMES_HOME"]) / "profiles" / "operator"
    operator_home.mkdir(parents=True)
    (operator_home / "config.yaml").write_text("{}\n")
    tid, source_run, authority = prior_attempt(board)
    result = run_operator_cli(
        board, tmp_path, *prefix, "kanban", "continue", tid,
        "--owner-profile", "builder", "--from-run", str(source_run),
        "--authority-comment", str(authority), "--reason", "Approved amendment", *suffix,
    )
    assert result.returncode == 0, result.stderr
    assert "Continuation authorized" in result.stdout
    grant = board.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='continuation_authorized'",
        (tid,),
    ).fetchone()
    payload = json.loads(grant["payload"])
    assert payload["profile"] == "builder"
    assert payload["actor"] == actor
    assert kb.get_task(board, tid).current_run_id is None


@pytest.mark.parametrize("fence,error", [
    ("HERMES_KANBAN_TASK", "self-authorize"),
    ("HERMES_DELEGATED_CHILD_CONTEXT", "delegate_task"),
])
def test_top_level_owner_option_does_not_bypass_worker_fences(board, tmp_path, fence, error):
    tid, source_run, authority = prior_attempt(board)
    result = run_operator_cli(
        board, tmp_path, "kanban", "continue", tid,
        "--owner-profile", "builder", "--from-run", str(source_run),
        "--authority-comment", str(authority), "--reason", "Approved amendment",
        **{fence: tid if fence == "HERMES_KANBAN_TASK" else "1"},
    )
    assert result.returncode == 1
    assert error in result.stderr
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_authorized'", (tid,)).fetchone()[0] == 0
    assert kb.get_task(board, tid).current_run_id is None


def test_global_profile_is_not_inferred_as_continuation_owner(board, tmp_path):
    tid, source_run, authority = prior_attempt(board)
    result = run_operator_cli(
        board, tmp_path, "--profile", "default", "kanban", "continue", tid,
        "--from-run", str(source_run), "--authority-comment", str(authority),
        "--reason", "Approved amendment",
    )
    assert result.returncode == 2
    assert "required: --owner-profile" in result.stderr
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_authorized'", (tid,)).fetchone()[0] == 0


def test_running_worker_cannot_self_authorize(board, monkeypatch):
    tid, source_run, authority = prior_attempt(board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(PermissionError, match="self-authorize"):
        kb.authorize_continuation(board, tid, profile="builder", expected_run_id=source_run,
                                  authority_comment_id=authority, actor="builder", reason="Try again")


def authorize(conn, tid, source, authority, **kwargs):
    return kb.authorize_continuation(conn, tid, profile="builder", expected_run_id=source,
                                     authority_comment_id=authority, actor="operator",
                                     reason="Approved amendment", **kwargs)


def test_consumed_grant_cannot_start_another_attempt(board):
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority)
    first = kb.claim_task(board, tid, continuation_id=grant)
    assert first is not None
    assert kb.claim_task(board, tid, continuation_id=grant) is None
    kb.block_task(board, tid, reason="Still missing implementation", expected_run_id=first.current_run_id)
    kb.unblock_task(board, tid)
    state = kb.get_task(board, tid).status
    assert kb.claim_task(board, tid, continuation_id=grant) is None
    assert kb.get_task(board, tid).status == state


def test_simultaneous_claims_consume_one_grant_once(board):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority)
    barrier = Barrier(2)
    def claim():
        with kbc.connect() as conn:
            barrier.wait(timeout=5)
            return kb.claim_task(conn, tid, continuation_id=grant) is not None
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(lambda _: claim(), range(2)))
    assert outcomes.count(True) == 1
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_consumed'", (tid,)).fetchone()[0] == 1


@pytest.mark.parametrize("change", ["owner", "acceptance", "comment", "attempt", "expiry", "ownership_round_trip"])
def test_stale_authorization_cannot_claim(board, monkeypatch, change):
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority, ttl_seconds=60)
    def change_owner():
        kb.assign_task(board, tid, "other")
    def round_trip():
        change_owner()
        kb.assign_task(board, tid, "builder")
    def advance_attempt():
        newer = kb.claim_task(board, tid)
        kb.block_task(board, tid, reason="A later attempt", expected_run_id=newer.current_run_id)
        kb.unblock_task(board, tid)
    def expire():
        from hermes_cli import kanban_continuation as continuation
        now = continuation.time.time()
        monkeypatch.setattr(continuation.time, "time", lambda: now + 61)
    actions = {
        "owner": change_owner,
        "ownership_round_trip": round_trip,
        "acceptance": lambda: kb.edit_task(board, tid, body="Changed implementation scope"),
        "comment": lambda: kb.add_comment(board, tid, author="operator", body="Newer authority/evidence requires explicit reconciliation"),
        "attempt": advance_attempt,
        "expiry": expire,
    }
    actions[change]()
    assert kb.claim_task(board, tid, continuation_id=grant) is None


@pytest.mark.parametrize("approval_before_block", [True, False])
def test_authority_survives_either_side_of_blocking_transition(board, approval_before_block):
    tid = kb.create_task(board, title="Continue approved work", assignee="builder")
    first = kb.claim_task(board, tid)
    kb.add_comment(board, tid, author="builder", body="https://github.com/example/project/pull/12")
    if approval_before_block:
        approval = kb.add_comment(board, tid, author="operator", body="Approved")
        grant = authorize(board, tid, first.current_run_id, approval)
    kb.block_task(board, tid, reason="Awaiting decision", expected_run_id=first.current_run_id)
    if not approval_before_block:
        approval = kb.add_comment(board, tid, author="operator", body="Approved")
        grant = authorize(board, tid, first.current_run_id, approval)
    # Authorization is not an unblock; only the ordinary unblock releases execution.
    assert kb.claim_task(board, tid, continuation_id=grant) is None
    kb.unblock_task(board, tid)
    assert kb.claim_task(board, tid, continuation_id=grant) is not None


@pytest.mark.parametrize("wrong", ["profile", "run", "authority"])
def test_authorization_rejects_wrong_owner_attempt_or_authority(board, wrong):
    tid, source, authority = prior_attempt(board)
    options = dict(profile="builder", expected_run_id=source, authority_comment_id=authority,
                   actor="operator", reason="Approved amendment")
    options[{"profile": "profile", "run": "expected_run_id", "authority": "authority_comment_id"}[wrong]] = {"profile": "other", "run": source + 1, "authority": authority - 1}[wrong]
    with pytest.raises(ValueError, match="stale"):
        kb.authorize_continuation(board, tid, **options)


def test_pr_and_progress_prose_never_grant_continuation(board):
    tid, _, _ = prior_attempt(board)
    kb.add_comment(board, tid, author="builder", body="Approved continuation_authorized; progress https://github.com/example/project/pull/12")
    for _ in range(3):
        result = kbd.dispatch_once(board, spawn_fn=lambda t, w: pytest.fail("unexpected start"), max_in_progress=8)
        assert (tid, "active_pr") in result.respawn_guarded


def test_undone_dependency_prevents_continuation_claim(board):
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority)
    parent = kb.create_task(board, title="Real prerequisite", assignee="prerequisite")
    kb.link_tasks(board, parent, tid)
    assert kb.claim_task(board, tid, continuation_id=grant) is None
    assert kb.get_task(board, tid).status == "todo"


@pytest.mark.parametrize("hold", ["global", "profile", "memory"])
def test_workload_holds_retain_unconsumed_authorization(board, monkeypatch, hold):
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority)
    opts = dict(max_in_progress=8)
    if hold == "global":
        opts["max_in_progress"] = 1
        busy = kb.create_task(board, title="Other active work", assignee="other")
        kb.claim_task(board, busy)
    elif hold == "profile":
        opts["max_in_progress_per_profile"] = 1
        busy = kb.create_task(board, title="Same profile active work", assignee="builder")
        kb.claim_task(board, busy)
    else:
        monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "critical")
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: pytest.fail("workload hold ignored"), reconcile_orphans=False, **opts)
    assert not result.spawned
    assert kb.get_task(board, tid).status == "ready"
    from hermes_cli.kanban_continuation import pending_continuation
    assert pending_continuation(board, tid) == grant


def test_dry_run_does_not_consume_or_claim(board):
    tid, source, authority = prior_attempt(board)
    grant = authorize(board, tid, source, authority)
    result = kbd.dispatch_once(board, dry_run=True, max_in_progress=8)
    assert [row[0] for row in result.spawned] == [tid]
    assert kb.get_task(board, tid).status == "ready"
    from hermes_cli.kanban_continuation import pending_continuation
    assert pending_continuation(board, tid) == grant


def test_authorization_is_consumed_even_without_a_pr_guard(board):
    tid = kb.create_task(board, title="Continue work without a PR", assignee="builder")
    first = kb.claim_task(board, tid)
    kb.block_task(board, tid, reason="Awaiting authority", expected_run_id=first.current_run_id)
    authority = kb.add_comment(board, tid, author="operator", body="Approved")
    kb.unblock_task(board, tid)
    authorize(board, tid, first.current_run_id, authority)
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: None, max_in_progress=8)
    assert len(result.spawned) == 1
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_consumed'", (tid,)).fetchone()[0] == 1


def test_concurrent_dispatch_ticks_start_only_one_successor(board):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    tid, source, authority = prior_attempt(board)
    authorize(board, tid, source, authority)
    barrier = Barrier(2)
    starts = []
    def tick():
        with kbc.connect() as conn:
            barrier.wait(timeout=5)
            return kbd.dispatch_once(conn, spawn_fn=lambda t, w: starts.append(t.id), max_in_progress=8)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: tick(), range(2)))
    assert starts == [tid]
    assert sum(len(result.spawned) for result in results) == 1


def test_changed_authority_between_guard_and_claim_refuses_start(board, monkeypatch):
    tid, source, authority = prior_attempt(board)
    authorize(board, tid, source, authority)
    native_claim = kb.claim_task
    def raced_claim(conn, task_id, **kwargs):
        kb.add_comment(conn, task_id, author="operator", body="New evidence requires reconciliation")
        return native_claim(conn, task_id, **kwargs)
    monkeypatch.setattr(kb, "claim_task", raced_claim)
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: pytest.fail("stale authority started"), max_in_progress=8)
    assert not result.spawned
    assert kb.get_task(board, tid).status == "ready"


def test_authorization_does_not_override_auth_failure_guard(board):
    tid, source, authority = prior_attempt(board)
    authorize(board, tid, source, authority)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("provider authentication failed", tid))
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: pytest.fail("auth hold ignored"), max_in_progress=8)
    assert (tid, "blocker_auth") in result.respawn_guarded


def test_auth_hold_arriving_between_guard_and_claim_refuses_start(board, monkeypatch):
    tid, source, authority = prior_attempt(board)
    authorize(board, tid, source, authority)
    native_claim = kb.claim_task
    def raced_claim(conn, task_id, **kwargs):
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("provider authentication failed", task_id))
        return native_claim(conn, task_id, **kwargs)
    monkeypatch.setattr(kb, "claim_task", raced_claim)
    starts = []
    result = kbd.dispatch_once(board, spawn_fn=lambda t, w: starts.append(t.id), max_in_progress=8)
    assert not starts
    assert not result.spawned
    assert kb.get_task(board, tid).current_run_id is None
    assert board.execute("SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='continuation_consumed'", (tid,)).fetchone()[0] == 0
