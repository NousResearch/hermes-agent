"""Quota cooldown uses failed execution provenance, not observation time."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch

QUOTA = "pid 12345 not alive Worker's last output: 'HTTP 429: The usage limit has been reached'"
FAILED_AT = 5_000_000


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setattr(dispatch.time, "time", lambda: FAILED_AT - 60)
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def fail(conn, monkeypatch, tid, outcome="crashed", error=QUOTA, at=FAILED_AT):
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    monkeypatch.setattr(dispatch.time, "time", lambda: at)
    dispatch._record_task_failure(conn, tid, error=error, outcome=outcome,
                                  release_claim=True, end_run=True, failure_limit=20)
    return claimed.current_run_id


@pytest.mark.parametrize("elapsed,reason", [(299, "infrastructure_cooldown"), (300, None)])
def test_infrastructure_cooldown_is_preserved(board, monkeypatch, elapsed, reason):
    tid = kb.create_task(board, title="host refused spawn", assignee="worker")
    assert kb.claim_task(board, tid) is not None
    monkeypatch.setattr(dispatch.time, "time", lambda: FAILED_AT)
    dispatch._record_task_failure(
        board, tid, error="host scope unavailable", outcome="spawn_failed",
        release_claim=True, end_run=True, infrastructure=True,
    )
    monkeypatch.setattr(dispatch.time, "time", lambda: FAILED_AT + elapsed)
    assert dispatch.check_respawn_guard(board, tid) == reason
    assert kb.get_task(board, tid).consecutive_failures == 0


@pytest.mark.parametrize("outcome", ["crashed", "rate_limited"])
@pytest.mark.parametrize("administrative", [False, True])
@pytest.mark.parametrize("lane", ["ready", "review"])
def test_quota_deadline_survives_administrative_history(board, monkeypatch, outcome, administrative, lane):
    tid = kb.create_task(board, title="quota", assignee="worker")
    rid = fail(board, monkeypatch, tid, outcome)
    if administrative:
        monkeypatch.setattr(dispatch.time, "time", lambda: FAILED_AT + 200)
        kb.block_task(board, tid, reason="operator hold", kind="needs_input")
    # Historical ready/review rows can retain a failure after administrative history.
    board.execute("UPDATE tasks SET status=? WHERE id=?", (lane, tid))
    board.commit()
    before = tuple(board.execute("SELECT consecutive_failures,last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone())
    for elapsed in (100, 299, 300, 600):
        monkeypatch.setattr(dispatch.time, "time", lambda: FAILED_AT + elapsed)
        expected = "rate_limit_cooldown" if elapsed < 300 else None
        assert dispatch.check_respawn_guard(board, tid, lane=lane) == expected
    assert tuple(board.execute("SELECT consecutive_failures,last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone()) == before
    assert board.execute("SELECT ended_at FROM task_runs WHERE id=?", (rid,)).fetchone()[0] == FAILED_AT


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("outcome", ["crashed", "rate_limited"])
@pytest.mark.parametrize("case,reason", [
    ("expired", None), ("admin_expired", None), ("admin_recent", "rate_limit_cooldown"),
    ("recent", "rate_limit_cooldown"),
    ("fresh_retry", "rate_limit_cooldown"),
    ("missing_auth", "blocker_auth"), ("revoked_auth", "blocker_auth"),
    ("permission", "blocker_auth"), ("mixed_auth", "blocker_auth"),
    ("new_auth", "blocker_auth"), ("same_second_auth", "blocker_auth"),
    ("other_model_success", "rate_limit_cooldown"),
    ("null_time", "blocker_auth_unknown_provenance"),
    ("invalid_time", "blocker_auth_unknown_provenance"),
    ("zero_time", "blocker_auth_unknown_provenance"),
    ("future_time", "rate_limit_cooldown"),
    ("no_run", "blocker_auth"), ("unmatched_error", "blocker_auth"),
    ("ambiguous_admin", "blocker_auth"), ("claimed_block", "blocker_auth"),
    ("blocked", None), ("manual_unblock", None),
    ("pr", "active_pr"), ("success", "recent_success"),
    ("disabled_pr", "active_pr"),
])
def test_native_dispatch_claims_only_eligible_quota_retries(board, monkeypatch, lane, outcome, case, reason):
    import hermes_cli.config as config
    import hermes_cli.profiles as profiles
    monkeypatch.setattr(config, "load_config", lambda *a, **kw: {"kanban": {"review_dispatch": True}})
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    tid = kb.create_task(board, title=case, assignee="worker", max_retries=20)
    error = {
        "missing_auth": "No Codex credentials stored",
        "revoked_auth": "revoked credentials",
        "permission": "permission denied",
        "mixed_auth": QUOTA + " invalid API key",
    }.get(case, QUOTA)
    # A completed run remains an independent guard even before a later quota failure.
    if case == "success":
        board.execute("INSERT INTO task_runs(task_id,profile,status,outcome,started_at,ended_at) VALUES (?,'worker','done','completed',?,?)", (tid, FAILED_AT - 120, FAILED_AT - 90))
        board.commit()
    rid = fail(board, monkeypatch, tid, outcome=outcome, error=error)
    now = FAILED_AT + 600
    if case in ("recent", "admin_recent", "other_model_success"):
        now = FAILED_AT + 100
    if case == "fresh_retry":
        fail(board, monkeypatch, tid, outcome=outcome, at=FAILED_AT + 550)
    if case in ("new_auth", "same_second_auth"):
        fail(board, monkeypatch, tid, error="invalid API key", at=FAILED_AT + (0 if case == "same_second_auth" else 10))
    if case == "other_model_success":
        other = kb.create_task(board, title="other model", assignee="worker", model_override="other-model")
        kb.claim_task(board, other)
        kb.complete_task(board, other, summary="different route succeeded")
    for key, value in (("null_time", None), ("invalid_time", "unknown"), ("zero_time", 0), ("future_time", now + 60)):
        if case == key:
            board.execute("UPDATE task_runs SET ended_at=? WHERE id=?", (value, rid))
    if case == "no_run":
        board.execute("DELETE FROM task_events WHERE task_id=?", (tid,))
        board.execute("DELETE FROM task_runs WHERE task_id=?", (tid,))
    if case == "unmatched_error":
        board.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (QUOTA + " newer", tid))
    board.commit()
    if case in ("admin_expired", "admin_recent", "ambiguous_admin", "claimed_block", "blocked", "manual_unblock"):
        if case == "claimed_block":
            kb.claim_task(board, tid)
        kb.block_task(board, tid, reason="owner action needed", kind="needs_input")
        if case == "ambiguous_admin":
            # Missing end timestamp must not be mistaken for an ignorable admin row.
            board.execute("UPDATE task_runs SET ended_at=NULL WHERE id=(SELECT MAX(id) FROM task_runs WHERE task_id=?)", (tid,))
        if case == "manual_unblock":
            kb.unblock_task(board, tid)
    if case in ("pr", "disabled_pr"):
        kb.add_comment(board, tid, author="worker", body="Opened https://github.com/example/repo/pull/123")
    if case == "disabled_pr":
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
        now = FAILED_AT + 1
    if case != "blocked":
        board.execute("UPDATE tasks SET status=? WHERE id=?", (lane, tid))
    board.commit()
    monkeypatch.setattr(dispatch.time, "time", lambda: now)
    if lane == "review" and reason in ("active_pr", "recent_success"):
        reason = None
    counters = tuple(board.execute("SELECT consecutive_failures,last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone())
    history = [tuple(r) for r in board.execute("SELECT id,outcome,error,ended_at FROM task_runs WHERE task_id=? ORDER BY id", (tid,))]
    calls = []
    def spawn(task, workspace, **kwargs):
        assert Path(workspace).is_dir()
        assert kb.get_task(board, tid).status == "running"
        calls.append(task)
    result = dispatch.dispatch_once(board, spawn_fn=spawn, max_spawn=1, max_in_progress=10,
                                    failure_limit=20, reconcile_orphans=False)
    should_claim = reason is None and case != "blocked"
    assert (tid in [s[0] for s in result.spawned]) == should_claim
    assert len(calls) == int(should_claim)
    assert dict(result.respawn_guarded).get(tid) == reason
    assert tuple(board.execute("SELECT consecutive_failures,last_failure_error FROM tasks WHERE id=?", (tid,)).fetchone()) == counters
    assert [tuple(r) for r in board.execute("SELECT id,outcome,error,ended_at FROM task_runs WHERE task_id=? ORDER BY id", (tid,))][:len(history)] == history
    current = kb.get_task(board, tid)
    assert current.status == ("running" if should_claim else "blocked" if case == "blocked" else lane)
    if should_claim:
        assert current.current_run_id != rid
        events = kb.list_events(board, tid)
        claim = [e for e in events if e.kind == "claimed"][-1]
        assert claim.run_id == current.current_run_id
        if lane == "review":
            assert claim.payload["source_status"] == "review"
