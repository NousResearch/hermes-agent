"""Machine acceptance: declared checks turn completion into a submission the dispatcher decides."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import types
from pathlib import Path

import pytest

from hermes_cli import kanban_acceptance as ka
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_acceptance as kba
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_pr_acceptance as pra

GIT = "/usr/bin/git"
needs_git = pytest.mark.skipif(not os.access(GIT, os.X_OK), reason="needs /usr/bin/git")


@pytest.fixture(autouse=True)
def _current_modules():
    """Other test files purge ``hermes_cli`` from sys.modules; bind the live module objects."""
    import importlib
    for alias, name in (("ka", "hermes_cli.kanban_acceptance"), ("kb", "hermes_cli.kanban_db"),
                        ("kba", "hermes_cli.kanban_db_acceptance"), ("kbc", "hermes_cli.kanban_db_connect"),
                        ("kbd", "hermes_cli.kanban_db_dispatch"), ("pra", "hermes_cli.kanban_pr_acceptance")):
        globals()[alias] = importlib.import_module(name)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _exited_worker(conn, tid, task):
    """Register a real worker process on the claimed run, then let it exit (its exit is provable)."""
    proc = subprocess.Popen(["sleep", "30"])
    kbd._set_worker_pid(conn, tid, proc.pid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    proc.kill()
    proc.wait()
    return task


def _claim(conn, tid, claimer=None):
    return _exited_worker(conn, tid, kb.claim_task(conn, tid, claimer=claimer))


def _claimed(conn, acceptance, *, assignee="builder", claimer=None, exited=True, **kw):
    tid = kb.create_task(conn, title="t", assignee=assignee, acceptance=acceptance, **kw)
    task = kb.claim_task(conn, tid, claimer=claimer)
    return tid, (_exited_worker(conn, tid, task) if exited else task)


def _submitted(conn, acceptance, **kw):
    tid, task = _claimed(conn, acceptance, **kw)
    assert kb.complete_task(conn, tid, summary="did it", expected_run_id=task.current_run_id)
    return tid


def _subs(conn, tid):
    return conn.execute("SELECT * FROM task_acceptance_submissions WHERE task_id = ? ORDER BY id", (tid,)).fetchall()


def _kinds(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _due_now(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE task_acceptance_submissions SET next_check_at = ? WHERE task_id = ? AND state = 'open'",
                     (int(time.time()), tid))


def _file_spec(path, exists=True):
    return {"kind": "file", "path": str(path), "exists": exists}


# --------------------------------------------------------------------------- legacy / pass / fail

def test_legacy_card_without_acceptance_completes_unchanged(conn):
    tid, task = _claimed(conn, None)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert kb.get_task(conn, tid).status == "done"
    assert _subs(conn, tid) == []
    assert not [e for e in kb.list_events(conn, tid) if e.kind.startswith("acceptance")]
    assert "acceptance" not in _kinds(conn, tid, "created")[0].payload
    assert kb.latest_run(conn, tid).outcome == "completed"


def test_pass_finalizes_once_with_hooks_and_dependants(conn, tmp_path, monkeypatch):
    hooks = []
    monkeypatch.setattr(kb, "_fire_task_hook", lambda event, *a, **k: event == "kanban_task_completed" and hooks.append(event))
    (tmp_path / "out.txt").write_text("x")
    tid = _submitted(conn, _file_spec(tmp_path / "out.txt"))
    child = kb.create_task(conn, title="child", assignee="builder", parents=[tid])
    task = kb.get_task(conn, tid)
    assert task.status == "review" and task.current_run_id is None
    assert kb.latest_run(conn, tid).outcome == "acceptance_submitted"
    assert _kinds(conn, tid, "completed") == [] and hooks == []
    assert kb.get_task(conn, child).status == "todo"

    assert [o["verdict"] for o in ka.run_acceptance_pass(conn)] == ["pass"]
    assert ka.run_acceptance_pass(conn) == []
    assert kb.get_task(conn, tid).status == "done"
    assert len(_kinds(conn, tid, "completed")) == 1 and hooks == ["kanban_task_completed"]
    assert kb.latest_run(conn, tid).outcome == "completed"
    assert kb.get_task(conn, child).status == "ready"
    verdict = _kinds(conn, tid, "acceptance_verdict")[-1].payload
    assert verdict["verdict"] == "pass" and verdict["results"][0]["receipt"]["observed_at"]
    assert _subs(conn, tid)[0]["state"] == "passed"


def test_fail_requeues_to_implementer_with_context_and_counts_once(conn, tmp_path):
    missing = tmp_path / "missing.pdf"
    tid = _submitted(conn, _file_spec(missing))
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "fail"
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee, task.consecutive_failures) == ("ready", "builder", 1)
    assert not kbd._RESPAWN_BLOCKER_RE.search(task.last_failure_error)
    assert kbd.check_respawn_guard(conn, tid) is None
    context = kb.build_worker_context(conn, tid)
    assert "acceptance_failed" in context and f"expected {missing} to exist" in context
    assert ka.run_acceptance_pass(conn) == []
    assert kb.get_task(conn, tid).consecutive_failures == 1

    # Repair and resubmit: a fresh submission, decided on its own.
    missing.write_text("now")
    again = _claim(conn, tid)
    assert kb.complete_task(conn, tid, summary="fixed", expected_run_id=again.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    assert [s["state"] for s in _subs(conn, tid)] == ["failed", "passed"]
    assert kb.get_task(conn, tid).consecutive_failures == 0


def test_fail_at_breaker_limit_blocks(conn, tmp_path):
    tid = _submitted(conn, _file_spec(tmp_path / "nope"), max_retries=1)
    ka.run_acceptance_pass(conn)
    assert kb.get_task(conn, tid).status == "blocked"
    assert _kinds(conn, tid, "gave_up")[-1].payload["trigger_outcome"] == "acceptance_failed"


def test_artifacts_staged_at_submission_and_workspace_kept_until_pass(conn, tmp_path):
    from hermes_cli import kanban_db_workspace as kbw
    tid, task = _claimed(conn, _file_spec(tmp_path / "gate"))
    ws = kbw.resolve_workspace(task)
    kbw.set_workspace_path(conn, tid, ws)
    (ws / "report.txt").write_text("r")
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id,
                            metadata={"artifacts": [str(ws / "report.txt")]})
    assert [a.filename for a in kb.list_attachments(conn, tid)] == ["report.txt"]
    ka.run_acceptance_pass(conn)
    assert kb.get_task(conn, tid).status == "ready" and (ws / "report.txt").exists()
    (tmp_path / "gate").write_text("x")
    again = _claim(conn, tid)
    assert kb.complete_task(conn, tid, summary="s2", expected_run_id=again.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    assert not ws.exists()
    assert Path(kb.list_attachments(conn, tid)[0].stored_path).exists()


# --------------------------------------------------------------------------- pending / tbd / machine wait

def test_tbd_and_pending_never_done_never_reviewed_no_hot_loop(conn, tmp_path, monkeypatch, all_assignees_spawnable):
    tbd = _submitted(conn, {"kind": "tbd"})
    pending = _submitted(conn, {"kind": "git_commit", "repo": str(tmp_path), "branch": "b", "base": "a" * 40})
    monkeypatch.setattr(ka, "evaluate_in_child", ka.evaluate_job)  # the stub below lives in this process
    monkeypatch.setattr(ka, "evaluate_git_commit", lambda spec, **kw: ka._result("pending", "infra"))
    assert ka.run_acceptance_pass(conn) == [{"task_id": pending, "submission_id": 2, "verdict": "pending", "reason": "infra"}]
    sub = _subs(conn, pending)[0]
    assert sub["next_check_at"] >= int(time.time()) + ka.BACKOFF_FIRST - 1
    assert _subs(conn, tbd)[0]["next_check_at"] is None
    assert ka.run_acceptance_pass(conn) == []  # neither is due: no hot loop
    _due_now(conn, pending)
    ka.run_acceptance_pass(conn)
    assert len(_kinds(conn, pending, "acceptance_verdict")) == 1  # unchanged reason: no event spam
    assert _subs(conn, pending)[0]["attempts"] == 2

    spawned = []
    res = kbd.dispatch_once(conn, spawn_fn=lambda t, ws, **k: spawned.append(t.id))
    assert spawned == [] and res.spawned == []
    assert not kbd.has_spawnable_review(conn)
    assert kbd._lane_rows(conn, "review") == []
    assert kb.claim_review_task(conn, tbd) is None
    assert {kb.get_task(conn, t).status for t in (tbd, pending)} == {"review"}
    assert kb.get_task(conn, tbd).consecutive_failures == 0


def test_public_complete_on_open_submission_is_refused_even_forced(conn, tmp_path):
    tid = _submitted(conn, {"kind": "tbd"})
    assert kb.complete_task(conn, tid, summary="again") is False
    assert kb.complete_task(conn, tid, summary="again", force=True) is False
    assert len(_subs(conn, tid)) == 1 and kb.get_task(conn, tid).status == "review"


def test_force_complete_on_live_claim_still_submits(conn, tmp_path):
    tid, task = _claimed(conn, _file_spec(tmp_path), exited=False)
    proc = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, proc.pid, run_id=task.current_run_id, claim_lock=task.claim_lock)
        with pytest.raises(kb.LiveClaimError):
            kb.complete_task(conn, tid, summary="s")
        assert kb.complete_task(conn, tid, summary="s", force=True)
        assert kba.is_machine_wait(conn, tid)
        assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_alive"
    finally:
        proc.kill()
        proc.wait()


def test_dashboard_done_submits(conn, tmp_path):
    import importlib.util
    pytest.importorskip("fastapi")
    path = Path(__file__).resolve().parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_dash_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tid = kb.create_task(conn, title="t", assignee="builder", acceptance={"kind": "tbd"})
    assert mod._STATUS_HANDLERS["done"](conn, tid, types.SimpleNamespace(result="r", summary="s", metadata=None))
    assert kba.is_machine_wait(conn, tid)


def test_cli_force_complete_submits_and_edit_rebinds(tmp_path, monkeypatch, conn):
    from hermes_cli import kanban as kc
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    target = tmp_path / "made"
    out = kc.run_slash(f"create cli-card --assignee builder --acceptance '{json.dumps({'kind': 'tbd'})}' --json")
    tid = json.loads(out)["id"]
    assert "Submitted" in kc.run_slash(f"complete {tid} --summary done --force")
    assert "cannot complete" in kc.run_slash(f"complete {tid} --summary done --force")
    target.write_text("x")
    kc.run_slash(f"edit {tid} --acceptance '{json.dumps(_file_spec(target))}'")
    with kbc.connect_closing() as c:
        assert ka.run_acceptance_pass(c)[0]["verdict"] == "pass"
        assert kb.get_task(c, tid).status == "done"
    assert "acceptance" in kc.run_slash("create bad --assignee b --acceptance '{{\"kind\":\"shell\"}}'")


# --------------------------------------------------------------------------- worker exit gate

def test_origin_worker_alive_then_exited(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"), exited=False)
    proc = subprocess.Popen(["sleep", "30"])
    kbd._set_worker_pid(conn, tid, proc.pid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert kb.get_task(conn, tid).current_run_id is None  # not proof of exit
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_alive"
    proc.kill()
    proc.wait()
    _due_now(conn, tid)
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"


def test_remote_worker_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"), claimer="some-other-host:42", exited=False)
    kbd._set_worker_pid(conn, tid, os.getpid(), run_id=task.current_run_id, claim_lock=task.claim_lock)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_remote"


# --------------------------------------------------------------------------- CAS / staleness / concurrency

def test_stale_run_does_not_submit(conn, tmp_path):
    tid, task = _claimed(conn, {"kind": "tbd"})
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id + 99) is False
    assert _subs(conn, tid) == [] and kb.get_task(conn, tid).status == "running"


def _leased(conn, tid, owner="A"):
    sub_id = _subs(conn, tid)[-1]["id"]
    lease = kba.lease_submission(conn, sub_id, owner, lease_seconds=90)
    assert lease and not lease.get("cancelled")
    return lease


def test_spec_edit_mid_evaluation_loses_cas_then_new_spec_decides(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = _submitted(conn, _file_spec(tmp_path / "nope"))
    lease = _leased(conn, tid)
    old = [ka.evaluate_file(_file_spec(tmp_path / "nope"))]
    assert kba.set_acceptance(conn, tid, _file_spec(tmp_path / "f"))
    assert kba.apply_verdict(conn, lease, old) is None
    assert kb.get_task(conn, tid).status == "review" and _subs(conn, tid)[0]["state"] == "open"
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    assert _kinds(conn, tid, "acceptance_updated")[0].payload["submission_id"] == 1


def test_reassign_ends_machine_wait_for_the_new_owner(conn, tmp_path):
    tid = _submitted(conn, _file_spec(tmp_path / "nope"))
    lease = _leased(conn, tid)
    assert kb.assign_task(conn, tid, "someone-else")
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee) == ("ready", "someone-else")
    assert _subs(conn, tid)[0]["state"] == "cancelled"
    assert kba.apply_verdict(conn, lease, [ka.evaluate_file(_file_spec(tmp_path / "nope"))]) is None
    assert kbd.check_respawn_guard(conn, tid) is None


def test_contract_change_mid_evaluation_cancels(conn, tmp_path):
    tid = _submitted(conn, _file_spec(tmp_path / "nope"))
    lease = _leased(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET completion_contract = 'acme/other' WHERE id = ?", (tid,))
    assert kba.apply_verdict(conn, lease, [ka.evaluate_file(_file_spec(tmp_path / "nope"))])["reason"] == "spec_changed"
    assert _subs(conn, tid)[0]["state"] == "cancelled" and kb.get_task(conn, tid).consecutive_failures == 0
    assert kb.get_task(conn, tid).status == "ready"  # not left as an orphan review card


def test_old_receipt_cannot_close_new_submission(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = _submitted(conn, _file_spec(tmp_path / "f"))
    lease = _leased(conn, tid)
    assert kb.reopen_review_task(conn, tid)
    task = _claim(conn, tid)
    assert kb.complete_task(conn, tid, summary="second", expected_run_id=task.current_run_id)
    assert [s["state"] for s in _subs(conn, tid)] == ["cancelled", "open"]
    assert kba.apply_verdict(conn, lease, [ka.evaluate_file(_file_spec(tmp_path / "f"))]) is None
    assert kb.get_task(conn, tid).status == "review"


def test_parent_reopened_mid_evaluation(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    parent = kb.create_task(conn, title="p", assignee="builder")
    assert kb.complete_task(conn, parent, summary="p", force=True)
    tid = kb.create_task(conn, title="c", assignee="builder", parents=[parent], acceptance=_file_spec(tmp_path / "f"))
    task = _claim(conn, tid)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    lease = _leased(conn, tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))  # reopened without invalidation
    assert kba.apply_verdict(conn, lease, [ka.evaluate_file(_file_spec(tmp_path / "f"))])["reason"] == "parents_unsatisfied"
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee) == ("todo", "builder") and _subs(conn, tid)[0]["state"] == "cancelled"


def test_parent_reopen_invalidation_cancels_and_returns_to_implementer(conn, tmp_path):
    parent = kb.create_task(conn, title="p", assignee="builder")
    assert kb.complete_task(conn, parent, summary="p", force=True)
    tid = kb.create_task(conn, title="c", assignee="builder", parents=[parent], acceptance={"kind": "tbd"})
    task = _claim(conn, tid)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))
        kb.invalidate_descendants_for_parent_reopen(conn, parent, author="op")
    assert _subs(conn, tid)[0]["state"] == "cancelled"
    assert kb.complete_task(conn, parent, summary="p2", force=True)
    assert kb.get_task(conn, tid).status == "ready"


def test_concurrent_passes_apply_once(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = _submitted(conn, _file_spec(tmp_path / "f"))
    other = kbc.connect(Path(conn.execute("PRAGMA database_list").fetchone()[2]))
    try:
        lease_a = _leased(conn, tid, "A")
        assert kba.lease_submission(other, 1, "B", lease_seconds=90) is None  # held
        with kb.write_txn(conn):  # A's lease expires while it is still evaluating
            conn.execute("UPDATE task_acceptance_submissions SET lease_until = 0")
        assert [o["verdict"] for o in ka.run_acceptance_pass(other)] == ["pass"]
        assert kba.apply_verdict(conn, lease_a, [ka.evaluate_file(_file_spec(tmp_path / "f"))]) is None
    finally:
        other.close()
    assert len(_kinds(conn, tid, "completed")) == 1


def test_estop_stops_the_pass(conn, tmp_path, monkeypatch):
    tid = _submitted(conn, _file_spec(tmp_path / "nope"))
    import gateway.kanban_watchers_common as common
    monkeypatch.setattr(common, "_kanban_dispatch_allowed", lambda: False)
    assert ka.run_acceptance_pass(conn) == []
    assert kba.is_machine_wait(conn, tid)


# --------------------------------------------------------------------------- review provenance

def test_review_approval_failure_returns_to_original_implementer(conn, tmp_path):
    tid, task = _claimed(conn, _file_spec(tmp_path / "nope"))
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    review = _exited_worker(conn, tid, kb.claim_review_task(conn, tid))
    assert review.assignee == "reviewer"
    assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
    assert _subs(conn, tid)[0]["implementer"] == "builder"
    ka.run_acceptance_pass(conn)
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee) == ("ready", "builder")


def test_review_approval_without_provenance_is_refused(conn):
    tid = kb.create_task(conn, title="t", assignee="reviewer", acceptance={"kind": "tbd"})
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    with pytest.raises(kb.AcceptanceSubmissionError):
        kb.complete_task(conn, tid, summary="ok")
    assert _subs(conn, tid) == [] and kb.get_task(conn, tid).status == "review"


# --------------------------------------------------------------------------- validation

@pytest.mark.parametrize("bad", [
    "file", ["kind", "file"], {"kind": "shell", "cmd": "rm -rf /"}, {"kind": "file"},
    {"kind": "file", "path": "relative/x"}, {"kind": "file", "path": "/a/../etc"},
    {"kind": "file", "path": "/a", "exists": 1}, {"kind": "file", "path": "/a", "extra": True},
    {"kind": "file", "path": "/a\x00b"}, {"kind": "file", "path": "//net/x"},
    {"kind": "tbd", "path": "/x"}, {"kind": "pr_checks", "pr_url": "https://github.com/a/b/pull/1/files"},
    {"kind": "git_commit", "repo": "/r", "branch": "main", "base": "A" * 40},
    {"kind": "git_commit", "repo": "/r", "branch": "main", "base": "abc"},
    *({"kind": "git_commit", "repo": "/r", "branch": b, "base": "a" * 40} for b in (
        "-x", "--upload-pack=evil", "a..b", "a@{1}", "a b", "x.lock", "a/", ".hidden", "a\\b", "a:b", "HEAD", "", 5)),
])
def test_strict_validation_rejects(conn, bad):
    with pytest.raises(ValueError):
        kb.create_task(conn, title="t", assignee="b", acceptance=bad)
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_cli_parse_rejects_non_json_constants_and_accepts_none():
    with pytest.raises(ValueError):
        ka.parse_acceptance_arg('{"kind":"file","path":"/a","exists":NaN}')
    assert ka.parse_acceptance_arg("none") is None
    assert json.loads(ka.parse_acceptance_arg('{"path":"/a/./b","kind":"file"}')) == {
        "kind": "file", "path": "/a/b", "exists": True}


def test_tool_create_rejects_malformed_and_stores_valid(monkeypatch, tmp_path):
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    bad = json.loads(kt._handle_create({"title": "t", "assignee": "b", "acceptance": {"kind": "file", "path": "x"}}))
    assert "absolute" in bad["error"]
    ok = json.loads(kt._handle_create({"title": "t", "assignee": "b", "acceptance": {"kind": "tbd"}}))
    with kbc.connect_closing() as c:
        assert kb.get_task(c, ok["task_id"]).acceptance == '{"kind":"tbd"}'
        assert c.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


# --------------------------------------------------------------------------- git_commit

def _git(repo, *args):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(repo), "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run([GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "commit", "-q", "--allow-empty", "-m", "base")
    base = _git(r, "rev-parse", "HEAD")
    _git(r, "checkout", "-q", "-b", "feature")
    _git(r, "commit", "-q", "--allow-empty", "-m", "work")
    _git(r, "checkout", "-q", "--orphan", "unrelated")
    _git(r, "commit", "-q", "--allow-empty", "-m", "other")
    _git(r, "checkout", "-q", "main")
    return r, base


def _commit_spec(repo, branch, base):
    return {"kind": "git_commit", "repo": str(repo), "branch": branch, "base": base}


@needs_git
@pytest.mark.parametrize("branch,verdict,reason", [
    ("feature", "pass", "descends_from_base"), ("main", "fail", "no_new_commit"),
    ("unrelated", "fail", "not_descendant"), ("ghost", "fail", "branch_missing"),
])
def test_git_commit_verdicts(repo, branch, verdict, reason):
    r, base = repo
    got = ka.evaluate_git_commit(_commit_spec(r, branch, base), deadline=time.monotonic() + 30)
    assert (got["verdict"], got["reason"]) == (verdict, reason)
    if verdict == "pass":
        assert ka._SHA1.fullmatch(got["receipt"]["tip"])


@needs_git
def test_git_commit_unknown_base_and_missing_repo(repo, tmp_path):
    r, _ = repo
    assert ka.evaluate_git_commit(_commit_spec(r, "feature", "b" * 40), deadline=time.monotonic() + 30)["verdict"] == "pending"
    assert ka.evaluate_git_commit(_commit_spec(tmp_path / "gone", "feature", "b" * 40),
                                  deadline=time.monotonic() + 30)["reason"] == "repo_missing"


@needs_git
def test_git_hardening_ignores_repo_config_programs_and_env(repo, tmp_path, monkeypatch):
    r, base = repo
    marker = tmp_path / "executed"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n")
    script.chmod(0o755)
    _git(r, "config", "core.fsmonitor", str(script))
    _git(r, "config", "core.hooksPath", str(tmp_path))
    _git(r, "config", "core.pager", str(script))
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _git(decoy, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", f"'core.fsmonitor'='{script}'")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    calls = []
    real_popen = subprocess.Popen
    monkeypatch.setattr(ka.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)) or real_popen(argv, **kw))
    got = ka.evaluate_git_commit(_commit_spec(r, "feature", base), deadline=time.monotonic() + 30)
    assert got["verdict"] == "pass"
    assert not marker.exists()
    for argv, kw in calls:
        assert argv[0] == ka.trusted_binary("git") and os.path.isabs(argv[0]) and kw["shell"] is False
        assert "core.fsmonitor=false" in argv and "protocol.allow=never" in argv and "core.useReplaceRefs=false" in argv
        assert not {k for k in kw["env"] if k in ("GIT_DIR", "GIT_CONFIG_PARAMETERS")}
        assert kw["env"]["GIT_NO_LAZY_FETCH"] == "1" and kw["env"]["PATH"] == "/usr/bin:/bin"
        assert kw["env"]["GIT_GRAFT_FILE"] == os.devnull and "core.commitGraph=false" in argv


def test_trusted_binary_never_uses_path(tmp_path, monkeypatch):
    fake = tmp_path / "git"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ka.trusted_binary("git") != str(fake)


# --------------------------------------------------------------------------- PR checks / contract AND

def _receipt(classification, *, ok=None, checks=(), complete=None):
    receipt = {"ok": classification == "success" if ok is None else ok, "classification": classification,
               "head_sha": "c" * 40, "pr_url": "https://github.com/acme/repo/pull/7",
               "checks": [{"name": n, "classification": c} for n, c in checks]}
    if complete if complete is not None else classification not in ("auth", "infra", "stale", "missing"):
        receipt["collection_complete"] = True
    return receipt


@pytest.fixture
def fake_pr(monkeypatch):
    state = {"receipt": _receipt("success", checks=[("ci", "success")]), "calls": []}

    def collect(contract, published_pr, assignee=None, *, deadline=None, gh_binary=None, mark_complete=False):
        state["calls"].append((contract, assignee, gh_binary, deadline))
        return state["receipt"]

    monkeypatch.setattr(pra, "collect_acceptance", collect)
    monkeypatch.setattr(ka, "trusted_binary", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ka, "evaluate_in_child", ka.evaluate_job)  # the fake collector lives in this process
    return state


@pytest.mark.parametrize("receipt,verdict", [
    (_receipt("success", checks=[("ci", "success")]), "pass"),
    (_receipt("failure", checks=[("ci", "failure"), ("lint", "success")]), "fail"),
    (_receipt("pending", checks=[("ci", "pending")]), "pending"),
    (_receipt("missing"), "pending"),
    (_receipt("unverifiable_app", checks=[("ci", "unverifiable_app")]), "pending"),
    (_receipt("auth"), "pending"), (_receipt("infra"), "pending"),
    (_receipt("stale", checks=[("ci", "failure")]), "pending"),
    (_receipt("pending", checks=[("ci", "pending"), ("lint", "failure")]), "fail"),
    (_receipt("infra", checks=[("aaa", "infra"), ("zzz", "failure")], complete=True), "fail"),
    (_receipt("pending", checks=[("ci", "failure")], complete=False), "pending"),
])
def test_pr_receipt_mapping(fake_pr, receipt, verdict):
    fake_pr["receipt"] = receipt
    got = ka.evaluate_pr("https://github.com/acme/repo/pull/7", assignee="builder", deadline=time.monotonic() + 30)
    assert got["verdict"] == verdict
    assert fake_pr["calls"][0][1:3] == ("builder", "/usr/bin/gh")


def test_pr_without_identity_is_pending_never_ambient(fake_pr):
    got = ka.evaluate_pr("https://github.com/acme/repo/pull/7", assignee=None, deadline=time.monotonic() + 30)
    assert got["reason"] == "no_identity" and fake_pr["calls"] == []


@pytest.mark.parametrize("contract_class,verdict", [("failure", "fail"), ("pending", "pending"), ("success", "pass")])
def test_completion_contract_and_acceptance_are_anded(conn, tmp_path, fake_pr, contract_class, verdict):
    (tmp_path / "f").write_text("x")
    fake_pr["receipt"] = _receipt(contract_class, checks=[("ci", contract_class)])
    _submitted(conn, _file_spec(tmp_path / "f"), completion_contract="https://github.com/acme/repo/pull/7")
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == verdict
    assert fake_pr["calls"][0][0] == "https://github.com/acme/repo/pull/7"


def test_owner_repo_contract_binds_at_submission_or_refuses(conn, fake_pr):
    tid, task = _claimed(conn, {"kind": "tbd"}, completion_contract="acme/repo")
    with pytest.raises(kb.AcceptanceSubmissionError):
        kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert kb.get_task(conn, tid).status == "running" and _subs(conn, tid) == []
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id,
                            metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
    assert kb.get_task(conn, tid).completion_contract == "https://github.com/acme/repo/pull/7"
    assert fake_pr["calls"] == []  # binding never collects


def test_collector_deadline_and_binary(monkeypatch, tmp_path):
    seen = []

    def capped(cmd, **kw):
        seen.append((cmd, kw["timeout"]))
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(ka, "run_capped", capped)
    monkeypatch.setattr(pra, "_assignee_profile_home", lambda a: str(tmp_path))
    got = pra.collect_acceptance("https://github.com/a/b/pull/1", None, assignee="b",
                                 deadline=time.monotonic() + 5, gh_binary="/opt/gh")
    assert got["classification"] == "infra"
    assert seen[0][0][0] == "/opt/gh" and seen[0][1] <= 5
    seen.clear()
    got = pra.collect_acceptance("https://github.com/a/b/pull/1", None, assignee="b",
                                 deadline=time.monotonic() - 1, gh_binary="/opt/gh")
    assert got["classification"] == "infra" and seen == []
    legacy = []
    monkeypatch.setattr(pra.subprocess, "run", lambda cmd, **kw: legacy.append((cmd, kw["timeout"])) or (_ for _ in ()).throw(
        subprocess.TimeoutExpired(cmd, kw["timeout"])))
    pra.collect_acceptance("https://github.com/a/b/pull/1", None)
    assert legacy[0][0][0] == "gh" and legacy[0][1] == 30 and seen == []  # legacy callers unchanged


@needs_git
def test_deadline_exhaustion_is_pending(conn, repo, tmp_path):
    r, base = repo
    assert ka.evaluate_git_commit(_commit_spec(r, "feature", base), deadline=time.monotonic() - 1)["reason"] == "deadline"
    tid = _submitted(conn, _file_spec(tmp_path / "nope"))
    assert ka.run_acceptance_pass(conn, total_deadline=0) == []
    assert kba.is_machine_wait(conn, tid) and _subs(conn, tid)[0]["attempts"] == 0


# --------------------------------------------------------------------------- goal mode

@pytest.fixture
def goal_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    def make(acceptance):
        with kbc.connect_closing() as c:
            tid = kb.create_task(c, title="g", assignee="test-worker", goal_mode=True, acceptance=acceptance)
            task = kb.claim_task(c, tid)
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
        return tid, task.current_run_id
    return make


@pytest.mark.parametrize("acceptance,gated", [({"kind": "tbd"}, False), (None, True)])
def test_goal_gate_skipped_only_with_acceptance(goal_worker, monkeypatch, acceptance, gated):
    from tools import kanban_tools as kt
    calls = []
    monkeypatch.setattr(kt, "_goal_gate", lambda *a: calls.append(a))
    tid, run_id = goal_worker(acceptance)
    out = json.loads(kt._handle_complete({"summary": "done"}))
    assert out["ok"] is True and bool(calls) is gated
    assert (out.get("acceptance") == "submitted") is (not gated)
    with kbc.connect_closing() as c:
        status = kb.goal_run_status(c, tid, run_id)
    assert status == ("done" if gated else "acceptance_submitted")


def test_cli_goal_gate_skipped_only_with_acceptance(goal_worker, monkeypatch):
    from hermes_cli import kanban as kc
    calls = []
    monkeypatch.setattr(kc, "_goal_gate_error", lambda *a, **k: calls.append(a) or None)
    monkeypatch.setattr(kc, "_worker_run_id_for", lambda tid: None)
    tid, _ = goal_worker({"kind": "tbd"})
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert "Submitted" in kc.run_slash(f"complete {tid} --summary s --force")
    assert calls == []
    plain, _ = goal_worker(None)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    kc.run_slash(f"complete {plain} --summary s --force")
    assert len(calls) == 1


def test_goal_loop_treats_submission_as_yielded_terminal():
    from hermes_cli.goals import run_kanban_goal_loop
    out = run_kanban_goal_loop(task_id="t", goal_text="g", run_turn=lambda p: "", block_fn=lambda m: None,
                               task_status_fn=lambda: "acceptance_submitted")
    assert out["outcome"] == "submitted_for_acceptance"


def test_tool_reports_open_submission_refusal(goal_worker):
    from tools import kanban_tools as kt
    goal_worker({"kind": "tbd"})
    assert json.loads(kt._handle_complete({"summary": "done"}))["acceptance"] == "submitted"
    out = json.loads(kt._handle_complete({"summary": "again"}))
    assert "waiting for machine acceptance" in out["error"]


# --------------------------------------------------------------------------- notifier

def test_notifier_wakes_on_fail_only():
    from gateway import kanban_watchers_notifier as nt
    ev = lambda verdict: types.SimpleNamespace(kind="acceptance_verdict", payload={"verdict": verdict, "implementer": "b",
                                                                                     "error": "Machine acceptance failed"})
    assert nt._wakes(ev("fail")) and not nt._wakes(ev("pending")) and not nt._wakes(ev("pass"))
    n = types.SimpleNamespace(head="Kanban t_1")
    assert nt._fmt_acceptance_verdict(ev("pending"), n)[0] is None
    assert "machine acceptance" in nt._fmt_acceptance_verdict(ev("fail"), n)[0]
    assert "acceptance_verdict" in nt.TERMINAL_KINDS


# --------------------------------------------------------------------------- review findings (regressions)

def test_probe_active_pr_machine_fail_is_a_spawnable_rework_handoff(conn, tmp_path, all_assignees_spawnable):
    tid, task = _claimed(conn, _file_spec(tmp_path / "missing"))
    kb.add_comment(conn, tid, author="builder", body="Published https://github.com/acme/repo/pull/7")
    assert kb.complete_task(conn, tid, summary="implemented", expected_run_id=task.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "fail"  # same second as the comment
    assert kbd.check_respawn_guard(conn, tid) is None
    spawned = []
    kbd.dispatch_once(conn, spawn_fn=lambda t, ws, **k: spawned.append(t.id))
    assert spawned == [tid]


def test_active_pr_still_guards_a_pr_comment_newer_than_the_fail(conn, tmp_path):
    tid = _submitted(conn, _file_spec(tmp_path / "missing"))
    ka.run_acceptance_pass(conn)
    kb.add_comment(conn, tid, author="builder", body="Published https://github.com/acme/repo/pull/8")
    assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_probe_aba_old_pass_cannot_close_new_manual_review(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"))
    assert kb.request_review(conn, tid, summary="v1", reviewer="reviewer", expected_run_id=task.current_run_id)
    approver = _exited_worker(conn, tid, kb.claim_review_task(conn, tid))
    assert kb.complete_task(conn, tid, summary="approved", expected_run_id=approver.current_run_id)
    lease = _leased(conn, tid)
    receipt = ka._evaluate_lease(lease, deadline=time.monotonic() + 30, per_check=30)
    assert receipt[0]["verdict"] == "pass"
    assert kb.reopen_review_task(conn, tid)
    new = _claim(conn, tid)
    assert kb.request_review(conn, tid, summary="v2 needs review", reviewer="reviewer",
                             expected_run_id=new.current_run_id)
    assert _subs(conn, tid)[0]["state"] == "cancelled"
    assert kb.claim_review_task(conn, tid) is not None  # an ordinary review again
    assert kba.apply_verdict(conn, lease, receipt) is None
    assert kb.get_task(conn, tid).status == "running"


def test_probe_spec_added_after_legacy_routing_never_writes_done(conn, tmp_path, monkeypatch):
    from hermes_cli import kanban_pr_acceptance_store as store
    tid, task = _claimed(conn, None)
    original = store.prepare_acceptance
    db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])

    def interleave(*args, **kwargs):
        other = kbc.connect(db_path)
        try:
            assert kba.set_acceptance(other, tid, _file_spec(tmp_path / "absent-required"))
        finally:
            other.close()
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "prepare_acceptance", interleave)
    with pytest.raises(kb.AcceptanceSubmissionError):
        kb.complete_task(conn, tid, summary="implemented", expected_run_id=task.current_run_id)
    assert kb.get_task(conn, tid).status == "running" and _subs(conn, tid) == []
    monkeypatch.setattr(store, "prepare_acceptance", original)
    assert kb.complete_task(conn, tid, summary="implemented", expected_run_id=task.current_run_id)
    assert kba.is_machine_wait(conn, tid)


def test_spec_removed_after_submit_routing_does_not_wedge(conn, tmp_path, monkeypatch):
    tid, task = _claimed(conn, {"kind": "tbd"})
    real = kba.task_acceptance

    def removed_after_routing(c, t):
        value = real(c, t)
        with kb.write_txn(c):
            c.execute("UPDATE tasks SET acceptance = NULL WHERE id = ?", (t,))
        return value

    monkeypatch.setattr(kba, "task_acceptance", removed_after_routing)
    with pytest.raises(kb.AcceptanceSubmissionError):
        kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    monkeypatch.setattr(kba, "task_acceptance", real)
    assert _subs(conn, tid) == []
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert kb.get_task(conn, tid).status == "done"


def test_probe_claimed_origin_without_pid_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"), exited=False)
    assert kb.complete_task(conn, tid, summary="s", expected_run_id=task.current_run_id)
    assert _subs(conn, tid)[0]["origin_kind"] == "worker"
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "identity_unknown"
    assert kb.get_task(conn, tid).status == "review"


def test_operator_completion_without_a_run_evaluates_at_once(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = kb.create_task(conn, title="t", assignee="builder", acceptance=_file_spec(tmp_path / "f"))
    assert kb.complete_task(conn, tid, summary="operator did it")
    assert _subs(conn, tid)[0]["origin_kind"] == "synthesized"
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"


def test_probe_dormant_tbd_does_not_suppress_a_new_review(conn):
    tid = _submitted(conn, {"kind": "tbd"})
    assert kb.reopen_review_task(conn, tid)
    new = _claim(conn, tid)
    assert kb.request_review(conn, tid, summary="new manual review", reviewer="reviewer",
                             expected_run_id=new.current_run_id)
    assert _subs(conn, tid)[0]["state"] == "cancelled"
    assert kb.claim_review_task(conn, tid) is not None


def _direct_triage(conn, tid):
    with kb.write_txn(conn):  # any writer, e.g. the dashboard's direct status write
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
    return True


@pytest.mark.parametrize("leave", [
    _direct_triage,
    lambda c, t: kb.archive_task(c, t),
    lambda c, t: kb.block_task(c, t, reason="hold"),
], ids=["direct-write", "archive", "block"])
def test_leaving_review_by_any_path_cancels_the_wait(conn, leave):
    tid = _submitted(conn, {"kind": "tbd"})
    moved = leave(conn, tid)
    if moved and kb.get_task(conn, tid).status != "review":
        assert _subs(conn, tid)[0]["state"] == "cancelled"
    else:
        assert kba.is_machine_wait(conn, tid)  # a refused transition keeps the wait


def test_removing_acceptance_routes_the_card_back(conn):
    tid = _submitted(conn, {"kind": "tbd"})
    assert kba.set_acceptance(conn, tid, None)
    task = kb.get_task(conn, tid)
    assert (task.status, task.assignee) == ("ready", "builder")
    assert not kbd._lane_rows(conn, "review")


@needs_git
def test_git_grafts_cannot_forge_ancestry(repo):
    r, base = repo
    tip = _git(r, "rev-parse", "unrelated")
    (r / ".git" / "info").mkdir(exist_ok=True)
    (r / ".git" / "info" / "grafts").write_text(f"{tip} {base}\n")
    got = ka.evaluate_git_commit(_commit_spec(r, "unrelated", base), deadline=time.monotonic() + 30)
    assert (got["verdict"], got["reason"]) == ("fail", "not_descendant")


@needs_git
def test_shallow_history_is_pending_not_a_false_fail(repo):
    r, base = repo
    tip = _git(r, "rev-parse", "feature")
    (r / ".git" / "shallow").write_text(tip + "\n")
    got = ka.evaluate_git_commit(_commit_spec(r, "feature", base), deadline=time.monotonic() + 30)
    assert (got["verdict"], got["reason"]) == ("pending", "history_incomplete")


def _fake_github(monkeypatch, *, head_changes=False):
    head = "a" * 40
    state = {"head": head}

    def api(endpoint, **kw):
        if endpoint == "graphql":
            return {"data": {"repository": {"pullRequest": {
                "headRefOid": head, "baseRefName": "main", "state": "OPEN",
                "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                    {"context": "aaa-infra", "app": None}, {"context": "zzz-failure", "app": None}]}}}}}}
        if "/rules/branches/" in endpoint:
            return [[]]
        if "/check-runs" in endpoint:
            runs = [{"id": 1, "name": "aaa-infra", "head_sha": head, "app": {"id": 1}, "status": "completed",
                     "conclusion": "timed_out"},
                    {"id": 2, "name": "zzz-failure", "head_sha": head, "app": {"id": 1}, "status": "completed",
                     "conclusion": "failure"}]
            if head_changes:
                state["head"] = "b" * 40
            return [{"total_count": 2, "check_runs": runs}]
        if "/statuses" in endpoint:
            return [[]]
        if "/pulls/" in endpoint:
            return {"head": {"sha": state["head"]}, "base": {"ref": "main"}, "state": "open"}
        raise AssertionError(endpoint)

    monkeypatch.setattr(pra, "_api", api)
    monkeypatch.setattr(pra, "_assignee_profile_home", lambda a: "/nonexistent")
    monkeypatch.setattr(ka, "trusted_binary", lambda name: f"/usr/bin/{name}")


def test_mixed_infra_and_failure_after_complete_collection_fails(monkeypatch):
    _fake_github(monkeypatch)
    got = ka.evaluate_pr("https://github.com/acme/repo/pull/7", assignee="b", deadline=time.monotonic() + 30)
    assert (got["verdict"], got["reason"]) == ("fail", "pr_checks_failed")


def test_head_change_during_collection_never_reuses_failures(monkeypatch):
    _fake_github(monkeypatch, head_changes=True)
    got = ka.evaluate_pr("https://github.com/acme/repo/pull/7", assignee="b", deadline=time.monotonic() + 30)
    assert (got["verdict"], got["reason"]) == ("pending", "pr_stale")


def test_run_capped_kills_on_overflow_of_either_stream():
    import sys
    for stream in ("stdout", "stderr"):
        with pytest.raises(ka.OutputOverflow):
            ka.run_capped([sys.executable, "-c", f"import sys; sys.{stream}.write('x' * 200000); sys.{stream}.flush()"],
                          env={"PATH": "/usr/bin:/bin"}, timeout=10, max_stdout=1000, max_stderr=1000)
    rc, out, err = ka.run_capped([sys.executable, "-c", "print('ok')"], env={"PATH": "/usr/bin:/bin"},
                                 timeout=10, max_stdout=1000, max_stderr=1000)
    assert (rc, out.strip(), err) == (0, b"ok", b"")


def test_truncated_gh_output_never_passes(monkeypatch, tmp_path):
    import sys
    gh = tmp_path / "gh"
    gh.write_text(f"#!{sys.executable}\nprint('{{\"data\": ' + ' ' * 100000 + '{{}}}}')\n")
    gh.chmod(0o755)
    monkeypatch.setattr(pra, "_MACHINE_GH_MAX_STDOUT", 1024)
    monkeypatch.setattr(pra, "_assignee_profile_home", lambda a: str(tmp_path))
    got = pra.collect_acceptance("https://github.com/a/b/pull/1", None, assignee="b",
                                 deadline=time.monotonic() + 30, gh_binary=str(gh), mark_complete=True)
    assert got["classification"] == "infra" and not got.get("collection_complete")


def test_receipt_cap_counts_utf8_bytes():
    results = [ka._result("fail", "file_missing", path="/" + "中" * 6000)]
    text = ka.bounded_receipt(results)
    assert len(text.encode("utf-8")) <= ka._MAX_RECEIPT_BYTES


def test_machine_gh_env_is_a_whitelist(monkeypatch, tmp_path):
    sentinels = {"CLAUDE_CODE_OAUTH_TOKEN": "s1", "AWS_ACCESS_KEY_ID": "s2", "AWS_SECRET_ACCESS_KEY": "s3",
                 "LD_PRELOAD": "/tmp/s4.so", "OPENAI_API_KEY": "s5", "ANTHROPIC_API_KEY": "s6",
                 "GH_TOKEN": "launcher-token", "PYTHONPATH": "/tmp/s7"}
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / ".env").write_text("GH_TOKEN=assignee-token\nHTTPS_PROXY=http://proxy.invalid:3128\nAWS_PROFILE=x\n")
    env = pra.machine_gh_env(str(profile), str(tmp_path / "home"))
    assert env["GH_TOKEN"] == "assignee-token" and env["HTTPS_PROXY"] == "http://proxy.invalid:3128"
    assert env["PATH"] == "/usr/bin:/bin" and env["HOME"] == str(tmp_path / "home")
    leaked = set(sentinels.values()) & set(env.values())
    assert not leaked, leaked
    allowed = {"PATH", "HOME", "XDG_CONFIG_HOME", "LANG", "LC_ALL", "GH_PROMPT_DISABLED",
               "GH_NO_UPDATE_NOTIFIER", "NO_COLOR", *pra._GH_GITHUB_LOGIN_KEYS, *pra._GH_TRANSPORT_KEYS}
    assert set(env) <= allowed
    with pytest.raises(pra._GateAuthError):
        pra.machine_gh_env(None, str(tmp_path))


def test_damaged_stored_spec_is_pending_and_one_bad_row_does_not_stop_the_pass(conn, tmp_path, monkeypatch):
    (tmp_path / "f").write_text("x")
    bad = _submitted(conn, _file_spec(tmp_path / "f"))
    damaged = '{"exists":1,"kind":"file","path":"%s"}' % (tmp_path / "f")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET acceptance = ? WHERE id = ?", (damaged, bad))
        conn.execute("UPDATE task_acceptance_submissions SET spec_fp = ? WHERE task_id = ?",
                     (ka.fingerprint(damaged), bad))
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "spec_invalid"
    good = _submitted(conn, _file_spec(tmp_path / "f"))
    _due_now(conn, bad)
    real = kba.apply_verdict

    def flaky(c, lease, results, **kw):
        if lease["task_id"] == bad:
            raise sqlite3_error()
        return real(c, lease, results, **kw)

    monkeypatch.setattr(kba, "apply_verdict", flaky)
    outcomes = ka.run_acceptance_pass(conn)
    assert [o["task_id"] for o in outcomes] == [good] and kb.get_task(conn, good).status == "done"


def sqlite3_error():
    import sqlite3
    return sqlite3.DatabaseError("damaged row")


def test_combine_of_nothing_is_never_pass():
    assert ka.combine([]) == ("pending", "no_predicates")
    assert ka.evaluate_file(_file_spec("/"))["receipt"]["check"] == "lstat_entry_exists"


# --------------------------------------------------------------------------- review round 2 (regressions)

def test_closed_unknown_claim_then_manual_approval_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"), exited=False)  # no pid ever registered
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    assert kb.complete_task(conn, tid, summary="approved by operator")  # no active run
    assert _subs(conn, tid)[0]["origin_kind"] == "worker"
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "identity_unknown"


def test_reaped_registered_worker_evaluates_after_operator_approval(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"))  # registered, then exited
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at = ended_at - 600 WHERE id = ?", (task.current_run_id,))
    kbd.reap_terminal_workers(conn)  # the pid is gone: the reaper clears the run's evidence
    assert kb.get_run(conn, task.current_run_id).worker_pid is None
    assert kb.complete_task(conn, tid, summary="approved")
    sub = _subs(conn, tid)[0]
    assert sub["origin_kind"] == "worker" and sub["worker_pid"]  # taken from the spawned event
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"


def test_ordinary_manual_review_reopen_still_restores_review_implementer(conn):
    tid, task = _claimed(conn, None)
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    assert kb.reopen_review_task(conn, tid)
    assert kb.get_task(conn, tid).assignee == "builder"


# --------------------------------------------------------------------------- review round 3 (L1)

def _live_worker(conn, tid, task):
    """Register a real, still-running worker process on the claimed run."""
    proc = subprocess.Popen(["sleep", "60"])
    kbd._set_worker_pid(conn, tid, proc.pid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    return proc


def _stop(proc):
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def _reviewer_claim(conn, tid, *, live):
    review = kb.claim_review_task(conn, tid)
    assert review is not None
    proc = _live_worker(conn, tid, review)
    if not live:
        _stop(proc)
    return review, proc


def test_live_implementer_and_exited_reviewer_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = kb.create_task(conn, title="t", assignee="builder", acceptance=_file_spec(tmp_path / "f"))
    task = kb.claim_task(conn, tid)
    impl = _live_worker(conn, tid, task)
    try:
        assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
        review, rproc = _reviewer_claim(conn, tid, live=False)
        assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
        sub = _subs(conn, tid)[0]
        assert (sub["worker_pid"], sub["approver_pid"], sub["approver_kind"]) == (impl.pid, rproc.pid, "worker")
        assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_alive"
        assert kb.get_task(conn, tid).status == "review"
        _stop(impl)
        _due_now(conn, tid)
        assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    finally:
        _stop(impl)


def test_exited_implementer_and_live_reviewer_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"))
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    review, rproc = _reviewer_claim(conn, tid, live=True)
    try:
        assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
        assert ka.run_acceptance_pass(conn)[0]["reason"] == "approver_worker_alive"
        assert kb.get_task(conn, tid).status == "review"
        _stop(rproc)
        _due_now(conn, tid)
        assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    finally:
        _stop(rproc)


def test_unregistered_reviewer_approval_stays_pending(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, task = _claimed(conn, _file_spec(tmp_path / "f"))
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    review = kb.claim_review_task(conn, tid)  # no pid ever registered for the reviewer
    assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["reason"] == "approver_identity_unknown"


def test_reviewer_approval_then_operator_reopen_and_recomplete_keeps_the_implementation(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid = kb.create_task(conn, title="t", assignee="builder", acceptance=_file_spec(tmp_path / "f"))
    task = kb.claim_task(conn, tid)
    impl = _live_worker(conn, tid, task)
    try:
        assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
        review, rproc = _reviewer_claim(conn, tid, live=False)
        assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
        assert kb.reopen_review_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="operator re-completes", force=True)
        sub = _subs(conn, tid)[-1]
        assert sub["state"] == "open" and (sub["worker_pid"], sub["approver_pid"]) == (impl.pid, rproc.pid)
        assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_alive"
        _stop(impl)
        _due_now(conn, tid)
        assert ka.run_acceptance_pass(conn)[0]["verdict"] == "pass"
    finally:
        _stop(impl)


def test_second_review_round_gates_the_newest_implementation_and_reviewer(conn, tmp_path):
    (tmp_path / "f").write_text("x")
    tid, first = _claimed(conn, _file_spec(tmp_path / "f"))
    assert kb.request_review(conn, tid, summary="impl 1", reviewer="reviewer", expected_run_id=first.current_run_id)
    first_review, _ = _reviewer_claim(conn, tid, live=False)
    assert kb.request_changes(conn, tid, reason="fix it", expected_run_id=first_review.current_run_id)[0]
    second = kb.claim_task(conn, tid)
    impl = _live_worker(conn, tid, second)
    try:
        assert kb.request_review(conn, tid, summary="impl 2", reviewer="reviewer", expected_run_id=second.current_run_id)
        review, rproc = _reviewer_claim(conn, tid, live=False)
        assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
        sub = _subs(conn, tid)[-1]
        assert (sub["worker_pid"], sub["approver_pid"]) == (impl.pid, rproc.pid)
        assert ka.run_acceptance_pass(conn)[0]["reason"] == "worker_alive"
    finally:
        _stop(impl)


def test_reimplementation_after_failed_approval_drops_the_old_reviewer(conn, tmp_path):
    tid, task = _claimed(conn, _file_spec(tmp_path / "missing"))
    assert kb.request_review(conn, tid, summary="impl", reviewer="reviewer", expected_run_id=task.current_run_id)
    review, _ = _reviewer_claim(conn, tid, live=False)
    assert kb.complete_task(conn, tid, summary="approved", expected_run_id=review.current_run_id)
    assert ka.run_acceptance_pass(conn)[0]["verdict"] == "fail"
    again = _claim(conn, tid)
    pid = kb.get_run(conn, again.current_run_id).worker_pid
    assert kb.complete_task(conn, tid, summary="impl 2", expected_run_id=again.current_run_id)
    sub = _subs(conn, tid)[-1]
    assert (sub["worker_pid"], sub["approver_kind"]) == (pid, None)


def _fifo_profile(tmp_path, name):
    profile = tmp_path / "home" / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    os.mkfifo(profile / ".env")  # no writer: any open() of it blocks forever


@pytest.mark.skipif(not os.access("/usr/bin/gh", os.X_OK), reason="needs /usr/bin/gh")
def test_blocking_profile_env_is_killed_and_the_next_card_still_runs(conn, tmp_path):
    _fifo_profile(tmp_path, "fifo-user")
    stuck = _submitted(conn, {"kind": "pr_checks", "pr_url": "https://github.com/acme/repo/pull/7"},
                       assignee="fifo-user")
    (tmp_path / "f").write_text("x")
    good = _submitted(conn, _file_spec(tmp_path / "f"))
    started = time.monotonic()
    outcomes = ka.run_acceptance_pass(conn, per_check=1.0, total_deadline=20)
    elapsed = time.monotonic() - started
    assert {o["task_id"]: (o["verdict"], o["reason"]) for o in outcomes} == {
        stuck: ("pending", "deadline"), good: ("pass", "file_present")}
    assert elapsed < 6, elapsed
    assert kb.get_task(conn, good).status == "done"


def test_child_evaluator_reports_crash_and_spent_budget_as_pending():
    assert ka.evaluate_in_child({"deadline": time.monotonic() - 1})[0]["reason"] == "deadline"
    assert ka.evaluate_in_child({"deadline": time.monotonic() + 10})[0]["reason"] == "evaluator_error"


def test_child_evaluator_decides_a_real_job(tmp_path):
    (tmp_path / "f").write_text("x")
    job = {"worker": {"worker_pid": None, "worker_host": None, "worker_fingerprint": None,
                      "origin_kind": "synthesized"},
           "acceptance": ka.validate_acceptance(_file_spec(tmp_path / "f")), "assignee": "b",
           "completion_contract": "local-only", "deadline": time.monotonic() + 20, "per_check": 20}
    assert ka.evaluate_in_child(job)[0]["verdict"] == "pass"
    job["worker"]["origin_kind"] = "worker"
    assert ka.evaluate_in_child(job)[0]["reason"] == "identity_unknown"


_ORPHAN = """
import os, sys, time
if os.fork() == 0:
    open(sys.argv[1], "w").write(str(os.getpid()))
    if sys.argv[2] == "overflow":
        time.sleep(0.3)
        sys.stdout.write("x" * 200000)
        sys.stdout.flush()
    time.sleep(60)
os._exit(0)
"""


def _gone(pid, wait=3.0):
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state in ("Z", "X"):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("mode", ["timeout", "overflow"])
def test_runner_kills_descendants_after_the_leader_exited(tmp_path, mode):
    import sys
    pidfile = tmp_path / "pid"
    # Resolved at run time: other test files purge and re-import hermes_cli, so a class captured at
    # collection is not the one the current module raises.
    error = subprocess.TimeoutExpired if mode == "timeout" else ka.OutputOverflow
    with pytest.raises(error):
        ka.run_capped([sys.executable, "-c", _ORPHAN, str(pidfile), mode], env={"PATH": "/usr/bin:/bin"},
                      timeout=1.5, max_stdout=1000, max_stderr=1000)
    pid = int(pidfile.read_text())
    try:
        assert _gone(pid), f"descendant {pid} survived"
    finally:
        with __import__("contextlib").suppress(ProcessLookupError):
            os.kill(pid, 9)
