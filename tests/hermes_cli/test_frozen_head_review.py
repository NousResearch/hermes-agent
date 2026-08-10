from __future__ import annotations

import json
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch, request):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    # The hermetic test runner may expose host processes through a restricted
    # /proc mount. Scanner behavior is exercised by the real helper test below;
    # admission tests isolate that unrelated host state.
    if "writer" not in request.node.name:
        monkeypatch.setattr(
            kb, "_worktree_writer_pids",
            lambda _path: kb._WorktreeWriterScan((), True, False, ()),
        )
    return home


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def git_tree(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "implementation"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "implemented.txt").write_text("done\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "implementation"],
        check=True,
    )
    return repo, git_head(repo)


def complete_with_evidence(conn, task_id: str, repo: Path) -> None:
    assert kb.complete_task(
        conn, task_id, summary="implementation complete",
        metadata={"changed_files": ["implemented.txt"], "tests_run": 3,
                  "head_sha": git_head(repo), "tree_sha": git_tree(repo)},
    )


def evidence(head: str, repo: Path, *, include_tree: bool = True) -> dict:
    result = {
        "head_sha": head,
        "worktree_path": str(repo),
        "clean": True,
        "implementation_complete": True,
        "source": "frozen-head-cli",
    }
    if include_tree:
        result["tree_sha"] = git_tree(repo)
    return result


def wait_for_helper_ready(helper: subprocess.Popen[str]) -> None:
    assert helper.stdout is not None
    ready, _, _ = select.select([helper.stdout], [], [], 5)
    assert ready, "writer helper did not signal readiness"
    assert helper.stdout.readline().strip() == "READY"


def start_writer_helper(
    code: str, *, cwd: Path, args: tuple[str, ...] = (),
) -> subprocess.Popen[str]:
    helper = subprocess.Popen(
        [sys.executable, "-c", code, *args],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wait_for_helper_ready(helper)
    return helper


def stop_writer_helper(helper: subprocess.Popen[str]) -> None:
    try:
        assert helper.stdin is not None
        helper.stdin.write("STOP\n")
        helper.stdin.flush()
        helper.wait(timeout=5)
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=5)


def isolate_proc_to_pid(monkeypatch, pid: int) -> None:
    real_iterdir = Path.iterdir

    def only_helper(path):
        if path == Path("/proc"):
            return iter([Path(f"/proc/{pid}")])
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", only_helper)


def submit_valid_frozen_head(
    conn, task_id: str, head: str, repo: Path,
) -> None:
    assert kb.submit_frozen_head_for_review(
        conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
    )


def test_db_accepts_completed_clean_exact_head_without_claim(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="frozen", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        assert kb.submit_frozen_head_for_review(
            conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
        )
        assert kb.get_task(conn, task_id).status == "review"
        event = kb.list_events(conn, task_id)[-1]
        assert event.kind == "review_submitted_frozen_head"
        assert event.payload["head_sha"] == head
        assert event.payload["tree_sha"] == git_tree(repo)
        assert event.payload["implementation_run_id"]


def test_committed_admission_freeze_persists_and_authorized_thaw_restores_modes(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    tracked = repo / "implemented.txt"
    root_mode = repo.stat().st_mode & 0o7777
    file_mode = tracked.stat().st_mode & 0o7777
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="durable freeze", workspace_kind="dir", workspace_path=str(repo))
        with kb._workspace_admission_lock(repo, exclusive=True, task_id=task_id, head_sha=head) as lease:
            assert not (repo.stat().st_mode & 0o222)
            assert not (tracked.stat().st_mode & 0o222)
            lease.commit()
        assert not (repo.stat().st_mode & 0o222)
        assert not (tracked.stat().st_mode & 0o222)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
        conn.commit()
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.thaw_frozen_workspace(conn, task_id, head_sha=head)
    assert repo.stat().st_mode & 0o7777 == root_mode
    assert tracked.stat().st_mode & 0o7777 == file_mode


def test_rejected_evidence_is_bounded_and_redacted(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    secret = "SUPER-SECRET-" + ("x" * 10000)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="bounded rejection", workspace_kind="dir", workspace_path=str(repo))
        with pytest.raises(kb.FrozenHeadReviewError):
            kb.submit_frozen_head_for_review(
                conn,
                task_id,
                head_sha=secret,
                worktree_path=secret,
                evidence={"head_sha": secret, "worktree_path": secret},
                actor=secret,
            )
        event = kb.list_events(conn, task_id)[-1]
        assert event is not None
        payload = json.dumps(event.payload)
        assert len(payload) < 2048
        assert secret not in payload
        assert event.payload["code"] == "FROZEN_HEAD_REVIEW_REJECTED"
        assert "SUPER-SECRET" not in payload


@pytest.mark.parametrize("status", ["scheduled", "ready"])
def test_scheduled_ready_submit_directly_without_transition_dance(kanban_home, tmp_path, status):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="scheduled recovery", workspace_kind="dir", workspace_path=str(repo))
        assert kb.schedule_task(conn, task_id, reason="today's frozen head")
        if status == "ready":
            assert kb.unblock_task(conn, task_id)
        assert kb.get_task(conn, task_id).status == status
        kb._synthesize_ended_run(
            conn, task_id, outcome="completed",
            summary="implementation complete",
            metadata={"changed_files": ["implemented.txt"], "tests_run": 3,
                       "head_sha": head, "tree_sha": git_tree(repo)},
        )
        assert kb.submit_frozen_head_for_review(
            conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
        )


@pytest.mark.parametrize("status", ["scheduled", "ready", "todo", "done"])
def test_every_accepted_dormant_state_is_explicitly_supported(kanban_home, tmp_path, status):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title=f"accepted {status}", workspace_kind="dir", workspace_path=str(repo))
        if status == "done":
            complete_with_evidence(conn, task_id, repo)
        else:
            conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
            conn.commit()
            kb._synthesize_ended_run(
                conn, task_id, outcome="completed", summary="implementation complete",
                metadata={"changed_files": ["implemented.txt"], "tests_run": 3,
                       "head_sha": head, "tree_sha": git_tree(repo)},
            )
        assert kb.submit_frozen_head_for_review(
            conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
        )


@pytest.mark.parametrize("status", ["triage", "running", "archived"])
def test_non_dormant_states_are_rejected(kanban_home, tmp_path, status):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title=f"reject {status}", workspace_kind="dir", workspace_path=str(repo))
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
        with pytest.raises(kb.FrozenHeadReviewError, match="cannot be submitted"):
            kb.submit_frozen_head_for_review(
                conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
            )


def test_legacy_review_required_blocked_state_is_accepted(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="blocked frozen head", workspace_kind="dir",
                                 workspace_path=str(repo), assignee="programmer")
        kb.claim_task(conn, task_id, claimer="programmer")
        run_id = kb.get_task(conn, task_id).current_run_id
        assert kb.block_task(conn, task_id, reason="review-required: inspect", expected_run_id=run_id)
        kb._synthesize_ended_run(
            conn, task_id, outcome="completed", summary="implementation complete",
            metadata={"changed_files": ["implemented.txt"], "tests_run": 3,
                      "head_sha": head, "tree_sha": git_tree(repo)},
        )
        assert kb.submit_frozen_head_for_review(
            conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
        )


def test_review_state_is_rejected(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="already review", workspace_kind="dir",
                                 workspace_path=str(repo))
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
        conn.commit()
        with pytest.raises(kb.FrozenHeadReviewError, match="status 'review'"):
            kb.submit_frozen_head_for_review(
                conn, task_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo)
            )


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda e, r: {**e, "head_sha": "0" * 40}, "evidence head_sha"),
        (lambda e, r: {**e, "clean": False}, "clean must be true"),
        (lambda e, r: {**e, "implementation_complete": False}, "implementation_complete"),
    ],
)
def test_db_rejects_malformed_or_mismatched_evidence(kanban_home, tmp_path, mutator, expected):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="reject", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        with pytest.raises(kb.FrozenHeadReviewError, match=expected):
            kb.submit_frozen_head_for_review(
                conn, task_id, head_sha=head, worktree_path=str(repo), evidence=mutator(evidence(head, repo), repo)
            )
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.list_events(conn, task_id)[-1].kind == "review_submission_rejected"


def test_db_rejects_missing_or_malformed_head_and_unrelated_state(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="head validation", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        valid = evidence(head, repo)
        with pytest.raises(kb.FrozenHeadReviewError, match="full 40-character"):
            kb.submit_frozen_head_for_review(conn, task_id, head_sha="", worktree_path=str(repo), evidence=valid)
        with pytest.raises(kb.FrozenHeadReviewError, match="full 40-character"):
            kb.submit_frozen_head_for_review(conn, task_id, head_sha="not-a-sha", worktree_path=str(repo), evidence=valid)

        unrelated_id = kb.create_task(conn, title="not done", workspace_kind="dir", workspace_path=str(repo))
        with pytest.raises(kb.FrozenHeadReviewError, match="completed implementation evidence is absent"):
            kb.submit_frozen_head_for_review(conn, unrelated_id, head_sha=head, worktree_path=str(repo), evidence=valid)


def test_db_rejects_dirty_head_live_claim_and_missing_implementation_evidence(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        dirty_id = kb.create_task(conn, title="dirty", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, dirty_id, repo)
        (repo / "dirty.txt").write_text("uncommitted\n")
        with pytest.raises(kb.FrozenHeadReviewError, match="dirty"):
            kb.submit_frozen_head_for_review(conn, dirty_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo))
        (repo / "dirty.txt").unlink()

        live_id = kb.create_task(conn, title="live", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, live_id, repo)
        conn.execute("UPDATE tasks SET claim_lock='live', worker_pid=1234 WHERE id=?", (live_id,))
        conn.commit()
        with pytest.raises(kb.FrozenHeadReviewError, match="live claim"):
            kb.submit_frozen_head_for_review(conn, live_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo))

        no_evidence_id = kb.create_task(conn, title="no evidence", workspace_kind="dir", workspace_path=str(repo))
        assert kb.complete_task(conn, no_evidence_id, result="done")
        with pytest.raises(kb.FrozenHeadReviewError, match="metadata is malformed"):
            kb.submit_frozen_head_for_review(conn, no_evidence_id, head_sha=head, worktree_path=str(repo), evidence=evidence(head, repo))


def test_cli_submit_review_uses_same_fail_closed_route(kanban_home, tmp_path, capsys):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="cli", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
    parser = __import__("argparse").ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args([
        "kanban", "submit-review", task_id, "--head-sha", head,
        "--worktree", str(repo), "--evidence", json.dumps(evidence(head, repo)),
    ])
    assert kc.kanban_command(args) == 0
    assert "Submitted" in capsys.readouterr().out
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "review"


def test_missing_task_is_frozen_head_error_without_fk_event_failure(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        with pytest.raises(kb.FrozenHeadReviewError, match="not found"):
            kb.submit_frozen_head_for_review(
                conn, "t_missing", head_sha=head, worktree_path=str(repo),
                evidence=evidence(head, repo),
            )


def test_os_writer_freeze_rejects_cwd_only_helper(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    helper = start_writer_helper(
        "import sys; print('READY', flush=True); sys.stdin.readline()",
        cwd=repo,
    )
    try:
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="writer", workspace_kind="dir", workspace_path=str(repo))
            complete_with_evidence(conn, task_id, repo)
            with pytest.raises(kb.FrozenHeadReviewError, match="active OS writers"):
                submit_valid_frozen_head(conn, task_id, head, repo)
    finally:
        stop_writer_helper(helper)


def test_os_writer_freeze_rejects_writable_fd_with_cwd_outside(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    helper = start_writer_helper(
        "import sys; f=open(sys.argv[1], 'r+b'); print('READY', flush=True); sys.stdin.readline()",
        cwd=tmp_path,
        args=(str(repo / "implemented.txt"),),
    )
    try:
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="fd writer", workspace_kind="dir", workspace_path=str(repo))
            complete_with_evidence(conn, task_id, repo)
            with pytest.raises(kb.FrozenHeadReviewError, match="active OS writers"):
                submit_valid_frozen_head(conn, task_id, head, repo)
    finally:
        stop_writer_helper(helper)


def test_os_writer_freeze_rejects_shared_writable_mmap_after_fd_close(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    helper = start_writer_helper(
        "import mmap, os, sys; fd=os.open(sys.argv[1], os.O_RDWR); mapping=mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE); os.close(fd); print('READY', flush=True); sys.stdin.readline()",
        cwd=tmp_path,
        args=(str(repo / "implemented.txt"),),
    )
    try:
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="mmap writer", workspace_kind="dir", workspace_path=str(repo))
            complete_with_evidence(conn, task_id, repo)
            with pytest.raises(kb.FrozenHeadReviewError, match="active OS writers"):
                submit_valid_frozen_head(conn, task_id, head, repo)
    finally:
        stop_writer_helper(helper)


def test_os_writer_freeze_allows_read_only_fd(kanban_home, tmp_path, monkeypatch):
    repo, head = make_repo(tmp_path)
    helper = start_writer_helper(
        "import sys; f=open(sys.argv[1], 'rb'); print('READY', flush=True); sys.stdin.readline()",
        cwd=tmp_path,
        args=(str(repo / "implemented.txt"),),
    )
    try:
        isolate_proc_to_pid(monkeypatch, helper.pid)
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="reader", workspace_kind="dir", workspace_path=str(repo))
            complete_with_evidence(conn, task_id, repo)
            submit_valid_frozen_head(conn, task_id, head, repo)
            task = kb.get_task(conn, task_id)
            assert task is not None and task.status == "review"
    finally:
        stop_writer_helper(helper)


def test_os_writer_freeze_allows_private_writable_mmap(kanban_home, tmp_path, monkeypatch):
    repo, head = make_repo(tmp_path)
    helper = start_writer_helper(
        "import ctypes, os, sys; fd=os.open(sys.argv[1], os.O_RDWR); size=os.fstat(fd).st_size; libc=ctypes.CDLL(None); mapping=libc.mmap(None, size, 3, 2, fd, 0); os.close(fd); print('READY', flush=True); sys.stdin.readline()",
        cwd=tmp_path,
        args=(str(repo / "implemented.txt"),),
    )
    try:
        isolate_proc_to_pid(monkeypatch, helper.pid)
        with kb.connect_closing() as conn:
            task_id = kb.create_task(conn, title="private mmap", workspace_kind="dir", workspace_path=str(repo))
            complete_with_evidence(conn, task_id, repo)
            submit_valid_frozen_head(conn, task_id, head, repo)
            task = kb.get_task(conn, task_id)
            assert task is not None and task.status == "review"
    finally:
        stop_writer_helper(helper)


def test_writer_scan_rejects_unsupported_platform(tmp_path, monkeypatch):
    monkeypatch.setattr(kb.sys, "platform", "win32")
    scan = kb._worktree_writer_pids(tmp_path)
    assert scan.writers == ()
    assert scan.complete is False
    assert scan.unsupported is True
    assert scan.errors == ("unsupported platform: win32",)


def test_writer_scan_rejects_absent_proc(tmp_path, monkeypatch):
    real_iterdir = Path.iterdir

    def fail_proc_enumeration(path):
        if path == Path("/proc"):
            raise FileNotFoundError("/proc disappeared")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_proc_enumeration)
    scan = kb._worktree_writer_pids(tmp_path)
    assert scan.writers == ()
    assert scan.complete is False
    assert scan.unsupported is False
    assert scan.errors == ("proc unavailable",)


def test_writer_scan_rejects_proc_enumeration_denial(tmp_path, monkeypatch):
    real_iterdir = Path.iterdir

    def deny_proc_enumeration(path):
        if path == Path("/proc"):
            raise PermissionError("hidden by policy")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", deny_proc_enumeration)
    scan = kb._worktree_writer_pids(tmp_path)
    assert scan.writers == ()
    assert scan.complete is False
    assert scan.unsupported is False
    assert scan.errors == ("proc enumeration denied",)


@pytest.mark.parametrize("process_kind", ["same uid", "root", "unknown"])
def test_writer_scan_rejects_unreadable_process_without_leaking_error(
    tmp_path, monkeypatch, process_kind,
):
    real_iterdir = Path.iterdir
    real_resolve = Path.resolve
    process = Path("/proc/424242")

    def fake_iterdir(path):
        if path == Path("/proc"):
            return iter([process])
        return real_iterdir(path)

    def deny_process(path, *args, **kwargs):
        if path == process / "cwd":
            raise PermissionError(f"private {process_kind} details")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(Path, "resolve", deny_process)
    scan = kb._worktree_writer_pids(tmp_path)
    assert scan.writers == ()
    assert scan.complete is False
    assert scan.unsupported is False
    assert len(scan.errors) <= 32
    assert scan.errors == ("pid 424242: permission denied",)
    assert process_kind not in " ".join(scan.errors)


@pytest.mark.parametrize(
    "changed_snapshot",
    [
        ("changed-head", lambda head, tree: ("0" * 40, tree, "")),
        ("changed-tree", lambda head, tree: (head, "1" * 40, "")),
        ("changed-dirty-status", lambda head, tree: (head, tree, " M implemented.txt\n")),
    ],
)
def test_git_snapshot_race_rejects_without_transition(
    kanban_home, tmp_path, monkeypatch, changed_snapshot,
):
    repo, head = make_repo(tmp_path)
    tree = git_tree(repo)
    label, second = changed_snapshot
    snapshots = [(head, tree, ""), second(head, tree)]
    calls = []

    def snapshots_with_race(path):
        calls.append(path)
        return snapshots.pop(0)

    monkeypatch.setattr(kb, "_git_worktree_snapshot", snapshots_with_race)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title=label, workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        with pytest.raises(kb.FrozenHeadReviewError, match="changed during verification"):
            submit_valid_frozen_head(conn, task_id, head, repo)
        assert len(calls) == 2
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "done"


@pytest.mark.parametrize("race", ["status", "workspace", "current_run", "claim", "pid", "run_id", "metadata"])
def test_final_transaction_races_reject_atomically(
    kanban_home, tmp_path, race,
):
    repo, head = make_repo(tmp_path)
    tree = git_tree(repo)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title=f"race {race}", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        run_row = conn.execute(
            "SELECT id FROM task_runs WHERE task_id=? AND outcome='completed' "
            "ORDER BY ended_at DESC, id DESC LIMIT 1", (task_id,),
        ).fetchone()
        assert run_row is not None
        run_id = int(run_row["id"])
        db_path = str(kb.kanban_db_path())
        racer_code = """
import json, sqlite3, sys, time
db, task_id, race, run_id, head, tree, other_workspace = sys.argv[1:]
conn = sqlite3.connect(db, timeout=30)
conn.execute("BEGIN IMMEDIATE")
print("READY", flush=True)
sys.stdin.readline()
if race == "status":
    conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (task_id,))
elif race == "workspace":
    conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (other_workspace, task_id))
elif race == "current_run":
    conn.execute("UPDATE tasks SET current_run_id=999999 WHERE id=?", (task_id,))
elif race == "claim":
    conn.execute("UPDATE tasks SET claim_lock='racer' WHERE id=?", (task_id,))
elif race == "pid":
    conn.execute("UPDATE tasks SET worker_pid=424242 WHERE id=?", (task_id,))
elif race == "run_id":
    conn.execute('''INSERT INTO task_runs
        (task_id, status, started_at, ended_at, outcome, summary, metadata)
        SELECT task_id, 'done', started_at, started_at + 1, 'completed',
               'newer run', metadata FROM task_runs WHERE id=?''', (int(run_id),))
elif race == "metadata":
    conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps({
        "changed_files": ["implemented.txt"], "tests_run": 4,
        "head_sha": head, "tree_sha": tree}), int(run_id)))
conn.commit()
conn.close()
"""
        racer = subprocess.Popen(
            [sys.executable, "-c", racer_code, db_path, task_id, race, str(run_id),
             head, tree, str(tmp_path / "different-worktree")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            wait_for_helper_ready(racer)
            result: dict[str, BaseException] = {}

            def submit() -> None:
                try:
                    with kb.connect_closing() as submit_conn:
                        submit_valid_frozen_head(submit_conn, task_id, head, repo)
                except BaseException as exc:  # assertion below checks the exact class
                    result["error"] = exc

            thread = threading.Thread(target=submit)
            thread.start()
            time.sleep(0.1)
            assert racer.stdin is not None
            racer.stdin.write("COMMIT\n")
            racer.stdin.flush()
            thread.join(timeout=10)
            assert not thread.is_alive()
            assert isinstance(result.get("error"), kb.FrozenHeadReviewError)
        finally:
            if racer.poll() is None:
                racer.kill()
            racer.wait(timeout=5)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status != "review"


def test_legacy_submit_task_for_review_keeps_running_and_blocked_contract(kanban_home):
    with kb.connect_closing() as conn:
        running_id = kb.create_task(conn, title="legacy running", assignee="programmer")
        kb.claim_task(conn, running_id, claimer="programmer")
        reviewed = kb.submit_task_for_review(conn, running_id, "reviewer")
        assert reviewed is not None and reviewed.status == "review"

        blocked_id = kb.create_task(conn, title="legacy blocked", assignee="programmer")
        kb.claim_task(conn, blocked_id, claimer="programmer")
        run_id = kb.get_task(conn, blocked_id).current_run_id
        assert kb.block_task(conn, blocked_id, reason="review-required: inspect", expected_run_id=run_id)
        reviewed = kb.submit_task_for_review(conn, blocked_id, "reviewer")
        assert reviewed is not None and reviewed.status == "review"


def test_frozen_head_rejects_unrelated_blocked_state(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="unrelated blocked", workspace_kind="dir",
            workspace_path=str(repo), assignee="programmer",
        )
        kb.claim_task(conn, task_id, claimer="programmer")
        task = kb.get_task(conn, task_id)
        assert task is not None
        run_id = task.current_run_id
        assert kb.block_task(
            conn, task_id, reason="waiting for operator input", expected_run_id=run_id,
        )
        with pytest.raises(kb.FrozenHeadReviewError, match="unrelated blocked state"):
            submit_valid_frozen_head(conn, task_id, head, repo)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"


def test_reviewer_claim_request_changes_and_next_implementation_claim_lifecycle(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    tracked = repo / "implemented.txt"
    original_file = tracked.stat().st_mode & 0o7777
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="review lifecycle", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        submit_valid_frozen_head(conn, task_id, head, repo)
        assert kb.claim_review_task(conn, task_id, claimer="reviewer") is not None
        assert not (repo.stat().st_mode & 0o222)
        assert not (tracked.stat().st_mode & 0o222)
        task = kb.get_task(conn, task_id)
        assert task is not None
        run_id = task.current_run_id
        assert kb.block_task(conn, task_id, reason="review-required: changes", expected_run_id=run_id)
        assert kb.unblock_task(conn, task_id)
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"
        assert not (repo.stat().st_mode & 0o222)
        assert kb.claim_task(conn, task_id, claimer="programmer") is not None
        assert tracked.stat().st_mode & 0o7777 == original_file


def test_terminal_review_completion_thaws_and_delete_requires_cleanup(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="terminal thaw", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        submit_valid_frozen_head(conn, task_id, head, repo)
        with pytest.raises(kb.FrozenHeadReviewError):
            kb.delete_task(conn, task_id)
        assert kb.complete_task(conn, task_id, result="approved")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "done"
        assert repo.stat().st_mode & 0o222
        assert kb.delete_task(conn, task_id)


def test_failed_implementation_thaw_blocks_without_spawn(kanban_home, tmp_path):
    repo, head = make_repo(tmp_path)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="thaw failure", workspace_kind="dir", workspace_path=str(repo))
        complete_with_evidence(conn, task_id, repo)
        submit_valid_frozen_head(conn, task_id, head, repo)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        manifests = kb._admission_state_root() / "manifests"
        for manifest in manifests.glob("*.json"):
            manifest.unlink()
        assert kb.claim_task(conn, task_id, claimer="programmer") is None
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "blocked"
        assert not (repo.stat().st_mode & 0o222)
