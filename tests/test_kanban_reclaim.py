"""Reclaiming the worktrees of DONE kanban cards.

Covers the defect that made ``disk-guard.sh``'s ``reclaim_merged_worktrees`` a
silent no-op, plus the guards that keep the reclaim from eating live work:

* **The leak** (squash merges are never ancestors of ``origin/main``, so nothing
  was ever eligible) — ``test_squash_merged_worktree_is_removed``.
* **Liveness hardening** — ``test_path_in_use_nonexistent_path_is_false`` is a
  negative control, plus ``test_path_in_use_detects_real_holder``. This is
  defence in depth, not a fix for an observed bug: the shell ``path_in_use``
  measured correct (0/100 false positives on a nonexistent path, correct
  positive control), and the earlier claim that it universally self-matched its
  own ``grep`` was a harness artifact and is retracted.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kbc
from hermes_cli import kanban as kanban_ops, kanban_reclaim as kbr

HOUR = 3600
DAY = 86400


# --- helpers -----------------------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    )
    return res.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "T")
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path


def _with_remote(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """``(repo, bare_remote)`` with ``origin`` wired and ``main`` pushed."""
    bare = tmp_path / f"{name}.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)], check=True)
    repo = _init_repo(tmp_path / name)
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", "origin", "main")
    return repo, bare


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "kanban.db"
    kbc.init_db(db_path=path)
    return path


def _card(db: Path, task_id: str, status: str, completed_at: int | None) -> None:
    with kbc.connect_closing(db_path=db) as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, created_at, completed_at) VALUES (?,?,?,?,?)",
            (task_id, f"task {task_id}", status, int(time.time()) - 10 * DAY, completed_at),
        )
        conn.commit()


def _reasons(decisions, task_id: str) -> str:
    return " | ".join(d.reason for d in decisions if d.task_id == task_id)


def _done_worktree(tmp_path: Path, db: Path, root: Path, task_id: str,
                   *, unpushed: bool = False) -> tuple[Path, Path]:
    """A card done 2 days ago with a pushed branch worktree under ``root``."""
    repo, _bare = _with_remote(tmp_path, "repo-" + task_id.replace("t_", "x"))
    branch = f"fix/{task_id}"
    wt = root / task_id
    _git(repo, "worktree", "add", "-q", "-b", branch, str(wt), "main")
    (wt / "change.txt").write_text("work\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "work")
    if not unpushed:
        _git(wt, "push", "-q", "origin", branch)
    _card(db, task_id, "done", int(time.time()) - 2 * DAY)
    return wt, repo


# --- defect A: liveness ------------------------------------------------------

def test_path_in_use_nonexistent_path_is_false():
    """Negative control: the shell guard reported even this path as in use."""
    assert kbr.path_in_use(Path("/nonexistent/ZZZNOPE")) is False


@pytest.mark.live_system_guard_bypass
def test_path_in_use_detects_real_holder(tmp_path: Path):
    held = tmp_path / "held"
    held.mkdir()
    proc = subprocess.Popen(["sleep", "30"], cwd=str(held))
    try:
        assert kbr.path_in_use(held) is True
        assert kbr.path_holder_pid(held) is not None
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.live_system_guard_bypass
def test_in_use_worktree_is_preserved(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_aaaaaaa1")
    proc = subprocess.Popen(["sleep", "30"], cwd=str(wt))
    try:
        decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    finally:
        proc.kill()
        proc.wait()
    assert wt.is_dir()
    assert "in use by pid" in _reasons(decisions, "t_aaaaaaa1")


# --- eligibility -------------------------------------------------------------

def test_done_and_pushed_worktree_is_removed(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_bbbbbbb2")
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert not wt.exists(), _reasons(decisions, "t_bbbbbbb2")
    assert any(d.removed and d.task_id == "t_bbbbbbb2" for d in decisions)


def test_unpushed_commits_preserve_the_worktree(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_ccccccc3", unpushed=True)
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert wt.is_dir()
    assert "unpushed" in _reasons(decisions, "t_ccccccc3")


def test_squash_merged_worktree_is_removed(tmp_path: Path, db: Path):
    """Defect B: main carries an equivalent commit with a DIFFERENT sha, so the
    branch tip is not an ancestor of origin/main — but nothing is unpushed."""
    task_id = "t_ddddddd4"
    repo, _bare = _with_remote(tmp_path, "squash-repo")
    root = tmp_path / "wts"
    root.mkdir()
    branch = f"fix/{task_id}"
    wt = root / task_id
    _git(repo, "worktree", "add", "-q", "-b", branch, str(wt), "main")
    (wt / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-qm", "feature")
    _git(wt, "push", "-q", "origin", branch)
    tip = _git(wt, "rev-parse", "HEAD")

    # Squash-merge: same tree, new sha, branch tip never becomes an ancestor.
    (repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feature (squashed)")
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    assert subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", tip, "origin/main"],
    ).returncode != 0, "fixture must not be a fast-forward merge"

    _card(db, task_id, "done", int(time.time()) - 2 * DAY)
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert not wt.exists(), _reasons(decisions, task_id)


def test_running_card_is_preserved(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    repo, _bare = _with_remote(tmp_path, "running-repo")
    wt = root / "t_eeeeeee5"
    _git(repo, "worktree", "add", "-q", "-b", "fix/run", str(wt), "main")
    _card(db, "t_eeeeeee5", "running", None)
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert wt.is_dir()
    assert "status=running" in _reasons(decisions, "t_eeeeeee5")


def test_unknown_task_id_is_preserved(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    (root / "t_fffffff6").mkdir()
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert (root / "t_fffffff6").is_dir()
    assert _reasons(decisions, "t_fffffff6") == "unknown task"


def test_recently_completed_card_is_under_the_age_floor(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_1111111a")
    with kbc.connect_closing(db_path=db) as conn:
        conn.execute("UPDATE tasks SET completed_at = ? WHERE id = ?",
                     (int(time.time()) - HOUR, "t_1111111a"))
        conn.commit()
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert wt.is_dir()
    assert "under the 6h floor" in _reasons(decisions, "t_1111111a")


def test_symlink_escaping_the_root_is_preserved(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    outside = tmp_path / "elsewhere" / "t_2222222b"
    outside.mkdir(parents=True)
    (outside / "keep.txt").write_text("precious\n", encoding="utf-8")
    (root / "t_2222222b").symlink_to(outside)
    _card(db, "t_2222222b", "done", int(time.time()) - 2 * DAY)
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert outside.is_dir() and (outside / "keep.txt").exists()
    assert "outside the worktrees root" in _reasons(decisions, "t_2222222b")


def test_registered_worktree_goes_through_git(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, repo = _done_worktree(tmp_path, db, root, "t_3333333c")
    assert str(wt) in _git(repo, "worktree", "list")
    kbr.reclaim_done_worktrees(roots=[root], db_path=db)
    assert not wt.exists()
    assert "t_3333333c" not in _git(repo, "worktree", "list")


def test_dry_run_removes_nothing(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_4444444d")
    decisions = kbr.reclaim_done_worktrees(roots=[root], db_path=db, dry_run=True)
    assert wt.is_dir()
    assert "would remove" in _reasons(decisions, "t_4444444d")


# --- sibling logs -------------------------------------------------------------

def test_sibling_logs_of_old_done_cards_move_to_logs_dir(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    done = root / "t_5555555e-pi.log"
    done.write_text("old run\n", encoding="utf-8")
    live = root / "t_6666666f-pi.log"
    live.write_text("live run\n", encoding="utf-8")
    _card(db, "t_5555555e", "done", int(time.time()) - 30 * DAY)
    _card(db, "t_6666666f", "running", None)

    decisions = kbr.reclaim_stale_sibling_logs(root=root, db_path=db)
    assert not done.exists()
    assert (root / "logs" / "t_5555555e-pi.log").read_text(encoding="utf-8") == "old run\n"
    assert live.is_file()
    assert "status=running" in _reasons(decisions, "t_6666666f")


def test_sibling_log_name_collision_is_suffixed(tmp_path: Path, db: Path):
    root = tmp_path / "wts"
    root.mkdir()
    (root / "logs").mkdir()
    (root / "logs" / "t_7777777a-pi.log").write_text("first\n", encoding="utf-8")
    (root / "t_7777777a-pi.log").write_text("second\n", encoding="utf-8")
    _card(db, "t_7777777a", "done", int(time.time()) - 30 * DAY)

    kbr.reclaim_stale_sibling_logs(root=root, db_path=db)
    assert (root / "logs" / "t_7777777a-pi.log").read_text(encoding="utf-8") == "first\n"
    assert (root / "logs" / "t_7777777a-pi.log.1").read_text(encoding="utf-8") == "second\n"


# --- wiring -------------------------------------------------------------------

def test_cmd_gc_actually_reclaims_worktrees(tmp_path: Path, db: Path, monkeypatch, capsys):
    """The real failure mode was a guard that ran and reclaimed nothing, so this
    calls the real ``_cmd_gc`` with the real reclaim wired in — no mocks."""
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_8888888b")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "scratch"))

    args = argparse.Namespace(
        event_retention_days=30, log_retention_days=30,
        worktree_min_age_hours=6, worktree_roots=[str(root)],
        no_worktrees=False, dry_run=False,
    )
    assert kanban_ops._cmd_gc(args) == 0
    out = capsys.readouterr().out
    assert not wt.exists(), out
    assert "done-card worktree(s)" in out


def test_kanban_reclaim_subcommand_parses_and_runs(tmp_path: Path, db: Path, monkeypatch, capsys):
    from hermes_cli import kanban

    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_9999999c")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    kanban.build_parser(sub)
    args = parser.parse_args(
        ["kanban", "reclaim", "--worktree-root", str(root), "--worktree-min-age-hours", "1"]
    )
    assert args.task_id is None
    assert kanban._cmd_reclaim(args) == 0
    assert not wt.exists(), capsys.readouterr().out


def test_reclaim_never_touches_paths_outside_the_given_roots(tmp_path: Path, db: Path):
    """No root, no removal — the containment contract, stated once."""
    outside = tmp_path / "outside" / "t_abcdef01"
    outside.mkdir(parents=True)
    _card(db, "t_abcdef01", "done", int(time.time()) - 5 * DAY)
    empty_root = tmp_path / "wts"
    empty_root.mkdir()
    decisions = kbr.reclaim_done_worktrees(roots=[empty_root], db_path=db)
    assert outside.is_dir()
    assert decisions == []


def test_default_roots_are_globbed_not_literal(tmp_path: Path, db: Path):
    home = tmp_path / "workspace"
    (home / "proj-worktrees").mkdir(parents=True)
    (home / "repo" / ".worktrees").mkdir(parents=True)
    (home / "unrelated").mkdir()
    roots = kbr.default_roots(db_path=db, workspace_home=home)
    assert (home / "proj-worktrees").resolve() in roots
    assert (home / "repo" / ".worktrees").resolve() in roots
    assert (home / "unrelated").resolve() not in roots


def test_os_getpid_is_never_reported_as_a_holder(tmp_path: Path):
    """Our own process must not make its own cwd un-reclaimable."""
    d = tmp_path / "selfcwd"
    d.mkdir()
    prev = os.getcwd()
    os.chdir(d)
    try:
        assert kbr.path_holder_pid(d) in (None,) or kbr.path_holder_pid(d) != os.getpid()
    finally:
        os.chdir(prev)


# --- deliverable 1: before/after free space recorded ON THE CARD --------------

def test_free_space_mb_matches_df_and_survives_a_missing_path(tmp_path: Path):
    """The number we record must be the same quantity ``df`` prints."""
    mine = kbr.free_space_mb(tmp_path)
    assert mine is not None and mine > 0
    import shutil as _sh
    assert abs(mine - _sh.disk_usage(tmp_path).free // (1024 * 1024)) <= 1
    # A path that does not exist must still yield the enclosing volume, not None.
    assert kbr.free_space_mb(tmp_path / "no" / "such" / "dir") is not None


def test_reclaim_records_before_after_free_space_on_the_card(
    tmp_path: Path, db: Path, monkeypatch, capsys,
):
    """Deliverable 1: the df evidence lands on the board FROM THE TICK.

    The prior round's evidence only existed because a human ran a script by
    hand; that is exactly the thing the next scheduled tick will not do.
    """
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_dfdfdf01")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))

    args = argparse.Namespace(
        worktree_min_age_hours=6, worktree_roots=[str(root)],
        no_worktrees=False, dry_run=False, no_comment=False,
    )
    assert kanban_ops._reclaim_worktrees(args) == 1
    out = capsys.readouterr().out
    assert not wt.exists(), out
    assert "free before" in out and "recorded the before/after df on 1 card(s)" in out

    with kbc.connect_closing(db_path=db) as conn:
        bodies = [r["body"] for r in conn.execute(
            "SELECT body FROM task_comments WHERE task_id = 't_dfdfdf01'"
        ).fetchall()]
    assert len(bodies) == 1, bodies
    body = bodies[0]
    assert "free before:" in body and "free after:" in body and "delta:" in body
    assert str(wt) in body


def test_dry_run_records_nothing_on_the_board(tmp_path: Path, db: Path, monkeypatch):
    root = tmp_path / "wts"
    root.mkdir()
    wt, _repo = _done_worktree(tmp_path, db, root, "t_dfdfdf02")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    args = argparse.Namespace(
        worktree_min_age_hours=6, worktree_roots=[str(root)],
        no_worktrees=False, dry_run=True, no_comment=False,
    )
    assert kanban_ops._reclaim_worktrees(args) == 0
    assert wt.is_dir()
    with kbc.connect_closing(db_path=db) as conn:
        n = conn.execute(
            "SELECT count(*) AS c FROM task_comments WHERE task_id = 't_dfdfdf02'"
        ).fetchone()["c"]
    assert n == 0


# --- deliverable 4a: the SCHEDULED TICK really invokes the reclaim ------------

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "disk-guard.sh"


def _run_guard(tmp_path: Path, stub_body: str, *, name: str) -> tuple[int, str]:
    """Run the real disk-guard with a stubbed hermes binary, isolated $HOME."""
    home = tmp_path / name
    (home / "workspace").mkdir(parents=True)
    (home / ".hermes" / "kanban").mkdir(parents=True)
    bindir = tmp_path / (name + "-bin")
    bindir.mkdir()
    stub = bindir / "hermes"
    stub.write_text(stub_body, encoding="utf-8")
    stub.chmod(0o755)
    marker = tmp_path / (name + ".calls")
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        DISK_GUARD_HERMES_BIN=str(stub),
        DISK_GUARD_FLOOR_GI="0",
        RECLAIM_MARKER=str(marker),
        HERMES_KANBAN_DB=str(home / ".hermes" / "kanban.db"),
    )
    res = subprocess.run(
        ["bash", str(GUARD), "--reclaim"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    calls = marker.read_text(encoding="utf-8") if marker.exists() else ""
    return res.returncode, res.stdout + res.stderr + "\n--CALLS--\n" + calls


def test_guard_tick_invokes_hermes_kanban_reclaim(tmp_path: Path):
    """INVARIANT: the scheduled tick must actually call the reclaim.

    Nothing asserted this before, which is how the live host ran 179 ticks
    taking the dead shell fallback while the log read as routine.
    """
    stub = (
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$RECLAIM_MARKER"\n'
        'exit 0\n'
    )
    rc, out = _run_guard(tmp_path, stub, name="ok")
    assert "kanban reclaim --dry-run" in out, out   # capability probe
    assert "kanban reclaim --logs" in out, out      # the real invocation
    assert "FAIL" not in out, out
    assert rc == 0, out


def test_guard_logs_a_probe_failure_as_a_failure(tmp_path: Path):
    """An old hermes that rejects the probe must NOT read as a routine tick.

    The shell fallback it falls back to is the proven no-op, so this branch
    means the host is still leaking and has to be greppable as FAIL.
    """
    stub = (
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$RECLAIM_MARKER"\n'
        'echo "usage: hermes kanban reclaim [-h] task_id" >&2\n'
        'exit 2\n'
    )
    _rc, out = _run_guard(tmp_path, stub, name="old")
    assert "kanban reclaim --dry-run" in out, out
    assert "kanban reclaim --logs" not in out, out
    assert "worktrees: FAIL hermes reclaim unavailable" in out, out
