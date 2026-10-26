"""The commit point's obligations: armed before the tree moves, handed back only when still ours.

``arm_commit_obligations`` owes the completion tail and the host fleet restart before git (or the
ZIP swap) writes a file; ``disarm_commit_obligations`` puts them back after a failure that left the
tree at its start commit. The host record is shared by every install of the OS user.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_commit as commit
from hermes_cli.update_host_obligation import host_obligation_path


@pytest.fixture(autouse=True)
def _fresh_commit_point(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    commit.begin_update_attempt()
    yield
    commit.begin_update_attempt()


@pytest.fixture
def root(tmp_path) -> Path:
    install = tmp_path / "install"
    install.mkdir()
    return install


def test_arm_refuses_when_no_store_accepts_the_fleet_restart(root):
    """Both the host record and the per-home breadcrumb unwritable: the move must not start (F26)."""
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path

    host_obligation_path().mkdir(parents=True)  # a directory where the record goes
    _fleet_restart_pending_marker_path().mkdir(parents=True)

    with pytest.raises(OSError):
        commit.arm_commit_obligations(root, "b" * 40)


def test_disarm_leaves_another_installs_newer_host_record(root):
    """A failed run hands back only what it armed: another install's record written since stays (F27)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    host.write_text("OTHER-INSTALL", encoding="utf-8")

    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OTHER-INSTALL"


def test_plain_arm_and_disarm_restores_what_the_run_found(root):
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    assert host.read_text(encoding="utf-8-sig") != "OLD"

    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OLD"


def test_an_unreadable_record_refuses_the_arm_and_survives(root, monkeypatch):
    """A read error is not absence: guessing 'none' would delete the record on disarm (F28)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("OLD", encoding="utf-8")
    real = Path.read_bytes

    def flaky(self):
        if self == host:
            raise PermissionError(13, "sharing violation", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    with pytest.raises(OSError):
        commit.arm_commit_obligations(root, "a" * 40)
    monkeypatch.setattr(Path, "read_bytes", real)
    commit.disarm_commit_obligations()

    assert host.read_text(encoding="utf-8-sig") == "OLD"


@pytest.mark.platforms("posix")  # unprivileged symlinks
def test_disarm_never_writes_through_a_planted_restore_alias(root, tmp_path):
    """A same-user link at a temp name must not redirect the restore (N05)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_text("ORIGINAL", encoding="utf-8")
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("PRECIOUS", encoding="utf-8")
    commit.arm_commit_obligations(root, "a" * 40)
    for alias in (host.with_name(host.name + ".restore"), host.with_name(f".{host.name}.{os.getpid()}.restore")):
        alias.symlink_to(sentinel)

    commit.disarm_commit_obligations()

    assert sentinel.read_text(encoding="utf-8-sig") == "PRECIOUS"
    assert not host.is_symlink() and host.read_text(encoding="utf-8-sig") == "ORIGINAL"


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8", env=env).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_refused_pull_after_the_branch_switch_owes_the_head_the_switch_landed_on(tmp_path, monkeypatch):
    """CP0 switched a parked feature branch (X) to main (A); CP1's arm for B then failed. The
    obligation used to stay on B, which the checkout at A never contains, while disarm refused
    (HEAD != X): an undischargeable debt (review C1). It must name A, and the refusal must not
    claim the checkout was untouched."""
    import hermes_cli.main as hermes_main
    from hermes_cli import update_cmd
    from hermes_cli._early_recovery import interrupted_pull_marker
    from hermes_cli.update_host_obligation import read_host_obligation

    up, clone = tmp_path / "up", tmp_path / "clone"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    (up / "f").write_text("A", encoding="utf-8")
    _git(up, "add", "-A")
    _git(up, "commit", "-qm", "A")
    a = _git(up, "rev-parse", "HEAD")
    _git(tmp_path, "clone", "-q", str(up), str(clone))
    _git(clone, "checkout", "-q", "-b", "feat")
    (clone / "g").write_text("X", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "X")
    x = _git(clone, "rev-parse", "HEAD")
    (up / "f").write_text("B", encoding="utf-8")
    _git(up, "commit", "-qam", "B")
    b = _git(up, "rev-parse", "HEAD")
    _git(clone, "fetch", "-q", "origin")
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(commit, "_owns_live_checkout", lambda _root: False)
    monkeypatch.chdir(clone)

    commit.record_run_start(["git"], clone)
    switched = update_cmd._switch_branch_at_commit_point(["git"], "main", "origin/main", pre=x, stash=None)
    assert switched.returncode == 0 and _git(clone, "rev-parse", "HEAD") == a
    interrupted_pull_marker(clone).mkdir()  # CP1's marker cannot be written
    reason = commit.arm_commit_point(["git"], clone, b, pre=a, target=b, stash=None)

    assert reason and "the checkout was not changed" not in reason and a[:10] in reason
    assert (read_host_obligation() or {}).get("expected_sha") == a


def _run_state() -> dict:
    return {name: getattr(commit, name, None) for name in ("_armed_snapshot", "_owner")} | {
        "_armed_bytes": dict(commit._armed_bytes)}


def _enter_run(state: dict) -> None:
    for name, value in state.items():
        if name == "_armed_bytes":
            commit._armed_bytes.clear()
            commit._armed_bytes.update(value)
        else:
            setattr(commit, name, value)


def test_one_installs_disarm_never_deletes_another_installs_debt_for_the_same_sha(root, tmp_path):
    """Install A and install B both arm SHA X. Same-SHA arms share one host record, so a byte
    compare cannot tell them apart: A's failed run used to delete the record B still owes through
    (review C2). A's disarm hands back only A's stake; the last owner to leave puts back what the first found."""
    from hermes_cli.update_host_obligation import read_host_obligation

    other = tmp_path / "other-install"
    other.mkdir()
    commit.arm_commit_obligations(root, "a" * 40)  # install A
    run_a = _run_state()
    commit.begin_update_attempt()
    commit.arm_commit_obligations(other, "a" * 40)  # install B, same pulled SHA
    run_b = _run_state()

    _enter_run(run_a)
    commit.disarm_commit_obligations()
    assert (read_host_obligation() or {}).get("expected_sha") == "a" * 40  # B still owes it

    _enter_run(run_b)
    commit.disarm_commit_obligations()
    assert not host_obligation_path().exists()  # the last owner puts back what the FIRST one found


def _abc_clone(tmp_path, monkeypatch):
    """Upstream commits A, B, C; a clone of it with PROJECT_ROOT pointed at it. Returns (clone, a, b, c)."""
    import hermes_cli.main as hermes_main

    up, clone = tmp_path / "up", tmp_path / "clone"
    up.mkdir()
    _git(up, "init", "-q", "-b", "main")
    shas = []
    for name in "ABC":
        (up / "f").write_text(name, encoding="utf-8")
        _git(up, "add", "-A")
        _git(up, "commit", "-qm", name)
        shas.append(_git(up, "rev-parse", "HEAD"))
    _git(tmp_path, "clone", "-q", str(up), str(clone))
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", clone)
    monkeypatch.setattr(commit, "_owns_live_checkout", lambda _root: False)
    monkeypatch.chdir(clone)
    return (clone, *shas)


def _move_ref_after_arm(monkeypatch, clone, ref, sha):
    """The race: ``ref`` moves to ``sha`` right after the commit point armed (marker + debt)."""
    real = commit.arm_commit_point

    def arm_then_move(*args, **kwargs):
        refused = real(*args, **kwargs)
        _git(clone, "update-ref", ref, sha)
        return refused

    monkeypatch.setattr(commit, "arm_commit_point", arm_then_move)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_the_pull_merges_the_armed_commit_not_a_tracking_ref_that_moved(tmp_path, monkeypatch):
    """CP1 armed B, then origin/main moved to C before ``merge --ff-only origin/main``: HEAD landed C
    while the debt named B and the marker was gone (review O3). The merge names the armed OID."""
    from hermes_cli import update_cmd
    from hermes_cli._early_recovery import interrupted_pull_marker
    from hermes_cli.update_host_obligation import read_host_obligation

    clone, a, b, c = _abc_clone(tmp_path, monkeypatch)
    _git(clone, "reset", "-q", "--hard", a)
    _git(clone, "update-ref", "refs/remotes/origin/main", b)
    commit.record_run_start(["git"], clone)
    _move_ref_after_arm(monkeypatch, clone, "refs/remotes/origin/main", c)
    update_cmd._pull_updates(["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)
    assert _git(clone, "rev-parse", "HEAD") == b
    assert (read_host_obligation() or {}).get("expected_sha") == b
    assert not interrupted_pull_marker(clone).exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_branch_that_moves_during_the_switch_leaves_head_and_debt_agreeing(tmp_path, monkeypatch):
    """CP0 resolved main to A and armed A, then main moved to B before ``checkout main``: the switch
    landed B, reported success and dropped the marker while the debt named A (review O2). It must
    refuse, with the debt following the HEAD it landed on."""
    from hermes_cli import update_cmd
    from hermes_cli._early_recovery import interrupted_pull_marker
    from hermes_cli.update_host_obligation import read_host_obligation

    clone, a, b, _c = _abc_clone(tmp_path, monkeypatch)
    _git(clone, "reset", "-q", "--hard", a)
    _git(clone, "checkout", "-q", "-b", "feat")
    (clone / "g").write_text("X", encoding="utf-8")
    _git(clone, "add", "-A")
    _git(clone, "commit", "-qm", "X")
    x = _git(clone, "rev-parse", "HEAD")
    commit.record_run_start(["git"], clone)
    _move_ref_after_arm(monkeypatch, clone, "refs/heads/main", b)
    switched = update_cmd._switch_branch_at_commit_point(["git"], "main", "origin/main", pre=x, stash=None)
    assert switched.returncode == 1 and "moved" in switched.stderr
    assert _git(clone, "rev-parse", "HEAD") == b
    assert (read_host_obligation() or {}).get("expected_sha") == b
    assert not interrupted_pull_marker(clone).exists()


def _repo_at_a(tmp_path, monkeypatch) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f").write_text("A", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "A")
    monkeypatch.setattr(commit, "_owns_live_checkout", lambda _root: False)
    return repo, _git(repo, "rev-parse", "HEAD")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_retarget_within_one_run_keeps_the_host_records_saved_baseline(tmp_path, monkeypatch):
    """One run arms B, then retargets to C (CP0 then the pull), and the checkout ends back on its
    start commit A. The second arm used to overwrite the saved baseline with "absent", so the
    disarm deleted the restart debt that stood before the run (review R1)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_bytes(b"PRIOR-DEBT")
    repo, _a = _repo_at_a(tmp_path, monkeypatch)
    commit.record_run_start(["git"], repo)

    commit.arm_commit_obligations(repo, "b" * 40)
    commit.arm_commit_obligations(repo, "c" * 40)
    commit.disarm_commit_obligations()

    assert host.read_bytes() == b"PRIOR-DEBT"


def test_a_later_runs_failed_retarget_puts_back_the_earlier_runs_debt(root):
    """Run 1 committed B and left its restart owed; run 2 (a new owner) armed C and failed before
    its move. Its release must put back run 1's record byte for byte, not delete it (review R1)."""
    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_bytes(b"PRIOR-DEBT")
    commit.arm_commit_obligations(root, "b" * 40)  # run 1: its move commits, nothing hands back
    run_one_record = host.read_bytes()
    commit.begin_update_attempt()

    commit.arm_commit_obligations(root, "c" * 40)  # run 2
    commit.disarm_commit_obligations()

    assert host.read_bytes() == run_one_record


def test_an_owner_leaving_a_shared_record_for_another_target_hands_the_others_stake_back(root, tmp_path):
    """Installs X and Y joined one record for SHA S; X then retargets to T and fails. Y's stake
    (and the first owner's baseline) comes back, and Y's own release then restores what stood
    before either of them (review R1)."""
    from hermes_cli.update_host_obligation import read_host_obligation

    host = host_obligation_path()
    host.parent.mkdir(parents=True, exist_ok=True)
    host.write_bytes(b"PRIOR-DEBT")
    other = tmp_path / "other-install"
    other.mkdir()
    commit.arm_commit_obligations(other, "a" * 40)  # install Y
    run_y = _run_state()
    commit.begin_update_attempt()
    commit.arm_commit_obligations(root, "a" * 40)  # install X joins
    commit.arm_commit_obligations(root, "d" * 40)  # X retargets
    commit.disarm_commit_obligations()

    record = read_host_obligation() or {}
    assert record.get("expected_sha") == "a" * 40 and record.get("owners") == [run_y["_owner"]]
    _enter_run(run_y)
    commit.disarm_commit_obligations()
    assert host.read_bytes() == b"PRIOR-DEBT"


def test_a_second_update_in_one_process_never_hands_back_the_first_updates_debt(root, monkeypatch):
    """Update 1 moved to B and left its tail and restart owed; update 2, in the same interpreter,
    armed C and was refused before its move. It used to reuse update 1's snapshot and owner and
    delete B's debt as its own undo (review O5). ``_cmd_update_impl`` starts a fresh attempt."""
    from types import SimpleNamespace

    from hermes_cli import update_cmd
    from hermes_cli.venv_sync import completion_pending_path

    commit.arm_commit_obligations(root, "b" * 40)  # update 1: committed, nothing handed back
    tail, host = completion_pending_path(root), host_obligation_path()
    owed = {tail: tail.read_bytes(), host: host.read_bytes()}

    class _Entered(Exception):
        pass

    def stop(_root):
        raise _Entered

    monkeypatch.setattr(update_cmd, "git_operation_in_progress", stop)
    with pytest.raises(_Entered):  # update 2's entry, stopped right after the attempt begins
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)
    commit.arm_commit_obligations(root, "c" * 40)
    commit.disarm_commit_obligations()

    assert {path: path.read_bytes() if path.exists() else None for path in owed} == owed


@pytest.mark.platforms("posix")  # unprivileged symlinks
@pytest.mark.parametrize("kind", ["regular", "symlink", "hardlink", "none"])
def test_disarm_restores_through_an_unpredictable_temp_and_leaves_every_sibling(root, tmp_path, kind):
    """The restore used a fixed ``.<name>.<pid>.restore`` temp and unlinked whatever stood there
    first: a user's ordinary file of that name was deleted (review O1). Nothing at the old name, or
    any alias target, is touched; the record still comes back."""
    from hermes_cli.venv_sync import completion_pending_path

    tail = completion_pending_path(root)
    tail.parent.mkdir(parents=True, exist_ok=True)
    tail.write_bytes(b"OLD-TAIL")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"PRECIOUS")
    commit.arm_commit_obligations(root, "a" * 40)
    planted = tail.with_name(f".{tail.name}.{os.getpid()}.restore")
    if kind == "regular":
        planted.write_bytes(b"USER-FILE")
    elif kind == "symlink":
        planted.symlink_to(sentinel)
    elif kind == "hardlink":
        os.link(sentinel, planted)

    commit.disarm_commit_obligations()

    assert tail.read_bytes() == b"OLD-TAIL" and not tail.is_symlink()
    assert sentinel.read_bytes() == b"PRECIOUS"
    if kind == "regular":
        assert planted.read_bytes() == b"USER-FILE"
    elif kind != "none":
        assert os.path.lexists(planted) and planted.read_bytes() == b"PRECIOUS"
    assert sorted(p.name for p in tail.parent.iterdir() if p.name.endswith(".restore")) == (
        [planted.name] if kind != "none" else [])


def test_an_arm_racing_a_release_is_never_undone_by_it(root, monkeypatch):
    """Run X's release judged the record (X its last owner, nothing found before it); another
    install armed SHA S2 before X's release acted, and X's unlink then deleted that fresh debt: a
    look-then-write with no compare-and-swap (kshitijk4poor F22/N05). Release and arm now judge and
    write under one mutex, so the racing arm lands after the release, never under it."""
    import threading

    from hermes_cli.update_host_obligation import read_host_obligation, write_host_obligation

    commit.arm_commit_obligations(root, "a" * 40)  # run X, over an absent record
    host, real_unlink, racer = host_obligation_path(), Path.unlink, []

    def unlink_after_a_racing_arm(self, *args, **kwargs):
        if self == host and not racer:
            racer.append(threading.Thread(target=write_host_obligation,
                                          kwargs={"expected_sha": "e" * 40, "owner": "other-install"}))
            racer[0].start()
            racer[0].join(timeout=1.0)  # under the mutex it can only wait for this release
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink_after_a_racing_arm)
    commit.disarm_commit_obligations()
    racer[0].join(timeout=15)
    monkeypatch.setattr(Path, "unlink", real_unlink)

    record = read_host_obligation() or {}
    assert record.get("expected_sha") == "e" * 40 and record.get("owners") == ["other-install"]
