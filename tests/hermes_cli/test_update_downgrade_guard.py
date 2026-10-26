"""`hermes update` refuses a target that predates the gateway runtime once a state.db holds ledger rows.

Downgrading is unsupported (#106742, jquesnelle item 6): the older release cannot delete or prune
the sessions with runtime history, so the move is refused before the tree changes and the message
names the pre-update snapshot that predates the cutover.
"""
import json
import sqlite3
import subprocess
from contextlib import closing
from unittest.mock import Mock

import pytest

import hermes_state_runtime as rt
from hermes_cli import update_downgrade_guard as guard
from hermes_state import SessionDB


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "T")
    (root / "hermes_state_common.py").write_text("SCHEMA_SQL = 'CREATE TABLE IF NOT EXISTS sessions (id TEXT)'\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "release before the cutover")
    _git(root, "tag", "old-release")
    (root / "hermes_state_common.py").write_text(
        "SCHEMA_SQL = '''CREATE TABLE IF NOT EXISTS session_admissions (seq INTEGER)'''\n")
    _git(root, "commit", "-am", "gateway runtime")
    return root


def _ledgered_store(home):
    home.mkdir(parents=True, exist_ok=True)
    with closing(SessionDB(home / "state.db")) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id="owner")
        db.create_session("s", source="cli")
        rt.admit_session_input(db, epoch=epoch, principal_id="p", session_id="s", request_id="r",
                               payload={"text": "x"})


def _pre_cutover_snapshot(home, snap_id):
    snap = home / "state-snapshots" / snap_id
    snap.mkdir(parents=True)
    with closing(sqlite3.connect(snap / "state.db")) as conn:
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")


def test_a_pre_runtime_target_is_refused_before_the_move_with_the_snapshot_that_predates_it(
        repo, tmp_path, monkeypatch, capsys):
    from hermes_cli import main as hermes_main
    from hermes_cli import update_cmd

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.backup._sibling_profile_homes", lambda h: [])
    monkeypatch.setattr(update_cmd._commit, "preflight_refusal", lambda *a, **k: None)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    _ledgered_store(home)
    _pre_cutover_snapshot(home, "20261001-000000-pre-update")
    # A newer snapshot already holding the runtime ledger is not a way back.
    newer = home / "state-snapshots" / "20261008-000000-pre-update"
    newer.mkdir()
    with closing(sqlite3.connect(newer / "state.db")) as conn:
        conn.execute("CREATE TABLE session_admissions (seq INTEGER)")
    head = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(SystemExit) as stopped:
        update_cmd._refuse_before_commit_point(["git"], _git(repo, "rev-parse", "old-release"), None)
    assert stopped.value.code == 1 and _git(repo, "rev-parse", "HEAD") == head
    out = capsys.readouterr().out
    assert "Downgrading is unsupported" in out and str(home / "state.db") in out
    assert "/snapshot restore 20261001-000000-pre-update" in out and "20261008" not in out
    assert "hermes gateway stop" in out and "No update was applied" in out


def test_forward_updates_and_stores_without_ledger_rows_are_never_refused(repo, tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.backup._sibling_profile_homes", lambda h: [])
    old = _git(repo, "rev-parse", "old-release")
    plain = tmp_path / "plain"
    plain.mkdir()
    SessionDB(plain / "state.db").close()  # runtime tables exist, but nothing was ever admitted
    assert guard.downgrade_refusal(["git"], repo, old, home=plain) is None

    ledgered = tmp_path / "ledgered"
    _ledgered_store(ledgered)
    head = _git(repo, "rev-parse", "HEAD")
    assert guard.downgrade_refusal(["git"], repo, head, home=ledgered) is None
    _git(repo, "checkout", "-q", "--detach", old)  # running the old release: the newer target is forward
    assert guard.downgrade_refusal(["git"], repo, head, home=ledgered) is None


@pytest.mark.parametrize("divergent", [True, False], ids=["beside-history", "in-history"])
def test_check_holds_a_release_that_update_would_refuse(repo, tmp_path, monkeypatch, capsys, divergent):
    """`hermes update --check` must neither offer nor promise a move back to a release the update then
    refuses (#139099): one cut beside this checkout's history, or one it already contains."""
    from types import SimpleNamespace

    from hermes_cli import source_releases, update_cmd_check

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.backup._sibling_profile_homes", lambda h: [])
    release = _git(repo, "rev-parse", "old-release")
    if divergent:
        _git(repo, "checkout", "-q", "-b", "release-line", "old-release")
        (repo / "notes.txt").write_text("release fix\n")
        _git(repo, "add", "notes.txt")
        _git(repo, "commit", "-qm", "release from the older line")
        release = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-q", "main")
    target = SimpleNamespace(commit=release, retired=False, ahead=False, label="v1.2.3", branch=None,
                             repository="NousResearch/hermes-agent")
    monkeypatch.setattr(source_releases, "resolve_source_target", lambda *a, **k: target)

    update_cmd_check.channel_compare_branch("stable", ["git"], repo)
    # No runtime history yet: a real update, or a real rollback.
    assert ("Selected release available" if divergent else "moves it back") in capsys.readouterr().out

    _ledgered_store(home)
    update_cmd_check.channel_compare_branch("stable", ["git"], repo)
    out = capsys.readouterr().out
    assert "Holding at this build" in out
    assert "Selected release available" not in out and "moves it back" not in out


def test_the_runtime_schema_marker_is_still_in_this_trees_state_schema():
    from hermes_state_common import SCHEMA_SQL

    assert guard.defines_runtime([SCHEMA_SQL])


@pytest.mark.parametrize("remote_result", [True, False, None, "no-reader"])
@pytest.mark.parametrize("runtime", [False, True], ids=["pre-runtime", "runtime"])
def test_present_commit_with_missing_schema_blobs_uses_remote_without_git_writes(
        repo, tmp_path, monkeypatch, remote_result, runtime):
    from hermes_cli.source_check import source_git_env

    # Release commits change the version, not the schema: pathspec history simplification must
    # not skip their inherited schema objects (Git 2.53 does; newer Git can mask the regression).
    _git(repo, "checkout", "-q", "-b", "release-line", "HEAD" if runtime else "old-release")
    (repo / "VERSION").write_text("1.2.3\n", encoding="utf-8")
    _git(repo, "add", "VERSION")
    _git(repo, "commit", "-qm", "release version bump")
    target = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    (repo / "hermes_state_common.py").write_text(
        "SCHEMA_SQL = '''CREATE TABLE IF NOT EXISTS session_admissions (seq INTEGER, extra TEXT)'''\n",
        encoding="utf-8")
    _git(repo, "commit", "-qam", "newer runtime schema")
    _git(repo, "config", "uploadpack.allowFilter", "true")
    clone = tmp_path / "partial"
    _git(tmp_path, "clone", "--filter=blob:none", "--no-checkout", repo.as_uri(), str(clone))
    trace = tmp_path / "git-trace.jsonl"
    monkeypatch.setenv("GIT_TRACE2_EVENT", str(trace))

    def probe(*args):
        return subprocess.run(["git", *args], cwd=clone, env=source_git_env(),
                              capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=10, check=False)

    def git(*args):
        return probe(*args).returncode

    before = {p.relative_to(clone): p.read_bytes() for p in (clone / ".git").rglob("*") if p.is_file()}
    assert git("cat-file", "-e", f"{target}^{{commit}}") == 0
    assert git("cat-file", "-e", f"{target}:hermes_state_common.py") != 0
    grep = probe("grep", "-q", "-F", guard._RUNTIME_SCHEMA_MARKER, target, "--", guard.STATE_MODULES)
    assert grep.returncode != 0 and grep.stderr
    remote = None if remote_result == "no-reader" else Mock(return_value=remote_result)
    assert guard.target_predates_runtime(None, clone, target, git=git, remote=remote) is (remote_result is True)
    if remote is not None:
        remote.assert_called_once_with(target)
    assert {p.relative_to(clone): p.read_bytes() for p in (clone / ".git").rglob("*") if p.is_file()} == before
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert not any(event["event"] == "child_start" for event in events)


@pytest.mark.parametrize("grep_error", [False, True])
def test_local_schema_decides_without_remote_unless_the_probe_fails(repo, grep_error):
    runtime = _git(repo, "rev-parse", "HEAD")
    (repo / "notes.txt").write_text("later build\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-qm", "later build")
    remote = Mock(side_effect=lambda ref: ref == "old-release")
    run = guard._updater_git(["git"], repo)

    def git(*args):
        return -1 if grep_error and args[0] == "grep" else run(*args)

    assert guard.target_predates_runtime(None, repo, "old-release", git=git, remote=remote)
    assert not guard.target_predates_runtime(None, repo, runtime, git=git, remote=remote)
    assert not guard.target_predates_runtime(None, repo, "HEAD", git=git, remote=remote)
    assert remote.call_count == (2 if grep_error else 0)
