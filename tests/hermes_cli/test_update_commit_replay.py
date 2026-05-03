from subprocess import CompletedProcess

from hermes_cli.main import (
    UpdateReplayResult,
    UpdateSnapshot,
    replay_missing_update_commits,
)


def _cp(cmd, stdout="", stderr="", returncode=0):
    return CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _snapshot(ahead_commits):
    return UpdateSnapshot(
        head_before="abc123deadbeef",
        branch_before="main",
        upstream_ref="origin/main",
        upstream_head_before="def456deadbeef",
        status_short="",
        ahead_commits=ahead_commits,
        ahead_count=len(ahead_commits),
        dirty_tree=False,
    )


def test_replay_missing_commits_returns_success_when_no_rescue_ref(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)
    monkeypatch.setattr("hermes_cli.main._update_commit_patch_id", lambda *args, **kwargs: None)

    result = replay_missing_update_commits(["git"], tmp_path, _snapshot(["a"]), None)

    assert result == UpdateReplayResult([], [], None, True)
    assert calls == []


def test_replay_missing_commits_replays_only_plus_commits_in_original_order(
    monkeypatch, tmp_path
):
    snapshot = _snapshot(["c1", "c2", "c3"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"]:
            return _cp(cmd, stdout="+ c3\n- c2\n+ c1\n")
        if cmd == ["git", "cherry-pick", "c1"]:
            return _cp(cmd)
        if cmd == ["git", "cherry-pick", "c3"]:
            return _cp(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)
    monkeypatch.setattr("hermes_cli.main._update_commit_patch_id", lambda *args, **kwargs: None)

    result = replay_missing_update_commits(
        ["git"], tmp_path, snapshot, "refs/hermes/update-rescue/test"
    )

    assert result.succeeded is True
    assert result.replayed_commits == ["c1", "c3"]
    assert result.skipped_commits == ["c2"]
    assert result.conflicted_commit is None
    assert calls == [
        ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"],
        ["git", "cherry-pick", "c1"],
        ["git", "cherry-pick", "c3"],
    ]


def test_replay_missing_commits_skips_all_minus_commits(monkeypatch, tmp_path):
    snapshot = _snapshot(["a", "b", "c"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"]:
            return _cp(cmd, stdout="- c\n- a\n- b\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)
    monkeypatch.setattr("hermes_cli.main._update_commit_patch_id", lambda *args, **kwargs: None)

    result = replay_missing_update_commits(
        ["git"], tmp_path, snapshot, "refs/hermes/update-rescue/test"
    )

    assert result.succeeded is True
    assert result.replayed_commits == []
    assert result.skipped_commits == ["a", "b", "c"]
    assert result.conflicted_commit is None
    assert calls == [["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"]]


def test_replay_missing_commits_aborts_and_stops_on_conflict(monkeypatch, tmp_path):
    snapshot = _snapshot(["a", "b"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"]:
            return _cp(cmd, stdout="+ a\n+ b\n")
        if cmd == ["git", "cherry-pick", "a"]:
            return _cp(cmd)
        if cmd == ["git", "cherry-pick", "b"]:
            return _cp(cmd, returncode=1, stderr="conflict")
        if cmd == ["git", "cherry-pick", "--abort"]:
            return _cp(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)
    monkeypatch.setattr("hermes_cli.main._update_commit_patch_id", lambda *args, **kwargs: None)

    result = replay_missing_update_commits(
        ["git"], tmp_path, snapshot, "refs/hermes/update-rescue/test"
    )

    assert result.succeeded is False
    assert result.replayed_commits == ["a"]
    assert result.skipped_commits == []
    assert result.conflicted_commit == "b"
    assert calls == [
        ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"],
        ["git", "cherry-pick", "a"],
        ["git", "cherry-pick", "b"],
        ["git", "cherry-pick", "--abort"],
    ]


def test_replay_missing_commits_skips_duplicate_patch_ids(monkeypatch, tmp_path):
    snapshot = _snapshot(["a", "b", "c"])
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"]:
            return _cp(cmd, stdout="+ a\n+ b\n+ c\n")
        if cmd == ["git", "cherry-pick", "a"]:
            return _cp(cmd)
        if cmd == ["git", "cherry-pick", "c"]:
            return _cp(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)
    monkeypatch.setattr(
        "hermes_cli.main._update_commit_patch_id",
        lambda git_cmd, cwd, commit: {"a": "pid-1", "b": "pid-1", "c": "pid-2"}[commit],
    )

    result = replay_missing_update_commits(
        ["git"], tmp_path, snapshot, "refs/hermes/update-rescue/test"
    )

    assert result.succeeded is True
    assert result.replayed_commits == ["a", "c"]
    assert result.skipped_commits == ["b"]
    assert result.conflicted_commit is None
    assert calls == [
        ["git", "cherry", "origin/main", "refs/hermes/update-rescue/test"],
        ["git", "cherry-pick", "a"],
        ["git", "cherry-pick", "c"],
    ]
