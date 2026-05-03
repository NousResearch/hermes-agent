from subprocess import CompletedProcess
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.main import (
    UpdateSnapshot,
    collect_update_snapshot,
    cmd_update,
    create_update_rescue_ref,
)


def _cp(cmd, stdout="", stderr="", returncode=0):
    return CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def test_collect_update_snapshot_detects_no_ahead_commits(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return _cp(cmd, stdout="main\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _cp(cmd, stdout="abc123\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return _cp(cmd, stdout="def456\n")
        if cmd == ["git", "status", "--porcelain"]:
            return _cp(cmd, stdout="")
        if cmd == ["git", "rev-list", "--reverse", "origin/main..HEAD"]:
            return _cp(cmd, stdout="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    snapshot = collect_update_snapshot(["git"], tmp_path)

    assert snapshot.branch_before == "main"
    assert snapshot.head_before == "abc123"
    assert snapshot.upstream_head_before == "def456"
    assert snapshot.status_short == ""
    assert snapshot.dirty_tree is False
    assert snapshot.ahead_commits == []
    assert snapshot.ahead_count == 0
    assert calls == [
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "origin/main"],
        ["git", "status", "--porcelain"],
        ["git", "rev-list", "--reverse", "origin/main..HEAD"],
    ]


def test_collect_update_snapshot_detects_ordered_ahead_commits(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return _cp(cmd, stdout="main\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _cp(cmd, stdout="abc123\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return _cp(cmd, stdout="def456\n")
        if cmd == ["git", "status", "--porcelain"]:
            return _cp(cmd, stdout="")
        if cmd == ["git", "rev-list", "--reverse", "origin/main..HEAD"]:
            return _cp(cmd, stdout="c1\nc2\nc3\n")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    snapshot = collect_update_snapshot(["git"], tmp_path)

    assert snapshot.ahead_commits == ["c1", "c2", "c3"]
    assert snapshot.ahead_count == 3
    assert snapshot.upstream_ref == "origin/main"


def test_collect_update_snapshot_records_dirty_tree_state(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return _cp(cmd, stdout="main\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _cp(cmd, stdout="abc123\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            return _cp(cmd, stdout="def456\n")
        if cmd == ["git", "status", "--porcelain"]:
            return _cp(cmd, stdout=" M hermes_cli/main.py\n?? notes.txt\n")
        if cmd == ["git", "rev-list", "--reverse", "origin/main..HEAD"]:
            return _cp(cmd, stdout="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    snapshot = collect_update_snapshot(["git"], tmp_path)

    assert snapshot.status_short == "M hermes_cli/main.py\n?? notes.txt"
    assert snapshot.dirty_tree is True


def test_create_update_rescue_ref_returns_none_when_no_ahead_commits(monkeypatch, tmp_path):
    snapshot = UpdateSnapshot(
        head_before="abc123deadbeef",
        branch_before="main",
        upstream_ref="origin/main",
        upstream_head_before="def456deadbeef",
        status_short="",
        ahead_commits=[],
        ahead_count=0,
        dirty_tree=False,
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    assert create_update_rescue_ref(["git"], tmp_path, snapshot) is None
    assert calls == []


def test_create_update_rescue_ref_creates_ref_for_ahead_commits(monkeypatch, tmp_path):
    snapshot = UpdateSnapshot(
        head_before="abc123deadbeef",
        branch_before="main",
        upstream_ref="origin/main",
        upstream_head_before="def456deadbeef",
        status_short="",
        ahead_commits=["c1", "c2"],
        ahead_count=2,
        dirty_tree=False,
    )
    calls = []
    created_ref = {}

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "update-ref"]:
            created_ref["value"] = cmd[2]
            assert cmd[3] == snapshot.head_before
            return _cp(cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    ref = create_update_rescue_ref(["git"], tmp_path, snapshot)

    assert ref is not None
    assert ref.startswith("refs/hermes/update-rescue/")
    assert ref.endswith("-abc123de")
    assert created_ref["value"] == ref
    assert calls == [["git", "update-ref", ref, snapshot.head_before]]


@patch("shutil.which")
@patch("hermes_cli.main._install_hangup_protection", return_value={})
@patch("hermes_cli.main._finalize_update_output")
@patch("hermes_cli.main._stash_local_changes_if_needed", return_value=None)
@patch("hermes_cli.main._invalidate_update_cache")
@patch("hermes_cli.main._clear_bytecode_cache", return_value=0)
@patch("hermes_cli.main._install_python_dependencies_with_optional_fallback")
@patch("hermes_cli.main._update_node_dependencies")
@patch("hermes_cli.main._build_web_ui")
@patch("hermes_cli.main._sync_with_upstream_if_needed")
@patch("hermes_cli.main.collect_update_snapshot")
@patch("hermes_cli.main.create_update_rescue_ref")
@patch("hermes_cli.main.subprocess.run")
def test_update_preflight_runs_before_destructive_update_operation(
    mock_run,
    mock_create_rescue_ref,
    mock_collect_snapshot,
    mock_sync_upstream,
    mock_build_web,
    mock_update_node_deps,
    mock_install_python_deps,
    mock_clear_bytecode,
    mock_invalidate_cache,
    mock_stash,
    mock_finalize_output,
    mock_install_hangup,
    mock_which,
    monkeypatch,
):
    order = []
    mock_which.side_effect = lambda name: "/usr/bin/uv" if name == "uv" else None
    mock_collect_snapshot.side_effect = lambda git_cmd, cwd: order.append("collect") or UpdateSnapshot(
        head_before="abc123deadbeef",
        branch_before="feature/topic",
        upstream_ref="origin/main",
        upstream_head_before="def456deadbeef",
        status_short="",
        ahead_commits=["c1"],
        ahead_count=1,
        dirty_tree=False,
    )
    mock_create_rescue_ref.side_effect = lambda git_cmd, cwd, snapshot, prefix="refs/hermes/update-rescue": order.append("rescue") or "refs/hermes/update-rescue/20260101-000000-abc123de"

    def fake_run(cmd, **kwargs):
        cmd_str = " ".join(str(part) for part in cmd)
        if cmd == ["git", "fetch", "origin"]:
            order.append("fetch")
            return _cp(cmd)
        if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            order.append("branch")
            return _cp(cmd, stdout="feature/topic\n")
        if cmd == ["git", "rev-parse", "HEAD"]:
            order.append("head")
            return _cp(cmd, stdout="abc123deadbeef\n")
        if cmd == ["git", "rev-parse", "origin/main"]:
            order.append("origin")
            return _cp(cmd, stdout="def456deadbeef\n")
        if cmd == ["git", "status", "--porcelain"]:
            order.append("status")
            return _cp(cmd, stdout="")
        if cmd == ["git", "status", "--short", "--branch"]:
            order.append("status-branch")
            return _cp(cmd, stdout="## main...origin/main [ahead 1]\n")
        if cmd == ["git", "rev-list", "--reverse", "origin/main..HEAD"]:
            order.append("ahead")
            return _cp(cmd, stdout="c1\n")
        if cmd == ["git", "update-ref", "refs/hermes/update-rescue/20260101-000000-abc123de", "abc123deadbeef"]:
            order.append("update-ref")
            return _cp(cmd)
        if cmd == ["git", "cherry", "origin/main", "refs/hermes/update-rescue/20260101-000000-abc123de"]:
            order.append("cherry")
            return _cp(cmd, stdout="- c1\n")
        if cmd == ["git", "checkout", "main"]:
            order.append("checkout")
            return _cp(cmd)
        if cmd == ["git", "checkout", "feature/topic"]:
            order.append("restore-branch")
            return _cp(cmd)
        if cmd == ["git", "rev-list", "HEAD..origin/main", "--count"]:
            order.append("count")
            return _cp(cmd, stdout="1\n")
        if cmd == ["git", "pull", "--ff-only", "origin", "main"]:
            order.append("pull")
            return _cp(cmd)
        raise AssertionError(f"unexpected command: {cmd_str}")

    mock_run.side_effect = fake_run

    args = SimpleNamespace(gateway=False)
    cmd_update(args)

    assert order.index("checkout") < order.index("collect")
    assert order.index("collect") < order.index("rescue")
    assert order.index("checkout") < order.index("pull")
    mock_collect_snapshot.assert_called_once()
    mock_create_rescue_ref.assert_called_once()
    mock_stash.assert_called_once()
    mock_sync_upstream.assert_not_called()
