from subprocess import CompletedProcess

from hermes_cli.main import (
    UpdateReplayResult,
    UpdateSnapshot,
    _collect_update_final_report,
    _print_update_final_report,
)



def _cp(cmd, stdout="", stderr="", returncode=0):
    return CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)



def test_collect_update_final_report_builds_expected_mapping(monkeypatch, tmp_path):
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
    replay_result = UpdateReplayResult(
        skipped_commits=["c2"],
        replayed_commits=["c1"],
        conflicted_commit=None,
        succeeded=True,
    )

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "rev-parse", "HEAD"]:
            return _cp(cmd, stdout="fedcba987654\n")
        if cmd == ["git", "status", "--short", "--branch"]:
            return _cp(
                cmd,
                stdout="## main...origin/main [ahead 1]\n M hermes_cli/main.py\n",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("hermes_cli.main.subprocess.run", fake_run)

    report = _collect_update_final_report(
        ["git"],
        tmp_path,
        snapshot,
        "refs/hermes/update-rescue/test",
        replay_result,
        "refs/stash@{0}",
        "restarted: hermes-gateway.service",
    )

    assert report["old_head"] == "abc123deadbeef"
    assert report["new_head"] == "fedcba987654"
    assert report["target_ref"] == "origin/main"
    assert report["rescue_ref"] == "refs/hermes/update-rescue/test"
    assert report["local_commits_preserved"] == ["c1", "c2"]
    assert report["local_commits_skipped"] == ["c2"]
    assert report["local_commits_replayed"] == ["c1"]
    assert report["replay_conflict_commit"] is None
    assert report["autostash_ref_preserved"] == "refs/stash@{0}"
    assert report["final_git_status"] == "## main...origin/main [ahead 1]\n M hermes_cli/main.py"
    assert report["gateway_service_health"] == "restarted: hermes-gateway.service"



def test_print_update_final_report_outputs_expected_labels(capsys):
    report = {
        "old_head": "abc123deadbeef",
        "new_head": "fedcba987654",
        "target_ref": "origin/main",
        "rescue_ref": "refs/hermes/update-rescue/test",
        "local_commits_preserved": ["c1", "c2"],
        "local_commits_skipped": ["c2"],
        "local_commits_replayed": ["c1"],
        "replay_conflict_commit": None,
        "autostash_ref_preserved": "refs/stash@{0}",
        "final_git_status": "## main...origin/main [ahead 1]\n M hermes_cli/main.py",
        "gateway_service_health": "restarted: hermes-gateway.service",
    }

    _print_update_final_report(report)

    out = capsys.readouterr().out
    assert "Update report" in out
    assert "Old HEAD:" in out
    assert "New HEAD:" in out
    assert "Target ref:" in out
    assert "Rescue ref:" in out
    assert "Local commits preserved:" in out
    assert "Local commits skipped as upstream-equivalent:" in out
    assert "Local commits replayed:" in out
    assert "Replay conflict commit:" in out
    assert "Autostash ref preserved:" in out
    assert "Final git status:" in out
    assert "Gateway/service health:" in out
