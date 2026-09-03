"""Diverged-fork sync skip must report the true upstream state (#100646).

When origin/main carries commits that are not on upstream/main, fork sync
skips to preserve them. The skip note used to hide two facts: how far behind
upstream the fork actually is, and that a fork sharing no merge-base with
upstream can never converge via the suggested ``git pull upstream main`` —
with no common ancestor a pull fails or unions two unrelated histories.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import update_cmd


def _run_fork_sync(tmp_path, *, origin_ahead, upstream_ahead, merge_base_rc):
    counts = {
        ("upstream/main", "origin/main"): origin_ahead,
        ("origin/main", "upstream/main"): upstream_ahead,
    }
    runs = [
        SimpleNamespace(returncode=0, stdout="", stderr=""),  # fetch upstream main
        SimpleNamespace(returncode=merge_base_rc, stdout="", stderr=""),  # merge-base
    ]
    with patch.object(
        update_cmd, "_has_upstream_remote", return_value=True
    ), patch.object(
        update_cmd,
        "_count_commits_between",
        side_effect=lambda _cmd, _cwd, base, head: counts[(base, head)],
    ), patch.object(update_cmd.subprocess, "run", side_effect=runs):
        return update_cmd._sync_with_upstream_if_needed(["git"], tmp_path)


class TestForkSyncSkipReportsDivergence:
    def test_skip_reports_upstream_delta_and_keeps_pull_hint(
        self, tmp_path, capsys
    ):
        checked = _run_fork_sync(
            tmp_path, origin_ahead=1, upstream_ahead=34, merge_base_rc=0
        )

        assert checked is True
        out = capsys.readouterr().out
        assert "Your fork has 1 commit(s) not on upstream" in out
        assert "also 34 commit(s) behind upstream" in out
        assert "git pull upstream main" in out
        assert "reset --hard" not in out

    def test_skip_without_merge_base_recommends_reset_not_pull(
        self, tmp_path, capsys
    ):
        checked = _run_fork_sync(
            tmp_path, origin_ahead=1, upstream_ahead=34, merge_base_rc=1
        )

        assert checked is True
        out = capsys.readouterr().out
        assert "share no merge-base" in out
        assert "git reset --hard upstream/main" in out
        assert "git pull upstream main" not in out

    def test_skip_with_fork_current_upstream_mentions_no_delta(
        self, tmp_path, capsys
    ):
        checked = _run_fork_sync(
            tmp_path, origin_ahead=2, upstream_ahead=0, merge_base_rc=0
        )

        assert checked is True
        out = capsys.readouterr().out
        assert "Your fork has 2 commit(s) not on upstream" in out
        assert "behind upstream" not in out
        assert "git pull upstream main" in out
