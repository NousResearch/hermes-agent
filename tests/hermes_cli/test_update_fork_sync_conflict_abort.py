"""Tests for merge-conflict recovery in the fork upstream sync.

When ``hermes update`` runs on a diverged fork, ``_sync_with_upstream_if_needed``
merges ``upstream/main`` into local ``main``. If that merge conflicts, the sync
is skipped — but the failed merge must be aborted first. A mid-merge tree
carries conflict markers in critical files (``hermes_cli/main.py``,
``apps/desktop/package.json``), which makes the CLI unbootable and fails every
later update stage (npm install, web UI build, desktop rebuild).

Reference incident: 2026-06-10 in-app update on the OmarB97 fork left
``~/.hermes/hermes-agent`` mid-merge (25 conflicted files); the desktop app
showed "Update didn't finish" and ``hermes`` itself raised SyntaxError on
the orphan ``<<<<<<< HEAD`` marker at main.py:471.
"""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import main as hermes_main


class _FakeGit:
    """Scripted ``subprocess.run`` for the diverged-fork sync flow.

    Plays a fork that is 57 ahead / 85 behind upstream so the sync takes the
    merge path, then fails the merge with a configurable outcome.
    """

    def __init__(self, *, merge_rc=1, merge_head_rc=0, abort_rc=0):
        self.calls: list[list[str]] = []
        self.merge_rc = merge_rc
        self.merge_head_rc = merge_head_rc
        self.abort_rc = abort_rc

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        args = list(cmd[1:])  # strip the "git" executable
        if args[:3] == ["remote", "get-url", "upstream"]:
            return SimpleNamespace(
                returncode=0, stdout="https://github.com/NousResearch/hermes-agent.git\n", stderr=""
            )
        if args[:2] == ["fetch", "upstream"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["rev-list", "--count"]:
            counts = {
                "upstream/main..origin/main": "57",
                "origin/main..upstream/main": "85",
            }
            return SimpleNamespace(returncode=0, stdout=counts[args[2]] + "\n", stderr="")
        if args[:2] == ["checkout", "main"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["merge", "upstream/main"]:
            return SimpleNamespace(
                returncode=self.merge_rc,
                stdout="",
                stderr="CONFLICT (content): Merge conflict in hermes_cli/main.py\n",
            )
        if args[0] == "rev-parse" and args[-1] == "MERGE_HEAD":
            return SimpleNamespace(returncode=self.merge_head_rc, stdout="", stderr="")
        if args[:2] == ["merge", "--abort"]:
            return SimpleNamespace(returncode=self.abort_rc, stdout="", stderr="")
        if args[0] == "push":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected git command: {cmd}")

    def issued(self, *suffix):
        return any(call[1 : 1 + len(suffix)] == list(suffix) for call in self.calls)


def test_conflicted_merge_is_aborted_and_tree_restored(monkeypatch, tmp_path, capsys):
    fake = _FakeGit(merge_rc=1, merge_head_rc=0, abort_rc=0)
    monkeypatch.setattr(hermes_main.subprocess, "run", fake)

    hermes_main._sync_with_upstream_if_needed(["git"], tmp_path)

    assert fake.issued("merge", "--abort")
    assert not fake.issued("push")
    out = capsys.readouterr().out
    assert "Merge failed (conflict)" in out
    assert "Working tree restored" in out


def test_failed_abort_warns_instead_of_claiming_restored(monkeypatch, tmp_path, capsys):
    fake = _FakeGit(merge_rc=1, merge_head_rc=0, abort_rc=1)
    monkeypatch.setattr(hermes_main.subprocess, "run", fake)

    hermes_main._sync_with_upstream_if_needed(["git"], tmp_path)

    assert fake.issued("merge", "--abort")
    out = capsys.readouterr().out
    assert "Could not abort the merge" in out
    assert "Working tree restored" not in out


def test_merge_error_without_merge_in_progress_skips_abort(monkeypatch, tmp_path, capsys):
    # e.g. merge refused before starting: no MERGE_HEAD, nothing to abort.
    fake = _FakeGit(merge_rc=1, merge_head_rc=1)
    monkeypatch.setattr(hermes_main.subprocess, "run", fake)

    hermes_main._sync_with_upstream_if_needed(["git"], tmp_path)

    assert not fake.issued("merge", "--abort")
    out = capsys.readouterr().out
    assert "Merge failed (conflict)" in out
    assert "Could not abort" not in out
