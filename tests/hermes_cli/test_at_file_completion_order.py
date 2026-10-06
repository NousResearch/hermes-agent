"""Regression test: the `@`-file completion walk must neither stat the whole tree to
order its result nor buffer a whole-tree listing into the completion timeout.

#127861: the first command tried was `rg --files --sortr=modified`, which upgrades the
walk into a stat-per-file pass (70 s on a 3.3M-file root here versus 1.3 s unsorted),
and even the plain follow-up walk could not be materialised inside the 2 s timeout on
that root. Both were killed, so the file picker came back empty. The fix bounds the
listing at the source and orders the already-collected page in Python.
"""

from __future__ import annotations

import os

import hermes_cli.commands_completion as cc


class _RecordingRun:
    """``subprocess.run`` stand-in: records argv and replays a canned rg listing."""

    def __init__(self, output: str) -> None:
        self.commands: list[list[str]] = []
        self.output = output

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        return cc.subprocess.CompletedProcess(cmd, 0, self.output, "")


def _stub_engine(monkeypatch, tmp_path, output: str, *, posix_shell: bool = False) -> _RecordingRun:
    monkeypatch.chdir(tmp_path)
    available = {"rg", "sh", "head"} if posix_shell else {"rg"}
    monkeypatch.setattr(cc.shutil, "which", lambda name: f"/usr/bin/{name}" if name in available else None)
    run = _RecordingRun(output)
    monkeypatch.setattr(cc.subprocess, "run", run)
    return run


def test_project_file_walk_does_not_ask_rg_to_sort_by_mtime(tmp_path, monkeypatch):
    run = _stub_engine(monkeypatch, tmp_path, f"{tmp_path}/a.py\n")

    assert cc.SlashCommandCompleter()._get_project_files() == ["a.py"]

    assert run.commands[0][-3:] == ["rg", "--files", str(tmp_path)]
    assert not any("--sortr" in arg for command in run.commands for arg in command)


def test_project_file_walk_is_cut_off_at_the_listing_limit(tmp_path, monkeypatch):
    run = _stub_engine(monkeypatch, tmp_path, f"{tmp_path}/a.py\n", posix_shell=True)

    assert cc.SlashCommandCompleter()._get_project_files() == ["a.py"]

    argv = run.commands[0]
    assert argv[:2] == ["sh", "-c"]
    assert f'exec "$@" | head -n {cc._PROJECT_FILE_LIMIT}' == argv[2]
    assert argv[3:] == ["sh", "rg", "--files", str(tmp_path)]


def test_project_files_are_listed_newest_first(tmp_path, monkeypatch):
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text("x")
    new.write_text("x")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    _stub_engine(monkeypatch, tmp_path, f"{old}\n{new}\n")

    assert cc.SlashCommandCompleter()._get_project_files() == ["new.py", "old.py"]
