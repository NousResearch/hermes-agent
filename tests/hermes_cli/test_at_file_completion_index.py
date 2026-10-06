"""#127861: `@`-file completion answers a typed name query from the locate index, and keeps
the bounded walk for what an index cannot serve.

The walk was the picker's only source, and it is unbounded in cost: `rg --files
--sortr=modified` stats every file (19 s on the reporting host's 3.3 M-file root) and even
the unsorted walk needs 2 s+ to materialise there, so the completion timeout killed it and
the picker came back empty. Its page is also a sample — the first files the walk reaches —
so on a large root it cannot find a name outside that slice, which is what an index query
answers for the whole tree.
"""

from __future__ import annotations

import os
import time

import hermes_cli.commands_completion as cc
from tools import file_search_index as index_engine


class _StubRun:
    """``subprocess.run`` stand-in: records argv, answers ``(returncode, stdout)`` per
    engine name (plocate, rg, ...).

    The engine is named by the argv element that is one of the stubbed names, not by
    ``argv[0]``: the walk wraps itself in ``sh -c`` for the listing bound.
    """

    def __init__(self, answers: dict[str, tuple[int, str]]) -> None:
        self.commands: list[list[str]] = []
        self.engines: list[str] = []
        self.answers = answers

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        name = next((part for part in cmd if part in self.answers), os.path.basename(cmd[0]))
        self.engines.append(name)
        code, stdout = self.answers.get(name, (1, ""))
        return cc.subprocess.CompletedProcess(cmd, code, stdout, "")


def _stub_host(monkeypatch, tmp_path, answers: dict[str, tuple[int, str]]) -> tuple[_StubRun, str]:
    """Chdir into *tmp_path* with a fresh locate database, a PATH that has the engines and
    a recording ``subprocess.run``. Returns the recorder and the database path."""
    database = tmp_path / "locate.db"
    database.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_FILE_SEARCH_LOCATE_DB", str(database))
    monkeypatch.delenv("HERMES_FILE_SEARCH_ENGINE", raising=False)
    monkeypatch.setattr(index_engine, "which_exists", lambda name: name in {"plocate", "rg"})
    run = _StubRun(answers)
    monkeypatch.setattr(cc.subprocess, "run", run)
    return run, str(database)


def _texts(completer, word: str, query: str, limit: int = 5) -> list[str]:
    return [c.text for c in completer._fuzzy_file_completions(f"@{word}", query, limit)]


def test_typed_query_is_answered_by_the_index(tmp_path, monkeypatch):
    (tmp_path / "kept.py").write_text("x")
    (tmp_path / ".hidden.py").write_text("x")
    (tmp_path / "subdir").mkdir()
    hits = f"{tmp_path}/.hidden.py\n{tmp_path}/kept.py\n{tmp_path}/subdir\n"
    run, database = _stub_host(monkeypatch, tmp_path, {
        "plocate": (0, hits), "rg": (0, f"{tmp_path}/unused.py\n")})

    # Hidden names and directories are dropped: `rg --files` completes neither.
    assert _texts(cc.SlashCommandCompleter(), "kept", "kept") == ["@file:kept.py"]

    assert run.engines == ["plocate"]
    argv = run.commands[0]
    assert argv[:2] == ["plocate", "-d"] and argv[2] == database
    # The query is root-scoped and a substring, not a suffix: the name is still being typed.
    assert argv[-2:] == ["--", f"{tmp_path}/*kept*"]


def test_indexed_matches_are_memoised_per_query(tmp_path, monkeypatch):
    (tmp_path / "kept.py").write_text("x")
    run, _ = _stub_host(monkeypatch, tmp_path, {"plocate": (0, f"{tmp_path}/kept.py\n")})
    completer = cc.SlashCommandCompleter()

    assert _texts(completer, "kept", "kept") == ["@file:kept.py"]
    assert _texts(completer, "kept", "kept") == ["@file:kept.py"]

    assert run.engines == ["plocate"]


def test_empty_query_lists_from_the_walk(tmp_path, monkeypatch):
    """No query = recently modified files, which an index cannot order."""
    (tmp_path / "a.py").write_text("x")
    run, _ = _stub_host(monkeypatch, tmp_path, {
        "rg": (0, f"{tmp_path}/a.py\n"), "plocate": (0, f"{tmp_path}/a.py\n")})

    assert _texts(cc.SlashCommandCompleter(), "", "") == ["@file:a.py"]
    assert run.engines == ["rg"]


def test_query_the_index_cannot_answer_keeps_the_walk_page(tmp_path, monkeypatch):
    """A subsequence the index's substring query cannot express still completes from the
    walk page, so fuzzy matching is not lost."""
    (tmp_path / "commands_completion.py").write_text("x")
    run, _ = _stub_host(monkeypatch, tmp_path, {
        "plocate": (0, ""), "rg": (0, f"{tmp_path}/commands_completion.py\n")})

    assert _texts(cc.SlashCommandCompleter(), "cc", "cc") == ["@file:commands_completion.py"]
    assert run.engines == ["plocate", "rg"]


def test_worktree_keeps_the_walk_for_a_query(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.py").write_text("x")
    run, _ = _stub_host(monkeypatch, tmp_path, {
        "rg": (0, f"{tmp_path}/a.py\n"), "plocate": (0, f"{tmp_path}/a.py\n")})

    # The walk is bounded and ignore-filtered in a work-tree, so rg keeps that query.
    assert _texts(cc.SlashCommandCompleter(), "a", "a") == ["@file:a.py"]
    assert run.engines == ["rg"]


def test_stale_database_keeps_the_walk_for_a_query(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("x")
    run, database = _stub_host(monkeypatch, tmp_path, {
        "rg": (0, f"{tmp_path}/a.py\n"), "plocate": (0, f"{tmp_path}/a.py\n")})
    stale = time.time() - 30 * 24 * 3600
    os.utime(database, (stale, stale))

    assert _texts(cc.SlashCommandCompleter(), "a", "a") == ["@file:a.py"]
    assert run.engines == ["rg"]


def test_listing_is_newest_first(tmp_path, monkeypatch):
    old, new = tmp_path / "old.py", tmp_path / "new.py"
    old.write_text("x")
    new.write_text("x")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    _stub_host(monkeypatch, tmp_path, {"rg": (0, f"{old}\n{new}\n")})

    assert cc.SlashCommandCompleter()._get_project_files() == ["new.py", "old.py"]
