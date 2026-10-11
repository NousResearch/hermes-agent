"""Regression test: `@folder:` completion must only surface directories and
`@file:` must only surface regular files.

Reported during TUI v2 blitz testing: typing `@folder:` showed .dockerignore,
.env, .gitignore, etc. alongside the actual directories because the path-
completion branch yielded every entry regardless of the explicit prefix, and
auto-switched the completion kind based on `is_dir`. That defeated the user's
explicit choice and rendered the `@folder:` / `@file:` prefixes useless for
filtering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hermes_cli.commands_completion import SlashCommandCompleter


def _run(tmp_path: Path, word: str) -> list[tuple[str, str]]:
    (tmp_path / "readme.md").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()

    completer = SlashCommandCompleter.__new__(SlashCommandCompleter)
    completions: Iterable = completer._context_completions(word)

    return [(c.text, c.display_meta) for c in completions if c.text.startswith(("@file:", "@folder:"))]


def test_at_folder_only_yields_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@folder:")]

    assert all(t.startswith("@folder:") for t in texts), texts
    assert any(t == "@folder:src/" for t in texts)
    assert any(t == "@folder:docs/" for t in texts)
    assert not any(t == "@folder:readme.md" for t in texts)
    assert not any(t == "@folder:.env" for t in texts)






def test_at_file_bare_without_colon_lists_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@file")]

    assert any(t == "@file:readme.md" for t in texts), texts
    assert not any(t == "@file:src/" for t in texts)


def test_project_file_completion_does_not_request_global_mtime_sort(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hermes_cli.commands_completion.shutil.which", lambda name: "/usr/bin/rg")
    calls = []

    class Result:
        returncode = 0
        stdout = "src/main.py\n"

    def run(command, **kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr("hermes_cli.commands_completion.subprocess.run", run)
    completer = SlashCommandCompleter.__new__(SlashCommandCompleter)
    completer._file_cache = []
    completer._file_cache_cwd = None
    completer._file_cache_time = 0.0

    assert completer._get_project_files() == ["src/main.py"]
    assert calls == [["rg", "--files", str(tmp_path)]]
