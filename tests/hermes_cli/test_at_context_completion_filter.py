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

from hermes_cli.commands import SlashCommandCompleter


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


def test_at_file_only_yields_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@file:")]

    assert all(t.startswith("@file:") for t in texts), texts
    assert any(t == "@file:readme.md" for t in texts)
    assert any(t == "@file:.env" for t in texts)
    assert not any(t == "@file:src/" for t in texts)
    assert not any(t == "@file:docs/" for t in texts)


def test_at_folder_preserves_prefix_on_empty_match(tmp_path, monkeypatch):
    """User typed `@folder:` (no partial) — completion text must keep the
    `@folder:` prefix even though the previous implementation auto-rewrote
    it to `@file:` for non-dir entries.
    """
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@folder:")]

    assert texts, "expected at least one directory completion"
    for t in texts:
        assert t.startswith("@folder:"), f"prefix leaked: {t}"


def test_at_folder_bare_without_colon_lists_directories(tmp_path, monkeypatch):
    """Typing `@folder` alone (no colon yet) should surface directories so
    users don't need to first accept the static `@folder:` hint before
    seeing what they're picking from.
    """
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@folder")]

    assert any(t == "@folder:src/" for t in texts), texts
    assert any(t == "@folder:docs/" for t in texts), texts
    assert not any(t == "@folder:readme.md" for t in texts)


def test_at_file_bare_without_colon_lists_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _ in _run(tmp_path, "@file")]

    assert any(t == "@file:readme.md" for t in texts), texts
    assert not any(t == "@file:src/" for t in texts)
