"""Regression tests for the TUI gateway's `complete.path` handler.

Reported during the TUI v2 blitz retest: typing `@folder:` (and `@folder`
with no colon yet) still surfaced files alongside directories in the
TUI composer, because the gateway-side completion lives in
`tui_gateway/server.py` and was never touched by the earlier fix to
`hermes_cli/commands.py`.

Covers:
  - `@folder:` only yields directories
  - `@file:` only yields regular files
  - Bare `@folder` / `@file` (no colon) lists cwd directly
  - Explicit prefix is preserved in the completion text
"""

from __future__ import annotations

from pathlib import Path

from tui_gateway import server


def _fixture(tmp_path: Path):
    (tmp_path / "readme.md").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()


def _items(word: str):
    resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": word}})

    return [(it["text"], it["display"], it.get("meta", "")) for it in resp["result"]["items"]]


def test_at_folder_colon_only_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)

    texts = [t for t, _, _ in _items("@folder:")]

    assert all(t.startswith("@folder:") for t in texts), texts
    assert any(t == "@folder:src/" for t in texts)
    assert any(t == "@folder:docs/" for t in texts)
    assert not any(t == "@folder:readme.md" for t in texts)
    assert not any(t == "@folder:.env" for t in texts)


def test_at_file_colon_only_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)

    texts = [t for t, _, _ in _items("@file:")]

    assert all(t.startswith("@file:") for t in texts), texts
    assert any(t == "@file:readme.md" for t in texts)
    assert not any(t == "@file:src/" for t in texts)
    assert not any(t == "@file:docs/" for t in texts)


def test_at_folder_bare_without_colon_lists_dirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)

    texts = [t for t, _, _ in _items("@folder")]

    assert any(t == "@folder:src/" for t in texts), texts
    assert any(t == "@folder:docs/" for t in texts), texts
    assert not any(t == "@folder:readme.md" for t in texts)


def test_at_file_bare_without_colon_lists_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fixture(tmp_path)

    texts = [t for t, _, _ in _items("@file")]

    assert any(t == "@file:readme.md" for t in texts), texts
    assert not any(t == "@file:src/" for t in texts)


def test_bare_at_still_shows_static_refs(tmp_path, monkeypatch):
    """`@` alone should list the static references so users discover the
    available prefixes.  (Unchanged behaviour; regression guard.)
    """
    monkeypatch.chdir(tmp_path)

    texts = [t for t, _, _ in _items("@")]

    for expected in ("@diff", "@staged", "@file:", "@folder:", "@url:", "@git:"):
        assert expected in texts, f"missing static ref {expected!r} in {texts!r}"
