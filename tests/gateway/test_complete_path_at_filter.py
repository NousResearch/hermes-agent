"""Regression tests for the TUI gateway's `complete.path` handler.

Reported during the TUI v2 blitz retest:
  - typing `@folder:` (and `@folder` with no colon yet) surfaced files
    alongside directories — the gateway-side completion lives in
    `tui_gateway/server.py` and was never touched by the earlier fix to
    `hermes_cli/commands.py`.
  - typing `@appChrome` required the full `@ui-tui/src/components/app…`
    path to find the file — users expect Cmd-P-style fuzzy basename
    matching across the repo, not a strict directory prefix filter.

Covers:
  - `@folder:` only yields directories
  - `@file:` only yields regular files
  - Bare `@folder` / `@file` (no colon) lists cwd directly
  - Explicit prefix is preserved in the completion text
  - `@<name>` with no slash fuzzy-matches basenames anywhere in the tree
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway import server


def _fixture(tmp_path: Path):
    (tmp_path / "readme.md").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()


def _items(word: str):
    resp = server.handle_request({"id": "1", "method": "complete.path", "params": {"word": word}})

    return [(it["text"], it["display"], it.get("meta", "")) for it in resp["result"]["items"]]


@pytest.fixture(autouse=True)
def _reset_fuzzy_cache(monkeypatch):
    # Each test walks a fresh tmp dir; clear the cached listing so prior
    # roots can't leak through the TTL window.
    server._fuzzy_cache.clear()
    yield
    server._fuzzy_cache.clear()


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


# ── Fuzzy basename matching ──────────────────────────────────────────────
# Users shouldn't have to know the full path — typing `@appChrome` should
# find `ui-tui/src/components/appChrome.tsx`.


def _nested_fixture(tmp_path: Path):
    (tmp_path / "readme.md").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "ui-tui/src/components").mkdir(parents=True)
    (tmp_path / "ui-tui/src/components/appChrome.tsx").write_text("x")
    (tmp_path / "ui-tui/src/components/appLayout.tsx").write_text("x")
    (tmp_path / "ui-tui/src/components/thinking.tsx").write_text("x")
    (tmp_path / "ui-tui/src/hooks").mkdir(parents=True)
    (tmp_path / "ui-tui/src/hooks/useCompletion.ts").write_text("x")
    (tmp_path / "tui_gateway").mkdir()
    (tmp_path / "tui_gateway/server.py").write_text("x")


def test_fuzzy_at_finds_file_without_directory_prefix(tmp_path, monkeypatch):
    """`@appChrome` — with no slash — should surface the nested file."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    entries = _items("@appChrome")
    texts = [t for t, _, _ in entries]

    assert "@file:ui-tui/src/components/appChrome.tsx" in texts, texts

    # Display is the basename, meta is the containing directory, so the
    # picker can show `appChrome.tsx  ui-tui/src/components` on one row.
    row = next(r for r in entries if r[0] == "@file:ui-tui/src/components/appChrome.tsx")
    assert row[1] == "appChrome.tsx"
    assert row[2] == "ui-tui/src/components"


def test_fuzzy_ranks_exact_before_prefix_before_subseq(tmp_path, monkeypatch):
    """Better matches sort before weaker matches regardless of path depth."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)
    (tmp_path / "server.py").write_text("x")  # exact basename match at root

    texts = [t for t, _, _ in _items("@server")]

    # Exact `server.py` beats `tui_gateway/server.py` (prefix match) — both
    # rank 1 on basename but exact basename wins on the sort key; shorter
    # rel path breaks ties.
    assert texts[0] == "@file:server.py", texts
    assert "@file:tui_gateway/server.py" in texts


def test_fuzzy_camelcase_word_boundary(tmp_path, monkeypatch):
    """Mid-basename camelCase pieces match without substring scanning."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    texts = [t for t, _, _ in _items("@Chrome")]

    # `Chrome` starts a camelCase word inside `appChrome.tsx`.
    assert "@file:ui-tui/src/components/appChrome.tsx" in texts, texts


def test_fuzzy_subsequence_catches_sparse_queries(tmp_path, monkeypatch):
    """`@uCo` → `useCompletion.ts` via subsequence, last-resort tier."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    texts = [t for t, _, _ in _items("@uCo")]

    assert "@file:ui-tui/src/hooks/useCompletion.ts" in texts, texts


def test_fuzzy_at_file_prefix_preserved(tmp_path, monkeypatch):
    """Explicit `@file:` prefix still wins the completion tag."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    texts = [t for t, _, _ in _items("@file:appChrome")]

    assert "@file:ui-tui/src/components/appChrome.tsx" in texts, texts


def test_fuzzy_skipped_when_path_has_slash(tmp_path, monkeypatch):
    """Any `/` in the query = user is navigating; keep directory listing."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    texts = [t for t, _, _ in _items("@ui-tui/src/components/app")]

    # Directory-listing mode prefixes with `@file:` / `@folder:` per entry.
    # It should only surface direct children of the named dir — not the
    # nested `useCompletion.ts`.
    assert any("appChrome.tsx" in t for t in texts), texts
    assert not any("useCompletion.ts" in t for t in texts), texts


def test_fuzzy_skipped_when_folder_tag(tmp_path, monkeypatch):
    """`@folder:<name>` still lists directories — fuzzy scanner only walks
    files (git-tracked + untracked), so defer to the dir-listing path."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    texts = [t for t, _, _ in _items("@folder:ui")]

    # Root has `ui-tui/` as a directory; the listing branch should surface it.
    assert any(t.startswith("@folder:ui-tui") for t in texts), texts


def test_fuzzy_hides_dotfiles_unless_asked(tmp_path, monkeypatch):
    """`.env` doesn't leak into `@env` but does show for `@.env`."""
    monkeypatch.chdir(tmp_path)
    _nested_fixture(tmp_path)

    assert not any(".env" in t for t, _, _ in _items("@env"))
    assert any(t.endswith(".env") for t, _, _ in _items("@.env"))


def test_fuzzy_caps_results(tmp_path, monkeypatch):
    """The 30-item cap survives a big tree."""
    monkeypatch.chdir(tmp_path)
    for i in range(60):
        (tmp_path / f"mod_{i:03d}.py").write_text("x")

    items = _items("@mod")

    assert len(items) == 30


def test_fuzzy_paths_relative_to_cwd_inside_subdir(tmp_path, monkeypatch):
    """When the gateway runs from a subdirectory of a git repo, fuzzy
    completion paths must resolve under that cwd — not under the repo root.

    Without this, `@appChrome` from inside `apps/web/` would suggest
    `@file:apps/web/src/foo.tsx` but the agent (resolving from cwd) would
    look for `apps/web/apps/web/src/foo.tsx` and fail. We translate every
    `git ls-files` result back to a `relpath(root)` and drop anything
    outside `root` so the completion contract stays "paths are cwd-relative".
    """
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)

    (tmp_path / "apps" / "web" / "src").mkdir(parents=True)
    (tmp_path / "apps" / "web" / "src" / "appChrome.tsx").write_text("x")
    (tmp_path / "apps" / "api" / "src").mkdir(parents=True)
    (tmp_path / "apps" / "api" / "src" / "server.ts").write_text("x")
    (tmp_path / "README.md").write_text("x")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    # Run from `apps/web/` — completions should be relative to here, and
    # files outside this subtree (apps/api, README.md at root) shouldn't
    # appear at all.
    monkeypatch.chdir(tmp_path / "apps" / "web")

    texts = [t for t, _, _ in _items("@appChrome")]

    assert "@file:src/appChrome.tsx" in texts, texts
    assert not any("apps/web/" in t for t in texts), texts

    server._fuzzy_cache.clear()
    other_texts = [t for t, _, _ in _items("@server")]

    assert not any("server.ts" in t for t in other_texts), other_texts

    server._fuzzy_cache.clear()
    readme_texts = [t for t, _, _ in _items("@README")]

    assert not any("README.md" in t for t in readme_texts), readme_texts
