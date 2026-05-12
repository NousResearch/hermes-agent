"""Tests for agent/context_includes.py — the pure @-include expander."""

from pathlib import Path

import pytest

from agent.context_includes import (
    CONTEXT_INCLUDE_MAX_DEPTH,
    INCLUDE_PATTERN,
    expand_includes,
    resolve_include_path,
)


# ---------------------------------------------------------------------------
# resolve_include_path
# ---------------------------------------------------------------------------


class TestResolveIncludePath:
    def test_absolute_path_kept_absolute(self, tmp_path):
        target = tmp_path / "f.md"
        out = resolve_include_path(str(target), Path("/some/other/dir"))
        assert out == target.resolve()

    def test_relative_path_resolves_against_base_dir(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        out = resolve_include_path("file.md", tmp_path)
        assert out == (tmp_path / "file.md").resolve()

    def test_tilde_expands_to_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        out = resolve_include_path("~/x.md", Path("/irrelevant"))
        assert out == (tmp_path / "x.md").resolve()

    def test_env_var_expands(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MYDIR", str(tmp_path))
        out = resolve_include_path("$MYDIR/x.md", Path("/irrelevant"))
        assert out == (tmp_path / "x.md").resolve()


# ---------------------------------------------------------------------------
# INCLUDE_PATTERN regex
# ---------------------------------------------------------------------------


class TestIncludePattern:
    def test_matches_line_only_token(self):
        assert INCLUDE_PATTERN.search("@/abs/path.md")

    def test_matches_with_leading_whitespace(self):
        assert INCLUDE_PATTERN.search("    @/abs/path.md")

    def test_does_not_match_inline(self):
        assert INCLUDE_PATTERN.search("see @bob for more") is None

    def test_does_not_match_with_trailing_text(self):
        assert INCLUDE_PATTERN.search("@/abs/path.md and more") is None


# ---------------------------------------------------------------------------
# expand_includes
# ---------------------------------------------------------------------------


class TestExpandIncludes:
    def test_no_includes_returns_unchanged(self, tmp_path):
        out = expand_includes("hello world", tmp_path)
        assert out == "hello world"

    def test_simple_include(self, tmp_path):
        target = tmp_path / "t.md"
        target.write_text("INCLUDED")
        out = expand_includes(f"before\n@{target}\nafter", tmp_path)
        assert "INCLUDED" in out
        assert "before" in out and "after" in out

    def test_nested_includes(self, tmp_path):
        c = tmp_path / "c.md"
        c.write_text("C")
        b = tmp_path / "b.md"
        b.write_text(f"B-pre\n@{c}\nB-post")
        a = tmp_path / "a.md"
        a.write_text(f"A\n@{b}")
        out = expand_includes(a.read_text(), tmp_path)
        assert "A" in out and "B-pre" in out and "B-post" in out and "C" in out

    def test_cycle_terminates(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(f"A\n@{b}")
        b.write_text(f"B\n@{a}")
        out = expand_includes(a.read_text(), tmp_path)
        assert "A" in out and "B" in out
        assert "@cycle" in out

    def test_missing_file_marker(self, tmp_path):
        out = expand_includes("@/no/such/file.md", tmp_path)
        assert "@missing" in out

    def test_max_depth_marker(self, tmp_path):
        # Build a chain longer than the depth cap.
        chain_len = CONTEXT_INCLUDE_MAX_DEPTH + 2
        files = [tmp_path / f"f{i}.md" for i in range(chain_len)]
        for i, f in enumerate(files):
            if i + 1 < chain_len:
                f.write_text(f"f{i}\n@{files[i+1]}")
            else:
                f.write_text(f"f{i}-LAST")
        out = expand_includes(f"@{files[0]}", tmp_path)
        assert "f0" in out
        assert f"f{chain_len-1}-LAST" not in out
        assert "@max-depth" in out

    def test_custom_max_depth(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text(f"A\n@{b}")
        b.write_text("B")
        # depth=0 means even the first include is over the budget.
        out = expand_includes(a.read_text(), tmp_path, max_depth=0)
        assert "@max-depth" in out
        assert "B" not in out

    def test_code_fence_is_inert(self, tmp_path):
        decoy = tmp_path / "decoy.md"
        decoy.write_text("DECOY-CONTENT")
        src = (
            "Example:\n"
            "```\n"
            f"@{decoy}\n"
            "```\n"
            "End."
        )
        out = expand_includes(src, tmp_path)
        assert "DECOY-CONTENT" not in out
        assert f"@{decoy}" in out

    def test_tilde_fence_is_inert(self, tmp_path):
        decoy = tmp_path / "d.md"
        decoy.write_text("DECOY")
        src = f"~~~\n@{decoy}\n~~~"
        out = expand_includes(src, tmp_path)
        assert "DECOY" not in out

    def test_scanner_invoked_on_included_chunk(self, tmp_path):
        target = tmp_path / "t.md"
        target.write_text("payload")
        captured: list[tuple[str, str]] = []

        def scanner(content: str, label: str) -> str:
            captured.append((content, label))
            return content + "::scanned"

        out = expand_includes(f"@{target}", tmp_path, scanner=scanner)
        assert "payload::scanned" in out
        assert any("payload" in c for c, _ in captured)

    def test_truncator_invoked_on_included_chunk(self, tmp_path):
        target = tmp_path / "t.md"
        target.write_text("BIG-PAYLOAD")

        def truncator(content: str, label: str) -> str:
            return f"[truncated:{label}]"

        out = expand_includes(f"@{target}", tmp_path, truncator=truncator)
        assert "BIG-PAYLOAD" not in out
        assert "[truncated:" in out

    def test_relative_path_resolves_against_included_dir(self, tmp_path):
        """Once we recurse into sub/a.md, its `@b.md` must resolve in sub/."""
        sub = tmp_path / "sub"
        sub.mkdir()
        b = sub / "b.md"
        b.write_text("SUB-B")
        a = sub / "a.md"
        a.write_text("SUB-A\n@b.md")
        out = expand_includes(f"@{a}", tmp_path)
        assert "SUB-A" in out and "SUB-B" in out

    def test_directory_target_treated_as_missing(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        out = expand_includes(f"@{d}", tmp_path)
        assert "@missing" in out

    def test_include_markers_present(self, tmp_path):
        target = tmp_path / "t.md"
        target.write_text("X")
        out = expand_includes(f"@{target}", tmp_path)
        assert "@include-begin" in out
        assert "@include-end" in out
