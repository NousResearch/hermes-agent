"""Tests for modular SOUL.md include/import expansion."""

from pathlib import Path

import pytest

from agent.soul_includes import (
    MAX_DEPTH,
    MAX_FILES,
    expand_soul_includes,
    has_soul_include_directive,
    parse_include_directive,
)


def _scan(content: str, label: str) -> str:
    if "MALICIOUS" in content:
        return f"[BLOCKED: {label} contained potential prompt injection (test). Content not loaded.]"
    return content


def _expand(root: Path, home: Path, *, max_depth: int = MAX_DEPTH, max_files: int = MAX_FILES) -> str:
    return expand_soul_includes(
        root.read_text(encoding="utf-8").strip(),
        root_path=root,
        hermes_home=home,
        scan_content=_scan,
        max_depth=max_depth,
        max_files=max_files,
    )


def test_parse_include_directive_accepts_include_and_import():
    assert parse_include_directive("@include soul/style.md") == "soul/style.md"
    assert parse_include_directive("@import soul/rules/*.md") == "soul/rules/*.md"


def test_parse_include_directive_requires_column_zero_line_only():
    assert parse_include_directive("Use @include soul/style.md") is None
    assert parse_include_directive("  @include soul/style.md") is None
    assert parse_include_directive("@include") is None


def test_has_soul_include_directive_ignores_fenced_code_and_escaped_lines():
    content = """Root
```md
@include soul/example.md
```
   ```md
@include soul/indented.md
   ```
~~~
```
@include soul/mismatched.md
~~~
\\@include soul/literal.md
"""
    assert has_soul_include_directive(content) is False


def test_inline_indented_fenced_and_escaped_include_lines_remain_literal(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "soul").mkdir()
    (home / "soul" / "style.md").write_text("SHOULD_NOT_LOAD", encoding="utf-8")
    root.write_text(
        "Use @include soul/style.md here\n"
        "  @include soul/style.md\n"
        "```md\n@include soul/style.md\n```\n"
        "   ```md\n@include soul/style.md\n   ```\n"
        "~~~\n```\n@include soul/style.md\n~~~\n"
        "\\@include soul/style.md\n",
        encoding="utf-8",
    )

    expanded = _expand(root, home)

    assert "SHOULD_NOT_LOAD" not in expanded
    assert "Use @include soul/style.md here" in expanded
    assert "  @include soul/style.md" in expanded
    assert "```md\n@include soul/style.md\n```" in expanded
    assert "   ```md\n@include soul/style.md\n   ```" in expanded
    assert "~~~\n```\n@include soul/style.md\n~~~" in expanded
    assert "\\@include soul/style.md" in expanded


def test_resolve_exact_markdown_relative_to_home(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "soul").mkdir()
    (home / "soul" / "style.md").write_text("Be concise.", encoding="utf-8")
    root.write_text("Root\n@include soul/style.md\nEnd", encoding="utf-8")

    expanded = _expand(root, home)

    assert expanded == "Root\nBe concise.\nEnd"


def test_nested_include_resolves_relative_to_hermes_home_not_fragment_dir(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "soul" / "nested").mkdir(parents=True)
    (home / "shared.md").write_text("Shared from home", encoding="utf-8")
    (home / "soul" / "nested" / "shared.md").write_text("Wrong relative file", encoding="utf-8")
    (home / "soul" / "nested" / "a.md").write_text("A\n@include shared.md", encoding="utf-8")
    root.write_text("@include soul/nested/a.md", encoding="utf-8")

    expanded = _expand(root, home)

    assert "Shared from home" in expanded
    assert "Wrong relative file" not in expanded


def test_resolve_glob_sorted_lexicographically(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    rules = home / "rules"
    rules.mkdir()
    (rules / "20.md").write_text("twenty", encoding="utf-8")
    (rules / "10.md").write_text("ten", encoding="utf-8")
    root.write_text("@include rules/*.md", encoding="utf-8")

    assert _expand(root, home) == "ten\ntwenty"


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("/tmp/outside.md", "absolute"),
        ("~/outside.md", "tilde"),
        ("../outside.md", "parent traversal"),
        ("secret.txt", "not a markdown"),
        ("rules/**/*.md", "recursive glob"),
        ("missing.md", "no include matches"),
    ],
)
def test_unsafe_or_missing_targets_render_placeholders(tmp_path: Path, target: str, message: str):
    home = tmp_path
    root = home / "SOUL.md"
    root.write_text(f"before\n@include {target}\nafter", encoding="utf-8")

    expanded = _expand(root, home)

    assert "before" in expanded
    assert "after" in expanded
    assert "[INCLUDE ERROR: SOUL.md:2:" in expanded
    assert message in expanded


def test_expand_unmatched_glob_renders_source_line_placeholder(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "rules").mkdir()
    root.write_text("before\n@include rules/*.md\nafter", encoding="utf-8")

    first = _expand(root, home)
    second = _expand(root, home)

    assert first == second
    assert "before" in first
    assert "after" in first
    assert "[INCLUDE ERROR: SOUL.md:2: no include matches: rules/*.md]" in first


def test_include_read_error_renders_source_line_placeholder(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "bad.md").write_bytes(b"\xff\xfe\x00")
    root.write_text("before\n@include bad.md\nafter", encoding="utf-8")

    expanded = _expand(root, home)

    assert "before" in expanded
    assert "after" in expanded
    assert "[INCLUDE ERROR: SOUL.md:2: could not read include bad.md: UnicodeDecodeError]" in expanded


def test_resolve_rejects_symlink_escape(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evil.md").write_text("outside", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / "soul").mkdir()
    root = home / "SOUL.md"
    link = home / "soul" / "link.md"
    try:
        link.symlink_to(outside / "evil.md")
    except OSError as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"symlink unavailable: {exc}")
    root.write_text("@include soul/link.md", encoding="utf-8")

    expanded = _expand(root, home)

    assert "outside" not in expanded
    assert "symlink escapes HERMES_HOME" in expanded


def test_expand_self_cycle_placeholder(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    root.write_text("A\n@include SOUL.md\nB", encoding="utf-8")

    expanded = _expand(root, home)

    assert "A" in expanded
    assert "B" in expanded
    assert "cycle detected" in expanded


def test_expand_indirect_cycle_placeholder(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "a.md").write_text("A\n@include b.md", encoding="utf-8")
    (home / "b.md").write_text("B\n@include a.md", encoding="utf-8")
    root.write_text("@include a.md", encoding="utf-8")

    expanded = _expand(root, home)

    assert "A" in expanded
    assert "B" in expanded
    assert "cycle detected" in expanded


def test_expand_depth_limit_placeholder_and_asserts_max_depth_constant(tmp_path: Path):
    assert MAX_DEPTH == 16
    home = tmp_path
    root = home / "SOUL.md"
    (home / "a.md").write_text("A\n@include b.md", encoding="utf-8")
    (home / "b.md").write_text("B", encoding="utf-8")
    root.write_text("@include a.md", encoding="utf-8")

    expanded = _expand(root, home, max_depth=1)

    assert "A" in expanded
    assert "max include depth exceeded" in expanded
    assert "\nB" not in expanded


def test_expand_file_count_limit_placeholder_and_asserts_max_files_constant(tmp_path: Path):
    assert MAX_FILES == 128
    home = tmp_path
    root = home / "SOUL.md"
    (home / "a.md").write_text("A", encoding="utf-8")
    (home / "b.md").write_text("B", encoding="utf-8")
    root.write_text("@include a.md\n@include b.md", encoding="utf-8")

    expanded = _expand(root, home, max_files=2)

    assert "A" in expanded
    assert "max include files exceeded" in expanded
    assert "\nB" not in expanded


def test_expand_scans_root_before_resolving_directives(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    root.write_text("MALICIOUS\n@include missing.md", encoding="utf-8")

    expanded = _expand(root, home)

    assert expanded.startswith("[BLOCKED: SOUL.md")
    assert "missing.md" not in expanded


def test_expand_scans_included_file_before_nested_expansion(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    (home / "safe.md").write_text("SHOULD_NOT_LOAD", encoding="utf-8")
    (home / "bad.md").write_text("MALICIOUS\n@include safe.md", encoding="utf-8")
    root.write_text("@include bad.md", encoding="utf-8")

    expanded = _expand(root, home)

    assert "[BLOCKED: bad.md" in expanded
    assert "SHOULD_NOT_LOAD" not in expanded


def test_repeated_expansion_is_byte_identical_for_same_inputs(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    rules = home / "rules"
    rules.mkdir()
    (rules / "b.md").write_text("B", encoding="utf-8")
    (rules / "a.md").write_text("A", encoding="utf-8")
    root.write_text("Root\n@include rules/*.md", encoding="utf-8")

    assert _expand(root, home) == _expand(root, home)


def test_repeated_expansion_is_byte_identical_for_placeholder_cases(tmp_path: Path):
    home = tmp_path
    root = home / "SOUL.md"
    root.write_text("@include missing.md", encoding="utf-8")

    assert _expand(root, home) == _expand(root, home)
