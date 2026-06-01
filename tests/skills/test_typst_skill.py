"""Smoke tests for the typst optional skill.

The typst skill wraps a binary we can't bundle (`typst` from the Typst
project, ~30 MB). These tests verify the skill conforms to the hardline
format, the shipped scripts parse as valid Python, and the scaffolder
produces a runnable project from its CLI. We do not invoke `typst` itself.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "typst"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "typst"


def test_platforms_covers_all_three_os(frontmatter) -> None:
    # Typst ships a single Rust binary for linux/macos/windows. No reason to
    # gate the skill by OS.
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}, (
        f"expected all three platforms, got {frontmatter['platforms']}"
    )


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Thomas Bale" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


@pytest.mark.parametrize(
    "path",
    [
        "scripts/new_project.py",
        "scripts/compile.py",
    ],
)
def test_shipped_scripts_parse(path: str) -> None:
    src = (SKILL_DIR / path).read_text()
    ast.parse(src)  # raises SyntaxError on broken Python


def test_shipped_templates_exist() -> None:
    templates_dir = SKILL_DIR / "templates"
    assert templates_dir.is_dir(), f"missing templates dir: {templates_dir}"
    assert (templates_dir / "minimal.typ").is_file(), "missing templates/minimal.typ"


@pytest.mark.parametrize(
    "path",
    [
        "templates/minimal.typ",
    ],
)
def test_shipped_templates_have_typst_set_directives(path: str) -> None:
    """Every shipped .typ template should at least configure a page and font,
    so the user has a working starting point and not a blank page.
    """
    src = (SKILL_DIR / path).read_text()
    assert "#set page" in src, f"{path} missing `#set page` directive"
    assert "#set text" in src, f"{path} missing `#set text` directive"


def test_new_project_script_help_exits_zero() -> None:
    """`python new_project.py --help` must succeed (stdlib argparse path)."""
    script = SKILL_DIR / "scripts" / "new_project.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"new_project.py --help failed: {result.returncode}\n{result.stderr}"
    )
    assert "scaffold" in result.stdout.lower() or "scaffold" in result.stdout, (
        "expected --help output to mention 'scaffold'"
    )


def test_new_project_script_creates_directory(tmp_path) -> None:
    """End-to-end smoke: run the scaffolder against a tmp dir and confirm the
    expected files appear. Does not invoke `typst` itself.
    """
    script = SKILL_DIR / "scripts" / "new_project.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "demo-paper",
            "--type",
            "article",
            "--path",
            str(tmp_path),
            "--no-gitignore",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"new_project.py failed: {result.returncode}\n{result.stderr}"
    )
    project = tmp_path / "demo-paper"
    assert project.is_dir()
    assert (project / "main.typ").is_file()
    assert (project / "refs.bib").is_file()
    assert (project / "figures").is_dir()
    main_src = (project / "main.typ").read_text()
    # The article template should at least mention the document title setter
    # and the section directive so the user has a clear starting point.
    assert "#set document" in main_src
    assert "= " in main_src, "article template missing at least one section heading"


def test_new_project_script_refuses_overwrite(tmp_path) -> None:
    """Scaffolder must not silently clobber an existing directory."""
    script = SKILL_DIR / "scripts" / "new_project.py"
    target = tmp_path / "existing"
    target.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "existing",
            "--type",
            "article",
            "--path",
            str(tmp_path),
            "--no-gitignore",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0, "expected non-zero exit when target exists"
    assert "refusing to overwrite" in result.stderr


def test_compile_script_self_test_handles_missing_binary(tmp_path, monkeypatch) -> None:
    """`compile.py --self-test` must fail gracefully (exit 1, helpful stderr)
    when `typst` is not on PATH. We force the failure by pointing PATH at an
    empty directory.
    """
    script = SKILL_DIR / "scripts" / "compile.py"
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    monkeypatch.setenv("PATH", str(empty_path))
    result = subprocess.run(
        [sys.executable, str(script), "--self-test"],
        capture_output=True,
        text=True,
        timeout=15,
        # Don't inherit the test process's PATH — we want a clean view.
        env={"PATH": str(empty_path)},
    )
    assert result.returncode == 1
    assert "typst binary not found" in result.stderr
