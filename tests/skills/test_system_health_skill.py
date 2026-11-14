"""Smoke tests for the system-health optional skill."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "system-health"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "system-health"


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "vijays365" in author, f"author should credit the contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


@pytest.mark.parametrize("path", ["scripts/system-health-report.py"])
def test_shipped_scripts_parse(path: str) -> None:
    src = (SKILL_DIR / path).read_text()
    ast.parse(src)


def test_script_imports_os_at_module_level() -> None:
    src = (SKILL_DIR / "scripts" / "system-health-report.py").read_text()
    tree = ast.parse(src)
    top_level_imports = {n.name for n in ast.walk(tree) if isinstance(n, ast.Import)}
    assert "os" in top_level_imports, "os must be imported at module level"


def test_script_has_no_local_os_import() -> None:
    """os should NOT be imported inside main() — that causes NameError in send_email()."""
    src = (SKILL_DIR / "scripts" / "system-health-report.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for child in ast.walk(node):
                if isinstance(child, ast.Import) and any(a.name == "os" for a in child.names):
                    pytest.fail("import os found inside main() — must be at module level")


def test_skill_has_pitfalls_section() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Common Pitfalls" in src or "## Pitfalls" in src
