"""Contract tests for the bundled uv-toolchain skill."""

from __future__ import annotations

import re
from pathlib import Path
import pytest
import yaml

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "uv-toolchain"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_uv_toolchain_skill_exists() -> None:
    assert SKILL_MD.is_file()


def test_uv_toolchain_frontmatter(skill_text: str) -> None:
    assert skill_text.startswith("---")
    m = re.search(r"\n---\s*\n", skill_text[3:])
    assert m is not None
    fm = yaml.safe_load(skill_text[3 : m.start() + 3])
    assert isinstance(fm, dict)

    assert fm.get("name") == "uv-toolchain"
    desc = fm.get("description", "")
    assert len(desc) <= 60
    assert desc.endswith(".")

    assert fm.get("version") == "1.0.0"
    assert "author" in fm
    assert fm.get("license") == "MIT"
    assert "linux" in fm.get("platforms", [])
    assert "macos" in fm.get("platforms", [])
    assert "windows" in fm.get("platforms", [])

    hermes_meta = fm.get("metadata", {}).get("hermes", {})
    assert "uv" in hermes_meta.get("tags", [])
    assert "python" in hermes_meta.get("tags", [])
    assert hermes_meta.get("category") == "software-development"


def test_uv_toolchain_content_guidelines(skill_text: str) -> None:
    # Must document PEP 723 and core uv commands
    assert "# /// script" in skill_text
    assert "uv run" in skill_text
    assert "uvx" in skill_text
    assert "uv venv" in skill_text
    assert "uv pip" in skill_text
