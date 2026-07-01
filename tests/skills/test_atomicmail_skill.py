"""Smoke tests for the bundled atomicmail email skill."""
from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "email" / "atomicmail"
)


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    match = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "atomicmail"


def test_blueprint_present(frontmatter) -> None:
    blueprint = frontmatter["metadata"]["hermes"]["blueprint"]
    assert blueprint["schedule"] == "0 * * * *"
    assert blueprint["no_agent"] is False
    assert "list_inbox.json" in blueprint["prompt"]


def test_launcher_exists_and_is_executable() -> None:
    launcher = SKILL_DIR / "scripts" / "atomicmail"
    assert launcher.is_file()
    assert launcher.stat().st_mode & stat.S_IXUSR


def test_required_bundle_paths_exist() -> None:
    for rel in (
        "lib/esm/skill/cli.js",
        "lib/presets/list_inbox.json",
        "lib/presets/send_mail.json",
    ):
        assert (SKILL_DIR / rel).is_file(), rel


def test_skill_lists_himalaya_as_related(frontmatter) -> None:
    related = frontmatter["metadata"]["hermes"].get("related_skills") or []
    assert "himalaya" in related


def test_himalaya_defers_to_atomicmail() -> None:
    himalaya_md = (
        Path(__file__).resolve().parents[2]
        / "skills"
        / "email"
        / "himalaya"
        / "SKILL.md"
    ).read_text()
    assert "atomicmail" in himalaya_md.lower()
    related_match = re.search(
        r"related_skills:\s*\[(.*?)\]",
        himalaya_md,
        re.DOTALL,
    )
    assert related_match
    assert "atomicmail" in related_match.group(1)
