"""Contract tests for the pytorch-fsdp optional skill."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mlops"
    / "pytorch-fsdp"
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    match = re.search(r"^---\n(.*?)\n---", skill_text, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_frontmatter_matches_optional_skill(frontmatter: dict) -> None:
    assert frontmatter["name"] == SKILL_DIR.name
    assert frontmatter["license"] == "MIT"
    assert set(frontmatter["platforms"]) == {"linux", "macos"}


def test_description_meets_hardline(frontmatter: dict) -> None:
    description = frontmatter["description"]
    assert len(description) <= 60
    assert description.endswith(".")


def test_author_preserves_provenance(frontmatter: dict) -> None:
    author = frontmatter["author"]
    assert "yinjianxxx" in author
    assert "Orchestra Research" in author


def test_main_skill_stays_loadable_and_readable(skill_text: str) -> None:
    assert len(skill_text) <= 100_000
    assert max(map(len, skill_text.splitlines())) < 500


@pytest.mark.parametrize(
    "section",
    [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ],
)
def test_modern_sections_are_present(skill_text: str, section: str) -> None:
    assert section in skill_text


def test_main_skill_routes_detail_to_references(skill_text: str) -> None:
    assert "references/index.md" in skill_text
    assert "references/other.md" in skill_text
    assert "generated from official documentation" not in skill_text.lower()
    assert "re-run the scraper" not in skill_text.lower()


def test_reference_index_maps_the_archive() -> None:
    index = (SKILL_DIR / "references" / "index.md").read_text(encoding="utf-8")
    archive = (SKILL_DIR / "references" / "other.md").read_text(encoding="utf-8")

    for heading in (
        "torch.distributed.fsdp.fully_shard",
        "FullyShardedDataParallel",
        "Distributed Checkpoint - torch.distributed.checkpoint",
    ):
        assert heading in index
        assert heading in archive


def test_current_official_docs_are_linked(skill_text: str) -> None:
    assert "docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html" in skill_text
    assert "docs.pytorch.org/docs/stable/fsdp.html" in skill_text
    assert "docs.pytorch.org/docs/stable/distributed.checkpoint.html" in skill_text
