"""Contract tests for the OrbStack operator optional skill."""

from pathlib import Path
import re

import pytest


REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "devops" / "orbstack-operator"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert match, "SKILL.md must have closed YAML frontmatter at byte zero"
    return match.group(1)


def _scalar(frontmatter: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", frontmatter, re.MULTILINE)
    assert match, f"missing frontmatter field: {key}"
    return match.group(1).strip()


def test_frontmatter_contract(frontmatter: str) -> None:
    assert _scalar(frontmatter, "name") == "orbstack-operator"
    description = _scalar(frontmatter, "description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert _scalar(frontmatter, "version") == "0.1.0"
    assert _scalar(frontmatter, "license") == "MIT"
    assert _scalar(frontmatter, "platforms") == "[macos]"
    assert "Thomas Oertel (tomraider4720), Hermes Agent" in _scalar(
        frontmatter, "author"
    )


def test_related_skill_resolves(frontmatter: str) -> None:
    assert "related_skills: [docker-management]" in frontmatter
    assert (
        REPO / "optional-skills" / "devops" / "docker-management" / "SKILL.md"
    ).is_file()


def test_operator_covers_all_orbstack_surfaces(skill_text: str) -> None:
    contracts = (
        "orb list",
        "docker context show",
        "orb logs docker",
        "orb start k8s",
        "kubectl config current-context",
        "host.docker.internal",
        "host.orb.internal",
        "docker.orb.internal",
    )
    missing = [contract for contract in contracts if contract not in skill_text]
    assert not missing, f"missing operator contracts: {missing}"


def test_destructive_operations_are_confirmation_gated(skill_text: str) -> None:
    destructive = ("orb reset", "orb delete docker", "orb delete k8s")
    for command in destructive:
        assert command in skill_text
    assert "require explicit confirmation" in skill_text
    assert "verify the archive exists before deletion" in skill_text


def test_uses_only_official_orbstack_references(skill_text: str) -> None:
    urls = re.findall(r"https://[^)\s]+", skill_text)
    assert urls
    assert all(url.startswith("https://docs.orbstack.dev/") for url in urls)
