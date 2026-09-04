"""Tests for the optional-skills/web-development/publish-site skill.

Structural + internal-consistency checks only (stdlib + pytest, no network).
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "web-development" / "publish-site"
SKILL_MD = SKILL_DIR / "SKILL.md"

VALID_PLATFORMS = {"linux", "macos", "windows"}


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text: str) -> dict:
    """Parse the YAML frontmatter with stdlib only (flat key: value fields)."""
    m = re.match(r"^---\n(.*?)\n---\n", skill_text, re.DOTALL)
    assert m, "SKILL.md must open with '---' delimited YAML frontmatter"
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if line.startswith((" ", "\t")) or ":" not in line:
            continue  # nested metadata keys — not needed here
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip()
    return fm


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_frontmatter_parses(frontmatter: dict):
    assert frontmatter.get("name") == "publish-site"
    assert frontmatter.get("version"), "version field required"
    assert frontmatter.get("license") == "MIT"
    assert "Hermes Agent" in frontmatter.get("author", "")


def test_description_length_and_period(frontmatter: dict):
    desc = frontmatter.get("description", "").strip().strip('"')
    assert desc, "no description field"
    assert len(desc) <= 60, f"description is {len(desc)} chars (>60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"


def test_platforms_list_valid(frontmatter: dict):
    raw = frontmatter.get("platforms", "")
    platforms = [p.strip() for p in raw.strip("[]").split(",") if p.strip()]
    assert platforms, "platforms list must be non-empty"
    assert set(platforms) <= VALID_PLATFORMS, f"invalid platforms: {platforms}"
    # gh/wrangler/netlify are all cross-platform — the skill keeps all three.
    assert set(platforms) == VALID_PLATFORMS


def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"


def test_no_dangling_skill_view_references(skill_text: str):
    """Every skill_view(name='...') reference must target a shipped skill."""
    targets = re.findall(r"skill_view\(\s*name\s*=\s*['\"]([^'\"]+)['\"]", skill_text)
    for name in targets:
        hits = list((REPO / "skills").glob(f"**/{name}/SKILL.md")) + list(
            (REPO / "optional-skills").glob(f"**/{name}/SKILL.md")
        )
        assert hits, f"skill_view reference to non-shipped skill: {name}"


def test_referenced_sibling_skills_ship(skill_text: str):
    """Prose-referenced companion skills must exist in the shipped trees."""
    for name in ("cloudflare-temporary-deploy",):
        if name in skill_text:
            hits = list((REPO / "skills").glob(f"**/{name}/SKILL.md")) + list(
                (REPO / "optional-skills").glob(f"**/{name}/SKILL.md")
            )
            assert hits, f"referenced skill does not ship: {name}"


def test_provider_ladder_documented(skill_text: str):
    """All three rungs of the provider ladder with their deploy commands."""
    assert "gh repo create" in skill_text
    assert "gh-pages" in skill_text
    assert "wrangler@latest pages deploy" in skill_text
    assert "netlify deploy --prod" in skill_text


def test_version_before_deploy_discipline(skill_text: str):
    assert "git tag" in skill_text, "deploys must be tagged"
    assert "Never deploy uncommitted files" in skill_text


def test_rollback_documented(skill_text: str):
    assert "Rollback" in skill_text
    assert re.search(r"git checkout deploy-", skill_text), "rollback must redeploy a previous tag"


def test_secrets_never_in_repo(skill_text: str):
    assert "NEVER commit secrets" in skill_text
    assert ".gitignore" in skill_text


def test_spa_404_pitfall_documented(skill_text: str):
    assert "404.html" in skill_text
    assert "_redirects" in skill_text


def test_verification_uses_real_http_check(skill_text: str):
    assert "%{http_code}" in skill_text
    assert "200" in skill_text
    assert "deploy log alone" in skill_text


def test_preview_before_deploy(skill_text: str):
    assert "http.server" in skill_text
    assert "cloudflared tunnel --url" in skill_text
    assert "trycloudflare.com" in skill_text


def test_category_description_exists():
    desc = REPO / "optional-skills" / "web-development" / "DESCRIPTION.md"
    assert desc.is_file(), "optional web-development category needs a DESCRIPTION.md"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
