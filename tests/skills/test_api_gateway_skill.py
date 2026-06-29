"""Tests for optional-skills/productivity/api-gateway skill documentation."""

import re
from pathlib import Path

import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "productivity" / "api-gateway"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES_DIR = SKILL_DIR / "references"


@pytest.fixture(scope="module")
def skill_text():
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(skill_text):
    match = re.match(r"^---\n(.+?)\n---", skill_text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    return yaml.safe_load(match.group(1))


# ── Frontmatter Validation ──────────────────────────────────────────────────


class TestFrontmatter:
    def test_required_fields_present(self, frontmatter):
        required = ["name", "description", "version", "author", "license", "platforms"]
        for field in required:
            assert field in frontmatter, f"Missing required frontmatter field: {field}"

    def test_name_matches_directory(self, frontmatter):
        assert frontmatter["name"] == "api-gateway"

    def test_description_under_60_chars(self, frontmatter):
        desc = frontmatter["description"]
        assert len(desc) <= 60, f"Description is {len(desc)} chars (max 60): {desc}"

    def test_description_ends_with_period(self, frontmatter):
        assert frontmatter["description"].endswith(".")

    def test_version_is_semver(self, frontmatter):
        assert re.match(r"^\d+\.\d+\.\d+$", frontmatter["version"])

    def test_platforms_includes_all_three(self, frontmatter):
        platforms = set(frontmatter["platforms"])
        assert {"linux", "macos", "windows"}.issubset(platforms)

    def test_required_environment_variables(self, frontmatter):
        env_vars = frontmatter["required_environment_variables"]
        assert len(env_vars) >= 1
        names = [v["name"] for v in env_vars]
        assert "MATON_API_KEY" in names

    def test_hermes_metadata_tags(self, frontmatter):
        tags = frontmatter["metadata"]["hermes"]["tags"]
        assert len(tags) >= 1
        assert all(isinstance(t, str) for t in tags)


# ── Required Sections ───────────────────────────────────────────────────────


class TestRequiredSections:
    REQUIRED_SECTIONS = [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Pitfalls",
        "Verification",
    ]

    @pytest.mark.parametrize("section", REQUIRED_SECTIONS)
    def test_section_exists(self, skill_text, section):
        pattern = rf"^##\s+{re.escape(section)}"
        assert re.search(pattern, skill_text, re.MULTILINE), (
            f"Missing required section: ## {section}"
        )


# ── Reference Directory Integrity ───────────────────────────────────────────


class TestReferenceIntegrity:
    TRIGGER_ONLY_SOURCES = {"time"}

    def test_references_directory_exists(self):
        assert REFERENCES_DIR.is_dir()

    def test_each_reference_has_readme(self):
        for provider_dir in sorted(REFERENCES_DIR.iterdir()):
            if provider_dir.is_dir() and provider_dir.name not in self.TRIGGER_ONLY_SOURCES:
                readme = provider_dir / "README.md"
                assert readme.is_file(), (
                    f"Missing README.md in references/{provider_dir.name}/"
                )

    def test_skill_md_links_match_existing_directories(self, skill_text):
        linked = set(re.findall(r"references/([a-z0-9-]+)/README\.md", skill_text))
        actual = {d.name for d in REFERENCES_DIR.iterdir() if d.is_dir()}
        missing_dirs = linked - actual
        assert not missing_dirs, (
            f"SKILL.md links to non-existent reference dirs: {missing_dirs}"
        )

    def test_all_directories_linked_in_skill_md(self, skill_text):
        linked = set(re.findall(r"references/([a-z0-9-]+)/README\.md", skill_text))
        actual = {
            d.name for d in REFERENCES_DIR.iterdir()
            if d.is_dir() and d.name not in self.TRIGGER_ONLY_SOURCES
        }
        unlinked = actual - linked
        assert not unlinked, (
            f"Reference dirs exist but are not linked in SKILL.md: {unlinked}"
        )

    def test_trigger_files_exist_for_linked_triggers(self, skill_text):
        trigger_links = set(
            re.findall(r"references/([a-z0-9-]+)/triggers\.md", skill_text)
        )
        for provider in trigger_links:
            trigger_file = REFERENCES_DIR / provider / "triggers.md"
            assert trigger_file.is_file(), (
                f"SKILL.md links to references/{provider}/triggers.md but file is missing"
            )


# ── Content Validation ──────────────────────────────────────────────────────


class TestContentValidation:
    def test_api_base_url_documented(self, skill_text):
        assert "api.maton.ai" in skill_text

    def test_authentication_header_documented(self, skill_text):
        assert "MATON_API_KEY" in skill_text
        assert "X-API-Key" in skill_text or "Authorization" in skill_text

    def test_cli_package_name_is_current(self, skill_text):
        assert "@maton/cli" in skill_text
        assert "@maton-ai/cli" not in skill_text

    def test_no_broken_markdown_links(self, skill_text):
        local_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", skill_text)
        for label, target in local_links:
            if target.startswith(("http", "mailto:")):
                continue
            target_path = target.split("#")[0]
            if not target_path:
                continue
            resolved = SKILL_DIR / target_path
            assert resolved.exists(), (
                f"Broken local link [{label}]({target}) — "
                f"resolved to {resolved} which does not exist"
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
