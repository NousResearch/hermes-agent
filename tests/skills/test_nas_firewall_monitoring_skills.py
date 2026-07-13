"""Frontmatter contract tests for optional monitoring skills:
pfsense-monitoring, truenas-monitoring, unraid-monitoring.

Validates:
- Frontmatter parses as YAML
- name and description present
- description <= 60 chars, one sentence, ends with period
- required_environment_variables declared
- Modern section headers present
- No env var table in body (redundant with frontmatter declaration)
"""

import re

import pytest
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS = [
    {
        "path": "optional-skills/devops/pfsense-monitoring/SKILL.md",
        "name": "pfsense-monitoring",
        "env_vars": ["PFSENSE_API_KEY", "PFSENSE_URL"],
        "related": ["truenas-monitoring", "unraid-monitoring"],
    },
    {
        "path": "optional-skills/devops/truenas-monitoring/SKILL.md",
        "name": "truenas-monitoring",
        "env_vars": ["TRUENAS_API_KEY", "TRUENAS_URL"],
        "related": ["pfsense-monitoring", "unraid-monitoring"],
    },
    {
        "path": "optional-skills/devops/unraid-monitoring/SKILL.md",
        "name": "unraid-monitoring",
        "env_vars": ["UNRAID_API_KEY", "UNRAID_URL"],
        "related": ["pfsense-monitoring", "truenas-monitoring"],
    },
]

MODERN_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## Quick Reference",
    "## Pitfalls",
    "## Verification",
]


@pytest.fixture(params=SKILLS, ids=lambda s: s["name"])
def skill(request):
    return request.param


def _load(skill):
    """Read SKILL.md and return (content, frontmatter dict)."""
    path = REPO_ROOT / skill["path"]
    content = path.read_text()
    assert content.startswith("---"), "Frontmatter must start at byte 0"
    # Extract frontmatter between first --- and second ---
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "Frontmatter closing --- not found"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), "Frontmatter is not a YAML mapping"
    body = content[m.end():]
    # Strip any leftover dashes/newlines from YAML closing fence
    body = body.lstrip("-\n ")
    return content, fm, body


class TestFrontmatterContract:
    def test_name_present(self, skill):
        _, fm, _ = _load(skill)
        assert "name" in fm, "Frontmatter must have 'name'"
        assert fm["name"] == skill["name"]

    def test_description_present(self, skill):
        _, fm, _ = _load(skill)
        assert "description" in fm, "Frontmatter must have 'description'"

    def test_description_max_60_chars(self, skill):
        _, fm, _ = _load(skill)
        desc = fm["description"]
        assert len(desc) <= 60, (
            f"Description is {len(desc)} chars (max 60): '{desc}'"
        )

    def test_description_ending(self, skill):
        _, fm, _ = _load(skill)
        desc = fm["description"]
        assert desc.endswith("."), "Description must end with a period"

    def test_description_one_sentence(self, skill):
        """Description should be a single sentence — at most one terminal period."""
        _, fm, _ = _load(skill)
        desc = fm["description"]
        # Count periods not part of abbreviations (e.g., 'v2.0')
        periods = re.findall(r'(?<![a-zA-Z0-9])\.(?!\w)', desc.rstrip("."))
        assert len(periods) <= 1, (
            f"Description has {len(periods)} sentence-ending periods: '{desc}'"
        )

    def test_description_no_marketing_words(self, skill):
        _, fm, _ = _load(skill)
        desc = fm["description"].lower()
        marketing = ["powerful", "comprehensive", "seamless", "advanced"]
        for word in marketing:
            assert word not in desc, f"Marketing word '{word}' in description"

    def test_version_present(self, skill):
        _, fm, _ = _load(skill)
        assert "version" in fm, "Frontmatter should have 'version'"

    def test_author_present(self, skill):
        _, fm, _ = _load(skill)
        assert "author" in fm, "Frontmatter should have 'author'"
        # Author should be the human contributor, not 'Hermes Agent'
        assert "Hermes Agent" not in fm["author"], (
            "Author should credit the human contributor, not 'Hermes Agent'"
        )

    def test_license_present(self, skill):
        _, fm, _ = _load(skill)
        assert "license" in fm, "Frontmatter should have 'license'"


class TestRequiredEnvVars:
    def test_required_env_vars_declared(self, skill):
        _, fm, _ = _load(skill)
        req = fm.get("required_environment_variables", [])
        assert isinstance(req, list), "required_environment_variables must be a list"
        names = [e.get("name") for e in req]
        for expected in skill["env_vars"]:
            assert expected in names, (
                f"Missing required_environment_variables entry for {expected}"
            )

    def test_api_key_has_prompt(self, skill):
        _, fm, _ = _load(skill)
        req = fm.get("required_environment_variables", [])
        keys = [e for e in req if "API_KEY" in e.get("name", "")]
        for k in keys:
            assert k.get("prompt"), f"{k['name']} must have a 'prompt'"
            assert k.get("help"), f"{k['name']} must have a 'help'"

    def test_no_env_var_table_in_body(self, skill):
        """Env vars documented in a markdown table are redundant with
        required_environment_variables frontmatter — the table does not
        trigger the setup/passthrough flow."""
        _, _, body = _load(skill)
        assert "| Variable" not in body and "| `Variable" not in body, (
            "Remove env var markdown table — required_environment_variables "
            "in frontmatter is the source of truth"
        )

    def test_url_in_config_not_required(self, skill):
        """Non-secret appliance URLs belong in metadata.hermes.config,
        not in required_environment_variables (they have defaults)."""
        _, fm, _ = _load(skill)
        # The URL var should be in required_environment_variables (for
        # passthrough) or in metadata.hermes.config — either is fine.
        req_names = [e.get("name") for e in fm.get("required_environment_variables", [])]
        config = (fm.get("metadata") or {}).get("hermes") or {}
        config_keys = list((config.get("config") or {}).keys())
        url_var = [v for v in skill["env_vars"] if "URL" in v][0]
        assert url_var in req_names or url_var in config_keys, (
            f"{url_var} should be in required_environment_variables or "
            "metadata.hermes.config"
        )


class TestModernSections:
    def test_has_modern_sections(self, skill):
        _, _, body = _load(skill)
        for section in MODERN_SECTIONS:
            assert section in body, f"Missing required section: {section}"

    def test_has_title(self, skill):
        _, _, body = _load(skill)
        assert body.lstrip().startswith("# "), "Body must start with '# <Title>'"

    def test_no_env_vars_required_section(self, skill):
        """The old '## Env Vars Required' section is redundant when
        required_environment_variables is in frontmatter."""
        _, _, body = _load(skill)
        assert "## Env Vars Required" not in body, (
            "Remove '## Env Vars Required' — use required_environment_variables "
            "in frontmatter instead"
        )


class TestRelatedSkills:
    def test_related_skills_in_metadata(self, skill):
        _, fm, _ = _load(skill)
        meta = (fm.get("metadata") or {}).get("hermes") or {}
        related = meta.get("related_skills", [])
        assert isinstance(related, list)
        for sibling in skill["related"]:
            assert sibling in related, (
                f"Should list {sibling} in metadata.hermes.related_skills"
            )


class TestBodyLength:
    def test_body_not_empty(self, skill):
        _, _, body = _load(skill)
        assert len(body.strip()) > 50, "Body must have substantial content"

    def test_total_under_limit(self, skill):
        content, _, _ = _load(skill)
        assert len(content) <= 100_000, "SKILL.md must be under 100,000 chars"
