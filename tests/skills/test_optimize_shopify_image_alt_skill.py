from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "optimize-shopify-image-alt"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_public_fetch_is_manual_and_resolves_dns():
    helper = (SKILL / "scripts" / "lib" / "public-fetch.mjs").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-alt-text-admin.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in helper + script
    assert 'redirect: "manual"' in helper and "dns.lookup" in helper


def test_preview_first_and_user_errors_are_explicit():
    script = (SKILL / "scripts" / "shopify-alt-text-admin.mjs").read_text(encoding="utf-8")
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "execute" in script or "--execute" in skill
    assert "userErrors" in script
