from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "shopify-markets-localization-auditor"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_storefront_fetches_use_manual_redirects_and_dns_checks():
    helper = (SKILL / "scripts" / "lib" / "public-fetch.mjs").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-markets-localization-auditor.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in helper + script
    assert 'redirect: "manual"' in helper and "dns.lookup" in helper


def test_mutations_are_review_first():
    script = (SKILL / "scripts" / "shopify-markets-localization-auditor.mjs").read_text(encoding="utf-8")
    assert "userErrors" in script
    assert "--execute" in (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert (SKILL / "templates" / "report-template.html").exists()
    assert '"templates", "report-template.html"' in script
