from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "shopify-product-serp-optimizer"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_public_mode_and_admin_writes_are_hardened():
    helper = (SKILL / "scripts" / "lib" / "public-fetch.mjs").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-product-serp-admin.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in helper + script
    assert 'redirect: "manual"' in helper and "dns.lookup" in helper
    assert "userErrors" in script


def test_custom_domain_is_described_as_public_storefront_mode():
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-product-serp-admin.mjs").read_text(encoding="utf-8")
    assert "public_storefront" in skill and "public_storefront" in script
    assert "media?.nodes" in script or "Array.isArray(product?.media)" in script
