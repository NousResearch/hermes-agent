from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "productivity" / "shopify-gmc-misrepresentation-auditor"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_crawlers_use_safe_manual_redirects():
    scripts = "".join(
        (SKILL / "scripts" / name).read_text(encoding="utf-8")
        for name in ("public-fetch.mjs", "gmc-store-audit.mjs", "gmc-product-audit.mjs")
    )
    assert 'redirect: "follow"' not in scripts
    assert 'redirect: "manual"' in scripts
    assert "dns.lookup" in scripts


def test_robots_block_is_terminal():
    script = (SKILL / "scripts" / "gmc-store-audit.mjs").read_text(encoding="utf-8")
    assert "gmc-auditor" in script
    assert "return result" in script
