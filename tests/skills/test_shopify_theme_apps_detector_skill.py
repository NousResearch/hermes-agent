from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "shopify-theme-apps-detector"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_report_template_and_fetch_guard_are_used():
    script = (SKILL / "scripts" / "store-scanner.mjs").read_text(encoding="utf-8")
    helper = (SKILL / "scripts" / "public-fetch.mjs").read_text(encoding="utf-8")
    assert (SKILL / "templates" / "report-template.html").exists()
    assert "templates/report-template.html" in script
    assert 'redirect: "follow"' not in script + helper
    assert 'redirect: "manual"' in helper and "dns.lookup" in helper


def test_report_data_is_script_safe():
    script = (SKILL / "scripts" / "store-scanner.mjs").read_text(encoding="utf-8")
    assert "\\u003c" in script and "application/json" in script
