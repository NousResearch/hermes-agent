from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "shopify-store-translator"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_authentication_contract_is_explicit_and_scope_gated():
    script = (SKILL / "scripts" / "lib" / "shopify-cli.mjs").read_text(encoding="utf-8")
    docs = (SKILL / "references" / "translation-api.md").read_text(encoding="utf-8")
    assert "assertRequiredScopes" in script
    assert "READ_SCOPES" in script and "REQUIRED_SCOPES" in script
    assert "Shopify CLI" in docs and "Dev Dashboard" in docs


def test_writes_remain_review_first():
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert "approval" in skill.lower()
    assert "write" in skill.lower()
