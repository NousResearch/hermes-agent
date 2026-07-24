from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "shopify-product-images-downloader"


def test_hermes_metadata_and_pinned_dependency():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata
    package = json.loads((SKILL / "package.json").read_text(encoding="utf-8"))
    assert package["dependencies"]["sharp"] == "0.35.3"


def test_downloader_has_no_runtime_install_or_implicit_redirects():
    script = (SKILL / "scripts" / "shopify-image-downloader.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in script
    assert "MAX_REDIRECTS" in script and "dns.lookup" in script
    assert "npm install sharp" not in script


def test_original_download_path_remains_available_without_sharp():
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-image-downloader.mjs").read_text(encoding="utf-8")
    assert "original-format downloads" in skill
    assert "WEBP_CONVERSION_UNAVAILABLE" in script
