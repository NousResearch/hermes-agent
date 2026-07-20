from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "productivity" / "shopify-blog-seo-optimizer"


def frontmatter_text():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    return text.split("---", 2)[1]


def test_hermes_metadata_contract():
    metadata = frontmatter_text()
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60
    assert description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata
    assert "related_skills:" in metadata


def test_public_fetch_does_not_follow_redirects_implicitly():
    helper = (SKILL / "scripts" / "lib" / "public-fetch.mjs").read_text(encoding="utf-8")
    script = (SKILL / "scripts" / "shopify-blog-seo-admin.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in helper + script
    assert "manual" in helper
    assert "dns.lookup" in helper


def test_article_html_sanitizer_covers_quoted_and_unquoted_handlers():
    script = (SKILL / "scripts" / "shopify-blog-seo-admin.mjs").read_text(encoding="utf-8")
    assert "on[a-z0-9_-]+" in script
    assert "srcdoc" in script
    assert "javascript:" in script
