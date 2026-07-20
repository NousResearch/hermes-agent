from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills" / "productivity" / "wechat-to-shopify-blog"


def test_hermes_metadata_contract():
    metadata = (SKILL / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[1]
    description = re.search(r'^description:\s*["\']?(.*?)["\']?\s*$', metadata, re.MULTILINE).group(1)
    assert len(description) <= 60 and description.endswith(".")
    for field in ("author:", "license:", "platforms:", "required_environment_variables:"):
        assert field in metadata
    assert "category:" in metadata and "related_skills:" in metadata


def test_wechat_fetches_validate_redirects_and_dns():
    helper = (SKILL / "scripts" / "lib" / "public-fetch.mjs").read_text(encoding="utf-8")
    fetcher = (SKILL / "scripts" / "fetch-wechat-article.mjs").read_text(encoding="utf-8")
    assert 'redirect: "follow"' not in helper + fetcher
    assert 'redirect: "manual"' in helper and "dns.lookup" in helper
    assert "mp.weixin.qq.com" in fetcher


def test_published_article_updates_are_blocked():
    script = (SKILL / "scripts" / "shopify-blog-admin.mjs").read_text(encoding="utf-8")
    assert "PUBLISHED_ARTICLE_UPDATE_BLOCKED" in script
    assert "ARTICLE_VERIFY" in script
    assert "isPublished: false" in script
