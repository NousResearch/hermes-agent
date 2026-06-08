from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "social-media" / "xurl" / "SKILL.md"
DOC_MD = (
    REPO_ROOT
    / "website"
    / "docs"
    / "user-guide"
    / "skills"
    / "bundled"
    / "social-media"
    / "social-media-xurl.md"
)


def test_xurl_article_ingestion_uses_raw_api_mode():
    skill_text = SKILL_MD.read_text(encoding="utf-8")
    docs_text = DOC_MD.read_text(encoding="utf-8")

    for text in (skill_text, docs_text):
        assert "For X Articles, use raw API mode" in text
        assert "`xurl read`" in text
        assert "do not put `read` before a `/2/tweets/...`" in text
        assert "tweet.fields=created_at,lang,public_metrics" in text
        assert "referenced_tweets,article" in text
        assert "data.article.plain_text" in text
        assert "read '/2/tweets/" not in text
