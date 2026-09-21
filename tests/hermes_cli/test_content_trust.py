"""
Tests for content_trust module (P2-4).

Verifies fence markers, tier tagging, and the prompt addendum.
"""

import pytest
from hermes_cli.content_trust import (
    TRUSTED,
    INTERNAL,
    EXTERNAL_UNTRUSTED,
    UNTRUSTED_BEGIN,
    UNTRUSTED_END,
    fence,
    fence_web_search,
    fence_web_extract,
    fence_github_readme,
    fence_social_post,
    untrusted_prompt_addendum,
)


class TestFence:
    def test_basic_fence_wraps_content(self):
        result = fence("test source", "hello world")
        assert UNTRUSTED_BEGIN in result
        assert UNTRUSTED_END in result
        assert "hello world" in result
        assert 'source="test source"' in result

    def test_fence_default_tier_is_external_untrusted(self):
        result = fence("test", "data")
        assert 'tier="external-untrusted"' in result

    def test_fence_explicit_trusted_tier(self):
        result = fence("docs", "trusted content", tier=TRUSTED)
        assert 'tier="trusted"' in result

    def test_fence_explicit_internal_tier(self):
        result = fence("kanban", "task body", tier=INTERNAL)
        assert 'tier="internal"' in result

    def test_fence_preserves_multiline_content(self):
        content = "line1\nline2\nline3"
        result = fence("src", content)
        lines = result.split("\n")
        assert lines[1] == "line1"
        assert lines[2] == "line2"
        assert lines[3] == "line3"

    def test_fence_content_between_markers(self):
        result = fence("src", "DATA")
        begin_idx = result.index(UNTRUSTED_BEGIN)
        end_idx = result.index(UNTRUSTED_END)
        assert begin_idx < end_idx
        assert result[begin_idx + len(UNTRUSTED_BEGIN)] != ">"


class TestFenceWebSearch:
    def test_formats_results(self):
        results = [
            {"title": "Page 1", "url": "https://a.com", "description": "First"},
            {"title": "Page 2", "url": "https://b.com", "description": "Second"},
        ]
        result = fence_web_search("test query", results)
        assert UNTRUSTED_BEGIN in result
        assert UNTRUSTED_END in result
        assert "Page 1" in result
        assert "Page 2" in result
        assert "https://a.com" in result
        assert "test query" in result

    def test_handles_missing_fields(self):
        results = [{"title": "Only Title"}]
        result = fence_web_search("q", results)
        assert "Only Title" in result
        assert "N/A" in result  # URL defaults to N/A

    def test_handles_empty_results(self):
        result = fence_web_search("q", [])
        assert UNTRUSTED_BEGIN in result


class TestFenceWebExtract:
    def test_wraps_url_and_content(self):
        result = fence_web_extract("https://example.com", "page content")
        assert "https://example.com" in result
        assert "page content" in result
        assert UNTRUSTED_BEGIN in result

    def test_truncates_long_url_in_source(self):
        long_url = "https://example.com/" + "a" * 200
        result = fence_web_extract(long_url, "content")
        # source attribute should be truncated
        assert len(long_url) > 120
        assert long_url[:120] in result


class TestFenceGitHub:
    def test_includes_repo_in_source(self):
        result = fence_github_readme("owner/repo", "# README")
        assert "owner/repo" in result
        assert "# README" in result
        assert UNTRUSTED_BEGIN in result


class TestFenceSocial:
    def test_includes_platform_and_author(self):
        result = fence_social_post("twitter", "someuser", "tweet text")
        assert "twitter" in result
        assert "@someuser" in result
        assert "tweet text" in result


class TestPromptAddendum:
    def test_includes_fence_markers(self):
        addendum = untrusted_prompt_addendum()
        assert UNTRUSTED_BEGIN in addendum
        assert UNTRUSTED_END in addendum

    def test_includes_trust_tier_descriptions(self):
        addendum = untrusted_prompt_addendum()
        assert "trusted" in addendum.lower()
        assert "internal" in addendum.lower()
        assert "external-untrusted" in addendum.lower()

    def test_is_reasonably_sized(self):
        addendum = untrusted_prompt_addendum()
        # Should be under 2KB — compact enough for system prompts
        assert len(addendum) < 2048
