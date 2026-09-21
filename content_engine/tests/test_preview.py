"""Tests for the blog visual preview generator (blog/preview.py).

Covers: frontmatter parsing, markdown→HTML conversion, full preview HTML
generation, integration with the approval tracker.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from blog.preview import (
    parse_frontmatter,
    md_to_html,
    build_preview_html,
    _image_to_data_uri,
    _read_mdx,
    _pending_slugs,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_MDX = """---
title: "Test Article"
description: "A test article for preview"
pubDate: 2026-06-29
heroImage: "/blog/test/hero.png"
tags: ["test", "preview", "ai"]
tier: ai
format: essay
approved: false
source: manual
---

# Test Article

This is a **bold** paragraph with *italic* and `inline code`.

## Section One

A paragraph with a [link](https://example.com).

> A blockquote with styled content.

- List item one
- List item two
- List item three

1. Ordered one
2. Ordered two

```python
def hello():
    print("world")
```

---

And a horizontal rule above with an image below.

![Section image alt](/blog/test/section-01.png)
"""

SAMPLE_FRONTMATTER_MDX = """---
title: "Agent memory is not a vector store"
pubDate: 2026-06-17
heroImage: "/blog/agent-memory-is-not-a-vector/hero.png"
tags: ["kensei", "build", "agent-memory"]
tier: builder
format: essay
approved: true
source: manual
---

Body text here.
"""


# ── Frontmatter parser ───────────────────────────────────────────────────

class TestParseFrontmatter:
    def test_parses_standard_frontmatter(self):
        fm, body = parse_frontmatter(SAMPLE_MDX)
        assert fm["title"] == "Test Article"
        assert fm["description"] == "A test article for preview"
        assert fm["tier"] == "ai"
        assert json.loads(fm["tags"]) == ["test", "preview", "ai"]
        assert "Body text here." not in body  # frontmatter only test
        
    def test_body_is_extracted(self):
        fm, body = parse_frontmatter(SAMPLE_MDX)
        assert "# Test Article" in body
        assert "This is a **bold** paragraph" in body

    def test_no_frontmatter_returns_empty(self):
        fm, body = parse_frontmatter("Just some text\nno frontmatter here")
        assert fm == {}
        assert body == "Just some text\nno frontmatter here"


# ── Markdown → HTML conversion ───────────────────────────────────────────

class TestMdToHtml:
    def test_h1_h2_h3(self):
        md = "# Title\n## Section\n### Subsection\n"
        html, heading = md_to_html(md)
        # h1 is extracted as heading, NOT rendered in body
        assert heading == "Title"
        assert "<h2" in html
        assert "Section" in html
        assert "<h3" in html
        assert "Subsection" in html


    def test_bold_italic_code(self):
        html, _ = md_to_html("**bold** *italic* `code`")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html
        assert "<code>code</code>" in html

    def test_link(self):
        html, _ = md_to_html("[click](https://example.com)")
        assert '<a href="https://example.com"' in html

    def test_image(self):
        html, _ = md_to_html("![alt](/path/img.png)")
        assert 'class="section-image"' in html
        assert 'src="/path/img.png"' in html

    def test_blockquote(self):
        html, _ = md_to_html("> A quote")
        assert "<blockquote" in html
        assert "A quote" in html

    def test_ul(self):
        html, _ = md_to_html("- item 1\n- item 2\n")
        assert "<ul>" in html
        assert "<li>item 1</li>" in html
        assert "<li>item 2</li>" in html

    def test_ol(self):
        html, _ = md_to_html("1. first\n2. second\n")
        assert "<ol>" in html
        assert "<li>first</li>" in html

    def test_code_block(self):
        md = "```python\nprint('hi')\n```\n"
        html, _ = md_to_html(md)
        assert "<pre>" in html
        assert "lang-python" in html
        assert "print('hi')" in html

    def test_hr(self):
        html, _ = md_to_html("---\n")
        assert "<hr>" in html


# ── Full preview HTML generation ─────────────────────────────────────────

class TestBuildPreviewHtml:
    def test_full_html_structure(self):
        html = build_preview_html(
            slug="test-post",
            title="Test Title",
            description="A description",
            stream="ai",
            tier="ai",
            tags=["test"],
            body_md="# Hello\n\nA paragraph.",
            pub_date="2026-06-29",
            hero_src="/blog/test/hero.png",
        )
        assert "<!DOCTYPE html>" in html
        assert "PREVIEW" in html
        assert "test-post" in html
        assert "Test Title" in html
        assert "A description" in html
        assert "ai" in html  # stream badge
        assert "◆ 2026-06-29" in html
        assert "PENDING REVIEW" in html
        assert "ARTICLE HERO IMAGE" in html
        
    def test_embeds_absolute_hero_image_as_data_uri(self, tmp_path):
        hero = tmp_path / "hero.png"
        hero.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 128)
        html = build_preview_html(
            slug="test-post",
            title="Test Title",
            stream="ai",
            tier="ai",
            tags=[],
            body_md="body",
            hero_src=str(hero),
        )
        assert "data:image/png;base64," in html
        assert "ARTICLE HERO IMAGE" in html
        assert "embedded · hero.png" in html

    def test_image_to_data_uri_reports_missing_image(self):
        src, meta = _image_to_data_uri("/tmp/definitely-missing-preview-image.png")
        assert src == "/tmp/definitely-missing-preview-image.png"
        assert meta == "image missing from preview bundle"
        
    def test_approved_status_badge(self):
        html = build_preview_html(
            slug="test",
            title="Test",
            stream="pm",
            tier="pm",
            tags=[],
            body_md="body",
            status="approved",
        )
        assert "APPROVED" in html

    def test_rejected_status_badge(self):
        html = build_preview_html(
            slug="test",
            title="Test",
            stream="builder",
            tier="builder",
            tags=[],
            body_md="body",
            status="rejected",
        )
        assert "REJECTED" in html

    def test_stream_class_mapped(self):
        html = build_preview_html(
            slug="test", title="Test", stream="ai", tier="ai",
            tags=[], body_md="body",
        )
        assert 'class="article-stream ai"' in html

    def test_builder_stream_class(self):
        html = build_preview_html(
            slug="test", title="Test", stream="builder", tier="builder",
            tags=[], body_md="body",
        )
        assert 'class="article-stream builder"' in html

    def test_escapes_html_in_body(self):
        html = build_preview_html(
            slug="test", title="Test", stream="ai", tier="ai",
            tags=[], body_md="<script>alert('xss')</script>",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ── Integration: reading from MDX ────────────────────────────────────────

class TestReadMdx:
    def test_no_file_returns_none(self):
        post = _read_mdx("this-post-definitely-does-not-exist-xyz")
        assert post is None

    def test_parses_existing_post(self):
        # Use a known real post
        post = _read_mdx("agent-memory-is-not-a-vector")
        if post:
            assert post["slug"] == "agent-memory-is-not-a-vector"
            assert post["title"]
            assert isinstance(post["tags"], list)
            assert isinstance(post["body_md"], str)
        # If not found (different environment), skip
        else:
            pytest.skip("Post not found on this system")


# ── Empty/invalid inputs ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_body(self):
        html, _ = md_to_html("")
        assert html == ""
    
    def test_only_frontmatter_and_heading(self):
        md = "---\ntitle: Test\n---\n\n# Heading only"
        _, heading = md_to_html(md)
        assert heading == "Heading only"

    def test_malformed_frontmatter(self):
        fm, body = parse_frontmatter("---\nnot yaml\njust\ntext\n---\nbody")
        assert "title" not in fm
        assert "body" in body

    def test_all_image_formats(self):
        html, _ = md_to_html("![png](/img.png)\n![jpg](/img.jpg)\n![webp](/img.webp)\n")
        assert html.count("section-image") == 3

    def test_nested_html_escaping(self):
        text = "AT&T < 5 & > 3"
        from blog.preview import _escape
        assert _escape(text) == "AT&amp;T &lt; 5 &amp; &gt; 3"
