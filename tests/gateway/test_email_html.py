"""Tests for email adapter HTML rendering (PR #73294)."""
import pytest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class TestMarkdownToHtmlEmail:
    """Test _markdown_to_html_email conversion."""

    def test_basic_markdown(self):
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email("**bold** and *italic*")
        assert "<strong>bold</strong>" in html
        assert "<em>italic</em>" in html

    def test_heading_styled(self):
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email("# Hello")
        assert '<h1 style=' in html
        assert "Hello" in html

    def test_fenced_code_no_duplicate_style(self):
        """<pre><code> blocks must NOT have duplicate style= attributes."""
        from plugins.platforms.email.adapter import _markdown_to_html_email
        md = "```python\nprint('hello')\n```"
        html = _markdown_to_html_email(md)
        # <pre> should have exactly one style=
        import re
        pre_tags = re.findall(r"<pre[^>]*>", html)
        for tag in pre_tags:
            assert tag.count("style=") == 1, f"Duplicate style= in: {tag}"
        # <code> should have exactly one style=
        code_tags = re.findall(r"<code[^>]*>", html)
        for tag in code_tags:
            assert tag.count("style=") == 1, f"Duplicate style= in: {tag}"

    def test_table_rendering(self):
        from plugins.platforms.email.adapter import _markdown_to_html_email
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = _markdown_to_html_email(md)
        assert "<table" in html
        assert "<td" in html

    def test_braces_not_escaped(self):
        """Body with { } braces must not break template substitution."""
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email("Use `{code}` here")
        # Verify the code span is rendered with inline code styling
        assert '<code style=' in html
        assert "code" in html
        # Verify no raw {body} placeholder remains
        assert "{body}" not in html


class TestAttachParts:
    """Test _attach_parts MIME structure."""

    def _make_adapter(self, html_format=True):
        """Create a minimal adapter mock for testing."""
        from unittest.mock import MagicMock
        adapter = MagicMock()
        adapter._html_format = html_format
        # Bind the real method
        from plugins.platforms.email.adapter import EmailAdapter
        adapter._attach_parts = EmailAdapter._attach_parts.__get__(adapter)
        return adapter

    def test_html_enabled_creates_two_parts(self):
        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart("alternative")
        adapter._attach_parts(msg, "**bold**")
        parts = msg.get_payload()
        assert len(parts) == 2
        assert parts[0].get_content_type() == "text/plain"
        assert parts[1].get_content_type() == "text/html"

    def test_html_disabled_creates_one_part(self):
        adapter = self._make_adapter(html_format=False)
        msg = MIMEMultipart("alternative")
        adapter._attach_parts(msg, "**bold**")
        parts = msg.get_payload()
        assert len(parts) == 1
        assert parts[0].get_content_type() == "text/plain"

    def test_importerror_falls_back_gracefully(self):
        """When markdown is not installed, should fall back to plain text."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "markdown":
                raise ImportError("No module named 'markdown'")
            return real_import(name, *args, **kwargs)

        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart("alternative")
        with pytest.MonkeyPatch.context() as m:
            m.setattr(builtins, "__import__", mock_import)
            # Should not raise — falls back to plain text only
            adapter._attach_parts(msg, "**bold**")
        parts = msg.get_payload()
        assert len(parts) == 1  # only plain text


class TestCreateBodyPart:
    """Test _create_body_part return type based on html_format."""

    def _make_adapter(self, html_format=True):
        from unittest.mock import MagicMock
        from plugins.platforms.email.adapter import EmailAdapter
        adapter = MagicMock()
        adapter._html_format = html_format
        adapter._create_body_part = EmailAdapter._create_body_part.__get__(adapter)
        adapter._attach_parts = EmailAdapter._attach_parts.__get__(adapter)
        return adapter

    def test_html_enabled_returns_alternative(self):
        adapter = self._make_adapter(html_format=True)
        part = adapter._create_body_part("**bold**")
        assert isinstance(part, MIMEMultipart)
        assert part.get_content_subtype() == "alternative"

    def test_html_disabled_returns_plain_text(self):
        adapter = self._make_adapter(html_format=False)
        part = adapter._create_body_part("**bold**")
        assert isinstance(part, MIMEText)
        assert part.get_content_type() == "text/plain"


class TestHtmlSanitization:
    """Test _sanitize_email_html security policy."""

    def test_script_tags_stripped(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<p>Hello</p><script>alert("xss")</script>')
        assert "script" not in result
        assert "alert" not in result
        assert "Hello" in result

    def test_event_handlers_stripped(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<p onclick="evil()">Click</p>')
        assert "onclick" not in result
        assert "Click" in result

    def test_javascript_urls_stripped(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<a href="javascript:alert(1)">link</a>')
        assert "javascript:" not in result

    def test_safe_urls_preserved(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<a href="https://example.com">link</a>')
        assert "https://example.com" in result

    def test_style_tags_stripped(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<p>Text</p><style>body{display:none}</style>')
        assert "style" not in result
        assert "display:none" not in result
        assert "Text" in result

    def test_inline_styles_preserved(self):
        """Inline styles are needed for email client compatibility."""
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<p style="margin:0;">Text</p>')
        assert 'style="margin:0;"' in result

    def test_disallowed_tags_stripped(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html('<p>OK</p><iframe src="evil"></iframe><embed>')
        assert "iframe" not in result
        assert "embed" not in result
        assert "OK" in result

    def test_allowed_tags_preserved(self):
        from plugins.platforms.email.adapter import _sanitize_email_html
        result = _sanitize_email_html("<p><strong>bold</strong> and <em>italic</em></p>")
        assert "<strong>" in result
        assert "<em>" in result


class TestHtmlDetection:
    """Test _is_html detection."""

    def test_html_detected(self):
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("<p>Hello</p>") is True
        assert _is_html("<div>content</div>") is True
        assert _is_html("<h1>Title</h1>") is True

    def test_markdown_not_detected_as_html(self):
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("**bold** and *italic*") is False
        assert _is_html("# Heading\n\nParagraph") is False
        assert _is_html("Just plain text.") is False
