"""Tests for HTML email rendering in the email platform adapter.

Covers:
- _style_html_email: inline CSS injection, <pre><code> override
- _attach_body: multipart/alternative wrapping
- _create_body_part: multipart/alternative for attachment paths
- html_format config: opt-in/opt-out behavior
"""

import pytest
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


# Import the module under test
import sys
import os

# Ensure the repo root is on sys.path so plugin imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plugins.platforms.email.adapter import (
    _style_html_email,
    _HTML_PREFIX,
    _HERMES_EMAIL_FOOTER,
    _HERMES_EMAIL_ELEMENT_STYLES,
)


class TestStyleHtmlEmail:
    """Tests for _style_html_email() inline CSS injection."""

    def test_code_inside_pre_gets_transparent_style(self):
        """<code> inside <pre> must get transparent background, not the default inline-code style."""
        html = '<pre><code>print("hello")</code></pre>'
        result = _style_html_email(html)
        # The <code> inside <pre> should have transparent background
        assert 'background:transparent' in result
        assert 'padding:0' in result
        # Should NOT have the default inline-code background
        assert 'background:#edf2f7' not in result.split('<pre')[1] if '<pre' in result else True

    def test_standalone_code_gets_inline_style(self):
        """<code> outside <pre> gets the inline-code background style."""
        html = '<p>Use <code>foo()</code> here</p>'
        result = _style_html_email(html)
        assert 'background:#edf2f7' in result
        assert 'border-radius:3px' in result

    def test_h1_gets_style(self):
        """Heading tags get styled."""
        html = '<h1>Title</h1>'
        result = _style_html_email(html)
        assert 'font-size:24px' in result
        assert 'border-bottom:2px solid #667eea' in result

    def test_pre_code_order_matters(self):
        """The <pre><code> override must run before the general <code> loop."""
        html = '<pre><code>x = 1</code></pre><p>inline <code>y</code></p>'
        result = _style_html_email(html)
        # Split at </pre> to isolate pre content from inline content
        pre_part = result.split('</pre>')[0]
        rest_part = result.split('</pre>')[1] if '</pre>' in result else ''
        # pre code should be transparent
        assert 'background:transparent' in pre_part
        # inline code should have the standard background
        if rest_part:
            assert 'background:#edf2f7' in rest_part

    def test_no_double_style_attribute(self):
        """Elements that already have style= are not double-styled."""
        html = '<p style="color:red;">already styled</p>'
        result = _style_html_email(html)
        # Should not add a second style= attribute
        assert result.count('style=') == 1

    def test_table_gets_style(self):
        """Table elements get styled."""
        html = '<table><tr><td>cell</td></tr></table>'
        result = _style_html_email(html)
        assert 'border-collapse:collapse' in result


class TestAttachBody:
    """Tests for EmailAdapter._attach_body() MIME structure."""

    def _make_adapter(self, html_format=True):
        """Create a minimal EmailAdapter mock with html_format config."""
        from plugins.platforms.email.adapter import EmailAdapter
        from gateway.config import PlatformConfig

        config = PlatformConfig.from_dict(
            {"extra": {"address": "test@test.com", "html_format": html_format}},
        )
        # Mock the __init__ to skip env var resolution
        with patch.object(EmailAdapter, '__init__', lambda self, *a, **kw: None):
            adapter = EmailAdapter.__new__(EmailAdapter)
            adapter._html_format = html_format
            adapter._address = "test@test.com"
            return adapter

    def test_attach_body_creates_multipart_alternative(self):
        """_attach_body should wrap body in multipart/alternative."""
        adapter = self._make_adapter(html_format=False)
        msg = MIMEMultipart()
        adapter._attach_body(msg, "Hello")

        # The root msg should have one child: multipart/alternative
        parts = msg.get_payload()
        assert len(parts) == 1
        assert parts[0].get_content_type() == "multipart/alternative"

    def _decode_payload(self, part):
        """Decode a MIME part's payload, handling base64 encoding."""
        decoded = part.get_payload(decode=True)
        if decoded:
            return decoded.decode("utf-8", errors="replace")
        return part.get_payload()

    def test_attach_body_with_html_disabled(self):
        """With html_format=False, alternative part has only text/plain."""
        adapter = self._make_adapter(html_format=False)
        msg = MIMEMultipart()
        adapter._attach_body(msg, "Hello **world**")

        alt = msg.get_payload()[0]
        payloads = alt.get_payload()
        assert len(payloads) == 1
        assert payloads[0].get_content_type() == "text/plain"
        assert self._decode_payload(payloads[0]) == "Hello **world**"

    def test_attach_body_with_html_enabled(self):
        """With html_format=True, alternative part has text/plain + text/html."""
        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart()
        adapter._attach_body(msg, "Hello **world**")

        alt = msg.get_payload()[0]
        payloads = alt.get_payload()
        assert len(payloads) == 2
        assert payloads[0].get_content_type() == "text/plain"
        assert payloads[1].get_content_type() == "text/html"
        html_content = self._decode_payload(payloads[1])
        assert "<strong" in html_content


class TestCreateBodyPart:
    """Tests for EmailAdapter._create_body_part() for attachment paths."""

    def _make_adapter(self, html_format=True):
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.object(EmailAdapter, '__init__', lambda self, *a, **kw: None):
            adapter = EmailAdapter.__new__(EmailAdapter)
            adapter._html_format = html_format
            return adapter

    def test_create_body_part_returns_multipart_alternative(self):
        """_create_body_part should return a multipart/alternative container."""
        adapter = self._make_adapter(html_format=False)
        part = adapter._create_body_part("Body text")
        assert part.get_content_type() == "multipart/alternative"

    def test_create_body_part_with_html(self):
        """With html_format=True, body part has plain + html."""
        adapter = self._make_adapter(html_format=True)
        part = adapter._create_body_part("Hello **bold**")
        payloads = part.get_payload()
        assert len(payloads) == 2
        assert payloads[0].get_content_type() == "text/plain"
        assert payloads[1].get_content_type() == "text/html"


class TestMimeStructure:
    """Integration tests for MIME structure across send paths."""

    def _make_adapter(self, html_format=True):
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.object(EmailAdapter, '__init__', lambda self, *a, **kw: None):
            adapter = EmailAdapter.__new__(EmailAdapter)
            adapter._html_format = html_format
            adapter._address = "test@test.com"
            return adapter

    def test_send_email_produces_multipart_mixed_with_alternative(self):
        """_send_email should produce multipart/mixed > multipart/alternative."""
        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart()
        adapter._attach_body(msg, "Hello **world**")

        assert msg.get_content_type() == "multipart/mixed"
        body_part = msg.get_payload()[0]
        assert body_part.get_content_type() == "multipart/alternative"
        alt_payloads = body_part.get_payload()
        assert len(alt_payloads) == 2
        assert alt_payloads[0].get_content_type() == "text/plain"
        assert alt_payloads[1].get_content_type() == "text/html"

    def test_attachment_path_produces_correct_structure(self):
        """_create_body_part inside multipart/mixed mimics real attachment flow."""
        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart()
        body_part = adapter._create_body_part("Body with **bold**")
        msg.attach(body_part)

        assert msg.get_content_type() == "multipart/mixed"
        alt = msg.get_payload()[0]
        assert alt.get_content_type() == "multipart/alternative"


class TestPreExistingHTML:
    """Tests for HTML that's already in the body (from cron/model output)."""

    def _make_adapter(self, html_format=True):
        from plugins.platforms.email.adapter import EmailAdapter
        with patch.object(EmailAdapter, '__init__', lambda self, *a, **kw: None):
            adapter = EmailAdapter.__new__(EmailAdapter)
            adapter._html_format = html_format
            return adapter

    def _decode_payload(self, part):
        decoded = part.get_payload(decode=True)
        if decoded:
            return decoded.decode("utf-8", errors="replace")
        return part.get_payload()

    def test_html_document_with_preamble(self):
        """HTML document with preamble text → preamble stripped, sent as HTML."""
        adapter = self._make_adapter()
        body = "Here is your daily briefing:\n<!DOCTYPE html><html><body><h1>Report</h1></body></html>"
        msg = MIMEMultipart()
        adapter._attach_body(msg, body)

        alt = msg.get_payload()[0]
        assert alt.get_content_type() == "multipart/alternative"
        payloads = alt.get_payload()
        assert len(payloads) == 2
        assert payloads[0].get_content_type() == "text/plain"
        assert payloads[1].get_content_type() == "text/html"
        html_content = self._decode_payload(payloads[1])
        assert html_content.startswith("<!DOCTYPE html>")
        assert "daily briefing" not in html_content

    def test_html_fragment_no_html_close(self):
        """HTML fragment (no </html>) → trailing commentary stripped."""
        adapter = self._make_adapter()
        body = "<div class='report'><h1>Status</h1><p>OK</p></div>\n\nThe report above shows the current status."
        msg = MIMEMultipart()
        adapter._attach_body(msg, body)

        alt = msg.get_payload()[0]
        payloads = alt.get_payload()
        html_content = self._decode_payload(payloads[1])
        assert html_content.endswith("</div>")
        assert "report above" not in html_content

    def test_duplicate_html_close_uses_first(self):
        """Duplicate </html> — first occurrence is used."""
        adapter = self._make_adapter()
        body = "<html><body><h1>Hello</h1></body></html>\ngarbage\n</html>"
        msg = MIMEMultipart()
        adapter._attach_body(msg, body)

        alt = msg.get_payload()[0]
        payloads = alt.get_payload()
        html_content = self._decode_payload(payloads[1])
        assert html_content.count("</html>") == 1

    def test_plain_text_not_detected_as_html(self):
        """Plain text body → sent as plain text, not HTML."""
        adapter = self._make_adapter(html_format=False)
        body = "Hello Felix,\n\nYour daily briefing is ready."
        msg = MIMEMultipart()
        adapter._attach_body(msg, body)

        alt = msg.get_payload()[0]
        payloads = alt.get_payload()
        assert len(payloads) == 1
        assert payloads[0].get_content_type() == "text/plain"
