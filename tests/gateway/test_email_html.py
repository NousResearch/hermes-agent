"""Tests for email adapter HTML rendering (PR #73294)."""
import logging

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
        import re
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email("Use `{code}` here")
        # Exact rendered code span: the content must survive conversion inside a
        # styled <code> element. A substring check ("code" in html) is
        # tautological — "{code}" and the surrounding prose both contain it, so
        # it could never fail. This assertion can.
        assert re.search(r"<code style=[^>]*>\{code\}</code>", html), html
        # Verify no raw {body} placeholder remains
        assert "{body}" not in html

    def test_markdown_with_quoted_tags_still_renders(self):
        """A tag quoted in prose/code must not skip Markdown rendering.

        Regression: the body mentions `<pre><code>` (common in agent output),
        which used to flip detection to the HTML path and leave every Markdown
        construct unrendered.
        """
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email(
            "**Wichtig:** Nutze `<pre><code>` fuer Codebloecke.\n\n- eins\n- zwei\n"
        )
        assert "<strong>" in html
        assert "<ul style=" in html
        assert "**Wichtig:**" not in html

    def test_markdown_table_with_bare_br_still_renders(self):
        """A bare `<br>` must not skip Markdown rendering of the whole body."""
        from plugins.platforms.email.adapter import _markdown_to_html_email
        html = _markdown_to_html_email("| A | B |\n|---|---|\n| 1 | 2 |\n<br>\n")
        assert "<table" in html


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

    def test_importerror_falls_back_gracefully(self, caplog):
        """When markdown is not installed, should fall back to plain text."""
        import builtins
        import logging
        import plugins.platforms.email.adapter as adapter_mod
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "markdown":
                raise ImportError("No module named 'markdown'")
            return real_import(name, *args, **kwargs)

        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart("alternative")
        with pytest.MonkeyPatch.context() as m:
            m.setattr(builtins, "__import__", mock_import)
            with caplog.at_level(logging.WARNING, logger=adapter_mod.logger.name):
                # Should not raise — falls back to plain text only
                adapter._attach_parts(msg, "**bold**")
        parts = msg.get_payload()
        assert len(parts) == 1  # only plain text
        # A missing optional dependency is expected, not a failure: no warning.
        assert "HTML conversion failed" not in caplog.text

    def test_conversion_error_falls_back_to_plain_only(self, caplog):
        """A real conversion failure (markdown present, rendering broken) warns."""
        import logging
        import plugins.platforms.email.adapter as adapter_mod

        def boom(_body):
            raise RuntimeError("kaboom")

        adapter = self._make_adapter(html_format=True)
        msg = MIMEMultipart("alternative")
        with pytest.MonkeyPatch.context() as m:
            m.setattr(adapter_mod, "_markdown_to_html_email", boom)
            with caplog.at_level(logging.WARNING, logger=adapter_mod.logger.name):
                adapter._attach_parts(msg, "**bold**")
        parts = msg.get_payload()
        assert len(parts) == 1
        assert parts[0].get_content_type() == "text/plain"
        assert "HTML conversion failed" in caplog.text
        assert "kaboom" in caplog.text


class TestReplySkeleton:
    """The shipped reply skeleton must nest plain+HTML inside multipart/alternative.

    Regression covered here: the body used to be attached straight to the
    ``multipart/mixed`` root, so ``text/plain`` and ``text/html`` were siblings.
    Clients then either ignore the HTML part or show it as an attachment — and the
    plain-text part is what a cron/mailbox user sees. The helper that nests
    correctly existed but had no production caller, so helper-level tests could
    never fail on the shipped structure.
    """

    def _make_adapter(self, html_format=True):
        from plugins.platforms.email.adapter import EmailAdapter
        adapter = EmailAdapter.__new__(EmailAdapter)
        adapter._html_format = html_format
        adapter._address = "hermes@example.ch"
        adapter._thread_context = {}
        return adapter

    def test_html_enabled_nests_body_as_alternative(self):
        adapter = self._make_adapter(html_format=True)
        msg, _, _ = adapter._new_reply("felix@example.ch", "**bold**")
        assert msg.get_content_subtype() == "mixed"
        children = msg.get_payload()
        assert isinstance(children, list)
        assert len(children) == 1, "body belongs in one child of mixed, not as two siblings"
        body = children[0]
        assert isinstance(body, MIMEMultipart)
        assert body.get_content_subtype() == "alternative"
        inner = body.get_payload()
        assert isinstance(inner, list)
        assert [p.get_content_type() for p in inner] == ["text/plain", "text/html"]

    def test_html_disabled_keeps_single_plain_child(self):
        adapter = self._make_adapter(html_format=False)
        msg, _, _ = adapter._new_reply("felix@example.ch", "**bold**")
        children = msg.get_payload()
        assert isinstance(children, list)
        assert len(children) == 1
        body = children[0]
        assert isinstance(body, MIMEText)
        assert body.get_content_type() == "text/plain"


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

    def test_tag_mentioned_in_inline_code_is_markdown(self):
        """A tag quoted inside a code span must not flip the body to HTML."""
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("Nutze `<pre><code>` fuer Bloecke.") is False
        assert _is_html("Ein `<br>` am Zeilenende.") is False

    def test_tag_inside_fenced_block_is_markdown(self):
        """HTML shown as a fenced example is still Markdown, not an HTML body."""
        from plugins.platforms.email.adapter import _is_html
        body = "Beispiel:\n\n```html\n<div>\n  <p>x</p>\n</div>\n```\n"
        assert _is_html(body) is False

    def test_bare_br_does_not_flip_to_html(self):
        """`<br>` alone carries no block structure — keep rendering Markdown."""
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("| A | B |\n|---|---|\n| 1 | 2 |\n<br>\n") is False

    def test_unclosed_block_tag_is_not_html(self):
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("<div>no closing tag here") is False

    def test_paired_block_tag_is_html(self):
        """Guards against over-tightening: real fragments must still be HTML."""
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("<div><b>Hi</b></div>") is True
        assert _is_html("<p>Hello</p>") is True
        assert _is_html("Cronjob Response: X\n-------------\n<div><b>Hi</b></div>") is True

    def test_doctype_detected_even_with_code_span(self):
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("<!DOCTYPE html>\n<html><body>hi</body></html>") is True

    def test_empty_body_is_not_html(self):
        from plugins.platforms.email.adapter import _is_html
        assert _is_html("") is False


class TestStandaloneSendSMTPPort:
    """_standalone_send must select SMTP_SSL for port 465 (implicit TLS)."""

    def test_port_465_uses_smtp_ssl(self):
        import asyncio
        from types import SimpleNamespace
        from plugins.platforms.email.adapter import _standalone_send
        from unittest.mock import patch

        captured = {}

        class FakeSSL:
            def __init__(self, *a, **k):
                captured["cls"] = "SMTP_SSL"
                captured["port"] = k.get("port") or (a[1] if len(a) > 1 else None)

            def login(self, *a, **k):
                return None

            def send_message(self, msg):
                return None

            def quit(self):
                return None

        class FakeSMTP:
            def __init__(self, *a, **k):
                captured["cls"] = "SMTP"
                captured["port"] = k.get("port") or (a[1] if len(a) > 1 else None)

            def starttls(self, context=None):
                return None

            def login(self, *a, **k):
                return None

            def send_message(self, msg):
                return None

            def quit(self):
                return None

        import os
        os.environ["EMAIL_ADDRESS"] = "a@b.ch"
        os.environ["EMAIL_PASSWORD"] = "x"
        os.environ["EMAIL_SMTP_HOST"] = "smtp.test.com"
        os.environ["EMAIL_SMTP_PORT"] = "465"

        pconfig = SimpleNamespace(token=None, api_key=None, extra={"address": "a@b.ch", "smtp_host": "smtp.test.com"})

        with patch("smtplib.SMTP_SSL", FakeSSL), patch("smtplib.SMTP", FakeSMTP):
            result = asyncio.run(_standalone_send(pconfig, "c@d.ch", "hello"))

        assert captured.get("cls") == "SMTP_SSL", f"port 465 must use SMTP_SSL, got {captured.get('cls')}"
        assert result.get("success") is True

    def test_port_587_uses_starttls(self):
        import asyncio
        from types import SimpleNamespace
        from plugins.platforms.email.adapter import _standalone_send
        from unittest.mock import patch

        captured = {}

        class FakeSMTP:
            def __init__(self, *a, **k):
                captured["cls"] = "SMTP"
                captured["starttls"] = False

            def starttls(self, context=None):
                captured["starttls"] = True
                return None

            def login(self, *a, **k):
                return None

            def send_message(self, msg):
                return None

            def quit(self):
                return None

        import os
        os.environ["EMAIL_ADDRESS"] = "a@b.ch"
        os.environ["EMAIL_PASSWORD"] = "x"
        os.environ["EMAIL_SMTP_HOST"] = "smtp.test.com"
        os.environ["EMAIL_SMTP_PORT"] = "587"

        pconfig = SimpleNamespace(token=None, api_key=None, extra={"address": "a@b.ch", "smtp_host": "smtp.test.com"})

        with patch("smtplib.SMTP", FakeSMTP):
            result = asyncio.run(_standalone_send(pconfig, "c@d.ch", "hello"))

        assert captured.get("cls") == "SMTP"
        assert captured.get("starttls") is True
        assert result.get("success") is True


class TestTrimHtmlPreamblePostamble:
    """HTML-body trimming ported from PR #36853 (chtse53)."""

    def test_strips_cron_wrapper_preamble(self):
        from plugins.platforms.email.adapter import _trim_html_preamble_postamble
        body = "Cronjob Response: Test\n-------------\n<div><b>Hi</b></div>"
        result = _trim_html_preamble_postamble(body)
        assert result.startswith("<div>")
        assert "Cronjob Response" not in result

    def test_strips_trailing_commentary_after_html_doc(self):
        from plugins.platforms.email.adapter import _trim_html_preamble_postamble
        body = "<html><body><p>x</p></body></html>\n\nModel commentary after"
        result = _trim_html_preamble_postamble(body)
        assert result.endswith("</html>")
        assert "commentary" not in result

    def test_strips_trailing_prose_after_fragment(self):
        from plugins.platforms.email.adapter import _trim_html_preamble_postamble
        body = "<div><p>Body</p></div>\n\nDas war mein Bericht."
        result = _trim_html_preamble_postamble(body)
        assert result == "<div><p>Body</p></div>"

    def test_plain_text_unchanged(self):
        from plugins.platforms.email.adapter import _trim_html_preamble_postamble
        body = "Nur Text, kein HTML hier"
        result = _trim_html_preamble_postamble(body)
        assert result == body

    def test_html_detection_path_applies_trim(self):
        from plugins.platforms.email.adapter import _markdown_to_html_email
        result = _markdown_to_html_email(
            "Cronjob Response: X\n-------------\n<div><b>Hi</b></div>"
        )
        assert "<b>Hi</b>" in result
        assert "Cronjob Response" not in result


class TestSanitizerFallback:
    """The sanitizer fallback must not swallow errors silently."""

    def test_failure_logs_warning_and_escapes(self, caplog):
        import plugins.platforms.email.adapter as adapter_mod

        class Boom(adapter_mod._EmailHtmlSanitizer):
            def feed(self, data):
                raise RuntimeError("boom")

        original = adapter_mod._EmailHtmlSanitizer
        adapter_mod._EmailHtmlSanitizer = Boom
        try:
            with caplog.at_level(logging.WARNING, logger=adapter_mod.logger.name):
                result = adapter_mod._sanitize_email_html("<p>x</p>")
        finally:
            adapter_mod._EmailHtmlSanitizer = original

        assert result == "&lt;p&gt;x&lt;/p&gt;"
        assert "sanitiz" in caplog.text.lower()


class TestStandaloneSendFallback:
    """A missing optional dependency must not warn on the standalone path either.

    ``_attach_parts`` (class path) separates ``ImportError`` — an expected,
    documented opt-out — from a real conversion failure. The standalone sender
    shipped with one broad ``except``, so a host without the optional ``markdown``
    package warned on every cron send.
    """

    def _pconfig(self, html_format=True):
        from types import SimpleNamespace
        return SimpleNamespace(
            token=None,
            api_key=None,
            extra={"address": "a@b.ch", "smtp_host": "smtp.test.com", "html_format": html_format},
        )

    def _send(self, monkeypatch, caplog, *, markdown_missing):
        import asyncio
        import builtins
        import plugins.platforms.email.adapter as adapter_mod

        sent = {}

        class FakeServer:
            def login(self, *a, **k):
                return None

            def send_message(self, msg):
                sent["msg"] = msg

            def quit(self):
                return None

        monkeypatch.setattr(adapter_mod, "_open_smtp", lambda *a, **k: FakeServer())
        monkeypatch.setattr(
            adapter_mod, "_get_secret", lambda key, default="": "x" if key == "EMAIL_PASSWORD" else default
        )

        if markdown_missing:
            real_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "markdown":
                    raise ImportError("No module named 'markdown'")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", mock_import)

        with caplog.at_level(logging.WARNING, logger=adapter_mod.logger.name):
            result = asyncio.run(adapter_mod._standalone_send(self._pconfig(), "c@d.ch", "**bold**"))
        return result, sent, caplog.text

    def test_missing_markdown_sends_plain_without_warning(self, monkeypatch, caplog):
        result, sent, log_text = self._send(monkeypatch, caplog, markdown_missing=True)
        assert result.get("success") is True
        assert "HTML conversion failed" not in log_text
        parts = sent["msg"].get_payload()
        assert isinstance(parts, list)
        assert [p.get_content_type() for p in parts] == ["text/plain"]

    def test_real_conversion_failure_warns(self, monkeypatch, caplog):
        import plugins.platforms.email.adapter as adapter_mod

        def boom(_body):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(adapter_mod, "_markdown_to_html_email", boom)
        result, sent, log_text = self._send(monkeypatch, caplog, markdown_missing=False)
        assert result.get("success") is True
        assert "HTML conversion failed" in log_text
        assert "kaboom" in log_text


class TestPlainAlternative:
    """Both alternatives of one email must carry the same content.

    Regression: an already-HTML body produced a trimmed, sanitized HTML part but a raw
    plain part, so a text-only client rendered the cron wrapper prefix plus naked
    ``<p>``/``<b>`` tags — visible in exactly the cron-report shape this feature targets.
    """

    CRON_BODY = "Cronjob Response: nightly\n-------------\n<div><b>Result</b><p>Line one</p></div>"

    def _adapter(self, html_format=True):
        from plugins.platforms.email.adapter import EmailAdapter
        adapter = EmailAdapter.__new__(EmailAdapter)
        adapter._html_format = html_format
        adapter._address = "hermes@example.ch"
        adapter._thread_context = {}
        return adapter

    def _parts(self, body, html_format=True):
        msg, _, _ = self._adapter(html_format)._new_reply("felix@example.ch", body)
        children = msg.get_payload()
        assert isinstance(children, list)
        body_part = children[0]
        assert isinstance(body_part, MIMEMultipart)
        inner = body_part.get_payload()
        assert isinstance(inner, list)
        decoded = []
        for part in inner:
            assert isinstance(part, MIMEText)
            payload = part.get_payload(decode=True)
            decoded.append((part.get_content_type(), payload.decode("utf-8") if isinstance(payload, bytes) else ""))
        return dict(decoded)

    def test_already_html_body_parts_agree(self):
        parts = self._parts(self.CRON_BODY)
        plain, html = parts["text/plain"], parts["text/html"]
        # The plain alternative is readable text: no tags and no cron wrapper prefix,
        # i.e. the same content the HTML alternative shows.
        assert "<" not in plain and ">" not in plain
        assert "Cronjob Response" not in plain
        assert "Result" in plain and "Line one" in plain
        # The HTML alternative still carries the markup and drops the wrapper.
        assert "<b>Result</b>" in html
        assert "Cronjob Response" not in html

    def test_markdown_body_plain_part_unchanged(self):
        body = "**bold** and `code`, mentioning `<pre><code>` in prose"
        parts = self._parts(body)
        # Markdown source IS the readable plain-text form — it must not be rewritten.
        assert parts["text/plain"] == body
