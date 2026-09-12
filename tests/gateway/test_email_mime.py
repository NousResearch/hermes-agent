"""Behavioral contract tests for the Email platform's outgoing MIME messages."""

from __future__ import annotations

import asyncio
from email import policy
from email import utils as email_utils
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.email import adapter as email_adapter
from plugins.platforms.email.mime import (
    MimeAttachment,
    MimeSignature,
    build_email_message,
    render_markdown_html,
    sanitize_message_html,
    sanitize_signature_html,
)


_DATE = "Sat, 23 Aug 2026 10:00:00 +0200"
_MESSAGE_ID = "<hermes-0123456789ab@test.com>"


def _make_adapter(monkeypatch, extra=None) -> email_adapter.EmailAdapter:
    monkeypatch.setenv("EMAIL_ADDRESS", "hermes@test.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_IMAP_HOST", "imap.test.com")
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.test.com")
    return email_adapter.EmailAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def _capture_adapter_message(monkeypatch, adapter):
    smtp = MagicMock()
    monkeypatch.setattr(adapter, "_connect_smtp", lambda: smtp)
    monkeypatch.setattr(email_adapter, "formatdate", lambda *, localtime: _DATE)
    monkeypatch.setattr(
        email_adapter.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef"),
    )
    return smtp


def _assert_utf8_plain_part(part, expected_body: str) -> None:
    assert part.get_content_type() == "text/plain"
    assert part.get_content_charset() == "utf-8"
    assert part["Content-Transfer-Encoding"] == "base64"
    assert part.get_payload(decode=True).decode("utf-8") == expected_body


def _build_message_with_attachment(filename: str):
    return build_email_message(
        from_address="hermes@test.com",
        to_address="user@test.com",
        subject="Attachment",
        body="Attached.",
        date=_DATE,
        attachments=(MimeAttachment(filename=filename, content=b"payload"),),
    )


def test_plain_reply_preserves_legacy_multipart_envelope_and_threading(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    smtp = _capture_adapter_message(monkeypatch, adapter)
    adapter._thread_context["user@test.com"] = {
        "subject": "Résumé du projet",
        "message_id": "<context@test.com>",
    }

    returned_id = adapter._send_email(
        "user@test.com",
        "Réponse en texte brut.",
        "<explicit@test.com>",
    )

    message = smtp.send_message.call_args.args[0]
    assert returned_id == _MESSAGE_ID
    assert message.get_content_type() == "multipart/mixed"
    assert list(message.keys()) == [
        "Content-Type",
        "MIME-Version",
        "From",
        "To",
        "Subject",
        "In-Reply-To",
        "References",
        "Date",
        "Message-ID",
    ]
    assert message["From"] == "hermes@test.com"
    assert message["To"] == "user@test.com"
    assert message["Subject"] == "Re: Résumé du projet"
    assert message["In-Reply-To"] == "<explicit@test.com>"
    assert message["References"] == "<explicit@test.com>"
    assert message["Date"] == _DATE
    assert message["Message-ID"] == _MESSAGE_ID
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "text/plain",
    ]
    _assert_utf8_plain_part(message.get_payload()[0], "Réponse en texte brut.")
    smtp.login.assert_called_once_with("hermes@test.com", "secret")
    smtp.quit.assert_called_once_with()


def test_single_attachment_preserves_legacy_shape_headers_and_payload(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _make_adapter(monkeypatch)
    smtp = _capture_adapter_message(monkeypatch, adapter)
    adapter._thread_context["user@test.com"] = {
        "subject": "Re: Existing subject",
        "message_id": "<original@test.com>",
    }
    source = tmp_path / "source.bin"
    source.write_bytes(b"\x00attachment\xff")

    returned_id = adapter._send_email_with_attachment(
        "user@test.com",
        "Pièce jointe.",
        str(source),
        "rapport final.bin",
    )

    message = smtp.send_message.call_args.args[0]
    assert returned_id == _MESSAGE_ID
    assert message.get_content_type() == "multipart/mixed"
    assert list(message.keys()) == [
        "Content-Type",
        "MIME-Version",
        "From",
        "To",
        "Subject",
        "In-Reply-To",
        "References",
        "Date",
        "Message-ID",
    ]
    assert message["Subject"] == "Re: Existing subject"
    assert message["In-Reply-To"] == "<original@test.com>"
    assert message["References"] == "<original@test.com>"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "text/plain",
        "application/octet-stream",
    ]
    plain_part, attachment = message.get_payload()
    _assert_utf8_plain_part(plain_part, "Pièce jointe.")
    assert attachment["Content-Transfer-Encoding"] == "base64"
    assert attachment["Content-Disposition"] == (
        'attachment; filename="rapport final.bin"'
    )
    assert attachment.get_filename() == "rapport final.bin"
    assert attachment.get_payload(decode=True) == b"\x00attachment\xff"


@pytest.mark.parametrize(
    "filename",
    [
        "report.txt",
        "report final.txt",
        'report "final".txt',
        "report;final.txt",
        "résumé-été.txt",
    ],
)
def test_attachment_filename_round_trips_through_serialization(filename):
    message = _build_message_with_attachment(filename)

    serialized = message.as_bytes(policy=policy.SMTP)
    parsed = BytesParser(policy=policy.default).parsebytes(serialized)
    attachment = next(parsed.iter_attachments())

    assert attachment.get_filename() == filename
    assert parsed["X-Injected"] is None


def test_attachment_filename_uses_quoted_and_rfc2231_parameters():
    spaced = _build_message_with_attachment("report final.txt").as_bytes(
        policy=policy.SMTP
    )
    punctuation = _build_message_with_attachment(
        'report; "final".txt'
    ).as_bytes(policy=policy.SMTP)
    unicode = _build_message_with_attachment("résumé-été.txt").as_bytes(
        policy=policy.SMTP
    )

    assert b'filename="report final.txt"' in spaced
    assert b'filename="report; \\"final\\".txt"' in punctuation
    assert b"filename*=utf-8''r%C3%A9sum%C3%A9-%C3%A9t%C3%A9.txt" in unicode


@pytest.mark.parametrize(
    "filename",
    [
        "report\rinjected.txt",
        "report\ninjected.txt",
        "report.txt\r\nX-Injected: true",
    ],
    ids=["cr", "lf", "crlf-header"],
)
def test_attachment_filename_rejects_header_newlines(filename):
    with pytest.raises(
        ValueError,
        match="attachment filename must not contain NUL, CR, or LF characters",
    ):
        _build_message_with_attachment(filename)


def test_attachment_filename_rejects_nul():
    with pytest.raises(
        ValueError,
        match="attachment filename must not contain NUL, CR, or LF characters",
    ):
        _build_message_with_attachment("report\x00final.txt")


def test_multiple_attachments_preserve_order_and_empty_body_semantics(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _make_adapter(monkeypatch)
    smtp = _capture_adapter_message(monkeypatch, adapter)
    first = tmp_path / "first.dat"
    second = tmp_path / "second.dat"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    adapter._send_email_with_attachments(
        "user@test.com",
        "",
        [str(first), str(second)],
    )

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/mixed"
    assert list(message.keys()) == [
        "Content-Type",
        "MIME-Version",
        "From",
        "To",
        "Subject",
        "Date",
        "Message-ID",
    ]
    assert message["Subject"] == "Re: Hermes Agent"
    assert message["In-Reply-To"] is None
    assert message["References"] is None
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "application/octet-stream",
        "application/octet-stream",
    ]
    attachments = message.get_payload()
    assert [part.get_filename() for part in attachments] == [
        "first.dat",
        "second.dat",
    ]
    assert [part.get_payload(decode=True) for part in attachments] == [
        b"first",
        b"second",
    ]


def test_html_with_attachments_and_omitted_empty_body_remains_attachments_only():
    message = build_email_message(
        from_address="hermes@test.com",
        to_address="user@test.com",
        subject="Attachment only",
        body="",
        date=_DATE,
        html_body="<p>Rendered empty body</p>",
        attachments=(MimeAttachment(filename="report.bin", content=b"report"),),
        include_empty_body=False,
    )

    assert message.get_content_type() == "multipart/mixed"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "application/octet-stream",
    ]
    attachment = message.get_payload()[0]
    assert attachment.get_filename() == "report.bin"
    assert attachment.get_payload(decode=True) == b"report"


def test_standalone_send_preserves_legacy_text_plain_envelope(monkeypatch):
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    smtp = MagicMock()
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", MagicMock(return_value=smtp))
    monkeypatch.setattr(email_adapter, "formatdate", lambda *, localtime: _DATE)
    config = SimpleNamespace(
        token=None,
        api_key=None,
        extra={"address": "hermes@test.com", "smtp_host": "smtp.test.com"},
    )

    result = asyncio.run(
        email_adapter._standalone_send(
            config,
            "user@test.com",
            "Message autonome accentué.",
        )
    )

    message = smtp.send_message.call_args.args[0]
    assert result == {
        "success": True,
        "platform": "email",
        "chat_id": "user@test.com",
    }
    assert message.get_content_type() == "text/plain"
    assert not message.is_multipart()
    assert list(message.keys()) == [
        "Content-Type",
        "MIME-Version",
        "Content-Transfer-Encoding",
        "From",
        "To",
        "Subject",
        "Date",
    ]
    assert message["From"] == "hermes@test.com"
    assert message["To"] == "user@test.com"
    assert message["Subject"] == "Hermes Agent"
    assert message["Date"] == _DATE
    assert message["Message-ID"] is None
    assert message["In-Reply-To"] is None
    assert message["References"] is None
    _assert_utf8_plain_part(message, "Message autonome accentué.")


def test_rich_html_reply_uses_sanitized_markdown_alternative(monkeypatch):
    adapter = _make_adapter(monkeypatch, {"rich_html_enabled": True})
    smtp = _capture_adapter_message(monkeypatch, adapter)
    adapter._thread_context["user@test.com"] = {
        "subject": "Formatting",
        "message_id": "<original@test.com>",
    }
    markdown_body = "# Résumé\n\nTexte **important** avec [lien](https://example.com)."

    adapter._send_email("user@test.com", markdown_body)

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/alternative"
    assert message["Subject"] == "Re: Formatting"
    assert message["In-Reply-To"] == "<original@test.com>"
    assert message["References"] == "<original@test.com>"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/alternative",
        "text/plain",
        "text/html",
    ]
    plain_part, html_part = message.get_payload()
    _assert_utf8_plain_part(plain_part, markdown_body)
    assert html_part.get_content_charset() == "utf-8"
    rendered = html_part.get_payload(decode=True).decode("utf-8")
    assert "<h1>Résumé</h1>" in rendered
    assert "<strong>important</strong>" in rendered
    assert 'href="https://example.com"' in rendered


def test_rich_html_with_attachment_nests_alternative_before_attachment(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _make_adapter(monkeypatch, {"rich_html_enabled": "true"})
    smtp = _capture_adapter_message(monkeypatch, adapter)
    source = tmp_path / "report.bin"
    source.write_bytes(b"report")

    adapter._send_email_with_attachment(
        "user@test.com",
        "Voir **rapport**.",
        str(source),
    )

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/mixed"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "multipart/alternative",
        "text/plain",
        "text/html",
        "application/octet-stream",
    ]
    alternative, attachment = message.get_payload()
    assert [part.get_content_type() for part in alternative.get_payload()] == [
        "text/plain",
        "text/html",
    ]
    assert attachment.get_filename() == "report.bin"
    assert attachment.get_payload(decode=True) == b"report"


def test_standalone_rich_html_uses_markdown_alternative(monkeypatch):
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    smtp = MagicMock()
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", MagicMock(return_value=smtp))
    monkeypatch.setattr(email_utils, "formatdate", lambda *, localtime: _DATE)
    config = SimpleNamespace(
        token=None,
        api_key=None,
        extra={
            "address": "hermes@test.com",
            "smtp_host": "smtp.test.com",
            "rich_html_enabled": True,
        },
    )

    result = asyncio.run(
        email_adapter._standalone_send(
            config,
            "user@test.com",
            "Message **riche**.",
        )
    )

    assert result["success"] is True
    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/alternative"
    plain_part, html_part = message.get_payload()
    _assert_utf8_plain_part(plain_part, "Message **riche**.")
    rendered = html_part.get_payload(decode=True).decode("utf-8")
    assert "<strong>riche</strong>" in rendered


def test_explicitly_disabled_rich_html_keeps_legacy_mime_shape(monkeypatch):
    adapter = _make_adapter(monkeypatch, {"rich_html_enabled": False})
    smtp = _capture_adapter_message(monkeypatch, adapter)
    renderer = MagicMock(side_effect=AssertionError("renderer must stay disabled"))
    monkeypatch.setattr(email_adapter, "render_markdown_html", renderer)

    adapter._send_email("user@test.com", "Texte **non rendu**.")

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/mixed"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "text/plain",
    ]
    _assert_utf8_plain_part(message.get_payload()[0], "Texte **non rendu**.")
    renderer.assert_not_called()


def test_markdown_rendering_sanitizes_dangerous_raw_html_and_urls():
    rendered = render_markdown_html(
        """# Safe heading

<script>alert('script')</script>
<p onclick="alert('event')">Paragraph</p>
[unsafe](javascript:alert('link'))
<a href="javascript:alert('raw')" onmouseover="alert('hover')">raw link</a>
"""
    )

    assert "<h1>Safe heading</h1>" in rendered
    assert "Paragraph" in rendered
    assert "unsafe" in rendered
    assert "raw link" in rendered
    lowered = rendered.lower()
    assert "<script" not in lowered
    assert "onclick" not in lowered
    assert "onmouseover" not in lowered
    assert "javascript:" not in lowered
    assert "script')" not in lowered


def test_message_sanitizer_preserves_safe_links_and_drops_active_content():
    cleaned = sanitize_message_html(
        '<p class="discarded">Safe <a href="https://example.com">link</a></p>'
        '<img src="https://tracker.example/pixel.png" onerror="alert(1)">'
        "<style>body { display: none }</style>"
    )

    assert cleaned.startswith("<p>Safe ")
    assert 'href="https://example.com"' in cleaned
    assert 'rel="noopener noreferrer"' in cleaned
    lowered = cleaned.lower()
    assert "class=" not in lowered
    assert "<img" not in lowered
    assert "onerror" not in lowered
    assert "<style" not in lowered
    assert "display: none" not in lowered


def test_message_sanitizer_drops_relative_urls():
    cleaned = sanitize_message_html(
        '<p><a href="/docs/setup">relative link</a> '
        '<a href="https://example.com/docs/setup">absolute link</a></p>'
    )

    assert "relative link" in cleaned
    assert 'href="/docs/setup"' not in cleaned
    assert 'href="https://example.com/docs/setup"' in cleaned


@pytest.mark.parametrize(
    "signature",
    [
        "John",
        {"enabled": True, "text": ""},
    ],
    ids=["not-a-mapping", "enabled-with-empty-text"],
)
def test_invalid_signature_config_disables_only_signature_and_keeps_rich_html(
    monkeypatch,
    caplog,
    signature,
):
    adapter = _make_adapter(
        monkeypatch,
        {"rich_html_enabled": True, "signature": signature},
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)

    adapter._send_email("user@test.com", "Message **riche**.")

    assert adapter._signature is None
    assert adapter._rich_html_enabled is True
    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/alternative"
    plain_part, html_part = message.get_payload()
    _assert_utf8_plain_part(plain_part, "Message **riche**.")
    assert "<strong>riche</strong>" in html_part.get_payload(decode=True).decode(
        "utf-8"
    )
    assert "Invalid Email signature configuration; signature disabled" in caplog.text


def test_valid_signature_config_initializes_unchanged(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        {
            "rich_html_enabled": True,
            "signature": {
                "enabled": True,
                "text": "Canonical signature",
                "html": "<strong>Rendered signature</strong>",
            },
        },
    )

    assert adapter._signature == MimeSignature(
        text="Canonical signature",
        html="<strong>Rendered signature</strong>",
    )


def test_disabled_signature_keeps_unsigned_legacy_body(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        {
            "signature": {
                "enabled": False,
                "text": "Must not appear",
                "html": "<strong>Must not appear</strong>",
            }
        },
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)

    adapter._send_email("user@test.com", "Original body")

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/mixed"
    _assert_utf8_plain_part(message.get_payload()[0], "Original body")
    assert "Must not appear" not in message.as_string()


def test_plain_signature_appends_canonical_text_without_changing_mime_shape(
    monkeypatch,
):
    adapter = _make_adapter(
        monkeypatch,
        {
            "signature": {
                "enabled": True,
                "text": "Hermes Agent\nInternal assistant",
            }
        },
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)

    adapter._send_email("user@test.com", "Original body")

    message = smtp.send_message.call_args.args[0]
    assert message.get_content_type() == "multipart/mixed"
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "text/plain",
    ]
    _assert_utf8_plain_part(
        message.get_payload()[0],
        "Original body\n\nHermes Agent\nInternal assistant",
    )


def test_rich_signature_derives_html_from_canonical_text(monkeypatch):
    signature_text = "**Hermes Agent**\n\nInternal assistant"
    adapter = _make_adapter(
        monkeypatch,
        {
            "rich_html_enabled": True,
            "signature": {"enabled": True, "text": signature_text},
        },
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)

    adapter._send_email("user@test.com", "Hello **there**.")

    message = smtp.send_message.call_args.args[0]
    plain_part, html_part = message.get_payload()
    plain = plain_part.get_payload(decode=True).decode("utf-8")
    rendered = html_part.get_payload(decode=True).decode("utf-8")
    assert plain == f"Hello **there**.\n\n{signature_text}"
    assert plain.count("Hermes Agent") == 1
    assert rendered.count("Hermes Agent") == 1
    assert "<strong>Hermes Agent</strong>" in rendered
    assert "<p>Internal assistant</p>" in rendered


def test_provided_signature_html_uses_separate_sanitizer_policy(monkeypatch):
    signature_html = (
        '<div style="color: #663399; position: fixed" onclick="alert(1)">'
        "<strong>Signature Team</strong>"
        "<script>alert('script')</script>"
        '<a href="javascript:alert(2)">unsafe</a>'
        '<a href="mailto:team@example.com">mail</a>'
        '<img src="https://tracker.example/pixel.png">'
        "</div>"
    )
    adapter = _make_adapter(
        monkeypatch,
        {
            "rich_html_enabled": True,
            "signature": {
                "enabled": True,
                "text": "Canonical signature",
                "html": signature_html,
            },
        },
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)

    adapter._send_email("user@test.com", "Message body")

    plain_part, html_part = smtp.send_message.call_args.args[0].get_payload()
    plain = plain_part.get_payload(decode=True).decode("utf-8")
    rendered = html_part.get_payload(decode=True).decode("utf-8")
    assert plain == "Message body\n\nCanonical signature"
    assert plain.count("Canonical signature") == 1
    assert rendered.count("Signature Team") == 1
    assert "<div" in rendered
    assert "color:" in rendered
    assert 'href="mailto:team@example.com"' in rendered
    lowered = rendered.lower()
    assert "position:" not in lowered
    assert "onclick" not in lowered
    assert "<script" not in lowered
    assert "javascript:" not in lowered
    assert "<img" not in lowered


def test_signature_sanitizer_keeps_safe_signature_layout_only():
    cleaned = sanitize_signature_html(
        '<div style="font-weight: bold; position: fixed">'
        '<span style="color: blue; background-image: url(https://tracker)">Team</span>'
        '<a href="tel:+33123456789">Call</a>'
        "</div>"
    )

    assert "<div" in cleaned
    assert "<span" in cleaned
    assert "font-weight:bold" in cleaned
    assert "color:blue" in cleaned
    assert 'href="tel:+33123456789"' in cleaned
    lowered = cleaned.lower()
    assert "position:" not in lowered
    assert "background-image" not in lowered


def test_signature_adds_plain_part_to_empty_attachment_message(
    monkeypatch,
    tmp_path: Path,
):
    adapter = _make_adapter(
        monkeypatch,
        {"signature": {"enabled": True, "text": "Attachment signature"}},
    )
    smtp = _capture_adapter_message(monkeypatch, adapter)
    source = tmp_path / "report.bin"
    source.write_bytes(b"report")

    adapter._send_email_with_attachment("user@test.com", "", str(source))

    message = smtp.send_message.call_args.args[0]
    assert [part.get_content_type() for part in message.walk()] == [
        "multipart/mixed",
        "text/plain",
        "application/octet-stream",
    ]
    plain_part, attachment = message.get_payload()
    _assert_utf8_plain_part(plain_part, "Attachment signature")
    assert attachment.get_payload(decode=True) == b"report"


def test_standalone_rich_signature_is_added_once_to_both_alternatives(monkeypatch):
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    smtp = MagicMock()
    monkeypatch.setattr(email_adapter.smtplib, "SMTP", MagicMock(return_value=smtp))
    monkeypatch.setattr(email_utils, "formatdate", lambda *, localtime: _DATE)
    config = SimpleNamespace(
        token=None,
        api_key=None,
        extra={
            "address": "hermes@test.com",
            "smtp_host": "smtp.test.com",
            "rich_html_enabled": True,
            "signature": {"enabled": True, "text": "Standalone signature"},
        },
    )

    result = asyncio.run(
        email_adapter._standalone_send(config, "user@test.com", "Standalone body")
    )

    assert result["success"] is True
    plain_part, html_part = smtp.send_message.call_args.args[0].get_payload()
    plain = plain_part.get_payload(decode=True).decode("utf-8")
    rendered = html_part.get_payload(decode=True).decode("utf-8")
    assert plain == "Standalone body\n\nStandalone signature"
    assert plain.count("Standalone signature") == 1
    assert rendered.count("Standalone signature") == 1
