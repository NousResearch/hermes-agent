"""Shared MIME construction for outgoing Email platform messages."""

from __future__ import annotations

from dataclasses import dataclass
from email import encoders
from email.message import Message
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional, Sequence

import markdown
import nh3


_MESSAGE_HTML_TAGS = {
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_MESSAGE_HTML_ATTRIBUTES = {"a": {"href", "title"}}
_MESSAGE_HTML_CLEAN_CONTENT_TAGS = {
    "embed",
    "iframe",
    "noscript",
    "object",
    "script",
    "style",
    "template",
}
_MESSAGE_HTML_URL_SCHEMES = {"http", "https", "mailto"}

_SIGNATURE_HTML_TAGS = _MESSAGE_HTML_TAGS | {
    "div",
    "small",
    "span",
    "sub",
    "sup",
}
_SIGNATURE_HTML_ATTRIBUTES = {
    "a": {"href", "style", "title"},
    "div": {"style"},
    "p": {"style"},
    "small": {"style"},
    "span": {"style"},
    "table": {"border", "cellpadding", "cellspacing", "role", "style", "width"},
    "td": {"align", "colspan", "rowspan", "style", "valign", "width"},
    "th": {"align", "colspan", "rowspan", "style", "valign", "width"},
}
_SIGNATURE_HTML_STYLE_PROPERTIES = {
    "border",
    "border-bottom",
    "border-color",
    "border-left",
    "border-right",
    "border-style",
    "border-top",
    "border-width",
    "color",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "line-height",
    "margin",
    "margin-bottom",
    "margin-left",
    "margin-right",
    "margin-top",
    "max-width",
    "padding",
    "padding-bottom",
    "padding-left",
    "padding-right",
    "padding-top",
    "text-align",
    "text-decoration",
    "vertical-align",
    "white-space",
    "width",
}
_SIGNATURE_HTML_URL_SCHEMES = _MESSAGE_HTML_URL_SCHEMES | {"tel"}


def sanitize_message_html(html: str) -> str:
    """Sanitize rendered message HTML using the Email message policy."""
    return nh3.clean(
        html,
        tags=_MESSAGE_HTML_TAGS,
        clean_content_tags=_MESSAGE_HTML_CLEAN_CONTENT_TAGS,
        attributes=_MESSAGE_HTML_ATTRIBUTES,
        strip_comments=True,
        link_rel="noopener noreferrer",
        url_schemes=_MESSAGE_HTML_URL_SCHEMES,
        url_relative="deny",
    )


def render_markdown_html(text: str) -> str:
    """Render Markdown and sanitize the resulting Email HTML fragment."""
    rendered = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html",
    )
    return sanitize_message_html(rendered)


def sanitize_signature_html(html: str) -> str:
    """Sanitize signature HTML using its separate layout-oriented policy."""
    return nh3.clean(
        html,
        tags=_SIGNATURE_HTML_TAGS,
        clean_content_tags=_MESSAGE_HTML_CLEAN_CONTENT_TAGS,
        attributes=_SIGNATURE_HTML_ATTRIBUTES,
        strip_comments=True,
        link_rel="noopener noreferrer",
        url_schemes=_SIGNATURE_HTML_URL_SCHEMES,
        filter_style_properties=_SIGNATURE_HTML_STYLE_PROPERTIES,
        url_relative="deny",
    )


def _render_signature_text_html(text: str) -> str:
    rendered = markdown.markdown(
        text,
        extensions=["nl2br", "sane_lists"],
        output_format="html",
    )
    return sanitize_signature_html(rendered)


@dataclass(frozen=True)
class MimeAttachment:
    """An attachment payload ready to be added to an outgoing message."""

    filename: str
    content: bytes


def _validate_attachment_filename(filename: str) -> None:
    if any(character in filename for character in ("\x00", "\r", "\n")):
        raise ValueError(
            "attachment filename must not contain NUL, CR, or LF characters"
        )


@dataclass(frozen=True)
class MimeSignature:
    """Validated plain-text and sanitized HTML signature variants."""

    text: str
    html: str


def prepare_signature(
    *,
    enabled: bool,
    text: Optional[str] = None,
    html: Optional[str] = None,
) -> Optional[MimeSignature]:
    """Validate and prepare a configured Email signature."""
    if not enabled:
        return None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "email signature.text is required when signature.enabled is true"
        )
    if html is not None and not isinstance(html, str):
        raise ValueError("email signature.html must be a string when provided")

    if html and html.strip():
        sanitized_html = sanitize_signature_html(html)
        if not sanitized_html.strip():
            sanitized_html = _render_signature_text_html(text)
    else:
        sanitized_html = _render_signature_text_html(text)
    return MimeSignature(text=text, html=sanitized_html)


def build_email_message(
    *,
    from_address: str,
    to_address: str,
    subject: str,
    body: str,
    date: str,
    message_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    references: Optional[str] = None,
    attachments: Sequence[MimeAttachment] = (),
    html_body: Optional[str] = None,
    signature: Optional[MimeSignature] = None,
    force_multipart: bool = False,
    include_empty_body: bool = True,
) -> Message:
    """Build an outgoing message while preserving the legacy plain-text MIME.

    ``force_multipart`` and ``include_empty_body`` encode two historical Email
    adapter behaviors: gateway replies always use ``multipart/mixed``, while
    attachment sends omit an empty text part.  The standalone sender remains a
    direct ``text/plain`` message when it has no attachments.
    """
    plain_body = body
    effective_html_body = html_body
    if signature is not None:
        plain_body = f"{body}\n\n{signature.text}" if body else signature.text
        if html_body is not None:
            effective_html_body = (
                f"{html_body}\n<br>\n{signature.html}" if html_body else signature.html
            )

    include_body = bool(plain_body) or include_empty_body
    alternative = None
    if effective_html_body is not None and include_body:
        alternative = MIMEMultipart("alternative")
        if attachments:
            message = MIMEMultipart()
        else:
            message = alternative
    elif force_multipart or attachments:
        message: Message = MIMEMultipart()
    else:
        message = MIMEText(plain_body, "plain", "utf-8")

    message["From"] = from_address
    message["To"] = to_address
    message["Subject"] = subject
    if in_reply_to:
        message["In-Reply-To"] = in_reply_to
    if references:
        message["References"] = references
    message["Date"] = date
    if message_id:
        message["Message-ID"] = message_id

    if alternative is not None:
        alternative.attach(MIMEText(plain_body, "plain", "utf-8"))
        alternative.attach(MIMEText(effective_html_body, "html", "utf-8"))
        if message is not alternative:
            message.attach(alternative)
    elif message.is_multipart():
        if include_body:
            message.attach(MIMEText(plain_body, "plain", "utf-8"))

    for attachment in attachments:
        _validate_attachment_filename(attachment.filename)
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.content)
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            "attachment",
            filename=attachment.filename,
        )
        message.attach(part)

    return message
