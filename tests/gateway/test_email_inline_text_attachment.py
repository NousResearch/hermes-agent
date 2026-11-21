"""A text file Apple Mail / iOS Mail attaches as ``inline; filename=...`` reaches the agent.

Those clients mark attached files ``Content-Disposition: inline`` with a filename. The adapter
skipped every inline ``text/plain`` / ``text/html`` part as a body part, so an attached ``.txt``
or ``.html`` file was neither an attachment nor the body (the body is the first plain part).
"""

import os
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig


@pytest.fixture()
def adapter():
    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret", "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter
        return EmailAdapter(PlatformConfig(enabled=True))


def _apple_mail(*, html_body: bool = False) -> bytes:
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["To"] = "hermes@test.com"
    msg["Subject"] = "Notes"
    if html_body:
        msg.set_content("<p>Summarize the attached notes.</p>", subtype="html")
    else:
        msg.set_content("Summarize the attached notes.")
    msg.add_attachment("line one\nline two\n", subtype="plain", disposition="inline", filename="meeting-notes.txt")
    msg.add_attachment("<b>report</b>", subtype="html", disposition="inline", filename="report.html")
    msg.add_attachment(b"a,b\n1,2\n", maintype="text", subtype="csv", disposition="inline", filename="data.csv")
    return msg.as_bytes()


@pytest.mark.parametrize("html_body", [False, True])
def test_inline_text_files_are_attachments_and_not_the_body(adapter, html_body):
    parsed = adapter._parse_fetched_message(b"1", _apple_mail(html_body=html_body))

    assert parsed["body"].strip() == "Summarize the attached notes."
    files = {a["filename"]: a for a in parsed["attachments"]}
    assert set(files) == {"meeting-notes.txt", "report.html", "data.csv"}
    assert Path(files["meeting-notes.txt"]["path"]).read_text(encoding="utf-8") == "line one\nline two\n"
    assert files["meeting-notes.txt"]["media_type"] == "text/plain"


def test_inline_body_part_without_filename_is_still_the_body(adapter):
    msg = EmailMessage()
    msg["From"] = "alice@example.com"
    msg["Subject"] = "Hi"
    msg.set_content("Plain body.", disposition="inline")
    msg.add_alternative("<p>Plain body.</p>", subtype="html", disposition="inline")

    parsed = adapter._parse_fetched_message(b"2", msg.as_bytes())

    assert parsed["body"].strip() == "Plain body."
    assert parsed["attachments"] == []
