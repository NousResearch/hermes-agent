"""A failed pre-authorization check is retried on the next polls before the message is dropped.

The header-only fetch that decides whether a sender may be read can answer ``OK`` with no header
literal for a just-delivered message; the same fetch reads fine a poll later. Marking that mail
read at once loses it for good.
"""

import logging
import os
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

HEADERS = b"From: user@test.com\r\n\r\n"
# A just-delivered message answered without its header literal (an unsolicited FLAGS line only).
UNUSABLE = [b"3 (UID 3 FLAGS (\\Recent))"]


def _adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter
    with patch.dict(os.environ, {"EMAIL_ADDRESS": "hermes@test.com", "EMAIL_PASSWORD": "secret",
                                 "EMAIL_IMAP_HOST": "imap.test.com", "EMAIL_SMTP_HOST": "smtp.test.com"}):
        return EmailAdapter(PlatformConfig(enabled=True))


def _raw_email():
    msg = MIMEText("body", "plain", "utf-8")
    msg["From"], msg["Subject"], msg["Message-ID"] = "user@test.com", "Hello", "<m1@test.com>"
    return msg.as_bytes()


class Mailbox:
    """A fake INBOX: the *unseen* UIDs, the header response per UID, and the UIDs stored ``\\Seen``."""

    def __init__(self, unseen=b"3", headers=None):
        self.unseen, self.headers, self.stored = unseen, headers or {}, []

    def imap(self):
        def uid(command, *args):
            if command == "search":
                return "OK", [self.unseen]
            if command == "fetch" and "BODY.PEEK" in args[1]:
                return "OK", self.headers.get(args[0], [(args[0] + b" (BODY[HEADER.FIELDS (FROM)] {24}", HEADERS)])
            if command == "fetch":
                return "OK", [(args[0] + b" (RFC822 {123}", _raw_email())]
            if command == "store":
                self.stored.append(args[0])
                return "OK", []
            return "NO", []

        imap = MagicMock()
        imap.uid.side_effect = uid
        return imap


def _poll(adapter, box, preauthorize=lambda _c: True):
    with patch("imaplib.IMAP4_SSL", return_value=box.imap()):
        return adapter._fetch_new_messages(preauthorize)


def test_an_unusable_header_response_is_retried_on_the_next_poll():
    adapter = _adapter()
    box = Mailbox(headers={b"3": UNUSABLE})
    assert _poll(adapter, box) == []
    assert box.stored == [] and b"3" not in adapter._seen_uids
    del box.headers[b"3"]
    [msg] = _poll(adapter, box)
    assert msg["sender_addr"] == "user@test.com"
    assert adapter._preauth_failures == {}


def test_a_header_response_that_stays_unusable_is_dropped_after_the_last_attempt(caplog):
    from plugins.platforms.email.adapter import _PREAUTH_ATTEMPTS
    adapter = _adapter()
    box = Mailbox(headers={b"3": UNUSABLE})
    with caplog.at_level(logging.WARNING):
        for _ in range(_PREAUTH_ATTEMPTS - 1):
            assert _poll(adapter, box) == [] and box.stored == []
        assert _poll(adapter, box) == []
    assert box.stored == [b"3"] and b"3" in adapter._seen_uids
    assert "Unusable pre-authorization headers for UID b'3'" in caplog.text
    assert "['bytes']" in caplog.text  # the retry line names the response's shape, never its content


def test_a_check_that_raises_is_retried_and_a_refused_sender_is_not():
    adapter = _adapter()
    box = Mailbox()

    def unavailable(_candidate):
        raise RuntimeError("authorization unavailable")

    assert _poll(adapter, box, unavailable) == [] and box.stored == []
    assert len(_poll(adapter, box)) == 1
    refused = Mailbox(unseen=b"4")
    assert _poll(_adapter(), refused, lambda _c: False) == [] and refused.stored == [b"4"]


def test_an_oversized_header_response_is_dropped_at_once():
    from plugins.platforms.email.adapter import _MAX_PREAUTH_HEADER_BYTES
    box = Mailbox(headers={b"3": [(b"3 (BODY[HEADER] {}", b"x" * (_MAX_PREAUTH_HEADER_BYTES + 1))]})
    assert _poll(_adapter(), box) == [] and box.stored == [b"3"]
