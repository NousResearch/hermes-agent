"""platforms.email.mark_seen: false leaves mail unread without re-dispatching it."""

import os
import unittest
import uuid
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch


def _make_adapter(**extra):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": "hermes@test.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        return EmailAdapter(PlatformConfig(enabled=True, extra=extra))


def _raw_email(sender="user@test.com"):
    msg = MIMEText("Test body", "plain", "utf-8")
    msg["From"] = sender
    msg["Subject"] = "Hello"
    msg["Message-ID"] = f"<{uuid.uuid4().hex[:8]}@test.com>"
    return msg.as_bytes()


class _FakeInbox:
    """IMAP double: answers UNSEEN with *unseen* and records every fetch/store call."""

    def __init__(self, unseen):
        self.unseen = unseen
        self.calls = []

    def uid(self, command, *args):
        self.calls.append((command, *args))
        if command == "search":
            return ("OK", [b" ".join(self.unseen)])
        if command == "fetch":
            return ("OK", [(args[0], _raw_email())])
        return ("OK", [None])

    def fetches(self):
        return [c[2] for c in self.calls if c[0] == "fetch" and "HEADER.FIELDS" not in c[2]]

    def stores(self):
        return [c for c in self.calls if c[0] == "store"]


def _fetch(adapter, inbox, accept=True):
    imap = MagicMock()
    imap.uid.side_effect = inbox.uid
    with patch("imaplib.IMAP4_SSL", return_value=imap):
        return adapter._fetch_new_messages(lambda _c: accept)


class TestMarkSeen(unittest.TestCase):
    def test_default_marks_seen(self):
        adapter = _make_adapter()
        inbox = _FakeInbox([b"1"])
        self.assertEqual(len(_fetch(adapter, inbox)), 1)
        self.assertEqual(inbox.fetches(), ["(RFC822)"])
        rejected = _FakeInbox([b"2"])
        _fetch(adapter, rejected, accept=False)
        self.assertEqual(len(rejected.stores()), 1)

    def test_mark_seen_false_peeks_and_never_stores(self):
        adapter = _make_adapter(mark_seen=False)
        inbox = _FakeInbox([b"1"])
        self.assertEqual(len(_fetch(adapter, inbox)), 1)
        self.assertEqual(inbox.fetches(), ["(BODY.PEEK[])"])
        rejected = _FakeInbox([b"2"])
        _fetch(adapter, rejected, accept=False)
        self.assertEqual(rejected.stores(), [])

    def test_env_var_overrides(self):
        with patch.dict(os.environ, {"EMAIL_MARK_SEEN": "false"}):
            adapter = _make_adapter()
        self.assertFalse(adapter._mark_seen)

    def test_unread_mail_is_dispatched_once(self):
        adapter = _make_adapter(mark_seen=False)
        self.assertEqual(len(_fetch(adapter, _FakeInbox([b"7"]))), 1)
        # Still UNSEEN on the server: the next poll must not dispatch it again.
        self.assertEqual(_fetch(adapter, _FakeInbox([b"7"])), [])

    def test_trim_does_not_redispatch_old_unread_mail(self):
        adapter = _make_adapter(mark_seen=False)
        # Startup baseline: every UID in a large inbox is remembered, then trimmed to the newest half.
        adapter._seen_uids = {str(i).encode() for i in range(1, adapter._seen_uids_max + 501)}
        adapter._trim_seen_uids()
        results = _fetch(adapter, _FakeInbox([b"5", b"1200", str(adapter._seen_uids_max + 600).encode()]))
        self.assertEqual([r["uid"] for r in results], [str(adapter._seen_uids_max + 600).encode()])

    def test_default_mode_has_no_floor(self):
        adapter = _make_adapter()
        adapter._seen_uids = {b"100", b"200"}
        self.assertEqual(adapter._unread_floor(), 0)


if __name__ == "__main__":
    unittest.main()
