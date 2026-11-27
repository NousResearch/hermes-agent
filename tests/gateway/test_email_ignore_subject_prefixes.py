"""Offline regression tests for ``platforms.email.ignore_subject_prefixes``.

An operator's script/cron reminder loop (e.g. a seminar scheduler mailing the agent's own
account) must not be answered. Before this filter, one such inbound message produced THREE
emails: the home-channel notice, the auto-TTS "couldn't deliver the audio attachment"
diagnostic, and the model's own reply.

The drop happens at PARSE time, so all three paths are unreachable at once. These tests pin
that ordering property, and that ordinary human mail from the same person is untouched.

No network, no SMTP send, no real mailbox: pure parse-level assertions on the real adapter.
"""

import os
import unittest
from unittest.mock import patch

MACHINE_RAW = (
    b"From: Chung Hwan Kim <chungkim@utdallas.edu>\r\n"
    b"To: hermes@example.com\r\n"
    b"Subject: [s3sem] Paper to present on Oct 9\r\n"
    b"Message-ID: <sem1@utdallas.edu>\r\n"
    b"Date: Fri, 2 Oct 2026 09:00:00 -0500\r\n"
    b"\r\n"
    b"Dear Jaehyun Park,\r\nIt is time to send me your paper selection.\r\n"
)

HUMAN_RAW = (
    b"From: Chung Hwan Kim <chungkim@utdallas.edu>\r\n"
    b"To: hermes@example.com\r\n"
    b"Subject: Question about the seminar schedule\r\n"
    b"Message-ID: <human1@utdallas.edu>\r\n"
    b"Date: Fri, 2 Oct 2026 09:30:00 -0500\r\n"
    b"\r\n"
    b"Can you move Thursday's seminar to Friday?\r\n"
)

# Same machine-generated reminder, but as a human REPLY in the thread.
REPLY_RAW = (
    b"From: Chung Hwan Kim <chungkim@utdallas.edu>\r\n"
    b"To: hermes@example.com\r\n"
    b"Subject: Re: [s3sem] Paper to present on Oct 9\r\n"
    b"Message-ID: <sem2@utdallas.edu>\r\n"
    b"In-Reply-To: <sem1@utdallas.edu>\r\n"
    b"Date: Fri, 2 Oct 2026 10:00:00 -0500\r\n"
    b"\r\n"
    b"I already approved Jaehyun's selection, please do not remind him again.\r\n"
)


def _adapter(extra=None):
    from gateway.config import PlatformConfig
    from plugins.platforms.email.adapter import EmailAdapter
    return EmailAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))


class TestIgnoreSubjectPrefixes(unittest.TestCase):
    """The machine-generated seminar reminder is dropped before ANY reply path."""

    ENV = {
        "EMAIL_ADDRESS": "hermes@example.com",
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
        "EMAIL_ALLOWED_USERS": "chungkim@utdallas.edu",
    }

    def _parse(self, raw, extra):
        with patch.dict(os.environ, self.ENV):
            return _adapter(extra)._parse_fetched_message(b"401", raw)

    def test_unconfigured_adapter_is_unchanged(self):
        """Default (no filter configured) dispatches as before — no behavior change."""
        self.assertIsNotNone(self._parse(MACHINE_RAW, None))

    def test_configured_prefix_drops_machine_reminder(self):
        """The seminar reminder never becomes a dispatchable message."""
        self.assertIsNone(self._parse(MACHINE_RAW, {"ignore_subject_prefixes": ["[s3sem]"]}))

    def test_prefix_match_is_case_insensitive(self):
        self.assertIsNone(self._parse(MACHINE_RAW, {"ignore_subject_prefixes": ["[S3SEM]"]}))

    def test_legitimate_human_mail_is_unaffected(self):
        """Same sender, unrelated subject: still reaches the agent."""
        parsed = self._parse(HUMAN_RAW, {"ignore_subject_prefixes": ["[s3sem]"]})
        self.assertIsNotNone(parsed)
        assert parsed is not None  # narrow for the type checker
        self.assertEqual(parsed["subject"], "Question about the seminar schedule")

    def test_human_reply_in_filtered_thread_still_delivered(self):
        """A real reply in the ignored thread is NOT swept away.

        The user replying about that exact seminar is a legitimate request; a prefix
        filter must not silence the human in the thread it silences the bot in.
        """
        parsed = self._parse(REPLY_RAW, {"ignore_subject_prefixes": ["[s3sem]"]})
        self.assertIsNotNone(parsed)
        assert parsed is not None  # narrow for the type checker
        self.assertEqual(parsed["subject"], "Re: [s3sem] Paper to present on Oct 9")

    def test_blank_entries_are_ignored(self):
        """Blank/whitespace entries never match everything."""
        self.assertIsNotNone(self._parse(MACHINE_RAW, {"ignore_subject_prefixes": ["", "   "]}))

    def test_empty_list_means_no_filtering(self):
        self.assertIsNotNone(self._parse(MACHINE_RAW, {"ignore_subject_prefixes": []}))

    def test_unrelated_prefix_leaves_reminder_alone(self):
        """The filter is scoped to what the operator listed."""
        self.assertIsNotNone(self._parse(MACHINE_RAW, {"ignore_subject_prefixes": ["[build-bot]"]}))

    def test_drop_happens_before_sender_gate_and_event(self):
        """No MessageEvent and no sender-gate work for a filtered message.

        This is the property that suppresses all three replies (home-channel notice,
        audio diagnostic, model reply): they are all downstream of dispatch.
        """
        with patch.dict(os.environ, self.ENV):
            adapter = _adapter({"ignore_subject_prefixes": ["[s3sem]"]})
            seen = []

            def _record(sender_addr, msg_data):
                seen.append(sender_addr)
                return True

            adapter._sender_accepted = _record
            self.assertIsNone(adapter._parse_fetched_message(b"402", MACHINE_RAW))
            self.assertEqual(seen, [], "sender gate ran for an operator-ignored message")


class TestParseDropMarksSeenExactlyOnce(unittest.TestCase):
    """A dropped message must not be re-fetched forever by the IMAP poll loop."""

    def test_dropped_message_is_not_retried(self):
        from unittest.mock import MagicMock
        with patch.dict(os.environ, TestIgnoreSubjectPrefixes.ENV):
            adapter = _adapter({"ignore_subject_prefixes": ["[s3sem]"]})
            imap = MagicMock()
            # search -> one unseen UID; fetch -> a real RFC822 payload, so the message
            # reaches _parse_fetched_message and is dropped by the filter itself.
            imap.uid.side_effect = lambda *a, **k: (
                ("OK", [b"401"]) if a and a[0] == "search" else ("OK", [(b"1 (RFC822 {n}", MACHINE_RAW)])
            )
            imap.__enter__.return_value = imap
            imap.__exit__.return_value = False
            adapter._inbox = MagicMock(return_value=imap)
            results = adapter._fetch_new_messages(lambda _c: True)
            self.assertEqual(results, [], "filtered message must not be dispatched")
            self.assertIn(b"401", adapter._seen_uids, "filtered message must not be re-fetched")


if __name__ == "__main__":
    unittest.main()