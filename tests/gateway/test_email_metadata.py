"""Email adapter honors send() metadata (subject/cc/fresh).

The base-class ``send(..., metadata=...)`` contract lets a caller (e.g. a
platform plugin) override the composed message: a verbatim ``subject``, a
``cc`` recipient list, and a ``fresh`` flag that suppresses reply threading.
The email adapter previously accepted ``metadata`` on ``send()`` but dropped
it before ``_send_email``, silently losing all three. These tests pin the
restored contract and its backward compatibility.
"""

import asyncio
import email.utils
import os
import unittest
from unittest.mock import MagicMock, patch


def _make_adapter(address="hermes@test.com"):
    from gateway.config import PlatformConfig

    with patch.dict(os.environ, {
        "EMAIL_ADDRESS": address,
        "EMAIL_PASSWORD": "secret",
        "EMAIL_IMAP_HOST": "imap.test.com",
        "EMAIL_SMTP_HOST": "smtp.test.com",
    }):
        from plugins.platforms.email.adapter import EmailAdapter

        adapter = EmailAdapter(PlatformConfig(enabled=True))
    return adapter


def _sent_message(mock_smtp):
    """The MIMEMultipart passed to smtp.send_message() during a send."""
    mock_server = mock_smtp.return_value
    return mock_server.send_message.call_args[0][0]


def _envelope_recipients(mock_smtp):
    """Recipients smtplib.send_message() would actually deliver to.

    Mirrors stdlib: an explicit ``to_addrs`` wins; otherwise the envelope is
    derived from the To/Bcc/Cc headers. This observes the adapter's real call
    shape, so a regression that passes ``to_addrs=[to_addr]`` (dropping cc from
    the wire while keeping a cosmetic Cc header) is caught here.
    """
    call = mock_smtp.return_value.send_message.call_args
    msg = call.args[0]
    to_addrs = call.args[2] if len(call.args) >= 3 else call.kwargs.get("to_addrs")
    if to_addrs is None:
        fields = [f for f in (msg["To"], msg["Bcc"], msg["Cc"]) if f is not None]
        to_addrs = [addr for _, addr in email.utils.getaddresses(fields)]
    return to_addrs


class TestSendMetadataSubject(unittest.TestCase):
    def test_metadata_subject_used_verbatim(self):
        """A metadata subject is used exactly as given — no 'Re:' prefix."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"subject": "[TDG-a1] Access request"},
            ))
            self.assertEqual(
                _sent_message(mock_smtp)["Subject"],
                "[TDG-a1] Access request",
            )

    def test_no_metadata_subject_falls_back_to_re_prefix(self):
        """Without a metadata subject, the thread-context 'Re:' fallback holds."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question"}
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "Body."))
            self.assertEqual(
                _sent_message(mock_smtp)["Subject"], "Re: Project question"
            )

    def test_empty_metadata_subject_falls_back_to_re_prefix(self):
        """A present-but-empty subject is treated as absent (a blank subject is
        never a useful override) → thread-context 'Re:' fallback."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question"}
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"subject": ""},
            ))
            self.assertEqual(
                _sent_message(mock_smtp)["Subject"], "Re: Project question"
            )


class TestSendMetadataCc(unittest.TestCase):
    def test_cc_list_sets_cc_header(self):
        """A metadata cc list becomes a comma-joined Cc header."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["team@test.com", "lead@test.com"]},
            ))
            self.assertEqual(
                _sent_message(mock_smtp)["Cc"], "team@test.com, lead@test.com"
            )

    def test_no_cc_header_when_absent(self):
        """No metadata cc → no Cc header at all."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "Body."))
            self.assertIsNone(_sent_message(mock_smtp)["Cc"])

    def test_cc_included_in_envelope_recipients(self):
        """cc lands on the SMTP envelope (RCPT), not just a cosmetic header."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["team@test.com", "lead@test.com"]},
            ))
            rcpts = _envelope_recipients(mock_smtp)
            self.assertIn("user@test.com", rcpts)
            self.assertIn("team@test.com", rcpts)
            self.assertIn("lead@test.com", rcpts)

    def test_cc_bare_string_is_one_recipient_not_char_split(self):
        """A single cc address passed as a string must not fragment per-character."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"cc": "team@test.com"},
            ))
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "team@test.com")
            self.assertIn("team@test.com", _envelope_recipients(mock_smtp))

    def test_cc_blank_and_non_string_entries_dropped(self):
        """Empty/whitespace/non-str cc entries are skipped — no spurious empty
        recipient, no crash."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["team@test.com", "", "  ", None]},
            ))
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "team@test.com")


class TestSendMetadataFresh(unittest.TestCase):
    def _seed_thread(self, adapter):
        adapter._thread_context["user@test.com"] = {
            "subject": "Old thread",
            "message_id": "<original@test.com>",
        }

    def test_fresh_suppresses_threading_headers(self):
        """fresh=True → no In-Reply-To/References even with a live thread context."""
        adapter = _make_adapter()
        self._seed_thread(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"subject": "[TDG-a1] Fresh outbound", "fresh": True},
            ))
            msg = _sent_message(mock_smtp)
            self.assertIsNone(msg["In-Reply-To"])
            self.assertIsNone(msg["References"])

    def test_not_fresh_still_threads_from_context(self):
        """Without fresh, a thread-context message_id still threads the reply."""
        adapter = _make_adapter()
        self._seed_thread(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send("user@test.com", "Body."))
            msg = _sent_message(mock_smtp)
            self.assertEqual(msg["In-Reply-To"], "<original@test.com>")
            self.assertEqual(msg["References"], "<original@test.com>")

    def test_fresh_non_bool_truthy_also_suppresses_threading(self):
        """Spec says 'fresh truthy' — a non-bool truthy value (e.g. 1) suppresses
        threading too, guarding against a future narrowing to `is True`."""
        adapter = _make_adapter()
        self._seed_thread(adapter)
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"fresh": 1},
            ))
            msg = _sent_message(mock_smtp)
            self.assertIsNone(msg["In-Reply-To"])
            self.assertIsNone(msg["References"])


class TestSendBackwardCompat(unittest.TestCase):
    def test_no_metadata_composition_unchanged(self):
        """No metadata + reply_to → Re: fallback subject, threaded, no Cc — the
        exact composition callers get today."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Weekly sync"}
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", reply_to="<inbound@test.com>",
            ))
            msg = _sent_message(mock_smtp)
            self.assertEqual(msg["To"], "user@test.com")
            self.assertEqual(msg["Subject"], "Re: Weekly sync")
            self.assertEqual(msg["In-Reply-To"], "<inbound@test.com>")
            self.assertEqual(msg["References"], "<inbound@test.com>")
            self.assertIsNone(msg["Cc"])


class TestSendMetadataHardening(unittest.TestCase):
    """Malformed or hostile metadata degrades gracefully — the send composes a
    valid message (or falls back) instead of crashing or injecting a header."""

    def test_non_string_subject_falls_back_to_re_prefix(self):
        """A non-string truthy subject (int/bool/list/dict) is ignored, not
        assigned to the header — which would crash at SMTP serialization time,
        after login, surfacing as a cryptic failed send indistinguishable from a
        transport error. The thread-context 'Re:' fallback holds instead."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question"}
        for bad in (123, True, ["x"], {"a": 1}):
            with patch("smtplib.SMTP") as mock_smtp:
                mock_smtp.return_value = MagicMock()
                result = asyncio.run(adapter.send(
                    "user@test.com", "Body.", metadata={"subject": bad},
                ))
                self.assertTrue(result.success)
                self.assertEqual(
                    _sent_message(mock_smtp)["Subject"], "Re: Project question"
                )

    def test_whitespace_only_subject_falls_back_to_re_prefix(self):
        """A whitespace-only subject is never a useful override → Re: fallback
        (pins the intent stated by test_empty_metadata_subject's docstring)."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question"}
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"subject": "   "},
            ))
            self.assertEqual(
                _sent_message(mock_smtp)["Subject"], "Re: Project question"
            )

    def test_crlf_subject_cannot_inject_headers(self):
        """A CR/LF-laced subject is flattened to a single line in the adapter's
        own code — so no smuggled header (e.g. Bcc) appears — rather than
        relying on stdlib to reject it at send time."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"subject": "Legit\r\nBcc: evil@evil.com"},
            ))
            msg = _sent_message(mock_smtp)
            self.assertNotIn("\n", msg["Subject"])
            self.assertNotIn("\r", msg["Subject"])
            # The real wire form: serialize and confirm no injected header line.
            self.assertNotIn("\nBcc:", msg.as_string())
            self.assertIsNone(msg["Bcc"])

    def test_non_iterable_cc_is_ignored_no_crash(self):
        """A non-iterable cc (int) is ignored, not iterated (which would raise
        TypeError and mislabel as a send failure); the send composes with no
        Cc header."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"cc": 123},
            ))
            self.assertTrue(result.success)
            self.assertIsNone(_sent_message(mock_smtp)["Cc"])

    def test_non_dict_metadata_is_ignored_no_crash(self):
        """A non-dict metadata (contract violation) degrades to no override
        rather than crashing on .get()."""
        adapter = _make_adapter()
        adapter._thread_context["user@test.com"] = {"subject": "Project question"}
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata=["not", "a", "dict"],
            ))
            self.assertTrue(result.success)
            self.assertEqual(
                _sent_message(mock_smtp)["Subject"], "Re: Project question"
            )

    def test_cc_equal_to_recipient_dropped_from_envelope(self):
        """A cc entry equal to the To address is dropped so smtplib does not
        issue a duplicate RCPT TO for the same recipient."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["user@test.com", "team@test.com"]},
            ))
            rcpts = _envelope_recipients(mock_smtp)
            self.assertEqual(rcpts.count("user@test.com"), 1)
            self.assertIn("team@test.com", rcpts)

    def test_duplicate_cc_entries_collapsed(self):
        """Repeated cc addresses collapse to a single Cc header / RCPT entry."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["dup@test.com", "dup@test.com"]},
            ))
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "dup@test.com")
            self.assertEqual(
                _envelope_recipients(mock_smtp).count("dup@test.com"), 1
            )

    def test_cc_entry_with_crlf_dropped_others_kept(self):
        """A newline-bearing cc entry is dropped (can't inject a header) while
        valid entries still send."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={
                    "cc": ["good@test.com", "evil@test.com\r\nBcc: z@test.com"],
                },
            ))
            msg = _sent_message(mock_smtp)
            self.assertEqual(msg["Cc"], "good@test.com")
            # The real wire form: no injected header, no smuggled recipient.
            wire = msg.as_string()
            self.assertNotIn("\nBcc:", wire)
            self.assertNotIn("z@test.com", wire)
            self.assertIsNone(msg["Bcc"])

    def test_cc_comma_joined_string_split_and_deduped(self):
        """A comma-joined cc string is split into individual addresses (matching
        the envelope smtplib derives), and an address equal to the primary
        recipient is dropped so it doesn't get a duplicate RCPT TO."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": "team@test.com, user@test.com"},
            ))
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "team@test.com")
            rcpts = _envelope_recipients(mock_smtp)
            self.assertEqual(rcpts.count("user@test.com"), 1)
            self.assertIn("team@test.com", rcpts)

    def test_cc_case_insensitive_dedup_vs_recipient(self):
        """A cc address that is the same mailbox as the recipient but differs
        only in letter case is dropped — real providers treat it as one."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com", "Body.", metadata={"cc": "User@Test.com"},
            ))
            self.assertIsNone(_sent_message(mock_smtp)["Cc"])
            self.assertEqual(
                _envelope_recipients(mock_smtp).count("user@test.com"), 1
            )

    def test_cc_tuple_accepted(self):
        """A tuple cc value is accepted the same as a list."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ("team@test.com", "lead@test.com")},
            ))
            self.assertEqual(
                _sent_message(mock_smtp)["Cc"], "team@test.com, lead@test.com"
            )

    def test_cc_display_name_wrapping_recipient_deduped(self):
        """A cc entry that wraps the recipient's own address in a display name
        ("Name <user@...>") is deduped on the parsed address, so the recipient
        does not get a second RCPT TO — matching how smtplib derives the
        envelope from the Cc header."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["User Name <user@test.com>", "team@test.com"]},
            ))
            rcpts = _envelope_recipients(mock_smtp)
            self.assertEqual(rcpts.count("user@test.com"), 1)
            self.assertIn("team@test.com", rcpts)

    def test_cc_quoted_comma_display_name_not_shredded(self):
        """Two distinct recipients whose display names contain commas
        ("Last, First <addr>") are both delivered — the quoted comma is not
        mistaken for an address separator that would drop or corrupt one."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "primary@test.com",
                "Body.",
                metadata={"cc": [
                    '"Smith, John" <john@test.com>',
                    '"Smith, Jane" <jane@test.com>',
                ]},
            ))
            rcpts = _envelope_recipients(mock_smtp)
            self.assertIn("john@test.com", rcpts)
            self.assertIn("jane@test.com", rcpts)

    def test_cc_semicolon_list_does_not_void_primary_recipient(self):
        """A semicolon-separated cc string (caller misuse) is dropped rather
        than corrupting the Cc header and voiding delivery to the primary
        recipient."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": "a@test.com; b@test.com"},
            ))
            self.assertTrue(result.success)
            self.assertIn("user@test.com", _envelope_recipients(mock_smtp))

    def test_cc_non_ascii_address_skipped_message_still_sends(self):
        """A non-ASCII (EAI) cc address the adapter can't deliver is skipped —
        it must not crash the whole send and lose the other recipients."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            result = asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["用户@例え.jp", "good@test.com"]},
            ))
            self.assertTrue(result.success)
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "good@test.com")

    def test_cc_unparseable_token_dropped(self):
        """A cc field getaddresses can't resolve into a real address (no "@")
        is dropped rather than becoming a doomed RCPT target."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "user@test.com",
                "Body.",
                metadata={"cc": ["not-an-address", "good@test.com"]},
            ))
            self.assertEqual(_sent_message(mock_smtp)["Cc"], "good@test.com")

    def test_cc_deduped_against_display_name_recipient(self):
        """When the recipient (To) is a display-name form, a bare cc for the
        same mailbox is still deduped — no duplicate RCPT TO."""
        adapter = _make_adapter()
        with patch("smtplib.SMTP") as mock_smtp:
            mock_smtp.return_value = MagicMock()
            asyncio.run(adapter.send(
                "User Name <user@test.com>",
                "Body.",
                metadata={"cc": ["user@test.com", "team@test.com"]},
            ))
            rcpts = _envelope_recipients(mock_smtp)
            self.assertEqual(rcpts.count("user@test.com"), 1)
            self.assertIn("team@test.com", rcpts)


if __name__ == "__main__":
    unittest.main()
