"""Tests for scrub.py — credential and PII scrubbing."""

import unittest
from unittest.mock import patch


class TestScrubContent(unittest.TestCase):
    """Tests for scrub_content with PII pattern matching."""

    def test_scrub_ssn_dashes(self) -> None:
        """SSN with dashes is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("SSN is 123-45-6789", allow_no_redact=True)
        self.assertEqual(result, "SSN is [SSN]")

    def test_scrub_ssn_spaces(self) -> None:
        """SSN with spaces is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("SSN is 123 45 6789", allow_no_redact=True)
        self.assertEqual(result, "SSN is [SSN]")

    def test_scrub_phone(self) -> None:
        """US phone number with parens is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("Call (555) 123-4567", allow_no_redact=True)
        self.assertEqual(result, "Call [PHONE]")

    def test_scrub_phone_intl(self) -> None:
        """International phone number with +1 prefix is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("Call +1 555-123-4567", allow_no_redact=True)
        self.assertEqual(result, "Call [PHONE]")

    def test_scrub_credit_card(self) -> None:
        """Credit card with spaces is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("Card: 4111 1111 1111 1111", allow_no_redact=True)
        self.assertEqual(result, "Card: [CC]")

    def test_scrub_credit_card_dashes(self) -> None:
        """Credit card with dashes is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("Card: 4111-1111-1111-1111", allow_no_redact=True)
        self.assertEqual(result, "Card: [CC]")

    def test_scrub_iban(self) -> None:
        """IBAN is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("IBAN: GB29 0000 0000 0000 0000 00", allow_no_redact=True)
        self.assertEqual(result, "IBAN: [IBAN]")

    def test_scrub_dob_born_context(self) -> None:
        """Date of birth with 'born on' context is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("born on 1990-01-15", allow_no_redact=True)
        self.assertEqual(result, "[DOB]")

    def test_scrub_dob_label_context(self) -> None:
        """Date of birth with 'DOB:' label is scrubbed."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("DOB: 1990/01/15", allow_no_redact=True)
        self.assertEqual(result, "[DOB]")

    def test_scrub_plain_date_not_scrubbed(self) -> None:
        """Plain date without PII context is NOT scrubbed as DOB."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("meeting on 2026-04-07", allow_no_redact=True)
        self.assertEqual(result, "meeting on 2026-04-07")

    def test_scrub_no_false_positive(self) -> None:
        """Plain year (2024) is NOT scrubbed as a date."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("The year 2024 was great", allow_no_redact=True)
        self.assertEqual(result, "The year 2024 was great")

    def test_scrub_plain_digits_not_phone(self) -> None:
        """10-digit number without separators is NOT scrubbed as a phone number."""
        from plugins.memory.tiered.scrub import scrub_content

        result = scrub_content("ID: 5551234567", allow_no_redact=True)
        self.assertEqual(result, "ID: 5551234567")

    def test_scrub_multiple_patterns(self) -> None:
        """Text with SSN, phone, and CC all scrubbed in one pass."""
        from plugins.memory.tiered.scrub import scrub_content

        text = "SSN 123-45-6789, phone (555) 123-4567, card 4111 1111 1111 1111"
        result = scrub_content(text, allow_no_redact=True)
        self.assertIn("[SSN]", result)
        self.assertIn("[PHONE]", result)
        self.assertIn("[CC]", result)
        self.assertNotIn("123-45-6789", result)
        self.assertNotIn("4111", result)

    def test_scrub_fail_closed(self) -> None:
        """Without agent.redact and allow_no_redact=False, raises RuntimeError."""
        with patch("plugins.memory.tiered.scrub._HAS_REDACT", False):
            from plugins.memory.tiered.scrub import scrub_content

            with self.assertRaises(RuntimeError) as ctx:
                scrub_content("some text", allow_no_redact=False)
            self.assertIn("refusing to store unscrubbed data", str(ctx.exception))

    def test_scrub_allows_dev_mode(self) -> None:
        """With allow_no_redact=True, works even without agent.redact."""
        with patch("plugins.memory.tiered.scrub._HAS_REDACT", False):
            from plugins.memory.tiered.scrub import scrub_content

            result = scrub_content("SSN is 123-45-6789", allow_no_redact=True)
            self.assertEqual(result, "SSN is [SSN]")


if __name__ == "__main__":
    unittest.main()
