"""The email adapter's sender identity is exactly one addr-spec.

The identity parsed from From: becomes the session's chat_id, the allowlist key and the reply To:, while sender
authentication only proves the domain.
"""

import unittest

from plugins.platforms.email.adapter import _extract_email_address


class TestSenderIdentityParsing(unittest.TestCase):

    def test_bracketed_from_with_more_than_one_address_is_rejected(self):
        # Each of these reached the single-bracket fallback and came back as a multi-recipient "address".
        for raw in ("Doe <victim@example.com,attacker@evil.test>",
                    "Doe <victim@example.com;attacker@evil.test>",
                    "Doe <victim@example.com@evil.test>"):
            self.assertEqual(_extract_email_address(raw), "", raw)
        # The fallback still serves the unquoted display names it exists for.
        self.assertEqual(_extract_email_address("Doe, John <John@Example.com>"), "john@example.com")


if __name__ == "__main__":
    unittest.main()
