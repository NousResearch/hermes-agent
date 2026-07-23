"""Tests for gateway E.164 phone-number preservation (issue #68911).

The gateway's ``_redact_gateway_user_facing_secrets`` forcibly redacts all
secret-like patterns including E.164 phone numbers.  With
``security.preserve_e164_chat_responses`` enabled, standalone E.164 numbers
should survive while API keys and bearer tokens remain masked.

Tests import the function directly without triggering gateway.run's module-level
side effects (dotenv, etc.) by importing via the module's __dict__ after
selective patching.
"""

import os
import re
import unittest
from unittest.mock import patch


class TestGatewayRedactE164(unittest.TestCase):
    """Functional tests for E.164 preservation through the gateway redactor."""

    @classmethod
    def setUpClass(cls):
        # Build reproducible synthetic test values (no real credentials).
        cls.phone = "+" + "".join(str((i % 9) + 1) for i in range(10))
        cls.token = "sk-" + "".join(chr(65 + i % 26) for i in range(48))
        cls.sample = f"Client ({cls.phone}); token {cls.token}"

    def _call_target(self, text: str) -> str:
        """Call _redact_gateway_user_facing_secrets with the opt-in env.

        We must import *inside* the test so that the env-var snapshot in
        the function body (``os.environ.get(...)``) reflects our patch.
        """
        from gateway.run import _redact_gateway_user_facing_secrets
        return _redact_gateway_user_facing_secrets(text)

    # --- Opt-out (default) behaviour ---

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": ""}, clear=False)
    def test_default_redacts_e164(self):
        """Without preserve_e164, E.164 phone numbers are masked."""
        result = self._call_target(self.sample)
        self.assertNotIn(self.phone, result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": ""}, clear=False)
    def test_default_still_redacts_token(self):
        """Without preserve_e164, synthetic sk- tokens are still masked."""
        result = self._call_target(self.sample)
        self.assertNotIn(self.token, result)

    # --- Opt-in behaviour ---

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_preserve_e164_opt_in_preserves_phone(self):
        """With preserve_e164, standalone E.164 numbers survive redaction."""
        result = self._call_target(self.sample)
        self.assertIn(self.phone, result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_preserve_e164_opt_in_still_redacts_token(self):
        """With preserve_e164, synthetic sk- tokens are still masked."""
        result = self._call_target(self.sample)
        self.assertNotIn(self.token, result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_preserve_e164_opt_in_still_redacts_bearer(self):
        """With preserve_e164, Bearer tokens are still masked."""
        sample_with_bearer = f"Phone: {self.phone}, Auth: Bearer {self.token}"
        result = self._call_target(sample_with_bearer)
        self.assertIn(self.phone, result)
        self.assertNotIn(self.token, result)

    # --- Edge cases ---

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_preserve_non_e164_not_affected(self):
        """Strings that are not standalone E.164 numbers are not affected."""
        sample = "No phone here, just a token sk-test1234567890"
        result = self._call_target(sample)
        self.assertNotIn("sk-test1234567890", result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_e164_and_token_with_opt_in(self):
        """With preserve_e164, E.164 survives and token is redacted."""
        text = "Call +15551234567 for support; key sk-test1234567890123456"
        result = self._call_target(text)
        self.assertIn("+15551234567", result)
        self.assertNotIn("sk-test1234567890123456", result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_multiple_e164_numbers_preserved(self):
        """Multiple standalone E.164 numbers all survive with opt-in."""
        sample = "Alice: +12025551234, Bob: +447700900123"
        result = self._call_target(sample)
        self.assertIn("+12025551234", result)
        self.assertIn("+447700900123", result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_short_e164_preserved(self):
        """Short E.164 numbers (7 digits) are preserved with opt-in."""
        sample = "Call +1234567 now"
        result = self._call_target(sample)
        self.assertIn("+1234567", result)

    @patch.dict(os.environ, {"HERMES_PRESERVE_E164_CHAT_RESPONSES": "true"}, clear=False)
    def test_long_e164_preserved(self):
        """Long E.164 numbers (15 digits) are preserved with opt-in."""
        sample = "Call +123456789012345 now"
        result = self._call_target(sample)
        self.assertIn("+123456789012345", result)


if __name__ == "__main__":
    unittest.main()
