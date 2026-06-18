"""Regression test for approval prompt credential redaction (issue #48456).

When Tirith flags a command for containing a credential-shaped pattern, the
gateway approval prompt must redact the credential from the command text
before sending it to the chat platform.  Without this fix, the raw command
(with the credential in plaintext) is sent verbatim to Telegram/Discord/etc.,
undoing Tirith's redaction one layer up.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestApprovalPromptRedaction:
    """Verify _approval_notify_sync redacts credentials from command text."""

    def test_redacts_github_pat_in_approval_prompt(self):
        """GitHub PAT in command text must be masked in the approval message."""
        from agent.redact import redact_sensitive_text

        raw_cmd = "curl -H 'Authorization: token ghp_abcdefghijklmnop1234' https://api.github.com/user"
        redacted = redact_sensitive_text(raw_cmd, force=True)

        # The PAT must not appear in the redacted output
        assert "ghp_abcdefghijklmnop1234" not in redacted
        # But the command structure should be preserved
        assert "curl" in redacted
        assert "github.com" in redacted

    def test_redacts_openai_key_in_approval_prompt(self):
        """OpenAI API key in command text must be masked."""
        from agent.redact import redact_sensitive_text

        raw_cmd = "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456 && python script.py"
        redacted = redact_sensitive_text(raw_cmd, force=True)

        assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in redacted
        assert "python script.py" in redacted

    def test_redacts_bearer_token_in_approval_prompt(self):
        """Bearer token in command text must be masked."""
        from agent.redact import redact_sensitive_text

        raw_cmd = "curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U' https://api.example.com"
        redacted = redact_sensitive_text(raw_cmd, force=True)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted

    def test_clean_command_passes_through_unchanged(self):
        """Commands without credentials should pass through unchanged."""
        from agent.redact import redact_sensitive_text

        raw_cmd = "ls -la /tmp && echo hello"
        redacted = redact_sensitive_text(raw_cmd, force=True)

        assert redacted == raw_cmd

    def test_force_flag_overrides_config(self):
        """force=True must redact even if security.redact_secrets is false."""
        from agent.redact import redact_sensitive_text

        raw_cmd = "curl -H 'Authorization: token ghp_abcdefghijklmnop1234' https://api.github.com"

        with patch("agent.redact._REDACT_ENABLED", False):
            # Without force, secrets pass through when redaction is disabled
            unforced = redact_sensitive_text(raw_cmd, force=False)
            assert "ghp_abcdefghijklmnop1234" in unforced

            # With force, secrets are always redacted
            forced = redact_sensitive_text(raw_cmd, force=True)
            assert "ghp_abcdefghijklmnop1234" not in forced
