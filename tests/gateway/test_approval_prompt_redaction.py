"""Regression test for approval prompt credential redaction (issue #48456).

When Tirith flags a command for containing a credential-shaped pattern, the
gateway approval prompt must redact the credential from the command text
before sending it to the chat platform. Without this fix, the raw command
(with the credential in plaintext) is sent verbatim to Telegram/Discord/etc.,
undoing Tirith's redaction one layer up.

The redaction is wired through the module-level ``_redact_approval_command``
seam. These tests bind that seam -- the production wiring -- not just the
underlying ``redact_sensitive_text`` helper, so they fail if the redaction
call is removed from either approval path.

Credential fixtures are built at runtime from a benign prefix + a run of
``X`` characters (the same trick tests/agent/test_redact.py uses): they match
the redactor regexes so the assertions stay meaningful, but contain no real
or real-looking key, so secret scanners do not flag this file.
"""

from gateway.run import _redact_approval_command

# Synthetic, scanner-safe credential fixtures. Each matches its redactor
# regex (ghp_/sk-/JWT) but is unmistakably fake -- a run of X's, never a
# real or real-format key.
_FAKE_GHP = "ghp_" + "X" * 36
_FAKE_OPENAI = "sk-proj-" + "X" * 40
_FAKE_JWT = "eyJ" + "X" * 20 + "." + "eyJ" + "X" * 24 + "." + "X" * 30


class TestRedactApprovalCommand:
    """Contract for the approval-prompt redaction seam used by the gateway."""

    def test_redacts_github_pat(self):
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com/user"
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out
        # command structure preserved so the operator can still judge the action
        assert "curl" in out
        assert "github.com" in out

    def test_redacts_openai_key(self):
        raw = "export OPENAI_API_KEY=" + _FAKE_OPENAI + " && python s.py"
        out = _redact_approval_command(raw)
        assert _FAKE_OPENAI not in out
        assert "python s.py" in out

    def test_redacts_bearer_token(self):
        raw = "curl -H 'Authorization: Bearer " + _FAKE_JWT + "' https://api.example.com"
        out = _redact_approval_command(raw)
        assert _FAKE_JWT not in out


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text

    def test_text_fallback_says_silence_means_no(self, monkeypatch):
        """Surfaces without buttons get the same deadline line as the button card."""
        from gateway.run import _format_exec_approval_fallback

        monkeypatch.setattr("gateway.platforms.base_exec_approval.approval_timeout_seconds", lambda: 300)
        text = _format_exec_approval_fallback("rm -rf /", "recursive delete", "/")
        assert "recursive delete" in text
        assert "5 minutes" in text
        for step in ("`/approve`", "`/approve session`", "`/approve always`", "`/deny`"):
            assert step in text

