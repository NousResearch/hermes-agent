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


class TestApprovalCommandWiring:
    """The chat-platform approval notify (TurnRunner._approval_notify_sync) must send the REDACTED
    command on both the card and the text fallback, without mutating the approval payload."""

    def test_chat_platform_path_redacts_before_send(self):
        import asyncio
        from types import SimpleNamespace
        from gateway.run_turn_runner import TurnRunner

        async def exercise():
            for card_success in (True, False):
                sent = []

                class Adapter:
                    def pause_typing_for_chat(self, chat_id):
                        pass

                    async def send_exec_approval(self, **kwargs):
                        sent.append(kwargs['command'])
                        return SimpleNamespace(success=card_success, error=None)

                    async def send(self, chat_id, text, **kwargs):
                        sent.append(text)
                        return SimpleNamespace(success=True)

                ctx = SimpleNamespace(_status_adapter=Adapter(), _status_chat_id='fixture',
                    _status_thread_metadata=None, session_key='redaction-fixture',
                    _loop_for_step=asyncio.get_running_loop(), stream_consumer_holder=[None])
                turn = TurnRunner(SimpleNamespace(), ctx)
                raw = 'curl -H "Authorization: token ' + _FAKE_GHP + '" https://example.test'
                approval = {'command': raw, 'description': 'fixture'}
                await asyncio.to_thread(turn._approval_notify_sync, approval)
                assert len(sent) == (1 if card_success else 2)
                assert all(_FAKE_GHP not in text and 'curl' in text for text in sent)
                assert approval['command'] == raw

        asyncio.run(exercise())


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

