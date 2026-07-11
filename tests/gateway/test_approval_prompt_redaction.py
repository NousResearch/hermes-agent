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

import pytest

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

    def test_clean_command_passes_through_unchanged(self):
        raw = "ls -la /tmp && echo hello"
        assert _redact_approval_command(raw) == raw

    def test_redacts_web_url_userinfo_and_all_query_values(self):
        sentinel = "opaque" + "X" * 24
        raw = (
            f"curl 'https://user:{sentinel}@example.com/run"
            f"?token={sentinel}&label=public'"
        )

        out = _redact_approval_command(raw)

        assert sentinel not in out
        assert "example.com/run" in out
        assert "token=[REDACTED]" in out
        assert "label=[REDACTED]" in out

    def test_redacts_secret_bearing_cli_flag_values(self):
        sentinel = "opaque" + "Y" * 24
        raw = (
            f"deploy --password {sentinel} --api-key={sentinel} "
            f"--access-token '{sentinel}'"
        )

        out = _redact_approval_command(raw)

        assert sentinel not in out
        assert "--password [REDACTED]" in out
        assert "--api-key=[REDACTED]" in out
        assert "--access-token [REDACTED]" in out

    @pytest.mark.parametrize(
        "command",
        [
            "curl -u user:{sentinel} https://example.com",
            "curl --user user:{sentinel} https://example.com",
            "wget --http-password={sentinel} https://example.com",
            "deploy --secret-key={sentinel}",
            "aws configure set aws_secret_access_key {sentinel}",
            "curl https://example.com/callback?{sentinel}",
        ],
    )
    def test_redacts_additional_credential_forms(self, command):
        sentinel = "opaque" + "Q" * 24

        out = _redact_approval_command(command.format(sentinel=sentinel))

        assert sentinel not in out

    def test_bounds_approval_preview(self):
        assert len(_redact_approval_command("x" * 5000)) == 4096

    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out

    def test_handles_none_and_empty(self):
        assert _redact_approval_command("") == ""
        assert _redact_approval_command(None) == ""


class TestApprovalCommandWiring:
    """Guard the production wiring on BOTH approval-notify transports:
    1. the chat-platform path (_approval_notify_sync in gateway/run.py), and
    2. the SSE/API path (_approval_notify in gateway/platforms/api_server.py),
    each of which must route the command through _redact_approval_command and
    REASSIGN the redacted value before any send/enqueue (so the raw command
    cannot reach a client). Uses AST (not char-offset string slicing) so a
    benign refactor doesn't cause a false failure, and so a discarded-result
    call (`_redact(cmd); send(cmd)`) does NOT pass."""

    def _assert_redacts_then_uses(
        self,
        module,
        func_name: str,
        sink_substr: str,
        helper_name: str = "_redact_approval_command",
    ):
        """Assert a helper result is assigned before the named egress sink.

        Walking the real AST (not a source slice) is refactor-robust and rejects
        discarded-result calls: the helper call must be an assignment, not a
        bare expression.
        """
        import ast
        import inspect

        source = inspect.getsource(module)
        tree = ast.parse(source)
        target_fn = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                target_fn = node
                break
        assert target_fn is not None, f"function {func_name} not found in {module.__name__}"

        redact_line = None
        for node in ast.walk(target_fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Name) and fn.id == helper_name:
                    redact_line = node.lineno
        assert redact_line is not None, (
            f"{func_name} must assign the result of {helper_name}(...) "
            "(a discarded-result call would still leak the raw command)"
        )

        sink_line = None
        for node in ast.walk(target_fn):
            seg = ast.get_source_segment(source, node)
            if seg and sink_substr in seg and getattr(node, "lineno", 0) > redact_line:
                sink_line = node.lineno
                break
        assert sink_line is not None, (
            f"`{sink_substr}` sink not found after the redaction in {func_name}"
        )

    def test_chat_platform_path_redacts_before_send(self):
        import gateway.run as run

        self._assert_redacts_then_uses(run, "_approval_notify_sync", "send_exec_approval")

    def test_sse_api_path_redacts_before_enqueue(self):
        from gateway.platforms import api_server

        self._assert_redacts_then_uses(
            api_server,
            "_approval_notify",
            "put_nowait",
            helper_name="_build_run_approval_request_event",
        )
