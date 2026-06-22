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

    def test_clean_command_passes_through_unchanged(self):
        raw = "ls -la /tmp && echo hello"
        assert _redact_approval_command(raw) == raw

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

    def _assert_redacts_then_uses(self, module, func_name: str, sink_substr: str):
        """Parse `module`'s full AST, locate the (possibly nested) function
        `func_name`, and assert it contains an assignment
        `<x> = _redact_approval_command(...)` whose result is then used by a
        statement matching `sink_substr` on a LATER line. Walking the real AST
        (not a source slice) is refactor-robust and rejects discarded-result
        calls (the call must be an assignment, not a bare expression)."""
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
                if isinstance(fn, ast.Name) and fn.id == "_redact_approval_command":
                    redact_line = node.lineno
        assert redact_line is not None, (
            f"{func_name} must assign the result of _redact_approval_command(...) "
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

        self._assert_redacts_then_uses(api_server, "_approval_notify", "put_nowait")
