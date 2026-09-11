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

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

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


    def test_forces_redaction_even_when_disabled(self, monkeypatch):
        """force=True must redact even if security.redact_secrets is off -- the
        approval prompt is a hard secret-egress boundary regardless of config."""
        raw = "curl -H 'Authorization: token " + _FAKE_GHP + "' https://api.github.com"
        # With redaction globally disabled, the seam must STILL redact (force=True).
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False, raising=False)
        out = _redact_approval_command(raw)
        assert _FAKE_GHP not in out


class TestApprovalCommandWiring:
    """Guard the production wiring on BOTH approval-notify transports:
    1. the chat-platform path (_approval_notify_sync in gateway/run.py), and
    2. the SSE/API path (_approval_notify in
       gateway/platforms/api_server_runs.py).

    The legacy chat assertion inspects its AST to require a reassigned redacted
    value before send. The SSE assertion executes the delivery path so transport
    refactors cannot invalidate the test while redaction remains effective.
    """

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
        import gateway.run_turn_runner as run

        self._assert_redacts_then_uses(run, "_approval_notify_sync", "send_exec_approval")

    @pytest.mark.asyncio
    async def test_sse_api_path_redacts_before_enqueue(self):
        from gateway.platforms.api_server_runs import (
            _make_approval_notify,
            _publish_run_event,
        )

        run_id = "run_redaction"
        pending = asyncio.Queue()
        subscriber = asyncio.Queue()
        adapter = SimpleNamespace(
            _run_streams={run_id: pending},
            _run_stream_subscribers={run_id: {subscriber}},
            _set_run_status=MagicMock(),
        )
        run = SimpleNamespace(
            run_id=run_id,
            put_event=lambda event: _publish_run_event(
                adapter, run_id, event, expected_queue=pending
            ),
        )
        api_server = SimpleNamespace(
            _approval_event_choices=lambda **_: ["once", "deny"]
        )
        notify = _make_approval_notify(adapter, run, _api_server=api_server)

        notify({"command": "curl -H 'Authorization: *** " + _FAKE_GHP + "' github.com"})
        await asyncio.sleep(0)

        event = subscriber.get_nowait()
        assert _FAKE_GHP not in event["command"]
        assert "curl" in event["command"]
        assert event["event"] == "approval.request"
        assert pending.empty()


class TestApprovalTextFallbackContract:
    def test_smart_deny_only_advertises_one_operation(self):
        from gateway.run import _format_exec_approval_fallback

        text = _format_exec_approval_fallback(
            "rm -rf /", "dangerous deletion", "/",
            allow_permanent=False, smart_denied=True,
        )
        assert "owner override" in text.lower()
        assert "one operation" in text.lower()
        assert "`/approve`" in text
        assert "approve session" not in text
        assert "approve always" not in text

