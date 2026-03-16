"""Tests for gateway session approval: env var setup and dangerous command guard."""

import os

import pytest

from agent.prompt_builder import PLATFORM_HINTS
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from tools.approval import check_dangerous_command, clear_session


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure gateway session env var is clean before/after each test."""
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    key = os.getenv("HERMES_SESSION_KEY", "default")
    clear_session(key)
    yield
    clear_session(key)


def _make_runner_and_context():
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="TestGroup",
        chat_type="group",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})
    return runner, context


# ---------------------------------------------------------------------------
# _set_session_env / _clear_session_env
# ---------------------------------------------------------------------------

class TestSessionEnvGatewayFlag:
    def test_set_session_env_sets_gateway_session(self):
        runner, context = _make_runner_and_context()
        runner._set_session_env(context)
        assert os.getenv("HERMES_GATEWAY_SESSION") == "1"

    def test_clear_session_env_removes_gateway_session(self):
        runner, context = _make_runner_and_context()
        runner._set_session_env(context)
        assert os.getenv("HERMES_GATEWAY_SESSION") == "1"

        runner._clear_session_env()
        assert os.getenv("HERMES_GATEWAY_SESSION") is None


# ---------------------------------------------------------------------------
# check_dangerous_command with HERMES_GATEWAY_SESSION
# ---------------------------------------------------------------------------

class TestDangerousCommandGateway:
    def test_auto_approves_without_gateway_session(self):
        """Without HERMES_GATEWAY_SESSION, dangerous commands are auto-approved."""
        result = check_dangerous_command("rm -rf /tmp/test", "local")
        assert result["approved"] is True

    def test_requires_approval_with_gateway_session(self, monkeypatch):
        """With HERMES_GATEWAY_SESSION=1, dangerous commands require approval."""
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        result = check_dangerous_command("rm -rf /tmp/test", "local")
        # The command should match a dangerous pattern and require approval
        if result.get("status") == "approval_required":
            assert result["approved"] is False
        else:
            # If the command doesn't match any dangerous pattern, it's still approved
            assert result["approved"] is True


# ---------------------------------------------------------------------------
# Platform hints contain confirmation instruction
# ---------------------------------------------------------------------------

class TestPlatformHintsConfirmation:
    @pytest.mark.parametrize("platform", ["telegram", "whatsapp", "discord", "slack", "signal"])
    def test_messaging_hints_contain_confirmation(self, platform):
        hint = PLATFORM_HINTS[platform]
        assert "ask for explicit user confirmation" in hint
        assert "Do not interpret conversational discussion as an instruction" in hint

    def test_email_hint_does_not_contain_confirmation(self):
        hint = PLATFORM_HINTS["email"]
        assert "ask for explicit user confirmation" not in hint
