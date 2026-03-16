"""Tests for gateway approval: dangerous command guard and platform hints."""

import os

import pytest

from agent.prompt_builder import PLATFORM_HINTS
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


# ---------------------------------------------------------------------------
# check_dangerous_command with HERMES_GATEWAY_SESSION
# ---------------------------------------------------------------------------

class TestDangerousCommandGateway:
    def test_auto_approves_without_gateway_session(self):
        """Without HERMES_GATEWAY_SESSION, dangerous commands are auto-approved
        via the early return at lines 319-320 of approval.py."""
        result = check_dangerous_command("rm -rf /tmp/test", "local")
        assert result["approved"] is True
        # Confirm this is the early-return auto-approve path (no status key),
        # not the non-dangerous path.
        assert "status" not in result

    def test_requires_approval_with_gateway_session(self, monkeypatch):
        """With HERMES_GATEWAY_SESSION=1, dangerous commands require approval."""
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        result = check_dangerous_command("rm -rf /tmp/test", "local")
        assert result["approved"] is False
        assert result["status"] == "approval_required"


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
