"""Tests for the system-message guard (fix for #29871).

When a provider's prepare_messages() hooks silently strip the role="system"
message, _build_kwargs_from_profile detects this and re-injects from the
original input so SOUL.md is never lost mid-flight.
"""

import pytest
from types import SimpleNamespace

from agent.transports.chat_completions import ChatCompletionsTransport


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


class TestSystemMessageGuard:
    """Verify the guard detects stripped system role and re-injects it."""

    def _build_mock_profile(self, strip_system=False):
        """Create a mock ProviderProfile where prepare_messages optionally strips system role."""

        class MockProfile:
            name = "test-ollama-cloud"
            fixed_temperature = None
            default_max_tokens = None
            supports_reasoning = False

            def __init__(self, strip_system=False):
                self._strip_system = strip_system

            def prepare_messages(self, msgs):
                if not msgs:
                    return msgs
                result = list(msgs)
                # Mimic provider hooks that strip system role from outgoing payload
                if self._strip_system and len(result) > 0 and isinstance(result[0], dict) and result[0].get("role") == "system":
                    result.pop(0)
                return result

            def build_api_kwargs_extras(self, **kwargs):
                return {}, {}

            def build_extra_body(self, **kwargs):
                return {}

        return MockProfile(strip_system=strip_system)

    def test_guard_no_strip_keeps_messages_intact(self, transport):
        """When prepare_messages does NOT strip system, guard should be a no-op."""
        profile = self._build_mock_profile(strip_system=False)
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=msgs,
            tools=None,
            params={"messages": msgs},
        )
        # System message should still be first
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"

    def test_guard_reinjected_when_stripped(self, transport):
        """When prepare_messages strips system role, guard must re-inject it."""
        profile = self._build_mock_profile(strip_system=True)
        soul_content = "# TestPersona\\nYou are TestPersona, marker: ZX7Q-MARKER-7L9K"
        msgs = [
            {"role": "system", "content": soul_content},
            {"role": "user", "content": "Hi"},
        ]
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=[{"role": "user", "content": "Hi"}],  # simulate what prepare_messages returned (stripped)
            tools=None,
            params={"messages": msgs},  # original input with system
        )
        # Guard should have re-injected the system message
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"
        assert soul_content in kwargs["messages"][0]["content"]

    def test_guard_no_action_when_no_system_in_input(self, transport):
        """When input has no system message, guard should not modify output."""
        profile = self._build_mock_profile(strip_system=False)
        msgs = [
            {"role": "user", "content": "Hi"},
        ]
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=msgs,
            tools=None,
            params={"messages": msgs},
        )
        # Should pass through unchanged — no system to guard
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_guard_no_action_when_system_also_stripped_but_not_in_input(self, transport):
        """Edge case: sanitized has no system (expected) and params also has no system — guard silent."""
        profile = self._build_mock_profile(strip_system=False)
        msgs = [{"role": "user", "content": "Hi"}]
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=msgs,
            tools=None,
            params={"messages": msgs},  # no system in params either
        )
        assert len(kwargs["messages"]) == 1
        assert kwargs["messages"][0]["role"] == "user"

    def test_guard_handles_empty_messages(self, transport):
        """Guard should handle empty messages gracefully."""
        profile = self._build_mock_profile(strip_system=False)
        msgs = []
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=msgs,
            tools=None,
            params={"messages": msgs},
        )
        assert len(kwargs["messages"]) == 0

    def test_guard_handles_system_at_non_first_position(self, transport):
        """Guard only checks first message. If system is elsewhere (non-standard), no re-injection."""
        profile = self._build_mock_profile(strip_system=False)
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "orphan system"},  # non-standard position
        ]
        kwargs = transport._build_kwargs_from_profile(
            profile=profile,
            model="test-model",
            sanitized=msgs,
            tools=None,
            params={"messages": msgs},
        )
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "user"  # unchanged, no re-injection
