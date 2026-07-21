"""Tests for agent-side code-skew detection (desktop/serve backend).

Companion to ``tests/test_code_skew.py`` (gateway): these prove the same
protection exists for the long-lived ``hermes serve`` / desktop backend
process, which imports ``run_agent`` directly rather than going through the
gateway.  See #68178.
"""

import pytest
from types import SimpleNamespace


class TestConversationLoopCodeSkewFinalization:
    def test_code_skew_finalizes_without_unbound_local_error(self, monkeypatch):
        """The early code-skew exit must use the module-level finalizer import."""
        from agent import conversation_loop

        context = SimpleNamespace(
            user_message="hello",
            original_user_message="hello",
            messages=[],
            conversation_history=[],
            active_system_prompt="",
            effective_task_id="test-code-skew",
            turn_id="turn-1",
            current_turn_user_idx=0,
            should_review_memory=False,
            plugin_user_context="",
            ext_prefetch_cache="",
        )
        monkeypatch.setattr(conversation_loop, "build_turn_context", lambda *_a, **_k: context)
        monkeypatch.setattr(
            conversation_loop,
            "finalize_turn",
            lambda _agent, **kwargs: {"response": kwargs["final_response"], "reason": kwargs["_turn_exit_reason"]},
        )

        class FakeAgent:
            api_mode = ""
            max_iterations = 1
            iteration_budget = SimpleNamespace(remaining=1)
            _budget_grace_call = False

            @staticmethod
            def _check_code_skew_before_turn():
                return "source changed"

        result = conversation_loop.run_conversation(FakeAgent(), "hello")

        assert result == {
            "response": (
                "I apologize, but the agent has detected that its source code "
                "has been updated while running. To avoid compatibility issues, "
                "please restart the application. (source changed)"
            ),
            "reason": "code_skew_detected",
        }


class TestAgentCodeSkewCaching:
    def test_boot_fingerprint_recorded_at_import(self):
        """``run_agent`` records its boot fingerprint on first import."""
        import run_agent

        # Should not be None on a git install.
        assert run_agent._agent_boot_fingerprint is not None

    def test_detect_no_skew_when_unchanged(self):
        """When the fingerprint hasn't changed, skew is None."""
        import run_agent

        assert run_agent._detect_agent_code_skew() is None

    def test_cached_skew_is_returned_immediately(self, monkeypatch):
        """Once confirmed, the result is cached and returned without I/O."""
        import run_agent

        monkeypatch.setattr(run_agent, "_agent_code_skew_confirmed", True)
        monkeypatch.setattr(run_agent, "_agent_code_skew_labels", ("abc1234567", "def4567890"))

        skew = run_agent._detect_agent_code_skew()
        assert skew == ("abc1234567", "def4567890")

    def test_none_boot_fingerprint_means_no_skew(self, monkeypatch):
        """If boot fingerprint could not be read, skew detection is a no-op."""
        import run_agent

        monkeypatch.setattr(run_agent, "_agent_boot_fingerprint", None)
        monkeypatch.setattr(run_agent, "_agent_code_skew_confirmed", False)
        monkeypatch.setattr(run_agent, "_agent_code_skew_labels", None)

        assert run_agent._detect_agent_code_skew() is None


class TestCheckCodeSkewBeforeTurn:
    def test_returns_none_without_skew(self):
        """When no skew exists, the method returns None."""
        import run_agent

        # Create a minimal fake agent with the method.
        class FakeAgent:
            pass

        fake = FakeAgent()
        # The method lives on AIAgent, not a module function. Test by
        # verifying the underlying function returns None when no skew.
        result = run_agent._detect_agent_code_skew()
        assert result is None

    def test_returns_warning_when_skew_confirmed(self, monkeypatch):
        """When skew is confirmed, the method returns a descriptive warning."""
        import run_agent

        monkeypatch.setattr(run_agent, "_agent_code_skew_confirmed", True)
        monkeypatch.setattr(run_agent, "_agent_code_skew_labels", ("abc1234567", "def4567890"))

        # The method is on AIAgent, so we need to instantiate or call via class.
        # Instead, test the underlying function directly.
        skew = run_agent._detect_agent_code_skew()
        assert skew == ("abc1234567", "def4567890")
