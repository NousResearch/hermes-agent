"""Tests for the pre-route heuristic in run_conversation.

Covers _should_delegate_to_claude_code and the pre-route block that
bypasses the main conversation loop for tool-heavy turns.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _should_delegate_to_claude_code — unit tests
# ---------------------------------------------------------------------------

class TestShouldDelegate:
    def _call(self, message: str, env: dict | None = None) -> bool:
        from agent.conversation_loop import _should_delegate_to_claude_code
        with patch.dict(os.environ, env or {}, clear=False):
            return _should_delegate_to_claude_code(message)

    # -- Flag disabled -------------------------------------------------------

    def test_disabled_by_env_zero(self):
        assert self._call("browse to example.com", {"HERMES_DELEGATE_HEURISTICS": "0"}) is False

    def test_disabled_by_env_false(self):
        assert self._call("screenshot", {"HERMES_DELEGATE_HEURISTICS": "false"}) is False

    def test_disabled_by_env_off(self):
        assert self._call("search the web", {"HERMES_DELEGATE_HEURISTICS": "off"}) is False

    # -- Non-string / empty inputs -------------------------------------------

    def test_non_string_returns_false(self):
        from agent.conversation_loop import _should_delegate_to_claude_code
        assert _should_delegate_to_claude_code(None) is False
        assert _should_delegate_to_claude_code(42) is False
        assert _should_delegate_to_claude_code([]) is False

    def test_empty_string_returns_false(self):
        assert self._call("") is False
        assert self._call("   ") is False

    # -- Capture prefixes ----------------------------------------------------

    def test_capture_this(self):
        assert self._call("capture this: interesting thought") is True

    def test_capture_colon(self):
        assert self._call("capture: some note here") is True

    def test_slash_capture(self):
        assert self._call("/capture idea for later") is True

    def test_save_this_thought(self):
        assert self._call("save this thought: remember X") is True

    def test_remember_this_colon(self):
        assert self._call("remember this: deadline friday") is True

    def test_log_this(self):
        assert self._call("log this: bug found in auth") is True

    def test_write_to_vault(self):
        assert self._call("write this to my vault: idea") is True

    # -- Tool markers --------------------------------------------------------

    def test_browse_to(self):
        assert self._call("browse to https://example.com") is True

    def test_screenshot(self):
        assert self._call("screenshot the current page") is True

    def test_screenshot_prefix_only(self):
        """'take a screenshot' should NOT pre-route — doesn't start with 'screenshot'."""
        assert self._call("take a screenshot of the current page") is False

    def test_screenshot_plural_not_delegated(self):
        """'screenshots work...' should NOT pre-route — trailing space required (F3 fix)."""
        assert self._call("screenshots work better than text descriptions") is False

    def test_screenshot_question_not_delegated(self):
        """'screenshots are broken...' should NOT pre-route — 'screenshot' must be followed by space."""
        assert self._call("screenshots are broken in hermes, how do I debug?") is False

    def test_search_the_web(self):
        assert self._call("search the web for recent news") is True

    def test_web_search(self):
        assert self._call("web search: python asyncio tutorial") is True

    def test_run_command(self):
        assert self._call("run command: ls -la") is True

    def test_run_this_command(self):
        assert self._call("run this command in the terminal") is True

    def test_shell_command(self):
        assert self._call("shell command: echo hello") is True

    def test_use_playwright(self):
        assert self._call("use playwright to fill the form") is True

    def test_open_browser(self):
        assert self._call("open browser and navigate to github") is True

    def test_inspect_this_page(self):
        assert self._call("inspect this page for accessibility issues") is True

    # -- Non-matching messages -----------------------------------------------

    def test_normal_chat_not_delegated(self):
        assert self._call("what time is it?") is False

    def test_memory_question_not_delegated(self):
        assert self._call("what did I tell you about my project?") is False

    def test_code_question_not_delegated(self):
        assert self._call("explain this Python function to me") is False

    def test_case_insensitive_marker(self):
        assert self._call("BROWSE TO example.com") is True

    def test_case_insensitive_capture(self):
        assert self._call("Capture This: my idea") is True

    # -- Negation guard (F3 fix: prefix-only prevents mid-sentence false positives) --

    def test_negation_screenshot_not_delegated(self):
        """'please do NOT screenshot this' should NOT pre-route."""
        assert self._call("please do NOT screenshot this, just describe it") is False

    def test_negation_run_command_not_delegated(self):
        """'is it safe to run command X?' should NOT pre-route."""
        assert self._call("is it safe to run command X without sudo?") is False

    def test_negation_shell_command_not_delegated(self):
        """'this is not a shell command' should NOT pre-route."""
        assert self._call("this is not a shell command") is False

    def test_negation_search_web_not_delegated(self):
        """'don't search the web for this' should NOT pre-route."""
        assert self._call("don't search the web for this, I know the answer") is False

    def test_mid_sentence_browse_not_delegated(self):
        """'I never said to browse to that page' should NOT pre-route."""
        assert self._call("I never said to browse to that page") is False


# ---------------------------------------------------------------------------
# Pre-route block integration — tests that the route fires and returns early
# ---------------------------------------------------------------------------

class TestPreRouteBlock:
    """Verify the pre-route block in run_conversation fires correctly."""

    def _make_agent(self, valid_tool_names=None):
        agent = MagicMock()
        agent.valid_tool_names = valid_tool_names or {"delegate_to_claude_code"}
        agent.session_id = "test-session"
        agent.platform = "test"
        agent.status_callback = None
        agent._persist_session = MagicMock()
        return agent

    def test_pre_route_fires_and_returns_early(self):
        """When heuristic matches and delegate tool is available, pre-route returns."""
        agent = self._make_agent()
        fake_result = json.dumps({"success": True, "result": "Done: opened browser"})

        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   return_value=fake_result):
            from agent.conversation_loop import run_conversation
            result = run_conversation(agent, "browse to example.com")

        assert result["api_calls"] == 0
        assert result["completed"] is True
        assert "Done: opened browser" in result["final_response"]
        assert result.get("delegate_result", {}).get("success") is True

    def test_pre_route_skipped_when_tool_not_available(self):
        """No pre-route when delegate_to_claude_code is not in valid_tool_names."""
        agent = self._make_agent(valid_tool_names={"memory", "web_search"})
        # If pre-route fires, it would call the delegate tool; if skipped,
        # run_conversation falls into the main loop which we don't stub here —
        # we only assert the pre-route doesn't trigger.
        with patch("agent.conversation_loop._should_delegate_to_claude_code",
                   return_value=True) as mock_heuristic, \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code") as mock_delegate:
            # The main loop will fail (agent mocked), but we only care that
            # delegate_to_claude_code was NOT called.
            try:
                from agent.conversation_loop import run_conversation
                run_conversation(agent, "browse to example.com")
            except Exception:
                pass
            mock_delegate.assert_not_called()

    def test_pre_route_skipped_when_heuristic_returns_false(self):
        """No pre-route when heuristic returns False."""
        agent = self._make_agent()
        with patch("agent.conversation_loop._should_delegate_to_claude_code",
                   return_value=False), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code") as mock_delegate:
            try:
                from agent.conversation_loop import run_conversation
                run_conversation(agent, "what time is it?")
            except Exception:
                pass
            mock_delegate.assert_not_called()

    def test_pre_route_falls_through_on_exception(self):
        """On delegate exception, pre-route falls through to normal loop."""
        agent = self._make_agent()
        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   side_effect=RuntimeError("delegate unavailable")):
            # Should NOT raise — should fall through to normal loop.
            # Normal loop will fail too (agent mocked), but we expect a non-pre-route error.
            try:
                from agent.conversation_loop import run_conversation
                result = run_conversation(agent, "browse to example.com")
                # If it returned, the pre-route returned a failure dict
                # (old code path) or fell through
            except Exception as exc:
                # The exception came from the normal loop, NOT from the pre-route
                assert "delegate unavailable" not in str(exc), (
                    "Pre-route exception should not propagate — should fall through"
                )

    def test_pre_route_handles_non_json_result(self):
        """Non-JSON delegate result is returned as raw string with completed=True."""
        agent = self._make_agent()
        raw = "Done in 3 steps."
        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   return_value=raw):
            from agent.conversation_loop import run_conversation
            result = run_conversation(agent, "browse to example.com")

        assert result["api_calls"] == 0
        assert raw in result["final_response"]
        # Non-JSON is a valid response, not a failure — completed must be True.
        assert result["completed"] is True, (
            "Non-JSON tool result was incorrectly marked completed=False"
        )

    def test_pre_route_increments_user_turn_count(self):
        """Pre-routed turns must increment _user_turn_count for nudge logic to work."""
        agent = self._make_agent()
        agent._user_turn_count = 0
        fake_result = json.dumps({"success": True, "result": "done"})
        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   return_value=fake_result):
            from agent.conversation_loop import run_conversation
            run_conversation(agent, "browse to example.com")

        assert agent._user_turn_count == 1, (
            "Pre-route must increment _user_turn_count so memory nudges accumulate"
        )

    def test_pre_route_no_double_count_on_delegate_exception(self):
        """On delegate exception, _user_turn_count must NOT be incremented by the pre-route."""
        agent = self._make_agent()
        agent._user_turn_count = 0
        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   side_effect=RuntimeError("delegate failed after import")):
            try:
                from agent.conversation_loop import run_conversation
                run_conversation(agent, "browse to example.com")
            except Exception:
                pass
        # Normal loop increments once on fallthrough — count must be exactly 1.
        # If pre-route incremented before the exception, then normal loop also
        # increments on fallthrough, giving 2 (double-count).
        assert agent._user_turn_count == 1, (
            "Expected exactly 1 increment (normal loop); "
            "got double-count — pre-route must not increment before delegate call"
        )

    def test_pre_route_uses_persist_user_message_for_history(self):
        """Gateway/API-server callers pass persist_user_message; that cleaned version must be stored."""
        agent = self._make_agent()
        fake_result = json.dumps({"success": True, "result": "Done."})
        raw_msg = "Voice context prefix: browse to example.com"
        clean_msg = "browse to example.com"
        with patch("agent.conversation_loop._should_delegate_to_claude_code", return_value=True), \
             patch("tools.claude_code_delegate_tool.delegate_to_claude_code",
                   return_value=fake_result):
            from agent.conversation_loop import run_conversation
            result = run_conversation(
                agent, raw_msg, persist_user_message=clean_msg
            )
        # The persisted user message must be the clean version, not the raw wrapper
        user_turn = next(
            (m for m in result["messages"] if m.get("role") == "user"), None
        )
        assert user_turn is not None
        assert user_turn["content"] == clean_msg, (
            f"Expected clean persist_user_message in history, got: {user_turn['content']!r}"
        )
