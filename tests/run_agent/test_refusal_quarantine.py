"""Regression tests for the content_filter quarantine + retry logic.

RCA: docs/rca-anthropic-refusal-session-poisoning.md

When Anthropic's input classifier fires on a tool result (empty refusal text,
sub-second latency, previous assistant turn had tool_calls), the loop must:

  1. Replace the last tool result with a placeholder (sha256 + length only),
  2. Persist the sanitised history,
  3. Retry exactly once,
  4. On success: return the successful response,
  5. On second refusal: surface the refusal without further retries,
  6. Persist the refusal as an assistant message (no user/user consecutive),
  7. After 2 consecutive refusals: surface the more-actionable repeated hint.

The original content must NEVER appear in logs or in the returned messages.
"""
from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_agent():
    """Minimal AIAgent with no network, no tools, no memory."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1/",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    return agent


def _response(*, content, finish_reason, tool_calls=None):
    """Minimal API response in OpenAI chat-completions shape."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test-model", usage=None)


def _refusal_response():
    """API response that triggers finish_reason=content_filter with empty text."""
    return _response(content=None, finish_reason="content_filter")


def _ok_response(text="All done."):
    """Normal stop response."""
    return _response(content=text, finish_reason="stop")


def _history_with_tool_result(tool_result_content: str = "some sensitive content"):
    """Pre-built conversation history ending with a tool call + tool result.

    Returns the list to pass as ``conversation_history`` to ``run_conversation``.
    The history looks like:

        user("do work") → assistant(tool_calls=[terminal]) → tool(result)

    so the detection scan will find a tool result preceded by an assistant
    with tool_calls.
    """
    return [
        {"role": "user", "content": "do some work"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tc1",
            "name": "terminal",
            "content": tool_result_content,
        },
    ]


# ── Test: quarantine + successful retry ────────────────────────────────────


class TestRefusalQuarantineSuccessfulRetry:
    """(a) empty refusal + low latency + tool result → quarantine + 1 retry → success."""

    def test_placeholder_replaces_tool_result(self):
        """After quarantine the returned messages must contain the placeholder,
        not the original tool result content."""
        agent = _make_agent()
        orig_content = "sensitive tool output"

        # Mock: fast refusal then success
        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),    # first call → content_filter
            _ok_response("Done."),  # quarantine retry → success
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue the work",
                conversation_history=_history_with_tool_result(orig_content),
            )

        # Must succeed
        assert result.get("failed") is not True

        # Original content must NOT appear anywhere in messages
        for m in result["messages"]:
            c = m.get("content") or ""
            if isinstance(c, list):
                for part in c:
                    assert orig_content not in str(part), (
                        f"Original content leaked into message part: {part!r}"
                    )
            else:
                assert orig_content not in c, (
                    f"Original content leaked into message: {m!r}"
                )

        # Placeholder must be present in a tool message
        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert tool_msgs, "No tool message found in result"
        last_tool_content = tool_msgs[-1].get("content", "")
        assert "quarantined after provider refusal" in last_tool_content
        assert "terminal" in last_tool_content

    def test_exactly_one_retry_on_first_refusal(self):
        """The API must be called exactly twice: once (refusal) + once (retry)."""
        agent = _make_agent()

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _ok_response(),
        ]

        with patch.object(agent, "_persist_session"):
            agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        assert agent.client.chat.completions.create.call_count == 2

    def test_sha256_in_placeholder(self):
        """The placeholder must include the first 16 hex chars of sha256."""
        agent = _make_agent()
        orig = "sensitive data here"

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _ok_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(orig),
            )

        expected_sha = hashlib.sha256(
            orig.encode("utf-8", errors="replace")
        ).hexdigest()[:16]

        tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
        assert expected_sha in tool_msgs[-1]["content"], (
            f"Expected sha256 prefix {expected_sha!r} in {tool_msgs[-1]['content']!r}"
        )


# ── Test: quarantine retry also refused ────────────────────────────────────


class TestQuarantineRetryAlsoRefused:
    """(b) retry also refusal → no further retries, return terminal result."""

    def test_no_third_api_call(self):
        """After quarantine + 2nd refusal the loop must not make a 3rd call."""
        agent = _make_agent()

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        # Exactly 2 API calls: original refusal + quarantine retry
        assert agent.client.chat.completions.create.call_count == 2
        assert result.get("failed") is True
        assert "content_policy_blocked" in (result.get("error") or "")

    def test_placeholder_still_present_after_double_refusal(self):
        """Even when both calls refuse, the tool result must stay quarantined."""
        agent = _make_agent()
        orig = "original tool output"

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(orig),
            )

        for m in result["messages"]:
            c = (m.get("content") or "")
            if isinstance(c, list):
                for part in c:
                    assert orig not in str(part)
            else:
                assert orig not in c


# ── Test: quarantine conditions NOT met ────────────────────────────────────


class TestQuarantineConditionsNotMet:
    """(c) If any condition is absent, quarantine must NOT fire."""

    def test_refusal_with_text_does_not_quarantine(self):
        """Non-empty refusal text → not an input-classifier case; no retry."""
        agent = _make_agent()

        # Refusal response with non-empty content
        msg = SimpleNamespace(
            content="This content violates our policy.",
            tool_calls=None,
            refusal="This content violates our policy.",
        )
        choice = SimpleNamespace(message=msg, finish_reason="content_filter")
        refusal_with_text = SimpleNamespace(
            choices=[choice], model="test-model", usage=None
        )
        agent.client.chat.completions.create.side_effect = [refusal_with_text]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        # Exactly 1 call — no quarantine retry
        assert agent.client.chat.completions.create.call_count == 1
        assert result.get("failed") is True

    def test_slow_refusal_does_not_quarantine(self):
        """Latency above threshold → no quarantine."""
        agent = _make_agent()

        agent.client.chat.completions.create.side_effect = [_refusal_response()]

        with (
            patch.object(agent, "_persist_session"),
            # Set threshold to effectively 0 so the real elapsed time (even a
            # millisecond) exceeds it.
            patch(
                "agent.conversation_loop._REFUSAL_QUARANTINE_LATENCY_THRESHOLD",
                0.0,
            ),
        ):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        assert agent.client.chat.completions.create.call_count == 1
        assert result.get("failed") is True

    def test_no_tool_result_does_not_quarantine(self):
        """History without a tool result → condition (c) false → no quarantine."""
        agent = _make_agent()
        # Only a user message, no tool result
        agent.client.chat.completions.create.side_effect = [_refusal_response()]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation("just a question")

        assert agent.client.chat.completions.create.call_count == 1
        assert result.get("failed") is True


# ── Test: assistant message persisted (no user/user consecutive) ────────────


class TestRefusalAssistantMessagePersisted:
    """(d) The refusal notice must be persisted as an assistant message."""

    def test_assistant_message_appended_on_terminal_refusal(self):
        """result['messages'] must end with an assistant role message."""
        agent = _make_agent()

        # Quarantine fires on first, retry also refusal → terminal
        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        last = result["messages"][-1]
        assert last.get("role") == "assistant", (
            f"Last message role must be 'assistant' to prevent user/user "
            f"consecutive; got {last.get('role')!r}"
        )

    def test_no_user_user_consecutive(self):
        """No two consecutive user-role messages in the returned history."""
        agent = _make_agent()

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        roles = [m.get("role") for m in result["messages"]]
        for i in range(len(roles) - 1):
            assert not (roles[i] == "user" and roles[i + 1] == "user"), (
                f"user/user consecutive at positions {i}/{i+1}: {roles}"
            )

    def test_assistant_message_contains_refusal_notice(self):
        """The appended assistant message must carry the refusal warning text."""
        agent = _make_agent()

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]

        with patch.object(agent, "_persist_session"):
            result = agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(),
            )

        last_content = result["messages"][-1].get("content") or ""
        assert "⚠️" in last_content or "safety refusal" in last_content.lower()


# ── Test: consecutive refusal hint replacement ──────────────────────────────


class TestConsecutiveRefusalHint:
    """(e) After ≥2 consecutive refusals in the same session, use a different hint."""

    def _run_refusal_turn(self, agent, hist):
        """Helper: run one turn that ends in a terminal double-refusal."""
        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _refusal_response(),
        ]
        with patch.object(agent, "_persist_session"):
            return agent.run_conversation("continue", conversation_history=hist)

    def test_first_refusal_uses_standard_hint(self):
        """First refusal: standard 'try rephrasing' hint."""
        from agent.conversation_loop import _CONTENT_POLICY_RECOVERY_HINT

        agent = _make_agent()
        result = self._run_refusal_turn(agent, _history_with_tool_result())

        last_content = result["messages"][-1].get("content") or ""
        assert _CONTENT_POLICY_RECOVERY_HINT in last_content, (
            f"Standard hint not found in: {last_content!r}"
        )

    def test_second_refusal_uses_repeated_hint(self):
        """Second consecutive refusal: replace hint with more-actionable text."""
        from agent.conversation_loop import _CONTENT_POLICY_REPEATED_HINT

        agent = _make_agent()
        # First turn → refusal (sets _refusal_consecutive_count = 1)
        self._run_refusal_turn(agent, _history_with_tool_result())
        # Second turn → refusal again (count becomes 2)
        result2 = self._run_refusal_turn(agent, _history_with_tool_result())

        last_content = result2["messages"][-1].get("content") or ""
        assert _CONTENT_POLICY_REPEATED_HINT in last_content, (
            f"Expected repeated hint in: {last_content!r}"
        )

    def test_successful_turn_resets_counter(self):
        """A successful turn resets the counter so next refusal shows standard hint."""
        from agent.conversation_loop import _CONTENT_POLICY_RECOVERY_HINT

        agent = _make_agent()
        # Refusal 1
        self._run_refusal_turn(agent, _history_with_tool_result())
        assert getattr(agent, "_refusal_consecutive_count", 0) == 1

        # Success
        agent.client.chat.completions.create.side_effect = [_ok_response("Done.")]
        with patch.object(agent, "_persist_session"):
            agent.run_conversation("simple question")
        assert getattr(agent, "_refusal_consecutive_count", 0) == 0

        # Next refusal → count is 1, should still show standard hint
        result3 = self._run_refusal_turn(agent, _history_with_tool_result())
        last_content = result3["messages"][-1].get("content") or ""
        assert _CONTENT_POLICY_RECOVERY_HINT in last_content


# ── Test: original content never logged ────────────────────────────────────


class TestOriginalContentNotLogged:
    """The original tool result must not appear in any log record."""

    def test_no_original_content_in_warning_logs(self, caplog):
        agent = _make_agent()
        orig = "super_secret_tool_output_xyz"

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _ok_response(),
        ]

        with (
            patch.object(agent, "_persist_session"),
            caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
        ):
            agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(orig),
            )

        for record in caplog.records:
            assert orig not in record.getMessage(), (
                f"Original content leaked into log: {record.getMessage()!r}"
            )

    def test_sha256_present_in_warning_log(self, caplog):
        agent = _make_agent()
        orig = "super_secret_tool_output_xyz"
        expected_sha = hashlib.sha256(
            orig.encode("utf-8", errors="replace")
        ).hexdigest()[:16]

        agent.client.chat.completions.create.side_effect = [
            _refusal_response(),
            _ok_response(),
        ]

        with (
            patch.object(agent, "_persist_session"),
            caplog.at_level(logging.WARNING, logger="agent.conversation_loop"),
        ):
            agent.run_conversation(
                "continue",
                conversation_history=_history_with_tool_result(orig),
            )

        log_text = " ".join(r.getMessage() for r in caplog.records)
        assert expected_sha in log_text, (
            f"sha256 prefix {expected_sha!r} not found in logs.\n"
            f"All log messages: {log_text!r}"
        )
