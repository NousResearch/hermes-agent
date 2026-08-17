"""Regression test for the thinking-only prefill reaching the wire.

A thinking-only response (reasoning tokens, no visible text) makes the loop
append an empty assistant turn and re-send so the model continues its own
reasoning. On providers that don't echo reasoning back, the API copy has its
reasoning fields stripped before ``_drop_thinking_only_and_merge_users`` runs,
so the drop pass used to see a bare ``{"role": "assistant", "content": ""}``
and let it through. Gemini rejects that with

    400 INVALID_ARGUMENT: Requests ending with a model turn are not supported.

classified as non-retryable, so the turn aborts outright.

Unlike the unit tests in ``test_thinking_only_sanitizer.py``, this drives
``run_conversation`` and asserts on the payload actually handed to the client,
so it exercises the API-copy build that decides whether ``_thinking_prefill``
survives as far as the drop pass.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client, mirroring the fixture in
    ``test_dropped_tool_call_recovery.py``."""
    from run_agent import AIAgent
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _thinking_only_response():
    """Reasoning tokens, no visible text — what triggers the prefill retry."""
    from tests.run_agent.test_run_agent import _mock_response
    return _mock_response(
        content="",
        finish_reason="stop",
        reasoning="Let me work through the request step by step.",
    )


def _final_response(text="Here is the answer."):
    """An ordinary text turn that ends the loop."""
    from tests.run_agent.test_run_agent import _mock_response
    return _mock_response(content=text, finish_reason="stop")


def _sent_messages(create_mock, call_index):
    call = create_mock.call_args_list[call_index]
    return call.kwargs.get("messages") or call.args[0].get("messages")


class TestThinkingPrefillTrailingTurn:

    def test_request_after_prefills_does_not_end_on_assistant(self, loop_agent):
        # Two thinking-only responses queue two prefill stubs, then the model
        # finally produces text. The third request is the one that used to go
        # out ending on a model turn.
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        create = loop_agent.client.chat.completions.create
        assert create.call_count >= 3, (
            "Two thinking-only responses should each trigger a prefill retry."
        )

        final_request = _sent_messages(create, 2)
        assert final_request[-1]["role"] != "assistant", (
            "Request ends on a model turn, which Gemini rejects with a "
            "non-retryable 400. The thinking-only prefill stubs must be "
            f"dropped before send. Got roles: {[m['role'] for m in final_request]}"
        )

    def test_prefill_stubs_are_absent_from_the_wire_payload(self, loop_agent):
        """The stubs should be gone entirely, not merely trailed by a nudge."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        sent = _sent_messages(loop_agent.client.chat.completions.create, 1)
        empty_assistants = [
            m for m in sent
            if m.get("role") == "assistant" and not (m.get("content") or "").strip()
        ]
        assert not empty_assistants, (
            f"Empty assistant stub(s) reached the wire: {empty_assistants}"
        )

    def test_thinking_only_retry_adds_visible_answer_nudge(self, loop_agent):
        """A thinking-only local response retries with new input, not a duplicate request."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        retry_payload = _sent_messages(loop_agent.client.chat.completions.create, 1)
        retry_text = "\n".join(
            str(message.get("content") or "")
            for message in retry_payload
            if message.get("role") == "user"
        )
        assert "visible answer" in retry_text.lower()
        assert "do not continue reasoning" in retry_text.lower()

    def test_internal_marker_never_reaches_the_wire(self, loop_agent):
        """``_thinking_prefill`` survives the API-copy build on purpose, but the
        transport must still keep it off the wire."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        sent = _sent_messages(loop_agent.client.chat.completions.create, 1)
        leaked = [m for m in sent if any(str(k).startswith("_") for k in m)]
        assert not leaked, f"Internal scaffolding keys reached the wire: {leaked}"


class _CapturingSessionDB:
    """Minimal SessionDB stand-in that records every appended message row."""

    def __init__(self):
        self.rows = []

    def append_message(self, session_id, role, content=None, **kwargs):
        self.rows.append({"role": role, "content": content})
        return len(self.rows)

    def append_messages_batch(self, session_id, messages, **kwargs):
        for m in messages:
            self.rows.append({"role": m.get("role"), "content": m.get("content")})
        return list(range(len(self.rows) - len(messages) + 1, len(self.rows) + 1))

    def flush_token_counts(self):
        pass


def _wire_user_text(create_mock, call_index):
    """Concatenated user-role content of one wire payload, lowercased."""
    sent = _sent_messages(create_mock, call_index)
    return "\n".join(
        str(m.get("content") or "")
        for m in sent
        if m.get("role") == "user"
    ).lower()


class TestThinkingPrefillNudgeRegression:
    """The four QA-required regression tests for the synthetic
    ``_THINKING_ONLY_VISIBLE_ANSWER_NUDGE`` injected on the first prefill."""

    def test_durable_transcript_excludes_synthetic_nudge(self, loop_agent):
        """The nudge is transport-only scaffolding: it must never reach the
        durable session store, even though it rides the wire on the retry."""
        loop_agent._session_db = _CapturingSessionDB()
        loop_agent._session_db_created = True
        loop_agent.session_id = "sess-nudge"

        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        persisted = loop_agent._session_db.rows
        assert persisted, "Expected at least the user + final assistant rows."
        for row in persisted:
            content = (row.get("content") or "") or ""
            assert "visible answer" not in content.lower(), (
                f"Synthetic nudge leaked into durable transcript: {row!r}"
            )
            assert "do not continue reasoning" not in content.lower(), (
                f"Synthetic nudge leaked into durable transcript: {row!r}"
            )

    def test_two_thinking_only_responses_inject_nudge_once_and_change_wire(
        self, loop_agent,
    ):
        """Two consecutive thinking-only responses must inject the nudge on
        exactly the first prefill retry, and that retry's wire payload must
        differ from the original request (not a byte-identical re-send)."""
        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _thinking_only_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            loop_agent.run_conversation("do the thing")

        create = loop_agent.client.chat.completions.create
        assert create.call_count >= 3, (
            "Two thinking-only responses should each trigger a prefill retry."
        )

        # The nudge is appended once to the live list, so it rides every
        # subsequent retry. "Injected once" means it is not re-appended on the
        # second prefill: the final retry must carry exactly one occurrence.
        assert "visible answer" in _wire_user_text(create, 1), (
            "The first prefill retry must carry the visible-answer nudge."
        )
        assert _wire_user_text(create, 2).count("visible answer") == 1, (
            "The nudge must be injected exactly once; the second prefill "
            "retry shows it duplicated."
        )

        original = _sent_messages(create, 0)
        retry = _sent_messages(create, 1)
        assert retry != original, (
            "The first prefill retry must differ from the original request; "
            "a byte-identical re-send would deterministically re-exhaust the "
            "reasoning budget."
        )

    def test_thinking_then_tool_call_leaves_clean_transcript(self, loop_agent):
        """thinking-only -> nudge -> tool call -> tool result -> final response
        must leave a clean durable transcript: no scaffolding flags and no
        nudge text reach the session store."""
        from tests.run_agent.test_run_agent import _mock_response, _mock_tool_call

        def _tool_call_response():
            return _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(name="read_file", arguments='{"path": "x"}')],
            )

        loop_agent.valid_tool_names = {"read_file"}
        loop_agent._session_db = _CapturingSessionDB()
        loop_agent._session_db_created = True
        loop_agent.session_id = "sess-tool"

        def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
            for tc in assistant_message.tool_calls:
                messages.append({
                    "role": "tool",
                    "name": tc.function.name,
                    "tool_call_id": tc.id,
                    "content": "file contents",
                })

        loop_agent.client.chat.completions.create.side_effect = [
            _thinking_only_response(),
            _tool_call_response(),
            _final_response(),
        ]

        with (
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            patch.object(loop_agent, "_execute_tool_calls", side_effect=_fake_execute),
        ):
            result = loop_agent.run_conversation("do the thing")

        assert result["final_response"] == "Here is the answer."
        assert result["completed"] is True

        # The durable transcript must be clean: the thinking-prefill stub and
        # the synthetic nudge are transport-only scaffolding and must never be
        # written to the session store, regardless of their in-memory position.
        persisted = loop_agent._session_db.rows
        assert persisted, "Expected durable rows for the completed turn."
        for row in persisted:
            content = (row.get("content") or "") or ""
            assert "visible answer" not in content.lower(), (
                f"Nudge text leaked into durable transcript: {row!r}"
            )
            assert "do not continue reasoning" not in content.lower(), (
                f"Nudge text leaked into durable transcript: {row!r}"
            )
        # The genuine turn survives in order: user -> assistant(tool_calls)
        # -> tool -> assistant(final).
        assert [r["role"] for r in persisted] == [
            "user", "assistant", "tool", "assistant",
        ], f"Unexpected durable transcript shape: {persisted!r}"
        assert persisted[-1]["content"] == "Here is the answer."

    def test_no_nudge_when_prefill_retry_is_not_first(self, loop_agent):
        """The nudge is gated on ``_thinking_prefill_retries == 1``. When the
        retry counter is already past the first attempt, no nudge may be
        injected — the wire must stay free of the synthetic continuation."""
        def _side_effect(**kwargs):
            if loop_agent.client.chat.completions.create.call_count == 1:
                # Simulate a non-first prefill: the counter is already 1, so
                # the next thinking-only response advances it to 2.
                loop_agent._thinking_prefill_retries = 1
                return _thinking_only_response()
            return _final_response()

        loop_agent.client.chat.completions.create.side_effect = _side_effect

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("do the thing")

        assert result["final_response"] == "Here is the answer."

        create = loop_agent.client.chat.completions.create
        for i in range(create.call_count):
            assert "visible answer" not in _wire_user_text(create, i), (
                f"Nudge injected on a non-first prefill retry (call {i})."
            )
