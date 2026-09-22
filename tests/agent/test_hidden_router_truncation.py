"""Regression: a router-HIDDEN truncation must enter bounded chunking recovery.

The provider/router can report ``finish_reason="tool_calls"`` (or ``"stop"``) while the
output limit actually cut a tool call's arguments off mid-stream. ``turn_tool_validation``
detects that from the broken JSON, but ends the turn immediately through ``_partial_exit``
— skipping the bounded chunking recovery entirely (and, on this base, skipping
``recover_from_truncation`` too, which never ran because finish_reason was not "length").

Contract pinned here:

- the broken call is never dispatched and never staged into the transcript;
- the turn does NOT end partial on the first hidden truncation;
- bounded chunking recovery runs instead (targeted smaller-calls guidance, no max_tokens
  boost) and shares ONE ``_ChunkingProgress`` budget with the ``length`` path;
- recovery stays hard-bounded when the model keeps repeating;
- a complete, genuinely malformed (not truncated) payload keeps the historical behaviour;
- the provider-flagged ``finish_reason="length"`` path is unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_truncation import _TRUNCATED_FINAL
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

# Unrepairable: JSON cut off inside a string, no closing brace — an output-limit cut of an
# oversized ``write_file`` payload as seen when a router rewrites finish_reason.
TRUNCATED_ARGS = '{"path": "/tmp/hidden.md", "content": "' + ("x" * 400)
# Complete but genuinely malformed: does not end in ``}``/``]``, yet the provider did NOT
# leave it incomplete. Must never be reclassified as an output-limit truncation.
MALFORMED_TERMINATED = '{"query": "abc"} trailing-garbage'


def _hidden_truncated_response(finish_reason="tool_calls", name="write_file",
                               arguments=TRUNCATED_ARGS, call_id="call_hidden"):
    return _mock_response(
        content="", finish_reason=finish_reason,
        tool_calls=[_mock_tool_call(name=name, arguments=arguments, call_id=call_id)],
    )


def _tool_round(*calls):
    return _mock_response(
        content="", finish_reason="tool_calls",
        tool_calls=[_mock_tool_call(name=n, arguments=a) for n, a in calls],
    )


@pytest.fixture()
def loop_agent():
    """AIAgent with one real tool schema and a mocked client so the loop runs for real."""
    tool_defs = [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "write",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="https://example.com/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.max_tokens = 4096
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _run(agent, user_message="write the file"):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(user_message)


def _request_messages(call):
    kwargs = call.kwargs or {}
    return kwargs.get("messages") or call.args[0].get("messages")


def _request_max_tokens(call):
    kwargs = call.kwargs or {}
    if "max_tokens" in kwargs:
        return kwargs["max_tokens"]
    return kwargs.get("max_output_tokens")


def _flat(messages):
    return json.dumps(messages, default=str)


def _last_user_text(messages):
    return next(m for m in reversed(messages) if m.get("role") == "user").get("content") or ""


class TestHiddenTruncationEntersBoundedRecovery:
    """CASE 1: finish_reason='tool_calls' + args cut off mid-stream (the reported failure)."""

    def _stage(self, agent, *responses):
        agent.client.chat.completions.create.side_effect = list(responses)
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        return dispatched, fake_dispatch

    def test_hidden_truncation_recovers_in_turn_instead_of_partial_exit(self, loop_agent):
        dispatched, fake_dispatch = self._stage(
            loop_agent,
            _hidden_truncated_response("tool_calls"),
            _tool_round(("write_file", '{"path": "/tmp/a.md"}')),
            _mock_response(content="done", finish_reason="stop"),
        )

        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 3, (
            "a hidden truncation must be recovered in-turn (chunking nudge + chunked round), "
            f"not end the turn (made {len(calls)} API calls)"
        )
        assert result["completed"] is True, (
            f"turn must complete after recovery (result={result.get('final_response')!r})"
        )
        assert result["final_response"] == "done"
        assert not result.get("partial")

        # The broken call never reached a dispatch path.
        assert all(args != TRUNCATED_ARGS for _, args in dispatched), dispatched
        assert [name for name, _ in dispatched] == ["write_file"]

        # The recovery request carries the chunking guidance and NOT the broken payload.
        second = _request_messages(calls[1])
        guidance = _last_user_text(second)
        assert "was too large" in guidance
        assert "multiple smaller tool calls" in guidance
        assert TRUNCATED_ARGS not in _flat(second), "broken response must be discarded"

        # No max_tokens boost: chunking is the recovery, not a bigger output budget.
        assert _request_max_tokens(calls[1]) == _request_max_tokens(calls[0])

    def test_hidden_truncation_with_stop_finish_reason_recovers_too(self, loop_agent):
        dispatched, fake_dispatch = self._stage(
            loop_agent,
            _hidden_truncated_response("stop"),
            _tool_round(("write_file", '{"path": "/tmp/b.md"}')),
            _mock_response(content="ok", finish_reason="stop"),
        )
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        assert result["completed"] is True
        assert "was too large" in _last_user_text(
            _request_messages(loop_agent.client.chat.completions.create.call_args_list[1])
        )
        assert all(args != TRUNCATED_ARGS for _, args in dispatched)


class TestBoundedAndSharedBudget:
    """CASE 2: recovery is hard-bounded and shares ONE progress budget across both paths."""

    def test_repeating_hidden_truncation_is_bounded(self, loop_agent):
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _hidden_truncated_response("tool_calls", call_id=f"c{i}") for i in range(40)
        ]

        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) <= 6, (
            f"a non-converging hidden truncation must stay bounded (made {len(calls)} calls)"
        )
        assert dispatched == [], "an incomplete tool call must never execute"
        assert result["completed"] is False
        # The turn ends with the canonical truncation copy and its verdict code. Upstream moved
        # the user-facing sentence into agent/turn_failure_copy (site_copy("truncated")), so the
        # old "truncated" substring assertion was an outdated expectation — the code plus the
        # canonical copy is the stable contract.
        assert result["failure_reason"] == "truncated"
        assert result["final_response"] == _TRUNCATED_FINAL
        # No exponential max_tokens boost ladder on this path.
        assert {_request_max_tokens(c) for c in calls} == {4096}

    def test_hidden_and_flag_truncations_share_one_chunking_budget(self, loop_agent):
        """Both kinds of truncation write into the SAME ``_ChunkingProgress``: the combined
        run must not get a fresh allowance per kind (that would double the bound)."""
        sizes = [4096, 3000, 2000, 1200, 800, 500, 300, 200, 100, 50]
        responses = []
        for i, size in enumerate(sizes):
            args = '{"path": "/tmp/h%d.md", "content": "' % i + ("x" * size)
            responses.append(_mock_response(
                content="",
                # Alternate the hidden reason and the provider-flagged one.
                finish_reason="tool_calls" if i % 2 == 0 else "length",
                tool_calls=[_mock_tool_call(name="write_file", arguments=args, call_id=f"c{i}")],
            ))
        responses.append(_mock_response(content="finished", finish_reason="stop"))

        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = responses
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        # Converging payloads across both kinds keep recovering, bounded by the SHARED budget
        # (base 4 + at most 4 extensions = 20 attempts), never by a per-kind allowance.
        assert len(calls) <= 21, f"shared budget must stay hard-bounded (made {len(calls)} calls)"
        assert result["completed"] is True, (
            f"converging run must finish (final={result.get('final_response')!r})"
        )
        assert dispatched == [], "no truncated payload may ever be dispatched"


class TestNoFalsePositive:
    """CASE 3: a COMPLETE payload the provider did not leave incomplete is not a truncation."""

    def test_complete_malformed_json_is_not_reclassified_as_truncation(self, loop_agent):
        """``{"query": "abc"} trailing-garbage`` does not end in ``}``/``]``, but the JSON is
        closed — the provider did not cut it off. It must never seed a chunking recovery."""
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content="", finish_reason="tool_calls",
                tool_calls=[_mock_tool_call(
                    name="write_file", arguments=MALFORMED_TERMINATED, call_id="bad")],
            ),
            _mock_response(content="recovered", finish_reason="stop"),
        ]
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        for c in calls:
            assert "was too large" not in _flat(_request_messages(c)), (
                "a complete (merely malformed) payload must not trigger chunking recovery"
            )
        assert all(args != MALFORMED_TERMINATED for _, args in dispatched)

    def test_malformed_and_incomplete_pair_is_recovered_but_nothing_dispatched(self, loop_agent):
        """A batch mixing a complete-but-malformed call with an incomplete one: the incomplete
        call governs (recovery), and NEITHER is dispatched."""
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(
                content="", finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(name="write_file", arguments=MALFORMED_TERMINATED, call_id="mf"),
                    _mock_tool_call(name="write_file", arguments=TRUNCATED_ARGS, call_id="tr"),
                ],
            ),
            _tool_round(("write_file", '{"path": "/tmp/ok.md"}')),
            _mock_response(content="done", finish_reason="stop"),
        ]
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        assert all(args not in (TRUNCATED_ARGS, MALFORMED_TERMINATED) for _, args in dispatched)
        assert result["completed"] is True
        assert "was too large" in _last_user_text(
            _request_messages(loop_agent.client.chat.completions.create.call_args_list[1])
        )


class TestProviderFlaggedPathUnchanged:
    """CASE 4: the ``finish_reason='length'`` path behaves exactly as before."""

    def test_length_finish_reason_with_unrepairable_args_still_chunks(self, loop_agent):
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _hidden_truncated_response("length"),
            _tool_round(("write_file", '{"path": "/tmp/l.md"}')),
            _mock_response(content="done", finish_reason="stop"),
        ]
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) == 3
        assert result["completed"] is True
        assert "was too large" in _last_user_text(_request_messages(calls[1]))
        assert all(args != TRUNCATED_ARGS for _, args in dispatched)
        assert _request_max_tokens(calls[1]) == _request_max_tokens(calls[0])

    def test_valid_tool_call_with_stop_is_unaffected(self, loop_agent):
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _tool_round(("write_file", '{"path": "/tmp/valid.md"}')),
            _mock_response(content="done", finish_reason="stop"),
        ]
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        assert result["completed"] is True
        assert [name for name, _ in dispatched] == ["write_file"]

    def test_repeated_length_truncation_is_bounded_by_shared_budget(self, loop_agent):
        """A non-converging ``length`` truncation must stay hard-bounded too — the chunking
        budget replaced the old max_tokens boost ladder for this path."""
        dispatched = []

        def fake_dispatch(name, args, task_id, *positional, **kwargs):
            dispatched.append((name, args))
            return json.dumps({"ok": True})

        loop_agent.client.chat.completions.create.side_effect = [
            _hidden_truncated_response("length", call_id=f"c{i}") for i in range(40)
        ]
        with (
            patch("model_tools.handle_function_call", side_effect=fake_dispatch),
            patch.object(loop_agent, "_invoke_tool", side_effect=fake_dispatch),
            patch("agent.tool_executor.maybe_persist_tool_result",
                  side_effect=lambda **kwargs: kwargs["content"]),
        ):
            result = _run(loop_agent)

        calls = loop_agent.client.chat.completions.create.call_args_list
        assert len(calls) <= 6, f"length-path chunking must stay bounded (made {len(calls)} calls)"
        assert dispatched == []
        assert result["completed"] is False
        assert {_request_max_tokens(c) for c in calls} == {4096}


class TestTurnScopedRetryStateSurvivesThePhase:
    """The recovery phase hands the loop's own verdict fields back (``_run_phase`` copies
    every field of the returned verdict onto ``_LoopState``), so it must not clobber
    turn-scoped retry state the turn had already accumulated — the text-continuation counter
    and the accumulated partial fragments both live on the loop, not on the per-iteration
    ``TurnRetryState``."""

    def test_phase_preserves_turn_scoped_continuation_state(self, loop_agent):
        from types import SimpleNamespace

        from agent.turn_retry_state import TurnRetryState
        from agent.turn_truncation import _ChunkingProgress, _recover_hidden_truncation_phase

        request = SimpleNamespace(
            finish_reason="tool_calls",
            assistant_message=_hidden_truncated_response("tool_calls").choices[0].message,
            broken_tools=["write_file"],
        )
        messages = [{"role": "user", "content": "write the file"}]

        verdict = _recover_hidden_truncation_phase(
            loop_agent, request, TurnRetryState(), messages=messages,
            conversation_history=None, api_call_count=1, effective_task_id="task",
            current_turn_user_idx=0, truncated_tool_call_retries=0, retry_count=0,
            compression_attempts=0, length_continue_retries=3,
            truncated_response_parts=["already stitched"], chunking_progress=_ChunkingProgress(),
        )

        assert verdict.action == "break"
        assert verdict.length_continue_retries == 3
        assert verdict.truncated_response_parts == ["already stitched"]

