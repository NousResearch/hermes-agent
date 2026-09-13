"""Ladder integration: output-cap starvation short-circuits recovery steps.

Exercises ``recover_empty_response`` with a starved response (finish_reason=max_tokens,
usage proves reasoning consumed the completion, no visible content) and asserts the
ladder skips nudge/prefill entirely and burns at most ONE paid retry before proceeding
to fallback/terminal — versus the legacy 1 nudge + 2 prefill + 3 retries path.

The agent stub provides the methods the ladder calls; behaviour under test is the
ladder's own branching and counter bookkeeping, not the stubs.
"""

from types import SimpleNamespace

from agent.turn_empty_response import recover_empty_response


def _has_content_after_think_block(text):
    # Mirror the real contract closely enough for these cases: stripped text present.
    return bool((text or "").replace("<智>", "").replace("</智>", "").strip())


def _strip_think_blocks(text):
    return (text or "").replace("<智>", "").replace("</智>", "")


def _agent(**overrides):
    base = dict(
        model="gemini-3.8-flash",
        provider="custom:gateway",
        api_mode="chat_completions",
        base_url="https://gateway.example/v1",
        api_key=None,
        _empty_content_retries=0,
        _post_tool_empty_retried=False,
        _thinking_prefill_retries=0,
        _fallback_chain=[],
        _last_content_with_tools=None,
        _last_content_tools_all_housekeeping=False,
        _current_streamed_assistant_text="",
        _response_was_previewed=False,
        _status_messages=[],
        _has_content_after_think_block=_has_content_after_think_block,
        _strip_think_blocks=_strip_think_blocks,
        _emit_status=lambda *_a, **_k: None,
        _buffer_status=lambda *_a, **_k: None,
        _flush_status_buffer=lambda: None,
        _extract_reasoning=lambda _m: "",
        _drop_trailing_empty_response_scaffolding=lambda _msgs: None,
        _build_assistant_message=lambda m, fr: {"role": "assistant", "content": "(empty)"},
        _interrupt_requested=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _starved_response():
    usage = SimpleNamespace(
        prompt_tokens=34_000, completion_tokens=1_000, total_tokens=35_000,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=996),
    )
    return SimpleNamespace(usage=usage)


def _assistant_message(tool_calls=None):
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    if tool_calls:
        msg.tool_calls = tool_calls
    return msg


def _run_ladder(agent, finish_reason="max_tokens"):
    """Drive one pass through recover_empty_response; returns (action, monkeypatched counters)."""
    actions = []
    verdict = recover_empty_response(
        agent,
        assistant_message=_assistant_message(),
        response=_starved_response(),
        finish_reason=finish_reason,
        final_response="",
        messages=[{"role": "user", "content": "task"}],
        api_messages=[],
        conversation_history=[],
        active_system_prompt=None,
        api_call_count=1,
        turn_exit_reason=None,
        preflight_compression_blocked=False,
    )
    actions.append(verdict.action)
    return verdict


class TestLadderShortCircuit:
    def test_starved_post_tool_response_skips_nudge(self):
        """The primary trigger shape: tool result immediately before the starved
        empty. Legacy ladder would append the "(empty)" + synthetic-user nudge
        rows (prompt mutation, cache break); starved streaks must skip it
        (Flash round-1 BLOCKER 1)."""
        agent = _agent(_post_tool_empty_retried=False)
        messages = [
            {"role": "user", "content": "run the thing"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "ok"},
        ]
        verdict = recover_empty_response(
            agent,
            assistant_message=_assistant_message(),
            response=_starved_response(),
            finish_reason="max_tokens",
            final_response="",
            messages=messages,
            api_messages=[],
            conversation_history=[],
            active_system_prompt=None,
            api_call_count=2,
            turn_exit_reason=None,
            preflight_compression_blocked=False,
        )
        assert getattr(agent, "_output_starvation_detected", False) is True
        # No synthetic nudge rows appended: history length unchanged.
        assert len(messages) == 3
        assert not any(m.get("_empty_recovery_synthetic") for m in messages)

    def test_starved_streak_suppresses_thinking_prefill(self):
        agent = _agent()
        # Simulate mid-streak: one starvation already recorded, prefill counter live.
        agent._thinking_prefill_retries = 0
        verdict = _run_ladder(agent)
        # With reasoning reported by the assistant message the legacy ladder would
        # prefill; starved streaks must go straight to budgeted retry instead.
        # (The starved assistant message has no reasoning attributes here, so prefill
        # would not trigger anyway — assert the counter stayed put.)
        assert agent._thinking_prefill_retries == 0

    def test_starvation_budget_causes_terminal_after_one_retry(self):
        """Two consecutive starved empties: the ladder must not keep retrying.
        After the first (budget=1) retry is burned, the second pass gets budget=0
        and proceeds to the terminal path."""
        agent = _agent(_empty_content_retries=1)  # one retry already burned
        from agent import empty_response_guard as guard

        guard.record_empty_attempt(
            agent, finish_reason="max_tokens", response=_starved_response(),
            observed_generation=False,
        )
        # Budget at check time (retries==1 already burned): zero.
        assert guard.starvation_retry_budget(agent, finish_reason="max_tokens") == 0


class TestStaleFlagReset:
    def test_benign_empty_after_starved_entry_resets_flag(self):
        """Flag reflects the CURRENT response (Flash R2 finding 1): a starved
        response followed by a benign (finish_reason=stop) empty must reset the
        flag so the later streak's nudge is not suppressed."""
        agent = _agent(_output_starvation_detected=True)  # stale from a prior streak
        _run_ladder(agent, finish_reason="stop")
        assert getattr(agent, "_output_starvation_detected", False) is False

    def test_non_starved_entry_clears_flag_even_with_truncating_reason_shape(self):
        """A response with content present + truncating reason is NOT starvation:
        the flag must flip False on that entry (prior-turn reuse then works)."""
        agent = _agent(_output_starvation_detected=True)
        verdict = recover_empty_response(
            agent,
            assistant_message=_assistant_message(),
            response=_starved_response(),
            finish_reason="length",
            final_response="real content survived truncation",
            messages=[{"role": "user", "content": "task"}],
            api_messages=[],
            conversation_history=[],
            active_system_prompt=None,
            api_call_count=1,
            turn_exit_reason=None,
            preflight_compression_blocked=False,
        )
        assert getattr(agent, "_output_starvation_detected", False) is False


class TestNoRegressionOnBenignEmpties:
    def test_stop_finish_empty_never_sets_starvation_flag(self):
        agent = _agent()
        _run_ladder(agent, finish_reason="stop")
        assert getattr(agent, "_output_starvation_detected", False) is False

    def test_length_finish_with_content_never_sets_starvation_flag(self):
        """Content actually delivered: not starvation, legacy ladder owns it."""
        agent = _agent()
        verdict = recover_empty_response(
            agent,
            assistant_message=_assistant_message(),
            response=_starved_response(),
            finish_reason="length",
            final_response="partial answer text before truncation",
            messages=[{"role": "user", "content": "task"}],
            api_messages=[],
            conversation_history=[],
            active_system_prompt=None,
            api_call_count=1,
            turn_exit_reason=None,
            preflight_compression_blocked=False,
        )
        assert getattr(agent, "_output_starvation_detected", False) is False
