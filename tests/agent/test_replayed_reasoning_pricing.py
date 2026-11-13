"""Replayed encrypted reasoning is priced at the provider-reported count (#125920).

The reporter's shape: an xAI/Codex session compacted once, then one long tool-calling turn whose
visible text is tiny while every tool round replays thousands of tokens of encrypted reasoning.
Priced at 0, that whole turn fit inside the protected tail, so the planned summary window held only
the earlier handoff row: compression found nothing to cut (and armed the structural backoff) while
the real bill sat over the window. Priced at the stamped count, the cut lands inside the active turn
and its older tool rounds (with their reasoning) are summarized.

Each row carries ITS OWN response's count (a truncated fragment built before the usage fold never
inherits the previous response's), the count survives into the per-call request estimate without
reaching the wire, and Claude signed thinking is charged only where conversion keeps the signature.
"""
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from agent.message_metadata import stamp_reasoning_tokens, stamped_reasoning_tokens


def _compacted_then_long_tool_turn(rounds: int = 12, reasoning_tokens: int = 8_000) -> list[dict]:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[CONTEXT COMPACTION] earlier work summarized."},
        {"role": "assistant", "content": "ok, continuing."},
        {"role": "user", "content": "Audit every file in the repo and report the totals."},
    ]
    for i in range(rounds):
        assistant = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "read_file", "arguments": f'{{"path": "f{i}"}}'}}],
            "codex_reasoning_items": [{"type": "reasoning", "encrypted_content": "x" * 400}],
        }
        stamp_reasoning_tokens(assistant, reasoning_tokens)
        messages += [assistant, {"role": "tool", "tool_call_id": f"c{i}", "content": f"value={i}"}]
    return messages


def test_window_reaches_into_a_reasoning_heavy_active_turn():
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        c = ContextCompressor(model="grok-4.7", threshold_percent=0.5, protect_first_n=1,
                              protect_last_n=2, quiet_mode=True)
        _ = c.context_length
    messages = _compacted_then_long_tool_turn()
    start, end = c._compress_window(messages)
    # ~96k of replayed reasoning against a 10k tail budget: most tool rounds must be summarizable.
    summarizable_rounds = sum(1 for m in messages[start:end] if m.get("codex_reasoning_items"))
    assert summarizable_rounds >= 8, f"window {start}-{end}: only {summarizable_rounds} rounds summarizable"


def test_each_row_carries_its_own_responses_count_into_the_request_estimate_not_onto_the_wire():
    """A real turn loop: a response reporting 8,000 reasoning tokens, then a TRUNCATED one reporting 73
    (its fragment is built before the usage fold), then the finish. The fragment is stamped 73, the
    next request's estimate sees the stamps, and no request-only key reaches the provider."""
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    def usage(reasoning):
        return {"prompt_tokens": 100, "completion_tokens": reasoning + 5, "total_tokens": reasoning + 105,
                "completion_tokens_details": {"reasoning_tokens": reasoning}}

    with (
        patch("hermes_agent.model_tools.get_tool_definitions", return_value=[]),
        patch("hermes_agent.model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = agent.compression_enabled = agent.save_trajectories = False
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[_mock_tool_call("unknown_tool", "{}", "c1")], usage=usage(8_000)),
        _mock_response(content="Partial answer", finish_reason="length", usage=usage(73)),
        _mock_response(content=" and the rest.", usage=usage(0)),
    ]
    estimates = []
    from agent import model_metadata
    original = model_metadata.estimate_messages_tokens_rough

    def spy(messages, **kwargs):
        estimates.append(sum(stamped_reasoning_tokens(m) for m in messages))
        return original(messages, **kwargs)

    # assemble_api_request imports the estimator inside the function, so the module attribute is the seam.
    with (
        patch("agent.model_metadata.estimate_messages_tokens_rough", side_effect=spy),
        patch.object(agent, "_persist_session"), patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("go")

    stamps = [stamped_reasoning_tokens(m) for m in result["messages"] if m.get("role") == "assistant"]
    assert stamps[0] == 8_000 and 73 in stamps and stamps.count(8_000) == 1, stamps
    assert max(estimates) >= 8_000, f"the request estimate lost the stamped counts: {estimates}"
    for call in agent.client.chat.completions.create.call_args_list:
        assert not any("_replayed_reasoning_tokens" in m for m in call.kwargs["messages"])


def test_replaced_reasoning_drops_the_old_stamp():
    row = {"role": "assistant", "content": ""}
    stamp_reasoning_tokens(row, 8_000)
    stamp_reasoning_tokens(row, 0)  # the new payload's count is unknown
    assert stamped_reasoning_tokens(row) == 0


def test_claude_thinking_is_charged_only_where_conversion_keeps_the_signature():
    from agent.anthropic_message_convert import convert_messages_to_anthropic
    from agent.message_sanitization import native_anthropic_accounting_projection

    def signed(text, tool_id=None):
        row = {"role": "assistant", "content": text,
               "reasoning_details": [{"type": "thinking", "thinking": "brief", "signature": "sig" * 40}]}
        if tool_id:
            row["tool_calls"] = [{"id": tool_id, "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        stamp_reasoning_tokens(row, 8_000)
        return row

    orphaned = [{"role": "user", "content": "q"}, signed("a", tool_id="t1"), {"role": "user", "content": "next"}]
    merged = [{"role": "user", "content": "q"}, signed("a"), signed("b"), {"role": "user", "content": "next"}]
    for history, kept in ((orphaned, 0), (merged, 1)):
        _, wire = convert_messages_to_anthropic(history, base_url="https://api.anthropic.com", model="claude-opus-5-5")
        sent = sum(1 for m in wire for b in m["content"] if isinstance(b, dict) and b.get("signature"))
        assert sent == kept
        _, thinking = native_anthropic_accounting_projection(history)
        assert sum(t >= 8_000 for t in thinking) == kept, thinking
