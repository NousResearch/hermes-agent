"""Replayed encrypted reasoning is priced at the provider-reported count (#125920).

The reporter's shape: an xAI/Codex session compacted once, then one long tool-calling turn whose
visible text is tiny while every tool round replays thousands of tokens of encrypted reasoning.
Priced at 0, that whole turn fit inside the protected tail, so the planned summary window held only
the earlier handoff row: compression found nothing to cut (and armed the structural backoff) while
the real bill sat over the window. Priced at the stamped count, the cut lands inside the active turn
and its older tool rounds (with their reasoning) are summarized.
"""
from unittest.mock import patch

from agent.context_compressor import ContextCompressor
from agent.message_metadata import stamp_reasoning_tokens


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
