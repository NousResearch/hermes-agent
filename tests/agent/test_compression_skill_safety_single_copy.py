"""A compacted transcript carries the Skill Safety Rule exactly once.

Prune-only attaches the rule to a ``skill_view`` prune stub (it has no handoff to ride) and protects the tail by row
count; a later full compaction keeps the tail by token budget, so that ruled stub can survive beside a handoff that
gets the rule too.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import SKILL_PRUNED_MARKER_PREFIX, ContextCompressor
from agent.context_compressor_tail import SKILL_SAFETY_RULE


def _texts(messages):
    return "\n".join(str(m.get("content") or "") for m in messages)


def _transcript():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(30):
        messages += [{"role": "user", "content": f"q{i} " + "facts " * 300},
                     {"role": "assistant", "content": f"a{i} " + "detail " * 1500}]
    messages.append({"role": "user", "content": "build the pdf"})
    calls = [("skill_view", '{"name":"pdf"}', "P" * 9000)]
    calls += [("terminal", '{"cmd":"ls"}', ch * 9000) for ch in "AB"]
    calls += [("terminal", '{"cmd":"ls"}', "ok")] * 5
    for i, (tool, args, content) in enumerate(calls):
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": tool, "arguments": args}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": content},
        ]
    messages += [{"role": "assistant", "content": "done"}, {"role": "user", "content": "latest ask"}]
    return messages


def test_prune_only_then_full_compaction_keeps_one_skill_safety_rule():
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        comp = ContextCompressor(
            model="test/model", quiet_mode=True, threshold_percent=0.5, protect_first_n=2, protect_last_n=4,
            proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000,
        )
    pruned, count = comp.prune_tool_results_only(_transcript(), current_tokens=120_000)
    assert count and _texts(pruned).count(SKILL_SAFETY_RULE) == 1

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "## Goal\nOld chat.\n\n## Completed Actions\n1. talked."
    with patch("agent.context_compressor.call_llm", return_value=response):
        compacted = comp.compress(pruned, force=True)

    assert SKILL_PRUNED_MARKER_PREFIX in _texts(compacted)
    assert _texts(compacted).count(SKILL_SAFETY_RULE) == 1
