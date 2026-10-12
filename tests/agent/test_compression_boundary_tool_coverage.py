"""Tool output replaced in the protected head/tail must reach the summarizer too.

The summarizable middle is picked AFTER pruning. A pressure-demoted head or
old lean-tail result can therefore fall outside that middle. Keeping only its
stub without summarizing its body silently loses facts at the boundary.
"""
import copy
from types import SimpleNamespace
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _response():
    message = SimpleNamespace(content="## Goal\nPublic fixture checkpoint.", refusal=None,
                              reasoning_content=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)


def _workload():
    messages: list[dict] = [{"role": "system", "content": "Public fictional fixture (S0)."}]
    for index in range(9):
        call_id = f"read-{index}"
        body = "a" * 17_003 + f" UNIQUE_FACT_{index}_MIDDLE " + "b" * 63_019 + f" UNIQUE_FACT_{index}_END "
        messages.extend([
            {"role": "user", "content": f"Read public fixture {index}."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function", "function": {
                    "name": "read_file", "arguments": '{"path":"fixture-' + str(index) + '.txt"}'}}]},
            {"role": "tool", "tool_call_id": call_id, "name": "read_file", "content": body},
            {"role": "assistant", "content": "Read."},
        ])
    messages.extend([{"role": "user", "content": "Continue the current public task."},
                     {"role": "assistant", "content": "Understood."}])
    return messages


def test_every_demoted_body_reaches_summary_even_outside_the_middle(monkeypatch):
    # All users in this fixture are real; do not load the unrelated platform loop.
    monkeypatch.setattr(ContextCompressor, "_is_synthetic_compression_user_turn",
                        classmethod(lambda cls, message: False))
    compressor = ContextCompressor(model="test/model", config_context_length=300_000,
                                   protect_first_n=3, protect_last_n=15,
                                   proactive_prune_tokens=0, quiet_mode=True)
    messages = _workload()
    original = copy.deepcopy(messages)
    with patch("agent.context_compressor.call_llm", return_value=_response()) as call:
        compressed = compressor.compress(messages, current_tokens=260_000, force=True)
    assert call.call_count > 0
    assert compressor._previous_summary and "Public fixture checkpoint" in compressor._previous_summary
    assert any("Public fixture checkpoint" in str(m.get("content")) for m in compressed)
    assert len(compressed) < len(messages), "coverage must come from an adopted compression, not an abort"
    prompts = "\n".join(c.kwargs["messages"][0]["content"] for c in call.call_args_list)
    intact = "\n".join(m["content"] for m in compressed if m.get("role") == "tool")
    retained_ids = {m.get("tool_call_id") for m in compressed if m.get("role") == "tool"}
    demoted = [m for m in messages if m.get("role") == "tool"
               and m.get("tool_call_id") in retained_ids and m["content"] not in intact]
    assert demoted, "the protected head/tail must actually contain demoted tool results"
    assert all(f"UNIQUE_FACT_{m['tool_call_id'].split('-')[-1]}_END" in prompts for m in demoted)
    for index in range(9):
        for marker in (f"UNIQUE_FACT_{index}_MIDDLE", f"UNIQUE_FACT_{index}_END"):
            assert marker in prompts or marker in intact, f"{marker} disappeared without reaching the summary"
    assert messages == original
    assert compressed[-1]["content"] == messages[-1]["content"]
