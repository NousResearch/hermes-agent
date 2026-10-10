"""MoA aggregator requests must grow as a byte-stable prefix across tool-loop iterations.

Issue #112358: the reference guidance used to be merged INTO a trailing user turn on
iteration 1 of every user turn, so iteration 2's ``user(task)`` byte-differed from the one
the provider had just cached and the prompt cache collapsed to the system prompt.
"""

from types import SimpleNamespace

from agent import moa_loop
from agent.anthropic_message_convert import convert_messages_to_anthropic


def test_attach_reference_guidance_never_mutates_the_trailing_user_turn():
    task = {"role": "user", "content": "ORIGINAL TASK"}
    messages = [{"role": "system", "content": "sys"}, task]
    moa_loop._attach_reference_guidance(messages, "REFERENCE BLOCK")

    assert messages[1] == {"role": "user", "content": "ORIGINAL TASK"}
    assert messages[-1] == {"role": "user", "content": "REFERENCE BLOCK"}
    assert moa_loop.peel_reference_guidance(messages, "REFERENCE BLOCK") == messages[:-1]


def test_prepared_aggregator_requests_share_a_byte_identical_prefix_across_iterations(monkeypatch):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", lambda **kw: calls.append(kw) or SimpleNamespace(choices=[]))
    monkeypatch.setattr(
        moa_loop, "_slot_runtime",
        lambda slot: {"provider": "nous", "model": "openai/gpt-6-astra", "api_mode": "chat_completions"},
    )
    facade = moa_loop.MoAChatCompletions.__new__(moa_loop.MoAChatCompletions)
    facade._pending_trace = None
    facade._agent = None
    aggregator = {"provider": "nous", "model": "openai/gpt-6-astra"}
    guidance = "[Mixture of Agents reference context]\nadvice"
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]

    # Iteration 1 ends on the user task; iteration 2 replays it plus the tool round.
    for messages in (
        history,
        [*history, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
         {"role": "tool", "tool_call_id": "1", "content": "result"}],
    ):
        prepared = facade.rebase_prepared_request({"guidance": guidance, "aggregator": aggregator,
                                                   "aggregator_temperature": None}, messages)
        facade._call_prepared_aggregator(prepared, {"tools": [{"type": "function", "function": {"name": "lookup"}}]})

    first, second = (c["messages"] for c in calls)
    assert second[: len(first) - 1] == first[:-1]
    assert second[-1] == first[-1] == {"role": "user", "content": guidance}


def test_anthropic_wire_keeps_the_task_block_byte_stable_with_guidance_as_its_own_block():
    """Anthropic Messages merges adjacent user turns; the merge must keep ``user(task)`` as an
    unchanged text block and add the guidance AFTER it, never fold both into one string
    (``"task\\n<guidance>"``) — otherwise the prefix diverges at the task on iteration 1 of every turn."""
    guidance = "[Mixture of Agents reference context]\nadvice"
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    iteration_1 = [*history]
    moa_loop._attach_reference_guidance(iteration_1, guidance)
    iteration_2 = [
        *history,
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    moa_loop._attach_reference_guidance(iteration_2, guidance)

    _, first = convert_messages_to_anthropic(iteration_1)
    _, second = convert_messages_to_anthropic(iteration_2)

    assert first[0]["content"] == [{"type": "text", "text": "task"}, {"type": "text", "text": guidance}]
    assert second[0]["content"] == "task"
    assert first[0]["content"][0]["text"] == second[0]["content"]
