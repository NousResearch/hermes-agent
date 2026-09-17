"""MoA aggregator requests must grow as a byte-stable prefix across tool-loop iterations.

Issue #112358: the reference guidance used to be merged INTO a trailing user turn on
iteration 1 of every user turn, so iteration 2's ``user(task)`` byte-differed from the one
the provider had just cached and the prompt cache collapsed to the system prompt.
"""

from types import SimpleNamespace

from agent import moa_loop


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
