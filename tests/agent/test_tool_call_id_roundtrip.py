"""Regression: a tool call whose response-item id is ``fc_…`` must replay with the SAME
call_id as its tool output.

Contract (not a snapshot): ``_replay_tool_call_items`` and ``_tool_output_items`` both
derive the id they send from the SAME stored pair. ``_assistant_tool_call_dict`` stores
the pair, so it must not hand one side a raw ``fc_…`` item id while the other side
canonicalizes it to ``call_…`` — DeepSeek rejects that with
``HTTP 400: No tool call found for tool output with call_id``, which is non-retryable and
sends every turn of the session to the fallback model.
"""
from agent.chat_completion_helpers import _assistant_tool_call_dict
from agent.codex_responses_adapter import (
    _canonical_call_id_from_fc,
    _derive_responses_function_call_id,
    _replay_tool_call_items,
    _split_responses_tool_id,
    _tool_output_items,
    _WireCallIds,
)

FC_ID = "fc_251e80a9-a21f-4be3-ba34-6efde31d1ab0"


class _Fn:
    name = "execute_code"
    arguments = '{"code": "x"}'


class _ToolCall:
    """Responses-shaped tool call: only ``id`` is set (the item id arrives without call_id)."""

    def __init__(self, item_id: str):
        self.id = item_id
        self.call_id = None
        self.response_item_id = item_id
        self.type = "function"
        self.function = _Fn()


class _Agent:
    """Only the helpers ``_assistant_tool_call_dict`` reaches for."""

    _split_responses_tool_id = staticmethod(_split_responses_tool_id)
    _derive_responses_function_call_id = staticmethod(_derive_responses_function_call_id)
    _canonical_call_id_from_fc = staticmethod(_canonical_call_id_from_fc)
    _deterministic_call_id = staticmethod(lambda *args: "det")


def _wire_ids_for(tool_call_item_id: str) -> tuple[str, str]:
    """The ids the assistant side and the tool side send for one stored call."""
    stored = _assistant_tool_call_dict(_Agent(), _ToolCall(tool_call_item_id), 0)
    assistant_wire = _replay_tool_call_items(
        {"role": "assistant", "tool_calls": [stored]}, start_index=0, wire_ids=_WireCallIds(),
    )[0]["call_id"]
    tool_wire = _tool_output_items(
        {"role": "tool", "tool_call_id": tool_call_item_id, "content": "ok"}, wire_ids=_WireCallIds(),
    )[0]["call_id"]
    return assistant_wire, tool_wire


def test_fc_item_id_replays_with_matching_call_id_on_both_sides():
    assistant_wire, tool_wire = _wire_ids_for(FC_ID)
    assert assistant_wire == tool_wire, (
        "assistant declared %r but the tool output answered %r — the replayed pair is "
        "misaligned and strict providers reject the request" % (assistant_wire, tool_wire)
    )


def test_canonical_call_id_is_preserved_verbatim():
    """A plain ``call_…`` id (the normal chat-completions shape) must pass through."""
    call_id = "call_7fc246be-03cb-4f8c-8bf8-46cd87ac17ec"
    assistant_wire, _ = _wire_ids_for(call_id)
    assert assistant_wire == call_id
