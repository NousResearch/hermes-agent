"""#114663: a streamed tool call that arrived without an id must claim a real one at assembly.

The accumulator keeps ``""`` when a provider omits the id on its tool-call deltas. The
request builder synthesizes a deterministic id for the assistant row while the tool result
row kept ``""`` (the executor persists ``tool_call.id``), so a call and its result stopped
matching on replay. The id is now claimed once, at assembly, from the same policy owner and
with the same inputs the request builder uses.
"""

from agent.chat_completion_helpers import _StreamingCall, _assistant_tool_call_dict
from agent.message_sanitization import deterministic_call_id


class _Agent:
    """The id policy the real agent exposes (``staticmethod(message_sanitization.deterministic_call_id)``)."""

    _deterministic_call_id = staticmethod(deterministic_call_id)

    @staticmethod
    def _split_responses_tool_id(raw):
        return (raw or "", None)

    @staticmethod
    def _derive_responses_function_call_id(call_id, response_item_id=None):
        return "fc_" + call_id


def _acc(entries):
    return {
        i: {
            "id": tool_id,
            "type": "function",
            "function": {"name": name, "arguments": args},
            "extra_content": None,
        }
        for i, (tool_id, name, args) in enumerate(entries)
    }


def test_blank_streamed_id_is_filled_at_assembly():
    assembled, truncated = _StreamingCall._assemble_tool_calls(
        _Agent(), _acc([("", "terminal", '{"cmd":"ls"}')]), "tool_calls")

    assert truncated is False
    assert assembled[0].id == deterministic_call_id("terminal", '{"cmd":"ls"}', 0)
    assert assembled[0].id.strip()


def test_filled_id_matches_the_row_sent_upstream():
    """The tool result row persists ``tool_call.id``; it must equal what the wire carries."""

    assembled, _ = _StreamingCall._assemble_tool_calls(
        _Agent(), _acc([("", "terminal", '{"cmd":"ls"}')]), "tool_calls")
    wire = _assistant_tool_call_dict(_Agent(), assembled[0], 0)

    assert wire["id"] == assembled[0].id
    assert wire["id"].strip()


def test_real_id_is_kept_verbatim():
    assembled, _ = _StreamingCall._assemble_tool_calls(
        _Agent(), _acc([("call_real", "terminal", '{"cmd":"ls"}')]), "tool_calls")

    assert assembled[0].id == "call_real"


def test_positions_keep_multiple_blank_ids_distinct():
    assembled, _ = _StreamingCall._assemble_tool_calls(
        _Agent(), _acc([("", "terminal", "{}"), ("", "read_file", "{}")]), "tool_calls")

    assert assembled[0].id == deterministic_call_id("terminal", "{}", 0)
    assert assembled[1].id == deterministic_call_id("read_file", "{}", 1)
    assert assembled[0].id != assembled[1].id
