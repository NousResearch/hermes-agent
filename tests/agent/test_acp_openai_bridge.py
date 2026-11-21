"""The ACP text bridge is what makes Hermes' own tools reachable on an ACP provider.

ACP has no OpenAI ``tools``/``tool_calls`` channel, so ``memory``,
``skill_manage``, ``todo`` and friends only work if the schemas travel into the
prompt as text and the calls are parsed back out of the response text. These
tests pin both halves plus the streaming shape, and check that the in-tree
consumer (``agent/copilot_acp_client.py``) still produces the same prompt it did
when it owned a private copy of this code.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent.acp_openai_bridge import (
    StreamChunks,
    _extract_tool_call_blocks,
    _find_matching_brace,
    completion_to_stream_chunks,
    extract_tool_calls_from_text,
    render_tool_bridge_sections,
    tool_specs_from_openai_tools,
)

_TOOLS = [
    {"type": "function", "function": {"name": "memory", "description": "d1", "parameters": {"a": 1}}},
    {"type": "function", "function": {"name": "read_file", "description": "d2", "parameters": {}}},
]


# ── prompt side ──────────────────────────────────────────────────────────────


def test_specs_are_flattened_and_malformed_entries_skipped():
    specs = tool_specs_from_openai_tools(
        [*_TOOLS, "junk", None, {"function": None}, {"function": {"name": "  "}}]
    )
    assert [s["name"] for s in specs] == ["memory", "read_file"]
    assert specs[0] == {"name": "memory", "description": "d1", "parameters": {"a": 1}}


def test_allowlist_forwards_only_the_named_tools():
    """An agent-as-provider runs its own read/edit tools; re-offering them would
    make Hermes re-run finished work, so those clients forward an allowlist."""
    specs = tool_specs_from_openai_tools(_TOOLS, allowlist=["memory"])
    assert [s["name"] for s in specs] == ["memory"]
    # No allowlist at all means "forward everything" — not "forward nothing".
    assert len(tool_specs_from_openai_tools(_TOOLS)) == 2
    assert tool_specs_from_openai_tools(_TOOLS, allowlist=[]) == []


def test_rendered_sections_carry_the_contract_and_the_schemas():
    sections = render_tool_bridge_sections(_TOOLS, {"type": "function"})
    assert len(sections) == 2
    assert "<tool_call>" in sections[0]
    payload = json.loads(sections[0].split("\n", 1)[1])
    assert [s["name"] for s in payload] == ["memory", "read_file"]
    assert sections[1].startswith("Tool choice hint:")


def test_no_tools_and_no_choice_render_nothing():
    """Callers splice the result unconditionally, so it must be safe to be empty."""
    assert render_tool_bridge_sections(None) == []
    assert render_tool_bridge_sections([]) == []
    assert render_tool_bridge_sections([{"function": {}}]) == []


# ── response side ────────────────────────────────────────────────────────────


def test_tool_call_block_is_parsed_and_stripped_from_the_text():
    calls, cleaned = extract_tool_calls_from_text(
        'Sure.\n<tool_call>{"id": "c1", "type": "function", "function": '
        '{"name": "memory", "arguments": "{\\"action\\": \\"add\\"}"}}</tool_call>\nDone.'
    )
    assert [c.function.name for c in calls] == ["memory"]
    assert calls[0].id == "c1"
    assert json.loads(calls[0].function.arguments) == {"action": "add"}
    # The user must not see the raw JSON.
    assert "<tool_call>" not in cleaned
    assert cleaned == "Sure.\nDone."


def test_multiple_blocks_are_all_parsed():
    text = "".join(
        f'<tool_call>{{"id": "c{i}", "type": "function", '
        f'"function": {{"name": "todo", "arguments": "{{}}"}}}}</tool_call>'
        for i in range(3)
    )
    calls, cleaned = extract_tool_calls_from_text(text)
    assert [c.id for c in calls] == ["c0", "c1", "c2"]
    assert cleaned == ""


def test_non_string_arguments_are_json_encoded_and_missing_ids_synthesised():
    calls, _ = extract_tool_calls_from_text(
        '<tool_call>{"type": "function", "function": '
        '{"name": "todo", "arguments": {"op": "list"}}}</tool_call>'
    )
    assert calls[0].function.arguments == '{"op": "list"}'
    assert calls[0].id == "acp_call_1"


def test_bare_json_is_a_fallback_only_when_no_block_matched():
    bare = '{"id": "c9", "type": "function", "function": {"name": "memory", "arguments": "{}"}}'
    calls, cleaned = extract_tool_calls_from_text(f"before {bare} after")
    assert [c.id for c in calls] == ["c9"]
    assert cleaned == "before\nafter"

    # With a real block present the bare-JSON scan must not double-count.
    both = f'<tool_call>{bare}</tool_call> and {bare}'
    calls, _ = extract_tool_calls_from_text(both)
    assert len(calls) == 1


def test_malformed_and_empty_input_never_raises():
    assert extract_tool_calls_from_text("") == ([], "")
    assert extract_tool_calls_from_text(None) == ([], "")
    calls, cleaned = extract_tool_calls_from_text("<tool_call>{not json}</tool_call>plain")
    assert calls == []
    assert cleaned == "plain"
    # Well-formed JSON that isn't a tool call is ignored, text preserved.
    calls, cleaned = extract_tool_calls_from_text('<tool_call>{"function": 5}</tool_call>hi')
    assert calls == []
    assert cleaned == "hi"


# ── streaming shape ──────────────────────────────────────────────────────────


def _completion(**extras):
    message = SimpleNamespace(
        content="hello",
        tool_calls=[
            SimpleNamespace(
                id="c1", type="function",
                function=SimpleNamespace(name="memory", arguments="{}"),
            )
        ],
        reasoning=None,
        reasoning_content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(total_tokens=3),
        model="acp",
        **extras,
    )


def test_stream_chunks_carry_the_delta_then_the_usage():
    chunks = completion_to_stream_chunks(_completion())
    assert len(chunks) == 2
    delta = chunks[0].choices[0].delta
    assert delta.content == "hello"
    assert delta.tool_calls[0].function.name == "memory"
    assert delta.tool_calls[0].index == 0
    assert chunks[0].choices[0].finish_reason == "tool_calls"
    # Usage arrives on its own trailing chunk, as OpenAI does it.
    assert chunks[0].usage is None
    assert chunks[1].usage.total_tokens == 3
    assert chunks[1].choices == []


def test_response_level_extras_survive_the_stream_conversion():
    """Hermes reads provider extras off the returned object; a plain list would
    drop them and silently disable the projection on stream=True."""
    chunks = completion_to_stream_chunks(
        _completion(hermes_projected_messages=[{"role": "tool", "content": "x"}])
    )
    assert isinstance(chunks, StreamChunks)
    assert isinstance(chunks, list)
    assert chunks.hermes_projected_messages == [{"role": "tool", "content": "x"}]


def test_a_text_only_completion_streams_without_tool_call_deltas():
    completion = _completion()
    completion.choices[0].message.tool_calls = []
    completion.choices[0].finish_reason = "stop"
    chunks = completion_to_stream_chunks(completion)
    assert chunks[0].choices[0].delta.tool_calls is None


# ── the in-tree consumer still speaks the same wire ──────────────────────────


def test_copilot_prompt_still_carries_the_contract_and_the_tools():
    """copilot-acp lost its private copy of the bridge; its prompt must not
    change shape."""
    from agent.acp_client import _format_messages_as_prompt

    prompt = _format_messages_as_prompt(
        [{"role": "user", "content": "hi"}], model="gpt-5", tools=_TOOLS,
    )
    assert "<tool_call>{...}</tool_call>" in prompt
    assert '"name": "memory"' in prompt
    assert '"name": "read_file"' in prompt  # copilot forwards everything
    assert "hi" in prompt


# ── nested JSON / in-string delimiters (PR #23888) ───────────────────────────
#
# The old lazy-regex extraction stopped at the first ``}`` that happened to be
# followed by ``</tool_call>``, so any brace inside a string value — or inside a
# nested ``arguments`` object — truncated the call and json.loads() failed. These
# pin the brace-depth scanner that replaced it.


def test_find_matching_brace_simple():
    text = '{"a": 1}'
    assert _find_matching_brace(text, 0) == len(text) - 1


def test_find_matching_brace_nested_objects():
    text = '{"a": {"b": {"c": 2}}}'
    assert _find_matching_brace(text, 0) == len(text) - 1


def test_find_matching_brace_ignores_braces_in_strings():
    text = '{"a": "}"}'
    assert _find_matching_brace(text, 0) == len(text) - 1


def test_find_matching_brace_ignores_close_tag_in_string():
    """An in-string ``</tool_call>`` must not be mistaken for the block end."""
    text = '{"a": "</tool_call>"}'
    assert _find_matching_brace(text, 0) == len(text) - 1


def test_find_matching_brace_handles_escaped_quotes():
    text = r'{"a": "\"}"}'
    assert _find_matching_brace(text, 0) == len(text) - 1


def test_find_matching_brace_unmatched_returns_minus_one():
    assert _find_matching_brace('{"a": 1', 0) == -1
    assert _find_matching_brace("", 0) == -1
    assert _find_matching_brace('{"a": 1}', 99) == -1


def test_extract_block_simple():
    text = '<tool_call>{"id": "c1"}</tool_call>'
    blocks = _extract_tool_call_blocks(text)
    assert len(blocks) == 1
    start, end, body = blocks[0]
    assert body == '{"id": "c1"}'
    assert text[start:end] == text


_NESTED = (
    '{"id": "c1", "type": "function", "function": {"name": "github_api", '
    '"arguments": {"query": "repo:owner/name } </tool_call> broken"}}}'
)


def test_extract_block_with_close_tag_inside_a_string_value():
    text = f"<tool_call>{_NESTED}</tool_call>"
    blocks = _extract_tool_call_blocks(text)
    assert len(blocks) == 1
    assert blocks[0][2] == _NESTED


def test_extract_block_with_nested_json():
    nested = '{"id": "c1", "function": {"name": "x", "arguments": {"a": {"b": 1}}}}'
    blocks = _extract_tool_call_blocks(f"<tool_call>{nested}</tool_call>")
    assert [b[2] for b in blocks] == [nested]


def test_extract_multiple_blocks():
    a = '{"id": "a"}'
    b = '{"id": "b", "nested": {"k": "}"}}'
    blocks = _extract_tool_call_blocks(f"<tool_call>{a}</tool_call> mid <tool_call>{b}</tool_call>")
    assert [blk[2] for blk in blocks] == [a, b]


def test_extract_no_blocks():
    assert _extract_tool_call_blocks("") == []
    assert _extract_tool_call_blocks("no calls here") == []
    assert _extract_tool_call_blocks("<tool_call>not json</tool_call>") == []


def test_extract_is_whitespace_tolerant():
    text = '<tool_call>\n  {"id": "c1"}\n</tool_call>'
    blocks = _extract_tool_call_blocks(text)
    assert [b[2] for b in blocks] == ['{"id": "c1"}']


def test_extract_requires_a_closing_tag():
    text = '<tool_call>{"id": "c1"}'
    assert _extract_tool_call_blocks(text) == []


def test_extract_handles_escaped_quotes_in_body():
    body = r'{"id": "c1", "function": {"name": "x", "arguments": "{\"a\": \"}\"}"}}'
    blocks = _extract_tool_call_blocks(f"<tool_call>{body}</tool_call>")
    assert [b[2] for b in blocks] == [body]


def test_pipeline_parses_a_call_whose_string_contains_the_close_tag():
    calls, cleaned = extract_tool_calls_from_text(f"<tool_call>{_NESTED}</tool_call>")
    assert len(calls) == 1
    assert calls[0].function.name == "github_api"
    assert json.loads(calls[0].function.arguments) == {
        "query": "repo:owner/name } </tool_call> broken"
    }
    assert cleaned == ""


def test_pipeline_nested_arguments_survive_without_id_or_type():
    """The bare-JSON fallback needs id+type, so this shape only ever worked if
    the XML pass itself was brace-aware."""
    calls, cleaned = extract_tool_calls_from_text(
        '<tool_call>{"function": {"name": "todo", "arguments": {"items": [{"a": "}"}]}}}</tool_call>'
    )
    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments) == {"items": [{"a": "}"}]}
    assert cleaned == ""


def test_pipeline_leaves_no_dangling_tags_or_braces():
    text = f"before<tool_call>{_NESTED}</tool_call>after"
    calls, cleaned = extract_tool_calls_from_text(text)
    assert len(calls) == 1
    assert cleaned == "before\nafter"
    for residue in ("<tool_call>", "</tool_call>", "}}"):
        assert residue not in cleaned


def test_pipeline_standard_text_round_trips():
    calls, cleaned = extract_tool_calls_from_text(
        'Hi\n<tool_call>{"id": "c1", "type": "function", "function": '
        '{"name": "memory", "arguments": "{}"}}</tool_call>\nbye'
    )
    assert [c.id for c in calls] == ["c1"]
    assert cleaned == "Hi\nbye"


def test_pipeline_empty_and_no_calls():
    assert extract_tool_calls_from_text("") == ([], "")
    calls, cleaned = extract_tool_calls_from_text("<tool_call>{bad}</tool_call>keep me")
    assert calls == []
    assert cleaned == "keep me"


def test_bare_tool_call_after_unbalanced_prose_brace():
    from agent.acp_openai_bridge import extract_tool_calls_from_text
    raw = '{"id":"later","type":"function","function":{"name":"example","arguments":{"nested":{"x":1}}}}'
    calls, cleaned = extract_tool_calls_from_text("unfinished { prose\n" + raw)
    assert [call.id for call in calls] == ["later"]
    assert cleaned == "unfinished { prose"
