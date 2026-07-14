"""Unit tests for AG-UI <-> Hermes translation. No agent/provider needed."""

from ag_ui.core import (
    AssistantMessage,
    Context,
    FunctionCall,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from agui_adapter.session import RunState, StateWriterSpec
from agui_adapter.translate import (
    agui_messages_to_hermes,
    agui_tools_to_openai,
    context_to_text,
    forwarded_props_to_text,
    frontend_tool_names,
    parse_state_writer_props,
    prepare_run,
    state_to_text,
)


def _user(text, id="u1"):
    return UserMessage(id=id, role="user", content=text)


def _assistant_toolcall(name, args, tc_id="tc1", msg_id="a1"):
    return AssistantMessage(
        id=msg_id,
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=tc_id, type="function", function=FunctionCall(name=name, arguments=args))],
    )


def test_message_roundtrip_shapes():
    msgs = [
        _user("hello"),
        _assistant_toolcall("change_background", '{"background":"red"}'),
        ToolMessage(id="t1", role="tool", content="ok", tool_call_id="tc1"),
    ]
    hermes = agui_messages_to_hermes(msgs)
    assert hermes[0] == {"role": "user", "content": "hello"}
    assert hermes[1]["role"] == "assistant"
    assert hermes[1]["tool_calls"][0] == {
        "id": "tc1",
        "type": "function",
        "function": {"name": "change_background", "arguments": '{"background":"red"}'},
    }
    assert hermes[2] == {"role": "tool", "tool_call_id": "tc1", "content": "ok"}


def test_prepare_run_fresh_turn():
    msgs = [_user("first"), _assistant_toolcall("x", "{}"), ToolMessage(id="t", role="tool", content="r", tool_call_id="tc1"), _user("second", id="u2")]
    prep = prepare_run(msgs)
    # Tail is a user turn -> fresh run, user text pulled out, rest is history.
    assert prep.user_message == "second"
    assert prep.is_resume is False
    assert prep.conversation_history[-1]["role"] == "tool"
    assert all(not (m["role"] == "user" and m["content"] == "second") for m in prep.conversation_history)


def test_prepare_run_resume_after_tool():
    # Tail is a tool result -> resume: keep the whole history (incl. the tool
    # result) and reuse the original user text (run_conversation re-appends it).
    msgs = [
        _user("change bg to forest"),
        _assistant_toolcall("change_background", '{"background":"forest"}'),
        ToolMessage(id="t1", role="tool", content='{"status":"success"}', tool_call_id="tc1"),
    ]
    prep = prepare_run(msgs)
    assert prep.user_message == "change bg to forest"
    assert prep.is_resume is True
    # Full history retained, ending in the tool result.
    assert prep.conversation_history[-1] == {
        "role": "tool",
        "tool_call_id": "tc1",
        "content": '{"status":"success"}',
    }


def test_context_injected_as_leading_system_not_in_user():
    msgs = [_user("who am i")]
    ctx = context_to_text([Context(description="display name", value="Ada")])
    prep = prepare_run(msgs, context_text=ctx)
    # Context lands as a system message in history, user message is untouched
    # (critical for aimock user-message fixture matching).
    assert prep.user_message == "who am i"
    assert prep.conversation_history[0]["role"] == "system"
    assert "Ada" in prep.conversation_history[0]["content"]


def test_tools_conversion_and_names():
    tools = [Tool(name="change_background", description="set bg", parameters={"type": "object", "properties": {"background": {"type": "string"}}})]
    schemas = agui_tools_to_openai(tools)
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "change_background"
    assert schemas[0]["function"]["parameters"]["properties"]["background"]["type"] == "string"
    assert frontend_tool_names(tools) == {"change_background"}


def test_multimodal_user_content_extracts_text():
    # content as a list of typed blocks -> text collapsed.
    msg = UserMessage(id="u", role="user", content=[{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}])
    hermes = agui_messages_to_hermes([msg])
    assert hermes[0]["content"] == "hi there"


# --- feature 1: forwarded_props -> agent config ----------------------------


def test_forwarded_props_renders_stable_sorted_skips_empty():
    text = forwarded_props_to_text(
        {"tone": "formal", "expertise": "expert", "responseLength": "", "extra": None}
    )
    # Empty/None skipped; keys sorted for determinism.
    assert "responseLength" not in text
    assert "extra" not in text
    assert text.index("expertise") < text.index("tone")
    assert "- tone: formal" in text
    assert "- expertise: expert" in text


def test_forwarded_props_empty_is_blank():
    assert forwarded_props_to_text({}) == ""
    assert forwarded_props_to_text(None) == ""


def test_forwarded_props_injected_as_system_not_in_user():
    msgs = [_user("hi")]
    props = forwarded_props_to_text({"tone": "formal"})
    prep = prepare_run(msgs, system_texts=[props])
    assert prep.user_message == "hi"
    sys_contents = [m["content"] for m in prep.conversation_history if m["role"] == "system"]
    assert any("tone: formal" in c for c in sys_contents)


# --- feature 2: inbound state -> injected context --------------------------


def test_state_to_text_renders_json():
    text = state_to_text({"recipe": {"title": "Pie", "servings": 4}})
    assert text.startswith("Current shared state:")
    assert "Pie" in text and "servings" in text


def test_state_to_text_empty_is_blank():
    assert state_to_text({}) == ""
    assert state_to_text(None) == ""


def test_state_injected_as_system_not_in_user():
    msgs = [_user("what am i cooking")]
    prep = prepare_run(msgs, system_texts=[state_to_text({"recipe": "Pie"})])
    assert prep.user_message == "what am i cooking"
    sys_contents = [m["content"] for m in prep.conversation_history if m["role"] == "system"]
    assert any("Current shared state" in c and "Pie" in c for c in sys_contents)


# --- feature 4: state-writer tool declaration + run-scoped state store ------


def test_parse_state_writer_props_list_form():
    specs, schemas = parse_state_writer_props(
        {
            "stateWriterTools": [
                {
                    "name": "set_notes",
                    "stateKey": "notes",
                    "arg": "notes",
                    "description": "Replace the notes.",
                    "parameters": {"type": "object", "properties": {"notes": {"type": "array"}}},
                }
            ]
        }
    )
    assert set(specs) == {"set_notes"}
    assert specs["set_notes"].state_key == "notes"
    assert specs["set_notes"].arg == "notes"
    assert specs["set_notes"].mode == "replace"
    assert schemas[0]["function"]["name"] == "set_notes"
    assert schemas[0]["function"]["parameters"]["properties"]["notes"]["type"] == "array"


def test_parse_state_writer_props_mapping_form_and_append_mode():
    specs, _ = parse_state_writer_props(
        {"stateWriterTools": {"append_delegation": {"stateKey": "delegations", "mode": "append"}}}
    )
    assert specs["append_delegation"].state_key == "delegations"
    assert specs["append_delegation"].mode == "append"
    assert specs["append_delegation"].arg is None


def test_parse_state_writer_props_unknown_mode_clamps_to_replace():
    # An unknown/garbage client `mode` is normalized to the Literal domain at
    # the parse boundary: only the exact string "append" is honored, everything
    # else (incl. case variants) collapses to "replace".
    specs, _ = parse_state_writer_props(
        {"stateWriterTools": {"x": {"stateKey": "k", "mode": "prepend"}}}
    )
    assert specs["x"].mode == "replace"
    specs2, _ = parse_state_writer_props(
        {"stateWriterTools": {"y": {"stateKey": "k", "mode": "APPEND"}}}
    )
    assert specs2["y"].mode == "replace"


def test_parse_state_writer_props_empty():
    assert parse_state_writer_props({}) == ({}, [])
    assert parse_state_writer_props(None) == ({}, [])
    assert parse_state_writer_props({"stateWriterTools": []}) == ({}, [])


def test_run_state_replace_merges_arg_into_key_and_keeps_seed():
    # Seeded with a UI-set key; a set_notes call adds the agent-written key.
    rs = RunState(
        state={"preferences": {"tone": "casual"}},
        specs={"set_notes": StateWriterSpec(state_key="notes", arg="notes")},
    )
    snap = rs.apply("set_notes", {"notes": ["a", "b"]})
    assert snap == {"preferences": {"tone": "casual"}, "notes": ["a", "b"]}
    # A second call replaces (last write wins).
    snap2 = rs.apply("set_notes", {"notes": ["c"]})
    assert snap2["notes"] == ["c"]
    # Snapshots are independent copies (first snapshot not mutated by later call).
    assert rs.snapshots[0]["notes"] == ["a", "b"]


def test_run_state_append_mode_accumulates_list():
    rs = RunState(specs={"deleg": StateWriterSpec(state_key="delegations", mode="append")})
    rs.apply("deleg", {"sub_agent": "research_agent", "task": "x"})
    snap = rs.apply("deleg", {"sub_agent": "writing_agent", "task": "y"})
    assert [d["sub_agent"] for d in snap["delegations"]] == ["research_agent", "writing_agent"]


def test_run_state_no_arg_merges_args_dict_into_key():
    rs = RunState(specs={"write_doc": StateWriterSpec(state_key="document", arg="document")})
    snap = rs.apply("write_doc", {"document": "hello world"})
    assert snap["document"] == "hello world"


# --- feature 3: multimodal image passthrough -------------------------------


def test_multimodal_image_passes_through_as_content_parts():
    msg = UserMessage(
        id="u",
        role="user",
        content=[
            {"type": "text", "text": "what is this"},
            {"type": "image", "source": {"type": "url", "value": "https://x/y.png"}},
        ],
    )
    hermes = agui_messages_to_hermes([msg])
    content = hermes[0]["content"]
    assert isinstance(content, list)
    assert {"type": "text", "text": "what is this"} in content
    assert {"type": "image_url", "image_url": {"url": "https://x/y.png"}} in content


def test_multimodal_data_source_becomes_data_uri():
    msg = UserMessage(
        id="u",
        role="user",
        content=[{"type": "image", "source": {"type": "data", "value": "QUJD", "mimeType": "image/png"}}],
    )
    hermes = agui_messages_to_hermes([msg])
    content = hermes[0]["content"]
    assert content == [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]


def test_pure_text_message_stays_plain_string():
    msg = UserMessage(id="u", role="user", content="just text")
    assert agui_messages_to_hermes([msg])[0]["content"] == "just text"


def test_prepare_run_fresh_turn_with_image_keeps_parts():
    msg = UserMessage(
        id="u",
        role="user",
        content=[
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "url", "value": "https://x/y.png"}},
        ],
    )
    prep = prepare_run([msg])
    assert isinstance(prep.user_message, list)
    assert any(p.get("type") == "image_url" for p in prep.user_message)
