"""Seeded session instructions survive wire conversion and a durable-session resume."""

from copy import deepcopy

from agent.anthropic_message_convert import convert_messages_to_anthropic
from agent.transports.codex import ResponsesApiTransport
from agent.turn_context import build_api_messages
from hermes_state import SessionDB


class _Agent:
    ephemeral_system_prompt = "Native profile overlay."
    _current_turn_timestamp = 10_000.0
    _copy_reasoning_content_for_api = staticmethod(lambda source, target: None)
    _should_sanitize_tool_calls = staticmethod(lambda: False)


def test_seeded_system_instructions_survive_resume_without_changing_history(tmp_path):
    native, seed, extra = "Native agent instructions.", "Render answers with ui4a/tsx.\n", "Save canvases in .artifacts/canvases/.\n"
    db_path = tmp_path / "sessions.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id="seeded", source="tui", system_prompt=native)
    for content in (seed, extra):
        db.append_message("seeded", role="system", content=content, display_kind="hidden")
    db.append_message("seeded", role="user", content="Compare the options.")
    db.append_message("seeded", role="assistant", content="Here is the comparison.")
    db.close()
    db = SessionDB(db_path=db_path)
    history = db.get_messages_as_conversation("seeded")
    db.close()
    assert history[0]["role"] == "system" and history[0]["display_kind"] == "hidden"
    history.append({"role": "user", "content": "Add the cost."})
    frozen = deepcopy(history)
    request, effective = build_api_messages(_Agent(), history, current_turn_user_idx=len(history) - 1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt=native)
    expected = "\n\n".join((native, _Agent.ephemeral_system_prompt, seed, extra))
    transport = ResponsesApiTransport()
    for explicit in ("", effective, "Explicit native override."):
        kwargs = transport.build_kwargs(model="gpt-6-astra", messages=request, instructions=explicit)
        assert kwargs["instructions"] == (expected if explicit in ("", effective) else "\n\n".join((explicit, seed, extra)))
        assert [item["role"] for item in kwargs["input"]] == ["user", "assistant", "user"]
        assert kwargs["input"][-1]["content"] == "Add the cost."
        assert seed not in str(kwargs["input"])
        assert transport.build_kwargs(model="gpt-6-astra", messages=request, instructions=explicit) == kwargs
    system, messages = convert_messages_to_anthropic(request)
    assert system == expected
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert seed not in str(messages)
    assert history == frozen


def test_system_blocks_keep_cache_breakpoints_and_single_prompt_behavior():
    blocks = [{"type": "text", "text": "Native cached prefix.", "cache_control": {"type": "ephemeral"}},
              {"type": "text", "text": "Native tail."}]
    base = [{"role": "system", "content": blocks}, {"role": "user", "content": "Hello"}]
    frozen = deepcopy(base)
    assert convert_messages_to_anthropic(base)[0] == blocks
    seeded = [base[0], {"role": "system", "content": "Inline UI."},
              {"role": "system", "content": [{"type": "text", "text": "Persistent canvas."}]}, base[1]]
    system, _ = convert_messages_to_anthropic(seeded)
    assert system == [*blocks, {"type": "text", "text": "Inline UI."}, {"type": "text", "text": "Persistent canvas."}]
    transport = ResponsesApiTransport()
    assert transport.build_kwargs(model="gpt-6-astra", messages=seeded)["instructions"] == "Native cached prefix.\nNative tail.\n\nInline UI.\n\nPersistent canvas."
    single = [{"role": "system", "content": "Native only."}, base[1]]
    assert transport.build_kwargs(model="gpt-6-astra", messages=single)["instructions"] == "Native only."
    assert convert_messages_to_anthropic(single)[0] == "Native only."
    assert base == frozen

    # Explicit native overrides do not deduplicate independently supplied seed rows.
    repeated = [{"role": "system", "content": "Replaced native."},
                {"role": "system", "content": "Explicit."}, None, base[1]]
    kwargs = transport.build_kwargs(model="gpt-6-astra", messages=repeated, instructions="Explicit.")
    assert kwargs["instructions"] == "Explicit.\n\nExplicit."
    assert kwargs["input"] == [{"role": "user", "content": "Hello"}]
    mixed = [{"role": "system", "content": [{"type": "input_text", "text": "Text guidance."},
              {"type": "input_image", "image_url": "https://example.test/ignored.png"}]}, base[1]]
    assert transport.build_kwargs(model="gpt-6-astra", messages=mixed)["instructions"] == "Text guidance."
