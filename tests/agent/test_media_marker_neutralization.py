"""llama.cpp media-marker regression tests (#108760; PR #108762).

llama-server randomizes its multimodal placeholder per process start
(``media_marker`` in ``/props``, e.g. ``<__media_CROZLhoWqzum1cQTzZQKMYA9gqWTHJ3E__>``)
so a prompt cannot forge one. Before any prompt is tokenized the server splits
it on that exact string and expects one attached bitmap per piece; a marker in
plain text with no image behind it makes ``mtmd`` throw and the request comes
back ``HTTP 400 {"message": "Failed to tokenize prompt"}`` in ~0.2s, before the
model runs. Retrying and failing over cannot help — every endpoint of the same
server shares the marker — so the whole session wedges.

A session poisons itself simply by reading the server's own ``/props`` into a
tool result, which is exactly how this was hit: one ``curl .../props`` dump
landed in the transcript and every subsequent turn 400'd.

The marker is neutralized on the API copy only, at the same chokepoint that
strips lone surrogates, so the stored transcript keeps what the tool actually
returned.
"""

import pytest

from agent.message_sanitization import (
    _neutralize_media_markers,
    _sanitize_messages_media_markers,
)

# The live marker from the 2026-09-12 incident, the fixed spelling older builds use,
# and the b8799-era image alias (same tokenizer 400).
RANDOM_MARKER = "<__media_CROZLhoWqzum1cQTzZQKMYA9gqWTHJ3E__>"
FIXED_MARKER = "<__media__>"
IMAGE_MARKER = "<__image__>"


@pytest.mark.parametrize("marker", [RANDOM_MARKER, FIXED_MARKER, IMAGE_MARKER])
def test_marker_is_removed_from_text(marker):
    out = _neutralize_media_markers(f"  media_marker: {marker}\n  model_alias: qwen-27b")
    assert marker not in out
    assert "model_alias: qwen-27b" in out


def test_non_marker_text_is_untouched():
    """Only the placeholder shape is rewritten; neighbouring angle-bracket
    tokens (chat-template specials quoted in a props dump) must survive."""
    text = "bos_token: <|endoftext|> eos_token: <|im_end|> <__mediaish__x>"
    assert _neutralize_media_markers(text) == text


def test_marker_neutralized_in_tool_result_content():
    messages = [
        {"role": "user", "content": "dump props"},
        {"role": "tool", "tool_call_id": "c1", "content": f"media_marker: {RANDOM_MARKER}"},
    ]
    assert _sanitize_messages_media_markers(messages) is True
    assert RANDOM_MARKER not in messages[1]["content"]
    assert messages[0]["content"] == "dump props"


def test_returns_false_when_no_marker_present():
    messages = [{"role": "user", "content": "no markers here"}]
    assert _sanitize_messages_media_markers(messages) is False


def test_auxiliary_kwargs_builder_neutralizes_markers():
    """Aux calls (title generation, compression, session search) assemble their own kwargs
    via _build_call_kwargs, bypassing assemble_api_request; the marker must be gone from
    the outgoing copy there too (review feedback on #108762)."""
    from agent.auxiliary_client import _build_call_kwargs

    messages = [
        {"role": "user", "content": "dump props"},
        {"role": "tool", "tool_call_id": "c1", "content": f"media_marker: {RANDOM_MARKER}"},
    ]
    kwargs = _build_call_kwargs("custom", "qwen-27b", messages, base_url="http://127.0.0.1:8080/v1")
    assert RANDOM_MARKER not in kwargs["messages"][1]["content"]
    # call_llm is a public API: the caller's list must survive untouched.
    assert messages[1]["content"] == f"media_marker: {RANDOM_MARKER}"
    assert messages[0]["content"] == "dump props"


def test_marker_inside_tool_call_arguments_is_neutralized():
    """Shallow already walks content / name / tool_call arguments; a marker there
    must be neutralized before the request leaves."""
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "shell", "arguments": f'{{"cmd": "cat {RANDOM_MARKER}"}}'},
        }]},
    ]
    assert _sanitize_messages_media_markers(messages) is True
    assert RANDOM_MARKER not in messages[0]["tool_calls"][0]["function"]["arguments"]


def test_marker_in_deep_only_field_is_neutralized():
    """deep=True must reach nested non-core fields (reasoning_details from byte-level
    models); flipping deep to False leaves a marker here — this test pins that."""
    messages = [{
        "role": "assistant",
        "content": "ok",
        "reasoning_details": [{"type": "reasoning.text", "text": RANDOM_MARKER}],
    }]
    assert _sanitize_messages_media_markers(messages) is True
    assert RANDOM_MARKER not in messages[0]["reasoning_details"][0]["text"]


def test_assemble_api_request_neutralizes_markers(monkeypatch, tmp_path):
    """Primary fix site (#108760): a marker quoted in a tool result must be gone from
    the assembled request before it reaches the wire."""
    import logging

    import hermes_cli.config
    from agent.turn_request_assembly import assemble_api_request
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = {}
    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: cfg)
    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: cfg)
    for key in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(key, raising=False)
    agent = AIAgent(
        platform="cli", provider="custom", api_key="fixture-key",
        base_url="http://127.0.0.1:8080/v1", model="qwen-27b",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, enabled_toolsets=["file"],
    )
    agent._current_turn_timestamp = 1.0
    history = [
        {"role": "system", "content": "Stable prompt"},
        {"role": "user", "content": "dump server props"},
        {"role": "tool", "tool_call_id": "c1", "content": f"media_marker: {RANDOM_MARKER}"},
    ]
    result = assemble_api_request(
        agent, messages=history, current_turn_user_idx=1, _ext_prefetch_cache="",
        _plugin_user_context="", moa_config=None, active_system_prompt="Stable prompt",
        original_user_message="dump server props", pending_moa_prepared_request=None,
        request_logger=logging.getLogger(__name__),
    )
    assert RANDOM_MARKER not in _messages_blob(result.api_messages)
    # The stored transcript keeps what the tool returned (API copy only).
    assert RANDOM_MARKER in history[2]["content"]


def test_iteration_summary_api_messages_neutralize_markers(monkeypatch, tmp_path):
    """The iteration-limit summary builds its own wire copy outside assemble_api_request;
    a marker must be neutralized there too (the bypass found in review of this PR)."""
    import agent.chat_completion_helpers as cch
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    agent = AIAgent(
        api_key="k", base_url="http://127.0.0.1:8080/v1", provider="custom", model="m",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
    )
    agent._cached_system_prompt = "SYS"
    history = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "c1", "content": RANDOM_MARKER},
    ]
    out = cch._iteration_summary_api_messages(agent, [dict(m) for m in history])
    for row in out:
        assert RANDOM_MARKER not in _messages_blob([row])


def _messages_blob(rows):
    import json

    return json.dumps(rows, ensure_ascii=False)


def test_sanitizer_is_idempotent():
    messages = [{"role": "tool", "tool_call_id": "c1", "content": RANDOM_MARKER}]
    assert _sanitize_messages_media_markers(messages) is True
    assert _sanitize_messages_media_markers(messages) is False
