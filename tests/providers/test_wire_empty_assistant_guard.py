"""Transport-boundary guard against wire-empty assistant messages.

Strict OpenAI-compatible providers (Moonshot/Kimi, ZAI) reject any request
containing an assistant message that serializes empty with HTTP 400 "the
message at position N with role 'assistant' must not be empty" — permanently
wedging the session. The semantic pre-call filters judge INTERNAL shapes and
keep anything payload-like, but exotic shapes (unknown content-block types,
junk tool_calls, stripped scaffolding) can serialize to nothing. Two live
wedges on 2026-07-19 (request dumps ..._063158_ca3fef and ..._074429_daef41)
each slipped a different variant past the filters, so convert_messages() now
drops anything empty AS SERIALIZED — the last line of defense.
"""

from agent.transports.chat_completions import (
    ChatCompletionsTransport,
    _drop_wire_empty_assistants,
    _wire_empty_assistant,
)


def test_predicate_catches_serialized_empty_variants():
    empties = [
        {"role": "assistant", "content": "", "tool_calls": []},   # both live wedges
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": []},
        {"role": "assistant", "content": [{}]},                    # unknown block, no text
        {"role": "assistant", "content": [{"type": "output_text", "text": ""}]},
        {"role": "assistant", "content": "", "reasoning": "", "reasoning_details": []},
    ]
    for msg in empties:
        assert _wire_empty_assistant(msg) is True, msg


def test_predicate_keeps_payload_bearing_messages():
    kept = [
        {"role": "assistant", "content": "real text"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        # reasoning replay is load-bearing for require-side providers
        {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
        {"role": "assistant", "content": "", "reasoning_details": [{"type": "reasoning.text"}]},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "user", "content": ""},   # only assistant messages are judged
        {"role": "tool", "content": ""},
    ]
    for msg in kept:
        assert _wire_empty_assistant(msg) is False, msg


def test_convert_messages_drops_empty_tail_on_both_exit_paths():
    transport = ChatCompletionsTransport()
    shell = {"role": "assistant", "content": "", "tool_calls": []}
    base = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "working on it"},
    ]
    # Fast path (no field-level sanitize needed).
    out = transport.convert_messages(base + [dict(shell)], model="moonshotai/kimi-k3")
    assert all(not _wire_empty_assistant(m) for m in out)
    assert len(out) == 2
    # Sanitize path (message carries a stripped internal marker).
    marked = dict(shell)
    marked["_empty_recovery_synthetic"] = True
    out2 = transport.convert_messages(base + [marked], model="moonshotai/kimi-k3")
    assert all(not _wire_empty_assistant(m) for m in out2)
    assert len(out2) == 2


def test_drop_returns_same_object_when_nothing_is_empty():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert _drop_wire_empty_assistants(msgs) is msgs


def test_summary_path_helper_is_importable_and_drops_shells():
    """The max-iterations summary path bypasses the transport and mirrors its
    sanitization by hand (chat_completion_helpers). It must apply the same
    wire-empty guard — a shell at any position wedges the summary request on
    Moonshot/Kimi (observed live 2026-07-19, 'position 3')."""
    import inspect

    from agent import chat_completion_helpers as helpers

    src = inspect.getsource(helpers)
    at = src.index('_drop_wire_empty_assistants')
    assert at > -1, 'summary path must apply the wire-empty guard'
    # Guard runs AFTER the semantic filters in the same build.
    semantic = src.index('_drop_thinking_only_and_merge_users(api_messages)')
    assert semantic < at
