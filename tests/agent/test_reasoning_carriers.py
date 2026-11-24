"""Gemini text-turn signatures survive capture and reach only their route."""

from agent.chat_completion_helpers import build_assistant_message
from agent.gemini_native_adapter import _build_gemini_contents, translate_gemini_response, translate_stream_event
from agent.reasoning_carriers import shape_wire_carriers
from agent.transports.chat_completions import ChatCompletionsTransport


class _Agent:
    stream_delta_callback = _stream_callback = reasoning_callback = None
    verbose_logging = False

    def _extract_reasoning(self, msg):
        from agent.agent_runtime_helpers import extract_reasoning
        return extract_reasoning(self, msg)

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, raw):
        return (None, None)

    def _derive_responses_function_call_id(self, call_id, item_id):
        return item_id

    def _deterministic_call_id(self, *a):
        return "call_x"


def _stored(response):
    return build_assistant_message(_Agent(), ChatCompletionsTransport().normalize_response(response), "stop")


def test_gemini_native_text_turn_signature_is_captured_and_replayed_on_the_last_part():
    resp = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                  {"text": "391", "thoughtSignature": "SIG_TEXT"}]},
                            "finishReason": "STOP"}]}
    stored = _stored(translate_gemini_response(resp, "gemini-3.6-flash"))
    # Streams deliver the signature on a trailing empty-text part.
    chunks = translate_stream_event({"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "SIG_S"}]}}]},
                                    "gemini-3.6-flash", {})
    assert [c.choices[0].delta.extra_content for c in chunks] == [{"google": {"thought_signature": "SIG_S"}}]

    history = [{"role": "user", "content": "q"}, stored, {"role": "user", "content": "again"}]
    resumed = [{k: v for k, v in m.items() if k != "extra_content"} for m in history]  # state.db keeps only the record
    for msgs in (history, resumed):
        wire = shape_wire_carriers(msgs, model="gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta")
        contents, _ = _build_gemini_contents(wire, is_gemini3=True)
        assert contents[1]["parts"][-1] == {"text": "391", "thoughtSignature": "SIG_TEXT"}
