"""``reasoning_details`` replay is route-scoped: OpenRouter/Nous read it, every other
chat-completions route gets a wire copy without it (strict schemas 400/422 on the field,
wedging the session after an in-session model switch — hermes-agent#70233)."""

from openai import OpenAI

from agent.auxiliary_wire import prepare_chat_messages
from agent.transports import get_transport

_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok", "reasoning_details": [{"type": "reasoning.text", "text": "x", "signature": "E"}]},
    {"role": "user", "content": "again"},
]


def test_auxiliary_wire_drops_reasoning_details_only_for_non_replaying_routes():
    with OpenAI(api_key="k", base_url="https://api.groq.com/openai/v1") as client:
        kwargs = prepare_chat_messages(client, {"model": "qwen/qwen3.6-27b", "messages": _HISTORY})
    assert all("reasoning_details" not in m for m in kwargs["messages"])
    assert "reasoning_details" in _HISTORY[1]  # durable history is untouched
    with OpenAI(api_key="k", base_url="https://openrouter.ai/api/v1") as client:
        kwargs = prepare_chat_messages(client, {"model": "m", "messages": _HISTORY})
    assert any("reasoning_details" in m for m in kwargs["messages"])


def test_openrouter_and_nous_routes_keep_reasoning_details():
    transport = get_transport("chat_completions")
    for base_url in ("https://openrouter.ai/api/v1", "https://inference-api.nousresearch.com/v1"):
        kwargs = transport.build_kwargs("m", _HISTORY, base_url=base_url)
        assert any("reasoning_details" in m for m in kwargs["messages"]), base_url


def test_openrouter_drops_reasoning_details_for_gemini_and_gemma_upstreams():
    """Regression test for #129037: OpenRouter forwards to strict upstream Google AI Studio.

    Gemini/Gemma endpoints 400 on unexpected 'reasoning_details', consuming 'thought_signature'
    instead. The wire copy must strip reasoning_details for Gemini/Gemma upstreams while
    keeping durable history intact and preserving reasoning_details for OpenRouter models
    that do replay it.
    """
    transport = get_transport("chat_completions")
    assert transport is not None
    openrouter_url = "https://openrouter.ai/api/v1"

    # Gemini/Gemma targets drop reasoning_details on wire
    for model in ("google/gemini-3.6-flash", "google/gemma-2-27b-it", "gemini-2.5-pro", "google/gemini-2.0-flash-exp:free"):
        kwargs = transport.build_kwargs(model, _HISTORY, base_url=openrouter_url)
        assert all("reasoning_details" not in m for m in kwargs["messages"]), f"reasoning_details leaked for {model}"
        assert "reasoning_details" in _HISTORY[1]  # durable history is untouched

    # OpenRouter-native reasoning models keep reasoning_details
    for model in ("deepseek/deepseek-r1", "qwen/qwen-2.5-coder-32b", "m"):
        kwargs = transport.build_kwargs(model, _HISTORY, base_url=openrouter_url)
        assert any("reasoning_details" in m for m in kwargs["messages"]), f"reasoning_details missing for {model}"


def test_auxiliary_wire_drops_reasoning_details_on_openrouter_gemini_aux_lane():
    """Auxiliary background review lane pointing to google/gemini-* via openrouter (#129037)."""
    with OpenAI(api_key="k", base_url="https://openrouter.ai/api/v1") as client:
        kwargs = prepare_chat_messages(client, {"model": "google/gemini-3.6-flash", "messages": _HISTORY})
    assert all("reasoning_details" not in m for m in kwargs["messages"])
    assert "reasoning_details" in _HISTORY[1]


def test_openrouter_gemini_preserves_thought_signature_while_stripping_reasoning_details():
    """Gemini targets require thought_signature (extra_content) but reject reasoning_details (#129037)."""
    transport = get_transport("chat_completions")
    assert transport is not None
    history_with_signature = [
        {"role": "user", "content": "run tool"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                    "extra_content": {"thought_signature": "sig_xyz"},
                }
            ],
            "reasoning_details": [{"type": "reasoning.text", "text": "thought", "signature": "E"}],
        },
    ]

    kwargs = transport.build_kwargs("google/gemini-3.6-flash", history_with_signature, base_url="https://openrouter.ai/api/v1")
    assistant_wire = kwargs["messages"][1]

    # reasoning_details stripped to prevent HTTP 400 from Google upstream
    assert "reasoning_details" not in assistant_wire
    # extra_content (thought_signature) kept because Gemini consumes it
    assert assistant_wire["tool_calls"][0]["extra_content"] == {"thought_signature": "sig_xyz"}
    # Original history unmodified
    assert "reasoning_details" in history_with_signature[1]
