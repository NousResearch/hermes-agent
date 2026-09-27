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
