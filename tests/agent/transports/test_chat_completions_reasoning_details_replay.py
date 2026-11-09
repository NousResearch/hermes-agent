"""``reasoning_details`` replay is route-scoped: OpenRouter-format gateways (OpenRouter, Kilo,
Vercel AI Gateway, Nous Portal) and vendors documenting it read it; strict schemas (Groq,
Mistral, Cerebras, Fireworks direct) 400/422 on the field and never receive it (#70233).
The Portal "replay budget" behind #118182 does not reproduce; it is back on the list."""

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


def test_openrouter_format_gateways_keep_and_strict_hosts_strip_reasoning_details():
    transport = get_transport("chat_completions")
    for url in ("https://openrouter.ai/api/v1", "https://inference-api.nousresearch.com/v1",
                "https://api.kilo.ai/api/gateway"):
        kwargs = transport.build_kwargs("m", _HISTORY, base_url=url)
        assert any("reasoning_details" in m for m in kwargs["messages"]), url
    for url in ("https://api.groq.com/openai/v1", "https://api.fireworks.ai/inference/v1",
                "https://api.mistral.ai/v1"):
        kwargs = transport.build_kwargs("m", _HISTORY, base_url=url)
        assert all("reasoning_details" not in m for m in kwargs["messages"]), url
    assert "reasoning_details" in _HISTORY[1]  # durable history is untouched
