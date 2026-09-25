"""``reasoning_details`` replay is route-scoped.

OpenRouter reads its unified reasoning array, provider profiles may opt into a
native carrier, and every other chat-completions route gets a wire copy without
foreign reasoning sidecars (strict schemas 400/422 on the field, wedging the
session after an in-session model switch — hermes-agent#70233/#122831).
"""

from types import SimpleNamespace

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


def test_openrouter_route_keeps_reasoning_details():
    transport = get_transport("chat_completions")
    assert transport is not None
    kwargs = transport.build_kwargs("m", _HISTORY, base_url="https://openrouter.ai/api/v1")
    assert any("reasoning_details" in m for m in kwargs["messages"])


def test_nous_portal_drops_foreign_reasoning_details():
    transport = get_transport("chat_completions")
    assert transport is not None
    kwargs = transport.build_kwargs("m", _HISTORY, base_url="https://inference-api.nousresearch.com/v1")
    assert all("reasoning_details" not in m for m in kwargs["messages"])
    assert "reasoning_details" in _HISTORY[1]  # durable history is untouched


def test_native_reasoning_details_survive_only_matching_profile():
    transport = get_transport("chat_completions")
    assert transport is not None
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_details": [
                {"type": "openrouter.native_assistant", "encrypted_content": "foreign"},
                {"type": "nous.native_assistant", "encrypted_content": "native"},
                {"type": "reasoning.text", "text": "generic"},
            ],
        },
    ]

    converted = transport.convert_messages(
        messages,
        base_url="https://inference-api.nousresearch.com/v1",
        provider_profile=SimpleNamespace(native_reasoning_details_type="nous.native_assistant"),
    )

    assert converted[1]["reasoning_details"] == [
        {"type": "nous.native_assistant", "encrypted_content": "native"},
        {"type": "reasoning.text", "text": "generic"},
    ]
    assert messages[1]["reasoning_details"][0]["encrypted_content"] == "foreign"
