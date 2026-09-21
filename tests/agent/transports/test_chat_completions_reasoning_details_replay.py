"""``reasoning_details`` replay is route-scoped and budget-aware on Nous Portal."""

from copy import deepcopy

from openai import OpenAI

from agent.auxiliary_wire import prepare_chat_messages
from agent.transports import get_transport

_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "ok", "reasoning_details": [{"type": "reasoning.text", "text": "x", "signature": "E"}]},
    {"role": "user", "content": "again"},
]

_PORTAL_HISTORY = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "first answer", "reasoning_details": [{"type": "reasoning.text", "text": "old", "signature": "OLD"}]},
    {"role": "user", "content": "second"},
    {"role": "assistant", "content": "second answer", "reasoning_details": [{"type": "reasoning.text", "text": "new", "signature": "NEW"}]},
    {"role": "user", "content": "continue"},
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


def test_portal_replays_only_newest_assistant_reasoning_details_without_mutating_history():
    transport = get_transport("chat_completions")
    history = deepcopy(_PORTAL_HISTORY)

    kwargs = transport.build_kwargs("m", history, base_url="https://inference-api.nousresearch.com/v1")
    replayed = [m for m in kwargs["messages"] if m.get("role") == "assistant"]

    assert "reasoning_details" not in replayed[0]
    assert replayed[1]["reasoning_details"] == _PORTAL_HISTORY[3]["reasoning_details"]
    assert history == _PORTAL_HISTORY


def test_auxiliary_portal_wire_uses_the_same_newest_turn_replay_policy():
    with OpenAI(api_key="k", base_url="https://inference-api.nousresearch.com/v1") as client:
        kwargs = prepare_chat_messages(client, {"model": "m", "messages": _PORTAL_HISTORY})

    replayed = [m for m in kwargs["messages"] if m.get("role") == "assistant"]
    assert "reasoning_details" not in replayed[0]
    assert replayed[1]["reasoning_details"] == _PORTAL_HISTORY[3]["reasoning_details"]


def test_strict_routes_drop_all_portal_style_replay_details():
    transport = get_transport("chat_completions")
    kwargs = transport.build_kwargs("m", _PORTAL_HISTORY, base_url="https://api.groq.com/openai/v1")

    assert all("reasoning_details" not in message for message in kwargs["messages"])
