"""Keep reasoning budgets consistent with the current replay contract.

Regression coverage following #99398: blank legacy placeholders must not
hide real reasoning that the active chat-completions route actually sends.
"""

from copy import deepcopy

import pytest

from agent.context_compressor import _estimate_msg_budget_tokens
from agent.message_sanitization import apply_reasoning_content_policy, reasoning_replay_route
from agent.model_metadata import _estimate_message_tokens_without_images
from agent.transports.chat_completions import ChatCompletionsTransport


_BUDGETS = [
    pytest.param(_estimate_msg_budget_tokens, id="tail"),
    pytest.param(_estimate_message_tokens_without_images, id="preflight"),
]
_ROUTES = [
    pytest.param("deepseek", "https://api.deepseek.com/v1", "deepseek-v4-flash", id="must-echo"),
    pytest.param("custom", "http://127.0.0.1:8080/v1", "local-model", id="local"),
]


def _wire_message(source, provider, base_url, model):
    route = reasoning_replay_route("chat_completions", provider, model, base_url)
    replay = deepcopy(source)
    apply_reasoning_content_policy(source, replay, route.pad, route.carriers)
    return ChatCompletionsTransport().convert_messages(
        [replay], model=model, base_url=base_url, provider=provider,
    )[0]


@pytest.mark.parametrize("estimate", _BUDGETS)
@pytest.mark.parametrize("provider,base_url,model", _ROUTES)
@pytest.mark.parametrize("placeholder", ["", " ", "\t\n"], ids=["empty", "space", "whitespace"])
def test_blank_placeholder_keeps_replayed_reasoning_chargeable(
    estimate, provider, base_url, model, placeholder,
):
    text = "synthetic replay reasoning " * 1_000
    base = {"role": "assistant", "content": "hi", "reasoning_content": placeholder}
    source = {**base, "reasoning": text}
    original = deepcopy(source)

    wire = _wire_message(source, provider, base_url, model)
    assert wire["reasoning_content"] == text
    assert estimate(source) - estimate(base) > 1_000
    assert source == original  # Replay and accounting must not rewrite canonical history.


@pytest.mark.parametrize("estimate", _BUDGETS)
def test_nonblank_reasoning_content_displaces_internal_reasoning_once(estimate):
    chosen = "synthetic chosen reasoning " * 1_000
    base = {"role": "assistant", "content": "hi", "reasoning_content": chosen}
    source = {**base, "reasoning": "synthetic displaced reasoning " * 1_000}
    original = deepcopy(source)

    wire = _wire_message(source, "deepseek", "https://api.deepseek.com/v1", "deepseek-v4-flash")
    assert wire["reasoning_content"] == chosen
    assert "reasoning" not in wire
    assert estimate(source) == estimate(base)
    assert source == original


@pytest.mark.parametrize("estimate", _BUDGETS)
def test_missing_reasoning_content_keeps_promoted_reasoning_chargeable(estimate):
    text = "synthetic promoted reasoning " * 1_000
    base = {"role": "assistant", "content": "hi"}
    source = {**base, "reasoning": text}
    original = deepcopy(source)

    wire = _wire_message(source, "deepseek", "https://api.deepseek.com/v1", "deepseek-v4-flash")
    assert wire["reasoning_content"] == text
    assert estimate(source) - estimate(base) > 1_000
    assert source == original
