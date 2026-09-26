"""A declared vocabulary also authorizes sending the level on the wire.

The picker offering a level is only half the job: on a plain OpenAI-compatible relay the
transport used to send no reasoning field at all (only Kimi / TokenHub / LM Studio got one),
so every level the user picked was a no-op at the endpoint. A route that declares
``models.<id>.reasoning_efforts`` has stated which levels it takes — that declaration is the
opt-in, and undeclared routes keep their old silence.
"""

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport

CONFIG = {
    "providers": {
        "relay": {
            "models": {
                "declared-model": {"reasoning_efforts": ["none", "low", "high", "max"]},
                "silent-model": {"context_length": 128000},
            }
        }
    }
}

MESSAGES = [{"role": "user", "content": "hola"}]


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: CONFIG)


def _kwargs(model, reasoning_config):
    return ChatCompletionsTransport().build_kwargs(
        model, MESSAGES, None, reasoning_config=reasoning_config, model_lower=model,
        max_tokens_param_fn=lambda value: {"max_tokens": value},
    )


def test_a_declared_route_carries_the_chosen_level():
    kwargs = _kwargs("declared-model", {"enabled": True, "effort": "high"})

    assert kwargs["reasoning_effort"] == "high"


def test_a_level_above_the_declaration_is_clamped_down_never_up():
    # ultra is a Hermes-internal step; the route declared max as its ceiling.
    kwargs = _kwargs("declared-model", {"enabled": True, "effort": "ultra"})

    assert kwargs["reasoning_effort"] == "max"


def test_disabled_reasoning_says_none_when_the_route_declares_it():
    kwargs = _kwargs("declared-model", {"enabled": False})

    assert kwargs["reasoning_effort"] == "none"


def test_an_undeclared_route_sends_nothing_as_before():
    kwargs = _kwargs("silent-model", {"enabled": True, "effort": "high"})

    assert "reasoning_effort" not in kwargs


def test_a_model_outside_the_config_sends_nothing():
    kwargs = _kwargs("unknown-model", {"enabled": True, "effort": "high"})

    assert "reasoning_effort" not in kwargs
