"""model_overrides.<provider>.<model>.extra_body — per-model wire additions.

Behaviour contract: the explicit per-model ``extra_body`` mapping reaches the
chat-completions wire ONLY through the ``extra_body`` request kwarg (the openai
SDK flattens it into the JSON body top level), never as top-level ``create()``
kwargs — openai>=2.54 type-checks those and raises TypeError on unknown names.
Explicit-only: a ``_default`` fill-gap never supplies wire params.
"""

import pytest

from agent.models_dev import explicit_model_extra_body


def _cfg(overrides):
    return {"model_overrides": overrides}


class _Profile:
    """Duck-typed provider profile for the transport path (no registration)."""

    name = "ollama-cloud"
    api_mode = "chat_completions"
    fixed_temperature = None
    native_reasoning_details_type = None

    def prepare_messages(self, messages):
        return messages

    def get_max_tokens(self, model):
        return None

    def build_api_kwargs_extras(self, **ctx):
        return {}, {}

    def build_extra_body(self, **ctx):
        return {}


def test_explicit_extra_body_lands_in_result():
    cfg = _cfg({"ollama-cloud": {"glm-5.3-flash": {"extra_body": {"repeat_penalty": 1.2}}}})
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3-flash", config=cfg) == {
        "repeat_penalty": 1.2
    }


def test_model_match_is_case_insensitive_like_catalog_lookup():
    cfg = _cfg({"ollama-cloud": {"glm-5.3-flash": {"extra_body": {"repeat_penalty": 1.2}}}})
    assert explicit_model_extra_body("ollama-cloud", "GLM-5.3-Flash", config=cfg) == {
        "repeat_penalty": 1.2
    }


def test_other_model_of_same_provider_gets_nothing():
    cfg = _cfg({"ollama-cloud": {"glm-5.3-flash": {"extra_body": {"repeat_penalty": 1.2}}}})
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3", config=cfg) == {}
    assert explicit_model_extra_body("ollama-cloud", "kimi-k3.1-pro", config=cfg) == {}


def test_default_fill_gap_never_supplies_wire_params():
    cfg = _cfg({"ollama-cloud": {"_default": {"extra_body": {"repeat_penalty": 1.5}}}})
    assert explicit_model_extra_body("ollama-cloud", "totally-unknown-model", config=cfg) == {}


def test_invalid_extra_body_value_is_ignored_not_raised():
    cfg = _cfg({"ollama-cloud": {"glm-5.3-flash": {"extra_body": "repeat_penalty=1.2"}}})
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3-flash", config=cfg) == {}


def test_non_string_keys_are_dropped():
    cfg = _cfg({"ollama-cloud": {"glm-5.3-flash": {"extra_body": {1: "x", "top_k": 3}}}})
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3-flash", config=cfg) == {"top_k": 3}


def test_empty_config_is_the_norm_not_an_error():
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3-flash", config={}) == {}
    assert explicit_model_extra_body("ollama-cloud", "glm-5.3-flash", config=None) == {}


@pytest.fixture
def overrides_config(monkeypatch):
    """Point the global (config=None) read at a fixed overrides mapping."""
    from agent import models_dev

    overrides = {"ollama-cloud": {"glm-5.3-flash": {"extra_body": {"repeat_penalty": 1.2}}}}

    def fake_cfg_get(*keys, default=None, config=None, **kwargs):
        if "model_overrides" in keys and config is None:
            return overrides
        return default

    monkeypatch.setattr(models_dev, "_cfg_get", fake_cfg_get)
    return overrides


def test_profile_path_merges_extra_body_into_request(overrides_config):
    from agent.transports.chat_completions import ChatCompletionsTransport

    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        "glm-5.3-flash",
        [{"role": "user", "content": "hi"}],
        None,
        provider_profile=_Profile(),
        provider_name="ollama-cloud",
        base_url="https://ollama.com/v1",
        session_id="test-session",
    )
    assert kwargs.get("extra_body", {}).get("repeat_penalty") == 1.2
    # The addition must ride ONLY the extra_body kwarg — a top-level repeat_penalty
    # is the TypeError the openai SDK raises on create() (the regression this guards).
    assert "repeat_penalty" not in kwargs


def test_legacy_path_merges_extra_body_into_request(overrides_config):
    from agent.transports.chat_completions import ChatCompletionsTransport

    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        "glm-5.3-flash",
        [{"role": "user", "content": "hi"}],
        None,
        provider_name="ollama-cloud",
        base_url="https://ollama.com/v1",
        session_id="test-session",
    )
    assert kwargs.get("extra_body", {}).get("repeat_penalty") == 1.2
    assert "repeat_penalty" not in kwargs


def test_unrelated_model_leaves_request_kwargs_untouched(overrides_config):
    from agent.transports.chat_completions import ChatCompletionsTransport

    transport = ChatCompletionsTransport()
    kwargs = transport.build_kwargs(
        "glm-5.3",
        [{"role": "user", "content": "hi"}],
        None,
        provider_profile=_Profile(),
        provider_name="ollama-cloud",
        base_url="https://ollama.com/v1",
        session_id="test-session",
    )
    assert not kwargs.get("extra_body")
