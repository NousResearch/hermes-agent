"""Auxiliary clients ride the wire a provider profile declares when no api_mode is given."""

import model_tools  # registers the bundled provider profiles

from agent.auxiliary_client import AnthropicAuxiliaryClient, CodexAuxiliaryClient, resolve_provider_client

assert model_tools


def test_anthropic_messages_profile_resolves_to_messages_adapter(monkeypatch):
    # Bare ``provider: commandcode-anthropic`` (no api_mode) must wrap the client on the profile's
    # declared wire, or the ``_reasoning_config`` kwarg would reach a plain OpenAI client and TypeError.
    monkeypatch.setenv("COMMANDCODE_API_KEY", "sk-test-" + "x" * 20)
    client, _ = resolve_provider_client("commandcode-anthropic", model="claude-haiku-4-5-20251001")
    assert isinstance(client, AnthropicAuxiliaryClient)


def test_codex_responses_profile_resolves_to_responses_adapter(monkeypatch):
    # Bare ``provider: meta-ai`` (and so vision/compression on a Meta main provider) must ride the
    # declared Responses wire like the main loop, not fall back to /chat/completions.
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-" + "x" * 20)
    client, _ = resolve_provider_client("meta-ai", model="muse-spark-1.3")
    assert isinstance(client, CodexAuxiliaryClient)


def test_explicit_chat_api_mode_beats_declared_responses_wire(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-test-" + "x" * 20)
    client, _ = resolve_provider_client("meta-ai", model="muse-spark-1.3", api_mode="chat_completions")
    assert not isinstance(client, CodexAuxiliaryClient)
