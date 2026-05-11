"""Tests for the OpenAI-compatible TTS branch of `hermes setup tts`.

Covers `_setup_tts_provider()` when the user picks the
"OpenAI-compatible (OpenRouter, Vercel AI Gateway, …)" option, which writes
``tts.openai.{base_url, model, voice, api_key}`` and persists ``provider=openai``
(the same OpenAI SDK code path drives any compatible /audio/speech endpoint).

Approach: mock ``prompt_choice``, ``prompt``, and ``save_config`` so the
wizard branch runs deterministically end-to-end without any I/O.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hermes_cli.setup import _setup_tts_provider


# Index of "OpenAI-compatible" in the provider list when the Nous-managed
# entry is NOT prepended (i.e. user has no paid subscription).
# Order: edge(0), elevenlabs(1), openai(2), openai-compatible(3), xai(4), …
OPENAI_COMPAT_IDX = 3


@pytest.fixture
def no_nous_subscription(monkeypatch):
    """Disable the Nous-managed TTS option so the index above stays stable."""
    from types import SimpleNamespace

    # Minimal duck-typed object: only the .nous_auth_present attribute is read
    # by `_setup_tts_provider` (gating the optional menu prefix).
    fake_features = SimpleNamespace(nous_auth_present=False)
    monkeypatch.setattr(
        "hermes_cli.setup.get_nous_subscription_features",
        lambda config: fake_features,
    )
    monkeypatch.setattr(
        "hermes_cli.setup.managed_nous_tools_enabled",
        lambda: False,
    )
    return fake_features


def _patch_io(prompt_choice_idx: int, prompt_responses: list[str]):
    """Build patchers for prompt_choice + prompt + save_config + print_*.

    prompt_choice_idx: index returned by prompt_choice.
    prompt_responses: list of strings returned in order by sequential prompt() calls.
    """
    prompt_iter = iter(prompt_responses)

    def _fake_prompt(question, default=None, password=False):  # noqa: ARG001
        try:
            return next(prompt_iter)
        except StopIteration:
            return default or ""

    return [
        patch("hermes_cli.setup.prompt_choice", return_value=prompt_choice_idx),
        patch("hermes_cli.setup.prompt", side_effect=_fake_prompt),
        patch("hermes_cli.setup.save_config"),
    ]


class TestOpenAICompatibleTTS:
    """Verify the new wizard branch for OpenRouter / Vercel AI Gateway / custom."""

    def test_openrouter_preset_writes_openai_section(self, no_nous_subscription):
        """OpenRouter preset (choice 1) pre-fills base_url + Gemini TTS model."""
        config: dict = {"tts": {"provider": "edge"}}

        # User flow:
        #   prompt_choice → "OpenAI-compatible" (index 3)
        #   prompt 1: endpoint preset → "1" (OpenRouter)
        #   prompt 2: model → "" (accept preset default)
        #   prompt 3: voice → "" (accept preset default)
        #   prompt 4: API key → "sk-or-v1-test"
        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            ["1", "", "", "sk-or-v1-test"],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        # Provider is normalized to "openai" (same code path drives all compatible endpoints).
        assert config["tts"]["provider"] == "openai"
        oai = config["tts"]["openai"]
        assert oai["base_url"] == "https://openrouter.ai/api/v1"
        assert oai["model"] == "google/gemini-3.1-flash-tts-preview"
        assert oai["voice"] == "Kore"
        assert oai["api_key"] == "sk-or-v1-test"

    def test_openrouter_preset_with_overridden_model_and_voice(self, no_nous_subscription):
        """User can override the preset model and voice."""
        config: dict = {"tts": {"provider": "edge"}}

        # endpoint=1 (OpenRouter), model=custom, voice=custom, key=custom
        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            [
                "1",
                "google/gemini-3-pro-tts",
                "Charon",
                "sk-or-v1-override",
            ],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        oai = config["tts"]["openai"]
        assert oai["base_url"] == "https://openrouter.ai/api/v1"
        assert oai["model"] == "google/gemini-3-pro-tts"
        assert oai["voice"] == "Charon"
        assert oai["api_key"] == "sk-or-v1-override"

    def test_vercel_ai_gateway_preset(self, no_nous_subscription):
        """Vercel AI Gateway preset (choice 2)."""
        config: dict = {"tts": {"provider": "edge"}}

        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            ["2", "", "", "vai-gateway-key"],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        oai = config["tts"]["openai"]
        assert oai["base_url"] == "https://ai-gateway.vercel.sh/v1"
        assert oai["model"] == "openai/tts-1"
        assert oai["voice"] == "alloy"
        assert oai["api_key"] == "vai-gateway-key"

    def test_custom_endpoint_requires_all_fields(self, no_nous_subscription):
        """Custom endpoint (choice 3): user must supply base_url, model, voice, key."""
        config: dict = {"tts": {"provider": "edge"}}

        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            [
                "3",
                "https://my-proxy.example.com/v1",
                "custom/tts-model",
                "MyVoice",
                "custom-key-123",
            ],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        oai = config["tts"]["openai"]
        assert oai["base_url"] == "https://my-proxy.example.com/v1"
        assert oai["model"] == "custom/tts-model"
        assert oai["voice"] == "MyVoice"
        assert oai["api_key"] == "custom-key-123"

    def test_missing_api_key_falls_back_to_edge(self, no_nous_subscription):
        """If the user does not supply an API key, fall back to Edge TTS."""
        config: dict = {"tts": {"provider": "edge"}}

        # endpoint=1 (OpenRouter), defaults for model/voice, EMPTY api_key
        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            ["1", "", "", ""],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        # No openai.api_key written, provider falls back to edge.
        assert config["tts"]["provider"] == "edge"

    def test_custom_endpoint_missing_base_url_falls_back(self, no_nous_subscription):
        """Custom endpoint with no base_url provided → fallback to Edge."""
        config: dict = {"tts": {"provider": "edge"}}

        # endpoint=3 (Custom), EMPTY base_url
        patchers = _patch_io(OPENAI_COMPAT_IDX, ["3", ""])
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        assert config["tts"]["provider"] == "edge"

    def test_invalid_endpoint_choice_defaults_to_openrouter(self, no_nous_subscription):
        """Out-of-range choice silently maps to "1" (OpenRouter preset)."""
        config: dict = {"tts": {"provider": "edge"}}

        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            ["99", "", "", "key"],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        oai = config["tts"]["openai"]
        assert oai["base_url"] == "https://openrouter.ai/api/v1"
        assert oai["model"] == "google/gemini-3.1-flash-tts-preview"

    def test_does_not_pollute_global_env_vars(self, no_nous_subscription, monkeypatch):
        """The branch must not write to OPENAI_API_KEY / VOICE_TOOLS_OPENAI_KEY."""
        config: dict = {"tts": {"provider": "edge"}}

        save_env_calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "hermes_cli.setup.save_env_value",
            lambda k, v: save_env_calls.append((k, v)),
        )

        patchers = _patch_io(
            OPENAI_COMPAT_IDX,
            ["1", "", "", "sk-or-v1-key"],
        )
        for p in patchers:
            p.start()
        try:
            _setup_tts_provider(config)
        finally:
            for p in patchers:
                p.stop()

        # Critical: the openai-compatible branch never touches env vars —
        # the key is always stored in config.yaml under tts.openai.api_key.
        assert save_env_calls == []
        assert config["tts"]["openai"]["api_key"] == "sk-or-v1-key"
