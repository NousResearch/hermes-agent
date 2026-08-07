"""OpenRouter runtime api_mode must not inherit another provider's persisted mode."""

from __future__ import annotations

import pytest

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def _resolve(model_cfg, monkeypatch):
    """Resolve an OpenRouter runtime with the arguments the CLI actually passes.

    ``_ensure_runtime_credentials`` forwards ``explicit_api_key`` /
    ``explicit_base_url`` (staged by the /model switch), which is what routes
    resolution into ``_resolve_openrouter_runtime``.
    """
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "load_config", lambda *a, **k: {"model": model_cfg})
    return rp.resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="sk-or-test",
        explicit_base_url=OPENROUTER_URL,
    )


def test_anthropic_config_does_not_pin_openrouter_to_messages_wire(monkeypatch):
    """A direct-Anthropic default must not force OpenRouter onto the Messages wire.

    Repro: config.yaml defaults to direct Anthropic
    (``provider: anthropic`` + ``api_mode: anthropic_messages``). A
    mid-session ``/model moonshotai/kimi-k3`` correctly restages
    provider/base_url to OpenRouter, but the persisted ``api_mode`` was
    honored unconditionally — so the agent was built on the Anthropic
    Messages transport and talked to OpenRouter's ``/v1/messages`` route
    instead of ``/v1/chat/completions``.
    """
    runtime = _resolve(
        {
            "default": "claude-sonnet-5",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        },
        monkeypatch,
    )

    assert runtime["provider"] == "openrouter"
    assert runtime["api_mode"] == "chat_completions", (
        "a persisted api_mode from a DIFFERENT provider's config block must not "
        f"leak into the OpenRouter runtime: {runtime['api_mode']}"
    )


def test_openrouter_config_still_honors_its_own_api_mode(monkeypatch):
    """The gate must not break a config that genuinely describes OpenRouter."""
    runtime = _resolve(
        {
            "default": "moonshotai/kimi-k3",
            "provider": "openrouter",
            "api_mode": "anthropic_messages",
        },
        monkeypatch,
    )

    assert runtime["api_mode"] == "anthropic_messages"


def test_openrouter_config_without_provider_still_honors_api_mode(monkeypatch):
    """No recorded provider = nothing to contradict; keep the persisted mode.

    Matches ``_provider_supports_explicit_api_mode``'s "no configured
    provider" allowance, so pre-existing configs that only set ``api_mode``
    keep working.
    """
    runtime = _resolve(
        {"default": "moonshotai/kimi-k3", "api_mode": "anthropic_messages"},
        monkeypatch,
    )

    assert runtime["api_mode"] == "anthropic_messages"


@pytest.mark.parametrize(
    "configured_provider",
    ["anthropic", "openai", "copilot", "bedrock", "openai-codex"],
)
def test_foreign_provider_modes_never_leak(monkeypatch, configured_provider):
    """Any foreign provider's persisted mode falls back to the endpoint default."""
    runtime = _resolve(
        {
            "default": "some-model",
            "provider": configured_provider,
            "api_mode": "codex_responses",
        },
        monkeypatch,
    )

    assert runtime["api_mode"] == "chat_completions"
