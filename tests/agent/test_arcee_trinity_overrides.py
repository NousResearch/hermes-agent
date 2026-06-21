"""Tests for Arcee Trinity Large Thinking per-model overrides.

Arcee Trinity Large Thinking is a reasoning model that wants:
- Fixed temperature=0.5 (vs the global default)
- Compression threshold=0.75 (delay compression to preserve reasoning context)

The helpers must match the bare model name, including when it arrives via
OpenRouter as ``arcee-ai/trinity-large-thinking``, but must NOT hit sibling
Arcee models like trinity-large-preview or trinity-mini.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    _compression_threshold_for_model,
    _fixed_temperature_for_model,
    _is_arcee_trinity_thinking,
    _is_codex_gpt55,
)
from run_agent import AIAgent


@pytest.mark.parametrize(
    "model",
    [
        "trinity-large-thinking",
        "arcee-ai/trinity-large-thinking",
        "Arcee-AI/Trinity-Large-Thinking",  # case-insensitive
        "  trinity-large-thinking  ",  # whitespace tolerant
    ],
)
def test_is_arcee_trinity_thinking_matches(model: str) -> None:
    assert _is_arcee_trinity_thinking(model) is True


@pytest.mark.parametrize(
    "model",
    [
        None,
        "",
        "trinity-large-preview",
        "arcee-ai/trinity-large-preview:free",
        "trinity-mini",
        "arcee-ai/trinity-mini",
        "trinity-large",  # prefix-only must not match
        "claude-sonnet-4.6",
        "gpt-5.4",
    ],
)
def test_is_arcee_trinity_thinking_rejects_non_matches(model) -> None:
    assert _is_arcee_trinity_thinking(model) is False


def test_fixed_temperature_for_trinity_thinking() -> None:
    assert _fixed_temperature_for_model("trinity-large-thinking") == 0.5
    assert _fixed_temperature_for_model("arcee-ai/trinity-large-thinking") == 0.5


def test_fixed_temperature_sibling_arcee_models_unaffected() -> None:
    # Preview and mini do not pin temperature — caller chooses its default.
    assert _fixed_temperature_for_model("trinity-large-preview") is None
    assert _fixed_temperature_for_model("trinity-mini") is None


def test_compression_threshold_for_trinity_thinking() -> None:
    assert _compression_threshold_for_model("trinity-large-thinking") == 0.75
    assert _compression_threshold_for_model("arcee-ai/trinity-large-thinking") == 0.75


def test_compression_threshold_default_none_for_other_models() -> None:
    # None means "leave the user's config value unchanged".
    assert _compression_threshold_for_model(None) is None
    assert _compression_threshold_for_model("") is None
    assert _compression_threshold_for_model("trinity-large-preview") is None
    assert _compression_threshold_for_model("claude-sonnet-4.6") is None
    assert _compression_threshold_for_model("kimi-k2") is None


# ---------------------------------------------------------------------------
# Codex gpt-5.5 compaction-threshold autoraise
#
# ChatGPT's Codex OAuth backend caps gpt-5.5 at a 272K window (verified live:
# ~330K-token request rejected with context_length_exceeded, ~250K accepted).
# The default 50% compaction trigger would fire at ~136K — half the usable
# window — so this route raises the trigger to 85%. Only the Codex OAuth route
# is affected; the same slug on OpenAI direct / OpenRouter / Copilot exposes a
# larger window and keeps the user's global threshold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.5-2026-04-23",  # dated snapshot
        "gpt-5.5-codex-mini",  # Codex variant of the 5.5 family (also 272K-capped)
        "openai/gpt-5.5",  # aggregator-prefixed (still on the codex route)
        "GPT-5.5",  # case-insensitive
        "  gpt-5.5  ",  # whitespace tolerant
    ],
)
def test_is_codex_gpt55_matches_on_codex_provider(model: str) -> None:
    assert _is_codex_gpt55(model, "openai-codex") is True


@pytest.mark.parametrize(
    "provider",
    ["openrouter", "openai", "copilot", "openai-api", "", None],
)
def test_is_codex_gpt55_rejects_non_codex_providers(provider) -> None:
    # gpt-5.5 on any non-Codex route keeps the larger window — no override.
    assert _is_codex_gpt55("gpt-5.5", provider) is False


@pytest.mark.parametrize(
    "model",
    ["gpt-5.4", "gpt-5", "gpt-5.55", "gpt-5.50", "", None],
)
def test_is_codex_gpt55_rejects_non_55_models(model) -> None:
    # gpt-5.55 / gpt-5.50 are different families and must NOT match — the
    # "gpt-5.5-" / "gpt-5.5." prefix guards require a separator after "5.5".
    assert _is_codex_gpt55(model, "openai-codex") is False


def test_compression_threshold_for_codex_gpt55() -> None:
    assert _compression_threshold_for_model("gpt-5.5", "openai-codex") == 0.85
    assert _compression_threshold_for_model("gpt-5.5-pro", "openai-codex") == 0.85
    assert _compression_threshold_for_model("openai/gpt-5.5", "openai-codex") == 0.85


def test_compression_threshold_codex_gpt55_other_routes_unaffected() -> None:
    # Same slug, different route → no override (keep the user's config value).
    assert _compression_threshold_for_model("gpt-5.5", "openrouter") is None
    assert _compression_threshold_for_model("gpt-5.5", "openai") is None
    assert _compression_threshold_for_model("gpt-5.5", "copilot") is None
    assert _compression_threshold_for_model("openai/gpt-5.5") is None  # no provider


def test_compression_threshold_codex_gpt55_opt_out() -> None:
    # allow_codex_gpt55_autoraise=False reverts to the global default (None).
    assert (
        _compression_threshold_for_model(
            "gpt-5.5", "openai-codex", allow_codex_gpt55_autoraise=False
        )
        is None
    )


def test_compression_threshold_opt_out_does_not_disable_trinity() -> None:
    # The opt-out flag is scoped to the Codex gpt-5.5 autoraise; the Arcee
    # Trinity override must still apply when the flag is False.
    assert (
        _compression_threshold_for_model(
            "trinity-large-thinking", "openrouter", allow_codex_gpt55_autoraise=False
        )
        == 0.75
    )


def test_codex_gpt55_autoraise_does_not_queue_gateway_warning() -> None:
    """Autoraise should tune compression without sending startup notices to chat."""

    def fake_load_config():
        return {
            "agent": {},
            "compression": {
                "enabled": True,
                "threshold": 0.70,
                "target_ratio": 0.15,
                "protect_last_n": 20,
                "codex_gpt55_autoraise": True,
            },
        }

    fake_client = type(
        "FakeClient",
        (),
        {
            "api_key": "test-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )()

    with (
        patch("hermes_cli.config.load_config", side_effect=fake_load_config),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fake_client, "gpt-5.5"),
        ),
    ):
        agent = AIAgent(
            provider="openai-codex",
            model="gpt-5.5",
            quiet_mode=True,
            enabled_toolsets=[],
            skip_memory=True,
            load_soul_identity=False,
        )

    assert agent.context_compressor.threshold_percent == 0.85
    assert agent._compression_threshold_autoraised == {"from": 0.70, "to": 0.85}
    assert agent._compression_warning is None
