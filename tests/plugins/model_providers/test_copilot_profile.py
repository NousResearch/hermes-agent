"""Unit + e2e tests for the Copilot provider profile's reasoning-effort wiring.

Regression coverage for the *second half* of the Copilot-hosted-Claude
``reasoning_effort`` bug.

PR #51953 fixed the ``run_agent.py`` gate (``_supports_reasoning_extra_body``)
and the legacy / no-profile path (``_github_models_reasoning_extra_body``) so
both resolve Claude efforts through the cached ``get_copilot_reasoning_efforts``
helper.  But the *live* path for ``provider='copilot'`` does not run either of
those: ``chat_completion_helpers.build_api_kwargs`` finds the registered
``CopilotProfile`` and delegates to ``ChatCompletionsTransport`` ->
``_build_kwargs_from_profile`` -> ``CopilotProfile.build_api_kwargs_extras``.

That profile resolved supported efforts through the *bare*
``github_model_reasoning_efforts(model)`` call (no ``catalog``, no
``api_key``), which falls through to the static GPT/o-series table and returns
``[]`` for Claude.  Net effect: the gate said "yes, reasoning is supported"
(``supports_reasoning=True``) while the profile simultaneously resolved "no
supported efforts", so both emit branches were dead and ``reasoning_effort:
high`` was silently dropped on *every* Copilot-hosted Claude turn, exactly the
bug the PR set out to fix, one layer deeper than it reached.

These tests pin the contract at two layers:

  1. ``CopilotProfile.build_api_kwargs_extras`` in isolation: the reviewer's
     repro (``provider='copilot'`` + Claude must emit ``extra_body.reasoning``).
  2. End-to-end through ``ChatCompletionsTransport.build_kwargs``: the actual
     live call path, asserting ``extra_body={'reasoning':{'effort':...}}`` is
     emitted on the wire.

Both go through ``providers.get_provider_profile`` so they stay honest: if the
registered profile is ever swapped for a plain ``ProviderProfile`` the
assertions collapse.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# A realistic Copilot ``/models`` catalog entry for a Claude model that
# advertises ``reasoning_effort`` support (mirrors the live API shape parsed by
# ``github_model_reasoning_efforts``: ``capabilities.supports.reasoning_effort``).
_CLAUDE_CATALOG = [
    {
        "id": "claude-opus-4.8",
        "capabilities": {
            "type": "chat",
            "supports": {"reasoning_effort": ["low", "medium", "high", "max"]},
        },
        "supported_endpoints": ["/chat/completions"],
    },
    {
        # A non-reasoning chat model, used to prove we don't over-emit.
        "id": "gpt-4.1",
        "capabilities": {"type": "chat", "supports": {}},
        "supported_endpoints": ["/chat/completions"],
    },
]


@pytest.fixture(autouse=True)
def _reset_copilot_catalog_cache():
    """Reset the module-level Copilot catalog cache around every test.

    ``get_copilot_reasoning_efforts`` memoises the catalog for an hour in a
    module global; without this reset, cross-test ordering would leak a warm
    (or empty) catalog and mask the very behaviour under test.
    """
    import hermes_cli.models as mod

    mod._copilot_catalog_cache = None
    mod._copilot_catalog_cache_time = 0.0
    yield
    mod._copilot_catalog_cache = None
    mod._copilot_catalog_cache_time = 0.0


@pytest.fixture
def copilot_profile():
    """Resolve the registered Copilot profile via the provider registry."""
    # Importing ``model_tools`` triggers plugin discovery, which registers the
    # Copilot profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("copilot")
    assert profile is not None, "copilot provider profile must be registered"
    assert type(profile).__name__ == "CopilotProfile", (
        "expected the registered CopilotProfile, not a bare ProviderProfile"
    )
    return profile


class TestCopilotProfileReasoningWireShape:
    """``CopilotProfile.build_api_kwargs_extras`` emits Claude reasoning effort."""

    def test_claude_high_effort_is_emitted(self, copilot_profile):
        """The core regression: ``provider='copilot'`` + Claude + ``effort=high``.

        With the catalog advertising ``reasoning_effort`` for Claude, the
        profile must emit ``extra_body={'reasoning':{'effort':'high'}}``.

        Before the fix, the profile called the bare
        ``github_model_reasoning_efforts('claude-opus-4.8')`` -> ``[]`` and this
        assertion failed with ``extra_body == {}`` (reasoning dropped).
        """
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, top_level = copilot_profile.build_api_kwargs_extras(
                model="claude-opus-4.8",
                reasoning_config={"effort": "high"},
                supports_reasoning=True,
                api_key="dummy-copilot-token",
                base_url="https://api.githubcopilot.com",
            )

        assert extra_body == {"reasoning": {"effort": "high"}}
        assert top_level == {}

    def test_claude_defaults_to_medium_without_reasoning_config(self, copilot_profile):
        """Reasoning supported but no explicit config -> server-side default medium."""
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, top_level = copilot_profile.build_api_kwargs_extras(
                model="claude-opus-4.8",
                reasoning_config=None,
                supports_reasoning=True,
                api_key="dummy-copilot-token",
            )

        assert extra_body == {"reasoning": {"effort": "medium"}}
        assert top_level == {}

    def test_xhigh_effort_normalizes_to_high(self, copilot_profile):
        """Non-standard ``xhigh`` clamps to the nearest supported level (``high``)."""
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, _ = copilot_profile.build_api_kwargs_extras(
                model="claude-opus-4.8",
                reasoning_config={"effort": "xhigh"},
                supports_reasoning=True,
                api_key="dummy-copilot-token",
            )

        assert extra_body == {"reasoning": {"effort": "high"}}

    def test_unsupported_effort_is_not_emitted(self, copilot_profile):
        """An effort the catalog doesn't list is dropped, not forced onto the wire."""
        # Catalog advertises only low/medium/high/max, so 'ultra' is not valid.
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, _ = copilot_profile.build_api_kwargs_extras(
                model="claude-opus-4.8",
                reasoning_config={"effort": "ultra"},
                supports_reasoning=True,
                api_key="dummy-copilot-token",
            )

        assert extra_body == {}

    def test_non_reasoning_model_emits_nothing(self, copilot_profile):
        """A chat model without ``reasoning_effort`` support emits no reasoning key."""
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, top_level = copilot_profile.build_api_kwargs_extras(
                model="gpt-4.1",
                reasoning_config={"effort": "high"},
                supports_reasoning=True,
                api_key="dummy-copilot-token",
            )

        assert extra_body == {}
        assert top_level == {}

    def test_supports_reasoning_false_emits_nothing(self, copilot_profile):
        """The gate is respected: ``supports_reasoning=False`` -> no reasoning key.

        Even with a fully reasoning-capable catalog, an upstream gate of
        ``False`` must suppress the reasoning payload entirely.
        """
        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            extra_body, top_level = copilot_profile.build_api_kwargs_extras(
                model="claude-opus-4.8",
                reasoning_config={"effort": "high"},
                supports_reasoning=False,
                api_key="dummy-copilot-token",
            )

        assert extra_body == {}
        assert top_level == {}


class TestCopilotReasoningEndToEnd:
    """The live path: ``ChatCompletionsTransport.build_kwargs`` -> ``extra_body``."""

    def test_build_kwargs_emits_reasoning_for_copilot_claude(self, copilot_profile):
        """End-to-end proof requested in review.

        Drives the real transport entry point the agent uses for
        ``provider='copilot'`` and asserts the final ``chat.completions.create``
        kwargs carry ``extra_body={'reasoning':{'effort':'high'}}``.

        This is the assertion that fails on the unpatched PR branch: the
        transport delegates to the profile, the profile resolves ``[]``, and no
        ``extra_body`` is attached.
        """
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        with patch(
            "hermes_cli.models.fetch_github_model_catalog",
            return_value=_CLAUDE_CATALOG,
        ):
            api_kwargs = transport.build_kwargs(
                model="claude-opus-4.8",
                messages=messages,
                tools=None,
                provider_profile=copilot_profile,
                supports_reasoning=True,
                reasoning_config={"effort": "high"},
                api_key="dummy-copilot-token",
                base_url="https://api.githubcopilot.com",
                max_tokens_param_fn=lambda n: {"max_tokens": n},
            )

        assert "extra_body" in api_kwargs, (
            "reasoning extra_body was dropped on the live copilot path"
        )
        assert api_kwargs["extra_body"].get("reasoning") == {"effort": "high"}
