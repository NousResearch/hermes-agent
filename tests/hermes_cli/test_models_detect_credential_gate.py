"""Auto-detection must never hand the user a provider they hold no credentials for.

Regression for the "accidental provider" class: ``/model <name>`` on provider A, where the name is
only known to provider B (static catalog or OpenRouter), used to switch the session to B even when
B had no key — an immediate 401 for most vendors, and for OpenRouter (whose runtime resolves with an
empty key instead of raising) a silent switch onto a metered aggregator.
"""

from __future__ import annotations

import pytest

from hermes_cli import models


@pytest.fixture
def no_live_catalog(monkeypatch):
    monkeypatch.setattr(models, "cached_provider_model_ids", lambda provider, **_: [])
    monkeypatch.setattr(models, "_find_openrouter_slug", lambda name: f"vendor/{name}")


@pytest.fixture
def authed(monkeypatch):
    """Pin which providers count as authenticated; everything else has no credentials."""
    from hermes_cli import models_detect

    granted: set[str] = set()
    monkeypatch.setattr(models_detect, "provider_has_credentials", lambda p: p in granted)
    return granted


class TestNoCredentialsNoSwitch:
    def test_openrouter_only_model_stays_when_no_openrouter_key(self, no_live_catalog, authed):
        assert models.detect_provider_for_model("some-model-only-openrouter-has", "deepseek") is None

    def test_openrouter_remap_allowed_with_key(self, no_live_catalog, authed):
        authed.add("openrouter")
        assert models.detect_provider_for_model("some-model-only-openrouter-has", "deepseek") == (
            "openrouter", "vendor/some-model-only-openrouter-has")

    def test_static_vendor_match_requires_that_vendors_credentials(self, no_live_catalog, authed, monkeypatch):
        monkeypatch.setattr(models, "detect_static_provider_for_model", lambda n, c: ("anthropic", n))
        assert models.detect_provider_for_model("claude-something", "deepseek") is None
        authed.add("anthropic")
        assert models.detect_provider_for_model("claude-something", "deepseek") == ("anthropic", "claude-something")

    def test_explicitly_named_provider_is_not_gated(self, no_live_catalog, authed, monkeypatch):
        """``/model nous`` names the provider: hand it back so the credential step can prompt/fail
        loudly instead of silently ignoring the request."""
        monkeypatch.setattr(models, "detect_static_provider_for_model", lambda n, c: ("nous", "hermes-4-405b"))
        assert models.detect_provider_for_model("nous", "deepseek") == ("nous", "hermes-4-405b")


class TestSharedSlugTiebreak:
    """A slug listed by several first-party catalogs goes to the one the user can use (#102775):
    ``gpt-5.6-luna`` sits in both ``openai-api`` and ``openai-codex``, and the first catalog hit
    used to be the only candidate — a Codex-only user was routed to a keyless ``openai-api``
    (``auto``) or left on the current provider (explicit switch)."""

    def test_shared_slug_goes_to_the_credentialed_sibling(self, no_live_catalog, authed):
        authed.add("openai-codex")
        assert models.detect_provider_for_model("gpt-5.6-luna", "deepseek") == ("openai-codex", "gpt-5.6-luna")
        assert models.detect_provider_for_model("gpt-5.6-luna", "auto") == ("openai-codex", "gpt-5.6-luna")

    def test_shared_slug_keeps_first_catalog_when_it_is_usable(self, no_live_catalog, authed):
        authed.update({"openai-api", "openai-codex"})
        assert models.detect_provider_for_model("gpt-5.6-luna", "deepseek") == ("openai-api", "gpt-5.6-luna")
        authed.clear()
        # Nothing usable anywhere: a fresh session still fails loudly on the first guess.
        assert models.detect_provider_for_model("gpt-5.6-luna", "auto") == ("openai-api", "gpt-5.6-luna")
