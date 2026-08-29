"""OpenCode Zen free-tier keyless routing (x-preview-f-free / "Ox Alpha").

The Zen relay serves ``*-free`` models ANONYMOUSLY: a request with no
Authorization header succeeds, while any non-empty bearer the relay doesn't
recognize — including our historical "no-key-required" placeholder and valid
OpenCode GO subscription keys — is rejected with 401 "Invalid API key".
The Go relay doesn't serve the free tier at all ("Model x is not supported").

These tests pin the keyless routing added for the community report where the
free Ox Alpha model failed under an OpenCode subscription:

1. ``is_opencode_zen_free_model`` recognizes free slugs (bare + prefixed).
2. ``opencode_zen_free_runtime`` pins free slugs to the Zen relay with the
   keyless placeholder + empty-Authorization headers, for BOTH family
   providers (Go selections heal to Zen).
3. ``resolve_runtime_provider`` routes free slugs keylessly with no
   OPENCODE_* credential present, and still fails closed for paid models.
4. The keyless placeholder never reaches the wire: client default_headers
   carry ``Authorization: ""`` overriding the SDK bearer.
"""

import os
from unittest import mock

import pytest

from hermes_cli.models import (
    OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
    is_opencode_zen_free_model,
    opencode_zen_free_headers,
    opencode_zen_free_runtime,
)


class TestFreeSlugDetection:
    def test_bare_free_slug(self):
        assert is_opencode_zen_free_model("x-preview-f-free")

    def test_provider_prefixed_slug(self):
        assert is_opencode_zen_free_model("opencode-zen/x-preview-f-free")

    def test_other_free_tier_slugs(self):
        for slug in (
            "hy3-free",
            "laguna-s-2.1-free",
            "mimo-v2.5-free",
            "nemotron-3-ultra-free",
        ):
            assert is_opencode_zen_free_model(slug), slug

    def test_paid_models_not_free(self):
        for slug in ("claude-sonnet-5", "glm-5.2", "kimi-k3", "gpt-5.6-sol"):
            assert not is_opencode_zen_free_model(slug), slug

    def test_empty_and_none(self):
        assert not is_opencode_zen_free_model("")
        assert not is_opencode_zen_free_model(None)

    def test_freedom_like_names_not_swept(self):
        # suffix match must be exact "-free", not substring "free"
        assert not is_opencode_zen_free_model("freeform-1")
        assert not is_opencode_zen_free_model("model-freedom")


class TestFreeRuntime:
    def test_zen_provider_free_model(self):
        rt = opencode_zen_free_runtime("opencode-zen", "hy3-free")
        assert rt is not None
        assert rt["base_url"] == "https://opencode.ai/zen/v1"
        assert rt["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert rt["api_mode"] == "chat_completions"
        assert rt["default_headers"]["Authorization"] == ""

    def test_go_provider_heals_to_zen(self):
        # Free slugs only exist on the Zen relay; a Go selection must be
        # routed to Zen (the Go relay rejects the model outright).
        rt = opencode_zen_free_runtime("opencode-go", "hy3-free")
        assert rt is not None
        assert rt["base_url"] == "https://opencode.ai/zen/v1"

    def test_go_ox_alpha_free_does_not_heal_to_zen(self):
        """ox-alpha-free is a KEYED Go-subscription model despite its -free
        suffix (Zen doesn't serve it; Go 401s anonymous). Membership in the
        verified keyless catalog — not the suffix — gates the heal."""
        assert opencode_zen_free_runtime("opencode-go", "ox-alpha-free") is None
        assert opencode_zen_free_runtime("opencode-zen", "ox-alpha-free") is None

    def test_paid_model_returns_none(self):
        assert opencode_zen_free_runtime("opencode-zen", "claude-sonnet-5") is None

    def test_non_opencode_provider_returns_none(self):
        assert opencode_zen_free_runtime("openrouter", "x-preview-f-free") is None
        assert opencode_zen_free_runtime(None, "x-preview-f-free") is None

    def test_headers_override_sdk_bearer(self):
        headers = opencode_zen_free_headers()
        assert headers["Authorization"] == ""
        assert headers["X-Title"] == "Hermes Agent"


class TestRuntimeProviderKeylessRouting:
    @pytest.fixture(autouse=True)
    def _no_opencode_creds(self, monkeypatch):
        for var in ("OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def _resolve(self, provider, model):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        with mock.patch(
            "hermes_cli.runtime_provider._get_model_config",
            return_value={"provider": provider, "model": model, "default": model},
        ):
            return resolve_runtime_provider(requested=provider, target_model=model)

    def test_zen_free_model_resolves_keyless(self):
        rt = self._resolve("opencode-zen", "hy3-free")
        assert rt["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert rt["base_url"] == "https://opencode.ai/zen/v1"
        assert rt["api_mode"] == "chat_completions"

    def test_go_free_model_resolves_keyless_on_zen(self):
        rt = self._resolve("opencode-go", "hy3-free")
        assert rt["api_key"] == OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER
        assert rt["base_url"] == "https://opencode.ai/zen/v1"

    def test_paid_model_still_fails_closed_without_key(self):
        from hermes_cli.auth import AuthError

        with pytest.raises(AuthError):
            self._resolve("opencode-zen", "claude-sonnet-5")


class TestKeylessProviderAlwaysAuthenticated:
    """opencode-free counts as authenticated everywhere, with zero keys.

    The provider is keyless: there is no credential to configure, so every
    surface that gates on auth (get_auth_status, provider:model listing,
    the /model picker source, the desktop explicit-only filter) must treat
    every install as logged in.
    """

    @pytest.fixture(autouse=True)
    def _no_creds(self, monkeypatch):
        for var in ("OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    def test_auth_status_logged_in(self):
        from hermes_cli.auth import get_auth_status

        st = get_auth_status("opencode-free")
        assert st["logged_in"] is True
        assert st["configured"] is True
        assert st["key_source"] == "keyless"

    def test_list_available_providers_authenticated(self):
        from hermes_cli.models import list_available_providers

        rows = {r["id"]: r["authenticated"] for r in list_available_providers()}
        assert rows.get("opencode-free") is True

    def test_picker_source_includes_provider_with_models(self):
        import model_tools  # noqa: F401 — plugin discovery
        from hermes_cli.model_switch import list_authenticated_providers

        provs = list_authenticated_providers(for_picker=True)
        free = [p for p in provs if p["slug"] == "opencode-free"]
        assert free, "opencode-free must appear in the picker with zero keys"
        assert free[0]["models"], "picker row must carry the curated models"

    def test_explicit_only_filter_keeps_keyless(self):
        from hermes_cli.inventory import _provider_is_keyless

        assert _provider_is_keyless("opencode-free") is True
        assert _provider_is_keyless("opencode-zen") is False
