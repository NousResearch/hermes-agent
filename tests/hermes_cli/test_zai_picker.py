"""Tests for the Z.AI / GLM provider picker integration.

Covers the GLM-5.2 addition (2026-06-13 — see
https://docs.z.ai/devpack/latest-model). Verifies:

  * The zai provider profile has the right `models_url` for the catalog probe.
  * `_PROVIDER_MODELS["zai"]` lists `glm-5.2` as the newest flagship.
  * `provider_model_ids("zai")` returns glm-5.2 even with no API key
    (the curated-catalog fallback path).
  * With a mocked live `/v1/models` response, glm-5.2 surfaces from
    the live fetch path.
  * `validate_requested_model("glm-5.2", "zai")` accepts the model
    against the curated catalog when the API is unreachable.

These are smoke tests; the deeper contract (gateway /v1/models caching,
picker-row formatting, OAuth-vs-api_key gating) is covered by
``test_api_key_providers.py`` and ``test_models_dev_preferred_merge.py``.
"""

from unittest.mock import patch

import pytest

from hermes_cli.models import (
    _MODELS_DEV_PREFERRED,
    _PROVIDER_MODELS,
    provider_model_ids,
    validate_requested_model,
)


# ---------------------------------------------------------------------------
# Provider profile
# ---------------------------------------------------------------------------


class TestZaiProviderProfile:
    def _profile(self):
        from providers import get_provider_profile
        p = get_provider_profile("zai")
        assert p is not None, "zai profile not registered — zai plugin broken"
        return p

    def test_zai_profile_loaded(self):
        assert self._profile().name == "zai"

    def test_zai_base_url(self):
        assert self._profile().base_url == "https://api.z.ai/api/paas/v4"

    def test_zai_models_url_set_explicitly(self):
        """Per model-provider-picker-integration skill: an explicit
        `models_url` is the defense-in-depth contract so the catalog
        probe doesn't accidentally hit `<inference_base>/models` if
        the base_url is ever changed."""
        assert self._profile().models_url == "https://api.z.ai/api/paas/v4/models"

    def test_zai_env_vars(self):
        # Order matters — auth.py checks each in sequence.
        assert self._profile().env_vars[:3] == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")

    def test_zai_fallback_models_lead_with_glm52(self):
        # Newest flagship must be the first fallback so /model picker
        # highlights it on no-key / 401 / network failure.
        fm = self._profile().fallback_models
        assert fm[0] == "glm-5.2"
        assert "glm-5.2" in fm

    def test_zai_default_aux_model_keeps_cheap(self):
        """Auxiliary tasks (compression, etc.) should NOT use the
        flagship glm-5.2 — overkill for summarisation."""
        aux = self._profile().default_aux_model
        assert aux != "glm-5.2"
        assert "flash" in aux or "air" in aux


# ---------------------------------------------------------------------------
# Static catalog (_PROVIDER_MODELS)
# ---------------------------------------------------------------------------


class TestZaiStaticCatalog:
    def test_zai_catalog_contains_glm52(self):
        assert "zai" in _PROVIDER_MODELS
        assert "glm-5.2" in _PROVIDER_MODELS["zai"]

    def test_zai_catalog_preserves_prior_flagships(self):
        # Regression guard: adding glm-5.2 must NOT drop the M-series /
        # older entries users were already picking.
        catalog = _PROVIDER_MODELS["zai"]
        for prior in ("glm-5.1", "glm-5", "glm-4.7", "glm-4.5-flash"):
            assert prior in catalog, f"{prior} missing from zai catalog"

    def test_zai_is_in_models_dev_preferred(self):
        """The live `/v1/models` merge must be enabled for zai so that
        a user with a ZAI_API_KEY sees fresh models without waiting
        for a Hermes release."""
        assert "zai" in _MODELS_DEV_PREFERRED


# ---------------------------------------------------------------------------
# provider_model_ids() — offline / no-key fallback
# ---------------------------------------------------------------------------


class TestZaiProviderModelIdsOffline:
    """With no API key reachable AND no models.dev data, the picker
    must still surface glm-5.2 via the curated catalog floor."""

    def _no_key(self, monkeypatch=None):
        """Patch the credential resolver at its SOURCE (hermes_cli.auth)
        because the models.py consumer imports it locally inside the
        function — patching the destination attribute doesn't reach it."""
        return patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "", "base_url": "", "source": "test"},
        )

    def test_glm52_appears_without_api_key(self, monkeypatch):
        # Don't change HERMES_HOME — the real one is fine.
        with self._no_key(monkeypatch):
            with patch("agent.models_dev.list_agentic_models", return_value=[]):
                out = provider_model_ids("zai")
        assert "glm-5.2" in out

    def test_provider_model_ids_uses_fallback_models_when_no_key(self, monkeypatch):
        """Per hermes_cli/models.py line 2245-2246: when no api_key
        resolves, the profile's `fallback_models` wins. glm-5.2 is
        first in that list."""
        with self._no_key(monkeypatch):
            with patch("agent.models_dev.list_agentic_models", return_value=[]):
                out = provider_model_ids("zai")
        # fallback_models is the floor; static catalog + models.dev
        # merge may add more — but glm-5.2 must be present and at or
        # near the top.
        assert "glm-5.2" in out
        assert out.index("glm-5.2") <= 2


# ---------------------------------------------------------------------------
# provider_model_ids() — live fetch path
# ---------------------------------------------------------------------------


class TestZaiProviderModelIdsLiveFetch:
    """When a ZAI_API_KEY is set, the profile-based generic path in
    `hermes_cli.models.provider_model_ids` calls
    `zai.fetch_models(api_key=...)`. We mock urllib via the same
    pattern used in test_minimax_picker.py — see
    model-provider-picker-integration/SKILL.md pitfall 1."""

    def test_live_fetch_surfaces_glm52(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-key-for-picker")

        # Real zai /v1/models response shape (openai-compat).
        live_body = (
            b'{"object": "list", "data": ['
            b'  {"id": "glm-5.2"},'
            b'  {"id": "glm-5.1"},'
            b'  {"id": "glm-5"}'
            b']}'
        )

        from providers import get_provider_profile
        profile = get_provider_profile("zai")
        assert profile is not None  # narrowed for the closure below

        def fake_fetch_models(*, api_key=None, timeout=8.0):
            return ["glm-5.2", "glm-5.1", "glm-5"]

        with patch.object(profile, "fetch_models", side_effect=fake_fetch_models):
            # Bypass the models.dev merge so we test only the live path.
            with patch("agent.models_dev.list_agentic_models", return_value=[]):
                out = provider_model_ids("zai")

        assert "glm-5.2" in out
        # And it should appear before older variants from the curated
        # catalog — the live list comes first per
        # `_merge_with_models_dev` semantics.
        glm52_idx = out.index("glm-5.2")
        glm5_idx = out.index("glm-5")
        assert glm52_idx < glm5_idx


# ---------------------------------------------------------------------------
# validate_requested_model — glm-5.2 acceptance
# ---------------------------------------------------------------------------


def _validate(model: str, provider: str = "zai"):
    return validate_requested_model(model, provider, api_key=None, base_url=None)


class TestValidateZaiGlm52:
    def test_glm52_accepted_via_static_catalog_when_api_down(self):
        """GLM-5.2 is in the curated catalog — even when /v1/models is
        unreachable, the static fallback should accept the model with
        `recognized=True`."""
        with patch("hermes_cli.models.provider_model_ids",
                   return_value=_PROVIDER_MODELS["zai"]):
            result = _validate("glm-5.2")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_glm52_accepted_against_live_catalog(self):
        """Live catalog returns glm-5.2 → validator returns recognized
        without the static fallback kicking in."""
        live = ["glm-5.2", "glm-5.1", "glm-5", "glm-4.7"]
        with patch("hermes_cli.models.provider_model_ids", return_value=live):
            result = _validate("glm-5.2")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True

    def test_unknown_zai_model_soft_accepted(self):
        """A model not in any catalog (e.g. a future glm-5.3 not yet
        published to the static list) is still accepted with a note —
        mirrors the comment at models.py:3768-3771 about Z.AI Pro/Max
        plans having access to models not in /models."""
        with patch("hermes_cli.models.provider_model_ids", return_value=[]):
            result = _validate("glm-5.3-future-flagship")
        assert result["accepted"] is True
        assert result["persist"] is True
        # Not recognized because not in any catalog.
        assert result["recognized"] is False
