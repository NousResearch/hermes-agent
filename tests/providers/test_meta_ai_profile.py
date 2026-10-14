"""Tests for the bundled Meta Model API (Muse Spark) provider plugin.

Adapted from the plugin's original suite at
https://github.com/albertodepaola/hermes-meta-provider — the plugin is now
bundled, so profiles resolve through normal registry discovery.
"""


from providers import get_provider_profile
from providers.base import ProviderProfile


def _profile():
    p = get_provider_profile("meta-ai")
    assert p is not None
    return p


class TestMetaAIProfile:
    def test_images_ride_user_turns_not_tool_results(self):
        """Muse accepts images on user turns but 400s on image parts inside
        tool-result envelopes (#101668): vision must stay on while tool-message
        vision stays off, so tool screenshots are routed as text/user turns."""
        p = _profile()
        assert p.supports_vision is True
        assert p.supports_vision_tool_messages is False

    def test_carried_profile_defaults(self):
        """Defaults carried with the meta-oauth branch: aux model,
        token ceiling, and spark fallbacks of the bundled meta-ai profile."""
        p = _profile()
        assert p.default_aux_model == "muse-spark-1.2-contributor"
        assert p.default_max_tokens == 16384
        assert p.fallback_models == ("muse-spark-1.3", "muse-spark-1.2")

    def test_live_catalog_filters_non_chat_models(self, monkeypatch):
        p = _profile()
        seen = []

        def fake_fetch_models(_self, **_kwargs):
            seen.append(True)
            return [
                "muse-voice-transcribe-1.0",
                "muse-spark-latest",
                "muse-image-1.0-eval",
                "muse-nova-test",
            ]

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch_models)

        assert p.fetch_models() == ["muse-spark-latest", "muse-nova-test"]
        assert seen



def _effort(cfg):
    _eb, top = _profile().build_api_kwargs_extras(reasoning_config=cfg)
    return top.get("reasoning_effort")


class TestReasoningEffort:
    def test_mapping(self):
        assert _effort(None) == "medium"  # unset -> medium
        assert _effort({"enabled": True, "effort": "high"}) == "high"
        assert _effort({"enabled": True, "effort": "low"}) == "low"
        assert _effort({"enabled": True, "effort": "max"}) == "xhigh"
        assert _effort({"enabled": False}) == "minimal"  # disabled -> minimal
        # Meta 400s on "none" — must map to "minimal"
        assert _effort({"effort": "none"}) == "minimal"

    def test_never_uses_extra_body(self):
        """The dial must be top-level, not extra_body.reasoning (core-gated path)."""
        eb, top = _profile().build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert eb == {}
        assert "reasoning_effort" in top


def _oauth_profile():
    p = get_provider_profile("meta-oauth")
    assert p is not None
    return p


class TestMetaOAuthProfile:
    """The subscription provider (core login in hermes_cli/auth_meta.py) must
    carry the same Meta wire quirks as meta-ai, or subscription turns fall
    back to generic defaults the endpoint rejects."""

    def test_images_ride_user_turns_not_tool_results(self):
        from providers import routed_model_rejects_vision_tool_messages

        p = _oauth_profile()
        assert p.supports_vision is True
        assert p.supports_vision_tool_messages is False
        # The routing helper the vision prep actually consults must agree:
        # api.meta.ai 400s on image parts inside tool-result envelopes.
        assert routed_model_rejects_vision_tool_messages("meta-oauth", "muse-spark-1.3") is True

    def test_reasoning_and_token_defaults_match_meta_ai(self):
        p = _oauth_profile()
        assert p.default_max_tokens == _profile().default_max_tokens
        _eb, top = p.build_api_kwargs_extras(reasoning_config={"enabled": False})
        assert top.get("reasoning_effort") == "minimal"  # Meta 400s on "none"
        _eb, top = p.build_api_kwargs_extras(reasoning_config={"effort": "none"})
        assert top.get("reasoning_effort") == "minimal"

    def test_live_catalog_filters_non_chat_models(self, monkeypatch):
        p = _oauth_profile()

        def fake_fetch_models(_self, **_kwargs):
            return ["muse-voice-transcribe-1.0", "muse-spark-latest", "muse-image-1.0-eval"]

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch_models)
        assert p.fetch_models() == ["muse-spark-latest"]

    def test_fallback_models_mirror_static_catalog(self):
        """Profile fallbacks and the static picker rows are two spellings of
        one curated list; drifting apart would offer different models
        depending on which surface reads first."""
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        assert set(_oauth_profile().fallback_models) == set(_PROVIDER_MODELS["meta-oauth"])

    def test_models_dev_mapping_matches_meta_ai(self):
        """Subscription turns must resolve the same models.dev catalog (and
        1M context window) as API-key turns, like xai-oauth -> xai."""
        from agent.models_dev import PROVIDER_TO_MODELS_DEV

        assert PROVIDER_TO_MODELS_DEV["meta-oauth"] == PROVIDER_TO_MODELS_DEV["meta-ai"]

    def test_profile_does_not_hijack_core_auth(self):
        """Registering the profile must not reroute auth: the core registry
        row and bespoke login stay authoritative, and `hermes auth add
        meta-oauth` must not trip the plugin missing-auth_handler error."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.auth_plugin_providers import (
            PLUGIN_MIRRORED_PROVIDERS, plugin_missing_auth_handler_error)

        assert PROVIDER_REGISTRY["meta-oauth"].auth_type == "oauth_external"
        assert "meta-oauth" not in PLUGIN_MIRRORED_PROVIDERS
        assert plugin_missing_auth_handler_error("meta-oauth", "add") is None
