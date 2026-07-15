"""Unit tests for the SCX.ai provider profile.

SCX.ai is a plain OpenAI-compatible api-key provider (sovereign Australian
inference), so this verifies the profile is registered correctly and wires the
expected identity, endpoint, auth, and catalog fields — the contract every
downstream layer (auth, models, doctor, runtime_provider, transport) reads
from.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def scx_profile():
    """Resolve the registered SCX profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    SCX profile. Going through ``get_provider_profile`` keeps the test honest
    about the actual registration path (name + alias resolution).
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("scx")
    assert profile is not None, "scx provider profile must be registered"
    return profile


class TestScxProfile:
    def test_identity_and_endpoint(self, scx_profile):
        assert scx_profile.name == "scx"
        assert scx_profile.api_mode == "chat_completions"
        assert scx_profile.auth_type == "api_key"
        assert scx_profile.base_url == "https://api.scx.ai/v1"
        assert scx_profile.get_hostname() == "api.scx.ai"

    def test_alias_resolves(self, scx_profile):
        import providers

        assert providers.get_provider_profile("scx-ai") is scx_profile

    def test_env_vars(self, scx_profile):
        # API key first, optional base-url override second (priority order).
        assert scx_profile.env_vars == ("SCX_API_KEY", "SCX_BASE_URL")

    def test_catalog_is_curated_flagships_only(self, scx_profile):
        # The picker is pinned to the curated flagship list: coder is the
        # setup default (entry [0], also the aux model), MAGPiE is SCX's
        # Australian-context flagship, MiniMax-M2.7 the hosted agentic model.
        assert scx_profile.fallback_models == ("coder", "MAGPiE", "MiniMax-M2.7")

    def test_no_live_catalog_merge(self, scx_profile):
        # fetch_models is disabled so the live /v1/models catalog (hosted
        # open-weights models, embeddings, STT, moderation) never merges
        # into the picker — the curated flagship list is authoritative.
        # Other hosted models remain usable when typed explicitly.
        assert scx_profile.fetch_models(api_key="scx-test-key") is None

    def test_aux_model_is_coder(self, scx_profile):
        # Cheap/fast side-task model (compression, session search, vision).
        assert scx_profile.default_aux_model == "coder"

    def test_no_provider_specific_request_knobs(self, scx_profile):
        # SCX is a standard chat-completions endpoint: no extra_body additions,
        # no top-level api_kwargs, no forced headers beyond auth.
        assert scx_profile.build_extra_body() == {}
        assert scx_profile.build_api_kwargs_extras() == ({}, {})
        assert scx_profile.default_headers == {}
