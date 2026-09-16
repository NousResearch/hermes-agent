"""Focused tests for Cheaper Inference provider wiring.

Cheaper Inference is an OpenAI-compatible gateway that serves the *same* model
ids other vendors publish (``glm-5.3``, ``kimi-k3``, ...) from whichever upstream
is cheapest, sometimes with a smaller context window than the vendor's own
endpoint. The contracts below cover the two ways that goes wrong: a half-wired
provider id (auth resolves but ``/model`` parsing or the resolver does not — see
``website/docs/developer-guide/adding-providers.md``), and a window borrowed from
the vendor's family default that is LARGER than the gateway actually serves.
"""

from __future__ import annotations

import sys
import types


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    sys.modules["dotenv"] = fake_dotenv


PROVIDER = "cheaperinference"
BASE_URL = "https://api.cheaperinference.com/v1"
ALIASES = ("cheaper-inference", "cheaperinference.com")


def _profile():
    from providers import get_provider_profile

    profile = get_provider_profile(PROVIDER)
    assert profile is not None, "cheaperinference profile did not register"
    return profile


class TestProfile:
    def test_profile_declares_endpoint_and_credential(self):
        profile = _profile()
        assert profile.name == PROVIDER
        assert profile.display_name == "Cheaper Inference"
        assert profile.base_url == BASE_URL
        assert profile.auth_type == "api_key"
        # The gateway's primary surface is chat/completions; its /v1/responses layer
        # is reachable through the codex transport, which already pins store=false.
        assert profile.api_mode == "chat_completions"
        assert profile.env_vars[0] == "CHEAPERINFERENCE_API_KEY"
        assert profile.get_hostname() == "api.cheaperinference.com"

    def test_curated_models_are_bare_ids(self):
        """The gateway takes the vendor's own id, not an aggregator ``vendor/model``
        slug — a prefixed entry here would 404 on every picker selection."""
        models = _profile().fallback_models
        assert models
        assert not [m for m in models if "/" in m]

    def test_aux_model_is_a_curated_model(self):
        """A ``default_aux_model`` outside the catalog spends a round-trip 404ing on
        every compression/vision/titling call before the retry net catches it."""
        profile = _profile()
        assert profile.default_aux_model in profile.fallback_models


class TestProviderIdAgreesEverywhere:
    """One canonical id across auth, catalog and resolver: if they disagree the
    provider authenticates while ``/model`` or runtime resolution silently misses."""

    def test_aliases_resolve_in_every_table(self):
        from hermes_cli.auth import resolve_provider
        from hermes_cli.models import _PROVIDER_ALIASES, normalize_provider
        from hermes_cli.providers import normalize_provider as normalize_in_providers

        for alias in (*ALIASES, PROVIDER):
            assert resolve_provider(alias) == PROVIDER
            assert normalize_provider(alias) == PROVIDER
            assert normalize_in_providers(alias) == PROVIDER
        for alias in ALIASES:
            assert _PROVIDER_ALIASES[alias] == PROVIDER

    def test_profile_aliases_match_the_static_tables(self):
        """Every alias the profile declares must also resolve through the static
        tables — a profile-only alias works for auth and fails in ``/model``."""
        from hermes_cli.models import normalize_provider

        for alias in _profile().aliases:
            assert normalize_provider(alias) == PROVIDER

    def test_provider_reaches_the_picker_and_labels(self):
        from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS
        from hermes_cli.providers import get_label

        assert PROVIDER in {p.slug for p in CANONICAL_PROVIDERS}
        assert _PROVIDER_LABELS[PROVIDER] == "Cheaper Inference"
        assert get_label(PROVIDER) == "Cheaper Inference"


class TestResolver:
    def test_resolve_provider_full_recognizes_the_provider(self):
        """None here means a configured ``provider: cheaperinference`` is discarded
        and env auto-detect wins instead (the Upstage regression class)."""
        from hermes_cli.providers import resolve_provider_full

        pdef = resolve_provider_full(PROVIDER, {}, [])
        assert pdef is not None
        assert pdef.id == PROVIDER
        assert pdef.base_url == BASE_URL
        assert "CHEAPERINFERENCE_API_KEY" in pdef.api_key_env_vars

    def test_gateway_is_flagged_as_an_aggregator(self):
        """It re-exposes other vendors' models, so model-switch must search its
        catalog rather than assume first-party ids."""
        from hermes_cli.providers import is_aggregator

        assert is_aggregator(PROVIDER)

    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import determine_api_mode, host_mandated_api_mode

        assert determine_api_mode(PROVIDER, BASE_URL) == "chat_completions"
        # The host serves both wires, so it must NOT be a host mandate: a mandate
        # would override an explicitly configured Responses route and vice versa.
        assert host_mandated_api_mode(BASE_URL) is None

    def test_credentials_honour_the_base_url_override(self, monkeypatch):
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials

        pconfig = PROVIDER_REGISTRY[PROVIDER]
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == BASE_URL
        assert "CHEAPERINFERENCE_API_KEY" in pconfig.api_key_env_vars
        assert pconfig.base_url_env_var == "CHEAPERINFERENCE_BASE_URL"

        monkeypatch.setenv("CHEAPERINFERENCE_API_KEY", "ci-secret")
        monkeypatch.setenv("CHEAPERINFERENCE_BASE_URL", "https://proxy.example/v1")
        creds = resolve_api_key_provider_credentials(PROVIDER)
        assert creds["provider"] == PROVIDER
        assert creds["api_key"] == "ci-secret"
        assert creds["base_url"] == "https://proxy.example/v1"


class TestEnvCatalog:
    """The dashboard/desktop Providers page lists only OPTIONAL_ENV_VARS keys whose
    category is "provider"; without them the gateway stays invisible there."""

    def test_optional_env_vars_expose_the_provider(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        key = OPTIONAL_ENV_VARS["CHEAPERINFERENCE_API_KEY"]
        assert key["category"] == "provider"
        assert key["password"] is True
        assert key["url"]
        base = OPTIONAL_ENV_VARS["CHEAPERINFERENCE_BASE_URL"]
        assert base["category"] == "provider"
        assert base["password"] is False


class TestModelCatalog:
    def test_picker_falls_back_to_curated_models_when_the_catalog_is_unreachable(self, monkeypatch):
        from hermes_cli.models import provider_model_ids

        profile = _profile()
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id, "api_key": "ci-live-key", "base_url": BASE_URL,
                "source": "CHEAPERINFERENCE_API_KEY",
            },
        )
        monkeypatch.setattr(profile, "fetch_models", lambda **_kw: None)

        assert provider_model_ids(PROVIDER) == list(profile.fallback_models)

    def test_live_catalog_merges_after_the_curated_list_without_duplicates(self, monkeypatch):
        """Curated-first merge policy (#46309): curated entries lead, live-only ids
        are appended, and a live duplicate of a curated id is not repeated."""
        from hermes_cli.models import provider_model_ids

        profile = _profile()
        curated = list(profile.fallback_models)
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id, "api_key": "ci-live-key", "base_url": BASE_URL,
                "source": "CHEAPERINFERENCE_API_KEY",
            },
        )
        monkeypatch.setattr(
            profile, "fetch_models", lambda **_kw: [curated[-1], "brand-new-upstream-model"],
        )

        result = provider_model_ids(PROVIDER)
        assert result == curated + ["brand-new-upstream-model"]


class TestModelsDevMapping:
    def test_provider_maps_into_the_models_dev_registry_both_ways(self):
        """The gateway has its own models.dev provider entry; without the mapping
        every capability/limit lookup for it misses the registry entirely."""
        from agent.models_dev import PROVIDER_TO_MODELS_DEV, _models_dev_to_hermes_ids

        assert PROVIDER in PROVIDER_TO_MODELS_DEV
        assert PROVIDER in _models_dev_to_hermes_ids(PROVIDER_TO_MODELS_DEV[PROVIDER])


class TestContextWindows:
    """Endpoint-scoped windows must beat the vendor family defaults.

    ``DEFAULT_CONTEXT_LENGTHS`` is keyed on the model NAME, so a gateway serving
    ``glm-5.3`` inherits Z.AI's larger window unless the endpoint says otherwise —
    and an over-reported window is packed past what the gateway accepts.
    """

    def _scoped_rows(self):
        from agent.model_metadata import _ENDPOINT_SCOPED_CONTEXT

        return [row for row in _ENDPOINT_SCOPED_CONTEXT if row[0] == "api.cheaperinference.com"]

    def test_endpoint_rows_win_over_the_family_default(self):
        from agent.model_metadata import get_model_context_length

        rows = self._scoped_rows()
        assert rows, "no endpoint-scoped context rows for the gateway"
        for _host, _paths, models, ctx in rows:
            for model in models:
                assert get_model_context_length(model, base_url=BASE_URL, provider=PROVIDER) == ctx

    def test_known_windows_match_the_published_catalog(self):
        """The rows above are checked against themselves, so they would still agree
        with a wrong number. These are the gateway's published windows, spot-checked
        by hand across the scoped table and the family table."""
        from agent.model_metadata import get_model_context_length

        published = {
            # scoped: the gateway serves less than the vendor's own endpoint
            "aion-labs.aion-2-0": 128_000,
            "claude-opus-4.5": 198_000,
            "deepseek-v4.1-flash": 1_048_576,
            "glm-5.3": 1_000_000,
            "gpt-5.5": 1_000_000,
            "minimax-m2.7": 198_000,
            "qwen3.6-27b": 256_000,
            # unscoped: the family default is already what the gateway serves
            "claude-opus-5": 1_000_000,
            "grok-4.5": 500_000,
        }
        for model, ctx in published.items():
            assert get_model_context_length(model, base_url=BASE_URL, provider=PROVIDER) == ctx, model

    def test_every_scoped_row_actually_corrects_the_family_default(self):
        """A row whose window already equals the family default is dead weight — the
        table is for divergences only. This is what keeps the list from growing into
        a second copy of the catalog as models.dev fills in."""
        from agent.model_metadata import (
            DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT, _longest_key_match,
        )

        for _host, _paths, models, ctx in self._scoped_rows():
            for model in models:
                hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
                family = hit[1] if hit else DEFAULT_FALLBACK_CONTEXT
                assert ctx != family, f"{model} scoped window duplicates the family default {family}"

    def test_models_without_a_scoped_row_keep_the_family_default(self):
        """The complement of the row above: a curated model the table does not name
        must resolve from the shared family table, so the two sources stay disjoint
        and nothing silently shadows a vendor window."""
        from agent.model_metadata import (
            DEFAULT_CONTEXT_LENGTHS, DEFAULT_FALLBACK_CONTEXT, _longest_key_match,
            get_model_context_length,
        )

        scoped = {m for _h, _p, models, _c in self._scoped_rows() for m in models}
        unscoped = [m for m in _profile().fallback_models if m not in scoped]
        assert unscoped, "expected some curated models to resolve from the family table"
        for model in unscoped:
            hit = _longest_key_match(DEFAULT_CONTEXT_LENGTHS, model.lower())
            family = hit[1] if hit else DEFAULT_FALLBACK_CONTEXT
            assert get_model_context_length(model, base_url=BASE_URL, provider=PROVIDER) == family

    def test_every_curated_model_clears_the_minimum_window(self):
        """agent_init rejects a session whose window is below the minimum, so a
        curated model that resolves under it is unusable from the picker."""
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length

        for model in _profile().fallback_models:
            ctx = get_model_context_length(model, base_url=BASE_URL, provider=PROVIDER)
            assert ctx >= MINIMUM_CONTEXT_LENGTH, f"{model} resolved {ctx}"


class TestModelNormalization:
    def test_redundant_provider_prefix_is_stripped(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        for prefix in (PROVIDER, *ALIASES):
            assert normalize_model_for_provider(f"{prefix}/glm-5.3", PROVIDER) == "glm-5.3"

    def test_a_foreign_vendor_prefix_is_left_alone(self):
        """The gateway's own ids are bare, but nothing in its docs says a
        ``vendor/model`` form is rejected — so the name goes to the wire as typed
        rather than being silently rewritten."""
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert normalize_model_for_provider("anthropic/claude-opus-5", PROVIDER) == "anthropic/claude-opus-5"
