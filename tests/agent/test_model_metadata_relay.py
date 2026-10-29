"""Relayed (gateway) context resolution — #142334.

A relay base URL (``https://llm.sgh.ng/v1``) is not a canonical provider host, so the host-keyed
tiers in :func:`get_model_context_length` all missed: :data:`_ENDPOINT_SCOPED_CONTEXT` matched
host+path only, and the registered provider-profile hook was asked for ``get_provider_profile
('auto')`` — ``None`` — so a picker-started session never reached it. Both left a model served
through a gateway on the 256K hard fallback (or a family catch-all) even though the provider had
declared its real window.

Contract pinned here:

* canonical host+path matching stays PRIMARY (a canonical host with another path never leaks);
* the same table answers by ``(provider, model)`` when the host is not canonical — the relay
  fallback, including vendor-prefixed relay ids (``kimi/k3``);
* ``auto``/blank resolves to a concrete provider before the profile hook;
* ``catalog_provider`` on a custom route names the upstream vendor for those lookups;
* an unknown model through a relay still lands on the 256K fallback.
"""

from contextlib import ExitStack
from unittest.mock import patch

import pytest

import agent.model_metadata as mm
import agent.model_metadata_relay as relay
from agent.model_metadata import get_model_context_length

RELAY = "https://llm.sgh.ng/v1"
KIMI_CTX = 1_048_576


def _quiet_tiers():
    """Silence every network/probe tier so a failed expectation returns a local verdict instead of
    dialing the relay. Values are the "nothing known" answers."""
    stack = ExitStack()
    for target, value in [
        ("agent.model_metadata.get_cached_context_length", None),
        ("agent.model_metadata.fetch_model_metadata", {}),
        ("agent.model_metadata.fetch_endpoint_model_metadata", {}),
        ("agent.model_metadata._query_ollama_api_show", None),
        ("agent.model_metadata._query_anthropic_context_length", None),
        ("agent.model_metadata._resolve_endpoint_context_length", None),
        ("agent.model_metadata._query_local_context_length", None),
        ("agent.models_dev.lookup_models_dev_context", None),
    ]:
        stack.enter_context(patch(target, return_value=value))
    return stack


@pytest.fixture(autouse=True)
def _reset_fallback_warnings():
    mm._FALLBACK_WARNED.clear()
    yield
    mm._FALLBACK_WARNED.clear()


class TestProviderScopedEndpointTable:
    """Part 1: ``_ENDPOINT_SCOPED_CONTEXT`` answers by provider when the host is not canonical."""

    def test_table_entries_carry_owning_provider(self):
        """Every entry names the provider that owns the window (the relay fallback key)."""
        for entry in relay._ENDPOINT_SCOPED_CONTEXT:
            host, paths, models, ctx, provider = entry
            assert isinstance(host, str) and host
            assert isinstance(paths, (set, frozenset)) and paths
            assert isinstance(models, (set, frozenset)) and models
            assert isinstance(ctx, int) and ctx > 0
            assert isinstance(provider, str) and provider

    def test_relayed_k3_resolves_through_provider(self):
        """``k3`` + ``kimi-coding`` resolves 1 Mi behind ANY relay base URL."""
        with _quiet_tiers():
            assert get_model_context_length(
                "k3", base_url=RELAY, provider="kimi-coding"
            ) == KIMI_CTX

    def test_relayed_k3_resolves_for_local_gateway_hosts(self):
        """The fallback is host-agnostic: two unrelated relay hosts both resolve."""
        with _quiet_tiers():
            for base_url in ("https://llm.sgh.ng/v1", "http://127.0.0.1:8090/v1"):
                assert get_model_context_length(
                    "k3", base_url=base_url, provider="kimi-coding"
                ) == KIMI_CTX

    def test_direct_k3_unchanged(self):
        """No regression on the canonical endpoint."""
        with _quiet_tiers():
            for base_url in (
                "https://api.kimi.com/coding",
                "https://api.kimi.com/coding/v1",
            ):
                assert get_model_context_length(
                    "k3", base_url=base_url, provider="kimi-coding"
                ) == KIMI_CTX

    def test_canonical_host_still_wins_over_provider(self):
        """A canonical host with another path is NOT the scoped surface, even with the provider."""
        with _quiet_tiers():
            for base_url in ("https://api.kimi.com/v1", "https://api.kimi.com/coding/other"):
                assert get_model_context_length(
                    "k3", base_url=base_url, provider="kimi-coding"
                ) != KIMI_CTX

    def test_unknown_provider_does_not_claim_the_kimi_entry(self):
        """Another provider's id must not borrow Kimi's window through a relay."""
        with _quiet_tiers():
            for provider in ("openrouter", "deepseek", "custom"):
                assert get_model_context_length(
                    "k3", base_url=RELAY, provider=provider
                ) != KIMI_CTX

    def test_vendor_prefixed_relay_id_resolves(self):
        """Gateways re-serve upstream ids vendor-prefixed (``kimi/k3``); the provider still owns it."""
        with _quiet_tiers():
            assert get_model_context_length(
                "kimi/k3", base_url=RELAY, provider="kimi-coding"
            ) == KIMI_CTX

    def test_unknown_model_through_relay_keeps_the_fallback(self):
        """A model the table does not know stays on the 256K fallback, relay or not."""
        with _quiet_tiers():
            assert get_model_context_length(
                "totally-unknown-model-xyz", base_url=RELAY, provider="kimi-coding"
            ) == mm.DEFAULT_FALLBACK_CONTEXT

    def test_helper_matches_provider_only_off_canonical_host(self):
        """Unit-level: the helper is provider-aware, and canonical hosts are path-strict."""
        assert mm._endpoint_scoped_context_length(
            "k3", RELAY, provider="kimi-coding"
        ) == KIMI_CTX
        assert mm._endpoint_scoped_context_length("k3", RELAY) is None
        assert mm._endpoint_scoped_context_length(
            "k3", "https://api.kimi.com/coding", provider="kimi-coding"
        ) == KIMI_CTX
        assert mm._endpoint_scoped_context_length(
            "k3", "https://api.kimi.com/v1", provider="kimi-coding"
        ) is None


class TestScopedWindowsEndToEnd:
    """The two scoped bounds, resolved through ``get_model_context_length`` — canonical and relayed.

    Moved here from ``test_model_metadata.py`` when the table and its lookup moved into
    ``agent.model_metadata_relay``: these cases are about the relay fallback, so they belong with
    the rest of the #142334 contract.
    """

    def test_nvidia_deepseek_v4_pro_context_is_endpoint_scoped(self):
        """NVIDIA's 262K NIM window must not lower DeepSeek V4 globally."""
        with _quiet_tiers():
            accepted_urls = (
                "https://integrate.api.nvidia.com/v1",
                "https://INTEGRATE.API.NVIDIA.COM/v1/",
                "https://integrate.api.nvidia.com:443/v1",
            )
            # Relay hosts (unknown to _URL_TO_PROVIDER) answer by provider too — a gateway that
            # fronts NIM still serves NVIDIA's 262K (#142334).
            relay_urls = ("https://gw.example.com/v1", "http://127.0.0.1:8090/v1")
            rejected_urls = (
                "http://integrate.api.nvidia.com/v1",
                "https://integrate.api.nvidia.com:8443/v1",
                "https://integrate.api.nvidia.com/v1/other",
                "https://integrate.api.nvidia.com/v1?route=other",
                # Another provider's own host is never a relay for NVIDIA's bound.
                "https://api.deepseek.com/v1",
                "https://openrouter.ai/api/v1",
            )

            for base_url in accepted_urls + relay_urls:
                assert get_model_context_length(
                    "deepseek-ai/deepseek-v4-pro",
                    provider="nvidia",
                    base_url=base_url,
                ) == 262_144

            for base_url in rejected_urls:
                assert get_model_context_length(
                    "deepseek-ai/deepseek-v4-pro",
                    provider="nvidia",
                    base_url=base_url,
                ) == 1_000_000

    def test_k3_context_is_scoped_to_confirmed_coding_endpoint(self):
        """The bare ``k3`` slug's 1 Mi context must not leak to unverified endpoints.

        The named ``kimi-k3`` / ``kimi-k3-cot`` slugs resolve to 1 Mi
        EVERYWHERE via DEFAULT_CONTEXT_LENGTHS — the window is a property of
        the model, served at 1M on api.moonshot.ai and api.moonshot.cn alike
        (verified against models.dev + OpenRouter live metadata). Only the
        bare ``k3`` slug, which exists solely on the Kimi Coding Plan
        endpoint, stays scoped: to that canonical host+path, plus (behind a
        relay whose host belongs to no known provider) to the owning
        ``kimi-coding`` provider — a gateway that fronts the coding plan
        serves the same 1 Mi window (#142334).
        """
        with _quiet_tiers():
            accepted_urls = (
                "https://api.kimi.com/coding",
                "https://API.KIMI.COM/coding/",
                "https://api.kimi.com:443/coding",
                "https://api.kimi.com/coding/v1",
                # Relays: unknown host, owning provider declared.
                "https://llm.sgh.ng/v1",
                "http://127.0.0.1:8090/v1",
            )
            rejected_urls = (
                "http://api.kimi.com/coding",
                "https://api.kimi.com:8443/coding",
                "https://api.kimi.com/coding/../other",
                "https://api.kimi.com/codingevil",
                "https://[api.kimi.com/coding",
                # Another provider's own host is never a relay for the coding plan's window.
                "https://api.moonshot.ai/v1",
                "https://api.moonshot.cn/v1",
            )

            for base_url in accepted_urls:
                for model in ("k3", "kimi-k3", "kimi-k3-cot"):
                    assert get_model_context_length(
                        model, provider="kimi-coding", base_url=base_url
                    ) == 1_048_576

            for base_url in rejected_urls:
                # Bare slug: scoped, must NOT leak to a surface that is not the coding plan.
                assert get_model_context_length(
                    "k3", provider="kimi-coding", base_url=base_url
                ) != 1_048_576
                # Named slugs: global DEFAULT_CONTEXT_LENGTHS entry applies
                # everywhere the model is actually named kimi-k3.
                for model in ("kimi-k3", "kimi-k3-cot"):
                    assert get_model_context_length(
                        model, provider="kimi-coding", base_url=base_url
                    ) == 1_048_576


class _RecordingProfile:
    """Provider profile whose declared window stands in for a plugin's ``get_model_context_length``."""

    def __init__(self, value):
        self.value = value
        self.seen = []

    def get_model_context_length(self, model):
        self.seen.append(model)
        return self.value


class TestAutoProviderBeforeProfileHook:
    """Part 2: ``auto``/blank resolves to a concrete provider, so registered profiles fire."""

    def test_auto_resolves_concrete_provider_for_profile_hook(self):
        profile = _RecordingProfile(1_234_567)
        asked = []

        def _profile(name):
            asked.append(name)
            return profile if name == "kimi-coding" else None

        with _quiet_tiers(), \
             patch("hermes_cli.auth.resolve_provider", return_value="kimi-coding"), \
             patch("providers.get_provider_profile", side_effect=_profile):
            ctx = get_model_context_length("relay-model", base_url=RELAY, provider="auto")

        assert ctx == 1_234_567
        assert "kimi-coding" in asked

    def test_blank_provider_resolves_too(self):
        profile = _RecordingProfile(999_000)
        with _quiet_tiers(), \
             patch("hermes_cli.auth.resolve_provider", return_value="kimi-coding"), \
             patch("providers.get_provider_profile",
                   side_effect=lambda name: profile if name == "kimi-coding" else None):
            assert get_model_context_length("relay-model", base_url=RELAY, provider="") == 999_000

    def test_known_host_infers_without_asking_the_resolver(self):
        """A direct connection's host names its provider: no config/credential read on that path."""
        asked = []

        def _profile(name):
            asked.append(name)

        with _quiet_tiers(), \
             patch("hermes_cli.auth.resolve_provider") as resolver, \
             patch("providers.get_provider_profile", side_effect=_profile):
            get_model_context_length("some-model", base_url="https://api.moonshot.ai/v1",
                                     provider="auto")

        resolver.assert_not_called()
        assert "kimi-coding" in asked

    def test_explicit_provider_does_not_consult_the_resolver(self):
        with _quiet_tiers(), \
             patch("hermes_cli.auth.resolve_provider") as resolver:
            get_model_context_length("relay-model", base_url=RELAY, provider="deepseek")

        resolver.assert_not_called()

    def test_resolution_failure_falls_through(self):
        """A resolver that cannot name a provider (AuthError, no config) must not break resolution."""
        with _quiet_tiers(), \
             patch("hermes_cli.auth.resolve_provider", side_effect=RuntimeError("no provider")):
            assert get_model_context_length(
                "totally-unknown-model-xyz", base_url=RELAY, provider="auto"
            ) == mm.DEFAULT_FALLBACK_CONTEXT


class TestCatalogProviderRelayAlias:
    """Part 3: ``catalog_provider`` names the upstream vendor a custom route resells."""

    @staticmethod
    def _cfg(config):
        def _cfg_get(*keys, default=None):
            node = config
            for key in keys:
                if not isinstance(node, dict) or key not in node:
                    return default
                node = node[key]
            return node
        return patch("agent.models_dev._cfg_get", side_effect=_cfg_get)

    def test_declared_vendor_scopes_relayed_metadata(self):
        config = {"providers": {"bifrost": {"api": RELAY, "catalog_provider": "kimi-coding"}}}
        with _quiet_tiers(), self._cfg(config):
            assert get_model_context_length(
                "k3", base_url=RELAY, provider="bifrost"
            ) == KIMI_CTX

    def test_legacy_custom_providers_entry_aliases_too(self):
        config = {"custom_providers": [
            {"name": "bifrost", "base_url": RELAY, "catalog_provider": "kimi-coding"}]}
        with _quiet_tiers(), self._cfg(config):
            assert get_model_context_length(
                "k3", base_url=RELAY, provider="custom:bifrost"
            ) == KIMI_CTX

    def test_without_alias_the_relay_stays_unscoped(self):
        config = {"providers": {"bifrost": {"api": RELAY}}}
        with _quiet_tiers(), self._cfg(config):
            assert get_model_context_length(
                "k3", base_url=RELAY, provider="bifrost"
            ) == mm.DEFAULT_FALLBACK_CONTEXT
