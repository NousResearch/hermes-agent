"""A named custom route ("custom:<name>") must inherit the vendor identity its base_url proves.

Hermes itself writes the ``custom:<name>`` spelling when a user configures a named entry under
``providers:``. The URL-inference gate in ``get_model_context_length`` only fired for a blank
provider, ``openrouter`` and bare ``custom``, so a route like ``custom:openrouter`` pointing at
``openrouter.ai`` matched no catalog and fell to the 256K step-9 fallback — which also flips the
compressor into the 75% small-window regime, compounding ~3x early (#133183).
"""

from unittest.mock import patch

from agent.model_metadata import get_model_context_length


def _lookup_side_effect(calls, vendor_window):
    def _lookup(provider, model, *, allow_network=False):
        calls.append(provider)
        return vendor_window if provider == "openrouter" else None

    return _lookup


class TestNamedCustomRouteVendorInference:
    def test_named_custom_route_inherits_vendor_identity_from_its_base_url(self):
        """`custom:openrouter` on openrouter.ai must resolve like bare `custom` does."""
        from agent import model_metadata as mm

        calls = []
        with (
            patch.object(mm, "get_cached_context_length", return_value=None),
            patch.object(mm, "fetch_endpoint_model_metadata", return_value={}),
            patch.object(mm, "_query_ollama_api_show", return_value=None),
            patch.object(mm, "is_local_endpoint", return_value=False),
            patch.object(mm, "fetch_model_metadata", return_value={}),
            patch.object(mm, "save_context_length"),
            patch(
                "agent.models_dev.lookup_models_dev_context",
                side_effect=_lookup_side_effect(calls, 1_000_000),
            ),
        ):
            ctx = get_model_context_length(
                "stealth/space-bunny-alpha",
                base_url="https://openrouter.ai/api/v1",
                provider="custom:openrouter",
            )
        assert ctx == 1_000_000
        assert (
            "openrouter" in calls
        )  # inference fired; on the bug it stayed "custom:openrouter"

    def test_all_route_spellings_agree_on_a_vendor_base_url(self):
        """The same route must not resolve differently depending on how the provider is spelled."""
        from agent import model_metadata as mm

        for provider in ("custom:openrouter", "custom", "openrouter"):
            calls = []
            with (
                patch.object(mm, "get_cached_context_length", return_value=None),
                patch.object(mm, "fetch_endpoint_model_metadata", return_value={}),
                patch.object(mm, "_query_ollama_api_show", return_value=None),
                patch.object(mm, "is_local_endpoint", return_value=False),
                patch.object(mm, "fetch_model_metadata", return_value={}),
                patch.object(mm, "save_context_length"),
                patch(
                    "agent.models_dev.lookup_models_dev_context",
                    side_effect=_lookup_side_effect(calls, 1_000_000),
                ),
            ):
                ctx = get_model_context_length(
                    "stealth/space-bunny-alpha",
                    base_url="https://openrouter.ai/api/v1",
                    provider=provider,
                )
            assert ctx == 1_000_000, provider

    def test_named_custom_route_on_a_generic_host_keeps_today_path(self):
        """A generic relay host must not gain an inferred vendor identity (step-2 probe unchanged)."""
        from agent import model_metadata as mm

        with (
            patch.object(mm, "get_cached_context_length", return_value=None),
            patch.object(
                mm, "_resolve_custom_endpoint_context_length", return_value=256_000
            ) as probe,
        ):
            ctx = get_model_context_length(
                "some-model",
                base_url="https://relay.example.com/v1",
                provider="custom:my-gateway",
            )
        assert ctx == 256_000
        probe.assert_called_once()
