"""Unit tests for the MiniMax provider profile.

Three MiniMax provider profiles (`minimax` direct API, `minimax-cn` China direct
API, `minimax-oauth` browser OAuth) all advertise a `default_aux_model` on
their `ProviderProfile`. The previous M2.7 / M2.7-highspeed values were
stale relative to the current frontier model (M3, released 2026-06-01) and
inconsistent with the `_PROVIDER_MODELS["minimax"]` catalog top entry in
`hermes_cli/models.py`.

This file pins the new defaults so the choice is reviewable and any future
revert shows up in a failing test rather than silent behavior drift.

Refs:
  - Issue #36196: M3 support request
  - PR #36205 (closed unmerged): Csrayz's M3 + 1M context work
  - PR #36212 (open): adds M3 to `_PROVIDER_MODELS["minimax"]` catalog
  - PR #6082: M2.7-highspeed → M2.7 for aux model (half-price fix)
  - Commit 773a0faca: same profile-layer fix pattern for `deepseek`
  - Issue #142648: the live catalog lives at ``<base>/v1/models``; ``<base>/models`` 404s
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest


@pytest.fixture(params=["minimax", "minimax-cn", "minimax-oauth"])
def minimax_profile(request):
    """Resolve each registered MiniMax profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    import model_tools
    import providers

    profile = providers.get_provider_profile(request.param)
    assert profile is not None, f"{request.param} provider profile must be registered"
    return profile, request.param


class TestMinimaxAuxModelM3:
    """MiniMax profile aux model is the new frontier M3, not the stale M2.7.

    The catalog top entry is ``MiniMax-M3`` in
    ``hermes_cli.models._PROVIDER_MODELS['minimax']`` and the
    user-facing ``model.default`` for a Token-Plan install is M3,
    so pinning the aux default to the same model keeps the runtime
    consistent (same auth, same billing pool, same rate limits, no
    surprise 2x-cost highspeed variant). M3 was released 2026-06-01
    — picking it as the aux default matches the forward-looking
    catalog order rather than the pre-M3 era.
    """


    def test_consumer_api_returns_non_empty_for_each_provider(self, minimax_profile):
        from agent.auxiliary_client import _get_aux_model_for_provider

        profile, provider_id = minimax_profile
        resolved = _get_aux_model_for_provider(provider_id)
        assert resolved != "", (
            f"_get_aux_model_for_provider({provider_id!r}) returned empty — "
            "the 'No auxiliary LLM provider configured' warning will fire on "
            f"every {provider_id} session even though the profile advertises "
            f"default_aux_model={profile.default_aux_model!r}"
        )
        assert resolved == profile.default_aux_model, (
            f"_get_aux_model_for_provider({provider_id!r}) returned "
            f"{resolved!r} but profile advertises {profile.default_aux_model!r} "
            "— the consumer API and the profile have drifted out of sync"
        )




class TestMinimaxM3OpenAIReasoningWireShape:
    """MiniMax-M3 on api.minimax.io/v1 gets MiniMax's OpenAI-compatible knobs."""

    def test_m3_openai_route_requests_reasoning_split_by_default(self):
        import model_tools
        import providers

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config=None,
            model="MiniMax-M3",
            base_url="https://api.minimax.io/v1",
        )
        assert extra_body == {"reasoning_split": True}
        assert top_level == {}


    @pytest.mark.parametrize(
        "model,base_url",
        [
            ("MiniMax-M2.7", "https://api.minimax.io/v1"),
            ("MiniMax-M3", "https://api.minimax.io/anthropic"),
            ("MiniMax-M3", "https://api.minimaxi.com/v1"),
        ],
    )
    def test_non_m3_or_non_global_openai_routes_emit_no_openai_reasoning_knobs(
        self, model, base_url
    ):
        import model_tools
        import providers

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
            base_url=base_url,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_transport_threads_base_url_to_profile(self):
        import model_tools
        import providers
        from agent.transports.chat_completions import ChatCompletionsTransport

        profile = providers.get_provider_profile("minimax")
        assert profile is not None
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="MiniMax-M3",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=profile,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.minimax.io/v1",
        )
        assert kwargs["extra_body"] == {
            "reasoning_split": True,
            "thinking": {"type": "adaptive"},
        }


class TestMinimaxOauthAliases:
    """Every ``--provider`` alias the user guide (website/docs/guides/minimax-oauth.md)
    promises for ``minimax-oauth`` must resolve through the plugin registry, not only the
    CLI alias tables — ``get_provider_profile(agent.provider)`` is what selects the
    anthropic_messages wire, extra_body and headers (#107928)."""

    def test_each_documented_oauth_alias_resolves_to_minimax_oauth(self):
        import model_tools
        import providers

        for alias in ("minimax_oauth", "minimax-portal", "minimax-global"):
            resolved = providers.get_provider_profile(alias)
            assert resolved is not None and resolved.name == "minimax-oauth", alias


class _CatalogHandler(BaseHTTPRequestHandler):
    """Serves a model list on the configured path, 404s everything else."""

    served_path = "/v1/models"
    model_ids = ["MiniMax-M3.1-Flash", "MiniMax-M3"]

    def do_GET(self):
        if self.path.rstrip("/") == self.served_path.rstrip("/"):
            body = json.dumps({"data": [{"id": mid} for mid in self.model_ids]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noise


def _serve(served_path, model_ids):
    """Start a local catalog server; returns (server, port)."""
    handler = type("_Handler", (_CatalogHandler,), {"served_path": served_path, "model_ids": model_ids})
    server = HTTPServer(("127.0.0.1", 0), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


class TestMinimaxCatalogDiscovery:
    """MiniMax's live catalog must be readable from the profile.

    The vendor publishes the catalog under ``<base>/v1/models``; the generic
    ``{base_url}/models`` probe answers 404, so every catalog surface silently
    degraded to the curated floor and new models never surfaced (#142648).
    """

    def test_catalog_is_read_from_the_v1_route(self):
        import providers

        server, port = _serve("/anthropic/v1/models", ["MiniMax-M3.1-Flash", "MiniMax-M3"])
        try:
            profile = providers.get_provider_profile("minimax")
            assert profile is not None
            models = profile.fetch_models(
                api_key="test-key", base_url=f"http://127.0.0.1:{port}/anthropic")
        finally:
            server.shutdown()
        assert models == ["MiniMax-M3.1-Flash", "MiniMax-M3"]

    def test_plain_models_route_still_serves_custom_endpoints(self):
        """A custom endpoint that publishes ``<base>/models`` keeps working."""
        import providers

        server, port = _serve("/proxy/models", ["proxy-model-a"])
        try:
            profile = providers.get_provider_profile("minimax")
            assert profile is not None
            models = profile.fetch_models(
                api_key="test-key", base_url=f"http://127.0.0.1:{port}/proxy")
        finally:
            server.shutdown()
        assert models == ["proxy-model-a"]

    def test_china_profile_reads_the_china_catalog(self):
        import providers

        server, port = _serve("/anthropic/v1/models", ["MiniMax-M3.1-Flash"])
        try:
            profile = providers.get_provider_profile("minimax-cn")
            assert profile is not None
            models = profile.fetch_models(
                api_key="test-key", base_url=f"http://127.0.0.1:{port}/anthropic")
        finally:
            server.shutdown()
        assert models == ["MiniMax-M3.1-Flash"]
