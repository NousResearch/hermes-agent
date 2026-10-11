"""CommandCode must not offer a model on a wire the API would reject.

CommandCode publishes ONE catalog shared by both profiles, and each row's
``supported_endpoints`` names the wires that accept it. Every ``claude-*`` row is
``/messages``-only, so the chat_completions profile listing a Claude model gave a
400 ``unsupported_model`` on every selection, while the Anthropic side happened to
stay correct by filtering ids with ``startswith("claude-")`` — a coincidence that
breaks the moment either family moves across wires.

Regression: https://github.com/NousResearch/hermes-agent/issues/123452
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

CATALOG = [
    {"id": "gpt-6-astra", "supported_endpoints": ["/chat/completions", "/responses"]},
    {"id": "deepseek/deepseek-v4-flash", "supported_endpoints": ["/chat/completions"]},
    {"id": "claude-sonnet-5-5", "supported_endpoints": ["/messages"]},
    {"id": "claude-opus-4-7", "supported_endpoints": ["/messages"]},
    # Annotated rows the wire does NOT accept must go; unannotated rows stay.
    {"id": "responses-only-model", "supported_endpoints": ["/responses"]},
    {"id": "unannotated-model"},
    {"id": "empty-annotation-model", "supported_endpoints": []},
]


def _serve(items):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": items}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


@pytest.fixture
def profiles():
    import model_tools  # noqa: F401 — triggers provider discovery
    import providers

    assert providers.get_provider_profile("commandcode") is not None
    assert providers.get_provider_profile("commandcode-anthropic") is not None


class TestCommandCodeCatalogWireFiltering:
    """Each profile lists only the rows its own wire accepts."""

    def test_chat_profile_drops_messages_only_rows(self, profiles):
        import providers

        server, port = _serve(CATALOG)
        try:
            models = providers.get_provider_profile("commandcode").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
        finally:
            server.shutdown()
        assert "claude-sonnet-5-5" not in models
        assert "claude-opus-4-7" not in models
        assert "deepseek/deepseek-v4-flash" in models
        assert "unannotated-model" in models

    def test_anthropic_profile_lists_only_messages_rows(self, profiles):
        import providers

        server, port = _serve(CATALOG)
        try:
            models = providers.get_provider_profile("commandcode-anthropic").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
        finally:
            server.shutdown()
        assert "claude-sonnet-5-5" in models
        assert "claude-opus-4-7" in models
        assert "deepseek/deepseek-v4-flash" not in models
        assert "gpt-6-astra" not in models

    def test_row_accepted_by_both_wires_appears_in_both(self, profiles):
        """A row naming two wires is legitimately selectable on either."""
        import providers

        items = [{"id": "dual-wire-model", "supported_endpoints": ["/chat/completions", "/messages"]}]
        server, port = _serve(items)
        try:
            chat = providers.get_provider_profile("commandcode").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
            anth = providers.get_provider_profile("commandcode-anthropic").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
        finally:
            server.shutdown()
        assert chat == ["dual-wire-model"]
        assert anth == ["dual-wire-model"]

    def test_custom_base_url_is_filtered_by_profile_wire(self, profiles):
        """A caller's custom endpoint is NOT exempt from the wire filter.

        A proxy that mirrors CommandCode's catalog shape keeps its
        annotations, and honoring them is what keeps the chat profile from
        offering a Messages-only row the proxy would 400.
        """
        import providers

        server, port = _serve([
            {"id": "claude-sonnet-5-5", "supported_endpoints": ["/messages"]},
            {"id": "deepseek/deepseek-v4-flash", "supported_endpoints": ["/chat/completions"]},
        ])
        try:
            chat = providers.get_provider_profile("commandcode").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
            anth = providers.get_provider_profile("commandcode-anthropic").fetch_models(
                api_key="k", base_url=f"http://127.0.0.1:{port}")
        finally:
            server.shutdown()
        assert chat == ["deepseek/deepseek-v4-flash"]
        assert anth == ["claude-sonnet-5-5"]


class TestSharedCatalogFilter:
    """The filter itself, independent of CommandCode."""

    def test_absent_or_empty_endpoints_are_admitted(self):
        from hermes_cli.chat_catalog import chat_catalog_ids

        items = [
            {"id": "no-annotation"},
            {"id": "empty-list", "supported_endpoints": []},
            {"id": "not-a-list", "supported_endpoints": "/messages"},
        ]
        assert chat_catalog_ids(items, endpoint="/messages") == [
            "no-annotation", "empty-list", "not-a-list",
        ]

    def test_generation_rows_still_dropped(self):
        from hermes_cli.chat_catalog import chat_catalog_ids

        items = [
            {"id": "claude-sonnet-5-5", "supported_endpoints": ["/messages"]},
            {"id": "wan2.7-image-pro", "supported_endpoints": ["/messages"]},
            {"id": "gen-x", "supported_endpoints": ["/messages"], "capabilities": {"type": "image"}},
        ]
        assert chat_catalog_ids(items, endpoint="/messages") == ["claude-sonnet-5-5"]

    def test_no_endpoint_filter_keeps_every_chat_row(self):
        from hermes_cli.chat_catalog import chat_catalog_ids

        items = [
            {"id": "a-only-responses", "supported_endpoints": ["/responses"]},
            {"id": "plain"},
        ]
        assert chat_catalog_ids(items) == ["a-only-responses", "plain"]


class TestCatalogEndpointDeclaration:
    """Both profiles must declare a wire, or the filter is a no-op."""

    def test_profiles_declare_distinct_wires(self, profiles):
        import providers

        chat = providers.get_provider_profile("commandcode").catalog_endpoint
        anth = providers.get_provider_profile("commandcode-anthropic").catalog_endpoint
        assert chat == "/chat/completions"
        assert anth == "/messages"
