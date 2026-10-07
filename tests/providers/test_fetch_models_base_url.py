"""Tests for ProviderProfile.fetch_models base_url override (issue #47009)."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch, MagicMock

from providers.base import ProviderProfile


class _FakeModelHandler(BaseHTTPRequestHandler):
    """Serves /models with a configurable model list."""

    models = [{"id": "custom-model-1"}, {"id": "custom-model-2"}]

    def do_GET(self):
        if self.path.rstrip("/") == "/models":
            body = json.dumps({"data": self.models}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noise


def _start_server(models=None):
    """Start a local HTTP server returning given models. Returns (server, port)."""
    if models is not None:
        _FakeModelHandler.models = models
    server = HTTPServer(("127.0.0.1", 0), _FakeModelHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


class TestFetchModelsBaseUrlOverride:
    """fetch_models() should use caller-provided base_url when given."""

    def test_base_url_override_used(self):
        """When base_url is passed, it overrides self.base_url."""
        server, port = _start_server([{"id": "proxy-model-a"}])
        try:
            profile = ProviderProfile(
                name="test",
                base_url="http://127.0.0.1:1",  # wrong port — should not be used
            )
            result = profile.fetch_models(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{port}",
            )
            assert result == ["proxy-model-a"]
        finally:
            server.shutdown()

    def test_custom_base_url_beats_models_url(self):
        """A caller base_url differing from the profile default overrides
        models_url — a user-configured proxy must win over the profile's
        hardcoded catalog endpoint (Discord report: CommandCode picker)."""
        server, port = _start_server([{"id": "proxy-model-b"}])
        try:
            profile = ProviderProfile(
                name="test",
                base_url="http://127.0.0.1:1",
                models_url="http://127.0.0.1:1/models",  # unreachable
            )
            result = profile.fetch_models(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{port}",
            )
            assert result == ["proxy-model-b"]
        finally:
            server.shutdown()

    def test_default_base_url_does_not_shadow_models_url(self):
        """Callers pass base_url unconditionally (profile default when the
        user configured nothing). Equality with self.base_url means "not
        customised" and must keep models_url as the endpoint."""
        server, port = _start_server([{"id": "catalog-model"}])
        try:
            profile = ProviderProfile(
                name="test",
                base_url="http://127.0.0.1:1",  # inference URL, unreachable
                models_url=f"http://127.0.0.1:{port}/models",
            )
            # Caller echoes the profile default back — models_url must win.
            result = profile.fetch_models(
                api_key="test-key",
                base_url="http://127.0.0.1:1/",  # same default, trailing slash
            )
            assert result == ["catalog-model"]
        finally:
            server.shutdown()





class TestCustomProviderBaseUrlPassthrough:
    """Custom provider (ollama/local) should pass base_url through to super."""

    def test_custom_passes_base_url(self):
        """CustomProfile.fetch_models passes base_url to super()."""
        server, port = _start_server([{"id": "ollama-model"}])
        try:
            from plugins.model_providers.custom import CustomProfile
            profile = CustomProfile(
                name="custom",
                base_url="http://127.0.0.1:1",  # wrong port
            )
            result = profile.fetch_models(
                api_key="",
                base_url=f"http://127.0.0.1:{port}",
            )
            assert result == ["ollama-model"]
        finally:
            server.shutdown()


class _RedirectingHandler(BaseHTTPRequestHandler):
    """Redirects /models to a configurable target and records received headers."""

    redirect_to = ""  # full URL to redirect /models to (set per test)
    received_headers: dict = {}

    def do_GET(self):
        if self.path.rstrip("/") == "/models":
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
        else:
            _RedirectingHandler.received_headers = dict(self.headers)
            body = json.dumps({"data": [{"id": "redirected-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class _NativeGeminiHandler(BaseHTTPRequestHandler):
    """Mimics paginated native ListModels with header authentication."""

    paths: list[str] = []
    repeat_page_token = False

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        type(self).paths.append(self.path)
        if (
            self.headers.get("Authorization")
            or self.headers.get("x-goog-api-key") != "test-key"
            or parse_qs(parsed.query).get("key")
        ):
            self.send_response(401)
            self.end_headers()
            return
        if parsed.path.rstrip("/") == "/models":
            page_token = parse_qs(parsed.query).get("pageToken", [""])[0]
            if not page_token:
                payload = {
                    "models": [
                        {
                            "name": "models/gemini-fixture-chat",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-fixture-preview-tts",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-embedding-fixture",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ],
                    "nextPageToken": "page-2",
                }
            else:
                payload = {
                    "models": [
                        {
                            "name": "models/gemini-fixture-chat-next",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/lyria-fixture",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                }
                if type(self).repeat_page_token:
                    payload["nextPageToken"] = "page-2"
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(401)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class TestFetchModelsRedirectCredentialStripping:
    """Credential headers must not follow a redirect outside the original origin."""

    def _run(self, redirect_to):
        """redirect_to is a callable (first_port, second_port) -> Location URL."""
        _RedirectingHandler.received_headers = {}
        server = HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
        second_server = HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
        port = server.server_address[1]
        second_port = second_server.server_address[1]
        _RedirectingHandler.redirect_to = redirect_to(port, second_port)
        Thread(target=server.serve_forever, daemon=True).start()
        Thread(target=second_server.serve_forever, daemon=True).start()
        try:
            profile = ProviderProfile(
                name="test",
                base_url=f"http://127.0.0.1:{port}",
                default_headers={"x-api-key": "default-header-secret"},
            )
            result = profile.fetch_models(api_key="bearer-secret")
        finally:
            server.shutdown()
            second_server.shutdown()
        headers = {k.lower(): v for k, v in _RedirectingHandler.received_headers.items()}
        return result, headers

    def test_cross_host_redirect_strips_credentials(self):
        result, headers = self._run(
            lambda port, _: f"http://localhost:{port}/redirected"
        )
        assert result == ["redirected-model"]  # fetch itself still works
        assert "authorization" not in headers
        assert "x-api-key" not in headers


    def test_same_origin_redirect_keeps_credentials(self):
        result, headers = self._run(
            lambda port, _: f"http://127.0.0.1:{port}/redirected"
        )
        assert result == ["redirected-model"]
        assert headers.get("authorization") == "Bearer bearer-secret"
        assert headers.get("x-api-key") == "default-header-secret"


class TestGeminiNativeFetchModels:
    """GeminiProfile.fetch_models uses the native ListModels contract (#62259)."""

    def test_native_catalog_uses_key_header_paginates_and_filters(self):
        _NativeGeminiHandler.paths = []
        _NativeGeminiHandler.repeat_page_token = False
        server = HTTPServer(("127.0.0.1", 0), _NativeGeminiHandler)
        port = server.server_address[1]
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            from plugins.model_providers.gemini import GeminiProfile
            profile = GeminiProfile(name="gemini", base_url=f"http://127.0.0.1:{port}")
            with patch(
                "agent.gemini_native_adapter.is_native_gemini_base_url",
                return_value=True,
            ):
                result = profile.fetch_models(api_key="test-key")
                assert result == ["gemini-fixture-chat", "gemini-fixture-chat-next"]
                assert len(_NativeGeminiHandler.paths) == 2
                assert "pageToken=page-2" in _NativeGeminiHandler.paths[1]

                _NativeGeminiHandler.paths = []
                _NativeGeminiHandler.repeat_page_token = True
                assert profile.fetch_models(api_key="test-key") is None
                assert len(_NativeGeminiHandler.paths) == 2
        finally:
            server.shutdown()

    def test_absent_key_skips_and_non_native_endpoints_delegate(self):
        from plugins.model_providers.gemini import GeminiProfile
        profile = GeminiProfile(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert profile.fetch_models(api_key="") is None
        with patch(
            "providers.base.ProviderProfile.fetch_models",
            return_value=["gemini-compat-model"],
        ) as base_fetch:
            for base_url in (
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "https://relay.example/v1",
            ):
                profile = GeminiProfile(name="gemini", base_url=base_url)
                assert profile.fetch_models(api_key="test-key") == ["gemini-compat-model"]
        assert base_fetch.call_count == 2


class TestModelPickerBaseUrlIntegration:
    """The /model picker path should pass model.base_url to fetch_models."""

    def test_picker_passes_base_url(self):
        """Verify models.py caller passes base_url to fetch_models."""
        mock_profile = MagicMock()
        mock_profile.auth_type = "api_key"
        mock_profile.base_url = "https://default.api.com"
        mock_profile.fetch_models.return_value = ["model-a"]

        with (
            patch("providers.get_provider_profile", return_value=mock_profile),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials",
                  return_value={"api_key": "sk-test", "base_url": "https://custom.proxy.com"}),
        ):
            from hermes_cli.models import provider_model_ids
            result = provider_model_ids("test-provider")
            # Verify fetch_models was called with base_url
            mock_profile.fetch_models.assert_called_once()
            call_kwargs = mock_profile.fetch_models.call_args
            assert call_kwargs.kwargs.get("base_url") == "https://custom.proxy.com"


def test_profiles_without_model_listing_never_hit_the_network():
    """SDK-backed profiles (bedrock, vertex) still carry a base_url the generic fetch_models
    would happily GET ``/models`` against; the flag must short-circuit first."""
    from providers import list_providers

    flagged = [p for p in list_providers() if not p.supports_model_listing]
    assert {p.name for p in flagged} >= {"bedrock", "vertex"}
    with patch("hermes_cli.urllib_security.open_credentialed_url") as opener:
        for profile in flagged:
            assert profile.fetch_models(api_key="k", base_url=profile.base_url) is None, profile.name
    opener.assert_not_called()
