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

    def test_fallback_to_self_base_url(self):
        """When base_url is None, falls back to self.base_url."""
        server, port = _start_server([{"id": "default-model"}])
        try:
            profile = ProviderProfile(
                name="test",
                base_url=f"http://127.0.0.1:{port}",
            )
            result = profile.fetch_models(api_key="test-key")
            assert result == ["default-model"]
        finally:
            server.shutdown()

    def test_no_base_url_returns_none(self):
        """When both base_url and self.base_url are empty, returns None."""
        profile = ProviderProfile(name="test", base_url="")
        result = profile.fetch_models(api_key="test-key", base_url="")
        assert result is None

    def test_base_url_override_with_models_url_set(self):
        """When self.models_url is set, base_url override is ignored (models_url wins)."""
        server, port = _start_server([{"id": "from-models-url"}])
        try:
            profile = ProviderProfile(
                name="test",
                base_url="http://127.0.0.1:1",
                models_url=f"http://127.0.0.1:{port}/models",
            )
            # base_url override should NOT be used because models_url takes priority
            result = profile.fetch_models(
                api_key="test-key",
                base_url="http://127.0.0.1:1",
            )
            assert result == ["from-models-url"]
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
    """Mimics the native /v1beta endpoint: rejects Bearer, honours ?key=."""

    def do_GET(self):
        # Native Gemini API rejects Bearer auth (this is the #62259 bug).
        if self.headers.get("Authorization"):
            self.send_response(401)
            self.end_headers()
            return
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") == "/models" and parse_qs(parsed.query).get("key"):
            body = json.dumps({
                "models": [
                    {"name": "models/gemini-2.5-pro"},
                    {"name": "models/gemini-3.5-flash"},
                ]
            }).encode()
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

    def test_same_host_different_port_redirect_strips_credentials(self):
        """A different port is a different origin — it can be a different service."""
        result, headers = self._run(
            lambda _, second_port: f"http://127.0.0.1:{second_port}/redirected"
        )
        assert result == ["redirected-model"]
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
    """GeminiProfile.fetch_models uses query-param auth against the native API (#62259)."""

    def test_native_query_param_auth_strips_prefix(self):
        server = HTTPServer(("127.0.0.1", 0), _NativeGeminiHandler)
        port = server.server_address[1]
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            from plugins.model_providers.gemini import GeminiProfile
            profile = GeminiProfile(name="gemini", base_url=f"http://127.0.0.1:{port}")
            result = profile.fetch_models(api_key="test-key")
            # models/ prefix stripped; Bearer path would have 401'd → None
            assert result == ["gemini-2.5-pro", "gemini-3.5-flash"]
        finally:
            server.shutdown()

    def test_no_api_key_returns_none(self):
        from plugins.model_providers.gemini import GeminiProfile
        profile = GeminiProfile(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert profile.fetch_models(api_key="") is None

    def test_openai_compat_base_url_delegates_to_base(self):
        """The OpenAI-compat ``/openai`` base URL speaks Bearer + data[].id, so
        the native override must delegate to ProviderProfile.fetch_models rather
        than force native ``?key=`` auth (which the compat endpoint rejects)."""
        from plugins.model_providers.gemini import GeminiProfile
        profile = GeminiProfile(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
        with patch(
            "providers.base.ProviderProfile.fetch_models",
            return_value=["gemini-compat-model"],
        ) as base_fetch:
            result = profile.fetch_models(api_key="test-key")
        assert result == ["gemini-compat-model"]
        base_fetch.assert_called_once()

    def test_native_fetch_exception_does_not_log_key_bearing_url(self, caplog):
        """urllib errors embed the request URL (which carries the ?key= api
        key); the failure path must not log the exception value."""
        import logging
        from plugins.model_providers.gemini import GeminiProfile

        profile = GeminiProfile(
            name="gemini",
            # Unroutable port → connection error carrying the URL in the exc.
            base_url="http://127.0.0.1:1/v1beta",
        )
        with caplog.at_level(logging.DEBUG):
            result = profile.fetch_models(api_key="super-secret-key")
        assert result is None
        assert "super-secret-key" not in caplog.text


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
