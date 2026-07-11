"""Tests for ProviderProfile.fetch_models base_url override (issue #47009)."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch, MagicMock

from providers.base import ProviderProfile, _FETCH_MODELS_MAX_BYTES


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

    def test_oversized_response_body_falls_back_without_unbounded_read(self):
        """Oversized provider catalogs should fail closed before reading the full body."""

        class OversizedResponse:
            headers = {}

            def __init__(self):
                self.read_sizes = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                self.read_sizes.append(size)
                if size is None or size < 0:
                    raise AssertionError("fetch_models() must not use an unbounded read")
                return b" " * (_FETCH_MODELS_MAX_BYTES + 1)

        response = OversizedResponse()
        profile = ProviderProfile(name="test", base_url="http://models.example")

        with patch("urllib.request.urlopen", return_value=response):
            result = profile.fetch_models(api_key="test-key")

        assert result is None
        assert response.read_sizes == [_FETCH_MODELS_MAX_BYTES + 1]

    def test_oversized_content_length_falls_back_without_reading_body(self):
        """A declared oversized catalog should be rejected before body allocation."""

        class OversizedResponse:
            headers = {"Content-Length": str(_FETCH_MODELS_MAX_BYTES + 1)}

            def __init__(self):
                self.read_called = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size=-1):
                self.read_called = True
                return b"{}"

        response = OversizedResponse()
        profile = ProviderProfile(name="test", base_url="http://models.example")

        with patch("urllib.request.urlopen", return_value=response):
            result = profile.fetch_models(api_key="test-key")

        assert result is None
        assert response.read_called is False


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
