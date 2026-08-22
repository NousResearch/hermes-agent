"""AI Gateway model list and pricing translation.

Vercel AI Gateway exposes ``/v1/models`` with a richer shape than OpenAI's
spec (type, tags, pricing). The pricing object uses ``input`` / ``output``
where hermes's shared picker expects ``prompt`` / ``completion``; these tests
pin the translation, the curated-list filtering, and the credential-redirect
guard the AI Gateway catalog calls share with the rest of ``hermes_cli.models``.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import patch, MagicMock

from hermes_cli import models as models_module
from hermes_cli.models import (
    VERCEL_AI_GATEWAY_MODELS,
    _ai_gateway_model_is_free,
    _fetch_ai_gateway_models,
    fetch_ai_gateway_models,
    fetch_ai_gateway_pricing,
)


def _mock_urlopen(payload):
    """Build a catalog-opener context manager mock returning the given payload."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    ctx = MagicMock()
    ctx.__enter__.return_value = resp
    ctx.__exit__.return_value = False
    return ctx


def _reset_caches():
    models_module._ai_gateway_catalog_cache = None
    models_module._pricing_cache.clear()


def test_ai_gateway_pricing_translates_input_output_to_prompt_completion():
    _reset_caches()
    payload = {
        "data": [
            {
                "id": "moonshotai/kimi-k2.5",
                "type": "language",
                "pricing": {
                    "input": "0.0000006",
                    "output": "0.0000025",
                    "input_cache_read": "0.00000015",
                    "input_cache_write": "0.0000006",
                },
            }
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_pricing(force_refresh=True)

    entry = result["moonshotai/kimi-k2.5"]
    assert entry["prompt"] == "0.0000006"
    assert entry["completion"] == "0.0000025"
    assert entry["input_cache_read"] == "0.00000015"
    assert entry["input_cache_write"] == "0.0000006"


def test_ai_gateway_pricing_returns_empty_on_fetch_failure():
    _reset_caches()
    with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=OSError("network down")):
        result = fetch_ai_gateway_pricing(force_refresh=True)
    assert result == {}


def test_ai_gateway_pricing_skips_entries_without_pricing_dict():
    _reset_caches()
    payload = {
        "data": [
            {"id": "x/y", "pricing": None},
            {"id": "a/b", "pricing": {"input": "0", "output": "0"}},
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_pricing(force_refresh=True)
    assert "x/y" not in result
    assert result["a/b"] == {"prompt": "0", "completion": "0"}


def test_ai_gateway_free_detector():
    assert _ai_gateway_model_is_free({"input": "0", "output": "0"}) is True
    assert _ai_gateway_model_is_free({"input": "0", "output": "0.01"}) is False
    assert _ai_gateway_model_is_free({"input": "0.01", "output": "0"}) is False
    assert _ai_gateway_model_is_free(None) is False
    assert _ai_gateway_model_is_free({"input": "not a number"}) is False


def test_fetch_ai_gateway_models_filters_against_live_catalog():
    _reset_caches()
    preferred = [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS]
    live_ids = preferred[:3]  # only first three exist live
    payload = {
        "data": [
            {"id": mid, "pricing": {"input": "0.001", "output": "0.002"}}
            for mid in live_ids
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    assert [mid for mid, _ in result] == live_ids
    assert result[0][1] == "recommended"


def test_fetch_ai_gateway_models_tags_free_models():
    _reset_caches()
    first_id = VERCEL_AI_GATEWAY_MODELS[0][0]
    second_id = VERCEL_AI_GATEWAY_MODELS[1][0]
    payload = {
        "data": [
            {"id": first_id, "pricing": {"input": "0.001", "output": "0.002"}},
            {"id": second_id, "pricing": {"input": "0", "output": "0"}},
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    by_id = dict(result)
    assert by_id[first_id] == "recommended"
    assert by_id[second_id] == "free"


def test_free_moonshot_model_auto_promoted_to_top_even_if_not_curated():
    _reset_caches()
    first_curated = VERCEL_AI_GATEWAY_MODELS[0][0]
    unlisted_free_moonshot = "moonshotai/kimi-coder-free-preview"
    payload = {
        "data": [
            {"id": first_curated, "pricing": {"input": "0.001", "output": "0.002"}},
            {"id": unlisted_free_moonshot, "pricing": {"input": "0", "output": "0"}},
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    assert result[0] == (unlisted_free_moonshot, "recommended")
    assert any(mid == first_curated for mid, _ in result)


def test_paid_moonshot_does_not_get_auto_promoted():
    _reset_caches()
    first_curated = VERCEL_AI_GATEWAY_MODELS[0][0]
    payload = {
        "data": [
            {"id": first_curated, "pricing": {"input": "0.001", "output": "0.002"}},
            {"id": "moonshotai/some-paid-variant", "pricing": {"input": "0.001", "output": "0.002"}},
        ]
    }
    with patch("hermes_cli.models._urlopen_model_catalog_request", return_value=_mock_urlopen(payload)):
        result = fetch_ai_gateway_models(force_refresh=True)

    assert result[0][0] == first_curated


def test_fetch_ai_gateway_models_falls_back_on_error():
    _reset_caches()
    with patch("hermes_cli.models._urlopen_model_catalog_request", side_effect=OSError("network")):
        result = fetch_ai_gateway_models(force_refresh=True)
    assert result == list(VERCEL_AI_GATEWAY_MODELS)


def _serve(handler_cls) -> ThreadingHTTPServer:
    """Start a throwaway loopback HTTP server on an ephemeral port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_ai_gateway_probe_drops_bearer_on_cross_origin_redirect(monkeypatch):
    """``_fetch_ai_gateway_models`` must not forward the API key across a redirect.

    The probe builds its own ``Request`` carrying ``Authorization: Bearer
    <AI_GATEWAY_API_KEY>`` and resolves its host from the user-settable
    ``AI_GATEWAY_BASE_URL``, so stdlib's default redirect handling would hand
    the key to whatever host a ``302`` names. Drives the real
    ``SafeCredentialRedirectHandler`` against two loopback servers -- one
    redirects, the other records what it received -- rather than mocking the
    security module, mirroring the wire-level tests in
    ``tests/hermes_cli/test_urllib_security.py``.
    """
    received: list[dict[str, str]] = []

    class _SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(
                {name.lower(): value for name, value in self.headers.items()}
            )
            body = json.dumps(
                {
                    "data": [
                        {
                            "id": "gateway/redirected-model",
                            "type": "language",
                            "tags": ["tool-use"],
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    sink = _serve(_SinkHandler)
    sink_port = sink.server_port

    class _RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink_port}/models")
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    source = _serve(_RedirectHandler)

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "ai-gateway-secret")
    monkeypatch.setenv(
        "AI_GATEWAY_BASE_URL", f"http://127.0.0.1:{source.server_port}"
    )
    try:
        result = _fetch_ai_gateway_models(timeout=5)
    finally:
        source.shutdown()
        sink.shutdown()

    # The redirect really was followed and its payload parsed, so the
    # assertions below are about a request that actually reached the sink.
    assert result == ["gateway/redirected-model"]
    assert len(received) == 1
    headers = received[0]
    assert "authorization" not in headers
    # Only the credential is dropped: the cross-origin-safe headers survive.
    assert headers.get("user-agent", "").startswith("hermes-cli/")
