"""Regression tests for the Actual Computer provider wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

from agent.auxiliary_client import _normalize_aux_provider
from hermes_cli import runtime_provider as rp
from hermes_cli.auth import (
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
    DEFAULT_ACTUAL_BASE_URL,
    DEFAULT_ACTUAL_LOCAL_BASE_URL,
    get_api_key_provider_status,
    normalize_actual_base_url,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import normalize_provider as normalize_model_provider
from hermes_cli.models import provider_model_ids
from hermes_cli.providers import determine_api_mode
from hermes_cli.providers import get_label
from hermes_cli.providers import normalize_provider as normalize_overlay_provider
from providers import get_provider_profile


def _clear_actual_env(monkeypatch):
    monkeypatch.delenv("ACTUAL_API_KEY", raising=False)
    monkeypatch.delenv("ACTUAL_BASE_URL", raising=False)


def test_actual_aliases_and_profile_metadata():
    profile = get_provider_profile("actual-computer")

    assert profile is not None
    assert profile.name == "actual"
    assert profile.display_name == "Actual Computer"
    assert profile.base_url == DEFAULT_ACTUAL_BASE_URL
    assert profile.api_mode == "codex_responses"
    assert profile.auth_type == "api_key"
    assert profile.env_vars == ("ACTUAL_API_KEY", "ACTUAL_BASE_URL")
    assert normalize_overlay_provider("aci") == "actual"
    assert normalize_model_provider("actualcomputer") == "actual"
    assert resolve_provider("actual-computer") == "actual"
    assert _normalize_aux_provider("aci") == "actual"
    assert get_label("actual") == "Actual Computer"
    assert determine_api_mode("actual", "https://api.actual.inc") == "codex_responses"


def test_actual_base_url_normalization():
    assert (
        normalize_actual_base_url("https://api.actual.inc") == DEFAULT_ACTUAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("https://api.actual.inc/v1")
        == DEFAULT_ACTUAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://127.0.0.1:8080")
        == DEFAULT_ACTUAL_LOCAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://127.0.0.1:8080/v1")
        == DEFAULT_ACTUAL_LOCAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://localhost:8080/")
        == "http://localhost:8080/v1"
    )


def test_actual_credentials_default_to_hosted_api(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")

    creds = resolve_api_key_provider_credentials("actual")

    assert creds["provider"] == "actual"
    assert creds["api_key"] == "actual-test-key"
    assert creds["base_url"] == DEFAULT_ACTUAL_BASE_URL


def test_actual_local_loopback_allows_no_auth(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")

    creds = resolve_api_key_provider_credentials("actual")
    status = get_api_key_provider_status("actual")

    assert creds["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert creds["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL
    assert creds["source"] == "local-offline"
    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["key_source"] == "local-offline"
    assert status["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_uses_hosted_default(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/test-model"},
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["api_key"] == "actual-test-key"
    assert resolved["base_url"] == DEFAULT_ACTUAL_BASE_URL


def test_actual_runtime_uses_local_env_without_key(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/local-model"},
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_uses_local_config_without_key(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "actual",
            "base_url": "http://127.0.0.1:8080",
            "default": "actual/local-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_normalizes_explicit_hosted_base_url(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/test-model"},
    )

    resolved = rp.resolve_runtime_provider(
        requested="actual",
        explicit_api_key="actual-test-key",
        explicit_base_url="https://api.actual.inc",
    )

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["api_key"] == "actual-test-key"
    assert resolved["base_url"] == DEFAULT_ACTUAL_BASE_URL
    assert resolved["source"] == "explicit"


def test_actual_profile_fetch_models_normalizes_env_base_url(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    profile = get_provider_profile("actual")
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "actual/local-model"}]}).encode()

    def _open(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _open)

    assert profile.fetch_models(api_key=None, timeout=1.5) == ["actual/local-model"]
    assert seen["url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL + "/models"
    assert seen["auth"] is None
    assert seen["timeout"] == 1.5


def test_actual_profile_fetch_models_sends_credential_only_to_original_origin(
    monkeypatch,
):
    """fetch_models must route through the shared redirect-credential guard.

    ActualProfile overrides ProviderProfile.fetch_models with its own
    base_url resolution, and previously called raw urllib.request.urlopen
    directly instead of the base class's open_credentialed_url — losing the
    protection that strips the Authorization header when a redirect leaves
    the original host. Exercises the real SafeCredentialRedirectHandler
    (no mocking of open_credentialed_url itself) against a local HTTP
    server that 302s to a different origin, mirroring
    test_urllib_security.py's end-to-end redirect tests.
    """
    import http.server
    import threading

    _clear_actual_env(monkeypatch)
    profile = get_provider_profile("actual")

    source_auth_headers: list[str | None] = []
    target_auth_headers: list[str | None] = []

    class _RedirectTargetHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            target_auth_headers.append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": "should-not-be-trusted"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    target_server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectTargetHandler)
    target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
    target_thread.start()
    target_port = target_server.server_address[1]

    class _RedirectingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            source_auth_headers.append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/models")
            self.end_headers()

        def log_message(self, *_args):
            pass

    redirect_server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
    redirect_thread = threading.Thread(
        target=redirect_server.serve_forever, daemon=True
    )
    redirect_thread.start()
    redirect_port = redirect_server.server_address[1]

    try:
        result = profile.fetch_models(
            api_key="actual-secret-token",
            base_url=f"http://127.0.0.1:{redirect_port}",
            timeout=5.0,
        )
    finally:
        redirect_server.shutdown()
        target_server.shutdown()
        redirect_thread.join(timeout=2.0)
        target_thread.join(timeout=2.0)

    assert result == ["should-not-be-trusted"], (
        "sanity check: the redirect must actually have been followed"
    )
    assert source_auth_headers == ["Bearer actual-secret-token"]
    assert target_auth_headers == [None], (
        "Authorization header leaked to a different origin after a redirect"
    )


def test_actual_provider_model_ids_use_local_profile_catalog(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    profile = get_provider_profile("actual")

    with patch.object(
        profile, "fetch_models", return_value=["actual/local-model"]
    ) as fetch:
        assert provider_model_ids("actual") == ["actual/local-model"]

    fetch.assert_called_once_with(
        api_key=ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
        base_url=DEFAULT_ACTUAL_LOCAL_BASE_URL,
    )


def test_actual_hosted_model_ids_send_resolved_credential(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    profile = get_provider_profile("actual")

    with patch.object(
        profile, "fetch_models", return_value=["actual/hosted-model"]
    ) as fetch:
        assert provider_model_ids("actual") == ["actual/hosted-model"]

    fetch.assert_called_once_with(
        api_key="actual-test-key",
        base_url=DEFAULT_ACTUAL_BASE_URL,
    )


def test_actual_hosted_model_ids_do_not_probe_without_credentials(monkeypatch):
    _clear_actual_env(monkeypatch)
    profile = get_provider_profile("actual")

    with patch.object(profile, "fetch_models") as fetch:
        assert provider_model_ids("actual") == []

    fetch.assert_not_called()


def test_actual_codex_transport_clamps_reasoning_effort():
    """Actual's SGLang/vLLM backends only accept none/low/medium/high/max.

    A global xhigh/ultra reasoning_effort must be clamped before the wire,
    otherwise every request fails with a wrapped HTTP 400. Regression for
    the field-reported reasoning_effort trap.
    """
    from agent.transports.codex import ResponsesApiTransport

    t = ResponsesApiTransport()
    for requested, expected in (("xhigh", "high"), ("ultra", "max"), ("high", "high")):
        kwargs = t.build_kwargs(
            model="glm-5.2-nvfp4",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            provider="actual",
            reasoning_config={"effort": requested},
            base_url=DEFAULT_ACTUAL_BASE_URL,
        )
        assert kwargs["reasoning"]["effort"] == expected, requested

    # Other providers keep the wider values untouched on the generic path.
    kwargs = t.build_kwargs(
        model="some-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider="openai-codex",
        reasoning_config={"effort": "xhigh"},
    )
    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_actual_runtime_config_local_base_url_without_key(monkeypatch):
    """Config-driven loopback base_url (not just env) reaches the no-auth path."""
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "actual",
            "base_url": "http://localhost:8080",
            "default": "actual/local-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["api_mode"] == "codex_responses"
