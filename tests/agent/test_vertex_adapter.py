"""Tests for the Vertex AI adapter (agent/vertex_adapter.py).

Vertex uses OAuth2 (short-lived access tokens from a service-account JSON or
ADC), NOT a static API key. These tests mock google-auth entirely — no network
calls — and cover token minting, the config.yaml→env precedence bridge, the
global vs regional base-URL shapes, and the ADC→service-account fallback.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_google_auth(monkeypatch, *, adc_ok=True, adc_project="adc-project",
                              sa_project="sa-project", token="ya29.FAKE"):
    """Register a fake google-auth tree in sys.modules and return the module set."""
    ga = types.ModuleType("google.auth")
    gt = types.ModuleType("google.auth.transport")
    gtr = types.ModuleType("google.auth.transport.requests")
    go = types.ModuleType("google.oauth2")
    gsa = types.ModuleType("google.oauth2.service_account")
    gp = types.ModuleType("google")

    gtr.Request = type("Request", (), {})

    class _Creds:
        def __init__(self):
            self.token = None
            self.expiry = None
            self.expired = False

        def refresh(self, req):
            self.token = token

    def _default(scopes=None):
        if not adc_ok:
            raise RuntimeError("Could not automatically determine credentials")
        return _Creds(), adc_project

    ga.default = _default
    ga.transport = gt
    gt.requests = gtr

    class _SA:
        @staticmethod
        def from_service_account_file(path, scopes=None):
            c = _Creds()
            c.project_id = sa_project
            return c

    gsa.Credentials = _SA
    go.service_account = gsa
    gp.auth = ga
    gp.oauth2 = go

    for name, mod in [
        ("google", gp), ("google.auth", ga), ("google.auth.transport", gt),
        ("google.auth.transport.requests", gtr), ("google.oauth2", go),
        ("google.oauth2.service_account", gsa),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return gp


@pytest.fixture
def vertex_adapter(monkeypatch):
    """Fresh vertex_adapter with a fake google-auth and clean caches/env."""
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_VERTEX_API_KEY", "GOOGLE_VERTEX_PROJECT",
                "GOOGLE_VERTEX_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    _install_fake_google_auth(monkeypatch)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    va._creds_cache.clear()
    # Neutralize config.yaml by default; individual tests re-patch _vertex_config.
    monkeypatch.setattr(va, "_vertex_config", lambda: {})
    return va


def test_build_base_url_global(vertex_adapter):
    url = vertex_adapter.build_vertex_base_url("proj", "global")
    assert url == (
        "https://aiplatform.googleapis.com/v1beta1/projects/proj/"
        "locations/global/endpoints/openapi"
    )


def test_build_base_url_regional(vertex_adapter):
    url = vertex_adapter.build_vertex_base_url("proj", "us-central1")
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/proj/"
        "locations/us-central1/endpoints/openapi"
    )


def test_get_vertex_config_uses_adc_and_default_region(vertex_adapter):
    token, base, auth_hdr = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
    assert auth_hdr == "Authorization"
    assert base == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/adc-project/"
        "locations/us-central1/endpoints/openapi"
    )


def test_config_yaml_supplies_project_and_region(vertex_adapter, monkeypatch):
    monkeypatch.setattr(
        vertex_adapter, "_vertex_config",
        lambda: {"project_id": "cfg-project", "region": "europe-west4"},
    )
    token, base, auth_hdr = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
    assert auth_hdr == "Authorization"
    assert "projects/cfg-project" in base
    assert "europe-west4-aiplatform.googleapis.com" in base
    assert "locations/europe-west4" in base


def test_env_overrides_config_yaml(vertex_adapter, monkeypatch):
    monkeypatch.setattr(
        vertex_adapter, "_vertex_config",
        lambda: {"project_id": "cfg-project", "region": "cfg-region"},
    )
    monkeypatch.setenv("VERTEX_PROJECT_ID", "env-project")
    monkeypatch.setenv("VERTEX_REGION", "us-east4")
    assert vertex_adapter._resolve_project_override() == "env-project"
    assert vertex_adapter._resolve_region() == "us-east4"


def test_has_vertex_credentials_via_config_project(vertex_adapter, monkeypatch):
    monkeypatch.setattr(vertex_adapter, "_vertex_config", lambda: {"project_id": "p"})
    assert vertex_adapter.has_vertex_credentials() is True


def test_has_vertex_credentials_false_when_nothing_set(vertex_adapter):
    assert vertex_adapter.has_vertex_credentials() is False


def test_missing_google_auth_returns_none(monkeypatch):
    for var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS",
                "VERTEX_PROJECT_ID", "VERTEX_REGION"):
        monkeypatch.delenv(var, raising=False)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    monkeypatch.setattr(va, "google", None)
    va._creds_cache.clear()
    assert va.get_vertex_credentials() == (None, None)


def test_multiplex_scope_takes_precedence_over_raw_environ(vertex_adapter, monkeypatch):
    """In a multiplex gateway, a profile's own secret scope must win over a
    stale value in process os.environ left behind by another profile's
    dotenv load at boot — otherwise Profile B's turn could resolve Profile
    A's Vertex project (or worse, its credentials file path)."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "other-profile-project")

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"VERTEX_PROJECT_ID": "this-profile-project"})
    try:
        assert vertex_adapter._resolve_project_override() == "this-profile-project"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_multiplex_unscoped_read_fails_closed(vertex_adapter, monkeypatch):
    """A credential read with no profile scope installed while multiplexing
    is active must raise rather than silently fall back to (possibly another
    profile's) raw os.environ value."""
    from agent import secret_scope

    monkeypatch.setenv("VERTEX_PROJECT_ID", "leaked-project")
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            vertex_adapter._resolve_project_override()
    finally:
        secret_scope.set_multiplex_active(False)


def test_adc_refuses_foreign_profile_google_application_credentials(
    vertex_adapter, monkeypatch, tmp_path
):
    """When this profile's scope defines no Vertex credentials, but os.environ
    still carries a *different* profile's GOOGLE_APPLICATION_CREDENTIALS (left
    there by python-dotenv at gateway boot), ADC must not silently mint a
    token under that foreign service account."""
    from agent import secret_scope

    sa_file = tmp_path / "other_profile_sa.json"
    sa_file.write_text('{"project_id": "other-profile"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))

    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({})  # this profile defines nothing
    try:
        assert vertex_adapter.get_vertex_credentials() == (None, None)
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_adc_still_works_when_not_multiplexed(vertex_adapter):
    """Single-profile (non-gateway) installs must see zero behavior change:
    ADC still resolves normally when multiplexing is off, scope or not."""
    token, base, auth_hdr = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
    assert auth_hdr == "Authorization"
    assert "adc-project" in base


def test_adc_failure_falls_back_to_service_account(monkeypatch, tmp_path):
    """When ADC refresh fails but a service-account JSON exists, use the SA."""
    for var in ("VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT",
                "GOOGLE_VERTEX_PROJECT", "GOOGLE_VERTEX_API_KEY",
                "GOOGLE_VERTEX_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    sa_file = tmp_path / "sa.json"
    sa_file.write_text('{"project_id": "sa-project"}')
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))
    monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
    _install_fake_google_auth(monkeypatch, adc_ok=False)
    import agent.vertex_adapter as va
    va = importlib.reload(va)
    va._creds_cache.clear()
    monkeypatch.setattr(va, "_vertex_config", lambda: {})
    # A resolvable SA path means the primary cache key is the file (not __adc__),
    # so this exercises the direct-SA path.
    token, project = va.get_vertex_credentials()
    assert token == "ya29.FAKE"
    assert project == "sa-project"


# ── API Key (Express Mode) tests ─────────────────────────────────────────────


def test_has_vertex_api_key_true_when_env_set(vertex_adapter, monkeypatch):
    """has_vertex_api_key returns True when GOOGLE_VERTEX_API_KEY is set."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyFakeKey123")
    assert vertex_adapter.has_vertex_api_key() is True


def test_has_vertex_api_key_false_when_not_set(vertex_adapter):
    """has_vertex_api_key returns False when the env var is absent."""
    assert vertex_adapter.has_vertex_api_key() is False


def test_resolve_vertex_api_key_returns_value(vertex_adapter, monkeypatch):
    """resolve_vertex_api_key returns the env var value."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyTestKey456")
    assert vertex_adapter.resolve_vertex_api_key() == "AIzaSyTestKey456"


def test_resolve_vertex_api_key_returns_none_when_not_set(vertex_adapter):
    """resolve_vertex_api_key returns None when env var is absent."""
    assert vertex_adapter.resolve_vertex_api_key() is None


def test_build_vertex_api_key_base_url_global(vertex_adapter):
    """Express Mode global endpoint uses aiplatform.googleapis.com."""
    url = vertex_adapter.build_vertex_api_key_base_url("my-project", "global")
    assert url == (
        "https://aiplatform.googleapis.com/v1beta1/projects/my-project/"
        "locations/global/endpoints/openapi"
    )


def test_build_vertex_api_key_base_url_regional(vertex_adapter):
    """Express Mode regional endpoint uses {region}-aiplatform.googleapis.com."""
    url = vertex_adapter.build_vertex_api_key_base_url("my-project", "us-central1")
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/"
        "locations/us-central1/endpoints/openapi"
    )


def test_build_vertex_api_key_base_url_europe(vertex_adapter):
    """Express Mode with europe-west4 region."""
    url = vertex_adapter.build_vertex_api_key_base_url("my-project", "europe-west4")
    assert url == (
        "https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/my-project/"
        "locations/europe-west4/endpoints/openapi"
    )


def test_get_vertex_config_with_api_key(vertex_adapter, monkeypatch):
    """get_vertex_config returns (api_key, base_url, x-goog-api-key) when API key is set."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyApiKey")
    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "api-key-project")
    monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "europe-west1")

    token_or_key, base_url, auth_hdr = vertex_adapter.get_vertex_config()
    assert token_or_key == "AIzaSyApiKey"
    assert auth_hdr == "x-goog-api-key"
    assert "projects/api-key-project" in base_url
    assert "europe-west1-aiplatform.googleapis.com" in base_url
    assert "locations/europe-west1" in base_url


def test_get_vertex_config_api_key_precedence_over_adc(vertex_adapter, monkeypatch):
    """API key path is used when BOTH API key and ADC credentials are available."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyKey")
    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "key-project")

    token_or_key, base_url, auth_hdr = vertex_adapter.get_vertex_config()
    assert token_or_key == "AIzaSyKey"  # API key, not OAuth token
    assert auth_hdr == "x-goog-api-key"
    assert "projects/key-project" in base_url


def test_get_vertex_config_api_key_missing_project(vertex_adapter, monkeypatch):
    """get_vertex_config returns (None, None, None) when API key is set but project is not."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyKey")
    # No project ID set anywhere

    result = vertex_adapter.get_vertex_config()
    assert result == (None, None, None)


def test_has_vertex_credentials_via_api_key(vertex_adapter, monkeypatch):
    """has_vertex_credentials returns True when only the API key is set."""
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AIzaSyKey")
    assert vertex_adapter.has_vertex_credentials() is True


def test_googole_vertex_location_region_precedence(vertex_adapter, monkeypatch):
    """GOOGLE_VERTEX_LOCATION takes precedence over VERTEX_REGION."""
    monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "us-west1")
    monkeypatch.setenv("VERTEX_REGION", "europe-west4")

    assert vertex_adapter._resolve_region() == "us-west1"


def test_googole_vertex_project_precedence(vertex_adapter, monkeypatch):
    """GOOGLE_VERTEX_PROJECT takes precedence over VERTEX_PROJECT_ID."""
    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "gv-project")
    monkeypatch.setenv("VERTEX_PROJECT_ID", "legacy-project")

    assert vertex_adapter._resolve_project_override() == "gv-project"


def test_googole_vertex_location_falls_back_to_vertex_region(vertex_adapter, monkeypatch):
    """VERTEX_REGION is used when GOOGLE_VERTEX_LOCATION is not set."""
    monkeypatch.setenv("VERTEX_REGION", "asia-east1")

    assert vertex_adapter._resolve_region() == "asia-east1"


# ── Model Discovery Tests ────────────────────────────────────────────────────


class _FakeUrlopenResult:
    """Mimics the result of urllib.request.urlopen."""
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_discover_vertex_models_parses_response(vertex_adapter, monkeypatch):
    """discover_vertex_models correctly parses a models.list response."""
    response_data = {
        "models": [
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent", "countTokens"],
            },
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/gemini-3-pro-preview",
                "displayName": "Gemini 3 Pro Preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/gemini-embedding-001",
                "displayName": "Gemini Embedding 001",
                "supportedGenerationMethods": ["embedding"],
            },
        ]
    }
    import urllib.request
    import json

    def fake_urlopen(req, timeout=10):
        return _FakeUrlopenResult(json.dumps(response_data).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == ["gemini-2.5-flash", "gemini-3-pro-preview"]


def test_discover_vertex_models_empty_when_no_generate_content(vertex_adapter, monkeypatch):
    """Models without generateContent are excluded from discovery."""
    response_data = {
        "models": [
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/textembedding-gecko",
                "displayName": "Gecko",
                "supportedGenerationMethods": ["embedding"],
            },
        ]
    }
    import urllib.request
    import json

    def fake_urlopen(req, timeout=10):
        return _FakeUrlopenResult(json.dumps(response_data).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == []


def test_discover_vertex_models_network_failure_returns_empty(vertex_adapter, monkeypatch):
    """Network errors during discovery return an empty list without crashing."""
    import urllib.error

    def fake_urlopen(req, timeout=10):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == []


def test_discover_vertex_models_http_error_returns_empty(vertex_adapter, monkeypatch):
    """HTTP 4xx/5xx during discovery return an empty list."""
    import urllib.error

    def fake_urlopen(req, timeout=10):
        raise urllib.error.HTTPError(
            url=req.full_url if hasattr(req, 'full_url') else "",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == []


def test_discover_vertex_models_malformed_json_returns_empty(vertex_adapter, monkeypatch):
    """Malformed JSON responses return an empty list."""
    import urllib.request

    def fake_urlopen(req, timeout=10):
        return _FakeUrlopenResult(b"not json at all")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == []


def test_discover_vertex_models_sorts_results(vertex_adapter, monkeypatch):
    """discover_vertex_models returns sorted model IDs."""
    response_data = {
        "models": [
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/gemini-3-pro-preview",
                "displayName": "Gemini 3 Pro Preview",
                "supportedGenerationMethods": ["generateContent"],
            },
            {
                "name": "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash",
                "displayName": "Gemini 2.5 Flash",
                "supportedGenerationMethods": ["generateContent"],
            },
        ]
    }
    import urllib.request
    import json

    def fake_urlopen(req, timeout=10):
        return _FakeUrlopenResult(json.dumps(response_data).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    models = vertex_adapter.discover_vertex_models("AIzaSyKey", "my-project", "us-central1")
    assert models == ["gemini-2.5-flash", "gemini-3-pro-preview"]  # sorted
    assert models == sorted(models)
