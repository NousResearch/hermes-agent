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
                "VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"):
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
    token, base = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
    assert base == (
        "https://aiplatform.googleapis.com/v1beta1/projects/adc-project/"
        "locations/global/endpoints/openapi"
    )


class _MetaCreds:
    def __init__(self):
        self.token = None
        self.expiry = None
        self.expired = False

    def refresh(self, req):
        self.token = "FAKE.TOKEN"


def test_adc_project_none_falls_back_to_metadata(vertex_adapter, monkeypatch):
    """GCE VM: default() mints a token but returns project_id=None; the adapter
    must resolve the project from the metadata server rather than discard a
    valid token. Regression guard for VM attached-service-account auth."""
    ga = sys.modules["google.auth"]
    monkeypatch.setattr(ga, "default", lambda scopes=None: (_MetaCreds(), None))
    monkeypatch.setattr(
        vertex_adapter, "_project_from_metadata", lambda: "metadata-project"
    )
    token, base = vertex_adapter.get_vertex_config()
    assert token == "FAKE.TOKEN"
    assert "projects/metadata-project" in base


def test_explicit_project_override_wins_over_metadata(vertex_adapter, monkeypatch):
    """Billing safety: an explicit VERTEX_PROJECT_ID must beat the VM host
    project even when the metadata fallback would supply one."""
    ga = sys.modules["google.auth"]
    monkeypatch.setattr(ga, "default", lambda scopes=None: (_MetaCreds(), None))
    monkeypatch.setattr(
        vertex_adapter, "_project_from_metadata", lambda: "metadata-project"
    )
    monkeypatch.setenv("VERTEX_PROJECT_ID", "billing-project")
    _token, base = vertex_adapter.get_vertex_config()
    assert "projects/billing-project" in base
    assert "metadata-project" not in base


def test_config_yaml_supplies_project_and_region(vertex_adapter, monkeypatch):
    monkeypatch.setattr(
        vertex_adapter, "_vertex_config",
        lambda: {"project_id": "cfg-project", "region": "europe-west4"},
    )
    token, base = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
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
    token, base = vertex_adapter.get_vertex_config()
    assert token == "ya29.FAKE"
    assert "adc-project" in base


def test_adc_failure_falls_back_to_service_account(monkeypatch, tmp_path):
    """When ADC refresh fails but a service-account JSON exists, use the SA."""
    for var in ("VERTEX_PROJECT_ID", "VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"):
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
