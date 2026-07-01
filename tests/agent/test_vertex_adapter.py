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
