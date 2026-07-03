"""Regression test for vertex provider overlay (#57539).

Ensures the ``vertex`` provider is present in ``HERMES_OVERLAYS`` and
its aliases resolve correctly, so ``/model --provider vertex`` works
from the CLI.
"""

from hermes_cli.providers import ALIASES, HERMES_OVERLAYS


def test_vertex_in_overlays():
    """Vertex must be registered in HERMES_OVERLAYS."""
    assert "vertex" in HERMES_OVERLAYS


def test_vertex_overlay_transport():
    overlay = HERMES_OVERLAYS["vertex"]
    assert overlay.transport == "openai_chat"
    assert overlay.auth_type == "vertex"


def test_vertex_overlay_base_url():
    overlay = HERMES_OVERLAYS["vertex"]
    assert overlay.base_url_override == "https://aiplatform.googleapis.com"


def test_vertex_overlay_env_vars():
    overlay = HERMES_OVERLAYS["vertex"]
    assert "GOOGLE_APPLICATION_CREDENTIALS" in overlay.extra_env_vars
    assert "VERTEX_PROJECT_ID" in overlay.extra_env_vars
    assert "VERTEX_REGION" in overlay.extra_env_vars


def test_vertex_aliases_resolve():
    for alias in ("google-vertex", "vertex-ai", "gcp-vertex", "google-cloud-vertex"):
        assert ALIASES.get(alias) == "vertex", f"Alias {alias!r} should resolve to 'vertex'"
