"""Tests for the FreeLLMAPI model-provider plugin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_INIT = _REPO_ROOT / "plugins" / "model-providers" / "freellmapi" / "__init__.py"


@pytest.fixture()
def freellmapi_profile():
    """Import the bundled profile without polluting the global registry."""
    module_name = "freellmapi_provider_test_module"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, _PLUGIN_INIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.freellmapi


def test_profile_registered_via_alias(freellmapi_profile):
    from providers import get_provider_profile

    assert get_provider_profile("freellmapi") is freellmapi_profile
    assert get_provider_profile("freellm") is freellmapi_profile
    assert get_provider_profile("free-llm-api") is freellmapi_profile


def test_sticky_session_header(freellmapi_profile):
    _, top_level = freellmapi_profile.build_api_kwargs_extras(
        session_id="hermes-session-abc123"
    )
    assert top_level["extra_headers"]["X-Session-Id"] == "hermes-session-abc123"


def test_no_session_header_when_absent(freellmapi_profile):
    _, top_level = freellmapi_profile.build_api_kwargs_extras(session_id=None)
    assert "extra_headers" not in top_level


def test_default_base_url(freellmapi_profile):
    assert freellmapi_profile.base_url == "http://127.0.0.1:3001/v1"


def test_fallback_models_include_auto(freellmapi_profile):
    assert freellmapi_profile.fallback_models[0] == "auto"
