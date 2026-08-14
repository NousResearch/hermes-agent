"""Tests for the llamacpp profile's residency reporting.

``resident_models`` reads llama-swap's /running - the proxy's own
management route, which never starts a model - and returns the resident
ids. On a bare llama-server or an unreachable endpoint it returns None so
the residency indicator degrades gracefully with no error.
"""

from __future__ import annotations

import shutil

import pytest

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_llamacpp_reasoning_kwargs import _probe_pkg
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)

BASE_URL = "http://192.168.77.10:8080/v1"


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    plugin_dir.parent.mkdir(parents=True)
    shutil.copytree(
        _installed_plugin_dir(),
        plugin_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None and profile.name == "llamacpp"
    yield profile
    _clear_provider_caches()


def test_swap_reports_resident_ids_including_starting(rig, monkeypatch):
    probe = _probe_pkg(rig)
    requested = []

    def fake_get(url, timeout):
        requested.append(url)
        return 200, {
            "running": [
                {"model": "qwen-small", "state": "ready"},
                {"model": "warming-up", "state": "starting"},
            ]
        }

    monkeypatch.setattr(probe, "_http_get_json", fake_get)

    assert rig.resident_models(base_url=BASE_URL) == ("qwen-small", "warming-up")
    # Only the management route was touched - never a model-scoped one.
    assert requested == ["http://192.168.77.10:8080/running"]


def test_bare_llama_server_degrades_to_none(rig, monkeypatch):
    probe = _probe_pkg(rig)
    monkeypatch.setattr(probe, "_http_get_json", lambda url, timeout: (404, None))

    assert rig.resident_models(base_url=BASE_URL) is None


def test_unreachable_endpoint_degrades_to_none(rig, monkeypatch):
    probe = _probe_pkg(rig)

    def fake_get(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(probe, "_http_get_json", fake_get)

    assert rig.resident_models(base_url=BASE_URL) is None


def test_swap_with_nothing_resident_reports_empty(rig, monkeypatch):
    probe = _probe_pkg(rig)
    monkeypatch.setattr(
        probe, "_http_get_json", lambda url, timeout: (200, {"running": []})
    )

    assert rig.resident_models(base_url=BASE_URL) == ()
