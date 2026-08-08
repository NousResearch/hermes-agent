"""Photon default-on env flags must treat 'off' as disabled."""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter, _markdown_enabled


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    return PhotonAdapter(PlatformConfig(enabled=True, token="", extra={}))


@pytest.mark.parametrize("raw", ["false", "0", "no", "off", "OFF"])
def test_markdown_falsy_aliases_disable(monkeypatch, raw):
    monkeypatch.setenv("PHOTON_MARKDOWN", raw)
    assert _markdown_enabled() is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE"])
def test_markdown_truthy_aliases_enable(monkeypatch, raw):
    monkeypatch.setenv("PHOTON_MARKDOWN", raw)
    assert _markdown_enabled() is True


def test_markdown_defaults_on(monkeypatch):
    monkeypatch.delenv("PHOTON_MARKDOWN", raising=False)
    assert _markdown_enabled() is True


@pytest.mark.parametrize("raw", ["false", "0", "no", "off"])
def test_sidecar_autostart_falsy_aliases_disable(monkeypatch, raw):
    monkeypatch.setenv("PHOTON_SIDECAR_AUTOSTART", raw)
    assert _make_adapter(monkeypatch)._autostart_sidecar is False


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on"])
def test_sidecar_autostart_truthy_aliases_enable(monkeypatch, raw):
    monkeypatch.setenv("PHOTON_SIDECAR_AUTOSTART", raw)
    assert _make_adapter(monkeypatch)._autostart_sidecar is True


def test_sidecar_autostart_defaults_on(monkeypatch):
    monkeypatch.delenv("PHOTON_SIDECAR_AUTOSTART", raising=False)
    assert _make_adapter(monkeypatch)._autostart_sidecar is True
