"""Google Chat dependency installation crosses PM's declared-feature interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import pytest

import pm
from plugins.platforms.google_chat import oauth


def test_stale_google_transitives_are_reported_missing(monkeypatch):
    installed = {
        "google-cloud-pubsub": "2.39.0",
        "google-api-python-client": "2.194.0",
        "google-auth": "2.55.0",
        "google-auth-oauthlib": "1.3.1",
        "google-auth-httplib2": "0.3.1",
        "httplib2": "0.31.2",
        "pyasn1": "0.6.3",
    }

    def fake_version(name):
        try:
            return installed[name]
        except KeyError:
            raise PackageNotFoundError(name) from None

    monkeypatch.setattr(oauth, "_distribution_version", fake_version)

    assert oauth._missing_required_packages() == [
        "google-auth==2.55.1",
        "httplib2==0.32.0",
        "pyasn1==0.6.4",
    ]


def test_installer_repairs_stale_transitives_through_pm(monkeypatch, capsys):
    states = iter([["google-auth==2.55.1", "httplib2==0.32.0", "pyasn1==0.6.4"], []])
    monkeypatch.setattr(oauth, "_missing_required_packages", lambda: next(states))
    calls = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras, **kwargs: calls.append((extras, kwargs)))

    assert oauth.install_deps() is True
    assert calls == [(["google", "google-chat"], {"explicit": True})]
    assert "restart" in capsys.readouterr().out.lower()


def test_ensure_deps_surfaces_install_reason(monkeypatch):
    """A refused install must reach the registry's log with its reason, not a bare False."""
    from plugins.platforms.google_chat import adapter

    monkeypatch.setattr(adapter, "GOOGLE_CHAT_AVAILABLE", False)

    def blocked(extra):
        raise RuntimeError(f"extra {extra!r}: lazy installs are disabled")

    monkeypatch.setattr(pm, "ensure_import", blocked)
    with pytest.raises(RuntimeError, match="lazy installs are disabled"):
        adapter.ensure_google_chat_deps()
